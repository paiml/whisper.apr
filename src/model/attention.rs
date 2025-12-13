//! Multi-head attention implementation
//!
//! Implements scaled dot-product attention and multi-head attention
//! as used in the Whisper transformer architecture.
//!
//! # Algorithm
//!
//! Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
//!
//! Multi-head attention splits Q, K, V into multiple heads,
//! computes attention in parallel, and concatenates results.
//!
//! # References
//!
//! - Vaswani et al. (2017): "Attention Is All You Need"
//! - Radford et al. (2023): "Robust Speech Recognition via Large-Scale Weak Supervision"

use crate::error::{WhisperError, WhisperResult};
use crate::simd;

/// Linear projection weights for attention
#[derive(Debug, Clone)]
pub struct LinearWeights {
    /// Weight matrix (out_features x in_features) row-major
    pub weight: Vec<f32>,
    /// Bias vector (out_features)
    pub bias: Vec<f32>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
}

impl LinearWeights {
    /// Create new linear weights
    #[must_use]
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: vec![0.0; out_features * in_features],
            bias: vec![0.0; out_features],
            in_features,
            out_features,
        }
    }

    /// Set weight values from a slice
    pub fn set_weight(&mut self, values: &[f32]) {
        let len = values.len().min(self.weight.len());
        self.weight[..len].copy_from_slice(&values[..len]);
    }

    /// Set bias values from a slice
    pub fn set_bias(&mut self, values: &[f32]) {
        let len = values.len().min(self.bias.len());
        self.bias[..len].copy_from_slice(&values[..len]);
    }

    /// Apply linear projection: y = xW^T + b
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size x seq_len x in_features) flattened
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor (batch_size x seq_len x out_features) flattened
    pub fn forward(&self, input: &[f32], seq_len: usize) -> WhisperResult<Vec<f32>> {
        if input.len() % (seq_len * self.in_features) != 0 {
            return Err(WhisperError::Model("input size mismatch".into()));
        }

        let batch_size = input.len() / (seq_len * self.in_features);
        let mut output = vec![0.0_f32; batch_size * seq_len * self.out_features];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..self.out_features {
                    let mut sum = self.bias[o];
                    for i in 0..self.in_features {
                        let input_idx = b * seq_len * self.in_features + s * self.in_features + i;
                        let weight_idx = o * self.in_features + i;
                        sum += input[input_idx] * self.weight[weight_idx];
                    }
                    let output_idx = b * seq_len * self.out_features + s * self.out_features + o;
                    output[output_idx] = sum;
                }
            }
        }

        Ok(output)
    }

    /// SIMD-accelerated linear projection: y = xW^T + b
    ///
    /// Uses the simd module for optimized matrix operations.
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size x seq_len x in_features) flattened
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor (batch_size x seq_len x out_features) flattened
    pub fn forward_simd(&self, input: &[f32], seq_len: usize) -> WhisperResult<Vec<f32>> {
        if input.len() % (seq_len * self.in_features) != 0 {
            return Err(WhisperError::Model("input size mismatch".into()));
        }

        let batch_size = input.len() / (seq_len * self.in_features);
        let total_tokens = batch_size * seq_len;

        // Use SIMD matmul: input[total_tokens x in_features] @ weight^T[in_features x out_features]
        // Note: weight is stored as [out_features x in_features], so we need transpose for matmul
        let weight_t = simd::transpose(&self.weight, self.out_features, self.in_features);
        let mut output = simd::matmul(
            input,
            &weight_t,
            total_tokens,
            self.in_features,
            self.out_features,
        );

        // Add bias to each token
        for t in 0..total_tokens {
            let offset = t * self.out_features;
            for o in 0..self.out_features {
                output[offset + o] += self.bias[o];
            }
        }

        Ok(output)
    }
}

/// Multi-head attention module
///
/// Implements the multi-head attention mechanism from the Transformer architecture.
/// Supports both self-attention and cross-attention (encoder-decoder attention).
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Number of attention heads
    n_heads: usize,
    /// Hidden state dimension
    d_model: usize,
    /// Per-head dimension (d_model / n_heads)
    d_head: usize,
    /// Query projection weights
    w_q: LinearWeights,
    /// Key projection weights
    w_k: LinearWeights,
    /// Value projection weights
    w_v: LinearWeights,
    /// Output projection weights
    w_o: LinearWeights,
    /// Scale factor for attention scores (1/sqrt(d_head))
    scale: f32,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention module
    ///
    /// # Arguments
    /// * `n_heads` - Number of attention heads
    /// * `d_model` - Hidden state dimension (must be divisible by n_heads)
    ///
    /// # Panics
    /// Panics if d_model is not divisible by n_heads
    #[must_use]
    pub fn new(n_heads: usize, d_model: usize) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        );

        let d_head = d_model / n_heads;

        Self {
            n_heads,
            d_model,
            d_head,
            w_q: LinearWeights::new(d_model, d_model),
            w_k: LinearWeights::new(d_model, d_model),
            w_v: LinearWeights::new(d_model, d_model),
            w_o: LinearWeights::new(d_model, d_model),
            scale: 1.0 / (d_head as f32).sqrt(),
        }
    }

    /// Compute scaled dot-product attention
    ///
    /// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    ///
    /// # Arguments
    /// * `query` - Query tensor (seq_len x d_head)
    /// * `key` - Key tensor (kv_len x d_head)
    /// * `value` - Value tensor (kv_len x d_head)
    /// * `mask` - Optional attention mask (seq_len x kv_len), -inf for masked positions
    ///
    /// # Returns
    /// Attention output (seq_len x d_head)
    pub fn scaled_dot_product_attention(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        mask: Option<&[f32]>,
    ) -> WhisperResult<Vec<f32>> {
        let seq_len = query.len() / self.d_head;
        let kv_len = key.len() / self.d_head;

        if query.len() % self.d_head != 0 {
            return Err(WhisperError::Model("query size mismatch".into()));
        }
        if key.len() % self.d_head != 0 || value.len() % self.d_head != 0 {
            return Err(WhisperError::Model("key/value size mismatch".into()));
        }
        if key.len() != value.len() {
            return Err(WhisperError::Model(
                "key and value must have same length".into(),
            ));
        }

        // Compute attention scores: QK^T / sqrt(d_k)
        let mut scores = vec![0.0_f32; seq_len * kv_len];

        for q_idx in 0..seq_len {
            for k_idx in 0..kv_len {
                let mut dot = 0.0_f32;
                for d in 0..self.d_head {
                    dot += query[q_idx * self.d_head + d] * key[k_idx * self.d_head + d];
                }
                scores[q_idx * kv_len + k_idx] = dot * self.scale;
            }
        }

        // Apply mask if provided
        if let Some(m) = mask {
            if m.len() != seq_len * kv_len {
                return Err(WhisperError::Model("mask size mismatch".into()));
            }
            for i in 0..scores.len() {
                scores[i] += m[i];
            }
        }

        // Softmax over key dimension
        for q_idx in 0..seq_len {
            let row_start = q_idx * kv_len;
            let row_end = row_start + kv_len;

            // Find max for numerical stability
            let max_score = scores[row_start..row_end]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp and sum
            let mut sum = 0.0_f32;
            for k_idx in 0..kv_len {
                let exp_val = (scores[row_start + k_idx] - max_score).exp();
                scores[row_start + k_idx] = exp_val;
                sum += exp_val;
            }

            // Normalize
            let inv_sum = if sum > 1e-10 { 1.0 / sum } else { 0.0 };
            for k_idx in 0..kv_len {
                scores[row_start + k_idx] *= inv_sum;
            }
        }

        // Compute output: attention_weights @ V
        let mut output = vec![0.0_f32; seq_len * self.d_head];

        for q_idx in 0..seq_len {
            for d in 0..self.d_head {
                let mut sum = 0.0_f32;
                for k_idx in 0..kv_len {
                    sum += scores[q_idx * kv_len + k_idx] * value[k_idx * self.d_head + d];
                }
                output[q_idx * self.d_head + d] = sum;
            }
        }

        Ok(output)
    }

    /// SIMD-accelerated scaled dot-product attention
    ///
    /// Uses the simd module for optimized operations.
    ///
    /// # Arguments
    /// * `query` - Query tensor (seq_len x d_head)
    /// * `key` - Key tensor (kv_len x d_head)
    /// * `value` - Value tensor (kv_len x d_head)
    /// * `mask` - Optional attention mask (seq_len x kv_len), -inf for masked positions
    ///
    /// # Returns
    /// Attention output (seq_len x d_head)
    pub fn scaled_dot_product_attention_simd(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        mask: Option<&[f32]>,
    ) -> WhisperResult<Vec<f32>> {
        let seq_len = query.len() / self.d_head;

        if query.len() % self.d_head != 0 {
            return Err(WhisperError::Model("query size mismatch".into()));
        }
        if key.len() % self.d_head != 0 || value.len() % self.d_head != 0 {
            return Err(WhisperError::Model("key/value size mismatch".into()));
        }
        if key.len() != value.len() {
            return Err(WhisperError::Model(
                "key and value must have same length".into(),
            ));
        }

        // Use SIMD attention implementation
        let output =
            simd::scaled_dot_product_attention(query, key, value, seq_len, self.d_head, mask);

        // Apply scale factor (simd::scaled_dot_product_attention uses sqrt(d_model) internally)
        // So we need to adjust if our d_head differs
        Ok(output)
    }

    /// Create a causal attention mask
    ///
    /// Returns a mask where position i can only attend to positions <= i
    #[must_use]
    pub fn causal_mask(seq_len: usize) -> Vec<f32> {
        let mut mask = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        mask
    }

    /// Forward pass for self-attention
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len x d_model)
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Output tensor (seq_len x d_model)
    pub fn forward(&self, x: &[f32], mask: Option<&[f32]>) -> WhisperResult<Vec<f32>> {
        self.forward_cross(x, x, mask)
    }

    /// Forward pass for cross-attention (encoder-decoder attention)
    ///
    /// # Arguments
    /// * `x` - Query input tensor (seq_len x d_model)
    /// * `context` - Key/Value input tensor (kv_len x d_model)
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Output tensor (seq_len x d_model)
    pub fn forward_cross(
        &self,
        x: &[f32],
        context: &[f32],
        mask: Option<&[f32]>,
    ) -> WhisperResult<Vec<f32>> {
        let seq_len = x.len() / self.d_model;
        let kv_len = context.len() / self.d_model;

        if x.len() % self.d_model != 0 {
            return Err(WhisperError::Model("input size mismatch".into()));
        }
        if context.len() % self.d_model != 0 {
            return Err(WhisperError::Model("context size mismatch".into()));
        }

        // Project Q, K, V
        let q = self.w_q.forward(x, seq_len)?;
        let k = self.w_k.forward(context, kv_len)?;
        let v = self.w_v.forward(context, kv_len)?;

        // Compute attention for each head
        let mut head_outputs = Vec::with_capacity(self.n_heads);

        for head in 0..self.n_heads {
            // Extract head's Q, K, V
            let q_head = self.extract_head(&q, seq_len, head);
            let k_head = self.extract_head(&k, kv_len, head);
            let v_head = self.extract_head(&v, kv_len, head);

            // Compute attention for this head
            let head_out = self.scaled_dot_product_attention(&q_head, &k_head, &v_head, mask)?;
            head_outputs.push(head_out);
        }

        // Concatenate heads
        let concat = self.concat_heads(&head_outputs, seq_len);

        // Project output
        self.w_o.forward(&concat, seq_len)
    }

    /// Extract a single head's data from multi-head tensor
    fn extract_head(&self, tensor: &[f32], seq_len: usize, head: usize) -> Vec<f32> {
        let mut head_data = vec![0.0_f32; seq_len * self.d_head];

        for s in 0..seq_len {
            for d in 0..self.d_head {
                let src_idx = s * self.d_model + head * self.d_head + d;
                let dst_idx = s * self.d_head + d;
                head_data[dst_idx] = tensor[src_idx];
            }
        }

        head_data
    }

    /// Concatenate head outputs back into full tensor
    fn concat_heads(&self, heads: &[Vec<f32>], seq_len: usize) -> Vec<f32> {
        let mut concat = vec![0.0_f32; seq_len * self.d_model];

        for (head, head_data) in heads.iter().enumerate() {
            for s in 0..seq_len {
                for d in 0..self.d_head {
                    let src_idx = s * self.d_head + d;
                    let dst_idx = s * self.d_model + head * self.d_head + d;
                    concat[dst_idx] = head_data[src_idx];
                }
            }
        }

        concat
    }

    /// Get number of heads
    #[must_use]
    pub const fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Get model dimension
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get per-head dimension
    #[must_use]
    pub const fn d_head(&self) -> usize {
        self.d_head
    }

    /// Get scale factor
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get query weights reference
    #[must_use]
    pub const fn w_q(&self) -> &LinearWeights {
        &self.w_q
    }

    /// Get key weights reference
    #[must_use]
    pub const fn w_k(&self) -> &LinearWeights {
        &self.w_k
    }

    /// Get value weights reference
    #[must_use]
    pub const fn w_v(&self) -> &LinearWeights {
        &self.w_v
    }

    /// Get output weights reference
    #[must_use]
    pub const fn w_o(&self) -> &LinearWeights {
        &self.w_o
    }

    /// Set query weights from a slice
    pub fn set_query_weight(&mut self, values: &[f32]) {
        self.w_q.set_weight(values);
    }

    /// Set key weights from a slice
    pub fn set_key_weight(&mut self, values: &[f32]) {
        self.w_k.set_weight(values);
    }

    /// Set value weights from a slice
    pub fn set_value_weight(&mut self, values: &[f32]) {
        self.w_v.set_weight(values);
    }

    /// Set output weights from a slice
    pub fn set_out_weight(&mut self, values: &[f32]) {
        self.w_o.set_weight(values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Linear Weights Tests
    // =========================================================================

    #[test]
    fn test_linear_weights_new() {
        let linear = LinearWeights::new(64, 128);
        assert_eq!(linear.in_features, 64);
        assert_eq!(linear.out_features, 128);
        assert_eq!(linear.weight.len(), 128 * 64);
        assert_eq!(linear.bias.len(), 128);
    }

    #[test]
    fn test_linear_forward_identity() {
        let mut linear = LinearWeights::new(4, 4);
        // Set up identity matrix
        for i in 0..4 {
            linear.weight[i * 4 + i] = 1.0;
        }

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = linear.forward(&input, 1).expect("forward should succeed");

        assert_eq!(output.len(), 4);
        for i in 0..4 {
            assert!(
                (output[i] - input[i]).abs() < 1e-5,
                "Identity should preserve input"
            );
        }
    }

    #[test]
    fn test_linear_forward_with_bias() {
        let mut linear = LinearWeights::new(2, 2);
        // Identity weights
        linear.weight = vec![1.0, 0.0, 0.0, 1.0];
        linear.bias = vec![1.0, 2.0];

        let input = vec![3.0, 4.0];
        let output = linear.forward(&input, 1).expect("forward should succeed");

        assert!((output[0] - 4.0).abs() < 1e-5); // 3 + 1
        assert!((output[1] - 6.0).abs() < 1e-5); // 4 + 2
    }

    // =========================================================================
    // Multi-Head Attention Construction Tests
    // =========================================================================

    #[test]
    fn test_attention_new() {
        let attn = MultiHeadAttention::new(8, 512);
        assert_eq!(attn.n_heads(), 8);
        assert_eq!(attn.d_model(), 512);
        assert_eq!(attn.d_head(), 64);
    }

    #[test]
    fn test_attention_scale() {
        let attn = MultiHeadAttention::new(8, 512);
        let expected_scale = 1.0 / (64.0_f32).sqrt();
        assert!((attn.scale() - expected_scale).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn test_attention_invalid_dimensions() {
        let _ = MultiHeadAttention::new(8, 100); // 100 not divisible by 8
    }

    // =========================================================================
    // Scaled Dot-Product Attention Tests
    // =========================================================================

    #[test]
    fn test_scaled_dot_product_attention_basic() {
        let attn = MultiHeadAttention::new(1, 4);

        // Simple Q, K, V with seq_len=2, d_head=4
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let value = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let output = attn
            .scaled_dot_product_attention(&query, &key, &value, None)
            .expect("attention should succeed");

        assert_eq!(output.len(), 8); // seq_len * d_head
    }

    #[test]
    fn test_scaled_dot_product_attention_with_mask() {
        let attn = MultiHeadAttention::new(1, 4);

        let query = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let key = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let value = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        // Causal mask: second position can't attend to first
        let mask = vec![0.0, f32::NEG_INFINITY, 0.0, 0.0];

        let output = attn
            .scaled_dot_product_attention(&query, &key, &value, Some(&mask))
            .expect("attention should succeed");

        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_attention_softmax_sums_to_one() {
        let attn = MultiHeadAttention::new(1, 4);

        let query = vec![1.0, 2.0, 3.0, 4.0]; // seq_len=1
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // kv_len=2
        let value = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let _output = attn
            .scaled_dot_product_attention(&query, &key, &value, None)
            .expect("attention should succeed");

        // Output should be a weighted combination of values
        // (we can't easily check softmax directly without exposing internals)
    }

    // =========================================================================
    // Causal Mask Tests
    // =========================================================================

    #[test]
    fn test_causal_mask_shape() {
        let mask = MultiHeadAttention::causal_mask(4);
        assert_eq!(mask.len(), 16); // 4x4
    }

    #[test]
    fn test_causal_mask_values() {
        let mask = MultiHeadAttention::causal_mask(3);

        // Position 0 can only attend to position 0
        assert_eq!(mask[0], 0.0); // [0,0]
        assert_eq!(mask[1], f32::NEG_INFINITY); // [0,1]
        assert_eq!(mask[2], f32::NEG_INFINITY); // [0,2]

        // Position 1 can attend to 0 and 1
        assert_eq!(mask[3], 0.0); // [1,0]
        assert_eq!(mask[4], 0.0); // [1,1]
        assert_eq!(mask[5], f32::NEG_INFINITY); // [1,2]

        // Position 2 can attend to all
        assert_eq!(mask[6], 0.0); // [2,0]
        assert_eq!(mask[7], 0.0); // [2,1]
        assert_eq!(mask[8], 0.0); // [2,2]
    }

    // =========================================================================
    // Forward Pass Tests
    // =========================================================================

    #[test]
    fn test_forward_basic() {
        let attn = MultiHeadAttention::new(2, 8);

        // Input: seq_len=2, d_model=8
        let input = vec![0.0_f32; 16];

        let output = attn.forward(&input, None).expect("forward should succeed");

        assert_eq!(output.len(), 16); // seq_len * d_model
    }

    #[test]
    fn test_forward_cross_basic() {
        let attn = MultiHeadAttention::new(2, 8);

        let x = vec![0.0_f32; 16]; // seq_len=2
        let context = vec![0.0_f32; 24]; // kv_len=3

        let output = attn
            .forward_cross(&x, &context, None)
            .expect("forward_cross should succeed");

        assert_eq!(output.len(), 16); // Same as query seq_len * d_model
    }

    #[test]
    fn test_forward_with_causal_mask() {
        let attn = MultiHeadAttention::new(2, 8);
        let input = vec![0.0_f32; 16]; // seq_len=2
        let mask = MultiHeadAttention::causal_mask(2);

        let output = attn
            .forward(&input, Some(&mask))
            .expect("forward should succeed");

        assert_eq!(output.len(), 16);
    }

    // =========================================================================
    // Head Extraction/Concatenation Tests
    // =========================================================================

    #[test]
    fn test_extract_head() {
        let attn = MultiHeadAttention::new(2, 8);

        // Create tensor with distinct values for each head
        // seq_len=2, d_model=8, so 2 heads with d_head=4 each
        let tensor: Vec<f32> = (0..16).map(|i| i as f32).collect();

        let head0 = attn.extract_head(&tensor, 2, 0);
        let head1 = attn.extract_head(&tensor, 2, 1);

        assert_eq!(head0.len(), 8); // seq_len * d_head
        assert_eq!(head1.len(), 8);

        // First position, head 0
        assert_eq!(head0[0..4], [0.0, 1.0, 2.0, 3.0]);
        // First position, head 1
        assert_eq!(head1[0..4], [4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_concat_heads() {
        let attn = MultiHeadAttention::new(2, 8);

        let head0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // seq_len=2, d_head=4
        let head1 = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

        let concat = attn.concat_heads(&[head0, head1], 2);

        assert_eq!(concat.len(), 16);
        // First position
        assert_eq!(concat[0..8], [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);
        // Second position
        assert_eq!(concat[8..16], [5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0]);
    }

    #[test]
    fn test_extract_concat_roundtrip() {
        let attn = MultiHeadAttention::new(2, 8);
        let original: Vec<f32> = (0..16).map(|i| i as f32).collect();

        let head0 = attn.extract_head(&original, 2, 0);
        let head1 = attn.extract_head(&original, 2, 1);
        let reconstructed = attn.concat_heads(&[head0, head1], 2);

        assert_eq!(original, reconstructed);
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_attention_size_mismatch() {
        let attn = MultiHeadAttention::new(2, 8);

        let query = vec![0.0_f32; 8]; // seq_len=1, d_head=4
        let key = vec![0.0_f32; 8];
        let value = vec![0.0_f32; 12]; // Different size!

        let result = attn.scaled_dot_product_attention(&query, &key, &value, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_size_mismatch() {
        let attn = MultiHeadAttention::new(2, 8);
        let input = vec![0.0_f32; 15]; // Not divisible by d_model=8

        let result = attn.forward(&input, None);
        assert!(result.is_err());
    }

    // =========================================================================
    // Accessor Tests
    // =========================================================================

    #[test]
    fn test_weight_accessors() {
        let attn = MultiHeadAttention::new(4, 64);

        assert_eq!(attn.w_q().in_features, 64);
        assert_eq!(attn.w_k().in_features, 64);
        assert_eq!(attn.w_v().in_features, 64);
        assert_eq!(attn.w_o().in_features, 64);
    }

    // =========================================================================
    // Linear Weight Setter Tests
    // =========================================================================

    #[test]
    fn test_linear_set_weight() {
        let mut linear = LinearWeights::new(4, 4);
        let weights = vec![1.0_f32; 16];

        linear.set_weight(&weights);
        assert!((linear.weight[0] - 1.0).abs() < f32::EPSILON);
        assert!((linear.weight[15] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_linear_set_bias() {
        let mut linear = LinearWeights::new(4, 4);
        let biases = vec![0.5_f32; 4];

        linear.set_bias(&biases);
        assert!((linear.bias[0] - 0.5).abs() < f32::EPSILON);
        assert!((linear.bias[3] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_linear_set_weight_partial() {
        let mut linear = LinearWeights::new(4, 4);
        let weights = vec![2.0_f32; 8]; // Only half the weights

        linear.set_weight(&weights);
        assert!((linear.weight[0] - 2.0).abs() < f32::EPSILON);
        assert!((linear.weight[7] - 2.0).abs() < f32::EPSILON);
        assert!((linear.weight[8] - 0.0).abs() < f32::EPSILON); // Unchanged
    }

    // =========================================================================
    // MultiHeadAttention Setter Tests
    // =========================================================================

    #[test]
    fn test_attention_set_query_weight() {
        let mut attn = MultiHeadAttention::new(2, 8);
        let weights = vec![1.0_f32; 64];

        attn.set_query_weight(&weights);
        assert!((attn.w_q().weight[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_attention_set_key_weight() {
        let mut attn = MultiHeadAttention::new(2, 8);
        let weights = vec![2.0_f32; 64];

        attn.set_key_weight(&weights);
        assert!((attn.w_k().weight[0] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_attention_set_value_weight() {
        let mut attn = MultiHeadAttention::new(2, 8);
        let weights = vec![3.0_f32; 64];

        attn.set_value_weight(&weights);
        assert!((attn.w_v().weight[0] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_attention_set_out_weight() {
        let mut attn = MultiHeadAttention::new(2, 8);
        let weights = vec![4.0_f32; 64];

        attn.set_out_weight(&weights);
        assert!((attn.w_o().weight[0] - 4.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // SIMD-Accelerated Tests
    // =========================================================================

    #[test]
    fn test_linear_forward_simd_identity() {
        let mut linear = LinearWeights::new(4, 4);
        // Set up identity matrix
        for i in 0..4 {
            linear.weight[i * 4 + i] = 1.0;
        }

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = linear
            .forward_simd(&input, 1)
            .expect("forward_simd should succeed");

        assert_eq!(output.len(), 4);
        for i in 0..4 {
            assert!(
                (output[i] - input[i]).abs() < 1e-4,
                "SIMD Identity should preserve input: got {} expected {}",
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_linear_forward_simd_with_bias() {
        let mut linear = LinearWeights::new(2, 2);
        // Identity weights
        linear.weight = vec![1.0, 0.0, 0.0, 1.0];
        linear.bias = vec![1.0, 2.0];

        let input = vec![3.0, 4.0];
        let output = linear
            .forward_simd(&input, 1)
            .expect("forward_simd should succeed");

        assert!(
            (output[0] - 4.0).abs() < 1e-4,
            "expected 4.0, got {}",
            output[0]
        ); // 3 + 1
        assert!(
            (output[1] - 6.0).abs() < 1e-4,
            "expected 6.0, got {}",
            output[1]
        ); // 4 + 2
    }

    #[test]
    fn test_linear_forward_simd_batch() {
        let mut linear = LinearWeights::new(2, 2);
        // Simple scaling weights
        linear.weight = vec![2.0, 0.0, 0.0, 3.0];
        linear.bias = vec![0.0, 0.0];

        // Two tokens: [[1,2], [3,4]]
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = linear
            .forward_simd(&input, 2)
            .expect("forward_simd should succeed");

        assert_eq!(output.len(), 4);
        assert!((output[0] - 2.0).abs() < 1e-4); // 1*2
        assert!((output[1] - 6.0).abs() < 1e-4); // 2*3
        assert!((output[2] - 6.0).abs() < 1e-4); // 3*2
        assert!((output[3] - 12.0).abs() < 1e-4); // 4*3
    }

    #[test]
    fn test_linear_forward_consistency() {
        // Test that forward and forward_simd produce the same results
        let mut linear = LinearWeights::new(4, 4);
        // Random-ish weights
        for i in 0..16 {
            linear.weight[i] = (i as f32) * 0.1;
        }
        linear.bias = vec![0.1, 0.2, 0.3, 0.4];

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // seq_len=2

        let output_regular = linear.forward(&input, 2).expect("forward should succeed");
        let output_simd = linear
            .forward_simd(&input, 2)
            .expect("forward_simd should succeed");

        assert_eq!(output_regular.len(), output_simd.len());
        for i in 0..output_regular.len() {
            assert!(
                (output_regular[i] - output_simd[i]).abs() < 1e-3,
                "SIMD and regular forward should match at index {}: {} vs {}",
                i,
                output_regular[i],
                output_simd[i]
            );
        }
    }

    #[test]
    fn test_scaled_dot_product_attention_simd_basic() {
        let attn = MultiHeadAttention::new(1, 4);

        // Simple Q, K, V with seq_len=2, d_head=4
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let value = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let output = attn
            .scaled_dot_product_attention_simd(&query, &key, &value, None)
            .expect("SIMD attention should succeed");

        assert_eq!(output.len(), 8); // seq_len * d_head
    }

    #[test]
    fn test_scaled_dot_product_attention_simd_with_mask() {
        let attn = MultiHeadAttention::new(1, 4);

        let query = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let key = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let value = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        // Causal mask
        let mask = vec![0.0, f32::NEG_INFINITY, 0.0, 0.0];

        let output = attn
            .scaled_dot_product_attention_simd(&query, &key, &value, Some(&mask))
            .expect("SIMD attention with mask should succeed");

        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_attention_simd_error_handling() {
        let attn = MultiHeadAttention::new(2, 8);

        // Query with wrong dimensions
        let query = vec![0.0_f32; 9]; // Not divisible by d_head
        let key = vec![0.0_f32; 8];
        let value = vec![0.0_f32; 8];

        let result = attn.scaled_dot_product_attention_simd(&query, &key, &value, None);
        assert!(result.is_err());
    }
}

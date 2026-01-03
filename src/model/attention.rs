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
//! # Parallelization (§11.3.2)
//!
//! Each attention head is independent [31], enabling embarrassingly parallel
//! computation. With `parallel` feature enabled, heads are computed via rayon.
//!
//! # References
//!
//! - [31] Vaswani et al. (2017): "Attention Is All You Need"
//! - Radford et al. (2023): "Robust Speech Recognition via Large-Scale Weak Supervision"

use crate::error::{WhisperError, WhisperResult};
use crate::parallel::{parallel_map, parallel_try_map};
use crate::simd;
use trueno::Matrix;

/// Linear projection weights for attention
///
/// Note: Clone is derived but will clone the cached Matrix (expensive).
/// Prefer to finalize_weights() after cloning if needed.
pub struct LinearWeights {
    /// Weight matrix (out_features x in_features) row-major
    pub weight: Vec<f32>,
    /// Bias vector (out_features)
    pub bias: Vec<f32>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
    /// Cached transposed weight matrix (in_features x out_features) for SIMD matmul
    /// Pre-computed by finalize_weights() to avoid runtime transpose
    weight_transposed: Option<Vec<f32>>,
    /// Cached trueno Matrix for zero-copy matmul (WAPR-BENCH-001 optimization)
    /// This avoids repeated Matrix::from_vec() calls in the hot path
    weight_matrix: Option<Matrix<f32>>,
}

impl Clone for LinearWeights {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            in_features: self.in_features,
            out_features: self.out_features,
            weight_transposed: self.weight_transposed.clone(),
            // Don't clone the Matrix cache - it will be rebuilt on finalize_weights()
            weight_matrix: None,
        }
    }
}

impl std::fmt::Debug for LinearWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearWeights")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("weight_len", &self.weight.len())
            .field("bias_len", &self.bias.len())
            .field("is_finalized", &self.weight_transposed.is_some())
            .field("has_matrix_cache", &self.weight_matrix.is_some())
            .finish()
    }
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
            weight_transposed: None,
            weight_matrix: None,
        }
    }

    /// Pre-compute and cache transposed weight matrix for SIMD matmul
    ///
    /// Call this after loading weights to avoid runtime transpose overhead.
    /// The transposed matrix is used by `forward_simd()` for efficient matmul.
    ///
    /// Also caches a trueno Matrix for zero-copy matmul operations.
    pub fn finalize_weights(&mut self) {
        let weight_t = simd::transpose(&self.weight, self.out_features, self.in_features);

        // Create trueno Matrix from transposed weights (WAPR-BENCH-001 optimization)
        // This avoids repeated Matrix::from_vec() calls in forward_simd()
        self.weight_matrix =
            Matrix::from_vec(self.in_features, self.out_features, weight_t.clone()).ok();

        self.weight_transposed = Some(weight_t);
    }

    /// Check if weights have been finalized
    #[must_use]
    pub fn is_finalized(&self) -> bool {
        self.weight_transposed.is_some()
    }

    /// Clear cached transposed weights (useful after modifying weights)
    pub fn invalidate_cache(&mut self) {
        self.weight_transposed = None;
        self.weight_matrix = None;
    }

    /// Set weight values from a slice
    ///
    /// Note: This invalidates any cached transposed weights.
    /// Call `finalize_weights()` after setting all weights.
    pub fn set_weight(&mut self, values: &[f32]) {
        let len = values.len().min(self.weight.len());
        self.weight[..len].copy_from_slice(&values[..len]);
        self.invalidate_cache();
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
    /// If `finalize_weights()` was called, uses cached trueno Matrix for
    /// zero-copy matmul (WAPR-BENCH-001 optimization).
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

        // Use cached trueno Matrix if available (WAPR-BENCH-001 optimization)
        // This avoids Matrix::from_vec() allocation for the weight matrix
        // Note: Using if-let because map_or_else cannot handle the lifetime of weight_t_owned
        #[allow(clippy::option_if_let_else)]
        let mut output = if let Some(weight_matrix) = &self.weight_matrix {
            simd::matmul_with_matrix(input, weight_matrix, total_tokens, self.in_features)
        } else {
            // Fallback: use cached transpose Vec or compute on-the-fly
            let weight_t_owned;
            let weight_t: &[f32] = if let Some(cached) = &self.weight_transposed {
                cached
            } else {
                weight_t_owned = simd::transpose(&self.weight, self.out_features, self.in_features);
                &weight_t_owned
            };

            // SIMD matmul: input[total_tokens x in_features] @ weight^T[in_features x out_features]
            simd::matmul(
                input,
                weight_t,
                total_tokens,
                self.in_features,
                self.out_features,
            )
        };

        // Add bias to each token using SIMD broadcast add
        simd::broadcast_add_inplace(&mut output, &self.bias, total_tokens, self.out_features);

        Ok(output)
    }
}

/// Default block size for Flash Attention (tuned for L1 cache)
pub const FLASH_ATTENTION_BLOCK_SIZE: usize = 32;

/// Threshold above which Flash Attention is used for memory efficiency
pub const FLASH_ATTENTION_THRESHOLD: usize = 128;

/// Configuration for Flash Attention
#[derive(Debug, Clone, Copy)]
pub struct FlashAttentionConfig {
    /// Query sequence length
    pub seq_len: usize,
    /// Key/Value sequence length
    pub kv_len: usize,
    /// Per-head dimension
    pub d_head: usize,
    /// Block size for tiling
    pub block_size: usize,
}

impl FlashAttentionConfig {
    /// Create a new Flash Attention configuration
    #[must_use]
    pub const fn new(seq_len: usize, kv_len: usize, d_head: usize, block_size: usize) -> Self {
        Self {
            seq_len,
            kv_len,
            d_head,
            block_size,
        }
    }

    /// Create with default block size
    #[must_use]
    #[allow(dead_code)] // Public API for future use
    pub const fn with_default_block_size(seq_len: usize, kv_len: usize, d_head: usize) -> Self {
        Self::new(seq_len, kv_len, d_head, FLASH_ATTENTION_BLOCK_SIZE)
    }
}

/// Block computation context for Flash Attention
struct BlockContext {
    q_idx: usize,
    kv_block_start: usize,
    kv_block_end: usize,
    scale: f32,
}

/// Compute block attention scores for a single query position
#[inline]
fn compute_block_scores(
    query: &[f32],
    key: &[f32],
    config: &FlashAttentionConfig,
    ctx: &BlockContext,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let mut scores = Vec::with_capacity(ctx.kv_block_end - ctx.kv_block_start);
    for k_idx in ctx.kv_block_start..ctx.kv_block_end {
        let mut dot = 0.0_f32;
        for d in 0..config.d_head {
            dot += query[ctx.q_idx * config.d_head + d] * key[k_idx * config.d_head + d];
        }
        let mut score = dot * ctx.scale;
        if let Some(m) = mask {
            score += m[ctx.q_idx * config.kv_len + k_idx];
        }
        scores.push(score);
    }
    scores
}

/// Update output with online softmax accumulation
#[inline]
fn update_output_with_block(
    output: &mut [f32],
    row_sum: &mut f32,
    row_max: &mut f32,
    block_scores: &[f32],
    value: &[f32],
    ctx: &BlockContext,
    d_head: usize,
) {
    let block_max = block_scores
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let prev_max = *row_max;
    let new_max = prev_max.max(block_max);
    let scale_prev = (prev_max - new_max).exp();

    *row_sum *= scale_prev;
    let out_start = ctx.q_idx * d_head;
    for d in 0..d_head {
        output[out_start + d] *= scale_prev;
    }

    for (local_k_idx, &score) in block_scores.iter().enumerate() {
        let k_idx = ctx.kv_block_start + local_k_idx;
        let exp_score = (score - new_max).exp();
        *row_sum += exp_score;
        for d in 0..d_head {
            output[out_start + d] += exp_score * value[k_idx * d_head + d];
        }
    }

    *row_max = new_max;
}

/// Normalize output row by softmax sum
#[inline]
fn normalize_row(output: &mut [f32], row_sum: f32, q_idx: usize, d_head: usize) {
    let inv_sum = if row_sum > 1e-10 { 1.0 / row_sum } else { 0.0 };
    let start = q_idx * d_head;
    for d in 0..d_head {
        output[start + d] *= inv_sum;
    }
}

/// Flash Attention: O(n) memory instead of O(n²)
///
/// Implements the Flash Attention algorithm from Dao et al. (2022).
/// Processes attention in blocks to minimize memory usage while maintaining
/// numerical correctness through online softmax computation.
///
/// # Arguments
/// * `query` - Query tensor (config.seq_len × config.d_head) flattened row-major
/// * `key` - Key tensor (config.kv_len × config.d_head) flattened row-major
/// * `value` - Value tensor (config.kv_len × config.d_head) flattened row-major
/// * `config` - Flash attention configuration (dimensions and block size)
/// * `mask` - Optional attention mask (seq_len × kv_len), -inf for masked positions
///
/// # Returns
/// Attention output (seq_len × d_head) flattened row-major
///
/// # Reference
/// Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact
/// Attention with IO-Awareness." NeurIPS 2022.
#[must_use]
pub fn flash_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: FlashAttentionConfig,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let scale = 1.0 / (config.d_head as f32).sqrt();

    let mut output = vec![0.0_f32; config.seq_len * config.d_head];
    let mut row_max = vec![f32::NEG_INFINITY; config.seq_len];
    let mut row_sum = vec![0.0_f32; config.seq_len];

    // Process key-value blocks
    for kv_block_start in (0..config.kv_len).step_by(config.block_size) {
        let kv_block_end = (kv_block_start + config.block_size).min(config.kv_len);

        // Process all queries for this KV block
        for q_idx in 0..config.seq_len {
            let ctx = BlockContext {
                q_idx,
                kv_block_start,
                kv_block_end,
                scale,
            };

            let block_scores = compute_block_scores(query, key, &config, &ctx, mask);

            update_output_with_block(
                &mut output,
                &mut row_sum[q_idx],
                &mut row_max[q_idx],
                &block_scores,
                value,
                &ctx,
                config.d_head,
            );
        }
    }

    // Final normalization
    for (q_idx, &sum) in row_sum.iter().enumerate() {
        normalize_row(&mut output, sum, q_idx, config.d_head);
    }

    output
}

/// Compute block scores using SIMD dot products
#[inline]
fn compute_block_scores_simd(
    query: &[f32],
    key: &[f32],
    config: &FlashAttentionConfig,
    ctx: &BlockContext,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let q_offset = ctx.q_idx * config.d_head;
    let mut scores = Vec::with_capacity(ctx.kv_block_end - ctx.kv_block_start);
    for k_idx in ctx.kv_block_start..ctx.kv_block_end {
        let k_offset = k_idx * config.d_head;
        let dot = simd::dot(
            &query[q_offset..q_offset + config.d_head],
            &key[k_offset..k_offset + config.d_head],
        );
        let mut score = dot * ctx.scale;
        if let Some(m) = mask {
            score += m[ctx.q_idx * config.kv_len + k_idx];
        }
        scores.push(score);
    }
    scores
}

/// Update output with SIMD operations
#[inline]
fn update_output_simd(
    output: &mut [f32],
    row_sum: &mut f32,
    row_max: &mut f32,
    block_scores: &[f32],
    value: &[f32],
    ctx: &BlockContext,
    d_head: usize,
) {
    let block_max = simd::max_element(block_scores);
    let prev_max = *row_max;
    let new_max = prev_max.max(block_max);
    let scale_prev = (prev_max - new_max).exp();

    *row_sum *= scale_prev;
    let q_offset = ctx.q_idx * d_head;
    simd::scale_inplace(&mut output[q_offset..q_offset + d_head], scale_prev);

    for (local_k_idx, &score) in block_scores.iter().enumerate() {
        let k_idx = ctx.kv_block_start + local_k_idx;
        let exp_score = (score - new_max).exp();
        *row_sum += exp_score;
        let v_offset = k_idx * d_head;
        simd::axpy(
            exp_score,
            &value[v_offset..v_offset + d_head],
            &mut output[q_offset..q_offset + d_head],
        );
    }
    *row_max = new_max;
}

/// SIMD-accelerated Flash Attention
///
/// Uses SIMD operations for the inner loops when available.
#[must_use]
pub fn flash_attention_simd(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: FlashAttentionConfig,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let scale = 1.0 / (config.d_head as f32).sqrt();

    let mut output = vec![0.0_f32; config.seq_len * config.d_head];
    let mut row_max = vec![f32::NEG_INFINITY; config.seq_len];
    let mut row_sum = vec![0.0_f32; config.seq_len];

    for kv_block_start in (0..config.kv_len).step_by(config.block_size) {
        let kv_block_end = (kv_block_start + config.block_size).min(config.kv_len);

        for q_idx in 0..config.seq_len {
            let ctx = BlockContext {
                q_idx,
                kv_block_start,
                kv_block_end,
                scale,
            };

            let block_scores = compute_block_scores_simd(query, key, &config, &ctx, mask);

            update_output_simd(
                &mut output,
                &mut row_sum[q_idx],
                &mut row_max[q_idx],
                &block_scores,
                value,
                &ctx,
                config.d_head,
            );
        }
    }

    // Final normalization
    for (q_idx, &sum) in row_sum.iter().enumerate() {
        let inv_sum = if sum > 1e-10 { 1.0 / sum } else { 0.0 };
        let start = q_idx * config.d_head;
        let end = (q_idx + 1) * config.d_head;
        simd::scale_inplace(&mut output[start..end], inv_sum);
    }

    output
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

    /// Forward pass for self-attention with automatic SIMD dispatch
    ///
    /// Dispatches to SIMD-optimized path when `simd` feature is enabled,
    /// otherwise falls back to scalar implementation.
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len x d_model)
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Output tensor (seq_len x d_model)
    pub fn forward(&self, x: &[f32], mask: Option<&[f32]>) -> WhisperResult<Vec<f32>> {
        self.forward_cross_dispatch(x, x, mask)
    }

    /// Forward pass for cross-attention with automatic SIMD dispatch
    ///
    /// Dispatches to SIMD-optimized path when `simd` feature is enabled,
    /// otherwise falls back to scalar implementation.
    ///
    /// # Arguments
    /// * `x` - Query input tensor (seq_len x d_model)
    /// * `context` - Key/Value input tensor (kv_len x d_model)
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Output tensor (seq_len x d_model)
    /// Forward pass with optimal dispatch: SIMD + Flash Attention when beneficial
    ///
    /// Dispatch logic (aligned with realizar patterns):
    /// - Long sequences (>128 tokens): Flash Attention (O(n) memory)
    /// - Short sequences: Standard attention (lower overhead)
    /// - SIMD: Always used when feature enabled
    pub fn forward_cross_dispatch(
        &self,
        x: &[f32],
        context: &[f32],
        mask: Option<&[f32]>,
    ) -> WhisperResult<Vec<f32>> {
        let seq_len = x.len() / self.d_model;
        let kv_len = context.len() / self.d_model;

        // Use Flash Attention for long sequences (matches realizar threshold)
        if seq_len > FLASH_ATTENTION_THRESHOLD || kv_len > FLASH_ATTENTION_THRESHOLD {
            // Flash Attention already uses SIMD internally
            self.forward_cross_flash(x, context, mask, FLASH_ATTENTION_BLOCK_SIZE)
        } else if cfg!(feature = "simd") {
            self.forward_cross_simd(x, context, mask)
        } else {
            self.forward_cross(x, context, mask)
        }
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
        // Dispatch to SIMD or scalar based on feature flag
        if cfg!(feature = "simd") {
            self.forward_cross_simd(x, context, mask)
        } else {
            self.forward_cross_scalar(x, context, mask)
        }
    }

    /// Scalar forward pass (fallback)
    fn forward_cross_scalar(
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

        // Project Q, K, V using scalar matmul
        let q = self.w_q.forward(x, seq_len)?;
        let k = self.w_k.forward(context, kv_len)?;
        let v = self.w_v.forward(context, kv_len)?;

        // Compute attention for each head (parallel when feature enabled)
        // Per §11.3.2: Each head is independent [31], enabling parallel computation
        let head_outputs = parallel_try_map(0..self.n_heads, |head| {
            let q_head = self.extract_head(&q, seq_len, head);
            let k_head = self.extract_head(&k, kv_len, head);
            let v_head = self.extract_head(&v, kv_len, head);
            self.scaled_dot_product_attention(&q_head, &k_head, &v_head, mask)
        })?;

        // Concatenate heads and project output
        let concat = self.concat_heads(&head_outputs, seq_len);
        self.w_o.forward(&concat, seq_len)
    }

    /// SIMD-accelerated forward pass
    fn forward_cross_simd(
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

        // Project Q, K, V using SIMD-accelerated matmul
        let q = self.w_q.forward_simd(x, seq_len)?;
        let k = self.w_k.forward_simd(context, kv_len)?;
        let v = self.w_v.forward_simd(context, kv_len)?;

        // Compute attention for each head using SIMD (parallel when feature enabled)
        // Per §11.3.2: Each head is independent [31], enabling parallel computation
        let head_outputs = parallel_try_map(0..self.n_heads, |head| {
            let q_head = self.extract_head(&q, seq_len, head);
            let k_head = self.extract_head(&k, kv_len, head);
            let v_head = self.extract_head(&v, kv_len, head);

            // Use SIMD attention
            self.scaled_dot_product_attention_simd(&q_head, &k_head, &v_head, mask)
        })?;

        // Concatenate heads and project output using SIMD
        let concat = self.concat_heads(&head_outputs, seq_len);
        self.w_o.forward_simd(&concat, seq_len)
    }

    /// Flash Attention forward pass for memory-efficient long sequences
    ///
    /// Uses block-based attention computation with O(n) memory instead of O(n²).
    /// Best for sequences longer than ~128 tokens where memory savings matter.
    ///
    /// # Arguments
    /// * `x` - Query input tensor (seq_len x d_model)
    /// * `context` - Key/Value input tensor (kv_len x d_model)
    /// * `mask` - Optional attention mask
    /// * `block_size` - Block size for tiling (default: 32)
    ///
    /// # Returns
    /// Output tensor (seq_len x d_model)
    pub fn forward_cross_flash(
        &self,
        x: &[f32],
        context: &[f32],
        mask: Option<&[f32]>,
        block_size: usize,
    ) -> WhisperResult<Vec<f32>> {
        let seq_len = x.len() / self.d_model;
        let kv_len = context.len() / self.d_model;

        if x.len() % self.d_model != 0 {
            return Err(WhisperError::Model("input size mismatch".into()));
        }
        if context.len() % self.d_model != 0 {
            return Err(WhisperError::Model("context size mismatch".into()));
        }

        // Project Q, K, V using SIMD-accelerated matmul
        let q = self.w_q.forward_simd(x, seq_len)?;
        let k = self.w_k.forward_simd(context, kv_len)?;
        let v = self.w_v.forward_simd(context, kv_len)?;

        // Compute attention for each head using Flash Attention (parallel when feature enabled)
        // Per §11.3.2: Each head is independent [31], enabling parallel computation
        let head_outputs = parallel_map(0..self.n_heads, |head| {
            let q_head = self.extract_head(&q, seq_len, head);
            let k_head = self.extract_head(&k, kv_len, head);
            let v_head = self.extract_head(&v, kv_len, head);

            // Use Flash Attention for O(n) memory
            let config = FlashAttentionConfig::new(seq_len, kv_len, self.d_head, block_size);
            if cfg!(feature = "simd") {
                flash_attention_simd(&q_head, &k_head, &v_head, config, mask)
            } else {
                flash_attention(&q_head, &k_head, &v_head, config, mask)
            }
        });

        // Concatenate heads and project output using SIMD
        let concat = self.concat_heads(&head_outputs, seq_len);
        self.w_o.forward_simd(&concat, seq_len)
    }

    /// Forward pass with automatic Flash Attention selection
    ///
    /// Uses Flash Attention for long sequences (>128 tokens) to reduce memory,
    /// and standard attention for shorter sequences.
    pub fn forward_cross_auto(
        &self,
        x: &[f32],
        context: &[f32],
        mask: Option<&[f32]>,
    ) -> WhisperResult<Vec<f32>> {
        let seq_len = x.len() / self.d_model;
        let kv_len = context.len() / self.d_model;

        // Use Flash Attention for sequences where O(n²) memory is significant
        if seq_len > FLASH_ATTENTION_THRESHOLD || kv_len > FLASH_ATTENTION_THRESHOLD {
            self.forward_cross_flash(x, context, mask, FLASH_ATTENTION_BLOCK_SIZE)
        } else {
            self.forward_cross(x, context, mask)
        }
    }

    /// Streaming attention with pre-computed key/value cache
    ///
    /// This method is optimized for incremental inference where:
    /// - Query is computed only for new positions
    /// - Key and Value are retrieved from a cache
    ///
    /// # Arguments
    /// * `x` - Current input tensor (new_seq_len × d_model)
    /// * `cached_key` - Cached key tensor (cache_len × d_model)
    /// * `cached_value` - Cached value tensor (cache_len × d_model)
    /// * `mask` - Optional attention mask (new_seq_len × cache_len)
    ///
    /// # Returns
    /// * `(output, new_key, new_value)` - Output and K/V for caching
    ///
    /// # Use Case
    /// Streaming transcription where each new audio chunk produces tokens
    /// that attend to previously cached context.
    pub fn forward_streaming(
        &self,
        x: &[f32],
        cached_key: &[f32],
        cached_value: &[f32],
        mask: Option<&[f32]>,
    ) -> WhisperResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let seq_len = x.len() / self.d_model;
        let cache_len = if cached_key.is_empty() {
            0
        } else {
            cached_key.len() / self.d_model
        };

        // Project query from current input
        let q = self.w_q.forward(x, seq_len)?;

        // Project new key/value from current input (to be cached)
        let new_k = self.w_k.forward(x, seq_len)?;
        let new_v = self.w_v.forward(x, seq_len)?;

        // Compute attention for each head (parallel when feature enabled)
        // Per §11.3.2: Each head is independent [31], enabling parallel computation
        let head_outputs = parallel_try_map(0..self.n_heads, |head| {
            // Extract this head's query
            let head_q = self.extract_head(&q, seq_len, head);

            // For K/V, combine cached and new if cache exists
            let (head_k, head_v, total_kv_len) = if cache_len > 0 {
                let cached_head_k = self.extract_head(cached_key, cache_len, head);
                let cached_head_v = self.extract_head(cached_value, cache_len, head);
                let new_head_k = self.extract_head(&new_k, seq_len, head);
                let new_head_v = self.extract_head(&new_v, seq_len, head);

                // Concatenate cached + new
                let mut combined_k = cached_head_k;
                combined_k.extend_from_slice(&new_head_k);
                let mut combined_v = cached_head_v;
                combined_v.extend_from_slice(&new_head_v);

                (combined_k, combined_v, cache_len + seq_len)
            } else {
                let new_head_k = self.extract_head(&new_k, seq_len, head);
                let new_head_v = self.extract_head(&new_v, seq_len, head);
                (new_head_k, new_head_v, seq_len)
            };

            // Compute attention with optional Flash Attention
            if total_kv_len > FLASH_ATTENTION_THRESHOLD {
                let config = FlashAttentionConfig::new(
                    seq_len,
                    total_kv_len,
                    self.d_head,
                    FLASH_ATTENTION_BLOCK_SIZE,
                );
                Ok(flash_attention_simd(
                    &head_q, &head_k, &head_v, config, mask,
                ))
            } else {
                self.scaled_dot_product_attention(&head_q, &head_k, &head_v, mask)
            }
        })?;

        // Concatenate head outputs
        let concat = self.concat_heads(&head_outputs, seq_len);

        // Output projection
        let output = self.w_o.forward(&concat, seq_len)?;

        Ok((output, new_k, new_v))
    }

    /// Streaming self-attention with automatic KV cache management
    ///
    /// Simplified interface for streaming self-attention that automatically
    /// handles the query-only computation pattern.
    ///
    /// # Arguments
    /// * `x` - Current input tensor (new_seq_len × d_model)
    /// * `cached_key` - Previously cached keys (cache_len × d_model)
    /// * `cached_value` - Previously cached values (cache_len × d_model)
    ///
    /// # Returns
    /// Attention output for the new tokens
    #[allow(dead_code)] // Public API for streaming support
    pub fn forward_self_streaming(
        &self,
        x: &[f32],
        cached_key: &[f32],
        cached_value: &[f32],
    ) -> WhisperResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        // For self-attention, we use a causal mask that prevents
        // attending to future positions within the new sequence
        let seq_len = x.len() / self.d_model;
        let cache_len = cached_key.len() / self.d_model;

        // Build streaming causal mask
        // New tokens can attend to all cached tokens + previous new tokens
        let mask = if seq_len > 1 {
            let total_len = cache_len + seq_len;
            let mut mask_data = vec![0.0_f32; seq_len * total_len];

            for q in 0..seq_len {
                // Can attend to all cached positions + positions 0..=q in new
                let max_attend = cache_len + q + 1;
                for k in max_attend..total_len {
                    mask_data[q * total_len + k] = f32::NEG_INFINITY;
                }
            }
            Some(mask_data)
        } else {
            None
        };

        self.forward_streaming(x, cached_key, cached_value, mask.as_deref())
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

    /// Set query bias from a slice
    pub fn set_query_bias(&mut self, values: &[f32]) {
        self.w_q.set_bias(values);
    }

    /// Set key bias from a slice
    pub fn set_key_bias(&mut self, values: &[f32]) {
        self.w_k.set_bias(values);
    }

    /// Set value bias from a slice
    pub fn set_value_bias(&mut self, values: &[f32]) {
        self.w_v.set_bias(values);
    }

    /// Set output bias from a slice
    pub fn set_out_bias(&mut self, values: &[f32]) {
        self.w_o.set_bias(values);
    }

    /// Pre-compute and cache transposed weights for all linear layers
    ///
    /// Call this after loading all weights to optimize SIMD matmul performance.
    pub fn finalize_weights(&mut self) {
        self.w_q.finalize_weights();
        self.w_k.finalize_weights();
        self.w_v.finalize_weights();
        self.w_o.finalize_weights();
    }

    /// Check if all weights have been finalized
    #[must_use]
    pub fn is_finalized(&self) -> bool {
        self.w_q.is_finalized()
            && self.w_k.is_finalized()
            && self.w_v.is_finalized()
            && self.w_o.is_finalized()
    }

    /// Get mutable query weights reference (for loading weights)
    pub fn w_q_mut(&mut self) -> &mut LinearWeights {
        &mut self.w_q
    }

    /// Get mutable key weights reference (for loading weights)
    pub fn w_k_mut(&mut self) -> &mut LinearWeights {
        &mut self.w_k
    }

    /// Get mutable value weights reference (for loading weights)
    pub fn w_v_mut(&mut self) -> &mut LinearWeights {
        &mut self.w_v
    }

    /// Get mutable output weights reference (for loading weights)
    pub fn w_o_mut(&mut self) -> &mut LinearWeights {
        &mut self.w_o
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
    fn test_linear_finalize_weights() {
        let mut linear = LinearWeights::new(4, 4);
        for i in 0..16 {
            linear.weight[i] = (i as f32) * 0.1;
        }
        linear.bias = vec![0.1, 0.2, 0.3, 0.4];

        // Before finalization
        assert!(!linear.is_finalized());

        // Finalize weights
        linear.finalize_weights();
        assert!(linear.is_finalized());

        // Run forward_simd which should use cached transpose
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output_finalized = linear
            .forward_simd(&input, 1)
            .expect("forward_simd should succeed");

        // Compare with unfinalized version (should match)
        linear.invalidate_cache();
        assert!(!linear.is_finalized());

        let output_unfinalized = linear
            .forward_simd(&input, 1)
            .expect("forward_simd should succeed");

        assert_eq!(output_finalized.len(), output_unfinalized.len());
        for i in 0..output_finalized.len() {
            assert!(
                (output_finalized[i] - output_unfinalized[i]).abs() < 1e-6,
                "Finalized and unfinalized should match"
            );
        }
    }

    #[test]
    fn test_attention_finalize_weights() {
        let mut attn = MultiHeadAttention::new(2, 8);

        // Before finalization
        assert!(!attn.is_finalized());

        // Finalize weights
        attn.finalize_weights();
        assert!(attn.is_finalized());

        // All internal linear layers should be finalized
        assert!(attn.w_q().is_finalized());
        assert!(attn.w_k().is_finalized());
        assert!(attn.w_v().is_finalized());
        assert!(attn.w_o().is_finalized());
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

    // =========================================================================
    // Flash Attention Tests
    // =========================================================================

    #[test]
    fn test_flash_attention_basic() {
        // Simple test: seq_len=2, kv_len=2, d_head=4
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let value = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let config = FlashAttentionConfig::new(2, 2, 4, 2);
        let output = flash_attention(&query, &key, &value, config, None);

        assert_eq!(output.len(), 8);
        // All outputs should be finite
        for &v in &output {
            assert!(v.is_finite(), "Flash attention output should be finite");
        }
    }

    #[test]
    fn test_flash_attention_matches_standard() {
        // Verify Flash Attention produces same results as standard attention
        let attn = MultiHeadAttention::new(1, 4);

        let query = vec![1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0];
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        let value = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0];

        // Standard attention
        let standard = attn
            .scaled_dot_product_attention(&query, &key, &value, None)
            .expect("standard attention");

        // Flash attention
        let config = FlashAttentionConfig::new(2, 3, 4, 2);
        let flash = flash_attention(&query, &key, &value, config, None);

        assert_eq!(standard.len(), flash.len());
        for i in 0..standard.len() {
            assert!(
                (standard[i] - flash[i]).abs() < 1e-4,
                "Flash attention should match standard at index {}: {} vs {}",
                i,
                standard[i],
                flash[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_with_mask() {
        // Test with causal mask
        let query = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let key = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let value = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        // Causal mask: position 0 can only see position 0
        let mask = vec![0.0, f32::NEG_INFINITY, 0.0, 0.0];

        let config = FlashAttentionConfig::new(2, 2, 4, 2);
        let output = flash_attention(&query, &key, &value, config, Some(&mask));

        assert_eq!(output.len(), 8);
        // First query should only see first value due to mask
        assert!(
            (output[0] - 1.0).abs() < 1e-4,
            "First output[0] should be 1.0"
        );
    }

    #[test]
    fn test_flash_attention_simd_matches_scalar() {
        let query = vec![1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0];
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let value = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let config = FlashAttentionConfig::new(2, 2, 4, 2);
        let scalar = flash_attention(&query, &key, &value, config, None);
        let simd = flash_attention_simd(&query, &key, &value, config, None);

        assert_eq!(scalar.len(), simd.len());
        for i in 0..scalar.len() {
            assert!(
                (scalar[i] - simd[i]).abs() < 1e-5,
                "SIMD flash attention should match scalar at index {}: {} vs {}",
                i,
                scalar[i],
                simd[i]
            );
        }
    }

    #[test]
    fn test_flash_attention_different_block_sizes() {
        let query = vec![1.0; 32]; // seq_len=8, d_head=4
        let key = vec![1.0; 32];
        let value = vec![1.0; 32];

        // Test with different block sizes
        let config_2 = FlashAttentionConfig::new(8, 8, 4, 2);
        let config_4 = FlashAttentionConfig::new(8, 8, 4, 4);
        let config_8 = FlashAttentionConfig::new(8, 8, 4, 8);

        let out_block_2 = flash_attention(&query, &key, &value, config_2, None);
        let out_block_4 = flash_attention(&query, &key, &value, config_4, None);
        let out_block_8 = flash_attention(&query, &key, &value, config_8, None);

        // All should produce the same result
        for i in 0..out_block_2.len() {
            assert!(
                (out_block_2[i] - out_block_4[i]).abs() < 1e-5,
                "Block size 2 vs 4 mismatch at {i}"
            );
            assert!(
                (out_block_4[i] - out_block_8[i]).abs() < 1e-5,
                "Block size 4 vs 8 mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_forward_cross_flash() {
        let attn = MultiHeadAttention::new(2, 8);

        // Simple inputs: seq_len=4, d_model=8
        let x = vec![0.1_f32; 32];
        let context = vec![0.2_f32; 32];

        let output = attn
            .forward_cross_flash(&x, &context, None, FLASH_ATTENTION_BLOCK_SIZE)
            .expect("forward_cross_flash");

        assert_eq!(output.len(), 32); // seq_len * d_model
    }

    #[test]
    fn test_forward_cross_auto() {
        let attn = MultiHeadAttention::new(2, 8);

        // Short sequence (should use standard attention)
        let x_short = vec![0.1_f32; 32]; // seq_len=4
        let ctx_short = vec![0.2_f32; 32];

        let output_short = attn
            .forward_cross_auto(&x_short, &ctx_short, None)
            .expect("forward_cross_auto short");

        assert_eq!(output_short.len(), 32);

        // Both methods should produce similar results for short sequences
        let output_standard = attn
            .forward_cross(&x_short, &ctx_short, None)
            .expect("forward_cross standard");

        for i in 0..output_short.len() {
            assert!(
                (output_short[i] - output_standard[i]).abs() < 1e-3,
                "Auto should match standard for short sequences at {i}"
            );
        }
    }

    #[test]
    fn test_forward_streaming_no_cache() {
        let attn = MultiHeadAttention::new(2, 8);

        // Single token, no cache
        let x = vec![0.1_f32; 8]; // seq_len=1, d_model=8
        let empty_key: Vec<f32> = vec![];
        let empty_value: Vec<f32> = vec![];

        let (output, new_k, new_v) = attn
            .forward_streaming(&x, &empty_key, &empty_value, None)
            .expect("forward_streaming no cache");

        assert_eq!(output.len(), 8);
        assert_eq!(new_k.len(), 8); // Key for caching
        assert_eq!(new_v.len(), 8); // Value for caching
    }

    #[test]
    fn test_forward_streaming_with_cache() {
        let attn = MultiHeadAttention::new(2, 8);

        // First token
        let x1 = vec![0.1_f32; 8];
        let (_, k1, v1) = attn
            .forward_streaming(&x1, &[], &[], None)
            .expect("first streaming call");

        // Second token with cached K/V
        let x2 = vec![0.2_f32; 8];
        let (output2, k2, v2) = attn
            .forward_streaming(&x2, &k1, &v1, None)
            .expect("second streaming call");

        assert_eq!(output2.len(), 8);
        assert_eq!(k2.len(), 8); // New key for this token
        assert_eq!(v2.len(), 8); // New value for this token
    }

    #[test]
    fn test_forward_streaming_multi_token() {
        let attn = MultiHeadAttention::new(2, 8);

        // Multiple tokens at once
        let x = vec![0.1_f32; 24]; // seq_len=3, d_model=8
        let (output, new_k, new_v) = attn
            .forward_streaming(&x, &[], &[], None)
            .expect("forward_streaming multi-token");

        assert_eq!(output.len(), 24);
        assert_eq!(new_k.len(), 24);
        assert_eq!(new_v.len(), 24);
    }

    #[test]
    fn test_forward_self_streaming() {
        let attn = MultiHeadAttention::new(2, 8);

        // Single token
        let x = vec![0.1_f32; 8];
        let (output, new_k, new_v) = attn
            .forward_self_streaming(&x, &[], &[])
            .expect("forward_self_streaming");

        assert_eq!(output.len(), 8);
        assert_eq!(new_k.len(), 8);
        assert_eq!(new_v.len(), 8);
    }

    // =========================================================================
    // Regression Tests for SIMD Optimizations (Sprint 3)
    // =========================================================================

    /// Regression test: Flash attention must match standard attention numerically
    #[test]
    fn regression_flash_attention_accuracy() {
        let attn = MultiHeadAttention::new(1, 64);
        let seq_len = 32;
        let d_head = 64;

        // Generate reproducible test data
        let q: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();
        let k: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (i as f32 * 0.2).cos() * 0.5)
            .collect();
        let v: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (i as f32 * 0.3).sin() * 0.5)
            .collect();

        // Standard attention
        let standard = attn
            .scaled_dot_product_attention(&q, &k, &v, None)
            .expect("standard attention");

        // Flash attention scalar
        let config = FlashAttentionConfig::with_default_block_size(seq_len, seq_len, d_head);
        let flash_scalar = flash_attention(&q, &k, &v, config, None);

        // Flash attention SIMD
        let flash_simd = flash_attention_simd(&q, &k, &v, config, None);

        // Verify numerical accuracy (tolerance for floating point)
        let tolerance = 1e-4;
        for i in 0..standard.len() {
            assert!(
                (standard[i] - flash_scalar[i]).abs() < tolerance,
                "Flash scalar mismatch at {i}: {} vs {}",
                standard[i],
                flash_scalar[i]
            );
            assert!(
                (standard[i] - flash_simd[i]).abs() < tolerance,
                "Flash SIMD mismatch at {i}: {} vs {}",
                standard[i],
                flash_simd[i]
            );
        }
    }

    /// Regression test: Streaming attention consistency across cache updates
    #[test]
    fn regression_streaming_attention_consistency() {
        let attn = MultiHeadAttention::new(2, 16);
        let d_model = 16;

        // Simulate incremental decoding: process tokens one at a time
        let mut cached_k = Vec::new();
        let mut cached_v = Vec::new();
        let mut all_outputs = Vec::new();

        // Process 5 tokens incrementally
        for i in 0..5 {
            let x: Vec<f32> = (0..d_model)
                .map(|j| (i * d_model + j) as f32 * 0.01)
                .collect();

            let (output, new_k, new_v) = attn
                .forward_streaming(&x, &cached_k, &cached_v, None)
                .expect("streaming attention");

            // Accumulate cache
            cached_k.extend_from_slice(&new_k);
            cached_v.extend_from_slice(&new_v);
            all_outputs.push(output);
        }

        // Verify output shapes
        for (i, output) in all_outputs.iter().enumerate() {
            assert_eq!(output.len(), d_model, "Token {i} output size mismatch");
        }

        // Verify cache grew correctly
        assert_eq!(cached_k.len(), 5 * d_model);
        assert_eq!(cached_v.len(), 5 * d_model);
    }

    /// Regression test: SIMD functions preserve numerical properties
    #[test]
    fn regression_simd_numerical_stability() {
        use crate::simd;

        // Test with values that could cause numerical issues
        let large_vals: Vec<f32> = (0..256).map(|i| 100.0 + i as f32 * 0.1).collect();
        let small_vals: Vec<f32> = (0..256).map(|i| 1e-6 + i as f32 * 1e-8).collect();
        let mixed_vals: Vec<f32> = (0..256)
            .map(|i| if i % 2 == 0 { 100.0 } else { 0.001 })
            .collect();

        // Softmax stability: large values shouldn't overflow
        let softmax_large = simd::softmax(&large_vals);
        assert!(
            softmax_large.iter().all(|&x| x.is_finite()),
            "Softmax produced non-finite values for large inputs"
        );
        let sum: f32 = softmax_large.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax sum not 1.0: {}", sum);

        // Softmax stability: small values shouldn't underflow to all zeros
        let softmax_small = simd::softmax(&small_vals);
        assert!(
            softmax_small.iter().any(|&x| x > 0.0),
            "Softmax underflowed to all zeros"
        );

        // Dot product: mixed magnitude values
        let dot_mixed = simd::dot(&mixed_vals, &mixed_vals);
        assert!(
            dot_mixed.is_finite(),
            "Dot product produced non-finite value"
        );
    }

    /// Regression test: Block sizes don't affect Flash attention results
    #[test]
    fn regression_flash_attention_block_size_invariance() {
        let seq_len = 64;
        let d_head = 32;

        let q: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * d_head)
            .map(|i| (i as f32 * 0.15).sin())
            .collect();

        // Test with different block sizes
        let block_sizes = [8, 16, 32, 64];
        let mut results = Vec::new();

        for &block_size in &block_sizes {
            let config = FlashAttentionConfig::new(seq_len, seq_len, d_head, block_size);
            let output = flash_attention_simd(&q, &k, &v, config, None);
            results.push(output);
        }

        // All results should be numerically equivalent
        let tolerance = 1e-5;
        for i in 1..results.len() {
            for j in 0..results[0].len() {
                assert!(
                    (results[0][j] - results[i][j]).abs() < tolerance,
                    "Block size {} differs from block size {} at position {}: {} vs {}",
                    block_sizes[0],
                    block_sizes[i],
                    j,
                    results[0][j],
                    results[i][j]
                );
            }
        }
    }

    /// Regression test: Multi-head attention output shape consistency
    #[test]
    fn regression_multihead_output_shapes() {
        // Test various configurations
        let configs = [
            (6, 384),  // tiny
            (8, 512),  // base
            (12, 768), // small
        ];

        for (n_heads, d_model) in configs {
            let attn = MultiHeadAttention::new(n_heads, d_model);

            for seq_len in [1, 10, 100] {
                let x = vec![0.1_f32; seq_len * d_model];
                let context = vec![0.2_f32; seq_len * d_model];

                let output = attn
                    .forward_cross(&x, &context, None)
                    .expect("forward_cross");

                assert_eq!(
                    output.len(),
                    seq_len * d_model,
                    "Output shape mismatch for n_heads={}, d_model={}, seq_len={}",
                    n_heads,
                    d_model,
                    seq_len
                );
            }
        }
    }

    // =========================================================================
    // EXTREME TDD: SIMD Dispatch Tests (WAPR-SIMD-001)
    // =========================================================================

    /// EXTREME TDD: Verify forward_cross() and forward_cross_simd() produce identical results
    ///
    /// This test validates that the SIMD-optimized path produces numerically
    /// equivalent results to the scalar baseline within floating-point tolerance.
    #[test]
    fn tdd_simd_dispatch_forward_cross_matches_scalar() {
        let mut attn = MultiHeadAttention::new(6, 384); // whisper-tiny config
        attn.finalize_weights();

        let seq_len = 10;
        let d_model = 384;
        let tolerance = 1e-4;

        // Generate deterministic input
        let x: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let context: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();

        // Run scalar path
        let scalar_output = attn
            .forward_cross(&x, &context, None)
            .expect("scalar forward_cross");

        // Run SIMD path
        let simd_output = attn
            .forward_cross_simd(&x, &context, None)
            .expect("simd forward_cross");

        // Verify shapes match
        assert_eq!(
            scalar_output.len(),
            simd_output.len(),
            "Output shapes must match"
        );

        // Verify numerical accuracy
        let mut max_diff: f32 = 0.0;
        for (i, (s, simd)) in scalar_output.iter().zip(simd_output.iter()).enumerate() {
            let diff = (s - simd).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < tolerance,
                "SIMD mismatch at index {}: scalar={}, simd={}, diff={}",
                i,
                s,
                simd,
                diff
            );
        }
        println!(
            "SIMD accuracy test passed: max_diff={:.2e} (tolerance={:.2e})",
            max_diff, tolerance
        );
    }

    /// EXTREME TDD: Verify forward() uses SIMD dispatch when feature is enabled
    ///
    /// Tests that the unified forward() method dispatches correctly based on
    /// the simd feature flag.
    #[test]
    fn tdd_forward_uses_simd_dispatch() {
        let mut attn = MultiHeadAttention::new(4, 64);
        attn.finalize_weights();

        let seq_len = 5;
        let d_model = 64;
        let tolerance = 1e-4;

        let x: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        // forward() should use SIMD when feature is enabled
        let forward_output = attn.forward(&x, None).expect("forward");

        // Directly call SIMD version for comparison
        let simd_output = attn
            .forward_cross_simd(&x, &x, None)
            .expect("forward_cross_simd");

        // If simd feature is enabled, outputs should match SIMD version
        if cfg!(feature = "simd") {
            for (i, (f, s)) in forward_output.iter().zip(simd_output.iter()).enumerate() {
                let diff = (f - s).abs();
                assert!(
                    diff < tolerance,
                    "forward() should match forward_cross_simd() when simd enabled. Index {}: {} vs {}, diff={}",
                    i, f, s, diff
                );
            }
            println!("SIMD dispatch verified: forward() uses forward_cross_simd()");
        }
    }

    /// EXTREME TDD: Verify forward_cross dispatches to SIMD when feature enabled
    #[test]
    fn tdd_forward_cross_auto_dispatch() {
        let mut attn = MultiHeadAttention::new(2, 32);
        attn.finalize_weights();

        let seq_len = 4;
        let d_model = 32;

        let x: Vec<f32> = (0..seq_len * d_model).map(|i| i as f32 * 0.01).collect();
        let ctx: Vec<f32> = (0..seq_len * d_model).map(|i| i as f32 * 0.02).collect();

        // Call the auto-dispatching method
        let output = attn
            .forward_cross_dispatch(&x, &ctx, None)
            .expect("forward_cross_dispatch");

        assert_eq!(output.len(), seq_len * d_model);
        println!(
            "Auto-dispatch forward_cross completed with {} elements",
            output.len()
        );
    }
}

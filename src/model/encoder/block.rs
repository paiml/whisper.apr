//! Transformer encoder block
//!
//! Self-attention with residual connections and feed-forward network.

use super::super::MultiHeadAttention;
use super::layers::{FeedForward, LayerNorm};
use crate::error::WhisperResult;

#[cfg(feature = "realizar-inference")]
use crate::error::WhisperError;
#[cfg(feature = "realizar-inference")]
use realizar::layers::FusedLayerNormLinear;

/// Element-wise vector addition for residual connections
fn add_residual(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(a, b)| a + b).collect()
}

/// Generate flat identity matrix of dimension d x d
#[cfg(feature = "realizar-inference")]
fn identity_flat(d: usize) -> Vec<f32> {
    (0..d)
        .flat_map(|i| (0..d).map(move |j| if i == j { 1.0 } else { 0.0 }))
        .collect()
}

/// Single transformer encoder block
#[derive(Debug, Clone)]
pub struct EncoderBlock {
    /// Self-attention layer
    pub self_attn: MultiHeadAttention,
    /// Layer norm before attention
    pub ln1: LayerNorm,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// Layer norm before FFN
    pub ln2: LayerNorm,
}

impl EncoderBlock {
    /// Create new encoder block
    #[must_use]
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(n_heads, d_model),
            ln1: LayerNorm::new(d_model),
            ffn: FeedForward::new(d_model, d_ff),
            ln2: LayerNorm::new(d_model),
        }
    }

    /// Forward pass: x + Attention(LN(x)) then x + FFN(LN(x))
    pub fn forward(&self, x: &[f32]) -> WhisperResult<Vec<f32>> {
        // Pre-norm self-attention with residual
        let normed = self.ln1.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, None)?;
        let mut residual = add_residual(x, &attn_out);

        // Pre-norm FFN with residual
        let normed = self.ln2.forward(&residual)?;
        let ffn_out = self.ffn.forward(&normed)?;

        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }

    /// Finalize weights by caching transposed/pre-computed data
    ///
    /// Encoder is compute-bound (1500-token batch matmul), not memory-bound.
    /// INT8 quantization makes encoder slower (per-token matvec overhead).
    /// Focus on efficient batch matmul via transposed weight caching.
    pub fn finalize_weights(&mut self) {
        self.self_attn.finalize_weights();
        self.ffn.finalize_weights();
    }

    /// Forward pass using fused kernels (WAPR-PERF-004 Phase 3)
    #[cfg(feature = "realizar-inference")]
    pub fn forward_fused(&self, x: &[f32]) -> WhisperResult<Vec<f32>> {
        let d_model = self.ln1.normalized_shape;
        let seq_len = x.len() / d_model;
        if x.len() % d_model != 0 {
            return Err(WhisperError::Model("input size mismatch".into()));
        }

        let mut fused_ln = FusedLayerNormLinear::new(d_model, d_model, self.ln1.eps)
            .map_err(|e| WhisperError::Model(format!("FusedLayerNormLinear error: {e}")))?;

        fused_ln.norm_weight_mut().copy_from_slice(&self.ln1.weight);
        fused_ln.norm_bias_mut().copy_from_slice(&self.ln1.bias);

        let identity = identity_flat(d_model);
        fused_ln.linear_weight_mut().copy_from_slice(&identity);
        let zeros = vec![0.0_f32; d_model];
        fused_ln.linear_bias_mut().copy_from_slice(&zeros);

        let ln_tensor = realizar::tensor::Tensor::from_vec(vec![seq_len, d_model], x.to_vec())
            .map_err(|e| WhisperError::Model(format!("Tensor error: {e}")))?;

        let normed_tensor = fused_ln
            .forward(&ln_tensor)
            .map_err(|e| WhisperError::Model(format!("FusedLayerNormLinear forward: {e}")))?;

        let normed = normed_tensor.data().to_vec();

        let attn_out = self.self_attn.forward(&normed, None)?;
        let mut residual = add_residual(x, &attn_out);

        let mut fused_ln2 = FusedLayerNormLinear::new(d_model, d_model, self.ln2.eps)
            .map_err(|e| WhisperError::Model(format!("FusedLayerNormLinear error: {e}")))?;

        fused_ln2
            .norm_weight_mut()
            .copy_from_slice(&self.ln2.weight);
        fused_ln2.norm_bias_mut().copy_from_slice(&self.ln2.bias);
        fused_ln2.linear_weight_mut().copy_from_slice(&identity);
        fused_ln2.linear_bias_mut().copy_from_slice(&zeros);

        let res_tensor =
            realizar::tensor::Tensor::from_vec(vec![seq_len, d_model], residual.clone())
                .map_err(|e| WhisperError::Model(format!("Tensor error: {e}")))?;

        let normed2_tensor = fused_ln2
            .forward(&res_tensor)
            .map_err(|e| WhisperError::Model(format!("FusedLayerNormLinear forward: {e}")))?;

        let normed2 = normed2_tensor.data().to_vec();
        let ffn_out = self.ffn.forward(&normed2)?;

        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_block_new() {
        let block = EncoderBlock::new(64, 4, 256);
        assert_eq!(block.self_attn.d_model(), 64);
        assert_eq!(block.ffn.d_model, 64);
    }

    #[test]
    fn test_encoder_block_forward() {
        let block = EncoderBlock::new(8, 2, 32);
        let input = vec![0.1_f32; 16];
        let output = block.forward(&input).expect("forward should succeed");
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_encoder_block_residual() {
        let block = EncoderBlock::new(8, 2, 32);
        let input = vec![1.0_f32; 8];
        let output = block.forward(&input).expect("forward should succeed");
        assert_eq!(output.len(), 8);
    }

    #[test]
    #[cfg(feature = "realizar-inference")]
    fn test_encoder_block_forward_fused() {
        let block = EncoderBlock::new(8, 2, 32);
        let input = vec![0.1_f32; 16];
        let output_fused = block.forward_fused(&input).expect("forward_fused");
        let output_regular = block.forward(&input).expect("forward");
        assert_eq!(output_fused.len(), output_regular.len());
        assert_eq!(output_fused.len(), 16);
    }

    #[test]
    #[cfg(feature = "realizar-inference")]
    fn test_encoder_block_forward_fused_invalid_input() {
        let block = EncoderBlock::new(8, 2, 32);
        let input = vec![0.1_f32; 17];
        let result = block.forward_fused(&input);
        assert!(result.is_err());
    }
}

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
        let mut residual = self.self_attn.forward(&normed, None)?;
        // In-place residual add: reuse attn_out buffer instead of allocating a new Vec
        for (r, &xi) in residual.iter_mut().zip(x.iter()) {
            *r += xi;
        }

        // Pre-norm FFN with residual
        let normed = self.ln2.forward(&residual)?;
        let ffn_out = self.ffn.forward(&normed)?;

        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }

    /// Forward pass with BrickProfiler instrumentation (WAPR-PROFILE-001 Gap 1)
    ///
    /// Identical to `forward()` but records per-operator timing and CPU cycles
    /// into the trueno `BrickProfiler` via O(1) `BrickId`-indexed arrays.
    ///
    /// BrickId mapping:
    /// - LN1, LN2 → `BrickId::LayerNorm` (category: Norm)
    /// - Self-attention → `BrickId::AttentionScore` (category: Attention)
    /// - FFN → `BrickId::GateProjection` (category: Ffn)
    pub fn forward_profiled(
        &self,
        x: &[f32],
        profiler: &mut trueno::BrickProfiler,
    ) -> WhisperResult<Vec<f32>> {
        let d_model = self.ln1.normalized_shape;
        let seq_len = (x.len() / d_model) as u64;

        // LN1: LayerNorm before attention
        let c0 = trueno::brick::cpu_cycles();
        let timer = profiler.start_brick(trueno::BrickId::LayerNorm);
        let normed = self.ln1.forward(x)?;
        let c1 = trueno::brick::cpu_cycles();
        profiler.stop_brick(timer, seq_len);
        let stats = profiler.brick_stats_mut(trueno::BrickId::LayerNorm);
        let cycles = c1.wrapping_sub(c0);
        stats.total_cycles += cycles;
        stats.min_cycles = stats.min_cycles.min(cycles);
        stats.max_cycles = stats.max_cycles.max(cycles);

        // Self-attention + residual
        let c0 = trueno::brick::cpu_cycles();
        let timer = profiler.start_brick(trueno::BrickId::AttentionScore);
        let mut residual = self.self_attn.forward(&normed, None)?;
        for (r, &xi) in residual.iter_mut().zip(x.iter()) {
            *r += xi;
        }
        let c1 = trueno::brick::cpu_cycles();
        profiler.stop_brick(timer, seq_len);
        let stats = profiler.brick_stats_mut(trueno::BrickId::AttentionScore);
        let cycles = c1.wrapping_sub(c0);
        stats.total_cycles += cycles;
        stats.min_cycles = stats.min_cycles.min(cycles);
        stats.max_cycles = stats.max_cycles.max(cycles);

        // LN2: LayerNorm before FFN
        let c0 = trueno::brick::cpu_cycles();
        let timer = profiler.start_brick(trueno::BrickId::LayerNorm);
        let normed = self.ln2.forward(&residual)?;
        let c1 = trueno::brick::cpu_cycles();
        profiler.stop_brick(timer, seq_len);
        let stats = profiler.brick_stats_mut(trueno::BrickId::LayerNorm);
        let cycles = c1.wrapping_sub(c0);
        stats.total_cycles += cycles;
        stats.min_cycles = stats.min_cycles.min(cycles);
        stats.max_cycles = stats.max_cycles.max(cycles);

        // FFN + residual
        let c0 = trueno::brick::cpu_cycles();
        let timer = profiler.start_brick(trueno::BrickId::GateProjection);
        let ffn_out = self.ffn.forward(&normed)?;
        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }
        let c1 = trueno::brick::cpu_cycles();
        profiler.stop_brick(timer, seq_len);
        let stats = profiler.brick_stats_mut(trueno::BrickId::GateProjection);
        let cycles = c1.wrapping_sub(c0);
        stats.total_cycles += cycles;
        stats.min_cycles = stats.min_cycles.min(cycles);
        stats.max_cycles = stats.max_cycles.max(cycles);

        Ok(residual)
    }

    /// Finalize weights by caching transposed/pre-computed data
    ///
    /// Encoder is compute-bound (1500-token batch matmul), not memory-bound.
    /// INT8 quantization makes encoder slower (per-token matvec overhead).
    /// Focus on efficient batch matmul via transposed weight caching.
    pub fn finalize_weights(&mut self) {
        self.self_attn.finalize_weights_encoder();
        self.ffn.finalize_weights_encoder();
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

        let mut residual = self.self_attn.forward(&normed, None)?;
        for (r, &xi) in residual.iter_mut().zip(x.iter()) {
            *r += xi;
        }

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

    #[test]
    fn test_encoder_block_forward_profiled() {
        let block = EncoderBlock::new(8, 2, 32);
        let input = vec![0.1_f32; 16];
        let mut profiler = trueno::BrickProfiler::enabled();
        let output = block
            .forward_profiled(&input, &mut profiler)
            .expect("forward_profiled");
        assert_eq!(output.len(), 16);
        // BrickProfiler should have recorded samples for each category
        let ln_stats = profiler.brick_stats(trueno::BrickId::LayerNorm);
        assert_eq!(ln_stats.count, 2, "LN1 + LN2 = 2 samples");
        assert!(ln_stats.total_ns > 0);
        let attn_stats = profiler.brick_stats(trueno::BrickId::AttentionScore);
        assert_eq!(attn_stats.count, 1);
        assert!(attn_stats.total_ns > 0);
        let ffn_stats = profiler.brick_stats(trueno::BrickId::GateProjection);
        assert_eq!(ffn_stats.count, 1);
        assert!(ffn_stats.total_ns > 0);
        // Category breakdown should work
        let cats = profiler.category_stats();
        assert!(cats[trueno::BrickCategory::Norm as usize].total_ns > 0);
        assert!(cats[trueno::BrickCategory::Attention as usize].total_ns > 0);
        assert!(cats[trueno::BrickCategory::Ffn as usize].total_ns > 0);
        // Profiled output should match non-profiled output
        let regular_output = block.forward(&input).expect("forward");
        assert_eq!(output, regular_output);
    }
}

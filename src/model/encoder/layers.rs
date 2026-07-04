//! Layer components for encoder blocks
//!
//! Layer normalization and feed-forward networks.

use super::super::LinearWeights;
use crate::error::{WhisperError, WhisperResult};

/// Layer normalization weights
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Scale parameter (gamma)
    pub weight: Vec<f32>,
    /// Shift parameter (beta)
    pub bias: Vec<f32>,
    /// Normalized dimension
    pub normalized_shape: usize,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl LayerNorm {
    /// Create new layer normalization
    #[must_use]
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            weight: vec![1.0; normalized_shape],
            bias: vec![0.0; normalized_shape],
            normalized_shape,
            eps: 1e-5,
        }
    }

    /// Apply layer normalization
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let mut output = vec![0.0_f32; input.len()];
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Apply layer normalization into a pre-allocated buffer (PMAT-014 O1).
    ///
    /// `output` must have the same length as `input`.
    pub fn forward_into(&self, input: &[f32], output: &mut [f32]) -> WhisperResult<()> {
        if input.len() % self.normalized_shape != 0 {
            return Err(WhisperError::Model(
                "input size mismatch for layer norm".into(),
            ));
        }
        debug_assert_eq!(input.len(), output.len());

        let seq_len = input.len() / self.normalized_shape;

        let chunks_in = input.chunks_exact(self.normalized_shape);
        let chunks_out = output.chunks_exact_mut(self.normalized_shape);
        
        #[cfg(not(feature = "parallel"))]
        {
            for (slice, out_slice) in chunks_in.zip(chunks_out) {
                crate::simd::optimized::layer_norm_into(slice, &self.weight, &self.bias, self.eps, out_slice);
            }
        }
        
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            chunks_in.zip(chunks_out).par_bridge().for_each(|(slice, out_slice)| {
                crate::simd::optimized::layer_norm_into(slice, &self.weight, &self.bias, self.eps, out_slice);
            });
        }

        Ok(())
    }
}

/// Feed-forward network (FFN) in transformer block
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// First linear layer (expansion)
    pub fc1: LinearWeights,
    /// Second linear layer (projection)
    pub fc2: LinearWeights,
    /// Hidden dimension (typically 4 * d_model)
    pub d_ff: usize,
    /// Model dimension
    pub d_model: usize,
}

impl FeedForward {
    /// Create new feed-forward network
    #[must_use]
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            fc1: LinearWeights::new(d_model, d_ff),
            fc2: LinearWeights::new(d_ff, d_model),
            d_ff,
            d_model,
        }
    }

    /// Forward pass with GELU activation
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        let seq_len = input.len() / self.d_model;

        // First linear + GELU
        let mut hidden = self.fc1.forward_simd(input, seq_len)?;
        for x in &mut hidden {
            *x = gelu(*x);
        }

        // Second linear
        self.fc2.forward_simd(&hidden, seq_len)
    }

    /// Forward pass into pre-allocated buffers (PMAT-014 O1).
    ///
    /// `hidden` must be `seq_len * d_ff` elements.
    /// `output` must be `seq_len * d_model` elements.
    pub fn forward_into(
        &self,
        input: &[f32],
        hidden: &mut [f32],
        output: &mut [f32],
    ) -> WhisperResult<()> {
        let seq_len = input.len() / self.d_model;

        // First linear into hidden + GELU in-place
        self.fc1.forward_simd_into(input, seq_len, hidden)?;
        for x in hidden.iter_mut() {
            *x = gelu(*x);
        }

        // Second linear into output
        self.fc2.forward_simd_into(hidden, seq_len, output)
    }

    /// Finalize weights for optimized SIMD matmul
    pub fn finalize_weights(&mut self) {
        self.fc1.finalize_weights();
        self.fc2.finalize_weights();
    }

    pub fn finalize_weights_encoder(&mut self) {
        self.fc1.finalize_weights_encoder();
        self.fc2.finalize_weights_encoder();
    }

    /// Check if weights have been finalized
    #[must_use]
    pub fn is_finalized(&self) -> bool {
        self.fc1.is_finalized() && self.fc2.is_finalized()
    }

    /// Convert all weights to fp16 in-place
    pub fn convert_to_f16(&mut self) {
        self.fc1.convert_to_f16();
        self.fc2.convert_to_f16();
    }
}

/// GELU activation function.
#[inline]
#[must_use]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_new() {
        let ln = LayerNorm::new(64);
        assert_eq!(ln.normalized_shape, 64);
        assert_eq!(ln.weight.len(), 64);
        assert_eq!(ln.bias.len(), 64);
    }

    #[test]
    fn test_layer_norm_forward() {
        let ln = LayerNorm::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.forward(&input).expect("forward should succeed");

        assert_eq!(output.len(), 4);
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    }

    #[test]
    fn test_layer_norm_batch() {
        let ln = LayerNorm::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = ln.forward(&input).expect("forward should succeed");
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_feed_forward_new() {
        let ffn = FeedForward::new(64, 256);
        assert_eq!(ffn.d_model, 64);
        assert_eq!(ffn.d_ff, 256);
    }

    #[test]
    fn test_feed_forward_forward() {
        let ffn = FeedForward::new(8, 32);
        let input = vec![0.0_f32; 16];
        let output = ffn.forward(&input).expect("forward should succeed");
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_gelu_at_zero() {
        let result = gelu(0.0);
        assert!(result.abs() < 1e-6, "GELU(0) should be ~0");
    }

    #[test]
    fn test_gelu_positive() {
        let result = gelu(1.0);
        assert!(result > 0.0, "GELU(1) should be positive");
        assert!(result < 1.0, "GELU(1) should be less than 1");
    }

    #[test]
    fn test_gelu_negative() {
        let result = gelu(-1.0);
        assert!(result < 0.0, "GELU(-1) should be negative");
        assert!(result > -0.2, "GELU(-1) should be > -0.2");
    }
}

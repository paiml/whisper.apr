//! SwiGLU Feed-Forward Network Implementation
//!
//! SwiGLU is a gated activation function that combines Swish and GLU:
//!
//! ```text
//! SwiGLU(x, W, V, W2) = (Swish(xW) ⊙ xV) W2
//! ```
//!
//! Where:
//! - Swish(x) = x * σ(x) = x * sigmoid(x)
//! - ⊙ is element-wise multiplication (Hadamard product)
//!
//! # Advantages
//!
//! - Better gradient flow than ReLU
//! - Gating mechanism provides adaptive nonlinearity
//! - Empirically better performance on language modeling
//!
//! # References
//!
//! - "GLU Variants Improve Transformer" https://arxiv.org/abs/2002.05202
//! - LLaMA, PaLM, and LFM2 all use SwiGLU

use crate::error::{WhisperError, WhisperResult};

/// SwiGLU FFN configuration
#[derive(Debug, Clone)]
pub struct SwiGluConfig {
    /// Input/output dimension (hidden_size)
    pub hidden_size: usize,
    /// Intermediate dimension (typically ~2.7x hidden_size for SwiGLU)
    pub intermediate_size: usize,
    /// Whether to use bias in linear layers
    pub bias: bool,
}

impl SwiGluConfig {
    /// Create config for LFM2-2.6B
    #[must_use]
    pub fn lfm2_2_6b() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 10752, // ~5.25x expansion
            bias: false,
        }
    }

    /// Validate configuration
    ///
    /// # Errors
    /// Returns error if configuration is invalid
    pub fn validate(&self) -> WhisperResult<()> {
        if self.hidden_size == 0 {
            return Err(WhisperError::Model("hidden_size must be > 0".into()));
        }
        if self.intermediate_size == 0 {
            return Err(WhisperError::Model("intermediate_size must be > 0".into()));
        }
        Ok(())
    }
}

/// SwiGLU Feed-Forward Network layer
///
/// Implements the gated FFN used in LFM2 and other modern LLMs.
#[derive(Debug, Clone)]
pub struct SwiGluFfn {
    /// Configuration
    pub config: SwiGluConfig,
    /// Gate projection weights [hidden_size, intermediate_size]
    pub w_gate: Vec<f32>,
    /// Up projection weights [hidden_size, intermediate_size]
    pub w_up: Vec<f32>,
    /// Down projection weights [intermediate_size, hidden_size]
    pub w_down: Vec<f32>,
    /// Gate bias (optional)
    pub b_gate: Option<Vec<f32>>,
    /// Up bias (optional)
    pub b_up: Option<Vec<f32>>,
    /// Down bias (optional)
    pub b_down: Option<Vec<f32>>,
}

impl SwiGluFfn {
    /// Create new SwiGLU FFN layer
    ///
    /// # Errors
    /// Returns error if config is invalid
    pub fn new(config: SwiGluConfig) -> WhisperResult<Self> {
        config.validate()?;

        let h = config.hidden_size;
        let i = config.intermediate_size;

        Ok(Self {
            config,
            w_gate: vec![0.0; h * i],
            w_up: vec![0.0; h * i],
            w_down: vec![0.0; i * h],
            b_gate: None,
            b_up: None,
            b_down: None,
        })
    }

    /// Forward pass through SwiGLU FFN
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [seq_len, hidden_size]
    ///
    /// # Returns
    /// Output tensor [seq_len, hidden_size]
    ///
    /// # Errors
    /// Returns error if dimensions are invalid
    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> WhisperResult<Vec<f32>> {
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;

        if hidden_states.len() != seq_len * h {
            return Err(WhisperError::Model(format!(
                "hidden_states length {} != seq_len * hidden_size ({})",
                hidden_states.len(),
                seq_len * h
            )));
        }

        // Gate projection: x @ W_gate
        let gate = self.linear(
            hidden_states,
            seq_len,
            &self.w_gate,
            self.b_gate.as_deref(),
            h,
            i,
        );

        // Up projection: x @ W_up
        let up = self.linear(
            hidden_states,
            seq_len,
            &self.w_up,
            self.b_up.as_deref(),
            h,
            i,
        );

        // Apply SwiGLU: Swish(gate) * up
        let mut intermediate = vec![0.0f32; seq_len * i];
        for j in 0..seq_len * i {
            // Swish(x) = x * sigmoid(x)
            let swish_gate = swish(gate[j]);
            intermediate[j] = swish_gate * up[j];
        }

        // Down projection: intermediate @ W_down
        let output = self.linear(
            &intermediate,
            seq_len,
            &self.w_down,
            self.b_down.as_deref(),
            i,
            h,
        );

        Ok(output)
    }

    /// Linear projection helper
    #[allow(clippy::unused_self)]
    fn linear(
        &self,
        input: &[f32],
        seq_len: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; seq_len * out_features];

        for s in 0..seq_len {
            for o in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..in_features {
                    // Weight layout: [in_features, out_features] (column-major for efficiency)
                    // But we'll use row-major: [out_features, in_features]
                    sum += input[s * in_features + k] * weight[o * in_features + k];
                }
                if let Some(b) = bias {
                    sum += b[o];
                }
                output[s * out_features + o] = sum;
            }
        }

        output
    }

    /// Total number of parameters
    #[must_use]
    pub fn num_params(&self) -> usize {
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        // 3 weight matrices
        3 * h * i
    }

    /// Memory usage in bytes (f32 weights only)
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.num_params() * std::mem::size_of::<f32>()
    }
}

/// Swish activation function: x * sigmoid(x)
///
/// Also known as SiLU (Sigmoid Linear Unit).
#[inline]
fn swish(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Sigmoid function (UCBD §4 ONE PATH: delegates to `trueno::sigmoid_scalar`)
#[inline]
fn sigmoid(x: f32) -> f32 {
    trueno::sigmoid_scalar(x)
}

/// GELU activation (UCBD §4 ONE PATH: delegates to `trueno::gelu_scalar`)
#[inline]
#[allow(dead_code)]
fn gelu(x: f32) -> f32 {
    trueno::gelu_scalar(x)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_config_lfm2() {
        let config = SwiGluConfig::lfm2_2_6b();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 10752);
        assert!(!config.bias);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_swiglu_new() {
        let config = SwiGluConfig::lfm2_2_6b();
        let ffn = SwiGluFfn::new(config).expect("should create SwiGLU");

        assert_eq!(ffn.w_gate.len(), 2048 * 10752);
        assert_eq!(ffn.w_up.len(), 2048 * 10752);
        assert_eq!(ffn.w_down.len(), 10752 * 2048);
        assert_eq!(ffn.num_params(), 3 * 2048 * 10752);
    }

    #[test]
    fn test_swiglu_forward_shape() {
        let config = SwiGluConfig {
            hidden_size: 16,
            intermediate_size: 32,
            bias: false,
        };
        let ffn = SwiGluFfn::new(config).expect("should create SwiGLU");

        let seq_len = 4;
        let input = vec![0.1f32; seq_len * 16];

        let output = ffn
            .forward(&input, seq_len)
            .expect("forward should succeed");
        assert_eq!(output.len(), seq_len * 16);
    }

    #[test]
    fn test_swiglu_forward_values() {
        let config = SwiGluConfig {
            hidden_size: 4,
            intermediate_size: 8,
            bias: false,
        };
        let mut ffn = SwiGluFfn::new(config).expect("should create SwiGLU");

        // Initialize weights with small values
        for (i, w) in ffn.w_gate.iter_mut().enumerate() {
            *w = ((i % 5) as f32 - 2.0) * 0.1;
        }
        for (i, w) in ffn.w_up.iter_mut().enumerate() {
            *w = ((i % 7) as f32 - 3.0) * 0.1;
        }
        for (i, w) in ffn.w_down.iter_mut().enumerate() {
            *w = ((i % 3) as f32 - 1.0) * 0.1;
        }

        let seq_len = 2;
        let input: Vec<f32> = (0..seq_len * 4).map(|i| (i as f32 * 0.1).sin()).collect();

        let output = ffn
            .forward(&input, seq_len)
            .expect("forward should succeed");
        assert_eq!(output.len(), seq_len * 4);

        // Output should be different from input
        let diff: f32 = output
            .iter()
            .zip(input.iter())
            .map(|(o, i)| (o - i).abs())
            .sum();
        assert!(diff > 0.001, "Output should differ from input");
    }

    #[test]
    fn test_swish_function() {
        // Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((swish(0.0) - 0.0).abs() < 1e-6);

        // Swish(x) ≈ x for large positive x
        let large_pos = swish(10.0);
        assert!((large_pos - 10.0).abs() < 0.01);

        // Swish(x) ≈ 0 for large negative x
        let large_neg = swish(-10.0);
        assert!(large_neg.abs() < 0.001);

        // Swish is smooth and has a minimum around x ≈ -1.28
        let at_minus_one = swish(-1.0);
        assert!(at_minus_one < 0.0); // Negative
    }

    #[test]
    fn test_sigmoid_function() {
        // sigmoid(0) = 0.5
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);

        // sigmoid(large_positive) ≈ 1
        assert!((sigmoid(10.0) - 1.0).abs() < 0.001);

        // sigmoid(large_negative) ≈ 0
        assert!(sigmoid(-10.0).abs() < 0.001);
    }

    #[test]
    fn test_gelu_function() {
        // GELU(0) ≈ 0
        assert!(gelu(0.0).abs() < 1e-6);

        // GELU(x) ≈ x for large positive x
        let large_pos = gelu(5.0);
        assert!((large_pos - 5.0).abs() < 0.01);

        // GELU is non-negative for x > 0
        assert!(gelu(1.0) > 0.0);
    }

    #[test]
    fn test_swiglu_memory() {
        let config = SwiGluConfig::lfm2_2_6b();
        let ffn = SwiGluFfn::new(config).expect("should create SwiGLU");

        let expected_params = 3 * 2048 * 10752;
        let expected_bytes = expected_params * 4; // f32 = 4 bytes

        assert_eq!(ffn.num_params(), expected_params);
        assert_eq!(ffn.memory_bytes(), expected_bytes);
    }
}

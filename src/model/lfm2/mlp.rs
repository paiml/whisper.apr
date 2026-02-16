//! Standard MLP Feed-Forward Network
//!
//! Two-projection FFN used by Moonshine and similar architectures:
//!
//! ```text
//! MLP(x) = fc2(activation(fc1(x)))
//! ```
//!
//! Supports GELU and SiLU activations. Unlike SwiGLU (3 projections),
//! this uses the standard 2-projection pattern with fc1 and fc2.

use crate::error::{WhisperError, WhisperResult};

/// Activation function for MLP FFN
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpActivation {
    /// GELU activation (used in Moonshine encoder)
    Gelu,
    /// SiLU/Swish activation (used in Moonshine decoder)
    Silu,
}

/// MLP FFN configuration
#[derive(Debug, Clone)]
pub struct MlpConfig {
    /// Input/output dimension (hidden_size)
    pub hidden_size: usize,
    /// Intermediate dimension (typically 4x hidden_size)
    pub intermediate_size: usize,
    /// Whether to use bias in linear layers
    pub bias: bool,
    /// Activation function
    pub activation: MlpActivation,
}

impl MlpConfig {
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

/// Standard MLP Feed-Forward Network layer
///
/// Two-projection FFN: fc1 (hidden → intermediate) → activation → fc2 (intermediate → hidden)
#[derive(Debug, Clone)]
pub struct MlpFfn {
    /// Configuration
    pub config: MlpConfig,
    /// First projection weights [hidden_size, intermediate_size] (row-major)
    pub fc1: Vec<f32>,
    /// Second projection weights [intermediate_size, hidden_size] (row-major)
    pub fc2: Vec<f32>,
    /// fc1 bias (optional)
    pub b1: Option<Vec<f32>>,
    /// fc2 bias (optional)
    pub b2: Option<Vec<f32>>,
}

impl MlpFfn {
    /// Create new MLP FFN layer with zero-initialized weights
    ///
    /// # Errors
    /// Returns error if config is invalid
    pub fn new(config: MlpConfig) -> WhisperResult<Self> {
        config.validate()?;

        let h = config.hidden_size;
        let i = config.intermediate_size;

        Ok(Self {
            config,
            fc1: vec![0.0; i * h],
            fc2: vec![0.0; h * i],
            b1: None,
            b2: None,
        })
    }

    /// Forward pass: fc1 → activation → fc2
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor `[seq_len, hidden_size]`
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor `[seq_len, hidden_size]`
    ///
    /// # Errors
    /// Returns error if dimensions are invalid
    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> WhisperResult<Vec<f32>> {
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;

        if hidden_states.len() != seq_len * h {
            return Err(WhisperError::Model(format!(
                "MLP input length {} != seq_len * hidden_size ({})",
                hidden_states.len(),
                seq_len * h
            )));
        }

        // fc1: [seq_len, hidden_size] → [seq_len, intermediate_size]
        let mut intermediate = linear(
            hidden_states,
            seq_len,
            &self.fc1,
            self.b1.as_deref(),
            h,
            i,
        );

        // Apply activation in-place
        match self.config.activation {
            MlpActivation::Gelu => {
                for val in &mut intermediate {
                    *val = gelu(*val);
                }
            }
            MlpActivation::Silu => {
                for val in &mut intermediate {
                    *val = silu(*val);
                }
            }
        }

        // fc2: [seq_len, intermediate_size] → [seq_len, hidden_size]
        let output = linear(
            &intermediate,
            seq_len,
            &self.fc2,
            self.b2.as_deref(),
            i,
            h,
        );

        Ok(output)
    }

    /// Total number of parameters
    #[must_use]
    pub fn num_params(&self) -> usize {
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size;
        2 * h * i
    }
}

/// Linear projection: input @ weight^T + bias
fn linear(
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

/// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x * x * x)).tanh())
}

/// SiLU/Swish activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_config_validate() {
        let config = MlpConfig {
            hidden_size: 288,
            intermediate_size: 1152,
            bias: false,
            activation: MlpActivation::Gelu,
        };
        assert!(config.validate().is_ok());

        let bad = MlpConfig {
            hidden_size: 0,
            intermediate_size: 1152,
            bias: false,
            activation: MlpActivation::Gelu,
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_mlp_new() {
        let config = MlpConfig {
            hidden_size: 288,
            intermediate_size: 1152,
            bias: false,
            activation: MlpActivation::Gelu,
        };
        let mlp = MlpFfn::new(config).expect("should create MLP");
        assert_eq!(mlp.fc1.len(), 1152 * 288);
        assert_eq!(mlp.fc2.len(), 288 * 1152);
        assert_eq!(mlp.num_params(), 2 * 288 * 1152);
    }

    #[test]
    fn test_mlp_forward_gelu_shape() {
        let config = MlpConfig {
            hidden_size: 16,
            intermediate_size: 64,
            bias: false,
            activation: MlpActivation::Gelu,
        };
        let mlp = MlpFfn::new(config).expect("should create MLP");

        let seq_len = 4;
        let input = vec![0.1f32; seq_len * 16];
        let output = mlp.forward(&input, seq_len).expect("forward");
        assert_eq!(output.len(), seq_len * 16);
    }

    #[test]
    fn test_mlp_forward_silu_shape() {
        let config = MlpConfig {
            hidden_size: 16,
            intermediate_size: 64,
            bias: false,
            activation: MlpActivation::Silu,
        };
        let mlp = MlpFfn::new(config).expect("should create MLP");

        let seq_len = 4;
        let input = vec![0.1f32; seq_len * 16];
        let output = mlp.forward(&input, seq_len).expect("forward");
        assert_eq!(output.len(), seq_len * 16);
    }

    #[test]
    fn test_mlp_forward_values() {
        let config = MlpConfig {
            hidden_size: 4,
            intermediate_size: 8,
            bias: false,
            activation: MlpActivation::Gelu,
        };
        let mut mlp = MlpFfn::new(config).expect("should create MLP");

        // Set non-zero weights
        for (i, w) in mlp.fc1.iter_mut().enumerate() {
            *w = ((i % 5) as f32 - 2.0) * 0.1;
        }
        for (i, w) in mlp.fc2.iter_mut().enumerate() {
            *w = ((i % 3) as f32 - 1.0) * 0.1;
        }

        let input = vec![1.0f32; 4];
        let output = mlp.forward(&input, 1).expect("forward");
        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gelu_activation() {
        assert!(gelu(0.0).abs() < 1e-6);
        assert!((gelu(5.0) - 5.0).abs() < 0.01);
        assert!(gelu(1.0) > 0.0);
    }

    #[test]
    fn test_silu_activation() {
        assert!(silu(0.0).abs() < 1e-6);
        assert!((silu(10.0) - 10.0).abs() < 0.01);
        assert!(silu(-10.0).abs() < 0.001);
    }

    #[test]
    fn test_mlp_dim_mismatch_error() {
        let config = MlpConfig {
            hidden_size: 16,
            intermediate_size: 64,
            bias: false,
            activation: MlpActivation::Gelu,
        };
        let mlp = MlpFfn::new(config).expect("should create MLP");

        let bad_input = vec![0.1f32; 10]; // Wrong size
        assert!(mlp.forward(&bad_input, 1).is_err());
    }
}

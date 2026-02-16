//! RoPE (Rotary Position Embedding) Implementation
//!
//! RoPE encodes position information by rotating query and key vectors
//! in 2D subspaces. This allows the model to extrapolate to longer sequences
//! than seen during training.
//!
//! # Formula
//!
//! For position m and dimension pair (2i, 2i+1):
//! ```text
//! q'[2i]   = q[2i] * cos(mθᵢ) - q[2i+1] * sin(mθᵢ)
//! q'[2i+1] = q[2i] * sin(mθᵢ) + q[2i+1] * cos(mθᵢ)
//!
//! where θᵢ = base^(-2i/d)
//! ```
//!
//! # LFM2 Configuration
//!
//! - Base (θ): 1,000,000 (enables long-context extrapolation)
//! - Head dimension: 64
//!
//! # References
//!
//! - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
//!   https://arxiv.org/abs/2104.09864

use crate::error::{WhisperError, WhisperResult};

/// RoPE configuration
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Head dimension (full dimension including non-rotary part)
    pub head_dim: usize,
    /// Base for frequency computation (θ in the paper)
    pub base: f32,
    /// Maximum sequence length for precomputation
    pub max_seq_len: usize,
    /// Number of dimensions to apply rotation to (must be even, <= head_dim).
    /// Remaining `head_dim - rotary_dim` dimensions pass through unchanged.
    /// Set to `head_dim` for full rotation (default behavior).
    /// Moonshine uses `int(head_dim * partial_rotary_factor)` where factor=0.9.
    pub rotary_dim: Option<usize>,
}

impl RopeConfig {
    /// Create config for LFM2-2.6B
    #[must_use]
    pub fn lfm2_2_6b() -> Self {
        Self {
            head_dim: 64,
            base: 1_000_000.0, // Long-context extrapolation
            max_seq_len: 4096, // WASM-friendly limit
            rotary_dim: None,  // Full rotation
        }
    }

    /// Effective rotary dimension (defaults to head_dim if not set)
    #[must_use]
    pub fn effective_rotary_dim(&self) -> usize {
        self.rotary_dim.unwrap_or(self.head_dim)
    }

    /// Validate configuration
    ///
    /// # Errors
    /// Returns error if configuration is invalid
    pub fn validate(&self) -> WhisperResult<()> {
        let rdim = self.effective_rotary_dim();
        if rdim % 2 != 0 {
            return Err(WhisperError::Model(format!(
                "rotary_dim ({rdim}) must be even for RoPE rotation pairs",
            )));
        }
        if rdim > self.head_dim {
            return Err(WhisperError::Model(format!(
                "rotary_dim ({rdim}) must be <= head_dim ({})",
                self.head_dim
            )));
        }
        if self.base <= 0.0 {
            return Err(WhisperError::Model("base must be positive".into()));
        }
        if self.max_seq_len == 0 {
            return Err(WhisperError::Model("max_seq_len must be > 0".into()));
        }
        Ok(())
    }
}

/// Rotary Position Embedding layer
///
/// Precomputes sin/cos tables for efficient position encoding.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Configuration
    pub config: RopeConfig,
    /// Precomputed cosine values [max_seq_len, head_dim/2]
    cos_cache: Vec<f32>,
    /// Precomputed sine values [max_seq_len, head_dim/2]
    sin_cache: Vec<f32>,
}

impl RotaryEmbedding {
    /// Create new RoPE layer with precomputed sin/cos tables
    ///
    /// # Errors
    /// Returns error if config is invalid
    pub fn new(config: RopeConfig) -> WhisperResult<Self> {
        config.validate()?;

        let rotary_dim = config.effective_rotary_dim();
        let half_rotary = rotary_dim / 2;
        let max_seq = config.max_seq_len;

        // Compute inverse frequencies: θᵢ = base^(-2i/d)
        // Uses rotary_dim for frequency computation (partial rotary)
        let inv_freq: Vec<f32> = (0..half_rotary)
            .map(|i| {
                let exp = -2.0 * (i as f32) / (rotary_dim as f32);
                config.base.powf(exp)
            })
            .collect();

        // Precompute sin/cos for all positions
        let mut cos_cache = vec![0.0f32; max_seq * half_rotary];
        let mut sin_cache = vec![0.0f32; max_seq * half_rotary];

        for pos in 0..max_seq {
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = (pos as f32) * freq;
                cos_cache[pos * half_rotary + i] = angle.cos();
                sin_cache[pos * half_rotary + i] = angle.sin();
            }
        }

        Ok(Self {
            config,
            cos_cache,
            sin_cache,
        })
    }

    /// Apply rotary embedding to query or key tensor
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, num_heads, head_dim]
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of attention heads
    /// * `position_offset` - Starting position (for incremental decoding)
    ///
    /// # Returns
    /// Rotated tensor with same shape
    ///
    /// # Errors
    /// Returns error if dimensions are invalid or position exceeds max_seq_len
    pub fn forward(
        &self,
        x: &[f32],
        seq_len: usize,
        num_heads: usize,
        position_offset: usize,
    ) -> WhisperResult<Vec<f32>> {
        let head_dim = self.config.head_dim;
        let rotary_dim = self.config.effective_rotary_dim();
        let half_rotary = rotary_dim / 2;
        let max_seq = self.config.max_seq_len;

        // Validate dimensions
        let expected_len = seq_len * num_heads * head_dim;
        if x.len() != expected_len {
            return Err(WhisperError::Model(format!(
                "input length {} != expected {} (seq={}, heads={}, dim={})",
                x.len(),
                expected_len,
                seq_len,
                num_heads,
                head_dim
            )));
        }

        if position_offset + seq_len > max_seq {
            return Err(WhisperError::Model(format!(
                "position {} + seq_len {} exceeds max_seq_len {}",
                position_offset, seq_len, max_seq
            )));
        }

        let mut output = x.to_vec();

        for s in 0..seq_len {
            let pos = position_offset + s;
            let cos = &self.cos_cache[pos * half_rotary..(pos + 1) * half_rotary];
            let sin = &self.sin_cache[pos * half_rotary..(pos + 1) * half_rotary];

            for h in 0..num_heads {
                let offset = (s * num_heads + h) * head_dim;

                // Apply rotation only to first rotary_dim dimensions
                for i in 0..half_rotary {
                    let x0 = x[offset + 2 * i];
                    let x1 = x[offset + 2 * i + 1];

                    output[offset + 2 * i] = x0 * cos[i] - x1 * sin[i];
                    output[offset + 2 * i + 1] = x0 * sin[i] + x1 * cos[i];
                }
                // Dimensions rotary_dim..head_dim are already copied from x.to_vec()
            }
        }

        Ok(output)
    }

    /// Apply rotary embedding in-place
    ///
    /// # Errors
    /// Returns error if dimensions are invalid
    pub fn forward_inplace(
        &self,
        x: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        position_offset: usize,
    ) -> WhisperResult<()> {
        let head_dim = self.config.head_dim;
        let rotary_dim = self.config.effective_rotary_dim();
        let half_rotary = rotary_dim / 2;
        let max_seq = self.config.max_seq_len;

        if position_offset + seq_len > max_seq {
            return Err(WhisperError::Model(format!(
                "position {} + seq_len {} exceeds max_seq_len {}",
                position_offset, seq_len, max_seq
            )));
        }

        for s in 0..seq_len {
            let pos = position_offset + s;
            let cos = &self.cos_cache[pos * half_rotary..(pos + 1) * half_rotary];
            let sin = &self.sin_cache[pos * half_rotary..(pos + 1) * half_rotary];

            for h in 0..num_heads {
                let offset = (s * num_heads + h) * head_dim;

                // Only rotate first rotary_dim dimensions
                for i in 0..half_rotary {
                    let x0 = x[offset + 2 * i];
                    let x1 = x[offset + 2 * i + 1];

                    x[offset + 2 * i] = x0 * cos[i] - x1 * sin[i];
                    x[offset + 2 * i + 1] = x0 * sin[i] + x1 * cos[i];
                }
                // Dimensions rotary_dim..head_dim remain unchanged
            }
        }

        Ok(())
    }

    /// Get cos values for a position range
    #[must_use]
    pub fn get_cos(&self, start: usize, len: usize) -> &[f32] {
        let half_rotary = self.config.effective_rotary_dim() / 2;
        &self.cos_cache[start * half_rotary..(start + len) * half_rotary]
    }

    /// Get sin values for a position range
    #[must_use]
    pub fn get_sin(&self, start: usize, len: usize) -> &[f32] {
        let half_rotary = self.config.effective_rotary_dim() / 2;
        &self.sin_cache[start * half_rotary..(start + len) * half_rotary]
    }

    /// Memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        2 * self.cos_cache.len() * std::mem::size_of::<f32>()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_config_lfm2() {
        let config = RopeConfig::lfm2_2_6b();
        assert_eq!(config.head_dim, 64);
        assert!((config.base - 1_000_000.0).abs() < 1.0);
        assert_eq!(config.max_seq_len, 4096);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_rope_config_validation() {
        // Invalid: odd head_dim
        let config = RopeConfig {
            head_dim: 63,
            base: 10000.0,
            max_seq_len: 100,
            rotary_dim: None,
        };
        assert!(config.validate().is_err());

        // Invalid: zero base
        let config = RopeConfig {
            head_dim: 64,
            base: 0.0,
            max_seq_len: 100,
            rotary_dim: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_rope_new() {
        let config = RopeConfig {
            head_dim: 8,
            base: 10000.0,
            max_seq_len: 10,
            rotary_dim: None,
        };
        let rope = RotaryEmbedding::new(config).expect("should create RoPE");

        assert_eq!(rope.cos_cache.len(), 10 * 4); // max_seq * half_dim
        assert_eq!(rope.sin_cache.len(), 10 * 4);
    }

    #[test]
    fn test_rope_forward_shape() {
        let config = RopeConfig {
            head_dim: 8,
            base: 10000.0,
            max_seq_len: 100,
            rotary_dim: None,
        };
        let rope = RotaryEmbedding::new(config).expect("should create RoPE");

        let seq_len = 5;
        let num_heads = 4;
        let head_dim = 8;
        let input = vec![1.0f32; seq_len * num_heads * head_dim];

        let output = rope
            .forward(&input, seq_len, num_heads, 0)
            .expect("forward should succeed");

        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_rope_rotation_preserves_norm() {
        let config = RopeConfig {
            head_dim: 4,
            base: 10000.0,
            max_seq_len: 10,
            rotary_dim: None,
        };
        let rope = RotaryEmbedding::new(config).expect("should create RoPE");

        // Single position, single head
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();

        let output = rope
            .forward(&input, 1, 1, 0)
            .expect("forward should succeed");
        let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Rotation should preserve L2 norm
        assert!(
            (input_norm - output_norm).abs() < 1e-5,
            "Rotation should preserve norm: {} vs {}",
            input_norm,
            output_norm
        );
    }

    #[test]
    fn test_rope_position_0_is_identity() {
        let config = RopeConfig {
            head_dim: 4,
            base: 10000.0,
            max_seq_len: 10,
            rotary_dim: None,
        };
        let rope = RotaryEmbedding::new(config).expect("should create RoPE");

        // At position 0, angle = 0, so cos=1, sin=0 → identity transform
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let output = rope
            .forward(&input, 1, 1, 0)
            .expect("forward should succeed");

        // For position 0: cos(0) = 1, sin(0) = 0
        // So x'[2i] = x[2i] * 1 - x[2i+1] * 0 = x[2i]
        //    x'[2i+1] = x[2i] * 0 + x[2i+1] * 1 = x[2i+1]
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - input[i]).abs() < 1e-5,
                "Position 0 should be identity: {} vs {}",
                v,
                input[i]
            );
        }
    }

    #[test]
    fn test_rope_different_positions_differ() {
        let config = RopeConfig {
            head_dim: 4,
            base: 10000.0,
            max_seq_len: 10,
            rotary_dim: None,
        };
        let rope = RotaryEmbedding::new(config).expect("should create RoPE");

        let input = vec![1.0, 2.0, 3.0, 4.0];

        let output_pos0 = rope
            .forward(&input, 1, 1, 0)
            .expect("forward should succeed");
        let output_pos5 = rope
            .forward(&input, 1, 1, 5)
            .expect("forward should succeed");

        // Different positions should produce different outputs
        let diff: f32 = output_pos0
            .iter()
            .zip(output_pos5.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "Different positions should differ");
    }

    #[test]
    fn test_rope_inplace() {
        let config = RopeConfig {
            head_dim: 4,
            base: 10000.0,
            max_seq_len: 10,
            rotary_dim: None,
        };
        let rope = RotaryEmbedding::new(config).expect("should create RoPE");

        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut inplace = original.clone();

        let output = rope
            .forward(&original, 1, 1, 3)
            .expect("forward should succeed");
        rope.forward_inplace(&mut inplace, 1, 1, 3)
            .expect("inplace should succeed");

        // Results should match
        for (i, &v) in inplace.iter().enumerate() {
            assert!(
                (v - output[i]).abs() < 1e-6,
                "Inplace should match forward: {} vs {}",
                v,
                output[i]
            );
        }
    }

    #[test]
    fn test_rope_position_overflow() {
        let config = RopeConfig {
            head_dim: 4,
            base: 10000.0,
            max_seq_len: 10,
            rotary_dim: None,
        };
        let rope = RotaryEmbedding::new(config).expect("should create RoPE");

        let input = vec![1.0f32; 4];

        // Position 15 exceeds max_seq_len 10
        let result = rope.forward(&input, 1, 1, 15);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_memory() {
        let config = RopeConfig {
            head_dim: 64,
            base: 10000.0,
            max_seq_len: 4096,
            rotary_dim: None,
        };
        let rope = RotaryEmbedding::new(config).expect("should create RoPE");

        // 2 caches * max_seq * half_dim * sizeof(f32)
        let expected = 2 * 4096 * 32 * 4;
        assert_eq!(rope.memory_bytes(), expected);
    }
}

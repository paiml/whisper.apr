//! Learned convolutional stem for Moonshine audio frontend
//!
//! Replaces the mel filterbank + encoder ConvFrontend for Moonshine models.
//! Three Conv1d layers with normalization process raw audio waveform directly:
//!
//! 1. Conv1d(1, C, kernel=441, stride=441, no bias) + GroupNorm + GELU
//! 2. Conv1d(C, C, kernel=7, stride=4, pad=3) + GELU
//! 3. Conv1d(C, D, kernel=7, stride=2, pad=3) + GELU + LayerNorm
//!
//! Total stride: 441 × 4 × 2 = 3528 samples per output frame (~220ms).
//! Output length is proportional to input duration (no 30s padding).

use crate::error::{WhisperError, WhisperResult};
use crate::model::Conv1d;

/// Group normalization layer
///
/// Normalizes across groups of channels. Used after conv1 in the Moonshine
/// preprocessing stem.
#[derive(Debug, Clone)]
pub struct GroupNorm {
    /// Scale parameter (gamma), one per channel
    pub weight: Vec<f32>,
    /// Shift parameter (beta), one per channel
    pub bias: Vec<f32>,
    /// Number of groups
    pub num_groups: usize,
    /// Number of channels
    pub num_channels: usize,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl GroupNorm {
    /// Create new group normalization
    #[must_use]
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        Self {
            weight: vec![1.0; num_channels],
            bias: vec![0.0; num_channels],
            num_groups,
            num_channels,
            eps: 1e-5,
        }
    }

    /// Apply group normalization
    ///
    /// Input layout: `[seq_len × num_channels]` (row-major)
    /// Normalizes within each group of channels for each position.
    ///
    /// # Errors
    /// Returns error if input dimensions don't match
    pub fn forward(&self, input: &[f32], seq_len: usize) -> WhisperResult<Vec<f32>> {
        let expected = seq_len * self.num_channels;
        if input.len() != expected {
            return Err(WhisperError::Model(format!(
                "GroupNorm input length {} != seq_len({}) * channels({})",
                input.len(),
                seq_len,
                self.num_channels
            )));
        }

        let channels_per_group = self.num_channels / self.num_groups;
        let mut output = vec![0.0_f32; input.len()];

        // GroupNorm normalizes across (seq_len × channels_per_group) for each group
        for g in 0..self.num_groups {
            let ch_start = g * channels_per_group;
            let ch_end = ch_start + channels_per_group;

            // Compute mean and variance across all positions and channels in this group
            let count = (seq_len * channels_per_group) as f32;
            let mut sum = 0.0_f32;
            for s in 0..seq_len {
                let row_start = s * self.num_channels;
                for c in ch_start..ch_end {
                    sum += input[row_start + c];
                }
            }
            let mean = sum / count;

            let mut var_sum = 0.0_f32;
            for s in 0..seq_len {
                let row_start = s * self.num_channels;
                for c in ch_start..ch_end {
                    let diff = input[row_start + c] - mean;
                    var_sum += diff * diff;
                }
            }
            let variance = var_sum / count;
            let inv_std = 1.0 / (variance + self.eps).sqrt();

            // Apply normalization with per-channel affine
            for s in 0..seq_len {
                let row_start = s * self.num_channels;
                for c in ch_start..ch_end {
                    let normalized = (input[row_start + c] - mean) * inv_std;
                    output[row_start + c] = normalized * self.weight[c] + self.bias[c];
                }
            }
        }

        Ok(output)
    }
}

/// Learned convolutional stem for Moonshine's variable-length audio frontend.
///
/// Processes raw audio waveform directly into encoder-ready features,
/// bypassing the mel filterbank entirely. Output length scales linearly
/// with input duration.
#[derive(Debug, Clone)]
pub struct ConvStem {
    /// Layer 1: raw audio → intermediate features (stride 441, no bias)
    pub conv1: Conv1d,
    /// Group normalization after conv1
    pub groupnorm: GroupNorm,
    /// Layer 2: intermediate → intermediate (stride 4)
    pub conv2: Conv1d,
    /// Layer 3: intermediate → d_model (stride 2)
    pub conv3: Conv1d,
    /// Layer normalization after conv3 (before encoder blocks)
    pub layer_norm: LayerNormStem,
    /// Intermediate channel count
    pub intermediate_channels: usize,
    /// Output model dimension
    pub d_model: usize,
}

/// Simple layer normalization for the conv stem output
///
/// Weight-only layer norm (no bias) applied after the final conv layer.
#[derive(Debug, Clone)]
pub struct LayerNormStem {
    /// Scale parameter (gamma)
    pub weight: Vec<f32>,
    /// Normalized dimension
    pub normalized_shape: usize,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl LayerNormStem {
    /// Create new layer norm
    #[must_use]
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            weight: vec![1.0; normalized_shape],
            normalized_shape,
            eps: 1e-5,
        }
    }

    /// Apply layer normalization
    ///
    /// # Errors
    /// Returns error if input size doesn't match
    pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
        if input.len() % self.normalized_shape != 0 {
            return Err(WhisperError::Model(
                "input size mismatch for LayerNormStem".into(),
            ));
        }

        let seq_len = input.len() / self.normalized_shape;
        let mut output = vec![0.0_f32; input.len()];

        for s in 0..seq_len {
            let start = s * self.normalized_shape;
            let end = start + self.normalized_shape;
            let slice = &input[start..end];

            let mean: f32 = slice.iter().sum::<f32>() / self.normalized_shape as f32;
            let variance: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                / self.normalized_shape as f32;
            let inv_std = 1.0 / (variance + self.eps).sqrt();

            for i in 0..self.normalized_shape {
                output[start + i] = (slice[i] - mean) * inv_std * self.weight[i];
            }
        }

        Ok(output)
    }
}

/// Total stride of the Moonshine conv stem (441 × 4 × 2)
pub const CONV_STEM_TOTAL_STRIDE: usize = 3528;

impl ConvStem {
    /// Create a new learned conv stem
    ///
    /// # Arguments
    /// * `intermediate_channels` - Intermediate feature channels (model-specific, ~64 for tiny)
    /// * `d_model` - Output dimension (288 for tiny, 416 for base)
    #[must_use]
    pub fn new(intermediate_channels: usize, d_model: usize) -> Self {
        Self {
            conv1: Conv1d::new(1, intermediate_channels, 441, 441, 0),
            groupnorm: GroupNorm::new(1, intermediate_channels),
            conv2: Conv1d::new(intermediate_channels, intermediate_channels, 7, 4, 3),
            conv3: Conv1d::new(intermediate_channels, d_model, 7, 2, 3),
            layer_norm: LayerNormStem::new(d_model),
            intermediate_channels,
            d_model,
        }
    }

    /// Forward pass: raw audio → encoder features
    ///
    /// Pipeline: conv1 → GroupNorm → GELU → conv2 → GELU → conv3 → GELU → LayerNorm
    ///
    /// # Arguments
    /// * `audio` - Raw audio samples (mono, 16kHz, f32)
    ///
    /// # Returns
    /// Feature tensor (output_frames × d_model) flattened row-major
    ///
    /// # Errors
    /// Returns error if audio is empty or convolution fails
    pub fn forward(&self, audio: &[f32]) -> WhisperResult<Vec<f32>> {
        if audio.is_empty() {
            return Err(WhisperError::Audio("empty audio input".into()));
        }

        // Layer 1: raw audio → intermediate (stride 441, no bias)
        let x = self.conv1.forward(audio)?;
        let seq_len = x.len() / self.intermediate_channels;
        let x = self.groupnorm.forward(&x, seq_len)?;
        let x = crate::simd::gelu(&x);

        // Layer 2: intermediate → intermediate (stride 4)
        let x = self.conv2.forward(&x)?;
        let x = crate::simd::gelu(&x);

        // Layer 3: intermediate → d_model (stride 2)
        let x = self.conv3.forward(&x)?;
        let x = crate::simd::gelu(&x);

        // Final LayerNorm before encoder blocks
        self.layer_norm.forward(&x)
    }

    /// Calculate output frame count for a given number of audio samples
    ///
    /// Each output frame corresponds to `CONV_STEM_TOTAL_STRIDE` (3528) input samples.
    #[must_use]
    pub fn output_frames(audio_samples: usize) -> usize {
        if audio_samples == 0 {
            return 0;
        }
        // Layer 1: (samples + 0 - 441) / 441 + 1
        let after_conv1 = if audio_samples >= 441 {
            (audio_samples - 441) / 441 + 1
        } else {
            0
        };
        // Layer 2: (after_conv1 + 6 - 7) / 4 + 1
        let after_conv2 = if after_conv1 >= 1 {
            (after_conv1 + 2 * 3 - 7) / 4 + 1
        } else {
            0
        };
        // Layer 3: (after_conv2 + 6 - 7) / 2 + 1
        if after_conv2 >= 1 {
            (after_conv2 + 2 * 3 - 7) / 2 + 1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_stem_new() {
        let stem = ConvStem::new(64, 288);
        assert_eq!(stem.conv1.in_channels, 1);
        assert_eq!(stem.conv1.out_channels, 64);
        assert_eq!(stem.conv1.kernel_size, 441);
        assert_eq!(stem.conv1.stride, 441);
        assert_eq!(stem.conv2.in_channels, 64);
        assert_eq!(stem.conv2.stride, 4);
        assert_eq!(stem.conv3.out_channels, 288);
        assert_eq!(stem.conv3.stride, 2);
        assert_eq!(stem.d_model, 288);
        assert_eq!(stem.groupnorm.num_channels, 64);
        assert_eq!(stem.layer_norm.normalized_shape, 288);
    }

    #[test]
    fn test_conv_stem_output_frames() {
        // 1.5s at 16kHz = 24,000 samples
        let frames = ConvStem::output_frames(24_000);
        assert!(
            frames > 0,
            "1.5s audio should produce >0 frames, got {frames}"
        );
        assert!(
            frames < 20,
            "1.5s audio should produce <20 frames, got {frames}"
        );

        // 30s at 16kHz = 480,000 samples
        let frames_30s = ConvStem::output_frames(480_000);
        assert!(
            frames_30s > 100,
            "30s audio should produce >100 frames, got {frames_30s}"
        );

        // Proportionality: 30s should produce ~20x more frames than 1.5s
        let ratio = frames_30s as f32 / frames as f32;
        assert!(
            (ratio - 20.0).abs() < 5.0,
            "30s/1.5s frame ratio should be ~20x, got {ratio:.1}x"
        );
    }

    #[test]
    fn test_conv_stem_output_frames_empty() {
        assert_eq!(ConvStem::output_frames(0), 0);
    }

    #[test]
    fn test_conv_stem_output_frames_short() {
        // Very short audio (less than one kernel)
        assert_eq!(ConvStem::output_frames(100), 0);
        assert_eq!(ConvStem::output_frames(440), 0);
    }

    #[test]
    fn test_conv_stem_forward_produces_output() {
        let stem = ConvStem::new(4, 8);
        // 2 seconds of audio = 32000 samples
        let audio = vec![0.0_f32; 32_000];
        let output = stem.forward(&audio).expect("forward should succeed");
        let expected_frames = ConvStem::output_frames(32_000);
        assert_eq!(
            output.len(),
            expected_frames * 8,
            "output should be frames × d_model"
        );
    }

    #[test]
    fn test_conv_stem_forward_empty_errors() {
        let stem = ConvStem::new(4, 8);
        let result = stem.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv_stem_total_stride() {
        assert_eq!(CONV_STEM_TOTAL_STRIDE, 441 * 4 * 2);
    }

    // =========================================================================
    // GroupNorm Tests
    // =========================================================================

    #[test]
    fn test_groupnorm_new() {
        let gn = GroupNorm::new(4, 16);
        assert_eq!(gn.num_groups, 4);
        assert_eq!(gn.num_channels, 16);
        assert_eq!(gn.weight.len(), 16);
        assert_eq!(gn.bias.len(), 16);
        // Defaults: weight=1, bias=0 (identity transform)
        assert!((gn.weight[0] - 1.0).abs() < f32::EPSILON);
        assert!((gn.bias[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_groupnorm_forward_identity() {
        // With weight=1, bias=0, GroupNorm normalizes to zero mean, unit variance
        let gn = GroupNorm::new(1, 4);
        // Single position, 4 channels, 1 group
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = gn.forward(&input, 1).expect("forward should succeed");
        assert_eq!(output.len(), 4);
        // Mean = 2.5, std ≈ 1.118
        // Normalized: (x - mean) / std
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "output mean should be ~0, got {mean}");
    }

    #[test]
    fn test_groupnorm_forward_multi_position() {
        let gn = GroupNorm::new(2, 4);
        // 2 positions, 4 channels, 2 groups (2 channels each)
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // pos 0
            5.0, 6.0, 7.0, 8.0, // pos 1
        ];
        let output = gn.forward(&input, 2).expect("forward should succeed");
        assert_eq!(output.len(), 8);
        // All values should be finite
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_groupnorm_forward_wrong_size() {
        let gn = GroupNorm::new(1, 4);
        let input = vec![1.0, 2.0, 3.0]; // Wrong size: not divisible by 4
        let result = gn.forward(&input, 1);
        assert!(result.is_err());
    }

    // =========================================================================
    // LayerNormStem Tests
    // =========================================================================

    #[test]
    fn test_layernorm_stem_new() {
        let ln = LayerNormStem::new(8);
        assert_eq!(ln.normalized_shape, 8);
        assert_eq!(ln.weight.len(), 8);
        assert!((ln.weight[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_layernorm_stem_forward_identity() {
        let ln = LayerNormStem::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.forward(&input).expect("forward should succeed");
        assert_eq!(output.len(), 4);
        // With weight=1, output should be normalized (zero mean)
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "output mean should be ~0, got {mean}");
    }

    #[test]
    fn test_layernorm_stem_forward_multi_position() {
        let ln = LayerNormStem::new(4);
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // pos 0
            5.0, 6.0, 7.0, 8.0, // pos 1
        ];
        let output = ln.forward(&input).expect("forward should succeed");
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_layernorm_stem_wrong_size() {
        let ln = LayerNormStem::new(4);
        let input = vec![1.0, 2.0, 3.0]; // Not divisible by 4
        let result = ln.forward(&input);
        assert!(result.is_err());
    }
}

//! Learned convolutional stem for Moonshine audio frontend
//!
//! Replaces the mel filterbank + encoder ConvFrontend for Moonshine models.
//! Three Conv1d layers with normalization process raw audio waveform directly:
//!
//! 1. Conv1d(1, C, kernel=127, stride=64, no bias) + tanh + GroupNorm
//! 2. Conv1d(C, 2*C, kernel=7, stride=3) + GELU
//! 3. Conv1d(2*C, C, kernel=3, stride=2) + GELU
//!
//! No LayerNorm in the stem — the encoder's post-block LayerNorm handles that.
//! Total stride: 64 × 3 × 2 = 384 samples per output frame.
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
    /// Intermediate channel count
    pub intermediate_channels: usize,
    /// Output model dimension
    pub d_model: usize,
}

/// Total stride of the Moonshine conv stem (64 × 3 × 2)
pub const CONV_STEM_TOTAL_STRIDE: usize = 384;

impl ConvStem {
    /// Create a new learned conv stem matching HF Moonshine architecture
    ///
    /// Architecture (from HuggingFace `MoonshineEncoder`):
    /// - conv1: `Conv1d(1, d_model, kernel=127, stride=64, pad=0, bias=False)`
    /// - tanh → GroupNorm(1, d_model)
    /// - conv2: `Conv1d(d_model, 2*d_model, kernel=7, stride=3, pad=0, bias=True)` → GELU
    /// - conv3: `Conv1d(2*d_model, d_model, kernel=3, stride=2, pad=0, bias=True)` → GELU
    ///
    /// No LayerNorm in the stem — `encoder.layer_norm` is applied after all encoder blocks.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (288 for tiny, 416 for base)
    #[must_use]
    pub fn new(d_model: usize) -> Self {
        Self {
            conv1: Conv1d::new(1, d_model, 127, 64, 0),
            groupnorm: GroupNorm::new(1, d_model),
            conv2: Conv1d::new(d_model, 2 * d_model, 7, 3, 0),
            conv3: Conv1d::new(2 * d_model, d_model, 3, 2, 0),
            intermediate_channels: 2 * d_model,
            d_model,
        }
    }

    /// Forward pass: raw audio → encoder features
    ///
    /// Pipeline: conv1 → tanh → GroupNorm → conv2 → GELU → conv3 → GELU
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

        // Layer 1: raw audio → d_model channels (stride 64, no bias)
        let x = self.conv1.forward(audio)?;
        let x = crate::simd::tanh_activation(&x);
        let seq_len = x.len() / self.d_model;
        let x = self.groupnorm.forward(&x, seq_len)?;

        // Layer 2: d_model → 2*d_model (stride 3)
        let x = self.conv2.forward(&x)?;
        let x = crate::simd::gelu(&x);

        // Layer 3: 2*d_model → d_model (stride 2)
        let x = self.conv3.forward(&x)?;
        let x = crate::simd::gelu(&x);

        Ok(x)
    }

    /// Forward pass with activation probing
    ///
    /// Same logic as [`forward()`](Self::forward) but records activation snapshots
    /// at each sub-layer boundary for numerical parity debugging.
    ///
    /// # Errors
    /// Returns error if audio is empty or convolution fails
    pub fn forward_probed(
        &self,
        audio: &[f32],
        probe: &mut crate::probe::ActivationProbe,
    ) -> crate::error::WhisperResult<Vec<f32>> {
        if audio.is_empty() {
            return Err(WhisperError::Audio("empty audio input".into()));
        }

        // Layer 1: raw audio → d_model channels (stride 64, no bias)
        let x = self.conv1.forward(audio)?;
        let seq_len = x.len() / self.d_model;
        probe.record("conv_stem.conv1_out", &x, &[seq_len, self.d_model]);

        let x = crate::simd::tanh_activation(&x);
        probe.record("conv_stem.tanh_out", &x, &[seq_len, self.d_model]);

        let x = self.groupnorm.forward(&x, seq_len)?;
        probe.record("conv_stem.groupnorm_out", &x, &[seq_len, self.d_model]);

        // Layer 2: d_model → 2*d_model (stride 3)
        let x = self.conv2.forward(&x)?;
        let seq_len2 = x.len() / self.intermediate_channels;
        probe.record(
            "conv_stem.conv2_out",
            &x,
            &[seq_len2, self.intermediate_channels],
        );

        let x = crate::simd::gelu(&x);

        // Layer 3: 2*d_model → d_model (stride 2)
        let x = self.conv3.forward(&x)?;
        let seq_len3 = x.len() / self.d_model;
        probe.record("conv_stem.conv3_out", &x, &[seq_len3, self.d_model]);

        let x = crate::simd::gelu(&x);
        probe.record("conv_stem.gelu3_out", &x, &[seq_len3, self.d_model]);

        Ok(x)
    }

    /// Calculate output frame count for a given number of audio samples
    ///
    /// HF Moonshine conv stem: conv1(k=127,s=64) → conv2(k=7,s=3) → conv3(k=3,s=2)
    /// All padding=0.
    #[must_use]
    pub fn output_frames(audio_samples: usize) -> usize {
        if audio_samples == 0 {
            return 0;
        }
        // Layer 1: (samples - 127) / 64 + 1
        let after_conv1 = if audio_samples >= 127 {
            (audio_samples - 127) / 64 + 1
        } else {
            0
        };
        // Layer 2: (after_conv1 - 7) / 3 + 1
        let after_conv2 = if after_conv1 >= 7 {
            (after_conv1 - 7) / 3 + 1
        } else {
            0
        };
        // Layer 3: (after_conv2 - 3) / 2 + 1
        if after_conv2 >= 3 {
            (after_conv2 - 3) / 2 + 1
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
        let stem = ConvStem::new(288);
        assert_eq!(stem.conv1.in_channels, 1);
        assert_eq!(stem.conv1.out_channels, 288);
        assert_eq!(stem.conv1.kernel_size, 127);
        assert_eq!(stem.conv1.stride, 64);
        assert_eq!(stem.conv2.in_channels, 288);
        assert_eq!(stem.conv2.out_channels, 576);
        assert_eq!(stem.conv2.kernel_size, 7);
        assert_eq!(stem.conv2.stride, 3);
        assert_eq!(stem.conv3.in_channels, 576);
        assert_eq!(stem.conv3.out_channels, 288);
        assert_eq!(stem.conv3.kernel_size, 3);
        assert_eq!(stem.conv3.stride, 2);
        assert_eq!(stem.d_model, 288);
        assert_eq!(stem.groupnorm.num_channels, 288);
    }

    #[test]
    fn test_conv_stem_output_frames() {
        // 1.5s at 16kHz = 24,000 samples
        let frames = ConvStem::output_frames(24_000);
        assert!(
            frames > 0,
            "1.5s audio should produce >0 frames, got {frames}"
        );
        // With stride 64*3*2=384, expect ~24000/384 ≈ 62 frames (minus kernel overlap)
        assert!(
            frames > 30 && frames < 100,
            "1.5s audio should produce 30-100 frames, got {frames}"
        );

        // 30s at 16kHz = 480,000 samples
        let frames_30s = ConvStem::output_frames(480_000);
        assert!(
            frames_30s > 500,
            "30s audio should produce >500 frames, got {frames_30s}"
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
        // Very short audio (less than conv1 kernel)
        assert_eq!(ConvStem::output_frames(0), 0);
        assert_eq!(ConvStem::output_frames(126), 0);
    }

    #[test]
    fn test_conv_stem_forward_produces_output() {
        let stem = ConvStem::new(8);
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
        let stem = ConvStem::new(8);
        let result = stem.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv_stem_total_stride() {
        assert_eq!(CONV_STEM_TOTAL_STRIDE, 64 * 3 * 2);
    }

    // =========================================================================
    // output_frames edge cases (WAPR-MOONSHINE-014)
    // =========================================================================

    #[test]
    fn test_output_frames_boundary_conditions() {
        // 0 samples → 0 frames
        assert_eq!(ConvStem::output_frames(0), 0);
        // 126 samples < conv1 kernel (127) → 0 frames
        assert_eq!(ConvStem::output_frames(126), 0);
        // 127 samples = conv1 kernel → conv1 produces 1 output,
        // but that's < conv2 kernel (7), so still 0 final frames
        assert_eq!(ConvStem::output_frames(127), 0);
        // 128 samples → conv1 produces 1 output, still < 7 for conv2
        assert_eq!(ConvStem::output_frames(128), 0);
        // First non-zero: need conv1 to produce ≥7 outputs,
        // then conv2 to produce ≥3, then conv3 to produce ≥1
        // conv1 ≥ 7: (n - 127) / 64 + 1 ≥ 7 → n ≥ 127 + 6*64 = 511
        // conv2 output from 7: (7 - 7)/3 + 1 = 1 → < 3 for conv3
        // conv2 ≥ 3 needs conv1 ≥ 13: n ≥ 127 + 12*64 = 895
        // conv3 from 3: (3 - 3)/2 + 1 = 1 ✓
        let first_nonzero = ConvStem::output_frames(895);
        assert_eq!(first_nonzero, 1, "895 samples should produce exactly 1 frame");
        assert_eq!(
            ConvStem::output_frames(894),
            0,
            "894 samples should still produce 0 frames"
        );
    }

    #[test]
    fn test_output_frames_monotonicity() {
        // output_frames(n) <= output_frames(n+1) for all n in [0, 100_000]
        let mut prev = ConvStem::output_frames(0);
        for n in 1..=100_000 {
            let curr = ConvStem::output_frames(n);
            assert!(
                curr >= prev,
                "Monotonicity violated: output_frames({}) = {} < output_frames({}) = {}",
                n,
                curr,
                n - 1,
                prev
            );
            prev = curr;
        }
    }

    #[test]
    fn test_output_frames_known_durations() {
        // Pre-computed expected values via the formula:
        // conv1: (n - 127) / 64 + 1
        // conv2: (conv1 - 7) / 3 + 1
        // conv3: (conv2 - 3) / 2 + 1

        // 1s = 16,000 samples
        // conv1: (16000 - 127)/64 + 1 = 248 + 1 = 249
        // conv2: (249 - 7)/3 + 1 = 80 + 1 = 81
        // conv3: (81 - 3)/2 + 1 = 39 + 1 = 40
        assert_eq!(ConvStem::output_frames(16_000), 40);

        // 1.5s = 24,000 samples
        // conv1: (24000 - 127)/64 + 1 = 372 + 1 = 373
        // conv2: (373 - 7)/3 + 1 = 122 + 1 = 123
        // conv3: (123 - 3)/2 + 1 = 60 + 1 = 61
        assert_eq!(ConvStem::output_frames(24_000), 61);

        // 3s = 48,000 samples
        // conv1: (48000 - 127)/64 + 1 = 747 + 1 = 748
        // conv2: (748 - 7)/3 + 1 = 247 + 1 = 248
        // conv3: (248 - 3)/2 + 1 = 122 + 1 = 123
        assert_eq!(ConvStem::output_frames(48_000), 123);

        // 10s = 160,000 samples
        // conv1: (160000 - 127)/64 + 1 = 2497 + 1 = 2498
        // conv2: (2498 - 7)/3 + 1 = 830 + 1 = 831
        // conv3: (831 - 3)/2 + 1 = 414 + 1 = 415
        assert_eq!(ConvStem::output_frames(160_000), 415);

        // 30s = 480,000 samples
        // conv1: (480000 - 127)/64 + 1 = 7497 + 1 = 7498
        // conv2: (7498 - 7)/3 + 1 = 2497 + 1 = 2498
        // conv3: (2498 - 3)/2 + 1 = 1247 + 1 = 1248
        assert_eq!(ConvStem::output_frames(480_000), 1248);
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

}

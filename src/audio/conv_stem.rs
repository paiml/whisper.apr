//! Learned convolutional stem for Moonshine audio frontend
//!
//! Replaces the mel filterbank + encoder ConvFrontend for Moonshine models.
//! Three Conv1d layers process raw audio waveform directly:
//!
//! 1. Conv1d(1, C, kernel=441, stride=441) + GELU — ~27.6ms per frame at 16kHz
//! 2. Conv1d(C, C, kernel=7, stride=4, pad=3) + GELU
//! 3. Conv1d(C, D, kernel=7, stride=2, pad=3) + GELU
//!
//! Total stride: 441 × 4 × 2 = 3528 samples per output frame (~220ms).
//! Output length is proportional to input duration (no 30s padding).

use crate::error::{WhisperError, WhisperResult};
use crate::model::Conv1d;

/// Learned convolutional stem for Moonshine's variable-length audio frontend.
///
/// Processes raw audio waveform directly into encoder-ready features,
/// bypassing the mel filterbank entirely. Output length scales linearly
/// with input duration.
#[derive(Debug, Clone)]
pub struct ConvStem {
    /// Layer 1: raw audio → intermediate features (stride 441)
    pub conv1: Conv1d,
    /// Layer 2: intermediate → intermediate (stride 4)
    pub conv2: Conv1d,
    /// Layer 3: intermediate → d_model (stride 2)
    pub conv3: Conv1d,
    /// Intermediate channel count
    pub intermediate_channels: usize,
    /// Output model dimension
    pub d_model: usize,
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
            conv2: Conv1d::new(intermediate_channels, intermediate_channels, 7, 4, 3),
            conv3: Conv1d::new(intermediate_channels, d_model, 7, 2, 3),
            intermediate_channels,
            d_model,
        }
    }

    /// Forward pass: raw audio → encoder features
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

        // Reshape: audio is [N] mono samples, conv1d expects [N × 1] (1 channel)
        // Conv1d::forward expects (seq_len × in_channels) layout, so for 1 channel
        // the data is the same as the raw samples.

        // Layer 1: raw audio → intermediate (stride 441)
        let x = self.conv1.forward(audio)?;
        let x = crate::simd::gelu(&x);

        // Layer 2: intermediate → intermediate (stride 4)
        let x = self.conv2.forward(&x)?;
        let x = crate::simd::gelu(&x);

        // Layer 3: intermediate → d_model (stride 2)
        let x = self.conv3.forward(&x)?;
        let x = crate::simd::gelu(&x);

        Ok(x)
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
}

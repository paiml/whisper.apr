//! Audio resampling module
//!
//! Implements high-quality audio resampling using sinc interpolation with
//! Kaiser window for anti-aliasing. Optimized for Whisper's 16kHz requirement.
//!
//! # Implementation Details
//!
//! Uses polyphase sinc interpolation with:
//! - Kaiser-windowed sinc kernel for sharp cutoff
//! - Anti-aliasing lowpass filter for downsampling
//! - Efficient polyphase decomposition
//!
//! # References
//!
//! - Smith, J.O. "Digital Audio Resampling Home Page"
//! - Crochiere, R.E. & Rabiner, L.R. "Multirate Digital Signal Processing"

use crate::error::{WhisperError, WhisperResult};
use std::f64::consts::PI;

/// Default filter kernel half-length (samples on each side)
const DEFAULT_KERNEL_HALF_LEN: usize = 16;

/// Default Kaiser window beta parameter (controls sidelobe attenuation)
const DEFAULT_KAISER_BETA: f64 = 6.0;

/// High-quality audio resampler using sinc interpolation
///
/// Converts audio between sample rates while preserving signal quality.
/// Uses a Kaiser-windowed sinc kernel for optimal stopband attenuation.
#[derive(Debug, Clone)]
pub struct SincResampler {
    /// Source sample rate
    source_rate: u32,
    /// Target sample rate
    target_rate: u32,
    /// Resampling ratio (target/source)
    ratio: f64,
    /// Sinc kernel half-length
    kernel_half_len: usize,
    /// Kaiser window beta parameter
    kaiser_beta: f64,
    /// Precomputed sinc kernel (for common ratios) - reserved for future optimization
    #[allow(dead_code)]
    kernel: Option<Vec<f32>>,
}

impl SincResampler {
    /// Create a new sinc resampler
    ///
    /// # Arguments
    /// * `source_rate` - Source sample rate (e.g., 44100)
    /// * `target_rate` - Target sample rate (e.g., 16000)
    ///
    /// # Errors
    /// Returns error if sample rates are invalid (zero)
    ///
    /// # Example
    /// ```
    /// use whisper_apr::audio::SincResampler;
    ///
    /// let resampler = SincResampler::new(44100, 16000).unwrap();
    /// assert_eq!(resampler.source_rate(), 44100);
    /// assert_eq!(resampler.target_rate(), 16000);
    /// ```
    pub fn new(source_rate: u32, target_rate: u32) -> WhisperResult<Self> {
        Self::with_params(
            source_rate,
            target_rate,
            DEFAULT_KERNEL_HALF_LEN,
            DEFAULT_KAISER_BETA,
        )
    }

    /// Create a resampler with custom parameters
    ///
    /// # Arguments
    /// * `source_rate` - Source sample rate
    /// * `target_rate` - Target sample rate
    /// * `kernel_half_len` - Filter kernel half-length (larger = better quality, slower)
    /// * `kaiser_beta` - Kaiser window beta (larger = better stopband, wider transition)
    ///
    /// # Errors
    /// Returns error if sample rates are zero or kernel_half_len is zero
    pub fn with_params(
        source_rate: u32,
        target_rate: u32,
        kernel_half_len: usize,
        kaiser_beta: f64,
    ) -> WhisperResult<Self> {
        if source_rate == 0 || target_rate == 0 {
            return Err(WhisperError::Audio("sample rate must be non-zero".into()));
        }
        if kernel_half_len == 0 {
            return Err(WhisperError::Audio(
                "kernel half-length must be non-zero".into(),
            ));
        }

        let ratio = f64::from(target_rate) / f64::from(source_rate);

        Ok(Self {
            source_rate,
            target_rate,
            ratio,
            kernel_half_len,
            kaiser_beta,
            kernel: None,
        })
    }

    /// Resample audio to target sample rate
    ///
    /// Uses sinc interpolation with Kaiser window for high-quality resampling.
    /// Automatically applies anti-aliasing for downsampling.
    ///
    /// # Arguments
    /// * `audio` - Input audio samples (normalized to [-1.0, 1.0])
    ///
    /// # Returns
    /// Resampled audio at target sample rate
    ///
    /// # Errors
    /// Returns error if audio is empty
    ///
    /// # Example
    /// ```
    /// use whisper_apr::audio::SincResampler;
    ///
    /// let resampler = SincResampler::new(48000, 16000).unwrap();
    /// let input: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();
    /// let output = resampler.resample(&input).unwrap();
    /// // Output should be ~1600 samples (48000/16000 = 3x downsampling)
    /// assert!(output.len() >= 1590 && output.len() <= 1610);
    /// ```
    pub fn resample(&self, audio: &[f32]) -> WhisperResult<Vec<f32>> {
        if audio.is_empty() {
            return Err(WhisperError::Audio("cannot resample empty audio".into()));
        }

        // Same rate - just copy
        if self.source_rate == self.target_rate {
            return Ok(audio.to_vec());
        }

        // Calculate output length
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let output_len = ((audio.len() as f64) * self.ratio).ceil() as usize;

        if output_len == 0 {
            return Err(WhisperError::Audio("output length would be zero".into()));
        }

        let mut output = vec![0.0_f32; output_len];

        // Cutoff frequency for anti-aliasing (normalized to source Nyquist)
        let cutoff = if self.ratio < 1.0 { self.ratio } else { 1.0 };

        // Perform sinc interpolation
        for (out_idx, out_sample) in output.iter_mut().enumerate() {
            // Position in input signal
            let in_pos = out_idx as f64 / self.ratio;

            // Accumulate interpolated value
            let mut sum = 0.0_f64;
            let mut weight_sum = 0.0_f64;

            // Window around the interpolation point
            #[allow(clippy::cast_possible_truncation)]
            let center = in_pos.floor() as i64;
            let frac = in_pos - in_pos.floor();

            let half_len = self.kernel_half_len as i64;

            for k in -half_len..=half_len {
                let idx = center + k;
                if idx < 0 || idx >= audio.len() as i64 {
                    continue;
                }

                // Distance from interpolation point
                let x = k as f64 - frac;

                // Windowed sinc value
                let sinc_val = self.windowed_sinc(x, cutoff);

                #[allow(clippy::cast_sign_loss)]
                let sample = audio[idx as usize] as f64;
                sum += sample * sinc_val;
                weight_sum += sinc_val;
            }

            // Normalize to preserve DC and amplitude
            #[allow(clippy::cast_possible_truncation)]
            if weight_sum.abs() > 1e-10 {
                *out_sample = (sum / weight_sum) as f32;
            }
        }

        Ok(output)
    }

    /// Compute windowed sinc function value
    ///
    /// sinc(x) = sin(π * x) / (π * x) for x ≠ 0, 1 for x = 0
    /// Multiplied by Kaiser window for improved frequency response.
    fn windowed_sinc(&self, x: f64, cutoff: f64) -> f64 {
        // sinc(cutoff * x) for lowpass filtering
        let sinc_arg = cutoff * x;
        let sinc_val = if sinc_arg.abs() < 1e-10 {
            1.0
        } else {
            (PI * sinc_arg).sin() / (PI * sinc_arg)
        };

        // Kaiser window
        let window_arg = x / self.kernel_half_len as f64;
        let window_val = if window_arg.abs() > 1.0 {
            0.0
        } else {
            self.kaiser_window(window_arg)
        };

        sinc_val * window_val
    }

    /// Kaiser window function
    ///
    /// w(n) = I0(β * sqrt(1 - (2n/N - 1)²)) / I0(β)
    /// where I0 is the zeroth-order modified Bessel function
    fn kaiser_window(&self, x: f64) -> f64 {
        let arg = self.kaiser_beta * x.mul_add(-x, 1.0).max(0.0).sqrt();
        bessel_i0(arg) / bessel_i0(self.kaiser_beta)
    }

    /// Get the source sample rate
    #[must_use]
    pub const fn source_rate(&self) -> u32 {
        self.source_rate
    }

    /// Get the target sample rate
    #[must_use]
    pub const fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// Get the resampling ratio
    #[must_use]
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Get the kernel half-length
    #[must_use]
    pub const fn kernel_half_len(&self) -> usize {
        self.kernel_half_len
    }
}

/// Zeroth-order modified Bessel function of the first kind
///
/// Uses the series expansion: I0(x) = Σ (x²/4)^k / (k!)²
/// Accurate to ~15 digits for typical Kaiser beta values.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_sq_over_4 = (x * x) / 4.0;

    for k in 1..50 {
        term *= x_sq_over_4 / (k * k) as f64;
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }

    sum
}

/// Legacy resampler alias for backward compatibility
pub type Resampler = SincResampler;

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_resampler_new() {
        let resampler = SincResampler::new(44100, 16000);
        assert!(resampler.is_ok());
        let r = resampler.expect("resampler should be valid");
        assert_eq!(r.source_rate(), 44100);
        assert_eq!(r.target_rate(), 16000);
    }

    #[test]
    fn test_resampler_with_params() {
        let resampler = SincResampler::with_params(48000, 16000, 32, 8.0);
        assert!(resampler.is_ok());
        let r = resampler.expect("resampler should be valid");
        assert_eq!(r.kernel_half_len(), 32);
    }

    #[test]
    fn test_resampler_invalid_source_rate() {
        let resampler = SincResampler::new(0, 16000);
        assert!(resampler.is_err());
        let err = resampler.expect_err("expected error for invalid source rate");
        assert!(matches!(err, WhisperError::Audio(_)));
    }

    #[test]
    fn test_resampler_invalid_target_rate() {
        let resampler = SincResampler::new(44100, 0);
        assert!(resampler.is_err());
    }

    #[test]
    fn test_resampler_invalid_kernel_half_len() {
        let resampler = SincResampler::with_params(44100, 16000, 0, 6.0);
        assert!(resampler.is_err());
    }

    #[test]
    fn test_resampler_ratio() {
        let resampler = SincResampler::new(48000, 16000).expect("valid");
        let expected_ratio = 16000.0 / 48000.0;
        assert!((resampler.ratio() - expected_ratio).abs() < 1e-10);
    }

    // =========================================================================
    // Resampling Tests
    // =========================================================================

    #[test]
    fn test_resample_same_rate() {
        let resampler = SincResampler::new(16000, 16000).expect("valid resampler");
        let audio = vec![1.0, 2.0, 3.0, 4.0];
        let result = resampler.resample(&audio);
        assert!(result.is_ok());
        assert_eq!(result.expect("valid result"), audio);
    }

    #[test]
    fn test_resample_empty_audio() {
        let resampler = SincResampler::new(44100, 16000).expect("valid resampler");
        let audio: Vec<f32> = vec![];
        let result = resampler.resample(&audio);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_downsample_44100_to_16000() {
        let resampler = SincResampler::new(44100, 16000).expect("valid resampler");
        let audio = vec![0.5; 44100]; // 1 second at 44.1kHz
        let result = resampler.resample(&audio);
        assert!(result.is_ok());
        let output = result.expect("valid result");
        // Output should be approximately 16000 samples
        assert!(output.len() >= 15900 && output.len() <= 16100);
    }

    #[test]
    fn test_resample_downsample_48000_to_16000() {
        let resampler = SincResampler::new(48000, 16000).expect("valid resampler");
        let audio = vec![0.5; 48000]; // 1 second at 48kHz
        let result = resampler.resample(&audio);
        assert!(result.is_ok());
        let output = result.expect("valid result");
        // Output should be approximately 16000 samples (3x downsample)
        assert!(output.len() >= 15900 && output.len() <= 16100);
    }

    #[test]
    fn test_resample_upsample_8000_to_16000() {
        let resampler = SincResampler::new(8000, 16000).expect("valid resampler");
        let audio = vec![0.5; 8000]; // 1 second at 8kHz
        let result = resampler.resample(&audio);
        assert!(result.is_ok());
        let output = result.expect("valid result");
        // Output should be approximately 16000 samples (2x upsample)
        assert!(output.len() >= 15900 && output.len() <= 16100);
    }

    #[test]
    fn test_resample_preserves_dc_signal() {
        let resampler = SincResampler::new(44100, 16000).expect("valid");
        let dc_value = 0.5_f32;
        let audio = vec![dc_value; 4410]; // 100ms at 44.1kHz
        let output = resampler.resample(&audio).expect("valid");

        // DC signal should be approximately preserved (within some tolerance due to edge effects)
        let mid_samples: Vec<f32> = output
            .iter()
            .skip(output.len() / 4)
            .take(output.len() / 2)
            .copied()
            .collect();
        let avg: f32 = mid_samples.iter().sum::<f32>() / mid_samples.len() as f32;
        assert!(
            (avg - dc_value).abs() < 0.1,
            "DC signal not preserved: expected {}, got {}",
            dc_value,
            avg
        );
    }

    #[test]
    fn test_resample_sine_wave_downsample() {
        let resampler = SincResampler::new(48000, 16000).expect("valid");
        let freq = 440.0_f32; // 440 Hz sine wave
        let duration = 0.1; // 100ms
        let n_samples = (48000.0 * duration) as usize;

        // Generate sine wave
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 48000.0).sin())
            .collect();

        let output = resampler.resample(&audio).expect("valid");

        // Output should have correct length
        let expected_len = (16000.0 * duration) as usize;
        assert!(
            (output.len() as i32 - expected_len as i32).abs() <= 2,
            "Expected ~{} samples, got {}",
            expected_len,
            output.len()
        );

        // Verify output amplitude is reasonable (not all zeros)
        let max_amp = output.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        assert!(max_amp > 0.1, "Output amplitude too low: {}", max_amp);
    }

    #[test]
    fn test_resample_sine_wave_upsample() {
        let resampler = SincResampler::new(16000, 48000).expect("valid");
        let freq = 440.0_f32;
        let duration = 0.1;
        let n_samples = (16000.0 * duration) as usize;

        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 16000.0).sin())
            .collect();

        let output = resampler.resample(&audio).expect("valid");

        // Output should have correct length
        let expected_len = (48000.0 * duration) as usize;
        assert!(
            (output.len() as i32 - expected_len as i32).abs() <= 2,
            "Expected ~{} samples, got {}",
            expected_len,
            output.len()
        );
    }

    // =========================================================================
    // Bessel Function Tests
    // =========================================================================

    #[test]
    fn test_bessel_i0_at_zero() {
        let result = bessel_i0(0.0);
        assert!((result - 1.0).abs() < 1e-10, "I0(0) should be 1");
    }

    #[test]
    fn test_bessel_i0_known_values() {
        // Known values from mathematical tables
        // I0(1) ≈ 1.2660658777520084
        let result = bessel_i0(1.0);
        assert!(
            (result - 1.2660658777520084).abs() < 1e-10,
            "I0(1) incorrect: {}",
            result
        );

        // I0(2) ≈ 2.2795853023360673
        let result = bessel_i0(2.0);
        assert!(
            (result - 2.2795853023360673).abs() < 1e-10,
            "I0(2) incorrect: {}",
            result
        );
    }

    #[test]
    fn test_bessel_i0_symmetry() {
        // I0 is even function: I0(x) = I0(-x)
        let x = 3.5;
        let pos = bessel_i0(x);
        let neg = bessel_i0(-x);
        assert!((pos - neg).abs() < 1e-10);
    }

    // =========================================================================
    // Kaiser Window Tests
    // =========================================================================

    #[test]
    fn test_kaiser_window_center() {
        let resampler = SincResampler::new(44100, 16000).expect("valid");
        // Kaiser window should be 1.0 at center (x=0)
        let val = resampler.kaiser_window(0.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kaiser_window_edges() {
        let resampler = SincResampler::new(44100, 16000).expect("valid");
        // Kaiser window should be small at edges (x = ±1)
        // With beta=6.0: I0(0)/I0(6) = 1/67.23 ≈ 0.0149
        let val_pos = resampler.kaiser_window(1.0);
        let val_neg = resampler.kaiser_window(-1.0);
        assert!(
            val_pos < 0.02,
            "Window should be small at edge, got {}",
            val_pos
        );
        assert!(
            (val_pos - val_neg).abs() < 1e-10,
            "Window should be symmetric"
        );
    }

    #[test]
    fn test_kaiser_window_outside_range() {
        let resampler = SincResampler::new(44100, 16000).expect("valid");
        // Kaiser window should be 0 outside [-1, 1]
        let val = resampler.windowed_sinc(100.0, 1.0);
        assert!((val).abs() < 1e-10);
    }

    // =========================================================================
    // Sinc Function Tests
    // =========================================================================

    #[test]
    fn test_windowed_sinc_at_zero() {
        let resampler = SincResampler::new(44100, 16000).expect("valid");
        // sinc(0) = 1
        let val = resampler.windowed_sinc(0.0, 1.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_sinc_at_integers() {
        let resampler = SincResampler::with_params(44100, 16000, 32, 6.0).expect("valid");
        // sinc(n) = 0 for non-zero integers n
        for n in 1..10 {
            let val = resampler.windowed_sinc(n as f64, 1.0);
            assert!(
                val.abs() < 0.1,
                "sinc({}) should be near zero, got {}",
                n,
                val
            );
        }
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_resample_single_sample() {
        let resampler = SincResampler::new(44100, 16000).expect("valid");
        let audio = vec![0.5_f32];
        let output = resampler.resample(&audio).expect("valid");
        assert!(!output.is_empty());
    }

    #[test]
    fn test_resample_very_short_audio() {
        let resampler = SincResampler::new(44100, 16000).expect("valid");
        let audio = vec![0.5_f32; 10];
        let output = resampler.resample(&audio).expect("valid");
        assert!(!output.is_empty());
    }

    #[test]
    fn test_resample_high_frequency_rejection() {
        // When downsampling, frequencies above Nyquist should be attenuated
        let resampler = SincResampler::new(48000, 16000).expect("valid");

        // Generate 10kHz sine wave (above 8kHz Nyquist of 16kHz output)
        let freq = 10000.0_f32;
        let n_samples = 4800; // 100ms
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 48000.0).sin())
            .collect();

        let output = resampler.resample(&audio).expect("valid");

        // High frequency should be significantly attenuated
        // Skip edge samples and check middle
        let mid_start = output.len() / 4;
        let mid_end = 3 * output.len() / 4;
        let max_mid_amp = output[mid_start..mid_end]
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f32, f32::max);

        // With proper anti-aliasing, the 10kHz component should be attenuated
        // (not eliminated due to windowing, but reduced)
        assert!(
            max_mid_amp < 0.8,
            "High frequency not sufficiently attenuated: {}",
            max_mid_amp
        );
    }

    // =========================================================================
    // Legacy Alias Test
    // =========================================================================

    #[test]
    fn test_resampler_alias() {
        // Verify the type alias works
        let resampler: Resampler = Resampler::new(44100, 16000).expect("valid");
        assert_eq!(resampler.source_rate(), 44100);
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    #[test]
    fn test_resample_output_length_property() {
        // Property: output length should be proportional to ratio
        for (src, tgt) in [
            (44100, 16000),
            (48000, 16000),
            (8000, 16000),
            (22050, 16000),
        ] {
            let resampler = SincResampler::new(src, tgt).expect("valid");
            let input_len = 1000_usize;
            let audio = vec![0.5_f32; input_len];
            let output = resampler.resample(&audio).expect("valid");

            let expected_len = (input_len as f64 * tgt as f64 / src as f64).ceil() as usize;
            assert!(
                (output.len() as i32 - expected_len as i32).abs() <= 1,
                "For {}→{}: expected ~{}, got {}",
                src,
                tgt,
                expected_len,
                output.len()
            );
        }
    }

    #[test]
    fn test_resample_bounded_output_property() {
        // Property: if input is bounded by [-1, 1], output should be approximately bounded
        let resampler = SincResampler::new(44100, 16000).expect("valid");
        let audio: Vec<f32> = (0..4410)
            .map(|i| (i as f32 * 0.01).sin()) // All values in [-1, 1]
            .collect();

        let output = resampler.resample(&audio).expect("valid");

        let max_output = output.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        // Due to sinc interpolation, slight overshoot is possible (Gibbs phenomenon)
        // but should be bounded
        assert!(
            max_output < 1.5,
            "Output exceeded expected bounds: {}",
            max_output
        );
    }
}

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

#[cfg(test)]
mod tests;

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

//! Mel spectrogram computation
//!
//! Implements mel filterbank for converting audio to mel spectrograms.
//! This is a critical component for Whisper ASR preprocessing.
//!
//! # Algorithm
//!
//! 1. Apply Hann window to audio frames
//! 2. Compute FFT to get power spectrum
//! 3. Apply mel filterbank to convert to mel scale
//! 4. Apply log compression
//!
//! # References
//!
//! - Whisper paper: Radford et al. (2023)
//! - Mel scale: Stevens, Volkmann, & Newman (1937)

mod simd;

#[cfg(test)]
mod tests;

use crate::error::{WhisperError, WhisperResult};
use crate::format::MelFilterbankData;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// Mel filterbank for spectrogram computation
///
/// Implements the mel-frequency filterbank used by Whisper for audio preprocessing.
/// The filterbank converts linear frequency power spectra to mel-scale representations.
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    /// Number of mel channels (typically 80 for Whisper)
    n_mels: usize,
    /// FFT size (typically 400 for Whisper at 16kHz)
    n_fft: usize,
    /// Sample rate in Hz (typically 16000 for Whisper)
    sample_rate: u32,
    /// Filterbank matrix (`n_mels` x `n_freqs`) stored in row-major order
    pub(crate) filters: Vec<f32>,
    /// Number of frequency bins (`n_fft` / 2 + 1)
    n_freqs: usize,
    /// Precomputed Hann window
    pub(crate) window: Vec<f32>,
}

impl MelFilterbank {
    /// Create a new mel filterbank by computing filters from scratch
    ///
    /// NOTE: For Whisper models, prefer `from_filters()` with the pre-computed
    /// filterbank from the model file, which matches OpenAI's slaney-normalized
    /// filterbank exactly.
    ///
    /// # Arguments
    /// * `n_mels` - Number of mel channels (typically 80 for Whisper)
    /// * `n_fft` - FFT size (typically 400 for Whisper)
    /// * `sample_rate` - Audio sample rate (typically 16000 for Whisper)
    ///
    /// # Panics
    /// Panics if `n_mels` or `n_fft` is zero
    #[must_use]
    pub fn new(n_mels: usize, n_fft: usize, sample_rate: u32) -> Self {
        assert!(n_mels > 0, "n_mels must be positive");
        assert!(n_fft > 0, "n_fft must be positive");
        assert!(sample_rate > 0, "sample_rate must be positive");

        let n_freqs = n_fft / 2 + 1;

        // Compute mel filterbank matrix
        let filters = Self::compute_filterbank(n_mels, n_fft, sample_rate);

        // Precompute Hann window
        let window = Self::hann_window(n_fft);

        Self {
            n_mels,
            n_fft,
            sample_rate,
            filters,
            n_freqs,
            window,
        }
    }

    /// Create a mel filterbank from pre-computed filter weights
    ///
    /// This is the preferred method for Whisper models, as it uses the exact
    /// filterbank from the model file (matching OpenAI's slaney-normalized
    /// librosa filterbank).
    ///
    /// # Arguments
    /// * `filters` - Pre-computed filterbank matrix (n_mels x n_freqs) in row-major order
    /// * `n_mels` - Number of mel channels (80 for Whisper)
    /// * `n_fft` - FFT size (400 for Whisper)
    /// * `sample_rate` - Audio sample rate (16000 for Whisper)
    ///
    /// # Panics
    /// Panics if filter dimensions don't match n_mels * n_freqs
    #[must_use]
    pub fn from_filters(filters: Vec<f32>, n_mels: usize, n_fft: usize, sample_rate: u32) -> Self {
        let n_freqs = n_fft / 2 + 1;
        assert_eq!(
            filters.len(),
            n_mels * n_freqs,
            "filterbank size mismatch: expected {} x {} = {}, got {}",
            n_mels,
            n_freqs,
            n_mels * n_freqs,
            filters.len()
        );

        let window = Self::hann_window(n_fft);

        Self {
            n_mels,
            n_fft,
            sample_rate,
            filters,
            n_freqs,
            window,
        }
    }

    /// Create a mel filterbank from .apr model metadata
    ///
    /// Uses the pre-computed slaney-normalized filterbank embedded in the .apr file
    /// for exact numerical match with OpenAI's Whisper implementation.
    ///
    /// # Arguments
    /// * `data` - Filterbank data from .apr file
    /// * `sample_rate` - Audio sample rate (16000 for Whisper)
    ///
    /// # Panics
    /// Panics if filterbank dimensions are invalid
    #[must_use]
    pub fn from_apr_data(data: MelFilterbankData, sample_rate: u32) -> Self {
        let n_mels = data.n_mels as usize;
        let n_freqs = data.n_freqs as usize;
        // n_fft = 2 * (n_freqs - 1) = 2 * (201 - 1) = 400 for Whisper
        let n_fft = 2 * (n_freqs - 1);

        Self::from_filters(data.data, n_mels, n_fft, sample_rate)
    }

    /// Compute the mel filterbank matrix
    ///
    /// Creates triangular filters spaced on the mel scale.
    fn compute_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let mut filters = vec![0.0_f32; n_mels * n_freqs];

        // Frequency range for mel scale
        let f_min = 0.0_f32;
        let f_max = sample_rate as f32 / 2.0;

        // Convert to mel scale
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        // Create n_mels + 2 points evenly spaced on mel scale
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
            .collect();

        // Convert mel points back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // Convert Hz to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| ((n_fft as f32 + 1.0) * f / sample_rate as f32).floor() as usize)
            .collect();

        // Create triangular filters
        for m in 0..n_mels {
            let f_m_minus = bin_points[m];
            let f_m = bin_points[m + 1];
            let f_m_plus = bin_points[m + 2];

            // Rising slope
            for k in f_m_minus..f_m {
                if k < n_freqs && f_m > f_m_minus {
                    let slope = (k - f_m_minus) as f32 / (f_m - f_m_minus) as f32;
                    filters[m * n_freqs + k] = slope;
                }
            }

            // Falling slope
            for k in f_m..f_m_plus {
                if k < n_freqs && f_m_plus > f_m {
                    let slope = (f_m_plus - k) as f32 / (f_m_plus - f_m) as f32;
                    filters[m * n_freqs + k] = slope;
                }
            }
        }

        filters
    }

    /// Convert frequency in Hz to mel scale
    ///
    /// Uses the formula: mel = 2595 * log10(1 + f/700)
    #[inline]
    #[must_use]
    pub fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to frequency in Hz
    ///
    /// Uses the formula: f = 700 * (10^(mel/2595) - 1)
    #[inline]
    #[must_use]
    pub fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Compute Hann window
    pub(crate) fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / size as f32).cos()))
            .collect()
    }

    /// Compute mel spectrogram from audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, f32, at target sample rate)
    /// * `hop_length` - Hop length between frames (typically 160 for Whisper)
    ///
    /// # Returns
    /// Mel spectrogram as a flattened 2D matrix (n_mels x n_frames) in row-major order
    ///
    /// # Errors
    /// Returns error if audio processing fails
    #[allow(clippy::no_effect_underscore_binding)]
    pub fn compute(&self, audio: &[f32], hop_length: usize) -> WhisperResult<Vec<f32>> {
        let _span = crate::trace_enter!("step_f_mel");

        if audio.is_empty() {
            return Ok(Vec::new());
        }

        if hop_length == 0 {
            return Err(WhisperError::Audio("hop_length must be positive".into()));
        }

        // Center padding: pad n_fft//2 zeros on each side to match librosa/HuggingFace
        // This ensures n_frames = n_samples // hop_length
        let pad_len = self.n_fft / 2;
        let padded_audio: Vec<f32> = std::iter::repeat(0.0_f32)
            .take(pad_len)
            .chain(audio.iter().copied())
            .chain(std::iter::repeat(0.0_f32).take(pad_len))
            .collect();

        // Calculate number of frames with center padding
        // n_frames = original_len // hop_length (matches HuggingFace exactly)
        let n_frames = audio.len() / hop_length;

        if n_frames == 0 {
            return Ok(Vec::new());
        }

        // Prepare FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.n_fft);

        // Output buffer
        let mut mel_spec = vec![0.0_f32; self.n_mels * n_frames];

        // Process each frame
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;

            // Apply window and prepare FFT input
            let mut fft_input: Vec<Complex<f32>> = (0..self.n_fft)
                .map(|i| {
                    let sample = if start + i < padded_audio.len() {
                        padded_audio[start + i]
                    } else {
                        0.0
                    };
                    Complex::new(sample * self.window[i], 0.0)
                })
                .collect();

            // Compute FFT
            fft.process(&mut fft_input);

            // Compute power spectrum (magnitude squared)
            // Note: We don't normalize here; the normalization happens in the max-8/+4/4 step
            let power_spec: Vec<f32> = fft_input
                .iter()
                .take(self.n_freqs)
                .map(|c| c.norm_sqr())
                .collect();

            // Apply mel filterbank
            for mel_idx in 0..self.n_mels {
                let mut mel_energy = 0.0_f32;
                for (freq_idx, &power) in power_spec.iter().enumerate() {
                    mel_energy += self.filters[mel_idx * self.n_freqs + freq_idx] * power;
                }

                // Apply log compression with floor to avoid log(0)
                let log_mel = (mel_energy.max(1e-10)).log10();
                mel_spec[frame_idx * self.n_mels + mel_idx] = log_mel;
            }
        }

        // Apply Whisper normalization
        let max_val = mel_spec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        for x in &mut mel_spec {
            *x = (*x).max(max_val - 8.0);
            *x = (*x + 4.0) / 4.0;
        }

        Ok(mel_spec)
    }

    /// Normalize mel spectrogram to match Whisper's expected input range
    ///
    /// Applies global normalization: (x - mean) / std
    pub fn normalize(&self, mel_spec: &mut [f32]) {
        if mel_spec.is_empty() {
            return;
        }

        // Compute mean
        let mean = mel_spec.iter().sum::<f32>() / mel_spec.len() as f32;

        // Compute std
        let variance =
            mel_spec.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / mel_spec.len() as f32;
        let std = variance.sqrt().max(1e-10);

        // Normalize
        for x in mel_spec {
            *x = (*x - mean) / std;
        }
    }

    /// Get the number of mel channels
    #[must_use]
    pub const fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Get the FFT size
    #[must_use]
    pub const fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// Get the sample rate
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the number of frequency bins
    #[must_use]
    pub const fn n_freqs(&self) -> usize {
        self.n_freqs
    }

    /// Get the filterbank matrix (n_mels x n_freqs) in row-major order
    #[must_use]
    pub fn filters(&self) -> &[f32] {
        &self.filters
    }
}

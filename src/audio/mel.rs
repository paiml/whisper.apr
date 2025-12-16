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
    filters: Vec<f32>,
    /// Number of frequency bins (`n_fft` / 2 + 1)
    n_freqs: usize,
    /// Precomputed Hann window
    window: Vec<f32>,
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
    fn hann_window(size: usize) -> Vec<f32> {
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

        // Calculate number of frames
        let n_frames = if audio.len() >= self.n_fft {
            (audio.len() - self.n_fft) / hop_length + 1
        } else {
            0
        };

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
                    let sample = if start + i < audio.len() {
                        audio[start + i]
                    } else {
                        0.0
                    };
                    Complex::new(sample * self.window[i], 0.0)
                })
                .collect();

            // Compute FFT
            fft.process(&mut fft_input);

            // Compute power spectrum (magnitude squared)
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

    /// SIMD-optimized mel filterbank application
    ///
    /// Applies the mel filterbank to a power spectrum using SIMD operations.
    /// This is significantly faster for batch processing.
    ///
    /// # Arguments
    /// * `power_spec` - Power spectrum (n_freqs values)
    ///
    /// # Returns
    /// Mel energies (n_mels values)
    pub fn apply_filterbank_simd(&self, power_spec: &[f32]) -> Vec<f32> {
        use crate::simd;

        if power_spec.len() != self.n_freqs {
            // Fallback to scalar if dimensions don't match
            return self.apply_filterbank_scalar(power_spec);
        }

        // Use SIMD dot product for each mel band
        let mut mel_energies = Vec::with_capacity(self.n_mels);

        for mel_idx in 0..self.n_mels {
            let filter_start = mel_idx * self.n_freqs;
            let filter_row = &self.filters[filter_start..filter_start + self.n_freqs];
            let energy = simd::dot(filter_row, power_spec);
            mel_energies.push(energy);
        }

        mel_energies
    }

    /// Scalar mel filterbank application (fallback)
    #[allow(clippy::needless_range_loop)]
    fn apply_filterbank_scalar(&self, power_spec: &[f32]) -> Vec<f32> {
        let mut mel_energies = vec![0.0_f32; self.n_mels];

        for mel_idx in 0..self.n_mels {
            let mut energy = 0.0_f32;
            let spec_len = power_spec.len().min(self.n_freqs);
            for freq_idx in 0..spec_len {
                energy += self.filters[mel_idx * self.n_freqs + freq_idx] * power_spec[freq_idx];
            }
            mel_energies[mel_idx] = energy;
        }

        mel_energies
    }

    /// SIMD-optimized normalization
    ///
    /// Uses SIMD operations for faster mean/variance computation.
    pub fn normalize_simd(&self, mel_spec: &mut [f32]) {
        use crate::simd;

        if mel_spec.is_empty() {
            return;
        }

        // Use SIMD for mean and variance
        let mean = simd::mean(mel_spec);
        let variance = simd::variance(mel_spec);
        let std = variance.sqrt().max(1e-10);

        // Normalize in place
        let inv_std = 1.0 / std;
        for x in mel_spec {
            *x = (*x - mean) * inv_std;
        }
    }

    /// Compute mel spectrogram with SIMD optimization
    ///
    /// Uses SIMD-accelerated filterbank application for faster processing.
    #[allow(clippy::no_effect_underscore_binding)]
    pub fn compute_simd(&self, audio: &[f32], hop_length: usize) -> WhisperResult<Vec<f32>> {
        let _span = crate::trace_enter!("step_f_mel_simd");

        if audio.is_empty() {
            return Ok(Vec::new());
        }

        if hop_length == 0 {
            return Err(WhisperError::Audio("hop_length must be positive".into()));
        }

        // Calculate number of frames
        let n_frames = if audio.len() >= self.n_fft {
            (audio.len() - self.n_fft) / hop_length + 1
        } else {
            0
        };

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
                    let sample = if start + i < audio.len() {
                        audio[start + i]
                    } else {
                        0.0
                    };
                    Complex::new(sample * self.window[i], 0.0)
                })
                .collect();

            // Compute FFT
            fft.process(&mut fft_input);

            // Compute power spectrum (magnitude squared)
            let power_spec: Vec<f32> = fft_input
                .iter()
                .take(self.n_freqs)
                .map(|c| c.norm_sqr())
                .collect();

            // Apply mel filterbank using SIMD
            let mel_energies = self.apply_filterbank_simd(&power_spec);

            // Apply log compression and store
            for (mel_idx, &energy) in mel_energies.iter().enumerate() {
                let log_mel = (energy.max(1e-10)).log10();
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // UNIT TESTS: Mel scale conversion
    // ============================================================

    #[test]
    fn test_hz_to_mel_zero() {
        let mel = MelFilterbank::hz_to_mel(0.0);
        assert!((mel - 0.0).abs() < 1e-5, "0 Hz should map to 0 mel");
    }

    #[test]
    fn test_hz_to_mel_1000hz() {
        // 1000 Hz is approximately 1000 mel (by design of the mel scale)
        let mel = MelFilterbank::hz_to_mel(1000.0);
        assert!(
            (mel - 1000.0).abs() < 50.0,
            "1000 Hz should be close to 1000 mel, got {mel}"
        );
    }

    #[test]
    fn test_mel_to_hz_roundtrip() {
        let frequencies = [0.0, 100.0, 500.0, 1000.0, 4000.0, 8000.0];
        for &hz in &frequencies {
            let mel = MelFilterbank::hz_to_mel(hz);
            let recovered = MelFilterbank::mel_to_hz(mel);
            assert!(
                (hz - recovered).abs() < 0.1,
                "Roundtrip failed for {hz} Hz: got {recovered}"
            );
        }
    }

    #[test]
    fn test_mel_scale_monotonic() {
        let mut prev_mel = -1.0_f32;
        for hz in (0..8000).step_by(100) {
            let mel = MelFilterbank::hz_to_mel(hz as f32);
            assert!(
                mel > prev_mel,
                "Mel scale should be monotonically increasing"
            );
            prev_mel = mel;
        }
    }

    // ============================================================
    // UNIT TESTS: Filterbank creation
    // ============================================================

    #[test]
    fn test_mel_filterbank_new() {
        let mel = MelFilterbank::new(80, 400, 16000);
        assert_eq!(mel.n_mels(), 80);
        assert_eq!(mel.n_fft(), 400);
        assert_eq!(mel.sample_rate(), 16000);
        assert_eq!(mel.n_freqs(), 201); // 400/2 + 1
    }

    #[test]
    fn test_mel_filterbank_filters_shape() {
        let mel = MelFilterbank::new(80, 400, 16000);
        assert_eq!(mel.filters.len(), 80 * 201);
    }

    #[test]
    fn test_mel_filterbank_filters_nonnegative() {
        let mel = MelFilterbank::new(80, 400, 16000);
        for &f in &mel.filters {
            assert!(f >= 0.0, "Filter values should be non-negative");
        }
    }

    #[test]
    fn test_mel_filterbank_filters_bounded() {
        let mel = MelFilterbank::new(80, 400, 16000);
        for &f in &mel.filters {
            assert!(f <= 1.0, "Filter values should be at most 1.0");
        }
    }

    #[test]
    fn test_mel_filterbank_window_size() {
        let mel = MelFilterbank::new(80, 400, 16000);
        assert_eq!(mel.window.len(), 400);
    }

    #[test]
    fn test_hann_window_endpoints() {
        let window = MelFilterbank::hann_window(100);
        // Hann window should be close to 0 at endpoints
        assert!(window[0] < 0.01, "Hann window should start near 0");
        assert!(
            window[99] < 0.01,
            "Hann window should end near 0, got {}",
            window[99]
        );
    }

    #[test]
    fn test_hann_window_peak() {
        let window = MelFilterbank::hann_window(100);
        // Hann window should peak in the middle
        let mid = window[50];
        assert!(
            mid > 0.9,
            "Hann window should peak near 1.0 in the middle, got {mid}"
        );
    }

    // ============================================================
    // UNIT TESTS: Spectrogram computation
    // ============================================================

    #[test]
    fn test_mel_compute_empty() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let result = mel.compute(&[], 160);
        assert!(result.is_ok());
        assert!(result.map_or(false, |v| v.is_empty()));
    }

    #[test]
    fn test_mel_compute_short_audio() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let audio = vec![0.0; 100]; // Too short for even one frame
        let result = mel.compute(&audio, 160);
        assert!(result.is_ok());
        assert!(result.map_or(false, |v| v.is_empty()));
    }

    #[test]
    fn test_mel_compute_exact_one_frame() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let audio = vec![0.0; 400]; // Exactly one FFT window
        let result = mel.compute(&audio, 160).expect("compute should succeed");
        // Should have exactly 1 frame
        assert_eq!(result.len(), 80 * 1);
    }

    #[test]
    fn test_mel_compute_multiple_frames() {
        let mel = MelFilterbank::new(80, 400, 16000);
        // 16000 samples = 1 second at 16kHz
        // With hop_length=160, we get (16000 - 400) / 160 + 1 = 98 frames
        let audio = vec![0.0; 16000];
        let result = mel.compute(&audio, 160).expect("compute should succeed");
        let n_frames = result.len() / 80;
        assert_eq!(n_frames, 98);
    }

    #[test]
    fn test_mel_compute_sine_wave() {
        let mel = MelFilterbank::new(80, 400, 16000);

        // Generate 1 second of 440 Hz sine wave
        let sample_rate = 16000.0;
        let freq = 440.0;
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let result = mel.compute(&audio, 160).expect("compute should succeed");

        // Should have reasonable energy (not all zeros or infinities)
        let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);

        assert!(max_val.is_finite(), "Max should be finite");
        assert!(min_val.is_finite(), "Min should be finite");
        assert!(max_val > min_val, "Should have some variation in output");
    }

    #[test]
    fn test_mel_compute_zero_hop_length() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let audio = vec![0.0; 1600];
        let result = mel.compute(&audio, 0);
        assert!(result.is_err());
    }

    // ============================================================
    // UNIT TESTS: Normalization
    // ============================================================

    #[test]
    fn test_normalize_empty() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let mut data: Vec<f32> = vec![];
        mel.normalize(&mut data); // Should not panic
        assert!(data.is_empty());
    }

    #[test]
    fn test_normalize_single_value() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let mut data = vec![5.0];
        mel.normalize(&mut data);
        // Single value normalized should be 0 (x - mean = 0)
        assert!((data[0]).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_mean_zero() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        mel.normalize(&mut data);

        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 1e-5, "Mean after normalization should be ~0");
    }

    #[test]
    fn test_normalize_std_one() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        mel.normalize(&mut data);

        let variance: f32 = data.iter().map(|&x| x.powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();
        assert!(
            (std - 1.0).abs() < 1e-5,
            "Std after normalization should be ~1, got {std}"
        );
    }

    // ============================================================
    // PROPERTY TESTS (with proptest)
    // ============================================================

    #[test]
    fn test_filterbank_energy_conservation() {
        // Each frequency bin should be covered by at least one filter
        // (except possibly edge bins)
        let mel = MelFilterbank::new(80, 400, 16000);
        let n_freqs = mel.n_freqs();

        for freq_idx in 10..n_freqs - 10 {
            // Skip edge bins
            let total_weight: f32 = (0..80)
                .map(|mel_idx| mel.filters[mel_idx * n_freqs + freq_idx])
                .sum();
            assert!(
                total_weight > 0.0,
                "Frequency bin {freq_idx} should be covered by filters"
            );
        }
    }

    #[test]
    fn test_output_shape_consistency() {
        let mel = MelFilterbank::new(80, 400, 16000);

        for audio_len in [400, 800, 1600, 8000, 16000] {
            let audio = vec![0.0; audio_len];
            let result = mel.compute(&audio, 160).expect("compute should succeed");

            // Output should always be a multiple of n_mels
            assert_eq!(
                result.len() % 80,
                0,
                "Output length {} should be multiple of 80",
                result.len()
            );
        }
    }

    #[test]
    fn test_silence_produces_low_energy() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let silence = vec![0.0; 16000];
        let result = mel.compute(&silence, 160).expect("compute should succeed");

        // Silence should produce very low (negative) log mel energies
        for &val in &result {
            assert!(
                val < 0.0,
                "Silence should produce negative log mel values, got {val}"
            );
        }
    }

    // ============================================================
    // UNIT TESTS: SIMD optimized methods
    // ============================================================

    #[test]
    fn test_apply_filterbank_simd_matches_scalar() {
        let mel = MelFilterbank::new(80, 400, 16000);

        // Generate a test power spectrum
        let power_spec: Vec<f32> = (0..mel.n_freqs())
            .map(|i| (i as f32 * 0.01).sin().powi(2))
            .collect();

        let simd_result = mel.apply_filterbank_simd(&power_spec);
        let scalar_result = mel.apply_filterbank_scalar(&power_spec);

        assert_eq!(simd_result.len(), scalar_result.len());
        for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
            assert!(
                (simd_val - scalar_val).abs() < 1e-5,
                "SIMD and scalar should match: {} vs {}",
                simd_val,
                scalar_val
            );
        }
    }

    #[test]
    fn test_apply_filterbank_simd_output_shape() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let power_spec = vec![1.0; mel.n_freqs()];

        let result = mel.apply_filterbank_simd(&power_spec);
        assert_eq!(result.len(), 80);
    }

    #[test]
    fn test_apply_filterbank_simd_dimension_mismatch() {
        let mel = MelFilterbank::new(80, 400, 16000);

        // Wrong size power spectrum - should fall back to scalar
        let power_spec = vec![1.0; 100]; // Wrong size
        let result = mel.apply_filterbank_simd(&power_spec);

        // Should still work via scalar fallback
        assert_eq!(result.len(), 80);
    }

    #[test]
    fn test_normalize_simd_matches_scalar() {
        let mel = MelFilterbank::new(80, 400, 16000);

        // Generate test data
        let mut simd_data: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut scalar_data = simd_data.clone();

        mel.normalize_simd(&mut simd_data);
        mel.normalize(&mut scalar_data);

        for (simd_val, scalar_val) in simd_data.iter().zip(scalar_data.iter()) {
            assert!(
                (simd_val - scalar_val).abs() < 1e-5,
                "SIMD and scalar normalization should match: {} vs {}",
                simd_val,
                scalar_val
            );
        }
    }

    #[test]
    fn test_normalize_simd_empty() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let mut data: Vec<f32> = vec![];
        mel.normalize_simd(&mut data); // Should not panic
        assert!(data.is_empty());
    }

    #[test]
    fn test_normalize_simd_mean_zero() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        mel.normalize_simd(&mut data);

        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(
            mean.abs() < 1e-5,
            "Mean after SIMD normalization should be ~0, got {mean}"
        );
    }

    #[test]
    fn test_normalize_simd_std_one() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        mel.normalize_simd(&mut data);

        let variance: f32 = data.iter().map(|&x| x.powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();
        assert!(
            (std - 1.0).abs() < 1e-5,
            "Std after SIMD normalization should be ~1, got {std}"
        );
    }

    #[test]
    fn test_compute_simd_matches_compute() {
        let mel = MelFilterbank::new(80, 400, 16000);

        // Generate test audio (sine wave)
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let compute_result = mel.compute(&audio, 160).expect("compute should succeed");
        let simd_result = mel
            .compute_simd(&audio, 160)
            .expect("compute_simd should succeed");

        assert_eq!(compute_result.len(), simd_result.len());
        for (compute_val, simd_val) in compute_result.iter().zip(simd_result.iter()) {
            assert!(
                (compute_val - simd_val).abs() < 1e-4,
                "compute and compute_simd should match: {} vs {}",
                compute_val,
                simd_val
            );
        }
    }

    #[test]
    fn test_compute_simd_empty() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let result = mel.compute_simd(&[], 160);
        assert!(result.is_ok());
        assert!(result.map_or(false, |v| v.is_empty()));
    }

    #[test]
    fn test_compute_simd_zero_hop() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let audio = vec![0.0; 1600];
        let result = mel.compute_simd(&audio, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_simd_short_audio() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let audio = vec![0.0; 100]; // Too short
        let result = mel.compute_simd(&audio, 160);
        assert!(result.is_ok());
        assert!(result.map_or(false, |v| v.is_empty()));
    }

    #[test]
    fn test_compute_simd_one_frame() {
        let mel = MelFilterbank::new(80, 400, 16000);
        let audio = vec![0.0; 400];
        let result = mel.compute_simd(&audio, 160).expect("should succeed");
        assert_eq!(result.len(), 80);
    }

    // ============================================================
    // ACCURACY TESTS: Reference validation (WAPR-013)
    // ============================================================
    // These tests validate mel spectrogram computation against
    // known reference values and properties from Whisper's original
    // implementation.

    #[test]
    fn test_mel_scale_matches_reference() {
        // Reference values from librosa/Whisper mel scale
        // mel(f) = 2595 * log10(1 + f/700)
        let test_cases = [
            (0.0, 0.0),
            (700.0, 781.9),   // mel(700) ≈ 782
            (1000.0, 999.99), // mel(1000) ≈ 1000 (by design)
            (4000.0, 2146.1), // mel(4000) ≈ 2146
            (8000.0, 2840.0), // mel(8000) ≈ 2840
        ];

        for (hz, expected_mel) in test_cases {
            let computed = MelFilterbank::hz_to_mel(hz);
            let error = (computed - expected_mel).abs();
            assert!(
                error < 10.0,
                "Mel conversion error for {hz} Hz: expected ~{expected_mel}, got {computed}"
            );
        }
    }

    #[test]
    fn test_filterbank_triangular_shape() {
        // Verify mel filters have proper triangular shape
        let mel = MelFilterbank::new(40, 512, 16000);
        let n_freqs = mel.n_freqs();

        for mel_idx in 1..39 {
            // Check filter peaks in middle (roughly)
            let filter_row = &mel.filters[mel_idx * n_freqs..(mel_idx + 1) * n_freqs];
            let max_val = filter_row.iter().fold(0.0_f32, |a, &b| a.max(b));

            // Each filter should have a peak
            assert!(max_val > 0.0, "Filter {mel_idx} should have positive peak");

            // Peak should not exceed 1.0
            assert!(
                max_val <= 1.0,
                "Filter {mel_idx} peak {} exceeds 1.0",
                max_val
            );
        }
    }

    #[test]
    fn test_whisper_standard_params() {
        // Whisper uses: n_mels=80, n_fft=400, sample_rate=16000, hop=160
        let mel = MelFilterbank::new(80, 400, 16000);

        // Verify standard parameters
        assert_eq!(mel.n_mels(), 80);
        assert_eq!(mel.n_fft(), 400);
        assert_eq!(mel.sample_rate(), 16000);
        assert_eq!(mel.n_freqs(), 201); // n_fft/2 + 1

        // Standard Whisper: 10ms frame shift (160 samples at 16kHz)
        let hop_length = 160;
        let audio_1s = vec![0.0; 16000];
        let result = mel.compute(&audio_1s, hop_length).expect("should work");

        // 1 second -> (16000 - 400) / 160 + 1 = 98 frames
        let n_frames = result.len() / 80;
        assert_eq!(n_frames, 98, "1s audio should produce 98 frames");
    }

    #[test]
    fn test_tone_produces_localized_energy() {
        // A pure tone should produce energy in specific mel bands
        let mel = MelFilterbank::new(80, 400, 16000);

        // Generate 440 Hz sine wave (A4 note)
        let sample_rate = 16000.0;
        let freq = 440.0;
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let result = mel.compute(&audio, 160).expect("compute should succeed");

        // Average energy per mel band across all frames
        // Data layout is [frame][mel], so index as: result[frame * 80 + mel_idx]
        let n_frames = result.len() / 80;
        let mut avg_energy = vec![0.0_f32; 80];
        for frame in 0..n_frames {
            for mel_idx in 0..80 {
                avg_energy[mel_idx] += result[frame * 80 + mel_idx];
            }
        }
        for e in &mut avg_energy {
            *e /= n_frames as f32;
        }

        // Find the mel band with maximum energy
        let max_mel = avg_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("avg_energy should not be empty");

        // 440 Hz corresponds to mel ~550, which maps to around mel bin 15-25
        // for 80 mel bands spanning 0-8kHz
        assert!(
            max_mel >= 10 && max_mel <= 35,
            "440 Hz should produce peak in lower mel bands, got bin {max_mel}"
        );
    }

    #[test]
    fn test_high_tone_energy_location() {
        // A high frequency tone should produce energy in higher mel bands
        let mel = MelFilterbank::new(80, 400, 16000);

        // Generate 4000 Hz sine wave
        let sample_rate = 16000.0;
        let freq = 4000.0;
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let result = mel.compute(&audio, 160).expect("compute should succeed");

        // Average energy per mel band
        // Data layout is [frame][mel], so index as: result[frame * 80 + mel_idx]
        let n_frames = result.len() / 80;
        let mut avg_energy = vec![0.0_f32; 80];
        for frame in 0..n_frames {
            for mel_idx in 0..80 {
                avg_energy[mel_idx] += result[frame * 80 + mel_idx];
            }
        }
        for e in &mut avg_energy {
            *e /= n_frames as f32;
        }

        let max_mel = avg_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("avg_energy should not be empty");

        // 4000 Hz corresponds to mel ~2146, in upper half of spectrum
        assert!(
            max_mel >= 40,
            "4000 Hz should produce peak in higher mel bands, got bin {max_mel}"
        );
    }

    #[test]
    fn test_energy_increases_with_amplitude() {
        let mel = MelFilterbank::new(80, 400, 16000);

        let freq = 1000.0;
        let sample_rate = 16000.0;

        // Generate tones with different amplitudes
        let quiet: Vec<f32> = (0..16000)
            .map(|i| 0.1 * (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();
        let loud: Vec<f32> = (0..16000)
            .map(|i| 1.0 * (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let result_quiet = mel.compute(&quiet, 160).expect("should work");
        let result_loud = mel.compute(&loud, 160).expect("should work");

        // Compute total energy
        let energy_quiet: f32 = result_quiet.iter().sum();
        let energy_loud: f32 = result_loud.iter().sum();

        // Loud signal should have more energy (in log domain, larger values)
        assert!(
            energy_loud > energy_quiet,
            "Louder signal should have higher energy: {} vs {}",
            energy_loud,
            energy_quiet
        );
    }

    #[test]
    fn test_output_range_is_reasonable() {
        // Log mel values should be in a reasonable range
        let mel = MelFilterbank::new(80, 400, 16000);

        // Generate typical speech-like signal (mix of frequencies)
        let audio: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                0.3 * (2.0 * PI * 200.0 * t).sin()
                    + 0.2 * (2.0 * PI * 500.0 * t).sin()
                    + 0.1 * (2.0 * PI * 1000.0 * t).sin()
            })
            .collect();

        let result = mel.compute(&audio, 160).expect("should work");

        // Check that values are in reasonable log domain range
        let min_val = result.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = result.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Log mel values typically range from about -15 to +5
        assert!(min_val > -30.0, "Min log mel value {} is too low", min_val);
        assert!(max_val < 20.0, "Max log mel value {} is too high", max_val);
        assert!(max_val > min_val, "Should have variation in output");
    }

    #[test]
    fn test_numerical_precision_consistency() {
        // Running the same computation multiple times should give identical results
        let mel = MelFilterbank::new(80, 400, 16000);

        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

        let result1 = mel.compute(&audio, 160).expect("should work");
        let result2 = mel.compute(&audio, 160).expect("should work");

        for (v1, v2) in result1.iter().zip(result2.iter()) {
            assert!(
                (v1 - v2).abs() < 1e-10,
                "Results should be identical: {} vs {}",
                v1,
                v2
            );
        }
    }

    #[test]
    fn test_simd_accuracy_matches_scalar() {
        // SIMD implementation should match scalar within floating point tolerance
        let mel = MelFilterbank::new(80, 400, 16000);

        let audio: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 880.0 * t).sin()
            })
            .collect();

        let scalar_result = mel.compute(&audio, 160).expect("scalar should work");
        let simd_result = mel.compute_simd(&audio, 160).expect("simd should work");

        assert_eq!(scalar_result.len(), simd_result.len());

        let max_diff = scalar_result
            .iter()
            .zip(simd_result.iter())
            .map(|(s, si)| (s - si).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "SIMD and scalar results differ by {}, should match closely",
            max_diff
        );
    }

    // =========================================================================
    // Property-Based Tests (WAPR-QA-002)
    // =========================================================================

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn property_mel_scale_monotonic(freq in 0.0f32..20000.0) {
                // Mel scale should be monotonically increasing
                let mel1 = MelFilterbank::hz_to_mel(freq);
                let mel2 = MelFilterbank::hz_to_mel(freq + 1.0);
                prop_assert!(mel2 >= mel1, "mel scale should be monotonic");
            }

            #[test]
            fn property_mel_hz_roundtrip(freq in 20.0f32..15000.0) {
                // hz -> mel -> hz should be close to original
                let mel = MelFilterbank::hz_to_mel(freq);
                let back = MelFilterbank::mel_to_hz(mel);
                let error = (freq - back).abs() / freq.max(1.0);
                prop_assert!(error < 0.01, "roundtrip error {} too large for freq {}", error, freq);
            }

            #[test]
            fn property_filterbank_nonnegative(n_mels in 20usize..128, n_fft in 256usize..1024) {
                let sample_rate = 16000;
                let mel = MelFilterbank::new(n_mels, n_fft, sample_rate);
                for val in &mel.filters {
                    prop_assert!(*val >= 0.0, "filterbank values must be non-negative");
                }
            }

            #[test]
            fn property_normalize_mean_zero(len in 10usize..1000) {
                // Create a mel filterbank and some test data
                let mel = MelFilterbank::new(80, 400, 16000);
                let mut data: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();

                // Normalize in place
                mel.normalize(&mut data);

                if data.len() > 1 {
                    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                    prop_assert!(mean.abs() < 1e-5, "normalized mean {} should be ~0", mean);
                }
            }

            #[test]
            fn property_simd_matches_scalar(audio_len in 1600usize..8000) {
                let mel = MelFilterbank::new(80, 400, 16000);
                let audio: Vec<f32> = (0..audio_len)
                    .map(|i| (i as f32 * 0.01).sin() * 0.5)
                    .collect();

                if let (Ok(scalar), Ok(simd)) = (mel.compute(&audio, 160), mel.compute_simd(&audio, 160)) {
                    prop_assert_eq!(scalar.len(), simd.len(), "output lengths must match");
                    for (s, si) in scalar.iter().zip(simd.iter()) {
                        prop_assert!((s - si).abs() < 1e-3, "scalar {} vs simd {} mismatch", s, si);
                    }
                }
            }
        }
    }
}

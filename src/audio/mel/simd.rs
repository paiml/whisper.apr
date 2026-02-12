//! SIMD-optimized mel filterbank operations
//!
//! Provides accelerated implementations of filterbank application and normalization.

use super::MelFilterbank;
use crate::error::{WhisperError, WhisperResult};
use rustfft::{num_complex::Complex, FftPlanner};

impl MelFilterbank {
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
    pub(crate) fn apply_filterbank_scalar(&self, power_spec: &[f32]) -> Vec<f32> {
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

        if hop_length == 0 {
            return Err(WhisperError::Audio("hop_length must be positive".into()));
        }

        if audio.is_empty() {
            return Ok(Vec::new());
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
                    let sample = padded_audio.get(start + i).copied().unwrap_or(0.0);
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

            // Apply mel filterbank using SIMD
            let mel_energies = self.apply_filterbank_simd(&power_spec);

            // Apply log compression and store
            for (mel_idx, &energy) in mel_energies.iter().enumerate() {
                let log_mel = (energy.max(1e-10)).log10();
                mel_spec[frame_idx * self.n_mels + mel_idx] = log_mel;
            }
        }

        // Apply Whisper normalization: clamp to max-8, then shift and scale
        let max_val = mel_spec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let floor = max_val - 8.0;
        for x in &mut mel_spec {
            *x = ((*x).max(floor) + 4.0) / 4.0;
        }

        Ok(mel_spec)
    }
}

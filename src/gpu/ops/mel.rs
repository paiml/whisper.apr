//! GPU mel filterbank computation (WAPR-WEBGPU-001)
//!
//! Provides GPU-accelerated mel spectrogram computation using WebGPU compute shaders.
//! This is the first stage of the Whisper pipeline and a key bottleneck.
//!
//! # Pipeline
//!
//! ```text
//! Audio → FFT → Power Spectrum → Mel Filterbank → Log Mel → Normalize
//! ```

use crate::gpu::error::{GpuError, GpuResult};

/// Mel filterbank configuration
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// Number of mel filterbank bins (typically 80 or 128)
    pub n_mels: u32,
    /// FFT size (typically 400 for 16kHz audio with 25ms window)
    pub n_fft: u32,
    /// Hop length in samples (typically 160 for 10ms)
    pub hop_length: u32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Minimum frequency for mel scale
    pub f_min: f32,
    /// Maximum frequency for mel scale
    pub f_max: f32,
    /// Whether to use log scaling
    pub log_scale: bool,
    /// Floor value for log to avoid log(0)
    pub log_floor: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self::whisper()
    }
}

impl MelConfig {
    /// Whisper model configuration (80 mels, 400 FFT, 160 hop)
    #[must_use]
    pub fn whisper() -> Self {
        Self {
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            sample_rate: 16000,
            f_min: 0.0,
            f_max: 8000.0,
            log_scale: true,
            log_floor: 1e-10,
        }
    }

    /// Whisper large model configuration (128 mels)
    #[must_use]
    pub fn whisper_large() -> Self {
        Self {
            n_mels: 128,
            ..Self::whisper()
        }
    }

    /// Number of frequency bins (n_fft / 2 + 1)
    #[must_use]
    pub fn n_freqs(&self) -> u32 {
        self.n_fft / 2 + 1
    }

    /// Calculate number of frames for given audio length
    #[must_use]
    pub fn n_frames(&self, audio_samples: usize) -> u32 {
        ((audio_samples as u32).saturating_sub(self.n_fft)) / self.hop_length + 1
    }

    /// Validate configuration
    pub fn validate(&self) -> GpuResult<()> {
        if self.n_mels == 0 {
            return Err(GpuError::compute("n_mels cannot be zero"));
        }
        if self.n_fft == 0 {
            return Err(GpuError::compute("n_fft cannot be zero"));
        }
        if self.hop_length == 0 {
            return Err(GpuError::compute("hop_length cannot be zero"));
        }
        if self.f_max <= self.f_min {
            return Err(GpuError::compute("f_max must be greater than f_min"));
        }
        Ok(())
    }
}

/// GPU mel filterbank operation
#[derive(Debug)]
pub struct GpuMelFilterbank {
    /// Configuration
    config: MelConfig,
    /// Precomputed filterbank weights (n_mels x n_freqs)
    filterbank: Vec<f32>,
}

impl GpuMelFilterbank {
    /// Create a new mel filterbank operation
    pub fn new(config: MelConfig) -> GpuResult<Self> {
        config.validate()?;
        let filterbank = Self::compute_filterbank(&config);
        Ok(Self { config, filterbank })
    }

    /// Create with Whisper default configuration
    pub fn whisper() -> GpuResult<Self> {
        Self::new(MelConfig::whisper())
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    /// Get precomputed filterbank weights
    #[must_use]
    pub fn filterbank(&self) -> &[f32] {
        &self.filterbank
    }

    /// Compute mel filterbank weights
    fn compute_filterbank(config: &MelConfig) -> Vec<f32> {
        let n_mels = config.n_mels as usize;
        let n_freqs = config.n_freqs() as usize;

        let mut filterbank = vec![0.0f32; n_mels * n_freqs];

        // Mel scale conversion
        let mel_min = Self::hz_to_mel(config.f_min);
        let mel_max = Self::hz_to_mel(config.f_max);

        // Compute FFT bin indices directly: mel scale → Hz → FFT bin
        let fft_bins: Vec<f32> = (0..=n_mels + 1)
            .map(|i| {
                let mel = mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32);
                let hz = Self::mel_to_hz(mel);
                (config.n_fft as f32 + 1.0) * hz / (config.sample_rate as f32)
            })
            .collect();

        // Create triangular filters
        for m in 0..n_mels {
            let f_start = fft_bins[m];
            let f_center = fft_bins[m + 1];
            let f_end = fft_bins[m + 2];

            for k in 0..n_freqs {
                let k_f = k as f32;
                let weight = if k_f < f_start || k_f > f_end {
                    0.0
                } else if k_f <= f_center {
                    (k_f - f_start) / (f_center - f_start + 1e-10)
                } else {
                    (f_end - k_f) / (f_end - f_center + 1e-10)
                };
                filterbank[m * n_freqs + k] = weight;
            }
        }

        filterbank
    }

    /// Convert Hz to mel scale
    #[must_use]
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).ln() / std::f32::consts::LN_10
    }

    /// Convert mel to Hz scale
    #[must_use]
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Generate WGSL shader for mel filterbank computation
    ///
    /// This shader applies the mel filterbank to a power spectrum.
    /// Input: power spectrum (n_frames x n_freqs)
    /// Output: mel spectrogram (n_frames x n_mels)
    #[must_use]
    pub fn generate_shader(&self) -> String {
        let n_mels = self.config.n_mels;
        let n_freqs = self.config.n_freqs();
        let log_floor = self.config.log_floor;

        format!(
            r#"// Mel filterbank shader (WAPR-WEBGPU-001)
// Applies mel filterbank to power spectrum
// Config: {n_mels} mels, {n_freqs} freq bins

struct MelParams {{
    n_frames: u32,
    n_freqs: u32,
    n_mels: u32,
    log_scale: u32,
    log_floor: f32,
    _padding: vec3<f32>,
}}

@group(0) @binding(0) var<uniform> params: MelParams;
@group(0) @binding(1) var<storage, read> power_spectrum: array<f32>;
@group(0) @binding(2) var<storage, read> filterbank: array<f32>;
@group(0) @binding(3) var<storage, read_write> mel_output: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {{
    let frame = global_id.y;
    let mel_bin = global_id.x;

    if (frame >= params.n_frames || mel_bin >= params.n_mels) {{
        return;
    }}

    // Apply filterbank: mel[frame, mel_bin] = sum(power[frame, :] * filterbank[mel_bin, :])
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.n_freqs; k = k + 1u) {{
        let power_val = power_spectrum[frame * params.n_freqs + k];
        let filter_val = filterbank[mel_bin * params.n_freqs + k];
        sum = sum + power_val * filter_val;
    }}

    // Apply log scaling if enabled
    var result = sum;
    if (params.log_scale != 0u) {{
        result = log(max(sum, {log_floor}));
    }}

    mel_output[frame * params.n_mels + mel_bin] = result;
}}
"#,
            n_mels = n_mels,
            n_freqs = n_freqs,
            log_floor = log_floor,
        )
    }

    /// Generate WGSL shader for full mel spectrogram pipeline
    ///
    /// This includes: windowing, FFT (simplified), power spectrum, mel filterbank, log
    /// Note: For production, FFT should use a proper GPU FFT library.
    #[must_use]
    pub fn generate_full_pipeline_shader(&self) -> String {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let n_mels = self.config.n_mels;
        #[allow(unused_variables)]
        let n_freqs = self.config.n_freqs();

        format!(
            r#"// Full mel spectrogram pipeline (WAPR-WEBGPU-001)
// Pipeline: Audio → Window → FFT → Power → Mel → Log
// Config: FFT={n_fft}, hop={hop_length}, mels={n_mels}

struct PipelineParams {{
    n_samples: u32,
    n_fft: u32,
    hop_length: u32,
    n_frames: u32,
    n_freqs: u32,
    n_mels: u32,
    log_floor: f32,
    _padding: u32,
}}

@group(0) @binding(0) var<uniform> params: PipelineParams;
@group(0) @binding(1) var<storage, read> audio: array<f32>;
@group(0) @binding(2) var<storage, read> hann_window: array<f32>;
@group(0) @binding(3) var<storage, read> filterbank: array<f32>;
@group(0) @binding(4) var<storage, read_write> mel_output: array<f32>;

// Workgroup shared memory for FFT
var<workgroup> fft_real: array<f32, {n_fft}>;
var<workgroup> fft_imag: array<f32, {n_fft}>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {{
    let frame = workgroup_id.x;
    let thread_idx = local_id.x;

    if (frame >= params.n_frames) {{
        return;
    }}

    let frame_start = frame * params.hop_length;

    // Step 1: Load and window the audio frame
    if (thread_idx < params.n_fft) {{
        let sample_idx = frame_start + thread_idx;
        var sample: f32 = 0.0;
        if (sample_idx < params.n_samples) {{
            sample = audio[sample_idx];
        }}
        fft_real[thread_idx] = sample * hann_window[thread_idx];
        fft_imag[thread_idx] = 0.0;
    }}

    workgroupBarrier();

    // Step 2: Simplified DFT (for demonstration - real impl would use FFT)
    // Each thread computes one frequency bin
    if (thread_idx < params.n_freqs) {{
        let k = thread_idx;
        var real_sum: f32 = 0.0;
        var imag_sum: f32 = 0.0;

        let angle_base = -2.0 * 3.14159265359 * f32(k) / f32(params.n_fft);

        for (var n: u32 = 0u; n < params.n_fft; n = n + 1u) {{
            let angle = angle_base * f32(n);
            real_sum = real_sum + fft_real[n] * cos(angle);
            imag_sum = imag_sum + fft_real[n] * sin(angle);
        }}

        // Power spectrum: |X|^2
        let power = real_sum * real_sum + imag_sum * imag_sum;

        // Store temporarily in fft_real (reuse shared memory)
        fft_real[k] = power;
    }}

    workgroupBarrier();

    // Step 3: Apply mel filterbank
    // Each thread handles one mel bin
    if (thread_idx < params.n_mels) {{
        let mel_bin = thread_idx;
        var mel_sum: f32 = 0.0;

        for (var k: u32 = 0u; k < params.n_freqs; k = k + 1u) {{
            let power_val = fft_real[k];
            let filter_val = filterbank[mel_bin * params.n_freqs + k];
            mel_sum = mel_sum + power_val * filter_val;
        }}

        // Log scaling
        let log_mel = log(max(mel_sum, params.log_floor));

        // Write output
        mel_output[frame * params.n_mels + mel_bin] = log_mel;
    }}
}}
"#,
            n_fft = n_fft,
            hop_length = hop_length,
            n_mels = n_mels,
        )
    }

    /// Calculate workgroups for dispatch
    #[must_use]
    pub fn workgroups(&self, n_frames: u32) -> (u32, u32, u32) {
        let x = self.config.n_mels.div_ceil(16);
        let y = n_frames.div_ceil(16);
        (x, y, 1)
    }

    /// Calculate memory requirement for mel output
    #[must_use]
    pub fn output_size(&self, n_frames: u32) -> usize {
        (self.config.n_mels as usize) * (n_frames as usize)
    }

    /// Calculate memory requirement in bytes
    #[must_use]
    pub fn output_bytes(&self, n_frames: u32) -> usize {
        self.output_size(n_frames) * 4
    }
}

/// Precomputed Hann window for Whisper
pub fn hann_window(n_fft: u32) -> Vec<f32> {
    (0..n_fft)
        .map(|i| {
            let x = std::f32::consts::PI * (i as f32) / (n_fft as f32 - 1.0);
            0.5 * (1.0 - (2.0 * x).cos())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_config_default() {
        let config = MelConfig::default();
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.hop_length, 160);
    }

    #[test]
    fn test_mel_config_whisper() {
        let config = MelConfig::whisper();
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_freqs(), 201); // 400/2 + 1
    }

    #[test]
    fn test_mel_config_validate() {
        assert!(MelConfig::whisper().validate().is_ok());

        let bad = MelConfig {
            n_mels: 0,
            ..MelConfig::whisper()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_mel_config_n_frames() {
        let config = MelConfig::whisper();
        // 16000 samples = 1 second, should give ~100 frames at 10ms hop
        assert_eq!(config.n_frames(16000), 98); // (16000 - 400) / 160 + 1
    }

    #[test]
    fn test_hz_mel_conversion() {
        // Test round-trip conversion
        let hz = 1000.0;
        let mel = GpuMelFilterbank::hz_to_mel(hz);
        let hz_back = GpuMelFilterbank::mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 0.01);
    }

    #[test]
    fn test_filterbank_creation() {
        let mel = GpuMelFilterbank::whisper().expect("create mel");
        let fb = mel.filterbank();

        // Should have n_mels * n_freqs weights
        assert_eq!(fb.len(), 80 * 201);

        // Weights should be non-negative
        assert!(fb.iter().all(|&w| w >= 0.0));

        // Each mel bin should have some non-zero weights
        for m in 0..80 {
            let row_sum: f32 = (0..201).map(|k| fb[m * 201 + k]).sum();
            assert!(row_sum > 0.0, "Mel bin {m} has zero weights");
        }
    }

    #[test]
    fn test_generate_shader() {
        let mel = GpuMelFilterbank::whisper().expect("create mel");
        let shader = mel.generate_shader();

        assert!(shader.contains("@compute"));
        assert!(shader.contains("@workgroup_size(16, 16, 1)"));
        assert!(shader.contains("mel_output"));
        assert!(shader.contains("filterbank"));
    }

    #[test]
    fn test_generate_full_pipeline_shader() {
        let mel = GpuMelFilterbank::whisper().expect("create mel");
        let shader = mel.generate_full_pipeline_shader();

        assert!(shader.contains("@compute"));
        assert!(shader.contains("hann_window"));
        assert!(shader.contains("fft_real"));
        assert!(shader.contains("mel_output"));
    }

    #[test]
    fn test_workgroups() {
        let mel = GpuMelFilterbank::whisper().expect("create mel");
        let (x, y, z) = mel.workgroups(100);

        assert_eq!(x, 5); // 80 mels / 16 = 5
        assert_eq!(y, 7); // 100 frames / 16 = 7 (rounded up)
        assert_eq!(z, 1);
    }

    #[test]
    fn test_output_size() {
        let mel = GpuMelFilterbank::whisper().expect("create mel");
        assert_eq!(mel.output_size(100), 8000); // 80 * 100
        assert_eq!(mel.output_bytes(100), 32000); // 8000 * 4
    }

    #[test]
    fn test_hann_window() {
        let window = hann_window(400);
        assert_eq!(window.len(), 400);

        // Window should start and end near zero
        assert!(window[0] < 0.01);
        assert!(window[399] < 0.01);

        // Window should peak in the middle
        assert!(window[200] > 0.99);
    }
}

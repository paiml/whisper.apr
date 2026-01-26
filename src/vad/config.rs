//! VAD configuration types
//!
//! Configuration structures for voice activity detection.

/// VAD configuration parameters
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Sample rate of input audio (default: 16000 Hz)
    pub sample_rate: u32,
    /// Frame size in samples (default: 480 = 30ms at 16kHz)
    pub frame_size: usize,
    /// Energy threshold relative to noise floor (default: 2.0)
    pub energy_threshold: f32,
    /// Zero-crossing rate threshold (default: 0.3)
    pub zcr_threshold: f32,
    /// Minimum speech duration in frames (default: 3)
    pub min_speech_frames: usize,
    /// Minimum silence duration to end speech (default: 10 frames)
    pub min_silence_frames: usize,
    /// Smoothing factor for adaptive thresholds (default: 0.95)
    pub smoothing: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_size: 480, // 30ms at 16kHz
            energy_threshold: 2.0,
            zcr_threshold: 0.3,
            min_speech_frames: 3,
            min_silence_frames: 10,
            smoothing: 0.95,
        }
    }
}

impl VadConfig {
    /// Create a new VAD configuration with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create config for low-latency streaming (10ms frames)
    #[must_use]
    pub fn low_latency() -> Self {
        Self {
            frame_size: 160, // 10ms at 16kHz
            min_speech_frames: 5,
            min_silence_frames: 15,
            ..Self::default()
        }
    }

    /// Create config for high-accuracy detection (50ms frames)
    #[must_use]
    pub fn high_accuracy() -> Self {
        Self {
            frame_size: 800, // 50ms at 16kHz
            min_speech_frames: 2,
            min_silence_frames: 6,
            ..Self::default()
        }
    }

    /// Set energy threshold (builder pattern)
    #[must_use]
    pub fn with_energy_threshold(mut self, threshold: f32) -> Self {
        self.energy_threshold = threshold;
        self
    }

    /// Set ZCR threshold (builder pattern)
    #[must_use]
    pub fn with_zcr_threshold(mut self, threshold: f32) -> Self {
        self.zcr_threshold = threshold;
        self
    }

    /// Set minimum speech frames (builder pattern)
    #[must_use]
    pub fn with_min_speech_frames(mut self, frames: usize) -> Self {
        self.min_speech_frames = frames;
        self
    }

    /// Set minimum silence frames (builder pattern)
    #[must_use]
    pub fn with_min_silence_frames(mut self, frames: usize) -> Self {
        self.min_silence_frames = frames;
        self
    }

    /// Set smoothing factor (builder pattern)
    #[must_use]
    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Set sample rate (builder pattern)
    #[must_use]
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set frame size (builder pattern)
    #[must_use]
    pub fn with_frame_size(mut self, frame_size: usize) -> Self {
        self.frame_size = frame_size;
        self
    }

    /// Get frame duration in seconds
    #[must_use]
    pub fn frame_duration(&self) -> f32 {
        self.frame_size as f32 / self.sample_rate as f32
    }

    /// Get frame duration in milliseconds
    #[must_use]
    pub fn frame_duration_ms(&self) -> f32 {
        self.frame_duration() * 1000.0
    }
}

/// Silence detection configuration (WAPR-092)
#[derive(Debug, Clone)]
pub struct SilenceConfig {
    /// Minimum silence duration in seconds to consider as a break (default: 0.3)
    pub min_silence_duration: f32,
    /// Maximum silence duration before forcing segment end (default: 2.0)
    pub max_silence_duration: f32,
    /// Energy threshold below which audio is considered silence (default: 0.001)
    pub silence_threshold: f32,
    /// Whether to use adaptive silence detection (default: true)
    pub adaptive: bool,
    /// Adaptation rate for noise floor (default: 0.01)
    pub adaptation_rate: f32,
}

impl Default for SilenceConfig {
    fn default() -> Self {
        Self {
            min_silence_duration: 0.3,
            max_silence_duration: 2.0,
            silence_threshold: 0.001,
            adaptive: true,
            adaptation_rate: 0.01,
        }
    }
}

impl SilenceConfig {
    /// Create a new silence configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum silence duration (builder pattern)
    #[must_use]
    pub fn with_min_silence_duration(mut self, duration: f32) -> Self {
        self.min_silence_duration = duration;
        self
    }

    /// Set maximum silence duration (builder pattern)
    #[must_use]
    pub fn with_max_silence_duration(mut self, duration: f32) -> Self {
        self.max_silence_duration = duration;
        self
    }

    /// Set silence threshold (builder pattern)
    #[must_use]
    pub fn with_silence_threshold(mut self, threshold: f32) -> Self {
        self.silence_threshold = threshold;
        self
    }

    /// Enable/disable adaptive detection (builder pattern)
    #[must_use]
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Set adaptation rate (builder pattern)
    #[must_use]
    pub fn with_adaptation_rate(mut self, rate: f32) -> Self {
        self.adaptation_rate = rate;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_config_default() {
        let config = VadConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.frame_size, 480);
    }

    #[test]
    fn test_vad_config_low_latency() {
        let config = VadConfig::low_latency();
        assert_eq!(config.frame_size, 160);
    }

    #[test]
    fn test_vad_config_high_accuracy() {
        let config = VadConfig::high_accuracy();
        assert_eq!(config.frame_size, 800);
    }

    #[test]
    fn test_vad_config_builder() {
        let config = VadConfig::new()
            .with_energy_threshold(3.0)
            .with_zcr_threshold(0.5)
            .with_min_speech_frames(5)
            .with_min_silence_frames(15)
            .with_smoothing(0.9);
        assert!((config.energy_threshold - 3.0).abs() < f32::EPSILON);
        assert!((config.zcr_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.min_speech_frames, 5);
    }

    #[test]
    fn test_vad_config_frame_duration() {
        let config = VadConfig::default();
        let duration = config.frame_duration();
        assert!((duration - 0.03).abs() < 0.001); // 30ms
    }

    #[test]
    fn test_vad_config_frame_duration_ms() {
        let config = VadConfig::default();
        let duration_ms = config.frame_duration_ms();
        assert!((duration_ms - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_silence_config_default() {
        let config = SilenceConfig::default();
        assert!((config.min_silence_duration - 0.3).abs() < f32::EPSILON);
        assert!(config.adaptive);
    }

    #[test]
    fn test_silence_config_builder() {
        let config = SilenceConfig::new()
            .with_min_silence_duration(0.5)
            .with_max_silence_duration(3.0)
            .with_silence_threshold(0.002)
            .with_adaptive(false)
            .with_adaptation_rate(0.02);
        assert!((config.min_silence_duration - 0.5).abs() < f32::EPSILON);
        assert!(!config.adaptive);
    }
}

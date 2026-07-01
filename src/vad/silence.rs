//! Silence detection module
//!
//! Provides silence detection with adaptive thresholds.

use super::config::SilenceConfig;

/// Detected silence segment (WAPR-092)
#[derive(Debug, Clone)]
pub struct SilenceSegment {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Average energy during silence (noise floor)
    pub noise_floor: f32,
}

impl SilenceSegment {
    /// Duration of the silence in seconds
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Check if this is a long silence (potential utterance boundary)
    #[must_use]
    pub fn is_utterance_boundary(&self, config: &SilenceConfig) -> bool {
        self.duration() >= config.min_silence_duration
    }
}

/// Silence detector with adaptive thresholds (WAPR-092)
#[derive(Debug, Clone)]
pub struct SilenceDetector {
    config: SilenceConfig,
    /// Adaptive noise floor estimate
    noise_floor: f32,
    /// Current silence start time (None if not in silence)
    silence_start: Option<f32>,
    /// Sample rate for time calculations
    sample_rate: u32,
    /// Total samples processed
    samples_processed: usize,
    /// Recent energy history for adaptation
    energy_history: Vec<f32>,
}

impl SilenceDetector {
    /// Create a new silence detector
    #[must_use]
    pub fn new(config: SilenceConfig, sample_rate: u32) -> Self {
        Self {
            config,
            noise_floor: 0.001,
            silence_start: None,
            sample_rate,
            samples_processed: 0,
            energy_history: Vec::with_capacity(100),
        }
    }

    /// Create with default config
    #[must_use]
    pub fn with_sample_rate(sample_rate: u32) -> Self {
        Self::new(SilenceConfig::default(), sample_rate)
    }

    /// Get current noise floor estimate
    #[must_use]
    pub fn noise_floor(&self) -> f32 {
        self.noise_floor
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> &SilenceConfig {
        &self.config
    }

    /// Check if currently in silence
    #[must_use]
    pub fn is_silence(&self) -> bool {
        self.silence_start.is_some()
    }

    /// Get current silence duration (0 if not in silence)
    #[must_use]
    pub fn current_silence_duration(&self) -> f32 {
        self.silence_start
            .map_or(0.0, |start| self.current_time() - start)
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.noise_floor = 0.001;
        self.silence_start = None;
        self.samples_processed = 0;
        self.energy_history.clear();
    }

    /// Process a frame and detect silence
    ///
    /// Returns Some(SilenceSegment) when a silence period ends, None otherwise
    pub fn process_frame(&mut self, frame: &[f32]) -> Option<SilenceSegment> {
        let energy = Self::compute_energy(frame);
        let current_time = self.current_time();

        // Update adaptive threshold
        if self.config.adaptive {
            self.update_noise_floor(energy);
        }

        let threshold = if self.config.adaptive {
            self.noise_floor.mul_add(2.0, self.config.silence_threshold)
        } else {
            self.config.silence_threshold
        };

        let is_silence = energy < threshold;
        self.samples_processed += frame.len();

        match (self.silence_start, is_silence) {
            (None, true) => {
                // Starting silence
                self.silence_start = Some(current_time);
                None
            }
            (Some(start), false) => {
                // Ending silence
                self.silence_start = None;
                let segment = SilenceSegment {
                    start,
                    end: current_time,
                    noise_floor: self.noise_floor,
                };
                // Only return if silence was long enough
                if segment.duration() >= self.config.min_silence_duration {
                    Some(segment)
                } else {
                    None
                }
            }
            (Some(start), true) => {
                // Continuing silence - check for max duration
                let duration = current_time - start;
                if duration >= self.config.max_silence_duration {
                    // Force end the silence segment
                    self.silence_start = Some(current_time);
                    Some(SilenceSegment {
                        start,
                        end: current_time,
                        noise_floor: self.noise_floor,
                    })
                } else {
                    None
                }
            }
            (None, false) => {
                // Continuing non-silence
                None
            }
        }
    }

    /// Detect all silence segments in audio
    #[must_use]
    pub fn detect(&mut self, audio: &[f32], frame_size: usize) -> Vec<SilenceSegment> {
        self.reset();
        let mut segments = Vec::new();

        for frame in audio.chunks(frame_size) {
            if frame.len() < frame_size / 2 {
                break;
            }
            if let Some(segment) = self.process_frame(frame) {
                segments.push(segment);
            }
        }

        // Handle ongoing silence at end
        if let Some(start) = self.silence_start {
            let end = self.current_time();
            if end - start >= self.config.min_silence_duration {
                segments.push(SilenceSegment {
                    start,
                    end,
                    noise_floor: self.noise_floor,
                });
            }
        }

        segments
    }

    /// Compute frame energy (RMS)
    fn compute_energy(frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }
        let sum: f32 = frame.iter().map(|x| x * x).sum();
        (sum / frame.len() as f32).sqrt()
    }

    /// Update adaptive noise floor
    fn update_noise_floor(&mut self, energy: f32) {
        self.energy_history.push(energy);
        if self.energy_history.len() > 100 {
            self.energy_history.remove(0);
        }

        // Use lower percentile of recent energy as noise floor
        if self.energy_history.len() >= 10 {
            let mut sorted = self.energy_history.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let percentile_idx = sorted.len() / 10;
            let estimated_floor = sorted[percentile_idx];

            self.noise_floor += self.config.adaptation_rate * (estimated_floor - self.noise_floor);
        }
    }

    /// Get current time in seconds
    fn current_time(&self) -> f32 {
        self.samples_processed as f32 / self.sample_rate as f32
    }
}

impl Default for SilenceDetector {
    fn default() -> Self {
        Self::with_sample_rate(16000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence_segment_duration() {
        let segment = SilenceSegment {
            start: 1.0,
            end: 2.5,
            noise_floor: 0.001,
        };
        assert!((segment.duration() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_silence_segment_is_utterance_boundary() {
        let segment = SilenceSegment {
            start: 0.0,
            end: 0.5,
            noise_floor: 0.001,
        };
        let config = SilenceConfig::default();
        assert!(segment.is_utterance_boundary(&config));

        let short_segment = SilenceSegment {
            start: 0.0,
            end: 0.1,
            noise_floor: 0.001,
        };
        assert!(!short_segment.is_utterance_boundary(&config));
    }

    #[test]
    fn test_silence_detector_new() {
        let config = SilenceConfig::default();
        let detector = SilenceDetector::new(config, 16000);
        assert!((detector.noise_floor() - 0.001).abs() < f32::EPSILON);
    }

    #[test]
    fn test_silence_detector_with_sample_rate() {
        let detector = SilenceDetector::with_sample_rate(48000);
        assert_eq!(detector.config().min_silence_duration, 0.3);
    }

    #[test]
    fn test_silence_detector_default() {
        let detector = SilenceDetector::default();
        assert!(!detector.is_silence());
    }

    #[test]
    fn test_silence_detector_reset() {
        let mut detector = SilenceDetector::default();
        detector.samples_processed = 1000;
        detector.reset();
        assert_eq!(detector.current_silence_duration(), 0.0);
    }

    #[test]
    fn test_silence_detector_process_silence() {
        let mut detector = SilenceDetector::default();
        let silence_frame = vec![0.0; 480];
        let result = detector.process_frame(&silence_frame);
        assert!(result.is_none()); // First frame doesn't return segment
        assert!(detector.is_silence());
    }

    #[test]
    fn test_silence_detector_process_speech() {
        let mut detector = SilenceDetector::default();
        let speech_frame: Vec<f32> = (0..480).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let result = detector.process_frame(&speech_frame);
        assert!(result.is_none());
        assert!(!detector.is_silence());
    }

    #[test]
    fn test_silence_detector_detect_empty() {
        let mut detector = SilenceDetector::default();
        let segments = detector.detect(&[], 480);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_silence_detector_detect_all_silence() {
        let mut detector = SilenceDetector::default();
        let audio = vec![0.0; 16000]; // 1 second of silence
        let segments = detector.detect(&audio, 480);
        // Should detect at least one silence segment
        assert!(!segments.is_empty());
    }

    #[test]
    fn test_compute_energy_empty() {
        let energy = SilenceDetector::compute_energy(&[]);
        assert!((energy - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_energy_silence() {
        let frame = vec![0.0; 480];
        let energy = SilenceDetector::compute_energy(&frame);
        assert!((energy - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_energy_signal() {
        let frame = vec![1.0; 480];
        let energy = SilenceDetector::compute_energy(&frame);
        assert!((energy - 1.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_silence_detector_silence_to_speech_transition() {
        let mut detector = SilenceDetector::default();

        // Start with silence
        let silence = vec![0.0; 480];
        detector.process_frame(&silence);
        assert!(detector.is_silence());

        // Transition to speech
        let speech: Vec<f32> = (0..480).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let result = detector.process_frame(&speech);
        // May return a silence segment if silence was long enough
        assert!(!detector.is_silence());
        // Result depends on whether silence was long enough
        let _ = result;
    }

    #[test]
    fn test_silence_detector_max_silence_duration() {
        let config = SilenceConfig::default().with_max_silence_duration(0.1);
        let mut detector = SilenceDetector::new(config, 16000);

        // Process enough silence to hit max duration
        let silence = vec![0.0; 480];
        let mut segments_found = 0;
        for _ in 0..100 {
            if let Some(_segment) = detector.process_frame(&silence) {
                segments_found += 1;
            }
        }
        // Should have found at least one segment due to max duration
        assert!(segments_found > 0);
    }

    #[test]
    fn test_silence_detector_adaptive_disabled() {
        let config = SilenceConfig::default().with_adaptive(false);
        let mut detector = SilenceDetector::new(config, 16000);

        let frame: Vec<f32> = (0..480).map(|i| (i as f32 * 0.01).sin() * 0.01).collect();
        detector.process_frame(&frame);
        // Noise floor should not be updated when adaptive is disabled
        assert!((detector.noise_floor() - 0.001).abs() < 0.01);
    }

    #[test]
    fn test_silence_detector_detect_speech_then_silence() {
        let mut detector = SilenceDetector::default();

        // Speech followed by silence
        let speech: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let silence = vec![0.0; 8000];

        let mut audio = speech;
        audio.extend(silence);

        let segments = detector.detect(&audio, 480);
        // Should detect silence after speech
        assert!(!segments.is_empty());
    }

    #[test]
    fn test_silence_detector_current_time() {
        let mut detector = SilenceDetector::default();
        let frame = vec![0.0; 480];

        // Process some frames
        for _ in 0..10 {
            detector.process_frame(&frame);
        }

        // Should have processed 10 * 480 = 4800 samples at 16kHz = 0.3 seconds
        let duration = detector.current_silence_duration();
        assert!(duration > 0.0);
    }

    #[test]
    fn test_silence_config_builder() {
        let config = SilenceConfig::new()
            .with_min_silence_duration(0.5)
            .with_max_silence_duration(3.0)
            .with_silence_threshold(0.005)
            .with_adaptive(false)
            .with_adaptation_rate(0.05);

        assert!((config.min_silence_duration - 0.5).abs() < f32::EPSILON);
        assert!((config.max_silence_duration - 3.0).abs() < f32::EPSILON);
        assert!((config.silence_threshold - 0.005).abs() < f32::EPSILON);
        assert!(!config.adaptive);
        assert!((config.adaptation_rate - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_silence_detector_short_frames() {
        let mut detector = SilenceDetector::default();
        // Very short frame (less than half frame size)
        let short_frame = vec![0.0; 100];
        let segments = detector.detect(&short_frame, 480);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_silence_segment_properties() {
        let segment = SilenceSegment {
            start: 1.5,
            end: 3.0,
            noise_floor: 0.002,
        };
        assert!((segment.duration() - 1.5).abs() < f32::EPSILON);
        assert!((segment.noise_floor - 0.002).abs() < f32::EPSILON);
    }
}

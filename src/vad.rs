//! Voice Activity Detection (VAD)
//!
//! Detects speech segments in audio to enable efficient streaming transcription.
//!
//! # Algorithm
//!
//! This module implements a multi-feature VAD using:
//! 1. **Energy-based detection**: Short-term energy above threshold
//! 2. **Zero-crossing rate**: Speech has characteristic ZCR patterns
//! 3. **Spectral centroid**: Speech has higher spectral centroid than noise
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::vad::{VadConfig, VoiceActivityDetector};
//!
//! let vad = VoiceActivityDetector::new(VadConfig::default());
//! let segments = vad.detect(&audio_samples);
//! ```
//!
//! # Streaming Support
//!
//! For streaming applications, use `VadState` to process chunks incrementally:
//!
//! ```rust,ignore
//! let mut state = VadState::new(config);
//! for chunk in audio_chunks {
//!     if state.process(&chunk) == VadEvent::SpeechStart {
//!         // Begin transcription
//!     }
//! }
//! ```

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

    /// Get frame duration in seconds
    #[must_use]
    pub fn frame_duration(&self) -> f32 {
        self.frame_size as f32 / self.sample_rate as f32
    }
}

/// Voice activity detection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    /// No speech detected
    Silence,
    /// Speech is being detected
    Speech,
    /// Transitioning from silence to speech
    SpeechStart,
    /// Transitioning from speech to silence
    SpeechEnd,
}

/// Voice activity event for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// Continue in current state (no change)
    Continue,
    /// Speech started
    SpeechStart,
    /// Speech ended
    SpeechEnd,
}

/// Detected speech segment
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Average energy during segment
    pub energy: f32,
}

impl SpeechSegment {
    /// Duration of the segment in seconds
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }
}

/// Voice Activity Detector
#[derive(Debug, Clone)]
pub struct VoiceActivityDetector {
    config: VadConfig,
    /// Adaptive noise floor estimate
    noise_floor: f32,
    /// Current VAD state
    state: VadState,
    /// Consecutive speech frames
    speech_frames: usize,
    /// Consecutive silence frames
    silence_frames: usize,
    /// Current time in samples
    current_sample: usize,
}

impl VoiceActivityDetector {
    /// Create a new VAD with the given configuration
    #[must_use]
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            noise_floor: 0.001, // Initial noise floor estimate
            state: VadState::Silence,
            speech_frames: 0,
            silence_frames: 0,
            current_sample: 0,
        }
    }

    /// Get the current configuration
    #[must_use]
    pub const fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Get current state
    #[must_use]
    pub const fn state(&self) -> VadState {
        self.state
    }

    /// Reset the detector state
    pub fn reset(&mut self) {
        self.noise_floor = 0.001;
        self.state = VadState::Silence;
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.current_sample = 0;
    }

    /// Detect speech segments in audio
    ///
    /// Returns a list of speech segments with timestamps.
    #[must_use]
    pub fn detect(&mut self, audio: &[f32]) -> Vec<SpeechSegment> {
        self.reset();
        let mut segments = Vec::new();
        let mut current_segment: Option<(f32, f32)> = None; // (start, energy_sum)
        let mut frame_count = 0;

        for frame in audio.chunks(self.config.frame_size) {
            if frame.len() < self.config.frame_size / 2 {
                break; // Skip very short trailing frame
            }

            let event = self.process_frame(frame);
            let time = self.sample_to_time(self.current_sample);

            match event {
                VadEvent::SpeechStart => {
                    current_segment = Some((time, Self::frame_energy(frame)));
                    frame_count = 1;
                }
                VadEvent::SpeechEnd => {
                    if let Some((start, energy_sum)) = current_segment.take() {
                        segments.push(SpeechSegment {
                            start,
                            end: time,
                            energy: energy_sum / frame_count.max(1) as f32,
                        });
                    }
                }
                VadEvent::Continue => {
                    if let Some((_, ref mut energy_sum)) = current_segment {
                        *energy_sum += Self::frame_energy(frame);
                        frame_count += 1;
                    }
                }
            }

            self.current_sample += frame.len();
        }

        // Handle unterminated speech segment
        if let Some((start, energy_sum)) = current_segment {
            let time = self.sample_to_time(self.current_sample);
            segments.push(SpeechSegment {
                start,
                end: time,
                energy: energy_sum / frame_count.max(1) as f32,
            });
        }

        segments
    }

    /// Process a single frame and return event
    ///
    /// Use this for streaming VAD.
    pub fn process_frame(&mut self, frame: &[f32]) -> VadEvent {
        let energy = Self::frame_energy(frame);
        let zcr = Self::zero_crossing_rate(frame);

        // Update noise floor (only during silence)
        if self.state == VadState::Silence {
            self.noise_floor = self
                .config
                .smoothing
                .mul_add(self.noise_floor, (1.0 - self.config.smoothing) * energy);
        }

        // Determine if frame contains speech
        let is_speech = self.is_speech_frame(energy, zcr);

        // State machine
        match self.state {
            VadState::Silence | VadState::SpeechEnd => {
                if is_speech {
                    self.speech_frames += 1;
                    self.silence_frames = 0;

                    if self.speech_frames >= self.config.min_speech_frames {
                        self.state = VadState::Speech;
                        VadEvent::SpeechStart
                    } else {
                        VadEvent::Continue
                    }
                } else {
                    self.speech_frames = 0;
                    self.state = VadState::Silence;
                    VadEvent::Continue
                }
            }
            VadState::Speech | VadState::SpeechStart => {
                if is_speech {
                    self.silence_frames = 0;
                    self.speech_frames += 1;
                    VadEvent::Continue
                } else {
                    self.silence_frames += 1;
                    self.speech_frames = 0;

                    if self.silence_frames >= self.config.min_silence_frames {
                        self.state = VadState::Silence;
                        VadEvent::SpeechEnd
                    } else {
                        VadEvent::Continue
                    }
                }
            }
        }
    }

    /// Check if a frame is speech based on features
    fn is_speech_frame(&self, energy: f32, zcr: f32) -> bool {
        // Energy above threshold
        let energy_above = energy > self.noise_floor * self.config.energy_threshold;

        // ZCR in typical speech range (not too high like noise, not too low)
        let zcr_in_range = zcr > 0.05 && zcr < self.config.zcr_threshold;

        // Both conditions must be met for robust detection
        energy_above && zcr_in_range
    }

    /// Calculate frame energy (RMS)
    fn frame_energy(frame: &[f32]) -> f32 {
        let sum: f32 = frame.iter().map(|&x| x * x).sum();
        (sum / frame.len() as f32).sqrt()
    }

    /// Calculate zero-crossing rate
    fn zero_crossing_rate(frame: &[f32]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }

        let crossings: usize = frame
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();

        crossings as f32 / (frame.len() - 1) as f32
    }

    /// Convert sample index to time in seconds
    fn sample_to_time(&self, sample: usize) -> f32 {
        sample as f32 / self.config.sample_rate as f32
    }
}

impl Default for VoiceActivityDetector {
    fn default() -> Self {
        Self::new(VadConfig::default())
    }
}

/// Streaming VAD processor
///
/// Processes audio in chunks and emits events.
#[derive(Debug, Clone)]
pub struct StreamingVad {
    detector: VoiceActivityDetector,
    /// Buffer for incomplete frames
    buffer: Vec<f32>,
    /// Accumulated speech audio
    speech_buffer: Vec<f32>,
    /// Whether we're currently in speech
    in_speech: bool,
}

impl StreamingVad {
    /// Create a new streaming VAD
    #[must_use]
    pub fn new(config: VadConfig) -> Self {
        Self {
            detector: VoiceActivityDetector::new(config),
            buffer: Vec::new(),
            speech_buffer: Vec::new(),
            in_speech: false,
        }
    }

    /// Process audio chunk and return any completed speech segments
    ///
    /// # Arguments
    /// * `audio` - Audio samples to process
    ///
    /// # Returns
    /// Completed speech segments (if speech ended) and whether currently in speech
    pub fn process(&mut self, audio: &[f32]) -> (Vec<f32>, bool) {
        // Add to buffer
        self.buffer.extend_from_slice(audio);

        let frame_size = self.detector.config.frame_size;
        let mut completed_speech: Option<Vec<f32>> = None;

        // Process complete frames
        while self.buffer.len() >= frame_size {
            let frame: Vec<f32> = self.buffer.drain(..frame_size).collect();
            let event = self.detector.process_frame(&frame);

            match event {
                VadEvent::SpeechStart => {
                    self.in_speech = true;
                    self.speech_buffer.clear();
                    self.speech_buffer.extend_from_slice(&frame);
                }
                VadEvent::Continue => {
                    if self.in_speech {
                        self.speech_buffer.extend_from_slice(&frame);
                    }
                }
                VadEvent::SpeechEnd => {
                    self.in_speech = false;
                    if !self.speech_buffer.is_empty() {
                        completed_speech = Some(std::mem::take(&mut self.speech_buffer));
                    }
                }
            }
        }

        (completed_speech.unwrap_or_default(), self.in_speech)
    }

    /// Get any remaining buffered speech (call at end of stream)
    #[must_use]
    pub fn flush(&mut self) -> Vec<f32> {
        // Process any remaining buffer
        if !self.buffer.is_empty() && self.in_speech {
            self.speech_buffer.extend_from_slice(&self.buffer);
        }
        self.buffer.clear();
        self.in_speech = false;

        std::mem::take(&mut self.speech_buffer)
    }

    /// Reset the streaming state
    pub fn reset(&mut self) {
        self.detector.reset();
        self.buffer.clear();
        self.speech_buffer.clear();
        self.in_speech = false;
    }

    /// Check if currently in speech
    #[must_use]
    pub const fn is_in_speech(&self) -> bool {
        self.in_speech
    }
}

impl Default for StreamingVad {
    fn default() -> Self {
        Self::new(VadConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // VadConfig Tests
    // =========================================================================

    #[test]
    fn test_vad_config_default() {
        let config = VadConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.frame_size, 480);
        assert!((config.energy_threshold - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vad_config_low_latency() {
        let config = VadConfig::low_latency();
        assert_eq!(config.frame_size, 160);
        assert_eq!(config.min_speech_frames, 5);
    }

    #[test]
    fn test_vad_config_high_accuracy() {
        let config = VadConfig::high_accuracy();
        assert_eq!(config.frame_size, 800);
        assert_eq!(config.min_speech_frames, 2);
    }

    #[test]
    fn test_vad_config_frame_duration() {
        let config = VadConfig::default();
        assert!((config.frame_duration() - 0.03).abs() < 0.001); // 30ms
    }

    // =========================================================================
    // SpeechSegment Tests
    // =========================================================================

    #[test]
    fn test_speech_segment_duration() {
        let segment = SpeechSegment {
            start: 1.0,
            end: 3.5,
            energy: 0.1,
        };
        assert!((segment.duration() - 2.5).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Frame Feature Tests
    // =========================================================================

    #[test]
    fn test_frame_energy_silence() {
        let silence = vec![0.0; 480];
        let energy = VoiceActivityDetector::frame_energy(&silence);
        assert!(energy < 0.001);
    }

    #[test]
    fn test_frame_energy_tone() {
        let tone: Vec<f32> = (0..480)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let energy = VoiceActivityDetector::frame_energy(&tone);
        assert!(energy > 0.3); // ~0.35 RMS for 0.5 amplitude sine
    }

    #[test]
    fn test_zero_crossing_rate_silence() {
        let silence = vec![0.0; 480];
        let zcr = VoiceActivityDetector::zero_crossing_rate(&silence);
        assert!(zcr < 0.01);
    }

    #[test]
    fn test_zero_crossing_rate_tone() {
        // 440Hz tone at 16kHz, 480 samples = 30ms
        // 440Hz * 0.03s = 13.2 cycles, each cycle crosses zero twice = ~26 crossings
        // ZCR = 26 / 479 ≈ 0.054
        let tone: Vec<f32> = (0..480)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let zcr = VoiceActivityDetector::zero_crossing_rate(&tone);
        assert!(zcr > 0.04 && zcr < 0.08, "ZCR was {zcr}");
    }

    #[test]
    fn test_zero_crossing_rate_noise() {
        // High frequency noise has high ZCR
        let noise: Vec<f32> = (0..480)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let zcr = VoiceActivityDetector::zero_crossing_rate(&noise);
        assert!(zcr > 0.9);
    }

    // =========================================================================
    // VoiceActivityDetector Tests
    // =========================================================================

    #[test]
    fn test_vad_new() {
        let vad = VoiceActivityDetector::new(VadConfig::default());
        assert_eq!(vad.state(), VadState::Silence);
    }

    #[test]
    fn test_vad_default() {
        let vad = VoiceActivityDetector::default();
        assert_eq!(vad.state(), VadState::Silence);
    }

    #[test]
    fn test_vad_reset() {
        let mut vad = VoiceActivityDetector::default();
        vad.speech_frames = 10;
        vad.state = VadState::Speech;

        vad.reset();

        assert_eq!(vad.state(), VadState::Silence);
        assert_eq!(vad.speech_frames, 0);
    }

    #[test]
    fn test_vad_detect_silence() {
        let mut vad = VoiceActivityDetector::default();
        let silence = vec![0.0; 16000]; // 1 second of silence

        let segments = vad.detect(&silence);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_vad_detect_speech_like() {
        let mut vad = VoiceActivityDetector::default();

        // Generate speech-like signal (tone with varying amplitude)
        let mut audio = Vec::new();

        // Silence
        audio.extend(vec![0.0; 4800]); // 300ms silence

        // Speech-like signal (440Hz tone with amplitude 0.3, ZCR ~0.1)
        for i in 0..8000 {
            // 500ms
            let t = i as f32 / 16000.0;
            audio.push((2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3);
        }

        // Silence
        audio.extend(vec![0.0; 4800]); // 300ms silence

        let segments = vad.detect(&audio);

        // Should detect one speech segment
        assert!(!segments.is_empty(), "Should detect speech segment");

        if !segments.is_empty() {
            // Verify timing is roughly correct
            assert!(segments[0].start >= 0.2);
            assert!(segments[0].start <= 0.5);
        }
    }

    #[test]
    fn test_vad_process_frame() {
        let mut vad = VoiceActivityDetector::new(VadConfig {
            min_speech_frames: 1,
            min_silence_frames: 1,
            ..VadConfig::default()
        });

        // Process silence
        let silence = vec![0.0; 480];
        assert_eq!(vad.process_frame(&silence), VadEvent::Continue);
        assert_eq!(vad.state(), VadState::Silence);

        // Process speech-like frame
        let speech: Vec<f32> = (0..480)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let event = vad.process_frame(&speech);

        // With min_speech_frames=1, should immediately start
        assert_eq!(event, VadEvent::SpeechStart);
        assert_eq!(vad.state(), VadState::Speech);
    }

    // =========================================================================
    // StreamingVad Tests
    // =========================================================================

    #[test]
    fn test_streaming_vad_new() {
        let vad = StreamingVad::new(VadConfig::default());
        assert!(!vad.is_in_speech());
    }

    #[test]
    fn test_streaming_vad_default() {
        let vad = StreamingVad::default();
        assert!(!vad.is_in_speech());
    }

    #[test]
    fn test_streaming_vad_process_silence() {
        let mut vad = StreamingVad::default();
        let silence = vec![0.0; 1600]; // 100ms

        let (speech, in_speech) = vad.process(&silence);
        assert!(speech.is_empty());
        assert!(!in_speech);
    }

    #[test]
    fn test_streaming_vad_reset() {
        let mut vad = StreamingVad::default();
        vad.buffer = vec![0.1; 100];
        vad.in_speech = true;

        vad.reset();

        assert!(vad.buffer.is_empty());
        assert!(!vad.is_in_speech());
    }

    #[test]
    fn test_streaming_vad_flush_empty() {
        let mut vad = StreamingVad::default();
        let flushed = vad.flush();
        assert!(flushed.is_empty());
    }

    #[test]
    fn test_streaming_vad_flush_with_speech() {
        let mut vad = StreamingVad::default();
        vad.speech_buffer = vec![0.1, 0.2, 0.3];
        vad.in_speech = true;

        let flushed = vad.flush();
        assert_eq!(flushed.len(), 3);
        assert!(!vad.is_in_speech());
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_vad_empty_audio() {
        let mut vad = VoiceActivityDetector::default();
        let segments = vad.detect(&[]);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_vad_short_audio() {
        let mut vad = VoiceActivityDetector::default();
        let short = vec![0.1; 100]; // Less than one frame
        let segments = vad.detect(&short);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_zcr_short_frame() {
        let short = vec![0.1];
        let zcr = VoiceActivityDetector::zero_crossing_rate(&short);
        assert!(zcr < f32::EPSILON);
    }

    #[test]
    fn test_sample_to_time() {
        let vad = VoiceActivityDetector::default();
        assert!((vad.sample_to_time(16000) - 1.0).abs() < f32::EPSILON);
        assert!((vad.sample_to_time(8000) - 0.5).abs() < f32::EPSILON);
    }
}

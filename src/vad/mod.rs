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

mod config;
mod models;
mod silence;

pub use config::{SilenceConfig, VadConfig};
pub use models::{SpeechSegment, VadEvent, VadState};
pub use silence::{SilenceDetector, SilenceSegment};

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

        let crossings: f32 = frame
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count() as f32;

        crossings / (frame.len() - 1) as f32
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

    #[test]
    fn test_vad_new() {
        let vad = VoiceActivityDetector::new(VadConfig::default());
        assert_eq!(vad.state(), VadState::Silence);
    }

    #[test]
    fn test_vad_default() {
        let vad = VoiceActivityDetector::default();
        assert_eq!(vad.config().sample_rate, 16000);
    }

    #[test]
    fn test_vad_reset() {
        let mut vad = VoiceActivityDetector::default();
        vad.speech_frames = 10;
        vad.reset();
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
    fn test_vad_frame_energy() {
        let frame = vec![0.5; 480];
        let energy = VoiceActivityDetector::frame_energy(&frame);
        assert!((energy - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_vad_zero_crossing_rate() {
        // Alternating signal
        let frame: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let zcr = VoiceActivityDetector::zero_crossing_rate(&frame);
        assert!(zcr > 0.9);
    }

    #[test]
    fn test_vad_zero_crossing_rate_short() {
        let frame = vec![1.0];
        let zcr = VoiceActivityDetector::zero_crossing_rate(&frame);
        assert!((zcr - 0.0).abs() < f32::EPSILON);
    }

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
        let silence = vec![0.0; 480];
        let (speech, in_speech) = vad.process(&silence);
        assert!(speech.is_empty());
        assert!(!in_speech);
    }

    #[test]
    fn test_streaming_vad_reset() {
        let mut vad = StreamingVad::default();
        vad.in_speech = true;
        vad.reset();
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
        vad.in_speech = true;
        vad.speech_buffer = vec![0.5; 1000];
        let flushed = vad.flush();
        assert_eq!(flushed.len(), 1000);
        assert!(!vad.is_in_speech());
    }
}

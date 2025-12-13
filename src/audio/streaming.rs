//! Streaming audio processor for real-time transcription
//!
//! Integrates the ring buffer, resampler, and VAD for continuous audio processing
//! as specified in sections 11.2-11.4 of the whisper.apr spec.
//!
//! # Architecture (per spec 11.3)
//!
//! ```text
//! AudioWorklet ──► Ring Buffer ──► Resampler ──► VAD ──► Chunk Accumulator ──► Inference
//!   (RT thread)    (lock-free)     (16kHz)       │              │
//!                                                │              │
//!                                     silence ◄──┘    speech ◄──┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::audio::{StreamingProcessor, StreamingConfig};
//!
//! let config = StreamingConfig::default();
//! let mut processor = StreamingProcessor::new(config);
//!
//! // Feed audio from AudioWorklet (any sample rate)
//! processor.push_audio(&samples_44100hz);
//!
//! // Check if a complete chunk is ready for inference
//! if let Some(chunk) = processor.get_chunk() {
//!     let result = whisper.transcribe(&chunk, options)?;
//! }
//! ```

use super::{RingBuffer, SincResampler, SAMPLE_RATE};
use crate::vad::{VadConfig, VoiceActivityDetector};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Default chunk duration in seconds (Whisper processes 30s segments)
pub const DEFAULT_CHUNK_DURATION: f32 = 30.0;

/// Default overlap between chunks (for smooth transcription)
pub const DEFAULT_CHUNK_OVERLAP: f32 = 1.0;

/// Minimum speech duration to trigger chunk (prevents spurious triggers)
pub const MIN_SPEECH_DURATION_MS: u32 = 500;

/// Configuration for the streaming audio processor
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Input sample rate (from AudioContext)
    pub input_sample_rate: u32,
    /// Target sample rate for Whisper (always 16000)
    pub output_sample_rate: u32,
    /// Chunk duration in seconds
    pub chunk_duration: f32,
    /// Overlap between chunks in seconds
    pub chunk_overlap: f32,
    /// Enable VAD filtering
    pub enable_vad: bool,
    /// VAD threshold (0.0-1.0)
    pub vad_threshold: f32,
    /// Minimum speech duration in ms before triggering
    pub min_speech_duration_ms: u32,
    /// Ring buffer duration in seconds
    pub buffer_duration: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            input_sample_rate: 44100,
            output_sample_rate: SAMPLE_RATE,
            chunk_duration: DEFAULT_CHUNK_DURATION,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
            enable_vad: true,
            vad_threshold: 0.5,
            min_speech_duration_ms: MIN_SPEECH_DURATION_MS,
            buffer_duration: 120.0, // 2 minutes per spec 11.3
        }
    }
}

impl StreamingConfig {
    /// Create config for a specific input sample rate
    #[must_use]
    pub fn with_sample_rate(input_sample_rate: u32) -> Self {
        Self {
            input_sample_rate,
            ..Default::default()
        }
    }

    /// Disable VAD (process all audio regardless of speech)
    #[must_use]
    pub fn without_vad(mut self) -> Self {
        self.enable_vad = false;
        self
    }

    /// Set VAD threshold
    #[must_use]
    pub fn vad_threshold(mut self, threshold: f32) -> Self {
        self.vad_threshold = threshold;
        self
    }

    /// Set chunk duration
    #[must_use]
    pub fn chunk_duration(mut self, duration: f32) -> Self {
        self.chunk_duration = duration;
        self
    }

    /// Set chunk overlap (WAPR-102)
    ///
    /// The overlap is the amount of audio from the end of the previous chunk
    /// that is prepended to the next chunk. This helps maintain context across
    /// chunk boundaries for better transcription accuracy.
    #[must_use]
    pub fn chunk_overlap(mut self, overlap: f32) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    /// Set minimum speech duration in milliseconds
    #[must_use]
    pub fn min_speech_duration_ms(mut self, duration: u32) -> Self {
        self.min_speech_duration_ms = duration;
        self
    }

    /// Get chunk size in samples at output rate
    #[must_use]
    pub fn chunk_samples(&self) -> usize {
        (self.chunk_duration * self.output_sample_rate as f32) as usize
    }

    /// Get overlap size in samples at output rate
    #[must_use]
    pub fn overlap_samples(&self) -> usize {
        (self.chunk_overlap * self.output_sample_rate as f32) as usize
    }
}

/// State of the streaming processor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorState {
    /// Waiting for speech to start
    WaitingForSpeech,
    /// Currently accumulating speech
    AccumulatingSpeech,
    /// Partial result is available (enough audio for interim transcription)
    PartialResultReady,
    /// Chunk ready for processing
    ChunkReady,
    /// Currently processing a chunk (transcription in progress)
    Processing,
    /// Error state (recoverable)
    Error,
}

/// Event emitted by the streaming processor (WAPR-100)
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingEvent {
    /// Speech detection started
    SpeechStart,
    /// Speech detection ended
    SpeechEnd,
    /// Partial result is available
    PartialReady {
        /// Audio accumulated so far (samples)
        accumulated_samples: usize,
        /// Duration in seconds
        duration_secs: f32,
    },
    /// Full chunk is ready for transcription
    ChunkReady {
        /// Chunk duration in seconds
        duration_secs: f32,
    },
    /// Processing started
    ProcessingStarted,
    /// Processing completed
    ProcessingCompleted,
    /// Error occurred
    Error(String),
    /// Reset occurred
    Reset,
}

/// Streaming audio processor for real-time transcription
///
/// This processor handles:
/// 1. Buffering incoming audio via lock-free ring buffer
/// 2. Resampling from native rate to 16kHz
/// 3. Voice activity detection to skip silence
/// 4. Accumulating speech into 30s chunks for inference
/// 5. Emitting events for state transitions (WAPR-100)
#[derive(Debug)]
pub struct StreamingProcessor {
    /// Configuration
    config: StreamingConfig,
    /// Ring buffer for incoming audio
    input_buffer: RingBuffer,
    /// Resampler (if input rate != 16kHz)
    resampler: Option<SincResampler>,
    /// Voice activity detector
    vad: VoiceActivityDetector,
    /// Accumulated chunk for inference
    chunk_buffer: Vec<f32>,
    /// Overlap buffer (last N samples of previous chunk)
    overlap_buffer: Vec<f32>,
    /// Current processor state
    state: ProcessorState,
    /// Previous state (for detecting transitions)
    prev_state: ProcessorState,
    /// Consecutive speech frames count
    speech_frames: u32,
    /// Consecutive silence frames count
    silence_frames: u32,
    /// Total samples processed
    samples_processed: u64,
    /// Pending events queue (WAPR-100)
    events: Vec<StreamingEvent>,
    /// Threshold for partial result (samples) - typically 3-5 seconds
    partial_threshold_samples: usize,
    /// Last partial result position (to avoid duplicate events)
    last_partial_position: usize,
}

/// Default partial result threshold: 3 seconds of audio at 16kHz
const DEFAULT_PARTIAL_THRESHOLD_SECS: f32 = 3.0;

impl StreamingProcessor {
    /// Create a new streaming processor with the given configuration
    #[must_use]
    pub fn new(config: StreamingConfig) -> Self {
        let input_buffer =
            RingBuffer::for_duration(config.buffer_duration, config.input_sample_rate);

        let resampler = if config.input_sample_rate != config.output_sample_rate {
            SincResampler::new(config.input_sample_rate, config.output_sample_rate).ok()
        } else {
            None
        };

        let vad_config = VadConfig {
            energy_threshold: config.vad_threshold * 4.0, // Scale 0-1 to typical energy threshold
            ..VadConfig::default()
        };
        let vad = VoiceActivityDetector::new(vad_config);

        let chunk_capacity = config.chunk_samples() + config.overlap_samples();
        let partial_threshold_samples =
            (DEFAULT_PARTIAL_THRESHOLD_SECS * config.output_sample_rate as f32) as usize;

        Self {
            config,
            input_buffer,
            resampler,
            vad,
            chunk_buffer: Vec::with_capacity(chunk_capacity),
            overlap_buffer: Vec::new(),
            state: ProcessorState::WaitingForSpeech,
            prev_state: ProcessorState::WaitingForSpeech,
            speech_frames: 0,
            silence_frames: 0,
            samples_processed: 0,
            events: Vec::new(),
            partial_threshold_samples,
            last_partial_position: 0,
        }
    }

    /// Create a processor with default config for the given sample rate
    #[must_use]
    pub fn with_sample_rate(sample_rate: u32) -> Self {
        Self::new(StreamingConfig::with_sample_rate(sample_rate))
    }

    /// Get current processor state
    #[must_use]
    pub const fn state(&self) -> ProcessorState {
        self.state
    }

    /// Get total samples processed
    #[must_use]
    pub const fn samples_processed(&self) -> u64 {
        self.samples_processed
    }

    /// Get current chunk buffer length
    #[must_use]
    pub fn chunk_len(&self) -> usize {
        self.chunk_buffer.len()
    }

    /// Get chunk progress (0.0 to 1.0)
    #[must_use]
    pub fn chunk_progress(&self) -> f32 {
        self.chunk_buffer.len() as f32 / self.config.chunk_samples() as f32
    }

    /// Check if a complete chunk is ready
    #[must_use]
    pub fn has_chunk(&self) -> bool {
        self.state == ProcessorState::ChunkReady
            || self.chunk_buffer.len() >= self.config.chunk_samples()
    }

    // =========================================================================
    // Chunk Overlap Management (WAPR-102)
    // =========================================================================

    /// Get the current overlap buffer length
    #[must_use]
    pub fn overlap_len(&self) -> usize {
        self.overlap_buffer.len()
    }

    /// Get the overlap duration in seconds
    #[must_use]
    pub fn overlap_duration(&self) -> f32 {
        self.overlap_buffer.len() as f32 / self.config.output_sample_rate as f32
    }

    /// Check if overlap buffer has data
    #[must_use]
    pub fn has_overlap(&self) -> bool {
        !self.overlap_buffer.is_empty()
    }

    /// Get the configured overlap size in samples
    #[must_use]
    pub fn configured_overlap_samples(&self) -> usize {
        self.config.overlap_samples()
    }

    /// Get the configured overlap duration in seconds
    #[must_use]
    pub fn configured_overlap_duration(&self) -> f32 {
        self.config.chunk_overlap
    }

    /// Clear the overlap buffer
    ///
    /// This is useful when you want to start fresh without using
    /// the previous chunk's context.
    pub fn clear_overlap(&mut self) {
        self.overlap_buffer.clear();
    }

    /// Get a copy of the overlap buffer for inspection
    #[must_use]
    pub fn get_overlap_buffer(&self) -> Vec<f32> {
        self.overlap_buffer.clone()
    }

    /// Set a custom overlap buffer
    ///
    /// This allows injecting context from a previous transcription
    /// when resuming a streaming session.
    pub fn set_overlap_buffer(&mut self, overlap: Vec<f32>) {
        self.overlap_buffer = overlap;
    }

    // =========================================================================
    // Event Handling (WAPR-100)
    // =========================================================================

    /// Check if there are pending events
    #[must_use]
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// Get number of pending events
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Drain all pending events
    pub fn drain_events(&mut self) -> Vec<StreamingEvent> {
        core::mem::take(&mut self.events)
    }

    /// Pop the next event (if any)
    pub fn pop_event(&mut self) -> Option<StreamingEvent> {
        if self.events.is_empty() {
            None
        } else {
            Some(self.events.remove(0))
        }
    }

    /// Peek at the next event without removing it
    #[must_use]
    pub fn peek_event(&self) -> Option<&StreamingEvent> {
        self.events.first()
    }

    /// Clear all pending events
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    // =========================================================================
    // Partial Results (WAPR-100)
    // =========================================================================

    /// Check if a partial result is available
    ///
    /// Returns true if enough audio has been accumulated for an interim transcription
    #[must_use]
    pub fn has_partial(&self) -> bool {
        self.state == ProcessorState::PartialResultReady
            || (self.state == ProcessorState::AccumulatingSpeech
                && self.chunk_buffer.len() >= self.partial_threshold_samples
                && self.chunk_buffer.len() > self.last_partial_position)
    }

    /// Get partial audio for interim transcription
    ///
    /// Returns the currently accumulated audio without consuming it.
    /// The chunk buffer continues to accumulate more audio.
    pub fn get_partial(&mut self) -> Option<Vec<f32>> {
        if !self.has_partial() {
            return None;
        }

        // Update last partial position to avoid duplicate events
        self.last_partial_position = self.chunk_buffer.len();

        // Return a copy of accumulated audio
        Some(self.chunk_buffer.clone())
    }

    /// Get partial audio duration in seconds
    #[must_use]
    pub fn partial_duration(&self) -> f32 {
        self.chunk_buffer.len() as f32 / self.config.output_sample_rate as f32
    }

    /// Set the partial result threshold in seconds
    ///
    /// Controls how much audio must accumulate before a partial result is triggered.
    pub fn set_partial_threshold(&mut self, seconds: f32) {
        self.partial_threshold_samples =
            (seconds * self.config.output_sample_rate as f32) as usize;
    }

    /// Get the partial result threshold in seconds
    #[must_use]
    pub fn partial_threshold(&self) -> f32 {
        self.partial_threshold_samples as f32 / self.config.output_sample_rate as f32
    }

    // =========================================================================
    // State Transitions (WAPR-100)
    // =========================================================================

    /// Mark processing as started
    ///
    /// Call this when you begin transcribing a chunk
    pub fn mark_processing_started(&mut self) {
        if self.state == ProcessorState::ChunkReady || self.state == ProcessorState::PartialResultReady {
            self.prev_state = self.state;
            self.state = ProcessorState::Processing;
            self.events.push(StreamingEvent::ProcessingStarted);
        }
    }

    /// Mark processing as completed
    ///
    /// Call this when transcription of a chunk is done
    pub fn mark_processing_completed(&mut self) {
        if self.state == ProcessorState::Processing {
            self.state = ProcessorState::WaitingForSpeech;
            self.events.push(StreamingEvent::ProcessingCompleted);
        }
    }

    /// Mark an error occurred (recoverable)
    pub fn mark_error(&mut self, message: &str) {
        self.prev_state = self.state;
        self.state = ProcessorState::Error;
        self.events.push(StreamingEvent::Error(message.to_string()));
    }

    /// Recover from error state
    pub fn recover_from_error(&mut self) {
        if self.state == ProcessorState::Error {
            self.state = ProcessorState::WaitingForSpeech;
            self.chunk_buffer.clear();
            self.last_partial_position = 0;
        }
    }

    /// Get the previous state (before last transition)
    #[must_use]
    pub const fn prev_state(&self) -> ProcessorState {
        self.prev_state
    }

    /// Emit an event internally
    fn emit_event(&mut self, event: StreamingEvent) {
        self.events.push(event);
    }

    /// Push audio samples into the processor
    ///
    /// Samples should be at the configured input sample rate
    pub fn push_audio(&mut self, samples: &[f32]) {
        self.input_buffer.write_overwrite(samples);
        self.samples_processed += samples.len() as u64;
    }

    /// Process buffered audio and update state
    ///
    /// Call this regularly (e.g., every 100ms) to process accumulated audio
    pub fn process(&mut self) {
        // Read available samples from ring buffer
        let available = self.input_buffer.available_read();
        if available == 0 {
            return;
        }

        // Process in small frames for VAD (30ms = 480 samples at 16kHz)
        let frame_size = (0.030 * self.config.input_sample_rate as f32) as usize;
        let mut input_frame = vec![0.0; frame_size];

        while self.input_buffer.available_read() >= frame_size {
            let read = self.input_buffer.read(&mut input_frame);
            if read < frame_size {
                break;
            }

            // Resample if needed
            let resampled = if let Some(ref resampler) = self.resampler {
                match resampler.resample(&input_frame) {
                    Ok(samples) => samples,
                    Err(_) => continue,
                }
            } else {
                input_frame.clone()
            };

            // VAD check using process_frame
            let is_speech = if self.config.enable_vad {
                // Process through VAD and check for speech events
                let event = self.vad.process_frame(&resampled);
                matches!(
                    event,
                    crate::vad::VadEvent::SpeechStart | crate::vad::VadEvent::Continue
                ) && self.vad.state() == crate::vad::VadState::Speech
            } else {
                true
            };

            self.update_state(is_speech, &resampled);
        }
    }

    /// Update internal state based on VAD result
    fn update_state(&mut self, is_speech: bool, samples: &[f32]) {
        let prev_speech_frames = self.speech_frames;
        let prev_silence_frames = self.silence_frames;

        if is_speech {
            self.speech_frames += 1;
            self.silence_frames = 0;
        } else {
            self.silence_frames += 1;
            self.speech_frames = 0;
        }

        // Track previous state for events
        self.prev_state = self.state;

        // State machine
        match self.state {
            ProcessorState::WaitingForSpeech => {
                if is_speech && self.speech_frames >= self.min_speech_frames() {
                    self.state = ProcessorState::AccumulatingSpeech;
                    // Start with overlap from previous chunk
                    self.chunk_buffer.clear();
                    self.chunk_buffer.extend(&self.overlap_buffer);
                    self.chunk_buffer.extend_from_slice(samples);
                    self.last_partial_position = 0;

                    // Emit speech start event (WAPR-100)
                    self.emit_event(StreamingEvent::SpeechStart);
                }
            }
            ProcessorState::AccumulatingSpeech => {
                self.chunk_buffer.extend_from_slice(samples);

                // Check for partial result threshold (WAPR-100)
                if self.chunk_buffer.len() >= self.partial_threshold_samples
                    && self.chunk_buffer.len() > self.last_partial_position
                    && self.last_partial_position == 0
                {
                    // First partial result ready
                    self.emit_event(StreamingEvent::PartialReady {
                        accumulated_samples: self.chunk_buffer.len(),
                        duration_secs: self.partial_duration(),
                    });
                }

                // Check if chunk is complete
                if self.chunk_buffer.len() >= self.config.chunk_samples() {
                    self.state = ProcessorState::ChunkReady;
                    let duration = self.chunk_buffer.len() as f32 / self.config.output_sample_rate as f32;
                    self.emit_event(StreamingEvent::ChunkReady { duration_secs: duration });
                }
                // Or if we hit extended silence (end of utterance)
                else if !is_speech && self.silence_frames >= self.max_silence_frames() {
                    // Emit speech end event
                    self.emit_event(StreamingEvent::SpeechEnd);

                    // Partial chunk is ready
                    if self.chunk_buffer.len() >= self.config.overlap_samples() * 2 {
                        self.state = ProcessorState::ChunkReady;
                        let duration = self.chunk_buffer.len() as f32 / self.config.output_sample_rate as f32;
                        self.emit_event(StreamingEvent::ChunkReady { duration_secs: duration });
                    } else {
                        // Too short, discard and wait for more speech
                        self.state = ProcessorState::WaitingForSpeech;
                        self.chunk_buffer.clear();
                        self.last_partial_position = 0;
                    }
                }
            }
            ProcessorState::PartialResultReady => {
                // Continue accumulating while partial is being processed
                self.chunk_buffer.extend_from_slice(samples);

                // Check if full chunk is now ready
                if self.chunk_buffer.len() >= self.config.chunk_samples() {
                    self.state = ProcessorState::ChunkReady;
                    let duration = self.chunk_buffer.len() as f32 / self.config.output_sample_rate as f32;
                    self.emit_event(StreamingEvent::ChunkReady { duration_secs: duration });
                }
            }
            ProcessorState::ChunkReady => {
                // Waiting for chunk to be consumed
            }
            ProcessorState::Processing => {
                // Waiting for processing to complete
                // Audio is being buffered in input_buffer during this time
            }
            ProcessorState::Error => {
                // Waiting for error recovery
            }
        }

        // Suppress unused variable warnings
        let _ = prev_speech_frames;
        let _ = prev_silence_frames;
    }

    /// Get minimum speech frames to trigger accumulation
    fn min_speech_frames(&self) -> u32 {
        let frame_duration_ms = 30;
        self.config.min_speech_duration_ms / frame_duration_ms
    }

    /// Get maximum silence frames before ending chunk
    fn max_silence_frames(&self) -> u32 {
        // 1 second of silence
        let frame_duration_ms = 30;
        1000 / frame_duration_ms
    }

    /// Get the accumulated chunk for inference
    ///
    /// Returns None if no chunk is ready. After calling this, the processor
    /// resets to wait for the next utterance.
    pub fn get_chunk(&mut self) -> Option<Vec<f32>> {
        if !self.has_chunk() {
            return None;
        }

        // Save overlap for next chunk
        let overlap_size = self.config.overlap_samples();
        if self.chunk_buffer.len() > overlap_size {
            let start = self.chunk_buffer.len() - overlap_size;
            self.overlap_buffer = self.chunk_buffer[start..].to_vec();
        }

        // Pad to full chunk size if needed
        let target_size = self.config.chunk_samples();
        if self.chunk_buffer.len() < target_size {
            self.chunk_buffer.resize(target_size, 0.0);
        }

        // Take the chunk
        let chunk = core::mem::take(&mut self.chunk_buffer);
        self.prev_state = self.state;
        self.state = ProcessorState::WaitingForSpeech;
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.last_partial_position = 0;

        Some(chunk)
    }

    /// Force flush any accumulated audio as a chunk
    ///
    /// Useful for end-of-stream processing
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        if self.chunk_buffer.is_empty() {
            return None;
        }

        // Process any remaining buffered audio
        self.process();

        // Pad to minimum size (overlap * 2)
        let min_size = self.config.overlap_samples() * 2;
        if self.chunk_buffer.len() < min_size {
            return None;
        }

        // Pad to full chunk size
        let target_size = self.config.chunk_samples();
        self.chunk_buffer.resize(target_size, 0.0);

        let chunk = core::mem::take(&mut self.chunk_buffer);
        self.prev_state = self.state;
        self.state = ProcessorState::WaitingForSpeech;
        self.overlap_buffer.clear();
        self.last_partial_position = 0;

        Some(chunk)
    }

    /// Reset the processor state
    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.chunk_buffer.clear();
        self.overlap_buffer.clear();
        self.prev_state = self.state;
        self.state = ProcessorState::WaitingForSpeech;
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.last_partial_position = 0;
        self.emit_event(StreamingEvent::Reset);
    }

    /// Get statistics about the processor
    #[must_use]
    pub fn stats(&self) -> ProcessorStats {
        ProcessorStats {
            samples_processed: self.samples_processed,
            buffer_available: self.input_buffer.available_read(),
            buffer_capacity: self.input_buffer.capacity(),
            chunk_progress: self.chunk_progress(),
            state: self.state,
        }
    }
}

/// Statistics about the streaming processor
#[derive(Debug, Clone)]
pub struct ProcessorStats {
    /// Total samples processed
    pub samples_processed: u64,
    /// Samples available in ring buffer
    pub buffer_available: usize,
    /// Ring buffer capacity
    pub buffer_capacity: usize,
    /// Current chunk progress (0.0-1.0)
    pub chunk_progress: f32,
    /// Current state
    pub state: ProcessorState,
}

impl ProcessorStats {
    /// Get buffer fill percentage
    #[must_use]
    pub fn buffer_fill(&self) -> f32 {
        if self.buffer_capacity == 0 {
            0.0
        } else {
            self.buffer_available as f32 / self.buffer_capacity as f32 * 100.0
        }
    }

    /// Get total duration processed in seconds
    #[must_use]
    pub fn duration_processed(&self, sample_rate: u32) -> f32 {
        self.samples_processed as f32 / sample_rate as f32
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.input_sample_rate, 44100);
        assert_eq!(config.output_sample_rate, 16000);
        assert!((config.chunk_duration - 30.0).abs() < f32::EPSILON);
        assert!(config.enable_vad);
    }

    #[test]
    fn test_config_with_sample_rate() {
        let config = StreamingConfig::with_sample_rate(48000);
        assert_eq!(config.input_sample_rate, 48000);
    }

    #[test]
    fn test_config_without_vad() {
        let config = StreamingConfig::default().without_vad();
        assert!(!config.enable_vad);
    }

    #[test]
    fn test_config_chunk_samples() {
        let config = StreamingConfig::default();
        // 30s * 16000 = 480000 samples
        assert_eq!(config.chunk_samples(), 480000);
    }

    #[test]
    fn test_config_overlap_samples() {
        let config = StreamingConfig::default();
        // 1s * 16000 = 16000 samples
        assert_eq!(config.overlap_samples(), 16000);
    }

    // =========================================================================
    // Processor Creation Tests
    // =========================================================================

    #[test]
    fn test_processor_new() {
        let processor = StreamingProcessor::new(StreamingConfig::default());
        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
        assert_eq!(processor.samples_processed(), 0);
        assert_eq!(processor.chunk_len(), 0);
    }

    #[test]
    fn test_processor_with_sample_rate() {
        let processor = StreamingProcessor::with_sample_rate(48000);
        assert_eq!(processor.config.input_sample_rate, 48000);
    }

    #[test]
    fn test_processor_same_sample_rate() {
        // When input == output rate, no resampler needed
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            ..Default::default()
        };
        let processor = StreamingProcessor::new(config);
        assert!(processor.resampler.is_none());
    }

    #[test]
    fn test_processor_different_sample_rate() {
        // When input != output rate, resampler is created
        let config = StreamingConfig::default(); // 44100 -> 16000
        let processor = StreamingProcessor::new(config);
        assert!(processor.resampler.is_some());
    }

    // =========================================================================
    // Audio Push Tests
    // =========================================================================

    #[test]
    fn test_push_audio() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        let samples = vec![0.0; 1000];
        processor.push_audio(&samples);
        assert_eq!(processor.samples_processed(), 1000);
    }

    #[test]
    fn test_push_audio_multiple() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.push_audio(&vec![0.0; 500]);
        processor.push_audio(&vec![0.0; 500]);
        assert_eq!(processor.samples_processed(), 1000);
    }

    // =========================================================================
    // State Tests
    // =========================================================================

    #[test]
    fn test_initial_state() {
        let processor = StreamingProcessor::new(StreamingConfig::default());
        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
        assert!(!processor.has_chunk());
    }

    #[test]
    fn test_chunk_progress_empty() {
        let processor = StreamingProcessor::new(StreamingConfig::default());
        assert!((processor.chunk_progress() - 0.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Reset Tests
    // =========================================================================

    #[test]
    fn test_reset() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.push_audio(&vec![0.1; 10000]);
        processor.reset();
        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
        assert_eq!(processor.chunk_len(), 0);
    }

    // =========================================================================
    // Stats Tests
    // =========================================================================

    #[test]
    fn test_stats() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.push_audio(&vec![0.0; 1000]);
        let stats = processor.stats();
        assert_eq!(stats.samples_processed, 1000);
        assert_eq!(stats.state, ProcessorState::WaitingForSpeech);
    }

    #[test]
    fn test_stats_buffer_fill() {
        let stats = ProcessorStats {
            samples_processed: 0,
            buffer_available: 500,
            buffer_capacity: 1000,
            chunk_progress: 0.0,
            state: ProcessorState::WaitingForSpeech,
        };
        assert!((stats.buffer_fill() - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stats_duration_processed() {
        let stats = ProcessorStats {
            samples_processed: 16000,
            buffer_available: 0,
            buffer_capacity: 1000,
            chunk_progress: 0.0,
            state: ProcessorState::WaitingForSpeech,
        };
        assert!((stats.duration_processed(16000) - 1.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // VAD Integration Tests
    // =========================================================================

    #[test]
    fn test_vad_disabled() {
        let config = StreamingConfig::default().without_vad();
        let processor = StreamingProcessor::new(config);
        assert!(!processor.config.enable_vad);
    }

    // =========================================================================
    // Flush Tests
    // =========================================================================

    #[test]
    fn test_flush_empty() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        assert!(processor.flush().is_none());
    }

    // =========================================================================
    // Get Chunk Tests
    // =========================================================================

    #[test]
    fn test_get_chunk_not_ready() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        assert!(processor.get_chunk().is_none());
    }

    // =========================================================================
    // Processing Tests
    // =========================================================================

    #[test]
    fn test_process_silence() {
        // Use same sample rate to avoid resampling complexity
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            enable_vad: true,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Push silence and process
        let silence = vec![0.0; 4800]; // 300ms at 16kHz
        processor.push_audio(&silence);
        processor.process();

        // Should stay in waiting state for silence
        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
    }

    #[test]
    fn test_process_with_vad_disabled() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            enable_vad: false,
            chunk_duration: 0.5, // Short chunk for testing
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // With VAD disabled, all audio is treated as speech
        let audio = vec![0.1; 8000]; // 500ms at 16kHz
        processor.push_audio(&audio);
        processor.process();

        // Should have accumulated some audio
        assert!(processor.chunk_len() > 0 || processor.has_chunk());
    }

    #[test]
    fn test_process_empty_buffer() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());

        // Process without pushing any audio
        processor.process();

        // Should stay in initial state
        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
        assert_eq!(processor.chunk_len(), 0);
    }

    #[test]
    fn test_config_vad_threshold() {
        let config = StreamingConfig::default().vad_threshold(0.8);
        assert!((config.vad_threshold - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_chunk_duration() {
        let config = StreamingConfig::default().chunk_duration(10.0);
        assert!((config.chunk_duration - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stats_buffer_fill_zero_capacity() {
        let stats = ProcessorStats {
            samples_processed: 0,
            buffer_available: 0,
            buffer_capacity: 0,
            chunk_progress: 0.0,
            state: ProcessorState::WaitingForSpeech,
        };
        assert!((stats.buffer_fill() - 0.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // State Machine Tests
    // =========================================================================

    #[test]
    fn test_update_state_accumulating() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            enable_vad: false,
            chunk_duration: 0.1,       // Very short chunk (1600 samples)
            min_speech_duration_ms: 0, // Immediate speech detection
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Manually trigger state transition
        let samples = vec![0.1; 320]; // 20ms frame
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.update_state(true, &samples);

        assert_eq!(processor.state(), ProcessorState::AccumulatingSpeech);
        assert_eq!(processor.chunk_buffer.len(), 320);
    }

    #[test]
    fn test_update_state_chunk_ready() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            enable_vad: false,
            chunk_duration: 0.02, // Very short chunk (320 samples)
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Fill chunk buffer to trigger ready state
        processor.chunk_buffer = vec![0.1; 320];
        processor.state = ProcessorState::AccumulatingSpeech;

        let samples = vec![0.1; 320];
        processor.update_state(true, &samples);

        // Should transition to ChunkReady
        assert_eq!(processor.state(), ProcessorState::ChunkReady);
    }

    #[test]
    fn test_update_state_silence_ends_speech() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 30.0,
            chunk_overlap: 0.1, // Small overlap
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Set up as if we're accumulating speech
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.chunk_buffer = vec![0.1; 5000]; // Some accumulated audio
        processor.silence_frames = 50; // Extended silence

        let samples = vec![0.0; 320];
        processor.update_state(false, &samples);

        // Should transition to ChunkReady due to extended silence
        assert_eq!(processor.state(), ProcessorState::ChunkReady);
    }

    #[test]
    fn test_update_state_short_segment_discarded() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 30.0,
            chunk_overlap: 1.0, // 16000 sample overlap
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Set up as if we're accumulating speech with very little audio
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.chunk_buffer = vec![0.1; 100]; // Very short
        processor.silence_frames = 50; // Extended silence

        let samples = vec![0.0; 320];
        processor.update_state(false, &samples);

        // Should go back to waiting (too short to be a valid chunk)
        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
        assert!(processor.chunk_buffer.is_empty());
    }

    #[test]
    fn test_update_state_chunk_ready_stays() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.state = ProcessorState::ChunkReady;

        let samples = vec![0.1; 320];
        processor.update_state(true, &samples);

        // Should stay in ChunkReady until consumed
        assert_eq!(processor.state(), ProcessorState::ChunkReady);
    }

    // =========================================================================
    // Get Chunk Tests
    // =========================================================================

    #[test]
    fn test_get_chunk_ready() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.02, // 320 samples
            chunk_overlap: 0.01,  // 160 samples
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Fill chunk buffer to be ready (larger than chunk size)
        processor.chunk_buffer = vec![0.1; 400];
        processor.state = ProcessorState::ChunkReady;

        let chunk = processor.get_chunk();
        assert!(chunk.is_some());

        let chunk = chunk.expect("chunk should exist");
        // get_chunk takes the whole buffer (doesn't trim)
        assert_eq!(chunk.len(), 400);

        // Should reset state
        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
    }

    #[test]
    fn test_get_chunk_saves_overlap() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.02, // 320 samples
            chunk_overlap: 0.01,  // 160 samples
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Fill chunk buffer
        processor.chunk_buffer = vec![0.1; 400];
        processor.state = ProcessorState::ChunkReady;

        let _ = processor.get_chunk();

        // Should have saved overlap
        assert_eq!(processor.overlap_buffer.len(), 160);
    }

    #[test]
    fn test_get_chunk_pads_short_chunk() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.02, // 320 samples
            chunk_overlap: 0.0,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Fill with less than full chunk
        processor.chunk_buffer = vec![0.1; 200];
        processor.state = ProcessorState::ChunkReady;

        let chunk = processor.get_chunk().expect("should get chunk");

        // Should be padded to full size
        assert_eq!(chunk.len(), 320);
    }

    // =========================================================================
    // Flush Tests
    // =========================================================================

    #[test]
    fn test_flush_with_accumulated_data() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.1, // 1600 samples
            chunk_overlap: 0.01, // 160 samples
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Accumulate some data
        processor.chunk_buffer = vec![0.1; 500]; // More than overlap * 2
        processor.state = ProcessorState::AccumulatingSpeech;

        let flushed = processor.flush();
        assert!(flushed.is_some());

        let flushed = flushed.expect("should flush");
        assert_eq!(flushed.len(), 1600); // Padded to full chunk

        // Should reset state
        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
        assert!(processor.overlap_buffer.is_empty());
    }

    #[test]
    fn test_flush_too_short() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.1,
            chunk_overlap: 0.05, // 800 samples, so need > 1600
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Too little data
        processor.chunk_buffer = vec![0.1; 100];

        let flushed = processor.flush();
        assert!(flushed.is_none());
    }

    // =========================================================================
    // Min/Max Frame Tests
    // =========================================================================

    #[test]
    fn test_min_speech_frames() {
        let config = StreamingConfig {
            min_speech_duration_ms: 300,
            ..Default::default()
        };
        let processor = StreamingProcessor::new(config);
        // 300ms / 30ms per frame = 10 frames
        assert_eq!(processor.min_speech_frames(), 10);
    }

    #[test]
    fn test_max_silence_frames() {
        let processor = StreamingProcessor::new(StreamingConfig::default());
        // 1000ms / 30ms per frame = 33 frames
        assert_eq!(processor.max_silence_frames(), 33);
    }

    // =========================================================================
    // Waiting for Speech State Tests
    // =========================================================================

    #[test]
    fn test_waiting_transitions_to_accumulating() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            min_speech_duration_ms: 0, // Immediate
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Pre-populate overlap buffer
        processor.overlap_buffer = vec![0.05; 100];
        processor.speech_frames = 1; // Already detected speech

        let samples = vec![0.1; 320];
        processor.update_state(true, &samples);

        // Should transition to AccumulatingSpeech
        assert_eq!(processor.state(), ProcessorState::AccumulatingSpeech);
        // Should include overlap + new samples
        assert!(processor.chunk_buffer.len() >= 320);
    }

    // =========================================================================
    // WAPR-100: Enhanced State Machine Tests
    // =========================================================================

    #[test]
    fn test_streaming_event_variants() {
        // Test all event variants can be created
        let speech_start = StreamingEvent::SpeechStart;
        let speech_end = StreamingEvent::SpeechEnd;
        let partial_ready = StreamingEvent::PartialReady {
            accumulated_samples: 48000,
            duration_secs: 3.0,
        };
        let chunk_ready = StreamingEvent::ChunkReady { duration_secs: 30.0 };
        let processing_started = StreamingEvent::ProcessingStarted;
        let processing_completed = StreamingEvent::ProcessingCompleted;
        let error = StreamingEvent::Error("test error".to_string());
        let reset = StreamingEvent::Reset;

        // Test Debug and Clone
        assert!(format!("{speech_start:?}").contains("SpeechStart"));
        assert!(format!("{speech_end:?}").contains("SpeechEnd"));
        assert!(format!("{partial_ready:?}").contains("PartialReady"));
        assert!(format!("{chunk_ready:?}").contains("ChunkReady"));
        assert!(format!("{processing_started:?}").contains("ProcessingStarted"));
        assert!(format!("{processing_completed:?}").contains("ProcessingCompleted"));
        assert!(format!("{error:?}").contains("Error"));
        assert!(format!("{reset:?}").contains("Reset"));

        // Test Clone
        let cloned = speech_start.clone();
        assert_eq!(cloned, StreamingEvent::SpeechStart);
    }

    #[test]
    fn test_processor_state_new_variants() {
        // Test new state variants
        let partial_ready = ProcessorState::PartialResultReady;
        let processing = ProcessorState::Processing;
        let error = ProcessorState::Error;

        assert!(format!("{partial_ready:?}").contains("PartialResultReady"));
        assert!(format!("{processing:?}").contains("Processing"));
        assert!(format!("{error:?}").contains("Error"));

        // Test equality
        assert_ne!(ProcessorState::PartialResultReady, ProcessorState::Processing);
        assert_ne!(ProcessorState::Processing, ProcessorState::Error);
    }

    #[test]
    fn test_event_handling_initial() {
        let processor = StreamingProcessor::new(StreamingConfig::default());
        assert!(!processor.has_events());
        assert_eq!(processor.event_count(), 0);
        assert!(processor.peek_event().is_none());
    }

    #[test]
    fn test_event_pop_and_drain() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());

        // Manually add events for testing
        processor.events.push(StreamingEvent::SpeechStart);
        processor.events.push(StreamingEvent::SpeechEnd);

        assert!(processor.has_events());
        assert_eq!(processor.event_count(), 2);

        // Pop first event
        let event = processor.pop_event();
        assert!(event.is_some());
        assert_eq!(event, Some(StreamingEvent::SpeechStart));
        assert_eq!(processor.event_count(), 1);

        // Drain remaining
        let remaining = processor.drain_events();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0], StreamingEvent::SpeechEnd);
        assert!(!processor.has_events());
    }

    #[test]
    fn test_event_peek() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.events.push(StreamingEvent::Reset);

        let peeked = processor.peek_event();
        assert!(peeked.is_some());
        assert_eq!(peeked, Some(&StreamingEvent::Reset));

        // Peeking doesn't consume
        assert_eq!(processor.event_count(), 1);
    }

    #[test]
    fn test_clear_events() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.events.push(StreamingEvent::SpeechStart);
        processor.events.push(StreamingEvent::SpeechEnd);

        processor.clear_events();
        assert!(!processor.has_events());
        assert_eq!(processor.event_count(), 0);
    }

    #[test]
    fn test_partial_result_threshold() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());

        // Default threshold is 3 seconds = 48000 samples at 16kHz
        assert!((processor.partial_threshold() - 3.0).abs() < 0.01);

        // Change threshold
        processor.set_partial_threshold(5.0);
        assert!((processor.partial_threshold() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_partial_duration() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Empty buffer
        assert!((processor.partial_duration() - 0.0).abs() < 0.01);

        // Add some samples
        processor.chunk_buffer = vec![0.1; 16000]; // 1 second
        assert!((processor.partial_duration() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_has_partial_not_accumulating() {
        let processor = StreamingProcessor::new(StreamingConfig::default());
        assert!(!processor.has_partial()); // Not accumulating
    }

    #[test]
    fn test_has_partial_below_threshold() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.chunk_buffer = vec![0.1; 16000]; // 1 second, below 3s threshold

        assert!(!processor.has_partial());
    }

    #[test]
    fn test_has_partial_above_threshold() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.chunk_buffer = vec![0.1; 64000]; // 4 seconds, above 3s threshold

        assert!(processor.has_partial());
    }

    #[test]
    fn test_get_partial() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.chunk_buffer = vec![0.1; 64000]; // 4 seconds

        let partial = processor.get_partial();
        assert!(partial.is_some());
        assert_eq!(partial.as_ref().map(|p| p.len()), Some(64000));

        // After getting partial, last_partial_position is updated
        assert_eq!(processor.last_partial_position, 64000);

        // Getting again without more audio returns None (already processed this position)
        assert!(!processor.has_partial());
    }

    #[test]
    fn test_processing_state_transitions() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.state = ProcessorState::ChunkReady;

        // Mark processing started
        processor.mark_processing_started();
        assert_eq!(processor.state(), ProcessorState::Processing);
        assert!(processor.has_events());

        // Should have emitted ProcessingStarted event
        let event = processor.pop_event();
        assert_eq!(event, Some(StreamingEvent::ProcessingStarted));

        // Mark processing completed
        processor.mark_processing_completed();
        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);

        let event = processor.pop_event();
        assert_eq!(event, Some(StreamingEvent::ProcessingCompleted));
    }

    #[test]
    fn test_error_state() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());

        processor.mark_error("Test error message");
        assert_eq!(processor.state(), ProcessorState::Error);

        let event = processor.pop_event();
        assert!(matches!(event, Some(StreamingEvent::Error(msg)) if msg == "Test error message"));
    }

    #[test]
    fn test_error_recovery() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.state = ProcessorState::Error;
        processor.chunk_buffer = vec![0.1; 1000];
        processor.last_partial_position = 500;

        processor.recover_from_error();

        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
        assert!(processor.chunk_buffer.is_empty());
        assert_eq!(processor.last_partial_position, 0);
    }

    #[test]
    fn test_prev_state_tracking() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            min_speech_duration_ms: 0,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        assert_eq!(processor.prev_state(), ProcessorState::WaitingForSpeech);

        // Transition to accumulating
        processor.speech_frames = 1;
        processor.update_state(true, &vec![0.1; 320]);

        // Previous state should be WaitingForSpeech
        assert_eq!(processor.prev_state(), ProcessorState::WaitingForSpeech);
        assert_eq!(processor.state(), ProcessorState::AccumulatingSpeech);
    }

    #[test]
    fn test_speech_start_event() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            min_speech_duration_ms: 0, // Immediate
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);
        processor.speech_frames = 1;

        processor.update_state(true, &vec![0.1; 320]);

        // Should have emitted SpeechStart event
        assert!(processor.has_events());
        let event = processor.pop_event();
        assert_eq!(event, Some(StreamingEvent::SpeechStart));
    }

    #[test]
    fn test_speech_end_and_chunk_ready_events() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 30.0,
            chunk_overlap: 0.1,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Set up as accumulating with enough audio
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.chunk_buffer = vec![0.1; 5000];
        processor.silence_frames = 50; // Extended silence

        processor.update_state(false, &vec![0.0; 320]);

        // Should have SpeechEnd then ChunkReady events
        let events = processor.drain_events();
        assert!(events.iter().any(|e| *e == StreamingEvent::SpeechEnd));
        assert!(events.iter().any(|e| matches!(e, StreamingEvent::ChunkReady { .. })));
    }

    #[test]
    fn test_reset_emits_event() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.chunk_buffer = vec![0.1; 1000];

        processor.reset();

        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
        assert!(processor.chunk_buffer.is_empty());

        // Should have emitted Reset event
        let event = processor.pop_event();
        assert_eq!(event, Some(StreamingEvent::Reset));
    }

    #[test]
    fn test_partial_ready_event_on_threshold() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            min_speech_duration_ms: 0,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);
        processor.set_partial_threshold(0.1); // 1600 samples

        // Transition to accumulating
        processor.speech_frames = 1;
        processor.update_state(true, &vec![0.1; 320]);
        processor.drain_events(); // Clear SpeechStart event

        // Accumulate past threshold
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.chunk_buffer = vec![0.1; 1500]; // Just below threshold

        processor.update_state(true, &vec![0.1; 320]);

        // Should have emitted PartialReady event
        let events = processor.drain_events();
        assert!(events.iter().any(|e| matches!(e, StreamingEvent::PartialReady { .. })));
    }

    #[test]
    fn test_processing_state_ignores_transitions() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.state = ProcessorState::Processing;
        processor.chunk_buffer = vec![0.1; 1000];

        // Calling update_state while processing should not change state
        processor.update_state(true, &vec![0.1; 320]);

        assert_eq!(processor.state(), ProcessorState::Processing);
    }

    #[test]
    fn test_error_state_ignores_transitions() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.state = ProcessorState::Error;

        processor.update_state(true, &vec![0.1; 320]);

        assert_eq!(processor.state(), ProcessorState::Error);
    }

    #[test]
    fn test_partial_result_ready_continues_accumulating() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.1, // 1600 samples for full chunk
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);
        processor.state = ProcessorState::PartialResultReady;
        processor.chunk_buffer = vec![0.1; 1000];

        processor.update_state(true, &vec![0.1; 320]);

        // Should have accumulated more samples
        assert_eq!(processor.chunk_buffer.len(), 1320);
    }

    #[test]
    fn test_partial_to_chunk_ready_transition() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.1, // 1600 samples
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);
        processor.state = ProcessorState::PartialResultReady;
        processor.chunk_buffer = vec![0.1; 1500]; // Close to full

        processor.update_state(true, &vec![0.1; 320]);

        // Should have transitioned to ChunkReady
        assert_eq!(processor.state(), ProcessorState::ChunkReady);
    }

    #[test]
    fn test_get_chunk_resets_partial_position() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.02,
            chunk_overlap: 0.01,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);
        processor.state = ProcessorState::ChunkReady;
        processor.chunk_buffer = vec![0.1; 400];
        processor.last_partial_position = 200;

        let _ = processor.get_chunk();

        // Should have reset partial position
        assert_eq!(processor.last_partial_position, 0);
    }

    #[test]
    fn test_flush_resets_partial_position() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.1,
            chunk_overlap: 0.01,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.chunk_buffer = vec![0.1; 500];
        processor.last_partial_position = 100;

        let _ = processor.flush();

        // Should have reset partial position
        assert_eq!(processor.last_partial_position, 0);
    }

    #[test]
    fn test_default_partial_threshold_constant() {
        // Verify the constant is set correctly
        assert!((DEFAULT_PARTIAL_THRESHOLD_SECS - 3.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // WAPR-102: Chunk Overlap Handling Tests
    // =========================================================================

    #[test]
    fn test_config_chunk_overlap_builder() {
        let config = StreamingConfig::default().chunk_overlap(0.5);
        assert!((config.chunk_overlap - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_config_min_speech_duration_builder() {
        let config = StreamingConfig::default().min_speech_duration_ms(500);
        assert_eq!(config.min_speech_duration_ms, 500);
    }

    #[test]
    fn test_overlap_len_initial() {
        let processor = StreamingProcessor::new(StreamingConfig::default());
        assert_eq!(processor.overlap_len(), 0);
    }

    #[test]
    fn test_overlap_duration_empty() {
        let processor = StreamingProcessor::new(StreamingConfig::default());
        assert!((processor.overlap_duration() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_has_overlap_initial() {
        let processor = StreamingProcessor::new(StreamingConfig::default());
        assert!(!processor.has_overlap());
    }

    #[test]
    fn test_configured_overlap_samples() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_overlap: 0.5, // 0.5 seconds
            ..Default::default()
        };
        let processor = StreamingProcessor::new(config);
        assert_eq!(processor.configured_overlap_samples(), 8000); // 0.5 * 16000
    }

    #[test]
    fn test_configured_overlap_duration() {
        let config = StreamingConfig {
            chunk_overlap: 0.75,
            ..Default::default()
        };
        let processor = StreamingProcessor::new(config);
        assert!((processor.configured_overlap_duration() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_clear_overlap() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.overlap_buffer = vec![0.1; 1000];

        processor.clear_overlap();

        assert!(!processor.has_overlap());
        assert_eq!(processor.overlap_len(), 0);
    }

    #[test]
    fn test_get_overlap_buffer() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.overlap_buffer = vec![0.5; 100];

        let overlap = processor.get_overlap_buffer();
        assert_eq!(overlap.len(), 100);
        assert!((overlap[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_set_overlap_buffer() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        let custom_overlap = vec![0.25; 200];

        processor.set_overlap_buffer(custom_overlap.clone());

        assert!(processor.has_overlap());
        assert_eq!(processor.overlap_len(), 200);
        let buffer = processor.get_overlap_buffer();
        assert!((buffer[0] - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_overlap_preserved_after_get_chunk() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.02, // 320 samples
            chunk_overlap: 0.01,  // 160 samples overlap
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Fill chunk buffer with recognizable pattern
        processor.chunk_buffer = (0..400).map(|i| i as f32 * 0.001).collect();
        processor.state = ProcessorState::ChunkReady;

        let _ = processor.get_chunk();

        // Overlap buffer should have last 160 samples
        assert!(processor.has_overlap());
        assert_eq!(processor.overlap_len(), 160);

        // Verify the overlap is from the end of the chunk
        let overlap = processor.get_overlap_buffer();
        // First sample of overlap should be sample 240 from original (400 - 160)
        assert!((overlap[0] - 0.240).abs() < 0.001);
    }

    #[test]
    fn test_overlap_used_when_starting_accumulation() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            min_speech_duration_ms: 0, // Immediate
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Pre-populate overlap buffer
        processor.overlap_buffer = vec![0.05; 100];
        processor.speech_frames = 1;

        let samples = vec![0.1; 320];
        processor.update_state(true, &samples);

        // Chunk buffer should include overlap + new samples
        assert!(processor.chunk_buffer.len() >= 420);
        // First 100 samples should be from overlap (0.05)
        assert!((processor.chunk_buffer[0] - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_overlap_duration_with_samples() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);
        processor.overlap_buffer = vec![0.1; 8000]; // 0.5 seconds

        assert!((processor.overlap_duration() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_overlap_reset_clears_buffer() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.overlap_buffer = vec![0.1; 1000];

        processor.reset();

        assert!(!processor.has_overlap());
    }

    #[test]
    fn test_flush_clears_overlap() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.1,
            chunk_overlap: 0.01,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);
        processor.state = ProcessorState::AccumulatingSpeech;
        processor.chunk_buffer = vec![0.1; 500];
        processor.overlap_buffer = vec![0.2; 100];

        let _ = processor.flush();

        assert!(!processor.has_overlap());
    }

    #[test]
    fn test_multiple_chunks_preserve_overlap_chain() {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            chunk_duration: 0.02,  // 320 samples
            chunk_overlap: 0.005, // 80 samples overlap
            min_speech_duration_ms: 0, // Immediate transition
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // First chunk
        processor.chunk_buffer = vec![0.1; 400];
        processor.state = ProcessorState::ChunkReady;
        let _ = processor.get_chunk();

        // Verify overlap preserved
        let first_overlap = processor.get_overlap_buffer();
        assert_eq!(first_overlap.len(), 80);

        // Second chunk - simulate accumulation that uses overlap
        // Set speech_frames high enough to trigger transition
        processor.speech_frames = 1;
        processor.state = ProcessorState::WaitingForSpeech;
        processor.update_state(true, &vec![0.2; 320]);

        // After transition, chunk buffer should include overlap + new samples
        assert!(processor.chunk_buffer.len() >= 320);
        // Verify overlap was prepended (first 80 samples should be 0.1)
        assert!((processor.chunk_buffer[0] - 0.1).abs() < 0.01);
    }
}

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
    /// Chunk ready for processing
    ChunkReady,
}

/// Streaming audio processor for real-time transcription
///
/// This processor handles:
/// 1. Buffering incoming audio via lock-free ring buffer
/// 2. Resampling from native rate to 16kHz
/// 3. Voice activity detection to skip silence
/// 4. Accumulating speech into 30s chunks for inference
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
    /// Consecutive speech frames count
    speech_frames: u32,
    /// Consecutive silence frames count
    silence_frames: u32,
    /// Total samples processed
    samples_processed: u64,
}

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

        Self {
            config,
            input_buffer,
            resampler,
            vad,
            chunk_buffer: Vec::with_capacity(chunk_capacity),
            overlap_buffer: Vec::new(),
            state: ProcessorState::WaitingForSpeech,
            speech_frames: 0,
            silence_frames: 0,
            samples_processed: 0,
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
        if is_speech {
            self.speech_frames += 1;
            self.silence_frames = 0;
        } else {
            self.silence_frames += 1;
            self.speech_frames = 0;
        }

        // State machine
        match self.state {
            ProcessorState::WaitingForSpeech => {
                if is_speech && self.speech_frames >= self.min_speech_frames() {
                    self.state = ProcessorState::AccumulatingSpeech;
                    // Start with overlap from previous chunk
                    self.chunk_buffer.clear();
                    self.chunk_buffer.extend(&self.overlap_buffer);
                    self.chunk_buffer.extend_from_slice(samples);
                }
            }
            ProcessorState::AccumulatingSpeech => {
                self.chunk_buffer.extend_from_slice(samples);

                // Check if chunk is complete
                if self.chunk_buffer.len() >= self.config.chunk_samples() {
                    self.state = ProcessorState::ChunkReady;
                }
                // Or if we hit extended silence (end of utterance)
                else if !is_speech && self.silence_frames >= self.max_silence_frames() {
                    // Partial chunk is ready
                    if self.chunk_buffer.len() >= self.config.overlap_samples() * 2 {
                        self.state = ProcessorState::ChunkReady;
                    } else {
                        // Too short, discard and wait for more speech
                        self.state = ProcessorState::WaitingForSpeech;
                        self.chunk_buffer.clear();
                    }
                }
            }
            ProcessorState::ChunkReady => {
                // Waiting for chunk to be consumed
            }
        }
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
        self.state = ProcessorState::WaitingForSpeech;
        self.speech_frames = 0;
        self.silence_frames = 0;

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
        self.state = ProcessorState::WaitingForSpeech;
        self.overlap_buffer.clear();

        Some(chunk)
    }

    /// Reset the processor state
    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.chunk_buffer.clear();
        self.overlap_buffer.clear();
        self.state = ProcessorState::WaitingForSpeech;
        self.speech_frames = 0;
        self.silence_frames = 0;
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
}

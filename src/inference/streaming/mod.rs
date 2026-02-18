//! Streaming inference for real-time transcription
//!
//! Provides real-time transcription with partial results as audio is processed.
//!
//! # Architecture
//!
//! ```text
//! Audio Input --> StreamingProcessor --> StreamingTranscriber --> Partial Results
//!   (RT)            (chunks)               (inference)           (callbacks)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use whisper_apr::inference::{StreamingTranscriber, StreamingConfig};
//!
//! let mut transcriber = StreamingTranscriber::new(model, config);
//!
//! // Feed audio samples as they arrive
//! for samples in audio_stream {
//!     transcriber.push_audio(&samples);
//!
//!     // Process and get any available results
//!     if let Some(result) = transcriber.process()? {
//!         println!("Partial: {}", result.text);
//!     }
//! }
//!
//! // Get final result
//! let final_result = transcriber.finalize()?;
//! ```

#[cfg(test)]
mod tests;

use crate::audio::{
    MelFilterbank, ProcessorState, StreamingConfig as AudioStreamingConfig, StreamingProcessor,
};
use crate::error::{WhisperError, WhisperResult};
use crate::TranscriptionResult;

/// Configuration for streaming transcription
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Audio streaming configuration
    pub audio: AudioStreamingConfig,
    /// Maximum tokens to generate per chunk
    pub max_tokens_per_chunk: usize,
    /// Overlap tokens between chunks for continuity
    pub overlap_tokens: usize,
    /// Temperature for decoding
    pub temperature: f32,
    /// Whether to return partial results
    pub return_partial: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            audio: AudioStreamingConfig::default(),
            max_tokens_per_chunk: 224, // Half of Whisper's max for faster streaming
            overlap_tokens: 10,
            temperature: 0.0,
            return_partial: true,
        }
    }
}

impl StreamingConfig {
    /// Create config for a specific input sample rate
    #[must_use]
    pub fn with_sample_rate(sample_rate: u32) -> Self {
        Self {
            audio: AudioStreamingConfig::with_sample_rate(sample_rate),
            ..Default::default()
        }
    }

    /// Disable VAD filtering (process all audio)
    #[must_use]
    pub fn without_vad(mut self) -> Self {
        self.audio = self.audio.without_vad();
        self
    }

    /// Set whether to return partial results
    #[must_use]
    pub const fn with_partial_results(mut self, enable: bool) -> Self {
        self.return_partial = enable;
        self
    }
}

/// Result from a streaming chunk
#[derive(Debug, Clone)]
pub struct StreamingResult {
    /// Transcribed text for this chunk
    pub text: String,
    /// Whether this is a final result (vs partial)
    pub is_final: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Chunk index (sequential)
    pub chunk_index: usize,
    /// Estimated latency in milliseconds
    pub latency_ms: u32,
}

/// Streaming transcriber state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriberState {
    /// Ready for audio input
    Ready,
    /// Processing a chunk
    Processing,
    /// Finalized (call reset to start new session)
    Finalized,
}

/// Streaming transcriber for real-time speech recognition
///
/// Integrates audio streaming with inference to provide low-latency
/// transcription with partial results.
pub struct StreamingTranscriber {
    /// Audio processor
    processor: StreamingProcessor,
    /// Mel filterbank for spectrogram computation
    mel: MelFilterbank,
    /// Configuration
    config: StreamingConfig,
    /// Current state
    state: TranscriberState,
    /// Accumulated text from all chunks
    accumulated_text: String,
    /// Current chunk index
    chunk_index: usize,
    /// Previous chunk's last tokens (for continuity)
    previous_tokens: Vec<u32>,
}

impl StreamingTranscriber {
    /// Create a new streaming transcriber
    #[must_use]
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            processor: StreamingProcessor::new(config.audio.clone()),
            mel: MelFilterbank::new(&aprender::audio::MelConfig::whisper()),
            config,
            state: TranscriberState::Ready,
            accumulated_text: String::new(),
            chunk_index: 0,
            previous_tokens: Vec::new(),
        }
    }

    /// Create with a specific sample rate
    #[must_use]
    pub fn with_sample_rate(sample_rate: u32) -> Self {
        Self::new(StreamingConfig::with_sample_rate(sample_rate))
    }

    /// Get current state
    #[must_use]
    pub const fn state(&self) -> TranscriberState {
        self.state
    }

    /// Get accumulated text so far
    #[must_use]
    pub fn text(&self) -> &str {
        &self.accumulated_text
    }

    /// Get current chunk index
    #[must_use]
    pub const fn chunk_index(&self) -> usize {
        self.chunk_index
    }

    /// Get chunk progress (0.0 to 1.0)
    #[must_use]
    pub fn chunk_progress(&self) -> f32 {
        self.processor.chunk_progress()
    }

    /// Push audio samples into the transcriber
    ///
    /// Samples should be at the configured input sample rate.
    pub fn push_audio(&mut self, samples: &[f32]) {
        if self.state == TranscriberState::Finalized {
            return;
        }
        self.processor.push_audio(samples);
    }

    /// Process buffered audio and return any available results
    ///
    /// This should be called regularly (e.g., every 100ms) to process audio
    /// and generate transcription results.
    pub fn process(&mut self) -> WhisperResult<Option<StreamingResult>> {
        if self.state == TranscriberState::Finalized {
            return Ok(None);
        }

        // Process audio through the streaming processor
        self.processor.process();

        // Check if we have a complete chunk
        if self.processor.state() != ProcessorState::ChunkReady {
            // No chunk ready yet
            if self.config.return_partial && self.processor.chunk_progress() > 0.3 {
                // Return partial result if we have significant audio
                return Ok(Some(self.create_partial_result()));
            }
            return Ok(None);
        }

        // Get the audio chunk
        let Some(chunk) = self.processor.get_chunk() else {
            return Ok(None);
        };

        self.state = TranscriberState::Processing;

        // Compute mel spectrogram
        let mel_spec = self.mel.compute(&chunk)
            .map_err(|e| WhisperError::Audio(e.to_string()))?;

        // Run inference on chunk
        let chunk_result = self.transcribe_chunk(&mel_spec)?;

        self.state = TranscriberState::Ready;
        self.chunk_index += 1;

        // Append to accumulated text
        if !chunk_result.text.is_empty() {
            if !self.accumulated_text.is_empty() {
                self.accumulated_text.push(' ');
            }
            self.accumulated_text.push_str(&chunk_result.text);
        }

        Ok(Some(chunk_result))
    }

    /// Create a partial result from current buffer state
    fn create_partial_result(&self) -> StreamingResult {
        StreamingResult {
            text: String::from("[listening...]"),
            is_final: false,
            confidence: 0.0,
            chunk_index: self.chunk_index,
            latency_ms: (self.processor.chunk_progress() * 30000.0) as u32,
        }
    }

    /// Transcribe a mel spectrogram chunk
    #[allow(clippy::unnecessary_wraps)]
    fn transcribe_chunk(&self, mel_spec: &[f32]) -> WhisperResult<StreamingResult> {
        // Placeholder: In full implementation, this would:
        // 1. Run encoder on mel spectrogram
        // 2. Run decoder with KV cache from previous chunk
        // 3. Return transcribed text

        // For now, return a placeholder result
        let _ = mel_spec; // Use mel_spec

        let result = StreamingResult {
            text: String::new(), // Would be filled by actual inference
            is_final: true,
            confidence: 1.0,
            chunk_index: self.chunk_index,
            latency_ms: 0,
        };

        Ok(result)
    }

    /// Flush any remaining audio and finalize transcription
    pub fn finalize(&mut self) -> WhisperResult<TranscriptionResult> {
        // Process any remaining audio in buffer
        if let Some(chunk) = self.processor.flush() {
            if !chunk.is_empty() {
                let mel_spec = self.mel.compute(&chunk)
                    .map_err(|e| WhisperError::Audio(e.to_string()))?;
                let chunk_result = self.transcribe_chunk(&mel_spec)?;
                if !chunk_result.text.is_empty() {
                    if !self.accumulated_text.is_empty() {
                        self.accumulated_text.push(' ');
                    }
                    self.accumulated_text.push_str(&chunk_result.text);
                }
            }
        }

        self.state = TranscriberState::Finalized;

        Ok(TranscriptionResult {
            text: self.accumulated_text.clone(),
            language: "en".into(),
            segments: vec![],
            profiling: None,
        })
    }

    /// Reset the transcriber for a new session
    pub fn reset(&mut self) {
        self.processor = StreamingProcessor::new(self.config.audio.clone());
        self.state = TranscriberState::Ready;
        self.accumulated_text.clear();
        self.chunk_index = 0;
        self.previous_tokens.clear();
    }

    /// Get statistics about the streaming session
    #[must_use]
    pub fn stats(&self) -> StreamingStats {
        let processor_stats = self.processor.stats();
        StreamingStats {
            chunks_processed: self.chunk_index,
            samples_processed: processor_stats.samples_processed,
            buffer_fill: processor_stats.buffer_fill(),
            total_text_length: self.accumulated_text.len(),
        }
    }
}

/// Statistics for streaming transcription
#[derive(Debug, Clone, Copy)]
pub struct StreamingStats {
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Total samples processed
    pub samples_processed: u64,
    /// Current buffer fill percentage
    pub buffer_fill: f32,
    /// Total accumulated text length
    pub total_text_length: usize,
}

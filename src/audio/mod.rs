//! Audio preprocessing module
//!
//! Handles audio loading, resampling, normalization, and mel spectrogram computation.

pub mod batch;
mod mel;
mod resampler;
mod ring_buffer;
mod streaming;

pub use batch::{AudioBatch, BatchMelResult, BatchPreprocessor, split_into_chunks};
pub use mel::MelFilterbank;
pub use resampler::{Resampler, SincResampler};
pub use ring_buffer::RingBuffer;
pub use streaming::{
    ProcessorState, ProcessorStats, StreamingConfig, StreamingEvent, StreamingProcessor,
    DEFAULT_CHUNK_DURATION, DEFAULT_CHUNK_OVERLAP, MIN_SPEECH_DURATION_MS,
};

// Re-export VAD types from root module for audio pipeline integration
pub use crate::vad::{SpeechSegment, StreamingVad, VadConfig, VadEvent, VadState, VoiceActivityDetector};

/// Default FFT size for Whisper (400 samples = 25ms at 16kHz)
pub const N_FFT: usize = 400;

/// Default hop length (160 samples = 10ms at 16kHz)
pub const HOP_LENGTH: usize = 160;

/// Default sample rate (16kHz)
pub const SAMPLE_RATE: u32 = 16000;

/// Audio preprocessing configuration
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Target sample rate (default: 16000 Hz for Whisper)
    pub sample_rate: u32,
    /// Number of mel filterbank channels (default: 80)
    pub n_mels: usize,
    /// FFT size (default: 400)
    pub n_fft: usize,
    /// Hop length between frames (default: 160)
    pub hop_length: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.hop_length, 160);
    }

    // ==========================================================================
    // VAD Integration Tests (WAPR-091)
    // ==========================================================================

    #[test]
    fn test_vad_reexport_config() {
        // Ensure VadConfig is accessible from audio module
        let config = VadConfig::new()
            .with_energy_threshold(2.5)
            .with_frame_size(320);
        assert!((config.energy_threshold - 2.5).abs() < f32::EPSILON);
        assert_eq!(config.frame_size, 320);
    }

    #[test]
    fn test_vad_reexport_detector() {
        // Ensure VoiceActivityDetector is accessible
        let vad = VoiceActivityDetector::default();
        assert_eq!(vad.state(), VadState::Silence);
    }

    #[test]
    fn test_vad_reexport_streaming() {
        // Ensure StreamingVad is accessible
        let vad = StreamingVad::default();
        assert!(!vad.is_in_speech());
    }

    #[test]
    fn test_vad_with_resampled_audio() {
        // Test VAD with audio from resampler
        let mut vad = VoiceActivityDetector::new(VadConfig::default());

        // Simulate resampled silence
        let silence = vec![0.0; 4800]; // 300ms at 16kHz
        let segments = vad.detect(&silence);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_vad_config_matches_audio_sample_rate() {
        // Ensure VAD sample rate matches audio pipeline
        let config = VadConfig::default();
        assert_eq!(config.sample_rate, SAMPLE_RATE);
    }

    #[test]
    fn test_streaming_processor_vad_integration() {
        // Test StreamingProcessor uses VAD correctly
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            enable_vad: true,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        // Push silence and verify VAD rejects it
        let silence = vec![0.0; 4800];
        processor.push_audio(&silence);
        processor.process();

        assert_eq!(processor.state(), ProcessorState::WaitingForSpeech);
    }

    #[test]
    fn test_vad_event_types() {
        // Ensure VadEvent is accessible
        let event = VadEvent::Continue;
        assert_eq!(event, VadEvent::Continue);

        let start = VadEvent::SpeechStart;
        assert_eq!(start, VadEvent::SpeechStart);

        let end = VadEvent::SpeechEnd;
        assert_eq!(end, VadEvent::SpeechEnd);
    }

    #[test]
    fn test_speech_segment_from_vad() {
        // Test SpeechSegment is accessible
        let segment = SpeechSegment {
            start: 1.0,
            end: 2.5,
            energy: 0.3,
        };
        assert!((segment.duration() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_audio_with_vad() {
        // Test VAD can process batch audio
        let mut vad = VoiceActivityDetector::default();

        // Create batch-like audio (multiple segments)
        let segment1 = vec![0.0; 1600];
        let segment2 = vec![0.0; 1600];

        vad.reset();
        let segments1 = vad.detect(&segment1);
        assert!(segments1.is_empty());

        vad.reset();
        let segments2 = vad.detect(&segment2);
        assert!(segments2.is_empty());
    }

    #[test]
    fn test_vad_low_latency_config_for_streaming() {
        let config = VadConfig::low_latency();
        // Low latency should have smaller frame size
        assert_eq!(config.frame_size, 160); // 10ms
        assert_eq!(config.frame_duration_ms(), 10.0);
    }

    #[test]
    fn test_vad_high_accuracy_config_for_batch() {
        let config = VadConfig::high_accuracy();
        // High accuracy should have larger frame size
        assert_eq!(config.frame_size, 800); // 50ms
        assert_eq!(config.frame_duration_ms(), 50.0);
    }
}

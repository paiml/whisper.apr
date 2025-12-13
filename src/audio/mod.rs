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
    ProcessorState, ProcessorStats, StreamingConfig, StreamingProcessor, DEFAULT_CHUNK_DURATION,
    DEFAULT_CHUNK_OVERLAP, MIN_SPEECH_DURATION_MS,
};

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
}

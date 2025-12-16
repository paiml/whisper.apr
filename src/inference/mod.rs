//! Inference engine
//!
//! Handles running the full transcription pipeline including streaming support.
//!
//! # Modules
//!
//! - [`beam`] - Beam search decoding for higher quality
//! - [`greedy`] - Fast greedy decoding
//! - [`processors`] - Logit processors for pre-sampling transforms
//! - [`streaming`] - Real-time streaming transcription
//!
//! # LogitProcessor Integration
//!
//! The `processors` module provides Whisper-specific implementations of
//! realizar's `LogitProcessor` trait, enabling composable pre-sampling transforms:
//!
//! ```rust,ignore
//! use whisper_apr::inference::processors::WhisperTokenSuppressor;
//!
//! let suppressor = WhisperTokenSuppressor::default();
//! suppressor.apply(&mut logits);
//! ```

mod beam;
mod greedy;
pub mod processors;
mod streaming;

pub use beam::{BeamSearchDecoder, Hypothesis};
pub use greedy::GreedyDecoder;
pub use processors::WhisperTokenSuppressor;
pub use streaming::{
    StreamingConfig, StreamingResult, StreamingStats, StreamingTranscriber, TranscriberState,
};

use crate::error::WhisperResult;
use crate::TranscriptionResult;

/// Inference engine configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 448,
            temperature: 0.0,
        }
    }
}

/// Run inference on audio features
///
/// # Arguments
/// * `audio_features` - Encoded audio features from encoder
/// * `config` - Inference configuration
///
/// # Errors
/// Returns error if inference fails
pub fn run_inference(
    audio_features: &[f32],
    config: &InferenceConfig,
) -> WhisperResult<TranscriptionResult> {
    let _ = (audio_features, config);

    Ok(TranscriptionResult {
        text: String::new(),
        language: "en".into(),
        segments: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_tokens, 448);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_run_inference() {
        let config = InferenceConfig::default();
        let result = run_inference(&[], &config);
        assert!(result.is_ok());
    }
}

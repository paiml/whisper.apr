//! Error types for Whisper.apr

use thiserror::Error;

/// Result type alias for Whisper operations
pub type WhisperResult<T> = Result<T, WhisperError>;

/// Errors that can occur during Whisper operations
#[derive(Debug, Error)]
pub enum WhisperError {
    /// Invalid audio format or parameters
    #[error("audio error: {0}")]
    Audio(String),

    /// Model loading or inference error
    #[error("model error: {0}")]
    Model(String),

    /// Invalid .apr format
    #[error("format error: {0}")]
    Format(String),

    /// Tokenization error
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Inference error
    #[error("inference error: {0}")]
    Inference(String),

    /// I/O error
    #[cfg(feature = "std")]
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// WASM-specific error
    #[cfg(feature = "wasm")]
    #[error("wasm error: {0}")]
    Wasm(String),

    /// Speaker diarization error
    #[error("diarization error: {0}")]
    Diarization(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = WhisperError::Audio("invalid sample rate".into());
        assert_eq!(err.to_string(), "audio error: invalid sample rate");
    }

    #[test]
    fn test_error_variants() {
        let audio_err = WhisperError::Audio("test".into());
        let model_err = WhisperError::Model("test".into());
        let format_err = WhisperError::Format("test".into());
        let tokenizer_err = WhisperError::Tokenizer("test".into());
        let inference_err = WhisperError::Inference("test".into());
        let diarization_err = WhisperError::Diarization("test".into());

        assert!(matches!(audio_err, WhisperError::Audio(_)));
        assert!(matches!(model_err, WhisperError::Model(_)));
        assert!(matches!(format_err, WhisperError::Format(_)));
        assert!(matches!(tokenizer_err, WhisperError::Tokenizer(_)));
        assert!(matches!(inference_err, WhisperError::Inference(_)));
        assert!(matches!(diarization_err, WhisperError::Diarization(_)));
    }

    #[test]
    fn test_diarization_error_display() {
        let err = WhisperError::Diarization("speaker identification failed".into());
        assert_eq!(
            err.to_string(),
            "diarization error: speaker identification failed"
        );
    }
}

//! APR Conversion Validation Module
//!
//! Implements the 25-point QA checklist from APR-SPEC.md for validating
//! converted APR model files.
//!
//! # Checklist Categories
//!
//! - A. Structural Integrity (5 points): Magic, header, tensor count, shapes, CRC
//! - B. Layer Norm Validation (5 points): LN weight/bias statistics
//! - C. Attention/Linear Validation (5 points): QKV, FFN weight statistics
//! - D. Embedding Validation (5 points): Token/positional embedding stats
//! - E. Functional Validation (5 points): Reference comparison, transcription tests

mod check;
mod stats;
mod validator;

pub use check::{ValidationCheck, ValidationReport};
pub use stats::TensorStats;
pub use validator::AprValidator;

use crate::error::{WhisperError, WhisperResult};
use crate::format::AprReader;

/// Validate an APR file from bytes
///
/// # Errors
/// Returns error if file cannot be parsed
pub fn validate_apr_bytes(data: Vec<u8>) -> WhisperResult<ValidationReport> {
    let reader = AprReader::new(data)?;
    let validator = AprValidator::new(&reader);
    Ok(validator.validate_all())
}

/// Quick validation - only critical checks
///
/// # Errors
/// Returns error if critical validation fails
pub fn quick_validate(reader: &AprReader) -> WhisperResult<()> {
    // Validate decoder layer norm weights are within expected statistical range
    if let Ok(data) = reader.load_tensor("decoder.layer_norm.weight") {
        let stats = TensorStats::compute("decoder.layer_norm.weight", &data);
        if stats.mean < 0.5 || stats.mean > 3.0 {
            return Err(WhisperError::Format(format!(
                "decoder.layer_norm.weight mean={:.4} outside valid range [0.5, 3.0]",
                stats.mean
            )));
        }
    }

    // Check encoder LN weight mean
    if let Ok(data) = reader.load_tensor("encoder.layer_norm.weight") {
        let stats = TensorStats::compute("encoder.layer_norm.weight", &data);
        if stats.mean < 0.5 || stats.mean > 3.0 {
            return Err(WhisperError::Format(format!(
                "encoder.layer_norm.weight mean={:.4} outside valid range [0.5, 3.0]",
                stats.mean
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;

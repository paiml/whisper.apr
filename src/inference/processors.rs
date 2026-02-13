//! Logit processors for Whisper decoding
//!
//! This module provides Whisper-specific logit processors that implement
//! realizar's `LogitProcessor` trait, enabling composable pre-sampling transforms.
//!
//! # Architecture
//!
//! Whisper.apr serves as a reference implementation demonstrating how to integrate
//! realizar's LogitProcessor system with domain-specific requirements.
//!
//! # Example
//!
//! ```rust,ignore
//! use whisper_apr::inference::processors::WhisperTokenSuppressor;
//!
//! let suppressor = WhisperTokenSuppressor::default();
//! suppressor.apply(&mut logits);
//! ```
//!
//! # References
//!
//! - Radford et al. (2023) - Robust Speech Recognition via Large-Scale Weak Supervision
//! - Holtzman et al. (2020) - The Curious Case of Neural Text Degeneration

#[cfg(feature = "realizar-inference")]
use realizar::generate::{LogitProcessor, LogitProcessorContext};

use crate::tokenizer::special_tokens;

/// Whisper-specific token suppressor
///
/// Suppresses special tokens that should never appear in transcription output:
/// - SOT (start of transcript)
/// - Language tokens
/// - Task tokens (translate, transcribe)
/// - Other control tokens
///
/// Based on whisper.cpp's suppress_tokens implementation.
#[derive(Debug, Clone)]
pub struct WhisperTokenSuppressor {
    /// Tokens to suppress (set to -inf)
    suppress_ids: Vec<u32>,
    /// Whether to suppress timestamp tokens
    suppress_timestamps: bool,
    /// Number of vocabulary entries (for bounds checking)
    n_vocab: usize,
}

impl WhisperTokenSuppressor {
    /// Create a new Whisper token suppressor with default suppressions
    ///
    /// Default suppressions include:
    /// - SOT (50257)
    /// - NO_SPEECH (50362)
    /// - TRANSLATE (50358)
    /// - TRANSCRIBE (50359)
    /// - PREV (50361)
    /// - SPEAKER_TURN (50363)
    /// - NO_TIMESTAMPS (50364)
    /// - All language tokens (50258-50357)
    #[must_use]
    pub fn new() -> Self {
        let mut suppress_ids = vec![
            special_tokens::SOT,
            special_tokens::NO_SPEECH,
            special_tokens::TRANSLATE,
            special_tokens::TRANSCRIBE,
            special_tokens::PREV,
            special_tokens::SPEAKER_TURN,
            special_tokens::NO_TIMESTAMPS,
        ];

        // Add all language tokens (50258 to 50357)
        for lang_id in special_tokens::LANG_BASE..special_tokens::TRANSLATE {
            suppress_ids.push(lang_id);
        }

        Self {
            suppress_ids,
            suppress_timestamps: true,
            n_vocab: 51865,
        }
    }

    /// Create with custom suppressions
    #[must_use]
    pub fn with_tokens(tokens: Vec<u32>) -> Self {
        Self {
            suppress_ids: tokens,
            suppress_timestamps: true,
            n_vocab: 51865,
        }
    }

    /// Set whether to suppress timestamp tokens
    #[must_use]
    pub fn with_timestamp_suppression(mut self, suppress: bool) -> Self {
        self.suppress_timestamps = suppress;
        self
    }

    /// Set vocabulary size (for bounds checking)
    #[must_use]
    pub fn with_vocab_size(mut self, n_vocab: usize) -> Self {
        self.n_vocab = n_vocab;
        self
    }

    /// Add additional tokens to suppress
    pub fn add_suppression(&mut self, token: u32) {
        if !self.suppress_ids.contains(&token) {
            self.suppress_ids.push(token);
        }
    }

    /// Get the list of suppressed token IDs
    #[must_use]
    pub fn suppressed_tokens(&self) -> &[u32] {
        &self.suppress_ids
    }

    /// Check if timestamps are suppressed
    #[must_use]
    pub const fn suppresses_timestamps(&self) -> bool {
        self.suppress_timestamps
    }

    /// Apply suppression to logits (standalone function for use without realizar)
    pub fn apply(&self, logits: &mut [f32]) {
        // Suppress configured tokens
        for &token_id in &self.suppress_ids {
            let idx = token_id as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }

        // Suppress timestamps if configured
        if self.suppress_timestamps {
            let timestamp_start = special_tokens::TIMESTAMP_BASE as usize;
            for logit in logits
                .iter_mut()
                .skip(timestamp_start)
                .take(self.n_vocab.saturating_sub(timestamp_start))
            {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

impl Default for WhisperTokenSuppressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "realizar-inference")]
impl LogitProcessor for WhisperTokenSuppressor {
    fn process(&self, logits: &mut [f32], _ctx: &LogitProcessorContext<'_>) {
        self.apply(logits);
    }

    fn name(&self) -> &'static str {
        "whisper_token_suppressor"
    }
}

// Re-export realizar types when feature is enabled
#[cfg(feature = "realizar-inference")]
pub use realizar::generate::{
    LogitProcessorChain, RepetitionPenalty, TemperatureScaler, TokenSuppressor,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_suppressor_default() {
        let suppressor = WhisperTokenSuppressor::default();

        // Should suppress SOT
        assert!(suppressor
            .suppressed_tokens()
            .contains(&special_tokens::SOT));

        // Should suppress language tokens
        assert!(suppressor
            .suppressed_tokens()
            .contains(&special_tokens::LANG_BASE));

        // Should suppress by default timestamps
        assert!(suppressor.suppresses_timestamps());
    }

    #[test]
    fn test_whisper_suppressor_apply() {
        let suppressor = WhisperTokenSuppressor::default();
        let mut logits = vec![1.0f32; 51865];

        suppressor.apply(&mut logits);

        // SOT should be -inf
        assert!(logits[special_tokens::SOT as usize] == f32::NEG_INFINITY);

        // Language tokens should be -inf
        assert!(logits[special_tokens::LANG_BASE as usize] == f32::NEG_INFINITY);

        // EOT should NOT be suppressed
        assert!(logits[special_tokens::EOT as usize] != f32::NEG_INFINITY);
    }

    #[test]
    fn test_whisper_suppressor_timestamps() {
        let suppressor = WhisperTokenSuppressor::new();
        let mut logits = vec![1.0f32; 51865];

        suppressor.apply(&mut logits);

        // Timestamp tokens should be -inf when suppress_timestamps=true
        assert!(logits[special_tokens::TIMESTAMP_BASE as usize] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_whisper_suppressor_no_timestamps() {
        let suppressor = WhisperTokenSuppressor::new().with_timestamp_suppression(false);
        let mut logits = vec![1.0f32; 51865];

        suppressor.apply(&mut logits);

        // Timestamp tokens should NOT be -inf
        assert!(logits[special_tokens::TIMESTAMP_BASE as usize] != f32::NEG_INFINITY);
    }

    #[test]
    fn test_whisper_suppressor_custom_tokens() {
        let suppressor = WhisperTokenSuppressor::with_tokens(vec![100, 200, 300]);
        let mut logits = vec![1.0f32; 51865];

        suppressor.apply(&mut logits);

        assert!(logits[100] == f32::NEG_INFINITY);
        assert!(logits[200] == f32::NEG_INFINITY);
        assert!(logits[300] == f32::NEG_INFINITY);
        assert!(logits[101] != f32::NEG_INFINITY);
    }

    #[test]
    fn test_whisper_suppressor_add_token() {
        let mut suppressor = WhisperTokenSuppressor::with_tokens(vec![]);
        suppressor.add_suppression(500);

        assert!(suppressor.suppressed_tokens().contains(&500));
    }

    #[test]
    fn test_whisper_suppressor_bounds_check() {
        let suppressor = WhisperTokenSuppressor::with_tokens(vec![100000]); // Out of bounds
        let mut logits = vec![1.0f32; 100];

        // Should not panic
        suppressor.apply(&mut logits);
    }

    #[test]
    fn property_suppressor_idempotent() {
        // Applying suppression twice should have same effect as once
        let suppressor = WhisperTokenSuppressor::default();

        let mut logits1 = vec![1.0f32; 51865];
        suppressor.apply(&mut logits1);

        let mut logits2 = logits1.clone();
        suppressor.apply(&mut logits2);

        for (a, b) in logits1.iter().zip(logits2.iter()) {
            assert!(
                (a - b).abs() < 1e-6 || (a.is_infinite() && b.is_infinite()),
                "Suppression should be idempotent"
            );
        }
    }

    #[test]
    fn property_suppressor_preserves_eot() {
        // EOT should NEVER be suppressed (critical for decoding termination)
        let suppressor = WhisperTokenSuppressor::default();
        let mut logits = vec![1.0f32; 51865];

        suppressor.apply(&mut logits);

        assert!(
            logits[special_tokens::EOT as usize].is_finite(),
            "EOT must never be suppressed"
        );
    }

    // =========================================================================
    // Builder Coverage Tests (PMAT-024)
    // =========================================================================

    #[test]
    fn test_with_vocab_size_affects_timestamp_suppression() {
        // with_vocab_size controls how many timestamp tokens are suppressed
        // Default: 51865, timestamps start at 50365, so 1500 tokens suppressed
        // With vocab_size=50400, only 35 timestamp tokens suppressed
        let suppressor = WhisperTokenSuppressor::new().with_vocab_size(50400);
        let mut logits = vec![1.0f32; 51865];
        suppressor.apply(&mut logits);

        // Token 50400 should NOT be suppressed (beyond n_vocab)
        assert!(
            logits[50400].is_finite(),
            "tokens beyond n_vocab should not be suppressed"
        );
    }

    #[test]
    fn test_with_vocab_size_large() {
        let suppressor = WhisperTokenSuppressor::new().with_vocab_size(100_000);
        let mut logits = vec![1.0f32; 100_000];
        suppressor.apply(&mut logits);
        // Timestamp tokens at 50365+ should be suppressed
        assert!(logits[special_tokens::TIMESTAMP_BASE as usize] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_with_vocab_size_chain() {
        // Verify builder chaining works correctly
        let suppressor = WhisperTokenSuppressor::new()
            .with_vocab_size(32000)
            .with_timestamp_suppression(false);
        let mut logits = vec![1.0f32; 51865];
        suppressor.apply(&mut logits);
        // Timestamps should NOT be suppressed (disabled)
        assert!(logits[special_tokens::TIMESTAMP_BASE as usize].is_finite());
    }

    #[test]
    fn test_add_suppression_deduplication() {
        let mut suppressor = WhisperTokenSuppressor::with_tokens(vec![100]);
        suppressor.add_suppression(100); // Duplicate
        suppressor.add_suppression(200); // New
                                         // 100 should appear only once
        assert_eq!(
            suppressor
                .suppressed_tokens()
                .iter()
                .filter(|&&t| t == 100)
                .count(),
            1
        );
        assert!(suppressor.suppressed_tokens().contains(&200));
    }

    // =========================================================================
    // with_vocab_size direct field verification (WAPR-QA-005)
    // =========================================================================

    #[test]
    fn test_with_vocab_size_sets_field_directly() {
        let suppressor = WhisperTokenSuppressor::new().with_vocab_size(32000);
        // Verify the n_vocab field is set by checking timestamp suppression range
        // With n_vocab=32000, timestamps at 50365+ are beyond vocab, so take(0)
        let mut logits = vec![1.0f32; 51865];
        suppressor.apply(&mut logits);

        // Token at index 32000 should NOT be suppressed as timestamp
        // (it's beyond n_vocab so the take() stops before it)
        assert!(
            logits[32000].is_finite(),
            "token at 32000 should not be suppressed when n_vocab=32000"
        );
    }

    #[test]
    fn test_with_vocab_size_zero() {
        // Edge case: n_vocab=0 means no timestamp suppression range
        let suppressor = WhisperTokenSuppressor::new().with_vocab_size(0);
        let mut logits = vec![1.0f32; 51865];
        suppressor.apply(&mut logits);

        // Timestamp base should NOT be suppressed (saturating_sub(50365) = 0)
        assert!(
            logits[special_tokens::TIMESTAMP_BASE as usize].is_finite(),
            "timestamp base should not be suppressed when n_vocab=0"
        );
    }
}

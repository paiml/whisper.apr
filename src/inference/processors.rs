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
    /// Size-derived special ids (whisper.apr#108)
    specials: special_tokens::SpecialTokens,
    /// Caller-added tokens, kept apart from the layout defaults so that
    /// `with_vocab_size` never drops or invents them.
    custom_ids: Vec<u32>,
    /// Whether the layout defaults are suppressed (false for `with_tokens`).
    layout_defaults: bool,
}

/// Default suppressions for one vocabulary layout (whisper.apr#108).
fn default_suppressions(specials: &special_tokens::SpecialTokens) -> Vec<u32> {
    // Every header control id. On the v2 layout start_of_lm is the old
    // SPEAKER_TURN slot (50360), so the v2 set is unchanged from main.
    let mut ids = vec![
        specials.sot,
        specials.no_speech,
        specials.translate,
        specials.transcribe,
        specials.start_of_lm,
        specials.prev,
        specials.no_timestamps,
    ];
    // Every language token, up to the task block.
    for lang_id in specials.lang_base..specials.translate {
        ids.push(lang_id);
    }
    ids
}

impl WhisperTokenSuppressor {
    /// Create a new suppressor with the default suppressions for the
    /// large-v2 / multilingual layout (vocab 51865). Call
    /// [`WhisperTokenSuppressor::with_vocab_size`] for other models.
    ///
    /// Default suppressions: SOT, NO_SPEECH, TRANSLATE, TRANSCRIBE, PREV,
    /// NO_TIMESTAMPS, SPEAKER_TURN, and every language token below TRANSLATE.
    #[must_use]
    pub fn new() -> Self {
        let specials = special_tokens::SpecialTokens::multilingual();
        Self {
            suppress_ids: default_suppressions(&specials),
            suppress_timestamps: true,
            n_vocab: 51865,
            specials,
            custom_ids: Vec::new(),
            layout_defaults: true,
        }
    }

    /// Create with custom suppressions only (layout 51865 for timestamp bounds).
    #[must_use]
    pub fn with_tokens(tokens: Vec<u32>) -> Self {
        Self {
            suppress_ids: tokens.clone(),
            suppress_timestamps: true,
            n_vocab: 51865,
            specials: special_tokens::SpecialTokens::multilingual(),
            custom_ids: tokens,
            layout_defaults: false,
        }
    }

    /// Recompute `suppress_ids` from the layout defaults (if enabled) and
    /// the caller's tokens.
    fn rebuild_suppress_ids(&mut self) {
        let mut ids = if self.layout_defaults {
            default_suppressions(&self.specials)
        } else {
            Vec::new()
        };
        for &t in &self.custom_ids {
            if !ids.contains(&t) {
                ids.push(t);
            }
        }
        self.suppress_ids = ids;
    }

    /// Set whether to suppress timestamp tokens
    #[must_use]
    pub fn with_timestamp_suppression(mut self, suppress: bool) -> Self {
        self.suppress_timestamps = suppress;
        self
    }

    /// Set the vocabulary size. Derives every special id from it and
    /// resets the layout defaults to that layout. Custom tokens added so
    /// far are kept. Panics on any size other than 51864, 51865 or
    /// 51866 (whisper.apr#108).
    #[must_use]
    pub fn with_vocab_size(mut self, n_vocab: usize) -> Self {
        self.specials = special_tokens::SpecialTokens::for_vocab_size(n_vocab);
        self.n_vocab = n_vocab;
        self.rebuild_suppress_ids();
        self
    }

    /// Add additional tokens to suppress
    pub fn add_suppression(&mut self, token: u32) {
        if !self.custom_ids.contains(&token) {
            self.custom_ids.push(token);
        }
        self.rebuild_suppress_ids();
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
            let timestamp_start = self.specials.timestamp_base as usize;
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
        // with_vocab_size bounds the timestamp suppression range. Only the
        // three supported sizes are accepted (whisper.apr#108).
        // With vocab_size=51864 the last id, 51864, is beyond n_vocab.
        let suppressor = WhisperTokenSuppressor::new().with_vocab_size(51864);
        let mut logits = vec![1.0f32; 51865];
        suppressor.apply(&mut logits);

        // Token 51864 should NOT be suppressed (beyond n_vocab)
        assert!(
            logits[51864].is_finite(),
            "tokens beyond n_vocab should not be suppressed"
        );
    }

    #[test]
    fn test_with_vocab_size_large() {
        let suppressor = WhisperTokenSuppressor::new().with_vocab_size(51866);
        let mut logits = vec![1.0f32; 51866];
        suppressor.apply(&mut logits);
        // v3 timestamp tokens (from 50365) should be suppressed
        assert!(
            logits[special_tokens::SpecialTokens::multilingual_v3().timestamp_base as usize]
                == f32::NEG_INFINITY
        );
    }

    #[test]
    fn test_with_vocab_size_chain() {
        // Verify builder chaining works correctly
        let suppressor = WhisperTokenSuppressor::new()
            .with_vocab_size(51864)
            .with_timestamp_suppression(false);
        let mut logits = vec![1.0f32; 51865];
        suppressor.apply(&mut logits);
        // Timestamps should NOT be suppressed (disabled)
        assert!(
            logits[special_tokens::SpecialTokens::english_only().timestamp_base as usize]
                .is_finite()
        );
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
        let suppressor = WhisperTokenSuppressor::new().with_vocab_size(51864);
        let mut logits = vec![1.0f32; 51865];
        suppressor.apply(&mut logits);

        // Token at index 51864 is beyond n_vocab, so the take() stops before it
        assert!(
            logits[51864].is_finite(),
            "token at 51864 should not be suppressed when n_vocab=51864"
        );
    }

    #[test]
    #[should_panic(expected = "unsupported n_vocab 0")]
    fn test_with_vocab_size_zero_is_rejected() {
        // n_vocab=0 is not a Whisper layout and must fail loudly (whisper.apr#108)
        let _ = WhisperTokenSuppressor::new().with_vocab_size(0);
    }
}

#[cfg(test)]
mod size_derived_suppression_tests {
    use super::WhisperTokenSuppressor;
    use crate::tokenizer::special_tokens::SpecialTokens;

    /// whisper.apr#108 falsifier: on large-v3/turbo (51866) the suppressed
    /// control ids are the v3 ids, not the v2 ids the suppressor hard-codes.
    /// Fails on main: the v3 no-timestamps id (50364) is not in the suppressed
    /// set, because main's default list is the v2 layout.
    #[test]
    fn suppressor_v3_suppresses_v3_control_ids() {
        let s = WhisperTokenSuppressor::new().with_vocab_size(51866);
        let v3 = SpecialTokens::multilingual_v3();
        let suppressed = s.suppressed_tokens();
        for id in [
            v3.sot,
            v3.no_speech,
            v3.translate,
            v3.transcribe,
            v3.prev,
            v3.no_timestamps,
        ] {
            assert!(suppressed.contains(&id), "v3 id {id} not suppressed");
        }
        for lang in v3.lang_base..v3.translate {
            assert!(
                suppressed.contains(&lang),
                "v3 language id {lang} not suppressed"
            );
        }
    }

    #[test]
    fn suppressor_v2_suppresses_v2_control_ids() {
        let s = WhisperTokenSuppressor::new().with_vocab_size(51865);
        let v2 = SpecialTokens::multilingual();
        for id in [v2.translate, v2.transcribe, v2.prev, v2.no_timestamps] {
            assert!(
                s.suppressed_tokens().contains(&id),
                "v2 id {id} not suppressed"
            );
        }
    }

    #[test]
    #[should_panic(expected = "unsupported n_vocab")]
    fn suppressor_unsupported_size_fails_loudly() {
        let _ = WhisperTokenSuppressor::new().with_vocab_size(51000);
    }
}

#[cfg(test)]
mod suppression_layout_tests {
    use super::WhisperTokenSuppressor;
    use crate::tokenizer::special_tokens::{self, SpecialTokens};

    /// The v2 default set must equal what main suppresses (no regression).
    #[test]
    fn v2_default_set_matches_main() {
        let s = WhisperTokenSuppressor::new();
        let mut expected = vec![
            special_tokens::SOT,
            special_tokens::NO_SPEECH,
            special_tokens::TRANSLATE,
            special_tokens::TRANSCRIBE,
            special_tokens::PREV,
            special_tokens::SPEAKER_TURN,
            special_tokens::NO_TIMESTAMPS,
        ];
        expected.extend(special_tokens::LANG_BASE..special_tokens::TRANSLATE);
        let mut got = s.suppressed_tokens().to_vec();
        got.sort_unstable();
        expected.sort_unstable();
        assert_eq!(got, expected);
    }

    /// Each special id is suppressed once, on every supported layout.
    #[test]
    fn each_control_id_suppressed_once() {
        for n in [51864usize, 51865, 51866] {
            let s = WhisperTokenSuppressor::new().with_vocab_size(n);
            let sp = SpecialTokens::for_vocab_size(n);
            for id in [
                sp.translate,
                sp.transcribe,
                sp.start_of_lm,
                sp.prev,
                sp.no_speech,
                sp.no_timestamps,
            ] {
                let count = s.suppressed_tokens().iter().filter(|&&t| t == id).count();
                assert_eq!(count, 1, "n_vocab {n}: id {id} suppressed {count} times");
            }
        }
    }
}

#[cfg(test)]
mod review_falsifier_tests {
    use super::WhisperTokenSuppressor;
    use crate::tokenizer::special_tokens::SpecialTokens;

    #[test]
    fn with_tokens_stays_custom_only_across_vocab_size() {
        let s = WhisperTokenSuppressor::with_tokens(vec![42]).with_vocab_size(51866);
        assert_eq!(s.suppressed_tokens(), &[42]);
        let empty = WhisperTokenSuppressor::with_tokens(vec![]).with_vocab_size(51866);
        assert!(empty.suppressed_tokens().is_empty());
    }

    #[test]
    fn custom_token_in_old_default_set_survives_resize() {
        // 50363 is no_timestamps on v2 but a timestamp slot on 51864.
        let s = WhisperTokenSuppressor::with_tokens(vec![50363])
            .with_vocab_size(51864)
            .with_timestamp_suppression(false);
        assert_eq!(s.suppressed_tokens(), &[50363]);
        let mut logits = vec![1.0f32; 51864];
        s.apply(&mut logits);
        assert_eq!(logits[50363], f32::NEG_INFINITY);
        assert!(logits[50257].is_finite());
    }

    #[test]
    fn with_vocab_size_keeps_custom_tokens() {
        let s = WhisperTokenSuppressor::with_tokens(vec![42]).with_vocab_size(51866);
        assert_eq!(s.suppressed_tokens(), &[42]); // custom-only: no defaults
        let mut base = WhisperTokenSuppressor::new();
        base.add_suppression(42);
        let s2 = base.with_vocab_size(51866);
        assert!(s2.suppressed_tokens().contains(&42), "custom token dropped");
        assert_eq!(s2.suppressed_tokens().len(), 108); // 107 v3 defaults + 42
    }

    #[test]
    fn v3_apply_without_timestamp_suppression_keeps_timestamps_finite() {
        let v3 = SpecialTokens::multilingual_v3();
        let s = WhisperTokenSuppressor::new()
            .with_vocab_size(51866)
            .with_timestamp_suppression(false);
        let mut logits = vec![1.0f32; 51866];
        s.apply(&mut logits);
        assert_eq!(logits[v3.no_timestamps as usize], f32::NEG_INFINITY);
        assert!(logits[v3.timestamp_base as usize].is_finite());
    }
}

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

use crate::tokenizer::special_tokens::{self, SpecialTokens};

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
    /// First timestamp token (`<|0.00|>`) of the vocabulary
    timestamp_base: u32,
}

impl WhisperTokenSuppressor {
    /// Create a new Whisper token suppressor with default suppressions for
    /// the v2 (51865) vocabulary only; use [`Self::for_specials`] with the
    /// model's [`SpecialTokens`] (issue #108).
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
        Self::for_specials(&SpecialTokens::multilingual())
    }

    /// The default suppressions of one vocabulary: sot, `<|nospeech|>`, both
    /// task tokens, `<|startofprev|>`, `<|startoflm|>`, `<|notimestamps|>`,
    /// every language token and (unless disabled) every timestamp.
    #[must_use]
    pub fn for_specials(specials: &SpecialTokens) -> Self {
        let mut suppress_ids = vec![
            specials.sot,
            specials.no_speech,
            specials.translate,
            specials.transcribe,
            specials.prev,
            specials.start_of_lm,
            specials.no_timestamps,
        ];
        suppress_ids.extend(specials.lang_base..specials.translate);

        Self {
            suppress_ids,
            suppress_timestamps: true,
            n_vocab: 51865,
            timestamp_base: specials.timestamp_base,
        }
    }

    /// Create with custom suppressions
    #[must_use]
    pub fn with_tokens(tokens: Vec<u32>) -> Self {
        Self {
            suppress_ids: tokens,
            suppress_timestamps: true,
            n_vocab: 51865,
            timestamp_base: special_tokens::TIMESTAMP_BASE,
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
            let timestamp_start = self.timestamp_base as usize;
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

/// OpenAI's `ApplyTimestampRules` and `SuppressBlank` for the timestamp
/// decode path (issue #106).
///
/// Without them a 30s window that ends in silence decodes to "." followed by
/// `<|endoftext|>`: nothing forces the first sampled token to be a timestamp.
/// The rules, given the tokens so far, the index where sampling began and the
/// next token's logits:
///
/// * `<|notimestamps|>` is never sampled.
/// * Pairing: after a timestamp pair (or a timestamp that is the first
///   sampled token) the next token is not a timestamp; after a lone timestamp
///   that follows text the next token is a timestamp or EOT, never text.
/// * Monotonic: timestamps never decrease. A timestamp may repeat only as the
///   start of the segment its predecessor closed (`timestamp_last`);
///   otherwise it must exceed the last one (`timestamp_last + 1`), so every
///   segment has a nonzero length.
/// * First sampled step: only a timestamp up to `max_initial_timestamp`
///   (1.0s, index 50) may be sampled, and blank (" ") and EOT are banned.
/// * Dominance: when the total probability of the timestamp tokens beats the
///   most likely token below them, a timestamp is sampled.
///
/// Contract `whisper-timestamp-rules-v1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimestampRules {
    eot: u32,
    no_timestamps: u32,
    timestamp_begin: u32,
    sample_begin: usize,
    max_initial_timestamp_index: u32,
    blank_tokens: Vec<u32>,
}

impl TimestampRules {
    /// `max_initial_timestamp` = 1.0s in 20ms timestamp steps.
    pub const MAX_INITIAL_TIMESTAMP_INDEX: u32 = 50;

    /// Rules for explicit special-token ids; `sample_begin` is the length of
    /// the initial (prompt) tokens, so `tokens[sample_begin..]` are sampled.
    #[must_use]
    pub fn new(eot: u32, no_timestamps: u32, timestamp_begin: u32, sample_begin: usize) -> Self {
        Self {
            eot,
            no_timestamps,
            timestamp_begin,
            sample_begin,
            max_initial_timestamp_index: Self::MAX_INITIAL_TIMESTAMP_INDEX,
            blank_tokens: Vec::new(),
        }
    }

    /// Rules for a Whisper vocabulary. The ids come from the one
    /// [`SpecialTokens`] source (issue #108); on large-v3 (51866 tokens)
    /// `<|notimestamps|>` = 50364 and `<|0.00|>` = 50365.
    #[must_use]
    pub fn for_vocab_size(n_vocab: usize, sample_begin: usize) -> Self {
        Self::for_specials(&SpecialTokens::for_vocab_size(n_vocab), sample_begin)
    }

    /// Rules over `specials`' vocabulary.
    #[must_use]
    pub fn for_specials(specials: &SpecialTokens, sample_begin: usize) -> Self {
        Self::new(
            specials.eot,
            specials.no_timestamps,
            specials.timestamp_base,
            sample_begin,
        )
    }

    /// Rules for a decode that starts from `initial_tokens`, or `None` when
    /// the prompt asks for no timestamps: that mode is left unchanged.
    #[must_use]
    pub fn for_prompt(n_vocab: usize, initial_tokens: &[u32]) -> Option<Self> {
        let rules = Self::for_vocab_size(n_vocab, initial_tokens.len());
        (!initial_tokens.contains(&rules.no_timestamps)).then_some(rules)
    }

    /// Tokens suppressed with EOT at the first sampled step (the encoding
    /// of " ", as OpenAI's `SuppressBlank`).
    #[must_use]
    pub fn with_blank_tokens(mut self, blank_tokens: Vec<u32>) -> Self {
        self.blank_tokens = blank_tokens;
        self
    }

    /// First timestamp token id.
    #[must_use]
    pub const fn timestamp_begin(&self) -> u32 {
        self.timestamp_begin
    }

    /// Apply the rules to the logits for the token after `tokens`.
    #[provable_contracts_macros::contract("whisper-timestamp-rules-v1", equation = "apply")]
    pub fn apply(&self, tokens: &[u32], logits: &mut [f32]) {
        let begin = (self.timestamp_begin as usize).min(logits.len());
        let sampled = tokens.get(self.sample_begin..).unwrap_or(&[]);
        if sampled.is_empty() {
            self.suppress_blank(logits);
        }
        suppress(
            logits,
            self.no_timestamps as usize..self.no_timestamps as usize + 1,
        );
        self.apply_pairing(sampled, logits, begin);
        if sampled.is_empty() {
            let last_allowed = begin + self.max_initial_timestamp_index as usize;
            suppress(logits, 0..begin);
            suppress(logits, last_allowed + 1..logits.len());
        }
        Self::apply_dominance(logits, begin);
    }

    /// `SuppressBlank`: no blank and no EOT as the first sampled token.
    fn suppress_blank(&self, logits: &mut [f32]) {
        for &t in self.blank_tokens.iter().chain(core::iter::once(&self.eot)) {
            suppress(logits, t as usize..t as usize + 1);
        }
    }

    /// Pairing and monotonic rules over the sampled tokens.
    fn apply_pairing(&self, sampled: &[u32], logits: &mut [f32], begin: usize) {
        let is_ts = |t: u32| t >= self.timestamp_begin;
        let n = sampled.len();
        let last_was_ts = n >= 1 && is_ts(sampled[n - 1]);
        let penultimate_was_ts = n < 2 || is_ts(sampled[n - 2]);
        if last_was_ts && penultimate_was_ts {
            suppress(logits, begin..logits.len());
        } else if last_was_ts {
            suppress(logits, 0..(self.eot as usize).min(begin));
        }
        if let Some(&last_ts) = sampled.iter().rev().find(|&&t| is_ts(t)) {
            let closing = last_was_ts && !penultimate_was_ts;
            let bound = last_ts as usize + usize::from(!closing);
            suppress(logits, begin..bound.min(logits.len()));
        }
    }

    /// Sample a timestamp when the timestamps' total log-probability beats
    /// the best log-probability below them (the shared normaliser cancels).
    fn apply_dominance(logits: &mut [f32], begin: usize) {
        let ts_mass = log_sum_exp(&logits[begin..]);
        let best_text = logits[..begin]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if ts_mass > best_text {
            suppress(logits, 0..begin);
        }
    }
}

/// Set `logits[range]` (clipped to the logits) to -inf.
fn suppress(logits: &mut [f32], range: core::ops::Range<usize>) {
    let end = range.end.min(logits.len());
    if let Some(slice) = logits.get_mut(range.start.min(end)..end) {
        slice.fill(f32::NEG_INFINITY);
    }
}

/// Numerically stable `ln(sum(exp(x)))`; -inf for an empty or all -inf slice.
fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return max;
    }
    max + values.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

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

    /// On large-v3/turbo (51866) the suppressor bans the v3 task, PREV and
    /// notimestamps ids, and its timestamp range starts at `<|0.00|>` = 50365
    /// (issue #108).
    #[test]
    fn the_v3_suppressor_uses_v3_ids() {
        let v3 = SpecialTokens::multilingual_v3();
        let suppressor = WhisperTokenSuppressor::for_specials(&v3).with_vocab_size(51_866);
        let banned = suppressor.suppressed_tokens();
        for id in [
            50_258, 50_259, 50_358, 50_359, 50_360, 50_361, 50_362, 50_363, 50_364,
        ] {
            assert!(banned.contains(&id), "{id} must be suppressed");
        }
        assert!(!banned.contains(&50_257), "eot stays allowed");

        let mut logits = vec![1.0f32; 51_866];
        suppressor.apply(&mut logits);
        assert!(logits[50_364] == f32::NEG_INFINITY);
        assert!(logits[50_365] == f32::NEG_INFINITY);
        assert!(logits[51_865] == f32::NEG_INFINITY);

        let mut logits = vec![1.0f32; 51_866];
        WhisperTokenSuppressor::for_specials(&v3)
            .with_timestamp_suppression(false)
            .with_vocab_size(51_866)
            .apply(&mut logits);
        assert!(
            logits[50_364] == f32::NEG_INFINITY,
            "notimestamps is banned"
        );
        assert!(logits[50_365].is_finite(), "<|0.00|> is allowed");

        assert_eq!(
            WhisperTokenSuppressor::new().suppressed_tokens(),
            WhisperTokenSuppressor::for_specials(&SpecialTokens::multilingual())
                .suppressed_tokens()
        );
    }
}

#[cfg(test)]
mod timestamp_rules_tests {
    use super::*;

    const N_VOCAB: usize = 51866; // large-v3 / large-v3-turbo
    const EOT: usize = 50257;
    const NO_TS: usize = 50364;
    const TB: usize = 50365;
    const PERIOD: usize = 13; // "."
    const BLANK: usize = 220; // " "
    const TEXT: u32 = 400;
    /// SOT, language, task: sampling begins at index 3.
    const PROMPT: [u32; 3] = [50258, 50259, 50360];

    fn rules() -> TimestampRules {
        TimestampRules::for_vocab_size(N_VOCAB, PROMPT.len()).with_blank_tokens(vec![BLANK as u32])
    }

    fn seq(sampled: &[u32]) -> Vec<u32> {
        PROMPT.iter().chain(sampled).copied().collect()
    }

    fn ts(index: u32) -> u32 {
        TB as u32 + index
    }

    fn flat() -> Vec<f32> {
        vec![0.0; N_VOCAB]
    }

    fn argmax(logits: &[f32]) -> usize {
        crate::inference::GreedyDecoder::new(1).decode_step(logits) as usize
    }

    fn banned(logits: &[f32], range: core::ops::Range<usize>) -> bool {
        logits[range].iter().all(|v| *v == f32::NEG_INFINITY)
    }

    fn allowed(logits: &[f32], range: core::ops::Range<usize>) -> bool {
        logits[range].iter().all(|v| v.is_finite())
    }

    #[test]
    fn special_token_ids_follow_the_vocabulary() {
        let v3 = TimestampRules::for_vocab_size(51866, 0);
        assert_eq!(
            (v3.no_timestamps, v3.timestamp_begin, v3.eot),
            (50364, 50365, 50257)
        );
        let v2 = TimestampRules::for_vocab_size(51865, 0);
        assert_eq!(
            (v2.no_timestamps, v2.timestamp_begin, v2.eot),
            (50363, 50364, 50257)
        );
        let en = TimestampRules::for_vocab_size(51864, 0);
        assert_eq!(
            (en.no_timestamps, en.timestamp_begin, en.eot),
            (50362, 50363, 50256)
        );
    }

    #[test]
    fn the_no_timestamps_mode_is_left_unchanged() {
        assert!(TimestampRules::for_prompt(N_VOCAB, &PROMPT).is_some());
        let no_ts: Vec<u32> = PROMPT.iter().copied().chain([NO_TS as u32]).collect();
        assert_eq!(TimestampRules::for_prompt(N_VOCAB, &no_ts), None);
    }

    /// FALSIFY-TSRULE-001 — the first sampled token is a timestamp at or
    /// below max_initial_timestamp (1.0s), and <|notimestamps|> is never
    /// sampled.
    #[test]
    fn the_first_token_is_a_timestamp_at_most_one_second() {
        let mut logits = flat();
        rules().apply(&seq(&[]), &mut logits);
        assert!(banned(&logits, 0..TB), "a non-timestamp survived step 0");
        assert!(
            allowed(&logits, TB..TB + 51),
            "<|0.00|>..<|1.00|> must stay"
        );
        assert!(
            banned(&logits, TB + 51..N_VOCAB),
            "a timestamp > 1.0s survived"
        );
        // Later steps keep <|notimestamps|> banned but no longer cap timestamps.
        let mut logits = flat();
        rules().apply(&seq(&[ts(0), TEXT]), &mut logits);
        assert_eq!(logits[NO_TS], f32::NEG_INFINITY);
        assert!(
            allowed(&logits, TB + 51..N_VOCAB),
            "the cap applies only at step 0"
        );
    }

    /// FALSIFY-TSRULE-002 — timestamps come in pairs: after a lone timestamp
    /// that follows text the next token is a timestamp or EOT; after a pair
    /// (or a first-token timestamp) it is not a timestamp.
    #[test]
    fn timestamps_come_in_pairs() {
        // text, <|5.00|>: a closing timestamp -> no text, EOT allowed. EOT is
        // made likely so the dominance rule does not ban it too.
        let mut logits = flat();
        logits[EOT] = 10.0;
        rules().apply(&seq(&[ts(0), TEXT, ts(250)]), &mut logits);
        assert!(banned(&logits, 0..EOT), "text after a lone timestamp");
        assert!(logits[EOT].is_finite(), "EOT must stay allowed");
        assert!(
            allowed(&logits, TB + 250..N_VOCAB),
            "the opening timestamp must stay"
        );
        // <|5.00|><|5.00|>: a pair -> no timestamp.
        let mut logits = flat();
        rules().apply(&seq(&[ts(0), TEXT, ts(250), ts(250)]), &mut logits);
        assert!(banned(&logits, TB..N_VOCAB), "a third timestamp in a row");
        assert!(allowed(&logits, 0..EOT), "text must follow a pair");
        // A first-token timestamp counts as a pair with the prompt.
        let mut logits = flat();
        rules().apply(&seq(&[ts(0)]), &mut logits);
        assert!(banned(&logits, TB..N_VOCAB));
        assert!(allowed(&logits, 0..EOT));
    }

    /// FALSIFY-TSRULE-003 — timestamps never decrease; a timestamp equal to
    /// the last is allowed only right after it closes a segment.
    #[test]
    fn timestamps_never_decrease() {
        let mut logits = flat();
        rules().apply(&seq(&[ts(0), TEXT, ts(250)]), &mut logits);
        assert!(banned(&logits, TB..TB + 250), "a timestamp below the last");
        assert!(
            logits[TB + 250].is_finite(),
            "the next segment may start where this ended"
        );
        // After text, the closing timestamp must exceed the opening one.
        let mut logits = flat();
        rules().apply(&seq(&[ts(0), TEXT, ts(250), ts(250), TEXT]), &mut logits);
        assert!(
            banned(&logits, TB..TB + 251),
            "a zero-length or backward segment"
        );
        assert!(allowed(&logits, TB + 251..N_VOCAB));
    }

    /// FALSIFY-TSRULE-004 — when the timestamps' total probability beats the
    /// best text token, a timestamp is sampled.
    #[test]
    fn timestamp_probability_mass_beats_the_best_text_token() {
        let mut logits = flat();
        logits[TEXT as usize] = 5.0; // best single token, but ~1500 timestamps at 0.0
        let mut out = logits.clone();
        rules().apply(&seq(&[ts(0), TEXT]), &mut out);
        assert!(
            banned(&out, 0..TB),
            "text survived a dominant timestamp mass"
        );
        assert!(argmax(&out) >= TB);
        // Anti-vacuity: weak timestamps leave text alone.
        let mut weak = logits;
        weak[TB..].fill(-20.0);
        rules().apply(&seq(&[ts(0), TEXT]), &mut weak);
        assert_eq!(argmax(&weak), TEXT as usize);
    }

    /// FALSIFY-TSRULE-005 — blank and EOT are banned at the first sampled
    /// step only.
    #[test]
    fn blank_and_eot_are_banned_at_step_zero() {
        let mut logits = flat();
        rules().apply(&seq(&[]), &mut logits);
        assert_eq!(logits[BLANK], f32::NEG_INFINITY);
        assert_eq!(logits[EOT], f32::NEG_INFINITY);
        let mut logits = flat();
        logits[TB..].fill(-20.0);
        rules().apply(&seq(&[ts(0), TEXT]), &mut logits);
        assert!(
            logits[BLANK].is_finite() && logits[EOT].is_finite(),
            "later steps keep them"
        );
    }

    /// FALSIFY-TSRULE-006 — the issue #106 table: a 30s window ending in
    /// silence put "." (then EOT) on top at step 0, and the decode emitted
    /// only ".". With the rules the first sampled token is a timestamp.
    #[test]
    fn a_leading_period_at_step_zero_yields_a_timestamp() {
        let mut logits = vec![-10.0; N_VOCAB];
        logits[PERIOD] = 12.0;
        logits[EOT] = 11.0;
        logits[BLANK] = 10.0;
        logits[TB] = 1.0; // <|0.00|>
        logits[TB + 400] = 13.0; // <|8.00|>: past max_initial_timestamp
        assert_eq!(argmax(&logits), TB + 400, "the fixture must start wrong");
        let mut raw = logits.clone();
        raw[TB + 400] = -10.0;
        assert_eq!(argmax(&raw), PERIOD, "without the rules \".\" wins");
        rules().apply(&seq(&[]), &mut logits);
        let first = argmax(&logits);
        assert!((TB..=TB + 50).contains(&first), "step 0 sampled {first}");
    }
}

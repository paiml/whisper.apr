//! Vocabulary handling
//!
//! Manages BPE vocabulary and special tokens for Whisper tokenization.
//!
//! # Overview
//!
//! Whisper uses GPT-2 style BPE tokenization with UTF-8 byte encoding.
//! The vocabulary contains:
//! - Base tokens (0-255): Individual bytes
//! - Merged tokens: BPE merge results
//! - Special tokens: Control tokens for decoding

use std::collections::HashMap;

#[cfg(test)]
mod tests;

/// Special token IDs for Whisper
///
/// These tokens control the decoder's behavior during transcription.
///
/// IMPORTANT: Whisper has three special-token layouts, told apart by the
/// vocabulary size:
/// - English-only models (tiny.en .. medium.en), 51864 tokens: EOT=50256
/// - Multilingual v1/v2 models, 51865 tokens: EOT=50257, `<|0.00|>`=50364
/// - large-v3 and large-v3-turbo, 51866 tokens: a 100th language moves every
///   id from translate on up by one (`<|transcribe|>`=50360, `<|0.00|>`=50365)
///
/// Use [`SpecialTokens::from_vocab_size`] (or `try_for_vocab_size`) for every
/// id. The plain constants and free functions below are the v2 (51865)
/// layout only and are kept for the public API; they are wrong on v3.
pub mod special_tokens {
    /// Vocabulary size threshold for multilingual models
    /// Models with vocab >= 51865 are multilingual
    pub const MULTILINGUAL_VOCAB_THRESHOLD: usize = 51865;

    // =========================================================================
    // English-only model tokens (GPT-2 tokenizer)
    // =========================================================================

    /// End of text token for English-only models
    pub const EOT_ENGLISH: u32 = 50256;
    /// Start of transcript token for English-only models
    pub const SOT_ENGLISH: u32 = 50257;

    // =========================================================================
    // Multilingual model tokens (extended tokenizer)
    // =========================================================================

    /// End of text token for multilingual models
    pub const EOT_MULTILINGUAL: u32 = 50257;
    /// Start of transcript token for multilingual models
    pub const SOT_MULTILINGUAL: u32 = 50258;
    /// Language token base for multilingual - language ID is LANG_BASE + lang_offset
    pub const LANG_BASE_MULTILINGUAL: u32 = 50259;
    /// Transcribe task token for multilingual
    pub const TRANSCRIBE_MULTILINGUAL: u32 = 50359;
    /// No timestamps token for multilingual
    pub const NO_TIMESTAMPS_MULTILINGUAL: u32 = 50363;

    // =========================================================================
    // Legacy constants: the multilingual v1/v2 (51865) layout ONLY. They are
    // wrong on large-v3/turbo (51866) and on English-only models; kept for
    // the public API. Use SpecialTokens::from_vocab_size() for every id.
    // =========================================================================

    /// End of text token - signals end of transcription
    /// WARNING: This is for multilingual models. Use SpecialTokens for English-only.
    pub const EOT: u32 = EOT_MULTILINGUAL;
    /// Start of transcript token - begins transcription
    pub const SOT: u32 = SOT_MULTILINGUAL;
    /// Language token base - language ID is LANG_BASE + lang_offset
    pub const LANG_BASE: u32 = LANG_BASE_MULTILINGUAL;
    /// Translate task token - translate audio to English
    pub const TRANSLATE: u32 = 50358;
    /// Transcribe task token - transcribe audio in original language
    pub const TRANSCRIBE: u32 = TRANSCRIBE_MULTILINGUAL;
    /// Speaker turn marker - used by tinydiarize models
    pub const SPEAKER_TURN: u32 = 50360;
    /// Previous context token
    pub const PREV: u32 = 50361;
    /// No speech token - indicates silence/no speech detected
    pub const NO_SPEECH: u32 = 50362;
    /// No timestamps token - disable timestamp generation
    pub const NO_TIMESTAMPS: u32 = NO_TIMESTAMPS_MULTILINGUAL;
    /// Begin timestamps token / Timestamp token base
    pub const TIMESTAMP_BASE: u32 = 50364;

    /// Vocabulary size of the English-only models (tiny.en .. medium.en).
    pub const ENGLISH_ONLY_VOCAB: usize = 51864;
    /// Vocabulary size of the multilingual v1/v2 models (99 languages).
    pub const MULTILINGUAL_V2_VOCAB: usize = 51865;
    /// Vocabulary size of large-v3 and large-v3-turbo (100 languages: `<|yue|>`
    /// was added, which moves every special id after the language block up
    /// by one).
    pub const MULTILINGUAL_V3_VOCAB: usize = 51866;

    /// Every special token id of one Whisper vocabulary — the single source
    /// for special ids (issue #108). Build it with [`SpecialTokens::from_vocab_size`]
    /// (strict) or [`SpecialTokens::try_for_vocab_size`]; never mix it with
    /// the v2-only legacy constants above.
    ///
    /// | id | 51864 (.en) | 51865 (v1/v2) | 51866 (v3, turbo) |
    /// |---|---|---|---|
    /// | eot | 50256 | 50257 | 50257 |
    /// | sot | 50257 | 50258 | 50258 |
    /// | lang_base | 50258 | 50259 | 50259 |
    /// | translate | 50357 | 50358 | 50359 |
    /// | transcribe | 50358 | 50359 | 50360 |
    /// | start_of_lm | 50359 | 50360 | 50361 |
    /// | prev | 50360 | 50361 | 50362 |
    /// | no_speech | 50361 | 50362 | 50363 |
    /// | no_timestamps | 50362 | 50363 | 50364 |
    /// | timestamp_base | 50363 | 50364 | 50365 |
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SpecialTokens {
        /// End of text token
        pub eot: u32,
        /// Start of transcript token
        pub sot: u32,
        /// Language token base (`<|en|>`)
        pub lang_base: u32,
        /// Number of language tokens (99, or 100 on v3)
        pub n_languages: u32,
        /// Translate task token
        pub translate: u32,
        /// Transcribe task token
        pub transcribe: u32,
        /// `<|startoflm|>` (the tinydiarize speaker-turn token on v2)
        pub start_of_lm: u32,
        /// `<|startofprev|>`: precedes the prompt tokens
        pub prev: u32,
        /// `<|nospeech|>` (`<|nocaptions|>` before v3)
        pub no_speech: u32,
        /// No timestamps token
        pub no_timestamps: u32,
        /// Timestamp base token (`<|0.00|>`)
        pub timestamp_base: u32,
        /// Whether this is a multilingual model
        pub is_multilingual: bool,
    }

    impl SpecialTokens {
        /// The ids laid out after `eot` exactly as Whisper's tokenizer does:
        /// sot, the language block, then translate .. notimestamps, then the
        /// 1501 timestamps.
        const fn layout(eot: u32, n_languages: u32, is_multilingual: bool) -> Self {
            let lang_base = eot + 2;
            let translate = lang_base + n_languages;
            Self {
                eot,
                sot: eot + 1,
                lang_base,
                n_languages,
                translate,
                transcribe: translate + 1,
                start_of_lm: translate + 2,
                prev: translate + 3,
                no_speech: translate + 4,
                no_timestamps: translate + 5,
                timestamp_base: translate + 6,
                is_multilingual,
            }
        }

        /// The special ids of a known Whisper vocabulary size (51864, 51865
        /// or 51866), or `None` for any other size.
        #[must_use]
        #[provable_contracts_macros::contract("whisper-special-tokens-v1", equation = "id_table")]
        pub const fn from_vocab_size(n_vocab: usize) -> Option<Self> {
            match n_vocab {
                ENGLISH_ONLY_VOCAB => Some(Self::english_only()),
                MULTILINGUAL_V2_VOCAB => Some(Self::multilingual()),
                MULTILINGUAL_V3_VOCAB => Some(Self::multilingual_v3()),
                _ => None,
            }
        }

        /// [`SpecialTokens::from_vocab_size`], refusing an unknown size.
        ///
        /// # Errors
        /// `WhisperError::Model` when `n_vocab` is not 51864, 51865 or 51866.
        #[provable_contracts_macros::contract("whisper-special-tokens-v1", equation = "refusal")]
        pub fn try_for_vocab_size(n_vocab: usize) -> crate::error::WhisperResult<Self> {
            Self::from_vocab_size(n_vocab).ok_or_else(|| {
                crate::error::WhisperError::Model(format!(
                    "unsupported Whisper vocabulary size {n_vocab}: expected \
                     {ENGLISH_ONLY_VOCAB} (English-only), {MULTILINGUAL_V2_VOCAB} \
                     (multilingual v1/v2) or {MULTILINGUAL_V3_VOCAB} (large-v3/turbo)"
                ))
            })
        }

        /// Special tokens for an already validated vocabulary size.
        ///
        /// Exact for 51864, 51865 and 51866. Model loading refuses any other
        /// size, so another size only reaches this from an in-memory test
        /// config; it then falls back to the v2 layout (>= 51865) or the
        /// English-only layout.
        #[must_use]
        pub fn for_vocab_size(n_vocab: usize) -> Self {
            Self::from_vocab_size(n_vocab).unwrap_or(if n_vocab >= MULTILINGUAL_VOCAB_THRESHOLD {
                Self::multilingual()
            } else {
                Self::english_only()
            })
        }

        /// Special tokens for multilingual v1/v2 models (51865 tokens)
        #[must_use]
        pub const fn multilingual() -> Self {
            Self::layout(EOT_MULTILINGUAL, 99, true)
        }

        /// Special tokens for large-v3 and large-v3-turbo (51866 tokens)
        #[must_use]
        pub const fn multilingual_v3() -> Self {
            Self::layout(EOT_MULTILINGUAL, 100, true)
        }

        /// Special tokens for English-only models (51864 tokens)
        #[must_use]
        pub const fn english_only() -> Self {
            Self::layout(EOT_ENGLISH, 99, false)
        }

        /// Get initial tokens for transcription
        ///
        /// Returns [SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS]
        #[must_use]
        pub fn initial_tokens(&self) -> [u32; 4] {
            [
                self.sot,
                self.lang_base, // English (lang_base + 0)
                self.transcribe,
                self.no_timestamps,
            ]
        }

        /// Language token for a language code, if this vocabulary has it.
        #[must_use]
        pub fn language_token(&self, lang_code: &str) -> Option<u32> {
            language_offset(lang_code)
                .filter(|&offset| offset < self.n_languages)
                .map(|offset| self.lang_base + offset)
        }

        /// Whether `token` is a language token.
        #[must_use]
        pub const fn is_language_token(&self, token: u32) -> bool {
            token >= self.lang_base && token < self.translate
        }

        /// Whether `token` is a timestamp token.
        #[must_use]
        pub const fn is_timestamp(&self, token: u32) -> bool {
            token >= self.timestamp_base
        }

        /// Timestamp token to seconds (20ms steps).
        #[must_use]
        pub fn timestamp_to_seconds(&self, token: u32) -> Option<f32> {
            self.is_timestamp(token)
                .then(|| (token - self.timestamp_base) as f32 * 0.02)
        }

        /// Whether `token` is a control token: eot, sot, a language, a task
        /// token, `<|startoflm|>`, `<|startofprev|>`, `<|nospeech|>` or
        /// `<|notimestamps|>` — every special id below the timestamps.
        #[must_use]
        pub const fn is_control_token(&self, token: u32) -> bool {
            token >= self.eot && token < self.timestamp_base
        }

        /// The token's name, for the special ids this table names.
        #[must_use]
        pub fn name_of(&self, token: u32) -> Option<&'static str> {
            let names = [
                (self.eot, "<|endoftext|>"),
                (self.sot, "<|startoftranscript|>"),
                (self.translate, "<|translate|>"),
                (self.transcribe, "<|transcribe|>"),
                (self.start_of_lm, "<|startoflm|>"),
                (self.prev, "<|startofprev|>"),
                (self.no_speech, "<|nospeech|>"),
                (self.no_timestamps, "<|notimestamps|>"),
                (self.timestamp_base, "<|0.00|>"),
            ];
            names.iter().find(|(id, _)| *id == token).map(|(_, n)| *n)
        }
    }

    impl Default for SpecialTokens {
        fn default() -> Self {
            Self::multilingual()
        }
    }

    /// Get language token ID for a language code
    ///
    /// v2 (51865) layout only; use [`SpecialTokens::language_token`].
    ///
    /// # Arguments
    /// * `lang_code` - Two-letter ISO 639-1 language code (e.g., "en", "es", "ja")
    ///
    /// # Returns
    /// Token ID for the language, or None if unsupported
    #[must_use]
    pub fn language_token(lang_code: &str) -> Option<u32> {
        language_offset(lang_code).map(|offset| LANG_BASE + offset)
    }

    /// Check if a token ID is a timestamp token
    ///
    /// v2 (51865) layout only; use [`SpecialTokens::is_timestamp`].
    #[must_use]
    pub const fn is_timestamp(token_id: u32) -> bool {
        token_id >= TIMESTAMP_BASE
    }

    /// Convert timestamp token to time in seconds
    ///
    /// Timestamps are in 20ms increments (50 per second)
    ///
    /// v2 (51865) layout only; use [`SpecialTokens::timestamp_to_seconds`].
    #[must_use]
    pub fn timestamp_to_seconds(token_id: u32) -> Option<f32> {
        if token_id >= TIMESTAMP_BASE {
            Some((token_id - TIMESTAMP_BASE) as f32 * 0.02)
        } else {
            None
        }
    }

    /// Get language offset for a language code
    ///
    /// Returns the offset from LANG_BASE (0 for English, 1 for Chinese, etc.)
    /// Use with SpecialTokens::lang_base to compute the actual token ID.
    ///
    /// # Arguments
    /// * `lang_code` - Two-letter ISO 639-1 language code (e.g., "en", "es", "ja")
    ///
    /// # Returns
    /// Language offset, or None if unsupported
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn language_offset(lang_code: &str) -> Option<u32> {
        crate::detection::SUPPORTED_LANGUAGES
            .iter()
            .position(|&c| c == lang_code)
            .map(|i| i as u32)
    }
}

/// BPE merge rule
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MergeRule {
    /// First token in the pair
    pub first: Vec<u8>,
    /// Second token in the pair
    pub second: Vec<u8>,
}

impl MergeRule {
    /// Create a new merge rule
    #[must_use]
    pub fn new(first: Vec<u8>, second: Vec<u8>) -> Self {
        Self { first, second }
    }

    /// Get the merged result
    #[must_use]
    pub fn merged(&self) -> Vec<u8> {
        let mut result = self.first.clone();
        result.extend_from_slice(&self.second);
        result
    }
}

/// Vocabulary for BPE tokenization
///
/// Contains token-to-bytes mappings and merge rules for encoding/decoding.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Token ID to byte sequence mapping
    id_to_bytes: Vec<Vec<u8>>,
    /// Byte sequence to token ID mapping (for encoding)
    bytes_to_id: HashMap<Vec<u8>, u32>,
    /// BPE merge rules in priority order
    merge_rules: Vec<MergeRule>,
    /// Merge lookup for fast pair checking
    merge_lookup: HashMap<(Vec<u8>, Vec<u8>), u32>,
}

impl Vocabulary {
    /// Create a new empty vocabulary
    #[must_use]
    pub fn new() -> Self {
        Self {
            id_to_bytes: Vec::new(),
            bytes_to_id: HashMap::new(),
            merge_rules: Vec::new(),
            merge_lookup: HashMap::new(),
        }
    }

    /// Create a vocabulary with base byte tokens (0-255)
    ///
    /// This initializes the vocabulary with single-byte tokens.
    #[must_use]
    pub fn with_base_tokens() -> Self {
        let mut vocab = Self::new();

        // Add single byte tokens (0-255)
        for byte in 0..=255u8 {
            vocab.add_token(vec![byte]);
        }

        vocab
    }

    /// Add a token to the vocabulary
    ///
    /// Returns the token ID assigned to this token.
    pub fn add_token(&mut self, bytes: Vec<u8>) -> u32 {
        let id = self.id_to_bytes.len() as u32;
        self.bytes_to_id.insert(bytes.clone(), id);
        self.id_to_bytes.push(bytes);
        id
    }

    /// Add a merge rule
    ///
    /// # Arguments
    /// * `first` - First token bytes
    /// * `second` - Second token bytes
    ///
    /// # Returns
    /// The token ID of the merged result
    pub fn add_merge(&mut self, first: Vec<u8>, second: Vec<u8>) -> u32 {
        let rule = MergeRule::new(first.clone(), second.clone());
        let merged = rule.merged();

        // Add the merged token if it doesn't exist
        let merged_id = if let Some(&id) = self.bytes_to_id.get(&merged) {
            id
        } else {
            self.add_token(merged)
        };

        // Add to merge lookup
        self.merge_lookup.insert((first, second), merged_id);
        self.merge_rules.push(rule);

        merged_id
    }

    /// Get vocabulary size
    #[must_use]
    pub fn len(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Check if vocabulary is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id_to_bytes.is_empty()
    }

    /// Get token bytes by ID
    #[must_use]
    pub fn get_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.id_to_bytes.get(token_id as usize).map(Vec::as_slice)
    }

    /// Get token ID by bytes
    #[must_use]
    pub fn get_id(&self, bytes: &[u8]) -> Option<u32> {
        self.bytes_to_id.get(bytes).copied()
    }

    /// Check if a merge exists for the given pair
    #[must_use]
    pub fn get_merge(&self, first: &[u8], second: &[u8]) -> Option<u32> {
        self.merge_lookup
            .get(&(first.to_vec(), second.to_vec()))
            .copied()
    }

    /// Get merge priority (lower is higher priority)
    #[must_use]
    pub fn merge_priority(&self, first: &[u8], second: &[u8]) -> Option<usize> {
        self.merge_rules
            .iter()
            .position(|r| r.first.as_slice() == first && r.second.as_slice() == second)
    }

    /// Decode token IDs to string
    ///
    /// # Arguments
    /// * `tokens` - Token IDs to decode
    ///
    /// # Returns
    /// Decoded string, or None if any token is invalid
    #[must_use]
    pub fn decode(&self, tokens: &[u32]) -> Option<String> {
        if tokens.is_empty() {
            return Some(String::new());
        }

        // Collect all bytes
        let mut bytes = Vec::new();
        for &token_id in tokens {
            // Skip special tokens for text output
            if token_id >= special_tokens::EOT {
                continue;
            }
            let token_bytes = self.get_bytes(token_id)?;
            bytes.extend_from_slice(token_bytes);
        }

        // Convert bytes to UTF-8 string (lossy conversion for robustness)
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Get number of merge rules
    #[must_use]
    pub fn num_merges(&self) -> usize {
        self.merge_rules.len()
    }

    /// Serialize vocabulary to bytes
    ///
    /// Format:
    /// - u32: number of tokens
    /// - u32: number of merge rules
    /// - For each token: u16 len, bytes
    /// - For each merge: u16 first_len, first_bytes, u16 second_len, second_bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write token count and merge count
        bytes.extend_from_slice(&(self.id_to_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.merge_rules.len() as u32).to_le_bytes());

        // Write tokens
        for token_bytes in &self.id_to_bytes {
            let len = token_bytes.len() as u16;
            bytes.extend_from_slice(&len.to_le_bytes());
            bytes.extend_from_slice(token_bytes);
        }

        // Write merge rules
        for rule in &self.merge_rules {
            let first_len = rule.first.len() as u16;
            bytes.extend_from_slice(&first_len.to_le_bytes());
            bytes.extend_from_slice(&rule.first);

            let second_len = rule.second.len() as u16;
            bytes.extend_from_slice(&second_len.to_le_bytes());
            bytes.extend_from_slice(&rule.second);
        }

        bytes
    }

    /// Read a length-prefixed byte sequence from data at the given offset
    fn read_length_prefixed(data: &[u8], offset: &mut usize) -> Option<Vec<u8>> {
        if *offset + 2 > data.len() {
            return None;
        }
        let len = u16::from_le_bytes([data[*offset], data[*offset + 1]]) as usize;
        *offset += 2;
        if *offset + len > data.len() {
            return None;
        }
        let bytes = data[*offset..*offset + len].to_vec();
        *offset += len;
        Some(bytes)
    }

    /// Deserialize vocabulary from bytes
    ///
    /// # Errors
    /// Returns None if parsing fails
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }

        let n_tokens = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let n_merges = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        let mut offset = 8;
        let mut vocab = Self::new();

        // Read tokens
        for _ in 0..n_tokens {
            let token_bytes = Self::read_length_prefixed(data, &mut offset)?;
            vocab.add_token(token_bytes);
        }

        // Read merge rules
        for _ in 0..n_merges {
            let first = Self::read_length_prefixed(data, &mut offset)?;
            let second = Self::read_length_prefixed(data, &mut offset)?;

            // Add merge (this also adds the merged token if not exists)
            vocab.merge_lookup.insert(
                (first.clone(), second.clone()),
                vocab.id_to_bytes.len() as u32,
            );
            vocab.merge_rules.push(MergeRule::new(first, second));
        }

        Some(vocab)
    }

    /// Get merge rules reference
    #[must_use]
    pub fn merge_rules(&self) -> &[MergeRule] {
        &self.merge_rules
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

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
/// IMPORTANT: Whisper has two tokenizer variants:
/// - English-only models (tiny.en, base.en, etc.): GPT-2 tokenizer, EOT=50256
/// - Multilingual models (tiny, base, etc.): Extended tokenizer, EOT=50257
///
/// Use `SpecialTokens::for_vocab_size(n_vocab)` to get correct token IDs.
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
    // Legacy constants (for backwards compatibility, assume multilingual)
    // Use SpecialTokens::for_vocab_size() for new code
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

    /// Dynamic special token lookup based on vocabulary size
    ///
    /// Whisper has two tokenizer variants with different token IDs:
    /// - English-only models use GPT-2 tokenizer (EOT=50256)
    /// - Multilingual models use extended tokenizer (EOT=50257)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SpecialTokens {
        /// End of text token
        pub eot: u32,
        /// Start of transcript token
        pub sot: u32,
        /// Language token base
        pub lang_base: u32,
        /// Transcribe task token
        pub transcribe: u32,
        /// No timestamps token
        pub no_timestamps: u32,
        /// Timestamp base token
        pub timestamp_base: u32,
        /// Whether this is a multilingual model
        pub is_multilingual: bool,
    }

    impl SpecialTokens {
        /// Create special tokens for the given vocabulary size
        ///
        /// # Arguments
        /// * `n_vocab` - Vocabulary size of the model
        ///
        /// # Returns
        /// Special tokens configured for the model type
        #[must_use]
        pub fn for_vocab_size(n_vocab: usize) -> Self {
            if n_vocab >= MULTILINGUAL_VOCAB_THRESHOLD {
                Self::multilingual()
            } else {
                Self::english_only()
            }
        }

        /// Special tokens for multilingual models
        #[must_use]
        pub const fn multilingual() -> Self {
            Self {
                eot: EOT_MULTILINGUAL,
                sot: SOT_MULTILINGUAL,
                lang_base: LANG_BASE_MULTILINGUAL,
                transcribe: TRANSCRIBE_MULTILINGUAL,
                no_timestamps: NO_TIMESTAMPS_MULTILINGUAL,
                timestamp_base: 50364,
                is_multilingual: true,
            }
        }

        /// Special tokens for English-only models
        #[must_use]
        pub const fn english_only() -> Self {
            Self {
                eot: EOT_ENGLISH,
                sot: SOT_ENGLISH,
                lang_base: 50258, // Same offset structure
                transcribe: 50358,
                no_timestamps: 50362,
                timestamp_base: 50363,
                is_multilingual: false,
            }
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
    }

    impl Default for SpecialTokens {
        fn default() -> Self {
            Self::multilingual()
        }
    }

    /// Get language token ID for a language code
    ///
    /// # Arguments
    /// * `lang_code` - Two-letter ISO 639-1 language code (e.g., "en", "es", "ja")
    ///
    /// # Returns
    /// Token ID for the language, or None if unsupported
    #[must_use]
    pub fn language_token(lang_code: &str) -> Option<u32> {
        // Language indices (Whisper's 99 supported languages)
        let lang_offset = match lang_code {
            "en" => 0,
            "zh" => 1,
            "de" => 2,
            "es" => 3,
            "ru" => 4,
            "ko" => 5,
            "fr" => 6,
            "ja" => 7,
            "pt" => 8,
            "tr" => 9,
            "pl" => 10,
            "ca" => 11,
            "nl" => 12,
            "ar" => 13,
            "sv" => 14,
            "it" => 15,
            "id" => 16,
            "hi" => 17,
            "fi" => 18,
            "vi" => 19,
            "he" => 20,
            "uk" => 21,
            "el" => 22,
            "ms" => 23,
            "cs" => 24,
            "ro" => 25,
            "da" => 26,
            "hu" => 27,
            "ta" => 28,
            "no" => 29,
            "th" => 30,
            "ur" => 31,
            "hr" => 32,
            "bg" => 33,
            "lt" => 34,
            "la" => 35,
            "mi" => 36,
            "ml" => 37,
            "cy" => 38,
            "sk" => 39,
            "te" => 40,
            "fa" => 41,
            "lv" => 42,
            "bn" => 43,
            "sr" => 44,
            "az" => 45,
            "sl" => 46,
            "kn" => 47,
            "et" => 48,
            "mk" => 49,
            // Additional languages...
            _ => return None,
        };
        Some(LANG_BASE + lang_offset)
    }

    /// Check if a token ID is a timestamp token
    #[must_use]
    pub const fn is_timestamp(token_id: u32) -> bool {
        token_id >= TIMESTAMP_BASE
    }

    /// Convert timestamp token to time in seconds
    ///
    /// Timestamps are in 20ms increments (50 per second)
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
    pub fn language_offset(lang_code: &str) -> Option<u32> {
        match lang_code {
            "en" => Some(0),
            "zh" => Some(1),
            "de" => Some(2),
            "es" => Some(3),
            "ru" => Some(4),
            "ko" => Some(5),
            "fr" => Some(6),
            "ja" => Some(7),
            "pt" => Some(8),
            "tr" => Some(9),
            "pl" => Some(10),
            "ca" => Some(11),
            "nl" => Some(12),
            "ar" => Some(13),
            "sv" => Some(14),
            "it" => Some(15),
            "id" => Some(16),
            "hi" => Some(17),
            "fi" => Some(18),
            "vi" => Some(19),
            "he" => Some(20),
            "uk" => Some(21),
            "el" => Some(22),
            "ms" => Some(23),
            "cs" => Some(24),
            "ro" => Some(25),
            "da" => Some(26),
            "hu" => Some(27),
            "ta" => Some(28),
            "no" => Some(29),
            "th" => Some(30),
            "ur" => Some(31),
            "hr" => Some(32),
            "bg" => Some(33),
            "lt" => Some(34),
            "la" => Some(35),
            "mi" => Some(36),
            "ml" => Some(37),
            _ => None,
        }
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
            if offset + 2 > data.len() {
                return None;
            }
            let len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + len > data.len() {
                return None;
            }
            let token_bytes = data[offset..offset + len].to_vec();
            offset += len;

            vocab.add_token(token_bytes);
        }

        // Read merge rules
        for _ in 0..n_merges {
            // Read first
            if offset + 2 > data.len() {
                return None;
            }
            let first_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + first_len > data.len() {
                return None;
            }
            let first = data[offset..offset + first_len].to_vec();
            offset += first_len;

            // Read second
            if offset + 2 > data.len() {
                return None;
            }
            let second_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + second_len > data.len() {
                return None;
            }
            let second = data[offset..offset + second_len].to_vec();
            offset += second_len;

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

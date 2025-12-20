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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Vocabulary Construction Tests
    // =========================================================================

    #[test]
    fn test_vocabulary_new() {
        let vocab = Vocabulary::new();
        assert!(vocab.is_empty());
        assert_eq!(vocab.len(), 0);
    }

    #[test]
    fn test_vocabulary_with_base_tokens() {
        let vocab = Vocabulary::with_base_tokens();
        assert_eq!(vocab.len(), 256);

        // Check some byte tokens
        assert_eq!(vocab.get_bytes(0), Some(&[0u8][..]));
        assert_eq!(vocab.get_bytes(65), Some(&[65u8][..])); // 'A'
        assert_eq!(vocab.get_bytes(255), Some(&[255u8][..]));
    }

    #[test]
    fn test_vocabulary_add_token() {
        let mut vocab = Vocabulary::new();
        let id1 = vocab.add_token(vec![104, 101, 108, 108, 111]); // "hello"
        let id2 = vocab.add_token(vec![119, 111, 114, 108, 100]); // "world"

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_vocabulary_get_bytes() {
        let mut vocab = Vocabulary::new();
        vocab.add_token(vec![104, 105]); // "hi"

        assert_eq!(vocab.get_bytes(0), Some(&[104u8, 105][..]));
        assert_eq!(vocab.get_bytes(1), None);
    }

    #[test]
    fn test_vocabulary_get_id() {
        let mut vocab = Vocabulary::new();
        vocab.add_token(vec![104, 105]); // "hi"

        assert_eq!(vocab.get_id(&[104, 105]), Some(0));
        assert_eq!(vocab.get_id(&[98, 121, 101]), None); // "bye"
    }

    // =========================================================================
    // Merge Rule Tests
    // =========================================================================

    #[test]
    fn test_merge_rule_new() {
        let rule = MergeRule::new(vec![104], vec![105]); // 'h' + 'i'
        assert_eq!(rule.first, vec![104]);
        assert_eq!(rule.second, vec![105]);
    }

    #[test]
    fn test_merge_rule_merged() {
        let rule = MergeRule::new(vec![104], vec![105]);
        assert_eq!(rule.merged(), vec![104, 105]); // "hi"
    }

    #[test]
    fn test_vocabulary_add_merge() {
        let mut vocab = Vocabulary::with_base_tokens();
        let merged_id = vocab.add_merge(vec![104], vec![105]); // 'h' + 'i' -> "hi"

        assert_eq!(merged_id, 256); // First merged token
        assert_eq!(vocab.len(), 257);
        assert_eq!(vocab.get_merge(&[104], &[105]), Some(256));
    }

    #[test]
    fn test_vocabulary_merge_priority() {
        let mut vocab = Vocabulary::with_base_tokens();
        vocab.add_merge(vec![104], vec![105]); // 'h' + 'i' (priority 0)
        vocab.add_merge(vec![116], vec![104]); // 't' + 'h' (priority 1)

        assert_eq!(vocab.merge_priority(&[104], &[105]), Some(0));
        assert_eq!(vocab.merge_priority(&[116], &[104]), Some(1));
        assert_eq!(vocab.merge_priority(&[97], &[98]), None);
    }

    // =========================================================================
    // Decode Tests
    // =========================================================================

    #[test]
    fn test_decode_empty() {
        let vocab = Vocabulary::new();
        let result = vocab.decode(&[]);
        assert_eq!(result, Some(String::new()));
    }

    #[test]
    fn test_decode_single_byte_tokens() {
        let vocab = Vocabulary::with_base_tokens();
        // "hi" = [104, 105]
        let result = vocab.decode(&[104, 105]);
        assert_eq!(result, Some("hi".to_string()));
    }

    #[test]
    fn test_decode_ascii() {
        let vocab = Vocabulary::with_base_tokens();
        // "Hello" = [72, 101, 108, 108, 111]
        let result = vocab.decode(&[72, 101, 108, 108, 111]);
        assert_eq!(result, Some("Hello".to_string()));
    }

    #[test]
    fn test_decode_invalid_token() {
        let vocab = Vocabulary::new();
        let result = vocab.decode(&[0]); // No tokens in empty vocab
        assert_eq!(result, None);
    }

    #[test]
    fn test_decode_skips_special_tokens() {
        let vocab = Vocabulary::with_base_tokens();
        // Include EOT token - should be skipped
        let result = vocab.decode(&[72, 105, special_tokens::EOT]); // "Hi" + EOT
        assert_eq!(result, Some("Hi".to_string()));
    }

    // =========================================================================
    // Special Token Tests
    // =========================================================================

    #[test]
    fn test_special_tokens_values_multilingual() {
        // Default constants are for multilingual models (whisper-tiny, whisper-base, etc.)
        // These match whisper.cpp's multilingual token IDs
        assert_eq!(special_tokens::EOT, 50257); // Multilingual EOT
        assert_eq!(special_tokens::SOT, 50258); // Multilingual SOT
        assert_eq!(special_tokens::LANG_BASE, 50259); // Multilingual lang base
        assert_eq!(special_tokens::TRANSLATE, 50358);
        assert_eq!(special_tokens::TRANSCRIBE, 50359); // Multilingual transcribe
        assert_eq!(special_tokens::SPEAKER_TURN, 50360);
        assert_eq!(special_tokens::PREV, 50361);
        assert_eq!(special_tokens::NO_SPEECH, 50362);
        assert_eq!(special_tokens::NO_TIMESTAMPS, 50363); // Multilingual no_timestamps
        assert_eq!(special_tokens::TIMESTAMP_BASE, 50364); // Multilingual timestamp base
    }

    #[test]
    fn test_special_tokens_english_only() {
        // English-only models (whisper-tiny.en, whisper-base.en) use GPT-2 tokenizer
        assert_eq!(special_tokens::EOT_ENGLISH, 50256);
        assert_eq!(special_tokens::SOT_ENGLISH, 50257);
    }

    #[test]
    fn test_special_tokens_for_vocab_size() {
        use special_tokens::SpecialTokens;

        // Multilingual model (vocab >= 51865)
        let multi = SpecialTokens::for_vocab_size(51865);
        assert!(multi.is_multilingual);
        assert_eq!(multi.eot, 50257);
        assert_eq!(multi.sot, 50258);

        // English-only model (vocab < 51865)
        let english = SpecialTokens::for_vocab_size(51864);
        assert!(!english.is_multilingual);
        assert_eq!(english.eot, 50256);
        assert_eq!(english.sot, 50257);
    }

    #[test]
    fn test_language_token() {
        // English is at LANG_BASE_MULTILINGUAL + 0 = 50259
        assert_eq!(special_tokens::language_token("en"), Some(50259));
        assert_eq!(special_tokens::language_token("zh"), Some(50260));
        assert_eq!(special_tokens::language_token("es"), Some(50262)); // es is index 3
        assert_eq!(special_tokens::language_token("invalid"), None);
    }

    #[test]
    fn test_is_timestamp() {
        assert!(!special_tokens::is_timestamp(50362)); // NO_SPEECH
        assert!(!special_tokens::is_timestamp(50363)); // NO_TIMESTAMPS_MULTILINGUAL
        assert!(special_tokens::is_timestamp(50364)); // TIMESTAMP_BASE (multilingual)
        assert!(special_tokens::is_timestamp(50365)); // First timestamp after base
    }

    #[test]
    fn test_timestamp_to_seconds() {
        // Timestamp 0 = 0.0 seconds
        assert_eq!(
            special_tokens::timestamp_to_seconds(special_tokens::TIMESTAMP_BASE),
            Some(0.0)
        );
        // Timestamp 50 = 1.0 second (50 * 0.02 = 1.0)
        assert_eq!(
            special_tokens::timestamp_to_seconds(special_tokens::TIMESTAMP_BASE + 50),
            Some(1.0)
        );
        // Non-timestamp token
        assert_eq!(special_tokens::timestamp_to_seconds(100), None);
    }

    // =========================================================================
    // Property Tests
    // =========================================================================

    #[test]
    fn test_vocabulary_roundtrip() {
        let mut vocab = Vocabulary::with_base_tokens();

        // Add some merges
        vocab.add_merge(vec![116], vec![104]); // "th"
        vocab.add_merge(vec![116, 104], vec![101]); // "the"

        // Check we can retrieve them
        assert!(vocab.get_merge(&[116], &[104]).is_some());
        assert!(vocab.get_merge(&[116, 104], &[101]).is_some());
    }

    #[test]
    fn test_num_merges() {
        let mut vocab = Vocabulary::with_base_tokens();
        assert_eq!(vocab.num_merges(), 0);

        vocab.add_merge(vec![104], vec![105]);
        assert_eq!(vocab.num_merges(), 1);

        vocab.add_merge(vec![116], vec![104]);
        assert_eq!(vocab.num_merges(), 2);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_vocabulary_to_bytes_empty() {
        let vocab = Vocabulary::new();
        let bytes = vocab.to_bytes();

        // 4 bytes for n_tokens (0) + 4 bytes for n_merges (0)
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[0..4], &0u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &0u32.to_le_bytes());
    }

    #[test]
    fn test_vocabulary_to_bytes_base_tokens() {
        let vocab = Vocabulary::with_base_tokens();
        let bytes = vocab.to_bytes();

        // Check header
        let n_tokens = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let n_merges = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        assert_eq!(n_tokens, 256);
        assert_eq!(n_merges, 0);

        // Total size: 8 header + 256 * (2 len + 1 byte) = 8 + 768 = 776 bytes
        assert_eq!(bytes.len(), 776);
    }

    #[test]
    fn test_vocabulary_roundtrip_empty() {
        let original = Vocabulary::new();
        let bytes = original.to_bytes();
        let restored = Vocabulary::from_bytes(&bytes).expect("should parse");

        assert_eq!(restored.len(), 0);
        assert_eq!(restored.num_merges(), 0);
    }

    #[test]
    fn test_vocabulary_roundtrip_base_tokens() {
        let original = Vocabulary::with_base_tokens();
        let bytes = original.to_bytes();
        let restored = Vocabulary::from_bytes(&bytes).expect("should parse");

        assert_eq!(restored.len(), 256);
        assert_eq!(restored.num_merges(), 0);

        // Verify all base tokens
        for i in 0..256u32 {
            assert_eq!(restored.get_bytes(i), Some(&[i as u8][..]));
        }
    }

    #[test]
    fn test_vocabulary_roundtrip_with_merges() {
        let mut original = Vocabulary::with_base_tokens();
        original.add_merge(vec![104], vec![105]); // "hi"
        original.add_merge(vec![116], vec![104]); // "th"
        original.add_merge(vec![116, 104], vec![101]); // "the"

        let bytes = original.to_bytes();
        let restored = Vocabulary::from_bytes(&bytes).expect("should parse");

        assert_eq!(restored.len(), original.len());
        assert_eq!(restored.num_merges(), 3);

        // Verify merge lookup works
        assert!(restored.merge_priority(&[104], &[105]).is_some());
        assert!(restored.merge_priority(&[116], &[104]).is_some());
        assert!(restored.merge_priority(&[116, 104], &[101]).is_some());
    }

    #[test]
    fn test_vocabulary_from_bytes_too_short() {
        let bytes = vec![0u8; 4]; // Too short
        assert!(Vocabulary::from_bytes(&bytes).is_none());
    }

    #[test]
    fn test_vocabulary_from_bytes_truncated_tokens() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_tokens = 1
        bytes.extend_from_slice(&0u32.to_le_bytes()); // n_merges = 0
        bytes.extend_from_slice(&10u16.to_le_bytes()); // token len = 10
                                                       // But no token data

        assert!(Vocabulary::from_bytes(&bytes).is_none());
    }

    #[test]
    fn test_merge_rules_accessor() {
        let mut vocab = Vocabulary::with_base_tokens();
        vocab.add_merge(vec![104], vec![105]);

        let rules = vocab.merge_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].first, vec![104]);
        assert_eq!(rules[0].second, vec![105]);
    }
}

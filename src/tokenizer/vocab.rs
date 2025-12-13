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
pub mod special_tokens {
    /// End of text token - signals end of transcription
    pub const EOT: u32 = 50257;
    /// Start of transcript token - begins transcription
    pub const SOT: u32 = 50258;
    /// Language token base - language ID is LANG_BASE + lang_offset
    pub const LANG_BASE: u32 = 50259;
    /// Transcribe task token - transcribe audio in original language
    pub const TRANSCRIBE: u32 = 50359;
    /// Translate task token - translate audio to English
    pub const TRANSLATE: u32 = 50358;
    /// No speech token - indicates silence/no speech detected
    pub const NO_SPEECH: u32 = 50362;
    /// No timestamps token - disable timestamp generation
    pub const NO_TIMESTAMPS: u32 = 50363;
    /// Timestamp token base - timestamp tokens start here
    pub const TIMESTAMP_BASE: u32 = 50364;

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
    fn test_special_tokens_values() {
        assert_eq!(special_tokens::EOT, 50257);
        assert_eq!(special_tokens::SOT, 50258);
        assert_eq!(special_tokens::LANG_BASE, 50259);
        assert_eq!(special_tokens::TRANSCRIBE, 50359);
        assert_eq!(special_tokens::TRANSLATE, 50358);
        assert_eq!(special_tokens::NO_SPEECH, 50362);
        assert_eq!(special_tokens::NO_TIMESTAMPS, 50363);
        assert_eq!(special_tokens::TIMESTAMP_BASE, 50364);
    }

    #[test]
    fn test_language_token() {
        assert_eq!(special_tokens::language_token("en"), Some(50259));
        assert_eq!(special_tokens::language_token("zh"), Some(50260));
        assert_eq!(special_tokens::language_token("es"), Some(50262));
        assert_eq!(special_tokens::language_token("invalid"), None);
    }

    #[test]
    fn test_is_timestamp() {
        assert!(!special_tokens::is_timestamp(50363)); // NO_TIMESTAMPS
        assert!(special_tokens::is_timestamp(50364)); // TIMESTAMP_BASE
        assert!(special_tokens::is_timestamp(50365)); // First timestamp
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
}

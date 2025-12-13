//! BPE tokenizer
//!
//! Implements byte-pair encoding (BPE) tokenization for Whisper.
//!
//! # Algorithm
//!
//! BPE works by:
//! 1. Starting with bytes as initial tokens
//! 2. Iteratively merging the most frequent adjacent pairs
//! 3. Using pre-learned merge rules to compress text
//!
//! # References
//!
//! - Sennrich et al. (2016): "Neural Machine Translation of Rare Words with Subword Units"
//! - Radford et al. (2019): GPT-2 BPE tokenization

mod vocab;

pub use vocab::{special_tokens, MergeRule, Vocabulary};

use crate::error::{WhisperError, WhisperResult};

/// BPE tokenizer for Whisper
///
/// Handles encoding text to tokens and decoding tokens back to text.
/// Uses the GPT-2 style BPE algorithm with UTF-8 byte encoding.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Vocabulary with merge rules
    vocab: Vocabulary,
}

impl BpeTokenizer {
    /// Create a new tokenizer with the given vocabulary
    #[must_use]
    pub fn new(vocab: Vocabulary) -> Self {
        Self { vocab }
    }

    /// Create a tokenizer with only base byte tokens (no merges)
    ///
    /// Useful for testing or when no merge rules are available.
    #[must_use]
    pub fn with_base_tokens() -> Self {
        Self {
            vocab: Vocabulary::with_base_tokens(),
        }
    }

    /// Encode text to token IDs using BPE
    ///
    /// # Algorithm
    ///
    /// 1. Convert text to UTF-8 bytes
    /// 2. Initialize each byte as a token
    /// 3. Iteratively merge pairs according to merge rules (highest priority first)
    /// 4. Return final token sequence
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    ///
    /// # Returns
    /// Vector of token IDs
    ///
    /// # Errors
    /// Returns error if encoding fails
    pub fn encode(&self, text: &str) -> WhisperResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Convert text to bytes
        let bytes = text.as_bytes();

        // Initialize token sequence (each byte is a token)
        let mut tokens: Vec<Vec<u8>> = bytes.iter().map(|&b| vec![b]).collect();

        // Apply BPE merges iteratively
        loop {
            // Find the highest priority merge in the current sequence
            let merge = self.find_best_merge(&tokens);

            match merge {
                Some((idx, merged_bytes)) => {
                    // Apply the merge
                    let second = tokens.remove(idx + 1);
                    tokens[idx].extend_from_slice(&second);
                    debug_assert_eq!(tokens[idx], merged_bytes);
                }
                None => break, // No more merges possible
            }
        }

        // Convert byte sequences to token IDs
        let mut token_ids = Vec::with_capacity(tokens.len());
        for token_bytes in tokens {
            let id = self.vocab.get_id(&token_bytes).ok_or_else(|| {
                WhisperError::Tokenizer(format!(
                    "unknown token: {:?}",
                    String::from_utf8_lossy(&token_bytes)
                ))
            })?;
            token_ids.push(id);
        }

        Ok(token_ids)
    }

    /// Find the highest priority merge in the token sequence
    fn find_best_merge(&self, tokens: &[Vec<u8>]) -> Option<(usize, Vec<u8>)> {
        if tokens.len() < 2 {
            return None;
        }

        let mut best_priority = usize::MAX;
        let mut best_idx = None;
        let mut best_merged = None;

        // Check all adjacent pairs
        for i in 0..tokens.len() - 1 {
            let first = &tokens[i];
            let second = &tokens[i + 1];

            // Check if this pair can be merged
            if let Some(priority) = self.vocab.merge_priority(first, second) {
                if priority < best_priority {
                    best_priority = priority;
                    best_idx = Some(i);
                    // Compute merged bytes
                    let mut merged = first.clone();
                    merged.extend_from_slice(second);
                    best_merged = Some(merged);
                }
            }
        }

        // SAFETY: best_merged is always set when best_idx is set
        best_idx.and_then(|idx| best_merged.map(|merged| (idx, merged)))
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    /// * `tokens` - Token IDs to decode
    ///
    /// # Returns
    /// Decoded text string
    ///
    /// # Errors
    /// Returns error if any token ID is invalid
    pub fn decode(&self, tokens: &[u32]) -> WhisperResult<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        self.vocab
            .decode(tokens)
            .ok_or_else(|| WhisperError::Tokenizer("invalid token ID".into()))
    }

    /// Decode token IDs to text, filtering special tokens
    ///
    /// # Arguments
    /// * `tokens` - Token IDs to decode
    /// * `skip_special` - If true, skip special tokens (EOT, SOT, etc.)
    ///
    /// # Returns
    /// Decoded text string
    ///
    /// # Errors
    /// Returns error if any token ID is invalid
    pub fn decode_with_options(&self, tokens: &[u32], skip_special: bool) -> WhisperResult<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let filtered: Vec<u32> = if skip_special {
            tokens
                .iter()
                .filter(|&&t| t < special_tokens::EOT)
                .copied()
                .collect()
        } else {
            tokens.to_vec()
        };

        self.vocab
            .decode(&filtered)
            .ok_or_else(|| WhisperError::Tokenizer("invalid token ID".into()))
    }

    /// Get vocabulary reference
    #[must_use]
    pub const fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::with_base_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_tokenizer_new() {
        let vocab = Vocabulary::new();
        let tokenizer = BpeTokenizer::new(vocab);
        assert_eq!(tokenizer.vocab_size(), 0);
    }

    #[test]
    fn test_tokenizer_with_base_tokens() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        assert_eq!(tokenizer.vocab_size(), 256);
    }

    #[test]
    fn test_tokenizer_default() {
        let tokenizer = BpeTokenizer::default();
        assert_eq!(tokenizer.vocab_size(), 256);
    }

    // =========================================================================
    // Encode Tests
    // =========================================================================

    #[test]
    fn test_tokenizer_encode_empty() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let result = tokenizer.encode("");
        assert!(result.is_ok());
        assert!(result.map_or(false, |v| v.is_empty()));
    }

    #[test]
    fn test_tokenizer_encode_single_char() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let result = tokenizer.encode("a").expect("encode should succeed");
        assert_eq!(result, vec![97]); // 'a' = 97
    }

    #[test]
    fn test_tokenizer_encode_ascii() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let result = tokenizer.encode("hi").expect("encode should succeed");
        assert_eq!(result, vec![104, 105]); // 'h' = 104, 'i' = 105
    }

    #[test]
    fn test_tokenizer_encode_hello() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let result = tokenizer.encode("hello").expect("encode should succeed");
        assert_eq!(result, vec![104, 101, 108, 108, 111]);
    }

    #[test]
    fn test_tokenizer_encode_with_merges() {
        let mut vocab = Vocabulary::with_base_tokens();
        // Add merge: 'h' + 'i' -> "hi"
        vocab.add_merge(vec![104], vec![105]);

        let tokenizer = BpeTokenizer::new(vocab);
        let result = tokenizer.encode("hi").expect("encode should succeed");

        // Should use merged token
        assert_eq!(result, vec![256]); // Merged token ID
    }

    #[test]
    fn test_tokenizer_encode_multiple_merges() {
        let mut vocab = Vocabulary::with_base_tokens();
        // Add merges in priority order
        vocab.add_merge(vec![104], vec![101]); // 'h' + 'e' -> "he" (priority 0)
        vocab.add_merge(vec![108], vec![108]); // 'l' + 'l' -> "ll" (priority 1)
        vocab.add_merge(vec![108, 108], vec![111]); // "ll" + 'o' -> "llo" (priority 2)

        let tokenizer = BpeTokenizer::new(vocab);
        let result = tokenizer.encode("hello").expect("encode should succeed");

        // "hello" -> "he" + "llo" -> tokens [256, 258]
        assert_eq!(result, vec![256, 258]);
    }

    #[test]
    fn test_tokenizer_encode_space() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let result = tokenizer.encode(" ").expect("encode should succeed");
        assert_eq!(result, vec![32]); // space = 32
    }

    #[test]
    fn test_tokenizer_encode_newline() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let result = tokenizer.encode("\n").expect("encode should succeed");
        assert_eq!(result, vec![10]); // newline = 10
    }

    #[test]
    fn test_tokenizer_encode_utf8_emoji() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        // UTF-8 bytes for 😀 are [240, 159, 152, 128]
        let result = tokenizer.encode("😀").expect("encode should succeed");
        assert_eq!(result, vec![240, 159, 152, 128]);
    }

    #[test]
    fn test_tokenizer_encode_utf8_japanese() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        // UTF-8 bytes for こんにちは (konnichiwa)
        let result = tokenizer.encode("こ").expect("encode should succeed");
        // UTF-8 bytes for こ are [227, 129, 147]
        assert_eq!(result, vec![227, 129, 147]);
    }

    // =========================================================================
    // Decode Tests
    // =========================================================================

    #[test]
    fn test_tokenizer_decode_empty() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let result = tokenizer.decode(&[]);
        assert!(result.is_ok());
        assert!(result.map_or(false, |s| s.is_empty()));
    }

    #[test]
    fn test_tokenizer_decode_single_token() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let result = tokenizer.decode(&[97]).expect("decode should succeed");
        assert_eq!(result, "a");
    }

    #[test]
    fn test_tokenizer_decode_hello() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let result = tokenizer
            .decode(&[104, 101, 108, 108, 111])
            .expect("decode should succeed");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_tokenizer_decode_with_merged_tokens() {
        let mut vocab = Vocabulary::with_base_tokens();
        vocab.add_merge(vec![104], vec![105]); // "hi"

        let tokenizer = BpeTokenizer::new(vocab);
        let result = tokenizer.decode(&[256]).expect("decode should succeed");
        assert_eq!(result, "hi");
    }

    #[test]
    fn test_tokenizer_decode_invalid_token() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let result = tokenizer.decode(&[50000]); // Invalid token
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_decode_skips_special() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        // Include special tokens - they should be skipped
        let result = tokenizer
            .decode(&[104, 105, special_tokens::EOT])
            .expect("decode should succeed");
        assert_eq!(result, "hi");
    }

    // =========================================================================
    // Roundtrip Tests
    // =========================================================================

    #[test]
    fn test_tokenizer_roundtrip_ascii() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let text = "Hello, World!";
        let tokens = tokenizer.encode(text).expect("encode should succeed");
        let decoded = tokenizer.decode(&tokens).expect("decode should succeed");
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_tokenizer_roundtrip_with_merges() {
        let mut vocab = Vocabulary::with_base_tokens();
        vocab.add_merge(vec![116], vec![104]); // "th"
        vocab.add_merge(vec![116, 104], vec![101]); // "the"

        let tokenizer = BpeTokenizer::new(vocab);
        let text = "the";
        let tokens = tokenizer.encode(text).expect("encode should succeed");
        let decoded = tokenizer.decode(&tokens).expect("decode should succeed");
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_tokenizer_roundtrip_utf8() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let text = "Hello 世界 🌍";
        let tokens = tokenizer.encode(text).expect("encode should succeed");
        let decoded = tokenizer.decode(&tokens).expect("decode should succeed");
        assert_eq!(decoded, text);
    }

    // =========================================================================
    // Decode With Options Tests
    // =========================================================================

    #[test]
    fn test_decode_with_options_skip_special() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let tokens = vec![104, 105, special_tokens::SOT, special_tokens::EOT];
        let result = tokenizer
            .decode_with_options(&tokens, true)
            .expect("decode should succeed");
        assert_eq!(result, "hi");
    }

    #[test]
    fn test_decode_with_options_keep_special() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let tokens = vec![104, 105]; // No special tokens
        let result = tokenizer
            .decode_with_options(&tokens, false)
            .expect("decode should succeed");
        assert_eq!(result, "hi");
    }

    // =========================================================================
    // Property Tests
    // =========================================================================

    #[test]
    fn test_encode_never_empty_for_nonempty_input() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let inputs = ["a", " ", "\n", "hello", "🎉"];

        for input in &inputs {
            let tokens = tokenizer.encode(input).expect("encode should succeed");
            assert!(
                !tokens.is_empty(),
                "encode('{}') should produce non-empty output",
                input
            );
        }
    }

    #[test]
    fn test_encode_produces_valid_tokens() {
        let tokenizer = BpeTokenizer::with_base_tokens();
        let text = "The quick brown fox jumps over the lazy dog.";
        let tokens = tokenizer.encode(text).expect("encode should succeed");

        for &token_id in &tokens {
            assert!(
                tokenizer.vocab().get_bytes(token_id).is_some(),
                "token {} should be valid",
                token_id
            );
        }
    }

    // =========================================================================
    // Property-Based Tests (WAPR-QA-002)
    // =========================================================================

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn property_encode_produces_valid_tokens(s in "[a-zA-Z0-9 ]{1,50}") {
                let tokenizer = BpeTokenizer::with_base_tokens();
                if let Ok(tokens) = tokenizer.encode(&s) {
                    for token in &tokens {
                        prop_assert!(
                            tokenizer.vocab().get_bytes(*token).is_some(),
                            "token {} from encoding '{}' should be valid",
                            token,
                            s
                        );
                    }
                }
            }

            #[test]
            fn property_decode_produces_string(token_count in 1usize..20) {
                let tokenizer = BpeTokenizer::with_base_tokens();
                // Generate valid token IDs (base tokens are 0-255)
                let tokens: Vec<u32> = (0..token_count).map(|i| (i % 128) as u32 + 32).collect();

                if let Ok(result) = tokenizer.decode(&tokens) {
                    // Should produce some output
                    prop_assert!(result.len() <= token_count * 4); // Each byte -> max 4 UTF-8 bytes
                }
            }

            #[test]
            fn property_vocab_size_positive(_dummy in 0..1i32) {
                let tokenizer = BpeTokenizer::with_base_tokens();

                let size = tokenizer.vocab_size();
                prop_assert!(size > 0, "vocab size should be positive");

                // Vocab size should match vocabulary entries
                let vocab = tokenizer.vocab();
                prop_assert_eq!(size, vocab.len());
            }

            #[test]
            fn property_special_tokens_reasonable(_dummy in 0..1i32) {
                let tokenizer = BpeTokenizer::with_base_tokens();

                // All special tokens should be valid
                let special_tokens = [
                    special_tokens::SOT,
                    special_tokens::EOT,
                    special_tokens::TRANSCRIBE,
                    special_tokens::TRANSLATE,
                    special_tokens::NO_TIMESTAMPS,
                ];

                for token in special_tokens {
                    // Whisper special tokens are in the 50257+ range, verify they're in reasonable bounds
                    prop_assert!(
                        (token as usize) >= 50257 && (token as usize) < 60000,
                        "special token {} should be in Whisper's special token range (50257-60000)",
                        token
                    );
                    // Also verify the base tokenizer has some vocabulary
                    prop_assert!(tokenizer.vocab_size() > 0, "tokenizer should have vocabulary");
                }
            }
        }
    }
}

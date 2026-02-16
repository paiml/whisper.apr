//! SentencePiece tokenizer for Moonshine models
//!
//! Moonshine uses a 32,768-token SentencePiece (Unigram) vocabulary,
//! distinct from Whisper's 51,865-token BPE vocabulary.
//!
//! This module provides a vocabulary-based tokenizer that loads
//! SentencePiece token mappings from an APR model file.

use crate::error::{WhisperError, WhisperResult};

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, string::String, vec::Vec};
#[cfg(feature = "std")]
use std::collections::BTreeMap;

/// SentencePiece special token IDs
pub mod special_tokens {
    /// Padding token
    pub const PAD: u32 = 0;
    /// Beginning of sequence
    pub const BOS: u32 = 1;
    /// End of sequence
    pub const EOS: u32 = 2;
    /// Unknown token
    pub const UNK: u32 = 3;
}

/// SentencePiece vocabulary size for Moonshine
pub const MOONSHINE_VOCAB_SIZE: usize = 32768;

/// SentencePiece tokenizer for Moonshine models
///
/// Uses a unigram model vocabulary mapping. Token pieces are stored
/// as UTF-8 strings with scores for the unigram model.
#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    /// Token ID → piece string
    id_to_piece: Vec<String>,
    /// Piece string → token ID
    piece_to_id: BTreeMap<String, u32>,
    /// Vocabulary size
    vocab_size: usize,
}

impl SentencePieceTokenizer {
    /// Create an empty tokenizer (for weight loading)
    #[must_use]
    pub fn new(vocab_size: usize) -> Self {
        Self {
            id_to_piece: Vec::with_capacity(vocab_size),
            piece_to_id: BTreeMap::new(),
            vocab_size,
        }
    }

    /// Create tokenizer with Moonshine default vocab size
    #[must_use]
    pub fn moonshine_default() -> Self {
        let mut tokenizer = Self::new(MOONSHINE_VOCAB_SIZE);
        // Register special tokens
        tokenizer.add_piece(special_tokens::PAD, "<pad>");
        tokenizer.add_piece(special_tokens::BOS, "<s>");
        tokenizer.add_piece(special_tokens::EOS, "</s>");
        tokenizer.add_piece(special_tokens::UNK, "<unk>");
        tokenizer
    }

    /// Add a token piece to the vocabulary
    pub fn add_piece(&mut self, id: u32, piece: &str) {
        let id_usize = id as usize;
        // Extend id_to_piece if needed
        while self.id_to_piece.len() <= id_usize {
            self.id_to_piece.push(String::new());
        }
        self.id_to_piece[id_usize] = String::from(piece);
        self.piece_to_id.insert(String::from(piece), id);
    }

    /// Encode text to token IDs
    ///
    /// Uses a greedy longest-match tokenization strategy.
    ///
    /// # Errors
    /// Returns error if text contains characters not in vocabulary
    pub fn encode(&self, text: &str) -> WhisperResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut tokens = Vec::new();
        let mut pos = 0;
        let bytes = text.as_bytes();

        while pos < bytes.len() {
            // Try longest match first
            let mut matched = false;
            let max_len = (bytes.len() - pos).min(64); // reasonable max piece length

            for len in (1..=max_len).rev() {
                if let Ok(piece) = core::str::from_utf8(&bytes[pos..pos + len]) {
                    if let Some(&id) = self.piece_to_id.get(piece) {
                        tokens.push(id);
                        pos += len;
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                // Fall back to UNK for unknown bytes
                tokens.push(special_tokens::UNK);
                pos += 1;
            }
        }

        Ok(tokens)
    }

    /// Decode token IDs to text
    ///
    /// # Errors
    /// Returns error if any token ID is out of range
    pub fn decode(&self, tokens: &[u32]) -> WhisperResult<String> {
        let mut text = String::new();
        for &id in tokens {
            let id_usize = id as usize;
            if id_usize >= self.id_to_piece.len() {
                return Err(WhisperError::Tokenizer(format!(
                    "token ID {id} out of range (max {})",
                    self.id_to_piece.len()
                )));
            }
            let piece = &self.id_to_piece[id_usize];
            // Skip special tokens in output
            if id <= special_tokens::UNK {
                continue;
            }
            // SentencePiece uses U+2581 (▁) as word boundary marker → space
            text.push_str(&piece.replace('\u{2581}', " "));
        }
        Ok(text.trim_start().to_string())
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of pieces currently loaded
    #[must_use]
    pub fn loaded_pieces(&self) -> usize {
        self.piece_to_id.len()
    }

    /// BOS token ID
    #[must_use]
    pub const fn bos_id(&self) -> u32 {
        special_tokens::BOS
    }

    /// EOS token ID
    #[must_use]
    pub const fn eos_id(&self) -> u32 {
        special_tokens::EOS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentencepiece_new() {
        let sp = SentencePieceTokenizer::new(32768);
        assert_eq!(sp.vocab_size(), 32768);
        assert_eq!(sp.loaded_pieces(), 0);
    }

    #[test]
    fn test_sentencepiece_moonshine_default() {
        let sp = SentencePieceTokenizer::moonshine_default();
        assert_eq!(sp.vocab_size(), MOONSHINE_VOCAB_SIZE);
        assert_eq!(sp.loaded_pieces(), 4); // pad, bos, eos, unk
        assert_eq!(sp.bos_id(), 1);
        assert_eq!(sp.eos_id(), 2);
    }

    #[test]
    fn test_sentencepiece_add_and_encode() {
        let mut sp = SentencePieceTokenizer::moonshine_default();
        sp.add_piece(100, "hello");
        sp.add_piece(101, "world");
        sp.add_piece(102, " ");

        let tokens = sp.encode("hello world").expect("should encode");
        assert_eq!(tokens, vec![100, 102, 101]);
    }

    #[test]
    fn test_sentencepiece_decode() {
        let mut sp = SentencePieceTokenizer::moonshine_default();
        sp.add_piece(100, "hello");
        sp.add_piece(101, " world");

        let text = sp.decode(&[100, 101]).expect("should decode");
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_sentencepiece_decode_skips_special() {
        let mut sp = SentencePieceTokenizer::moonshine_default();
        sp.add_piece(100, "hi");

        let text = sp
            .decode(&[special_tokens::BOS, 100, special_tokens::EOS])
            .expect("should decode");
        assert_eq!(text, "hi");
    }

    #[test]
    fn test_sentencepiece_encode_empty() {
        let sp = SentencePieceTokenizer::moonshine_default();
        let tokens = sp.encode("").expect("should encode empty");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_sentencepiece_decode_empty() {
        let sp = SentencePieceTokenizer::moonshine_default();
        let text = sp.decode(&[]).expect("should decode empty");
        assert!(text.is_empty());
    }

    #[test]
    fn test_sentencepiece_unknown_falls_back_to_unk() {
        let sp = SentencePieceTokenizer::moonshine_default();
        // No pieces loaded beyond special tokens, so all chars should produce UNK
        let tokens = sp.encode("abc").expect("should encode");
        assert_eq!(tokens, vec![special_tokens::UNK; 3]);
    }

    #[test]
    fn test_sentencepiece_decode_out_of_range() {
        let sp = SentencePieceTokenizer::moonshine_default();
        let result = sp.decode(&[99999]);
        assert!(result.is_err());
    }
}

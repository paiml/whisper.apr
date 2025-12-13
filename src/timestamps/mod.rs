//! Timestamp extraction and word-level alignment (WAPR-160 to WAPR-163)
//!
//! Provides accurate word-level timestamps using cross-attention alignment.
//!
//! # Overview
//!
//! This module implements:
//! 1. Segment-level timestamps from decoder output
//! 2. Word-level timestamps via cross-attention alignment
//! 3. Sub-word token interpolation for accurate boundaries
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::timestamps::{WordTimestampExtractor, extract_segments};
//!
//! // Basic segment extraction (existing)
//! let segments = extract_segments(&tokens, |t| tokenizer.decode(t));
//!
//! // Word-level timestamps (new)
//! let extractor = WordTimestampExtractor::new(config);
//! let word_timestamps = extractor.extract(&cross_attention_weights, &tokens)?;
//! ```

pub mod alignment;
pub mod boundaries;
pub mod interpolation;
mod segment;

pub use alignment::{
    AlignmentConfig, CrossAttentionAlignment, TokenAlignment, WordAlignment,
    WordTimestampExtractor,
};
pub use boundaries::{BoundaryConfig, BoundaryDetector, WordBoundary};
pub use interpolation::{InterpolationConfig, TimestampInterpolator, TokenTimestamp};
pub use segment::{
    count_text_tokens, estimate_duration_from_tokens, extract_segments, get_timestamps,
    has_timestamps, is_control_token, is_timestamp, merge_segments, parse_timestamp_pairs,
    seconds_to_timestamp_token, split_long_segments, timestamp_to_seconds, MAX_TIMESTAMP_SECONDS,
    MAX_TIMESTAMP_TOKENS, TIMESTAMP_RESOLUTION,
};


/// Word with timestamp and confidence score
#[derive(Debug, Clone)]
pub struct WordWithTimestamp {
    /// The word text
    pub word: String,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Token indices that make up this word
    pub token_indices: Vec<usize>,
}

impl WordWithTimestamp {
    /// Create a new word with timestamp
    #[must_use]
    pub fn new(word: String, start: f32, end: f32, confidence: f32) -> Self {
        Self {
            word,
            start,
            end,
            confidence,
            token_indices: Vec::new(),
        }
    }

    /// Get word duration in seconds
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Check if timestamp is high confidence
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }

    /// Add a token index
    pub fn add_token_index(&mut self, index: usize) {
        self.token_indices.push(index);
    }

    /// Set token indices
    #[must_use]
    pub fn with_token_indices(mut self, indices: Vec<usize>) -> Self {
        self.token_indices = indices;
        self
    }
}

/// Word-level timestamp result for a segment
#[derive(Debug, Clone)]
pub struct WordTimestampResult {
    /// Words with timestamps
    pub words: Vec<WordWithTimestamp>,
    /// Segment start time
    pub segment_start: f32,
    /// Segment end time
    pub segment_end: f32,
    /// Overall alignment confidence
    pub alignment_confidence: f32,
}

impl WordTimestampResult {
    /// Create new timestamp result
    #[must_use]
    pub fn new(words: Vec<WordWithTimestamp>, segment_start: f32, segment_end: f32) -> Self {
        let alignment_confidence = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32
        };

        Self {
            words,
            segment_start,
            segment_end,
            alignment_confidence,
        }
    }

    /// Get total word count
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Get words at a specific time
    #[must_use]
    pub fn words_at_time(&self, time: f32) -> Vec<&WordWithTimestamp> {
        self.words
            .iter()
            .filter(|w| time >= w.start && time < w.end)
            .collect()
    }

    /// Get word by index
    #[must_use]
    pub fn get_word(&self, index: usize) -> Option<&WordWithTimestamp> {
        self.words.get(index)
    }

    /// Check if result has high-quality timestamps
    #[must_use]
    pub fn is_high_quality(&self) -> bool {
        self.alignment_confidence >= 0.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_with_timestamp_new() {
        let word = WordWithTimestamp::new("hello".to_string(), 0.5, 1.0, 0.95);
        assert_eq!(word.word, "hello");
        assert!((word.start - 0.5).abs() < f32::EPSILON);
        assert!((word.end - 1.0).abs() < f32::EPSILON);
        assert!((word.confidence - 0.95).abs() < f32::EPSILON);
        assert!(word.token_indices.is_empty());
    }

    #[test]
    fn test_word_with_timestamp_duration() {
        let word = WordWithTimestamp::new("test".to_string(), 1.0, 2.5, 0.9);
        assert!((word.duration() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_with_timestamp_high_confidence() {
        let high = WordWithTimestamp::new("high".to_string(), 0.0, 1.0, 0.85);
        let low = WordWithTimestamp::new("low".to_string(), 0.0, 1.0, 0.5);

        assert!(high.is_high_confidence());
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_word_with_timestamp_token_indices() {
        let mut word = WordWithTimestamp::new("test".to_string(), 0.0, 1.0, 0.9);
        word.add_token_index(5);
        word.add_token_index(6);

        assert_eq!(word.token_indices, vec![5, 6]);
    }

    #[test]
    fn test_word_with_timestamp_with_token_indices() {
        let word = WordWithTimestamp::new("test".to_string(), 0.0, 1.0, 0.9)
            .with_token_indices(vec![1, 2, 3]);

        assert_eq!(word.token_indices, vec![1, 2, 3]);
    }

    #[test]
    fn test_word_timestamp_result_new() {
        let words = vec![
            WordWithTimestamp::new("hello".to_string(), 0.0, 0.5, 0.9),
            WordWithTimestamp::new("world".to_string(), 0.5, 1.0, 0.8),
        ];

        let result = WordTimestampResult::new(words, 0.0, 1.0);

        assert_eq!(result.word_count(), 2);
        assert!((result.alignment_confidence - 0.85).abs() < f32::EPSILON);
        assert!(result.is_high_quality());
    }

    #[test]
    fn test_word_timestamp_result_empty() {
        let result = WordTimestampResult::new(vec![], 0.0, 1.0);

        assert_eq!(result.word_count(), 0);
        assert!((result.alignment_confidence - 0.0).abs() < f32::EPSILON);
        assert!(!result.is_high_quality());
    }

    #[test]
    fn test_word_timestamp_result_words_at_time() {
        let words = vec![
            WordWithTimestamp::new("hello".to_string(), 0.0, 0.5, 0.9),
            WordWithTimestamp::new("world".to_string(), 0.5, 1.0, 0.8),
        ];

        let result = WordTimestampResult::new(words, 0.0, 1.0);

        let at_025 = result.words_at_time(0.25);
        assert_eq!(at_025.len(), 1);
        assert_eq!(at_025[0].word, "hello");

        let at_075 = result.words_at_time(0.75);
        assert_eq!(at_075.len(), 1);
        assert_eq!(at_075[0].word, "world");
    }

    #[test]
    fn test_word_timestamp_result_get_word() {
        let words = vec![
            WordWithTimestamp::new("hello".to_string(), 0.0, 0.5, 0.9),
            WordWithTimestamp::new("world".to_string(), 0.5, 1.0, 0.8),
        ];

        let result = WordTimestampResult::new(words, 0.0, 1.0);

        assert_eq!(result.get_word(0).map(|w| w.word.as_str()), Some("hello"));
        assert_eq!(result.get_word(1).map(|w| w.word.as_str()), Some("world"));
        assert!(result.get_word(2).is_none());
    }
}

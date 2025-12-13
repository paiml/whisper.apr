//! Cross-attention alignment extraction (WAPR-160)
//!
//! Extracts word-level timestamps from cross-attention weights.
//!
//! # Overview
//!
//! Whisper's cross-attention weights encode alignment between audio frames
//! and text tokens. By analyzing these weights, we can determine when each
//! word was spoken in the audio.
//!
//! # Algorithm
//!
//! 1. Extract cross-attention weights from decoder layers
//! 2. Average weights across heads and layers
//! 3. Find peak attention for each token
//! 4. Convert frame positions to timestamps
//! 5. Group tokens into words

use crate::error::{WhisperError, WhisperResult};

/// Audio frame rate (frames per second)
pub const AUDIO_FRAME_RATE: f32 = 50.0;

/// Alignment configuration
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// Layers to use for alignment (e.g., last 6 layers)
    pub layers: Vec<usize>,
    /// Heads to use for alignment (None = all heads)
    pub heads: Option<Vec<usize>>,
    /// Minimum attention threshold for valid alignment
    pub min_attention: f32,
    /// Temperature for attention softmax
    pub temperature: f32,
    /// Use median instead of mean for averaging
    pub use_median: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            layers: vec![0, 1, 2, 3, 4, 5], // First 6 layers
            heads: None,                    // All heads
            min_attention: 0.1,
            temperature: 1.0,
            use_median: false,
        }
    }
}

impl AlignmentConfig {
    /// Create config optimized for accuracy
    #[must_use]
    pub fn for_accuracy() -> Self {
        Self {
            layers: vec![2, 3, 4, 5], // Middle-late layers
            heads: None,
            min_attention: 0.05,
            temperature: 0.5,
            use_median: true,
        }
    }

    /// Create config optimized for speed
    #[must_use]
    pub fn for_speed() -> Self {
        Self {
            layers: vec![3, 4], // Only 2 layers
            heads: Some(vec![0, 1, 2, 3]), // Subset of heads
            min_attention: 0.15,
            temperature: 1.0,
            use_median: false,
        }
    }

    /// Set layers to use
    #[must_use]
    pub fn with_layers(mut self, layers: Vec<usize>) -> Self {
        self.layers = layers;
        self
    }

    /// Set minimum attention threshold
    #[must_use]
    pub fn with_min_attention(mut self, threshold: f32) -> Self {
        self.min_attention = threshold;
        self
    }
}

/// Token alignment information
#[derive(Debug, Clone)]
pub struct TokenAlignment {
    /// Token index in the sequence
    pub token_index: usize,
    /// Token ID
    pub token_id: u32,
    /// Peak audio frame position
    pub frame_position: usize,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Alignment confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Attention weights for this token across frames
    pub attention_weights: Vec<f32>,
}

impl TokenAlignment {
    /// Create new token alignment
    #[must_use]
    pub fn new(
        token_index: usize,
        token_id: u32,
        frame_position: usize,
        confidence: f32,
    ) -> Self {
        let start_time = frame_position as f32 / AUDIO_FRAME_RATE;
        Self {
            token_index,
            token_id,
            frame_position,
            start_time,
            end_time: start_time,
            confidence,
            attention_weights: Vec::new(),
        }
    }

    /// Set end time
    pub fn set_end_time(&mut self, end_frame: usize) {
        self.end_time = end_frame as f32 / AUDIO_FRAME_RATE;
    }

    /// Get duration in seconds
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }

    /// Check if alignment is high confidence
    #[must_use]
    pub fn is_confident(&self) -> bool {
        self.confidence >= 0.5
    }

    /// Set attention weights
    #[must_use]
    pub fn with_attention_weights(mut self, weights: Vec<f32>) -> Self {
        self.attention_weights = weights;
        self
    }
}

/// Word alignment with timing
#[derive(Debug, Clone)]
pub struct WordAlignment {
    /// Word text
    pub word: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Confidence score
    pub confidence: f32,
    /// Token alignments that make up this word
    pub tokens: Vec<TokenAlignment>,
}

impl WordAlignment {
    /// Create new word alignment
    #[must_use]
    pub fn new(word: String, tokens: Vec<TokenAlignment>) -> Self {
        let start_time = tokens.first().map_or(0.0, |t| t.start_time);
        let end_time = tokens.last().map_or(0.0, |t| t.end_time);
        let confidence = if tokens.is_empty() {
            0.0
        } else {
            tokens.iter().map(|t| t.confidence).sum::<f32>() / tokens.len() as f32
        };

        Self {
            word,
            start_time,
            end_time,
            confidence,
            tokens,
        }
    }

    /// Get word duration
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }

    /// Get token count
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

/// Cross-attention alignment extractor
#[derive(Debug, Clone)]
pub struct CrossAttentionAlignment {
    /// Configuration
    config: AlignmentConfig,
}

impl CrossAttentionAlignment {
    /// Create new alignment extractor
    #[must_use]
    pub fn new(config: AlignmentConfig) -> Self {
        Self { config }
    }

    /// Extract token alignments from cross-attention weights
    ///
    /// # Arguments
    /// * `attention_weights` - Cross-attention weights [layers][heads][tokens][frames]
    /// * `token_ids` - Token IDs
    /// * `num_frames` - Number of audio frames
    ///
    /// # Returns
    /// Token alignments with frame positions and confidence
    pub fn extract_token_alignments(
        &self,
        attention_weights: &[Vec<Vec<Vec<f32>>>],
        token_ids: &[u32],
        num_frames: usize,
    ) -> WhisperResult<Vec<TokenAlignment>> {
        if attention_weights.is_empty() {
            return Err(WhisperError::Inference(
                "No attention weights provided".to_string(),
            ));
        }

        if token_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Average attention across selected layers and heads
        let averaged = self.average_attention(attention_weights, num_frames, token_ids.len())?;

        // Find peak frame for each token
        let mut alignments = Vec::with_capacity(token_ids.len());

        for (token_idx, &token_id) in token_ids.iter().enumerate() {
            if token_idx >= averaged.len() {
                break;
            }

            let token_attention = &averaged[token_idx];
            let (peak_frame, peak_value) = self.find_peak(token_attention);

            let confidence = self.compute_confidence(token_attention, peak_frame, peak_value);

            let mut alignment =
                TokenAlignment::new(token_idx, token_id, peak_frame, confidence)
                    .with_attention_weights(token_attention.clone());

            // Set end time based on next token or frame boundary
            if token_idx + 1 < averaged.len() {
                let next_attention = &averaged[token_idx + 1];
                let (next_peak, _) = self.find_peak(next_attention);
                alignment.set_end_time(next_peak);
            } else {
                alignment.set_end_time(num_frames);
            }

            alignments.push(alignment);
        }

        Ok(alignments)
    }

    /// Average attention weights across layers and heads
    #[allow(clippy::unnecessary_wraps)]
    fn average_attention(
        &self,
        attention_weights: &[Vec<Vec<Vec<f32>>>],
        num_frames: usize,
        num_tokens: usize,
    ) -> WhisperResult<Vec<Vec<f32>>> {
        let mut averaged = vec![vec![0.0f32; num_frames]; num_tokens];
        let mut count = 0;

        for (layer_idx, layer) in attention_weights.iter().enumerate() {
            if !self.config.layers.contains(&layer_idx) {
                continue;
            }

            for (head_idx, head) in layer.iter().enumerate() {
                if let Some(ref heads) = self.config.heads {
                    if !heads.contains(&head_idx) {
                        continue;
                    }
                }

                for (token_idx, token_attention) in head.iter().enumerate() {
                    if token_idx >= num_tokens {
                        break;
                    }

                    for (frame_idx, &weight) in token_attention.iter().enumerate() {
                        if frame_idx >= num_frames {
                            break;
                        }
                        averaged[token_idx][frame_idx] += weight;
                    }
                }

                count += 1;
            }
        }

        if count > 0 {
            for token_attention in &mut averaged {
                for weight in token_attention.iter_mut() {
                    *weight /= count as f32;
                }
            }
        }

        Ok(averaged)
    }

    /// Find peak attention frame
    fn find_peak(&self, attention: &[f32]) -> (usize, f32) {
        let _ = self; // Method for consistency
        attention
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((0, 0.0), |(idx, &val)| (idx, val))
    }

    /// Compute alignment confidence
    fn compute_confidence(&self, attention: &[f32], peak_frame: usize, peak_value: f32) -> f32 {
        if attention.is_empty() || peak_value < self.config.min_attention {
            return 0.0;
        }

        // Confidence based on:
        // 1. Peak value (higher is better)
        // 2. Peakiness (concentration around peak)
        let sum: f32 = attention.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }

        let concentration = peak_value / sum;

        // Check attention spread around peak
        let window = 5;
        let start = peak_frame.saturating_sub(window);
        let end = (peak_frame + window + 1).min(attention.len());
        let local_sum: f32 = attention[start..end].iter().sum();
        let locality = local_sum / sum;

        // Combined confidence
        concentration.mul_add(0.5, locality * 0.5).min(1.0)
    }
}

impl Default for CrossAttentionAlignment {
    fn default() -> Self {
        Self::new(AlignmentConfig::default())
    }
}

/// Word timestamp extractor
#[derive(Debug, Clone)]
pub struct WordTimestampExtractor {
    /// Cross-attention alignment extractor
    alignment: CrossAttentionAlignment,
}

impl WordTimestampExtractor {
    /// Create new extractor
    #[must_use]
    pub fn new(config: AlignmentConfig) -> Self {
        Self {
            alignment: CrossAttentionAlignment::new(config),
        }
    }

    /// Extract word alignments from cross-attention weights
    ///
    /// # Arguments
    /// * `attention_weights` - Cross-attention weights
    /// * `token_ids` - Token IDs
    /// * `token_texts` - Decoded text for each token
    /// * `num_frames` - Number of audio frames
    pub fn extract_word_alignments(
        &self,
        attention_weights: &[Vec<Vec<Vec<f32>>>],
        token_ids: &[u32],
        token_texts: &[String],
        num_frames: usize,
    ) -> WhisperResult<Vec<WordAlignment>> {
        // Get token alignments
        let token_alignments =
            self.alignment
                .extract_token_alignments(attention_weights, token_ids, num_frames)?;

        // Group tokens into words
        let words = self.group_tokens_into_words(&token_alignments, token_texts);

        Ok(words)
    }

    /// Group token alignments into word alignments
    fn group_tokens_into_words(
        &self,
        alignments: &[TokenAlignment],
        token_texts: &[String],
    ) -> Vec<WordAlignment> {
        let _ = self; // Method for consistency
        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut current_tokens: Vec<TokenAlignment> = Vec::new();

        for (alignment, text) in alignments.iter().zip(token_texts.iter()) {
            // Check if this starts a new word (starts with space or is first token)
            let starts_new_word = text.starts_with(' ') || text.starts_with('▁');

            if starts_new_word && !current_word.is_empty() {
                // Save current word
                words.push(WordAlignment::new(
                    current_word.trim().to_string(),
                    current_tokens.clone(),
                ));
                current_word.clear();
                current_tokens.clear();
            }

            current_word.push_str(text.trim_start_matches([' ', '▁']));
            current_tokens.push(alignment.clone());
        }

        // Add final word
        if !current_word.is_empty() {
            words.push(WordAlignment::new(
                current_word.trim().to_string(),
                current_tokens,
            ));
        }

        words
    }
}

impl Default for WordTimestampExtractor {
    fn default() -> Self {
        Self::new(AlignmentConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // AlignmentConfig Tests
    // =========================================================================

    #[test]
    fn test_alignment_config_default() {
        let config = AlignmentConfig::default();
        assert_eq!(config.layers.len(), 6);
        assert!(config.heads.is_none());
        assert!((config.min_attention - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_alignment_config_for_accuracy() {
        let config = AlignmentConfig::for_accuracy();
        assert_eq!(config.layers, vec![2, 3, 4, 5]);
        assert!(config.use_median);
    }

    #[test]
    fn test_alignment_config_for_speed() {
        let config = AlignmentConfig::for_speed();
        assert_eq!(config.layers, vec![3, 4]);
        assert!(config.heads.is_some());
    }

    #[test]
    fn test_alignment_config_with_layers() {
        let config = AlignmentConfig::default().with_layers(vec![1, 2, 3]);
        assert_eq!(config.layers, vec![1, 2, 3]);
    }

    #[test]
    fn test_alignment_config_with_min_attention() {
        let config = AlignmentConfig::default().with_min_attention(0.2);
        assert!((config.min_attention - 0.2).abs() < f32::EPSILON);
    }

    // =========================================================================
    // TokenAlignment Tests
    // =========================================================================

    #[test]
    fn test_token_alignment_new() {
        let alignment = TokenAlignment::new(0, 100, 50, 0.9);
        assert_eq!(alignment.token_index, 0);
        assert_eq!(alignment.token_id, 100);
        assert_eq!(alignment.frame_position, 50);
        assert!((alignment.confidence - 0.9).abs() < f32::EPSILON);
        assert!((alignment.start_time - 1.0).abs() < f32::EPSILON); // 50 / 50 fps
    }

    #[test]
    fn test_token_alignment_set_end_time() {
        let mut alignment = TokenAlignment::new(0, 100, 50, 0.9);
        alignment.set_end_time(100);
        assert!((alignment.end_time - 2.0).abs() < f32::EPSILON); // 100 / 50 fps
    }

    #[test]
    fn test_token_alignment_duration() {
        let mut alignment = TokenAlignment::new(0, 100, 50, 0.9);
        alignment.set_end_time(100);
        assert!((alignment.duration() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_token_alignment_is_confident() {
        let high = TokenAlignment::new(0, 100, 50, 0.6);
        let low = TokenAlignment::new(0, 100, 50, 0.4);
        assert!(high.is_confident());
        assert!(!low.is_confident());
    }

    #[test]
    fn test_token_alignment_with_attention_weights() {
        let alignment =
            TokenAlignment::new(0, 100, 50, 0.9).with_attention_weights(vec![0.1, 0.2, 0.7]);
        assert_eq!(alignment.attention_weights, vec![0.1, 0.2, 0.7]);
    }

    // =========================================================================
    // WordAlignment Tests
    // =========================================================================

    #[test]
    fn test_word_alignment_new() {
        let tokens = vec![
            TokenAlignment::new(0, 100, 50, 0.9),
            TokenAlignment::new(1, 101, 60, 0.8),
        ];
        let word = WordAlignment::new("hello".to_string(), tokens);

        assert_eq!(word.word, "hello");
        assert!((word.start_time - 1.0).abs() < f32::EPSILON);
        assert!((word.confidence - 0.85).abs() < f32::EPSILON);
        assert_eq!(word.token_count(), 2);
    }

    #[test]
    fn test_word_alignment_empty_tokens() {
        let word = WordAlignment::new("empty".to_string(), vec![]);
        assert!((word.start_time - 0.0).abs() < f32::EPSILON);
        assert!((word.end_time - 0.0).abs() < f32::EPSILON);
        assert!((word.confidence - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_alignment_duration() {
        let mut token1 = TokenAlignment::new(0, 100, 50, 0.9);
        token1.set_end_time(60);
        let mut token2 = TokenAlignment::new(1, 101, 60, 0.8);
        token2.set_end_time(80);

        let word = WordAlignment::new("test".to_string(), vec![token1, token2]);
        assert!((word.duration() - 0.6).abs() < 0.01); // (80-50) / 50 fps
    }

    // =========================================================================
    // CrossAttentionAlignment Tests
    // =========================================================================

    #[test]
    fn test_cross_attention_alignment_new() {
        let alignment = CrossAttentionAlignment::new(AlignmentConfig::default());
        assert!((alignment.config.min_attention - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cross_attention_alignment_default() {
        let alignment = CrossAttentionAlignment::default();
        assert_eq!(alignment.config.layers.len(), 6);
    }

    #[test]
    fn test_extract_token_alignments_empty() {
        let alignment = CrossAttentionAlignment::default();
        let result = alignment.extract_token_alignments(&[], &[], 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_token_alignments_no_tokens() {
        let alignment = CrossAttentionAlignment::default();
        // Create minimal attention weights
        let weights = vec![vec![vec![vec![0.1f32; 10]; 1]; 4]; 6];
        let result = alignment.extract_token_alignments(&weights, &[], 10);
        assert!(result.is_ok());
        assert!(result.expect("should succeed").is_empty());
    }

    #[test]
    fn test_extract_token_alignments_single_token() {
        let config = AlignmentConfig::default().with_layers(vec![0]);
        let alignment = CrossAttentionAlignment::new(config);

        // Create attention weights with peak at frame 5
        let mut token_attention = vec![0.1f32; 10];
        token_attention[5] = 0.9;

        let weights = vec![vec![vec![token_attention]; 1]];
        let token_ids = vec![100u32];

        let result = alignment
            .extract_token_alignments(&weights, &token_ids, 10)
            .expect("should succeed");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].frame_position, 5);
        assert!(result[0].confidence > 0.0);
    }

    #[test]
    fn test_find_peak() {
        let alignment = CrossAttentionAlignment::default();
        let attention = vec![0.1, 0.2, 0.8, 0.3, 0.1];
        let (peak_idx, peak_val) = alignment.find_peak(&attention);

        assert_eq!(peak_idx, 2);
        assert!((peak_val - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_find_peak_empty() {
        let alignment = CrossAttentionAlignment::default();
        let (peak_idx, peak_val) = alignment.find_peak(&[]);
        assert_eq!(peak_idx, 0);
        assert!((peak_val - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_confidence() {
        let alignment = CrossAttentionAlignment::default();

        // High concentration attention
        let attention = vec![0.0, 0.0, 0.9, 0.1, 0.0];
        let confidence = alignment.compute_confidence(&attention, 2, 0.9);
        assert!(confidence > 0.5);

        // Flat attention
        let flat_attention = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let flat_confidence = alignment.compute_confidence(&flat_attention, 2, 0.2);
        assert!(flat_confidence < confidence);
    }

    // =========================================================================
    // WordTimestampExtractor Tests
    // =========================================================================

    #[test]
    fn test_word_timestamp_extractor_new() {
        let extractor = WordTimestampExtractor::new(AlignmentConfig::default());
        assert!((extractor.alignment.config.min_attention - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_word_timestamp_extractor_default() {
        let extractor = WordTimestampExtractor::default();
        assert_eq!(extractor.alignment.config.layers.len(), 6);
    }

    #[test]
    fn test_group_tokens_into_words_simple() {
        let extractor = WordTimestampExtractor::default();

        let alignments = vec![
            TokenAlignment::new(0, 100, 0, 0.9),
            TokenAlignment::new(1, 101, 10, 0.8),
            TokenAlignment::new(2, 102, 20, 0.85),
        ];

        let texts = vec![
            "hello".to_string(),
            " world".to_string(),
            "!".to_string(),
        ];

        let words = extractor.group_tokens_into_words(&alignments, &texts);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[1].word, "world!");
    }

    #[test]
    fn test_group_tokens_into_words_sentencepiece() {
        let extractor = WordTimestampExtractor::default();

        let alignments = vec![
            TokenAlignment::new(0, 100, 0, 0.9),
            TokenAlignment::new(1, 101, 10, 0.8),
        ];

        let texts = vec!["▁hello".to_string(), "▁world".to_string()];

        let words = extractor.group_tokens_into_words(&alignments, &texts);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[1].word, "world");
    }

    #[test]
    fn test_group_tokens_into_words_empty() {
        let extractor = WordTimestampExtractor::default();
        let words = extractor.group_tokens_into_words(&[], &[]);
        assert!(words.is_empty());
    }
}

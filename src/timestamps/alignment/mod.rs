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

#[cfg(test)]
mod tests;

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
            layers: vec![3, 4],            // Only 2 layers
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
    pub fn new(token_index: usize, token_id: u32, frame_position: usize, confidence: f32) -> Self {
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
        let n = tokens.len();
        let confidence = tokens.iter().map(|t| t.confidence).sum::<f32>() / n.max(1) as f32;

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

        for (token_idx, (&token_id, token_attention)) in
            token_ids.iter().zip(averaged.iter()).enumerate()
        {
            let (peak_frame, peak_value) = self.find_peak(token_attention);

            let confidence = self.compute_confidence(token_attention, peak_frame, peak_value);

            let mut alignment = TokenAlignment::new(token_idx, token_id, peak_frame, confidence)
                .with_attention_weights(token_attention.clone());

            // Set end time based on next token or frame boundary
            let end_frame = averaged
                .get(token_idx + 1)
                .map_or(num_frames, |next| self.find_peak(next).0);
            alignment.set_end_time(end_frame);

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
        let mut count = 0usize;

        let selected_heads: Vec<&Vec<Vec<f32>>> = attention_weights
            .iter()
            .enumerate()
            .filter(|(li, _)| self.config.layers.contains(li))
            .flat_map(|(_, layer)| {
                layer
                    .iter()
                    .enumerate()
                    .filter(|(hi, _)| self.config.heads.as_ref().map_or(true, |h| h.contains(hi)))
            })
            .map(|(_, head)| head)
            .collect();

        for head in &selected_heads {
            for (token_idx, token_attention) in head.iter().enumerate().take(num_tokens) {
                for (frame_idx, &weight) in token_attention.iter().enumerate().take(num_frames) {
                    averaged[token_idx][frame_idx] += weight;
                }
            }
            count += 1;
        }

        let scale = 1.0 / (count.max(1) as f32);
        for token_attention in &mut averaged {
            for weight in token_attention.iter_mut() {
                *weight *= scale;
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
        let sum: f32 = attention.iter().sum();
        if attention.is_empty() || peak_value < self.config.min_attention || sum <= 0.0 {
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

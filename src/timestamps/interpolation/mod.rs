//! Timestamp interpolation for sub-word tokens (WAPR-162)
//!
//! Interpolates timestamps for BPE sub-word tokens within words.
//!
//! # Overview
//!
//! Whisper uses BPE tokenization, so words are often split into multiple tokens.
//! This module provides methods to:
//! 1. Interpolate timestamps within words
//! 2. Handle special tokens (punctuation, etc.)
//! 3. Smooth timestamp sequences

#[cfg(test)]
mod tests;

use crate::error::WhisperResult;

/// Interpolation configuration
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// Interpolation method
    pub method: InterpolationMethod,
    /// Smoothing window size (0 = no smoothing)
    pub smoothing_window: usize,
    /// Weight for character-length proportional timing
    pub char_weight: f32,
    /// Weight for uniform timing
    pub uniform_weight: f32,
}

/// Interpolation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Linear interpolation based on token position
    Linear,
    /// Character-length proportional
    CharacterProportional,
    /// Weighted combination
    Weighted,
    /// Attention-guided (requires attention weights)
    AttentionGuided,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            method: InterpolationMethod::Weighted,
            smoothing_window: 3,
            char_weight: 0.7,
            uniform_weight: 0.3,
        }
    }
}

impl InterpolationConfig {
    /// Create config for linear interpolation
    #[must_use]
    pub fn linear() -> Self {
        Self {
            method: InterpolationMethod::Linear,
            smoothing_window: 0,
            char_weight: 0.0,
            uniform_weight: 1.0,
        }
    }

    /// Create config for character-proportional interpolation
    #[must_use]
    pub fn character_proportional() -> Self {
        Self {
            method: InterpolationMethod::CharacterProportional,
            smoothing_window: 0,
            char_weight: 1.0,
            uniform_weight: 0.0,
        }
    }

    /// Set smoothing window
    #[must_use]
    pub fn with_smoothing(mut self, window: usize) -> Self {
        self.smoothing_window = window;
        self
    }

    /// Set interpolation method
    #[must_use]
    pub fn with_method(mut self, method: InterpolationMethod) -> Self {
        self.method = method;
        self
    }
}

/// Token timestamp information
#[derive(Debug, Clone)]
pub struct TokenTimestamp {
    /// Token index
    pub index: usize,
    /// Token text
    pub text: String,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Whether this timestamp was interpolated
    pub interpolated: bool,
    /// Confidence (lower for interpolated)
    pub confidence: f32,
}

impl TokenTimestamp {
    /// Create new token timestamp
    #[must_use]
    pub fn new(index: usize, text: String, start: f32, end: f32) -> Self {
        Self {
            index,
            text,
            start,
            end,
            interpolated: false,
            confidence: 1.0,
        }
    }

    /// Get duration
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Mark as interpolated
    pub fn mark_interpolated(&mut self, confidence: f32) {
        self.interpolated = true;
        self.confidence = confidence;
    }

    /// Create interpolated timestamp
    #[must_use]
    pub fn interpolated(index: usize, text: String, start: f32, end: f32, confidence: f32) -> Self {
        Self {
            index,
            text,
            start,
            end,
            interpolated: true,
            confidence,
        }
    }
}

/// Timestamp interpolator
#[derive(Debug, Clone)]
pub struct TimestampInterpolator {
    /// Configuration
    config: InterpolationConfig,
}

impl TimestampInterpolator {
    /// Create new interpolator
    #[must_use]
    pub fn new(config: InterpolationConfig) -> Self {
        Self { config }
    }

    /// Interpolate timestamps for tokens within a word
    ///
    /// # Arguments
    /// * `word_start` - Word start time
    /// * `word_end` - Word end time
    /// * `tokens` - Token texts within the word
    pub fn interpolate_word_tokens(
        &self,
        word_start: f32,
        word_end: f32,
        tokens: &[String],
        start_index: usize,
    ) -> WhisperResult<Vec<TokenTimestamp>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        if tokens.len() == 1 {
            return Ok(vec![TokenTimestamp::new(
                start_index,
                tokens[0].clone(),
                word_start,
                word_end,
            )]);
        }

        match self.config.method {
            InterpolationMethod::Linear => {
                self.interpolate_linear(word_start, word_end, tokens, start_index)
            }
            InterpolationMethod::CharacterProportional => {
                self.interpolate_char_proportional(word_start, word_end, tokens, start_index)
            }
            InterpolationMethod::Weighted => {
                self.interpolate_weighted(word_start, word_end, tokens, start_index)
            }
            InterpolationMethod::AttentionGuided => {
                // Falls back to weighted without attention weights
                self.interpolate_weighted(word_start, word_end, tokens, start_index)
            }
        }
    }

    /// Interpolate with attention guidance
    pub fn interpolate_with_attention(
        &self,
        word_start: f32,
        word_end: f32,
        tokens: &[String],
        attention_weights: &[Vec<f32>],
        start_index: usize,
        frame_rate: f32,
    ) -> WhisperResult<Vec<TokenTimestamp>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        if attention_weights.len() != tokens.len() {
            return self.interpolate_word_tokens(word_start, word_end, tokens, start_index);
        }

        let duration = word_end - word_start;
        let mut timestamps = Vec::with_capacity(tokens.len());
        let mut current_time = word_start;

        for (i, (text, attention)) in tokens.iter().zip(attention_weights.iter()).enumerate() {
            // Find peak attention frame for this token
            let (peak_frame, _) = attention
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            let peak_time = peak_frame as f32 / frame_rate;

            // Constrain to word boundaries
            let token_center = peak_time.clamp(word_start, word_end);

            // Estimate token duration
            let token_duration = if i + 1 < tokens.len() {
                let next_peak = attention_weights[i + 1]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx);
                let next_time = (next_peak as f32 / frame_rate).clamp(word_start, word_end);
                (next_time - token_center).max(duration / tokens.len() as f32 * 0.5)
            } else {
                word_end - token_center
            };

            let token_start = current_time;
            let token_end = (token_start + token_duration).min(word_end);

            timestamps.push(TokenTimestamp::interpolated(
                start_index + i,
                text.clone(),
                token_start,
                token_end,
                0.8, // Higher confidence with attention guidance
            ));

            current_time = token_end;
        }

        Ok(timestamps)
    }

    /// Linear interpolation
    #[allow(clippy::unnecessary_wraps)]
    fn interpolate_linear(
        &self,
        word_start: f32,
        word_end: f32,
        tokens: &[String],
        start_index: usize,
    ) -> WhisperResult<Vec<TokenTimestamp>> {
        let _ = self; // Method for consistency
        let duration = word_end - word_start;
        let token_duration = duration / tokens.len() as f32;

        let mut timestamps = Vec::with_capacity(tokens.len());
        let mut current_time = word_start;

        for (i, text) in tokens.iter().enumerate() {
            let token_end = current_time + token_duration;

            timestamps.push(TokenTimestamp::interpolated(
                start_index + i,
                text.clone(),
                current_time,
                token_end,
                0.5, // Low confidence for linear interpolation
            ));

            current_time = token_end;
        }

        Ok(timestamps)
    }

    /// Character-proportional interpolation
    #[allow(clippy::unnecessary_wraps)]
    fn interpolate_char_proportional(
        &self,
        word_start: f32,
        word_end: f32,
        tokens: &[String],
        start_index: usize,
    ) -> WhisperResult<Vec<TokenTimestamp>> {
        let _ = self; // Method for consistency
        let duration = word_end - word_start;
        let total_chars: usize = tokens.iter().map(|t| t.chars().count().max(1)).sum();

        let mut timestamps = Vec::with_capacity(tokens.len());
        let mut current_time = word_start;

        for (i, text) in tokens.iter().enumerate() {
            let char_count = text.chars().count().max(1);
            let token_duration = (char_count as f32 / total_chars as f32) * duration;
            let token_end = current_time + token_duration;

            timestamps.push(TokenTimestamp::interpolated(
                start_index + i,
                text.clone(),
                current_time,
                token_end,
                0.6, // Medium confidence
            ));

            current_time = token_end;
        }

        Ok(timestamps)
    }

    /// Weighted interpolation (combination of linear and character-proportional)
    #[allow(clippy::unnecessary_wraps)]
    fn interpolate_weighted(
        &self,
        word_start: f32,
        word_end: f32,
        tokens: &[String],
        start_index: usize,
    ) -> WhisperResult<Vec<TokenTimestamp>> {
        let duration = word_end - word_start;
        let total_chars: usize = tokens.iter().map(|t| t.chars().count().max(1)).sum();
        let uniform_duration = duration / tokens.len() as f32;

        let mut timestamps = Vec::with_capacity(tokens.len());
        let mut current_time = word_start;

        for (i, text) in tokens.iter().enumerate() {
            let char_count = text.chars().count().max(1);
            let char_duration = (char_count as f32 / total_chars as f32) * duration;

            let weighted_duration = self
                .config
                .char_weight
                .mul_add(char_duration, self.config.uniform_weight * uniform_duration);

            let token_end = (current_time + weighted_duration).min(word_end);

            timestamps.push(TokenTimestamp::interpolated(
                start_index + i,
                text.clone(),
                current_time,
                token_end,
                0.65, // Medium-high confidence for weighted
            ));

            current_time = token_end;
        }

        // Ensure last token ends at word_end
        if let Some(last) = timestamps.last_mut() {
            last.end = word_end;
        }

        Ok(timestamps)
    }

    /// Smooth timestamps using moving average
    pub fn smooth_timestamps(&self, timestamps: &mut [TokenTimestamp]) {
        if self.config.smoothing_window == 0 || timestamps.len() < 3 {
            return;
        }

        let window = self.config.smoothing_window;
        let mut smoothed_starts = Vec::with_capacity(timestamps.len());
        let mut smoothed_ends = Vec::with_capacity(timestamps.len());

        for i in 0..timestamps.len() {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(timestamps.len());

            let avg_start: f32 =
                timestamps[start..end].iter().map(|t| t.start).sum::<f32>() / (end - start) as f32;
            let avg_end: f32 =
                timestamps[start..end].iter().map(|t| t.end).sum::<f32>() / (end - start) as f32;

            smoothed_starts.push(avg_start);
            smoothed_ends.push(avg_end);
        }

        for (i, ts) in timestamps.iter_mut().enumerate() {
            if ts.interpolated {
                ts.start = smoothed_starts[i];
                ts.end = smoothed_ends[i];
            }
        }

        // Fix any overlaps
        for i in 1..timestamps.len() {
            if timestamps[i].start < timestamps[i - 1].end {
                let mid = (timestamps[i].start + timestamps[i - 1].end) / 2.0;
                timestamps[i - 1].end = mid;
                timestamps[i].start = mid;
            }
        }
    }
}

impl Default for TimestampInterpolator {
    fn default() -> Self {
        Self::new(InterpolationConfig::default())
    }
}

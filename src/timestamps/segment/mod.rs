//! Timestamp extraction and segmentation
//!
//! Extracts timestamps and segments from decoder token output.
//!
//! # Whisper Timestamp Format
//!
//! Whisper encodes timestamps as special tokens:
//! - Tokens >= TIMESTAMP_BASE are timestamp tokens
//! - Each timestamp represents 20ms (50 timestamps per second)
//! - Maximum timestamp: 30 seconds (1500 tokens)
//!
//! # Segment Structure
//!
//! Tokens follow the pattern:
//! `<|startoftranscript|> <|lang|> <|task|> <|notimestamps|> text... <|endoftext|>`
//!
//! Or with timestamps:
//! `<|startoftranscript|> <|lang|> <|task|> <|0.00|> text... <|2.50|> <|2.50|> more... <|5.00|> <|endoftext|>`

use crate::tokenizer::special_tokens;
use crate::Segment;

#[cfg(test)]
mod tests;

/// Maximum timestamp value in seconds (30 seconds)
pub const MAX_TIMESTAMP_SECONDS: f32 = 30.0;

/// Timestamp resolution in seconds (20ms)
pub const TIMESTAMP_RESOLUTION: f32 = 0.02;

/// Maximum number of timestamp tokens (30s / 20ms = 1500)
pub const MAX_TIMESTAMP_TOKENS: u32 = 1500;

/// Segment extraction state
struct SegmentExtractor {
    segments: Vec<Segment>,
    current_start: Option<f32>,
    current_tokens: Vec<u32>,
}

impl SegmentExtractor {
    fn new() -> Self {
        Self {
            segments: Vec::new(),
            current_start: None,
            current_tokens: Vec::new(),
        }
    }

    /// Try to finalize current segment and push to results
    fn try_finalize_segment<F>(&mut self, end_time: f32, tokenizer_decode: &mut F)
    where
        F: FnMut(&[u32]) -> Option<String>,
    {
        let Some(start) = self.current_start else {
            return;
        };

        if let Some(segment) = self.create_segment(start, end_time, tokenizer_decode) {
            self.segments.push(segment);
        }
        self.current_tokens.clear();
    }

    /// Create a segment from current tokens if valid
    fn create_segment<F>(&self, start: f32, end: f32, tokenizer_decode: &mut F) -> Option<Segment>
    where
        F: FnMut(&[u32]) -> Option<String>,
    {
        if self.current_tokens.is_empty() {
            return None;
        }

        let text = tokenizer_decode(&self.current_tokens)
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())?;

        Some(Segment {
            start,
            end,
            text,
            tokens: self.current_tokens.clone(),
        })
    }

    /// Handle a timestamp token
    fn handle_timestamp<F>(&mut self, time: f32, tokenizer_decode: &mut F)
    where
        F: FnMut(&[u32]) -> Option<String>,
    {
        self.try_finalize_segment(time, tokenizer_decode);
        self.current_start = Some(time);
    }

    /// Handle trailing tokens without end timestamp
    fn finalize_remaining<F>(&mut self, tokenizer_decode: &mut F)
    where
        F: FnMut(&[u32]) -> Option<String>,
    {
        let Some(start) = self.current_start else {
            return;
        };

        // Estimate end time based on token count (~60ms per token)
        let estimated_duration = (self.current_tokens.len() as f32) * 0.06;
        let end = start + estimated_duration;

        if let Some(segment) = self.create_segment(start, end, tokenizer_decode) {
            self.segments.push(segment);
        }
    }
}

/// Extract segments with timestamps from token sequence
///
/// # Arguments
/// * `tokens` - Token sequence from decoder
/// * `tokenizer_decode` - Function to decode tokens to text
///
/// # Returns
/// Vector of segments with timestamps and text
pub fn extract_segments<F>(tokens: &[u32], mut tokenizer_decode: F) -> Vec<Segment>
where
    F: FnMut(&[u32]) -> Option<String>,
{
    let mut extractor = SegmentExtractor::new();

    for &token in tokens {
        if is_control_token(token) {
            continue;
        }

        if special_tokens::is_timestamp(token) {
            let time = special_tokens::timestamp_to_seconds(token).unwrap_or(0.0);
            extractor.handle_timestamp(time, &mut tokenizer_decode);
        } else {
            extractor.current_tokens.push(token);
        }
    }

    extractor.finalize_remaining(&mut tokenizer_decode);
    extractor.segments
}

/// Check if a token is a control token (SOT, EOT, LANG, TASK, etc.)
pub fn is_control_token(token: u32) -> bool {
    token == special_tokens::SOT
        || token == special_tokens::EOT
        || token == special_tokens::TRANSCRIBE
        || token == special_tokens::TRANSLATE
        || token == special_tokens::NO_TIMESTAMPS
        || token == special_tokens::NO_SPEECH
        || is_language_token(token)
}

/// Check if a token is a language token
fn is_language_token(token: u32) -> bool {
    // Language tokens are in range [LANG_BASE, TRANSCRIBE)
    (special_tokens::LANG_BASE..special_tokens::TRANSCRIBE).contains(&token)
}

/// Check if a token is a timestamp token
#[must_use]
pub fn is_timestamp(token: u32) -> bool {
    special_tokens::is_timestamp(token)
}

/// Convert timestamp token to seconds
#[must_use]
pub fn timestamp_to_seconds(token: u32) -> Option<f32> {
    special_tokens::timestamp_to_seconds(token)
}

/// Parse timestamps from a token sequence (without decoding text)
///
/// Returns pairs of (start_time, end_time) for each segment.
pub fn parse_timestamp_pairs(tokens: &[u32]) -> Vec<(f32, f32)> {
    let timestamps: Vec<f32> = tokens
        .iter()
        .filter_map(|&token| special_tokens::timestamp_to_seconds(token))
        .collect();

    timestamps
        .windows(2)
        .filter(|w| w[1] > w[0])
        .map(|w| (w[0], w[1]))
        .collect()
}

/// Convert seconds to timestamp token
#[must_use]
pub fn seconds_to_timestamp_token(seconds: f32) -> u32 {
    let clamped = seconds.clamp(0.0, MAX_TIMESTAMP_SECONDS);
    let offset = (clamped / TIMESTAMP_RESOLUTION).round() as u32;
    special_tokens::TIMESTAMP_BASE + offset.min(MAX_TIMESTAMP_TOKENS)
}

/// Get all timestamp tokens in a token sequence
pub fn get_timestamps(tokens: &[u32]) -> Vec<(usize, f32)> {
    tokens
        .iter()
        .enumerate()
        .filter_map(|(idx, &token)| {
            special_tokens::timestamp_to_seconds(token).map(|time| (idx, time))
        })
        .collect()
}

/// Check if token sequence has timestamps
#[must_use]
pub fn has_timestamps(tokens: &[u32]) -> bool {
    tokens.iter().any(|&t| special_tokens::is_timestamp(t))
}

/// Count text tokens (non-special, non-timestamp)
#[must_use]
pub fn count_text_tokens(tokens: &[u32]) -> usize {
    tokens
        .iter()
        .filter(|&&t| !special_tokens::is_timestamp(t) && !is_control_token(t))
        .count()
}

/// Estimate duration from token count
///
/// Uses heuristic of ~60ms per token (based on typical speech rate)
#[must_use]
pub fn estimate_duration_from_tokens(token_count: usize) -> f32 {
    (token_count as f32) * 0.06
}

/// Merge adjacent segments if gap is small enough
pub fn merge_segments(segments: &[Segment], max_gap: f32) -> Vec<Segment> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut merged = Vec::new();
    let mut current = segments[0].clone();

    for segment in segments.iter().skip(1) {
        if segment.start - current.end <= max_gap {
            // Merge segments
            current.end = segment.end;
            current.text.push(' ');
            current.text.push_str(&segment.text);
            current.tokens.extend_from_slice(&segment.tokens);
        } else {
            merged.push(current);
            current = segment.clone();
        }
    }

    merged.push(current);
    merged
}

/// Split long segments at sentence boundaries
pub fn split_long_segments(segments: &[Segment], max_duration: f32) -> Vec<Segment> {
    let mut result = Vec::new();

    for segment in segments {
        if segment.end - segment.start <= max_duration {
            result.push(segment.clone());
        } else {
            // Split at sentence boundaries
            let sentences = split_sentences(&segment.text);
            if sentences.len() > 1 {
                let total_duration = segment.end - segment.start;
                let total_chars: usize = sentences.iter().map(|s| s.len()).sum();

                let mut current_time = segment.start;
                for sentence in sentences {
                    let sentence_duration =
                        (sentence.len() as f32 / total_chars as f32) * total_duration;
                    result.push(Segment {
                        start: current_time,
                        end: current_time + sentence_duration,
                        text: sentence,
                        tokens: vec![], // Tokens not preserved in split
                    });
                    current_time += sentence_duration;
                }
            } else {
                result.push(segment.clone());
            }
        }
    }

    result
}

/// Split text at sentence boundaries
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Add remaining text
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

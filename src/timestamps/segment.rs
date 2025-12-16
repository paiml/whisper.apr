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

        let text = tokenizer_decode(&self.current_tokens)?;
        let text = text.trim().to_string();
        if text.is_empty() {
            return None;
        }

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
        if self.current_start.is_some() {
            self.try_finalize_segment(time, tokenizer_decode);
        }
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

        if self.current_tokens.is_empty() {
            return;
        }

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
    let mut pairs = Vec::new();
    let mut timestamps: Vec<f32> = Vec::new();

    for &token in tokens {
        if special_tokens::is_timestamp(token) {
            if let Some(time) = special_tokens::timestamp_to_seconds(token) {
                timestamps.push(time);

                // Every two timestamps form a pair
                if timestamps.len() >= 2 {
                    let start = timestamps[timestamps.len() - 2];
                    let end = timestamps[timestamps.len() - 1];
                    if end > start {
                        pairs.push((start, end));
                    }
                }
            }
        }
    }

    pairs
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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Timestamp Token Tests
    // =========================================================================

    #[test]
    fn test_seconds_to_timestamp_token() {
        assert_eq!(
            seconds_to_timestamp_token(0.0),
            special_tokens::TIMESTAMP_BASE
        );
        assert_eq!(
            seconds_to_timestamp_token(1.0),
            special_tokens::TIMESTAMP_BASE + 50
        );
        assert_eq!(
            seconds_to_timestamp_token(0.5),
            special_tokens::TIMESTAMP_BASE + 25
        );
    }

    #[test]
    fn test_seconds_to_timestamp_token_clamped() {
        // Should clamp to 30 seconds max
        assert_eq!(
            seconds_to_timestamp_token(35.0),
            special_tokens::TIMESTAMP_BASE + MAX_TIMESTAMP_TOKENS
        );
        assert_eq!(
            seconds_to_timestamp_token(-1.0),
            special_tokens::TIMESTAMP_BASE
        );
    }

    #[test]
    fn test_timestamp_roundtrip() {
        for seconds in [0.0, 0.5, 1.0, 5.5, 10.0, 29.98] {
            let token = seconds_to_timestamp_token(seconds);
            let recovered = special_tokens::timestamp_to_seconds(token).unwrap_or(0.0);
            assert!((seconds - recovered).abs() < 0.02, "Failed at {seconds}");
        }
    }

    // =========================================================================
    // Control Token Tests
    // =========================================================================

    #[test]
    fn test_is_control_token() {
        assert!(is_control_token(special_tokens::SOT));
        assert!(is_control_token(special_tokens::EOT));
        assert!(is_control_token(special_tokens::TRANSCRIBE));
        assert!(is_control_token(special_tokens::TRANSLATE));
        assert!(is_control_token(special_tokens::NO_TIMESTAMPS));
        assert!(!is_control_token(100)); // Regular token
    }

    #[test]
    fn test_is_language_token() {
        assert!(is_language_token(special_tokens::LANG_BASE)); // English
        assert!(is_language_token(special_tokens::LANG_BASE + 50)); // Some language
        assert!(!is_language_token(special_tokens::SOT));
        assert!(!is_language_token(special_tokens::TRANSCRIBE));
    }

    // =========================================================================
    // Timestamp Extraction Tests
    // =========================================================================

    #[test]
    fn test_has_timestamps() {
        let with_ts = vec![100, special_tokens::TIMESTAMP_BASE, 200];
        assert!(has_timestamps(&with_ts));

        let without_ts = vec![100, 200, 300];
        assert!(!has_timestamps(&without_ts));
    }

    #[test]
    fn test_get_timestamps() {
        let tokens = vec![
            special_tokens::SOT,
            special_tokens::TIMESTAMP_BASE, // 0.0s
            100,
            101,
            special_tokens::TIMESTAMP_BASE + 50, // 1.0s
        ];

        let timestamps = get_timestamps(&tokens);
        assert_eq!(timestamps.len(), 2);
        assert_eq!(timestamps[0], (1, 0.0));
        assert!((timestamps[1].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_count_text_tokens() {
        let tokens = vec![
            special_tokens::SOT,
            special_tokens::TIMESTAMP_BASE,
            100,
            101,
            102,
            special_tokens::TIMESTAMP_BASE + 50,
            special_tokens::EOT,
        ];

        assert_eq!(count_text_tokens(&tokens), 3);
    }

    // =========================================================================
    // Timestamp Pair Parsing Tests
    // =========================================================================

    #[test]
    fn test_parse_timestamp_pairs() {
        let tokens = vec![
            special_tokens::TIMESTAMP_BASE, // 0.0s
            100,
            special_tokens::TIMESTAMP_BASE + 50, // 1.0s
            special_tokens::TIMESTAMP_BASE + 50, // 1.0s (repeated)
            200,
            special_tokens::TIMESTAMP_BASE + 100, // 2.0s
        ];

        let pairs = parse_timestamp_pairs(&tokens);
        assert_eq!(pairs.len(), 2);
        assert!((pairs[0].0 - 0.0).abs() < 0.01);
        assert!((pairs[0].1 - 1.0).abs() < 0.01);
        assert!((pairs[1].0 - 1.0).abs() < 0.01);
        assert!((pairs[1].1 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_timestamp_pairs_empty() {
        let pairs = parse_timestamp_pairs(&[100, 200, 300]);
        assert!(pairs.is_empty());
    }

    // =========================================================================
    // Segment Extraction Tests
    // =========================================================================

    #[test]
    fn test_extract_segments_basic() {
        let tokens = vec![
            special_tokens::TIMESTAMP_BASE, // 0.0s
            104,
            105,                                 // "hi"
            special_tokens::TIMESTAMP_BASE + 50, // 1.0s
        ];

        let decode = |ts: &[u32]| -> Option<String> {
            // Simple mock decoder
            if ts == &[104, 105] {
                Some("hi".to_string())
            } else {
                Some(ts.iter().map(|t| format!("{t}")).collect())
            }
        };

        let segments = extract_segments(&tokens, decode);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "hi");
        assert!((segments[0].start - 0.0).abs() < 0.01);
        assert!((segments[0].end - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_segments_multiple() {
        let tokens = vec![
            special_tokens::TIMESTAMP_BASE, // 0.0s
            1,
            2,
            special_tokens::TIMESTAMP_BASE + 50, // 1.0s
            special_tokens::TIMESTAMP_BASE + 50, // 1.0s (repeated start)
            3,
            4,
            special_tokens::TIMESTAMP_BASE + 100, // 2.0s
        ];

        let decode = |ts: &[u32]| -> Option<String> { Some(format!("{:?}", ts)) };

        let segments = extract_segments(&tokens, decode);
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_extract_segments_skips_control_tokens() {
        let tokens = vec![
            special_tokens::SOT,
            special_tokens::LANG_BASE, // English
            special_tokens::TRANSCRIBE,
            special_tokens::TIMESTAMP_BASE, // 0.0s
            100,
            special_tokens::TIMESTAMP_BASE + 50, // 1.0s
            special_tokens::EOT,
        ];

        let decode = |ts: &[u32]| -> Option<String> { Some(format!("{:?}", ts)) };

        let segments = extract_segments(&tokens, decode);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].tokens, vec![100]);
    }

    #[test]
    fn test_extract_segments_empty_text_skipped() {
        let tokens = vec![
            special_tokens::TIMESTAMP_BASE,      // 0.0s
            special_tokens::TIMESTAMP_BASE + 50, // 1.0s (no text between)
        ];

        let decode = |_ts: &[u32]| -> Option<String> { Some(String::new()) };

        let segments = extract_segments(&tokens, decode);
        assert!(segments.is_empty());
    }

    // =========================================================================
    // Duration Estimation Tests
    // =========================================================================

    #[test]
    fn test_estimate_duration_from_tokens() {
        assert!((estimate_duration_from_tokens(10) - 0.6).abs() < 0.01);
        assert!((estimate_duration_from_tokens(100) - 6.0).abs() < 0.01);
    }

    // =========================================================================
    // Segment Merging Tests
    // =========================================================================

    #[test]
    fn test_merge_segments_close() {
        let segments = vec![
            Segment {
                start: 0.0,
                end: 1.0,
                text: "Hello".to_string(),
                tokens: vec![1, 2],
            },
            Segment {
                start: 1.1,
                end: 2.0,
                text: "World".to_string(),
                tokens: vec![3, 4],
            },
        ];

        let merged = merge_segments(&segments, 0.2);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "Hello World");
        assert_eq!(merged[0].tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_merge_segments_far() {
        let segments = vec![
            Segment {
                start: 0.0,
                end: 1.0,
                text: "Hello".to_string(),
                tokens: vec![1, 2],
            },
            Segment {
                start: 5.0,
                end: 6.0,
                text: "World".to_string(),
                tokens: vec![3, 4],
            },
        ];

        let merged = merge_segments(&segments, 0.2);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_segments_empty() {
        let merged = merge_segments(&[], 0.5);
        assert!(merged.is_empty());
    }

    // =========================================================================
    // Sentence Splitting Tests
    // =========================================================================

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello. World! How are you?");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello.");
        assert_eq!(sentences[1], "World!");
        assert_eq!(sentences[2], "How are you?");
    }

    #[test]
    fn test_split_sentences_no_punctuation() {
        let sentences = split_sentences("Hello world");
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "Hello world");
    }

    #[test]
    fn test_split_long_segments() {
        let segments = vec![Segment {
            start: 0.0,
            end: 20.0,
            text: "Hello. World.".to_string(),
            tokens: vec![],
        }];

        let split = split_long_segments(&segments, 5.0);
        assert_eq!(split.len(), 2);
    }

    #[test]
    fn test_split_long_segments_short_segment() {
        let segments = vec![Segment {
            start: 0.0,
            end: 2.0,
            text: "Hello".to_string(),
            tokens: vec![1],
        }];

        let split = split_long_segments(&segments, 5.0);
        assert_eq!(split.len(), 1);
        assert_eq!(split[0].tokens, vec![1]); // Tokens preserved
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_constants() {
        assert!((MAX_TIMESTAMP_SECONDS - 30.0).abs() < f32::EPSILON);
        assert!((TIMESTAMP_RESOLUTION - 0.02).abs() < f32::EPSILON);
        assert_eq!(MAX_TIMESTAMP_TOKENS, 1500);
    }
}

//! Tests for timestamp segment extraction

use super::*;
use crate::tokenizer::special_tokens;
use crate::Segment;

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
        if ts == [104, 105] {
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

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_is_timestamp_wrapper() {
    assert!(is_timestamp(special_tokens::TIMESTAMP_BASE));
    assert!(is_timestamp(special_tokens::TIMESTAMP_BASE + 100));
    assert!(!is_timestamp(100));
}

#[test]
fn test_timestamp_to_seconds_wrapper() {
    let seconds = timestamp_to_seconds(special_tokens::TIMESTAMP_BASE + 50);
    assert!(seconds.is_some());
    assert!((seconds.unwrap() - 1.0).abs() < 0.01);

    let not_timestamp = timestamp_to_seconds(100);
    assert!(not_timestamp.is_none());
}

#[test]
fn test_is_control_token_no_speech() {
    assert!(is_control_token(special_tokens::NO_SPEECH));
}

#[test]
fn test_extract_segments_with_trailing_tokens() {
    // Tokens after timestamp without closing timestamp
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE, // 0.0s
        100,
        101,
        102,
        // No closing timestamp - should trigger finalize_remaining
    ];

    let decode = |ts: &[u32]| -> Option<String> { Some(format!("tokens:{}", ts.len())) };

    let segments = extract_segments(&tokens, decode);
    assert_eq!(segments.len(), 1);
    // End time should be estimated from token count
    assert!(segments[0].end > segments[0].start);
}

#[test]
fn test_extract_segments_decoder_returns_none() {
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE,
        100,
        special_tokens::TIMESTAMP_BASE + 50,
    ];

    // Decoder returns None - segment should be skipped
    let decode = |_ts: &[u32]| -> Option<String> { None };

    let segments = extract_segments(&tokens, decode);
    assert!(segments.is_empty());
}

#[test]
fn test_extract_segments_whitespace_only_text() {
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE,
        100,
        special_tokens::TIMESTAMP_BASE + 50,
    ];

    // Decoder returns only whitespace - segment should be skipped
    let decode = |_ts: &[u32]| -> Option<String> { Some("   \t\n  ".to_string()) };

    let segments = extract_segments(&tokens, decode);
    assert!(segments.is_empty());
}

#[test]
fn test_parse_timestamp_pairs_invalid_order() {
    // Timestamps where end <= start should not be added
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE + 100, // 2.0s
        special_tokens::TIMESTAMP_BASE + 50,  // 1.0s (end < start)
    ];

    let pairs = parse_timestamp_pairs(&tokens);
    assert!(pairs.is_empty());
}

#[test]
fn test_parse_timestamp_pairs_equal() {
    // Timestamps where end == start should not be added
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE + 50, // 1.0s
        special_tokens::TIMESTAMP_BASE + 50, // 1.0s (end == start)
    ];

    let pairs = parse_timestamp_pairs(&tokens);
    assert!(pairs.is_empty());
}

#[test]
fn test_split_long_segments_single_sentence() {
    // Long segment with only one sentence (no split possible)
    let segments = vec![Segment {
        start: 0.0,
        end: 20.0,
        text: "Hello world no punctuation here".to_string(),
        tokens: vec![1, 2, 3],
    }];

    let split = split_long_segments(&segments, 5.0);
    assert_eq!(split.len(), 1);
    assert_eq!(split[0].tokens, vec![1, 2, 3]); // Original preserved
}

#[test]
fn test_split_sentences_empty() {
    let sentences = split_sentences("");
    assert!(sentences.is_empty());
}

#[test]
fn test_split_sentences_only_punctuation() {
    // Each punctuation creates a sentence with just the punctuation
    let sentences = split_sentences("...");
    assert_eq!(sentences.len(), 3);
    assert!(sentences.iter().all(|s| s == "."));
}

#[test]
fn test_merge_segments_three_segments() {
    let segments = vec![
        Segment {
            start: 0.0,
            end: 1.0,
            text: "A".to_string(),
            tokens: vec![1],
        },
        Segment {
            start: 1.1,
            end: 2.0,
            text: "B".to_string(),
            tokens: vec![2],
        },
        Segment {
            start: 2.1,
            end: 3.0,
            text: "C".to_string(),
            tokens: vec![3],
        },
    ];

    let merged = merge_segments(&segments, 0.2);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].text, "A B C");
}

#[test]
fn test_get_timestamps_empty() {
    let timestamps = get_timestamps(&[]);
    assert!(timestamps.is_empty());
}

#[test]
fn test_has_timestamps_empty() {
    assert!(!has_timestamps(&[]));
}

#[test]
fn test_count_text_tokens_empty() {
    assert_eq!(count_text_tokens(&[]), 0);
}

#[test]
fn test_count_text_tokens_all_special() {
    let tokens = vec![
        special_tokens::SOT,
        special_tokens::EOT,
        special_tokens::TIMESTAMP_BASE,
    ];
    assert_eq!(count_text_tokens(&tokens), 0);
}

#[test]
fn test_extract_segments_no_tokens_between_timestamps() {
    // No text tokens between consecutive timestamps
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE,
        special_tokens::TIMESTAMP_BASE + 50,
        special_tokens::TIMESTAMP_BASE + 50,
        special_tokens::TIMESTAMP_BASE + 100,
    ];

    let decode = |ts: &[u32]| -> Option<String> {
        if ts.is_empty() {
            Some(String::new())
        } else {
            Some(format!("{:?}", ts))
        }
    };

    let segments = extract_segments(&tokens, decode);
    // Empty segments should be skipped
    assert!(segments.is_empty());
}

#[test]
fn test_segment_extractor_handle_timestamp_no_prior_start() {
    // First timestamp should just set start, not finalize anything
    let tokens = vec![
        100, // Text before any timestamp (should be ignored)
        special_tokens::TIMESTAMP_BASE,
        101,
        special_tokens::TIMESTAMP_BASE + 50,
    ];

    let decode = |ts: &[u32]| -> Option<String> { Some(format!("{:?}", ts)) };

    let segments = extract_segments(&tokens, decode);
    // Only the segment between timestamps should be extracted
    // Text before first timestamp is accumulated but never finalized
    assert_eq!(segments.len(), 1);
}

// =========================================================================
// Additional Coverage Tests for Edge Cases
// =========================================================================

#[test]
fn test_extract_segments_trailing_tokens_estimated_duration() {
    // Tokens after final timestamp - duration estimated from token count
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE, // 0.0s
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109, // 10 text tokens
             // No closing timestamp
    ];

    let decode = |ts: &[u32]| -> Option<String> { Some(format!("text:{}", ts.len())) };

    let segments = extract_segments(&tokens, decode);
    assert_eq!(segments.len(), 1);
    // Duration should be estimated: 10 tokens * 0.06s = 0.6s
    assert!(
        (segments[0].end - segments[0].start - 0.6).abs() < 0.01,
        "Expected 0.6s duration, got {}",
        segments[0].end - segments[0].start
    );
}

#[test]
fn test_extract_segments_finalize_remaining_no_start() {
    // No timestamp at all - finalize_remaining should do nothing
    let tokens = vec![100, 101, 102]; // Text only, no timestamps

    let decode = |ts: &[u32]| -> Option<String> { Some(format!("{:?}", ts)) };

    let segments = extract_segments(&tokens, decode);
    // No start timestamp means finalize_remaining returns early
    assert!(segments.is_empty());
}

#[test]
fn test_extract_segments_finalize_remaining_empty_tokens() {
    // Start timestamp but no text tokens
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE, // 0.0s - sets start
                                        // No text tokens, no closing timestamp
    ];

    let decode = |ts: &[u32]| -> Option<String> { Some(format!("{:?}", ts)) };

    let segments = extract_segments(&tokens, decode);
    // current_tokens is empty, so finalize_remaining returns early
    assert!(segments.is_empty());
}

#[test]
fn test_split_long_segments_multiple_sentences() {
    // Test splitting with more than 2 sentences
    let segments = vec![Segment {
        start: 0.0,
        end: 30.0,
        text: "First sentence. Second sentence. Third sentence.".to_string(),
        tokens: vec![1, 2, 3, 4, 5],
    }];

    let split = split_long_segments(&segments, 5.0);
    assert_eq!(split.len(), 3);

    // Verify segments have valid times
    for seg in &split {
        assert!(seg.end > seg.start);
        assert!(seg.start >= 0.0);
        assert!(seg.end <= 30.0);
    }

    // Tokens not preserved in split segments
    assert!(split[0].tokens.is_empty());
}

#[test]
fn test_merge_segments_single() {
    // Single segment - should return as-is
    let segments = vec![Segment {
        start: 0.0,
        end: 1.0,
        text: "Hello".to_string(),
        tokens: vec![1],
    }];

    let merged = merge_segments(&segments, 0.5);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].text, "Hello");
}

#[test]
fn test_split_sentences_with_spaces() {
    // Test sentence splitting with extra spaces
    let sentences = split_sentences("  Hello.  World!  ");
    assert_eq!(sentences.len(), 2);
    assert_eq!(sentences[0], "Hello.");
    assert_eq!(sentences[1], "World!");
}

#[test]
fn test_try_finalize_segment_no_start() {
    // Test when handle_timestamp is called with no prior start set
    // The first timestamp just sets start, doesn't finalize anything
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE, // First timestamp - sets start
        // No text tokens between
        special_tokens::TIMESTAMP_BASE + 50, // Second timestamp - try_finalize with empty tokens
    ];

    let decode = |ts: &[u32]| -> Option<String> {
        if ts.is_empty() {
            Some(String::new())
        } else {
            Some(format!("{:?}", ts))
        }
    };

    let segments = extract_segments(&tokens, decode);
    // No text between timestamps, so segments should be empty
    // (empty text segments are filtered out)
    assert!(segments.is_empty());
}

#[test]
fn test_create_segment_decoder_none() {
    // Test when decoder returns None in create_segment
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE,
        100,
        special_tokens::TIMESTAMP_BASE + 50,
    ];

    // First call returns text, second returns None
    let mut call_count = 0;
    let decode = |_ts: &[u32]| -> Option<String> {
        call_count += 1;
        if call_count == 1 {
            None
        } else {
            Some("text".to_string())
        }
    };

    let segments = extract_segments(&tokens, decode);
    // First decode returns None, so segment is skipped
    assert!(segments.is_empty());
}

#[test]
fn test_parse_timestamp_pairs_single_timestamp() {
    // Only one timestamp - can't form a pair
    let tokens = vec![special_tokens::TIMESTAMP_BASE];
    let pairs = parse_timestamp_pairs(&tokens);
    assert!(pairs.is_empty());
}

#[test]
fn test_split_long_segments_empty() {
    let split = split_long_segments(&[], 5.0);
    assert!(split.is_empty());
}

#[test]
fn test_segment_extractor_multiple_finalize() {
    // Multiple consecutive timestamps - each triggers try_finalize
    let tokens = vec![
        special_tokens::TIMESTAMP_BASE,      // 0.0s
        100,                                 // text
        special_tokens::TIMESTAMP_BASE + 25, // 0.5s
        special_tokens::TIMESTAMP_BASE + 25, // 0.5s (repeat - triggers finalize with empty)
        special_tokens::TIMESTAMP_BASE + 50, // 1.0s (triggers finalize with empty)
        101,                                 // text
        special_tokens::TIMESTAMP_BASE + 75, // 1.5s
    ];

    let decode = |ts: &[u32]| -> Option<String> {
        if ts.is_empty() {
            Some(String::new())
        } else {
            Some(format!("{:?}", ts))
        }
    };

    let segments = extract_segments(&tokens, decode);
    // Only segments with actual text should be included
    assert_eq!(segments.len(), 2);
}

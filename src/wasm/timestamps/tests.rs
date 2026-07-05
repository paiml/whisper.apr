#![allow(clippy::expect_used)]
//! Tests for WASM timestamp bindings

use super::*;

#[test]
fn test_alignment_config_wasm_default() {
    let config = AlignmentConfigWasm::new();
    assert_eq!(config.layers.len(), 6);
    assert!((config.min_attention - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_alignment_config_wasm_for_accuracy() {
    let config = AlignmentConfigWasm::for_accuracy();
    assert!(config.use_median);
    assert!((config.min_attention - 0.05).abs() < f32::EPSILON);
}

#[test]
fn test_alignment_config_wasm_for_speed() {
    let config = AlignmentConfigWasm::for_speed();
    assert_eq!(config.layers.len(), 2);
}

#[test]
fn test_alignment_config_wasm_setters() {
    let mut config = AlignmentConfigWasm::new();
    config.set_min_attention(0.2);
    assert!((config.min_attention() - 0.2).abs() < f32::EPSILON);
}

#[test]
fn test_word_with_timestamp_wasm() {
    let word = WordWithTimestamp::new("hello".to_string(), 0.0, 1.0, 0.9);
    let wasm: WordWithTimestampWasm = word.into();

    assert_eq!(wasm.word(), "hello");
    assert!((wasm.start() - 0.0).abs() < f32::EPSILON);
    assert!((wasm.end() - 1.0).abs() < f32::EPSILON);
    assert!((wasm.duration() - 1.0).abs() < f32::EPSILON);
    assert!((wasm.confidence() - 0.9).abs() < f32::EPSILON);
    assert!(wasm.is_high_confidence());
}

#[test]
fn test_word_timestamp_result_wasm() {
    let words = vec![
        WordWithTimestamp::new("hello".to_string(), 0.0, 0.5, 0.9),
        WordWithTimestamp::new("world".to_string(), 0.5, 1.0, 0.8),
    ];
    let result = WordTimestampResult::new(words, 0.0, 1.0);
    let wasm: WordTimestampResultWasm = result.into();

    assert_eq!(wasm.word_count(), 2);
    assert!((wasm.segment_start() - 0.0).abs() < f32::EPSILON);
    assert!((wasm.segment_end() - 1.0).abs() < f32::EPSILON);
    assert!(wasm.is_high_quality());
}

#[test]
fn test_word_timestamp_result_wasm_get_word() {
    let words = vec![WordWithTimestamp::new("test".to_string(), 0.0, 1.0, 0.85)];
    let result = WordTimestampResult::new(words, 0.0, 1.0);
    let wasm: WordTimestampResultWasm = result.into();

    let word = wasm.get_word(0).expect("should have word");
    assert_eq!(word.word(), "test");
    assert!(wasm.get_word(1).is_none());
}

#[test]
fn test_word_timestamp_result_wasm_arrays() {
    let words = vec![
        WordWithTimestamp::new("a".to_string(), 0.0, 0.5, 0.9),
        WordWithTimestamp::new("b".to_string(), 0.5, 1.0, 0.8),
    ];
    let result = WordTimestampResult::new(words, 0.0, 1.0);
    let wasm: WordTimestampResultWasm = result.into();

    assert_eq!(wasm.word_texts(), vec!["a", "b"]);
    assert_eq!(wasm.word_starts().len(), 2);
    assert_eq!(wasm.word_ends().len(), 2);
    assert_eq!(wasm.word_confidences().len(), 2);
}

#[test]
fn test_word_timestamp_result_wasm_to_json() {
    let words = vec![WordWithTimestamp::new("hello".to_string(), 0.0, 0.5, 0.9)];
    let result = WordTimestampResult::new(words, 0.0, 1.0);
    let wasm: WordTimestampResultWasm = result.into();

    let json = wasm.to_json();
    assert!(json.contains("\"word\":\"hello\""));
    assert!(json.contains("\"start\":0"));
}

#[test]
fn test_token_timestamp_wasm() {
    let ts = TokenTimestamp::interpolated(0, "hel".to_string(), 0.0, 0.3, 0.6);
    let wasm: TokenTimestampWasm = ts.into();

    assert_eq!(wasm.index(), 0);
    assert_eq!(wasm.text(), "hel");
    assert!(wasm.interpolated());
    assert!((wasm.confidence() - 0.6).abs() < f32::EPSILON);
}

#[test]
fn test_word_boundary_wasm() {
    let boundary = WordBoundary::new(0.0, 1.0)
        .with_confidence(0.8, 0.9)
        .with_audio_refined(true);
    let wasm: WordBoundaryWasm = boundary.into();

    assert!((wasm.start() - 0.0).abs() < f32::EPSILON);
    assert!((wasm.end() - 1.0).abs() < f32::EPSILON);
    assert!((wasm.confidence() - 0.85).abs() < f32::EPSILON);
    assert!(wasm.audio_refined());
}

#[test]
fn test_word_timestamp_extractor_wasm_new() {
    let _extractor = WordTimestampExtractorWasm::new();
}

#[test]
fn test_word_timestamp_extractor_wasm_with_config() {
    let config = AlignmentConfigWasm::for_accuracy();
    let _extractor = WordTimestampExtractorWasm::with_config(config);
}

#[test]
fn test_word_timestamp_extractor_wasm_for_accuracy() {
    let _extractor = WordTimestampExtractorWasm::for_accuracy();
}

#[test]
fn test_word_timestamp_extractor_wasm_for_speed() {
    let _extractor = WordTimestampExtractorWasm::for_speed();
}

#[test]
fn test_timestamp_interpolator_wasm_new() {
    let _interpolator = TimestampInterpolatorWasm::new();
}

#[test]
fn test_timestamp_interpolator_wasm_linear() {
    let _interpolator = TimestampInterpolatorWasm::linear();
}

#[test]
fn test_timestamp_interpolator_wasm_character_proportional() {
    let _interpolator = TimestampInterpolatorWasm::character_proportional();
}

#[test]
fn test_get_word_timestamp_recommendation() {
    let karaoke = get_word_timestamp_recommendation("karaoke");
    assert!(karaoke.contains("forAccuracy"));

    let realtime = get_word_timestamp_recommendation("realtime");
    assert!(realtime.contains("forSpeed"));
}

// ============================================================
// Additional Coverage Tests
// ============================================================

#[test]
fn test_alignment_config_wasm_setters_coverage() {
    let mut config = AlignmentConfigWasm::new();
    config.set_layers(vec![1, 2, 3]);
    config.set_min_attention(0.5);
    config.set_temperature(0.1);
    config.set_use_median(true);

    assert!((config.min_attention() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_word_with_timestamp_wasm_high_confidence() {
    let word = WordWithTimestamp::new("hello".to_string(), 0.0, 1.0, 0.95);
    let wasm: WordWithTimestampWasm = word.into();
    assert!(wasm.is_high_confidence());
}

#[test]
fn test_word_with_timestamp_wasm_low_confidence() {
    let word = WordWithTimestamp::new("hello".to_string(), 0.0, 1.0, 0.5);
    let wasm: WordWithTimestampWasm = word.into();
    assert!(!wasm.is_high_confidence());
}

#[test]
fn test_word_timestamp_result_wasm_quality() {
    let words = vec![
        WordWithTimestamp::new("a".to_string(), 0.0, 0.5, 0.95),
        WordWithTimestamp::new("b".to_string(), 0.5, 1.0, 0.9),
    ];
    let result = WordTimestampResult::new(words, 0.0, 1.0);
    let wasm: WordTimestampResultWasm = result.into();

    assert!(wasm.is_high_quality());
}

#[test]
fn test_word_timestamp_result_wasm_segment_times() {
    let words = vec![WordWithTimestamp::new("test".to_string(), 1.0, 2.0, 0.9)];
    let result = WordTimestampResult::new(words, 1.0, 2.0);
    let wasm: WordTimestampResultWasm = result.into();

    assert!((wasm.segment_start() - 1.0).abs() < f32::EPSILON);
    assert!((wasm.segment_end() - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_token_timestamp_wasm_duration() {
    let ts = TokenTimestamp::interpolated(0, "hel".to_string(), 0.0, 0.5, 0.6);
    let wasm: TokenTimestampWasm = ts.into();

    assert!((wasm.duration() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_word_boundary_wasm_duration() {
    let boundary = WordBoundary::new(0.0, 0.5);
    let wasm: WordBoundaryWasm = boundary.into();

    assert!((wasm.duration() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_get_word_timestamp_recommendation_subtitles() {
    let rec = get_word_timestamp_recommendation("subtitles");
    assert!(!rec.is_empty());
}

#[test]
fn test_get_word_timestamp_recommendation_unknown() {
    let rec = get_word_timestamp_recommendation("unknown_use_case");
    assert!(rec.contains("Unknown use case"));
}

#[test]
fn test_alignment_config_wasm_default_impl_trait() {
    let config = AlignmentConfigWasm::default();
    // Default layers should be the same as new()
    assert_eq!(config.layers.len(), 6);
    assert!((config.min_attention - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_word_alignment_to_word_with_timestamp_wasm() {
    use crate::timestamps::alignment::WordAlignment;
    let alignment = WordAlignment {
        word: "hello".to_string(),
        start_time: 0.5,
        end_time: 1.0,
        confidence: 0.9,
        tokens: vec![],
    };
    let wasm: WordWithTimestampWasm = alignment.into();
    assert_eq!(wasm.word(), "hello");
    assert!((wasm.start() - 0.5).abs() < f32::EPSILON);
    assert!((wasm.end() - 1.0).abs() < f32::EPSILON);
    assert!((wasm.confidence() - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_word_timestamp_result_wasm_alignment_confidence() {
    let words = vec![WordWithTimestamp::new("test".to_string(), 0.0, 1.0, 0.95)];
    let result = WordTimestampResult::new(words, 0.0, 1.0);
    let wasm: WordTimestampResultWasm = result.into();

    // alignment_confidence should be average of word confidences
    assert!(wasm.alignment_confidence() >= 0.0);
    assert!(wasm.alignment_confidence() <= 1.0);
}

//! Tests for parity testing framework

use super::*;

/// Generate a word sequence "word0 word1 word2 ..." for parity testing
fn word_sequence(count: usize) -> String {
    (0..count)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// -------------------------------------------------------------------------
// ParityResult tests
// -------------------------------------------------------------------------

#[test]
fn test_parity_result_pass() {
    let result = ParityResult::Pass {
        wer: 0.005,
        timestamp_tolerance_ms: Some(50),
    };
    assert!(result.is_pass());
    assert!(!result.is_fail());
}

#[test]
fn test_parity_result_fail() {
    let result = ParityResult::Fail {
        wer: 0.15,
        expected: "hello world".to_string(),
        actual: "hello word".to_string(),
        differences: vec![],
    };
    assert!(result.is_fail());
    assert!(!result.is_pass());
}

// -------------------------------------------------------------------------
// ParityConfig tests
// -------------------------------------------------------------------------

#[test]
fn test_parity_config_default() {
    let config = ParityConfig::default();
    assert!((config.max_wer - 0.01).abs() < f64::EPSILON);
    assert_eq!(config.timestamp_tolerance_ms, 50);
    assert!(config.normalize_whitespace);
    assert!(!config.normalize_punctuation);
    assert!(!config.case_insensitive);
}

// -------------------------------------------------------------------------
// calculate_wer tests
// -------------------------------------------------------------------------

#[test]
fn test_wer_identical() {
    let wer = calculate_wer("hello world", "hello world");
    assert!((wer - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_wer_one_word_different() {
    // "hello world" vs "hello word" - 1 substitution out of 2 words = 0.5
    let wer = calculate_wer("hello world", "hello word");
    assert!((wer - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_wer_completely_different() {
    let wer = calculate_wer("hello world", "goodbye earth");
    assert!((wer - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_wer_empty_reference() {
    let wer = calculate_wer("", "hello");
    assert!((wer - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_wer_empty_both() {
    let wer = calculate_wer("", "");
    assert!((wer - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_wer_insertion() {
    // "a b" vs "a x b" - 1 insertion out of 2 words
    let wer = calculate_wer("a b", "a x b");
    assert!((wer - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_wer_deletion() {
    // "a b c" vs "a c" - 1 deletion out of 3 words
    let wer = calculate_wer("a b c", "a c");
    assert!((wer - 1.0 / 3.0).abs() < 0.01);
}

// -------------------------------------------------------------------------
// levenshtein_distance tests
// -------------------------------------------------------------------------

#[test]
fn test_levenshtein_identical() {
    let dist = levenshtein_distance(&["a", "b", "c"], &["a", "b", "c"]);
    assert_eq!(dist, 0);
}

#[test]
fn test_levenshtein_one_substitution() {
    let dist = levenshtein_distance(&["a", "b", "c"], &["a", "x", "c"]);
    assert_eq!(dist, 1);
}

#[test]
fn test_levenshtein_one_insertion() {
    let dist = levenshtein_distance(&["a", "b"], &["a", "x", "b"]);
    assert_eq!(dist, 1);
}

#[test]
fn test_levenshtein_one_deletion() {
    let dist = levenshtein_distance(&["a", "b", "c"], &["a", "c"]);
    assert_eq!(dist, 1);
}

#[test]
fn test_levenshtein_empty_first() {
    let dist = levenshtein_distance::<&str>(&[], &["a", "b"]);
    assert_eq!(dist, 2);
}

#[test]
fn test_levenshtein_empty_second() {
    let dist = levenshtein_distance(&["a", "b"], &[]);
    assert_eq!(dist, 2);
}

// -------------------------------------------------------------------------
// ParityTest tests
// -------------------------------------------------------------------------

#[test]
fn test_parity_test_exact_match() {
    let test = ParityTest::new(
        PathBuf::from("test.wav"),
        "Hello world".to_string(),
        "Hello world".to_string(),
    );

    let result = test.verify_text_parity();
    assert!(result.is_pass());
}

#[test]
fn test_parity_test_whitespace_normalization() {
    let test = ParityTest::new(
        PathBuf::from("test.wav"),
        "Hello   world".to_string(),
        "Hello world".to_string(),
    );

    let result = test.verify_text_parity();
    assert!(result.is_pass());
}

#[test]
fn test_parity_test_within_tolerance() {
    // 1 word different out of 100 = 1% WER, at threshold
    let reference = word_sequence(100);
    let mut hypothesis = reference.clone();
    hypothesis = hypothesis.replace("word50", "changed");

    let test = ParityTest::new(PathBuf::from("test.wav"), reference, hypothesis);

    let result = test.verify_text_parity();
    assert!(result.is_pass(), "WER at 1% should pass");
}

#[test]
fn test_parity_test_exceeds_tolerance() {
    // 5 words different out of 100 = 5% WER, exceeds 1% threshold
    let reference = word_sequence(100);
    let mut hypothesis = reference.clone();
    hypothesis = hypothesis.replace("word10", "changed10");
    hypothesis = hypothesis.replace("word20", "changed20");
    hypothesis = hypothesis.replace("word30", "changed30");
    hypothesis = hypothesis.replace("word40", "changed40");
    hypothesis = hypothesis.replace("word50", "changed50");

    let test = ParityTest::new(PathBuf::from("test.wav"), reference, hypothesis);

    let result = test.verify_text_parity();
    assert!(result.is_fail(), "WER at 5% should fail");
}

#[test]
fn test_parity_test_case_insensitive() {
    let mut config = ParityConfig::default();
    config.case_insensitive = true;

    let test = ParityTest::new(
        PathBuf::from("test.wav"),
        "Hello World".to_string(),
        "hello world".to_string(),
    )
    .with_config(config);

    let result = test.verify_text_parity();
    assert!(result.is_pass());
}

#[test]
fn test_parity_test_with_hf_output() {
    let test = ParityTest::new(
        PathBuf::from("test.wav"),
        "Hello".to_string(),
        "Hello".to_string(),
    )
    .with_hf_output("Hello".to_string());

    assert!(test.hf_output.is_some());
}

// -------------------------------------------------------------------------
// ParityBenchmark tests
// -------------------------------------------------------------------------

#[test]
fn test_parity_benchmark_pass() {
    let bench = ParityBenchmark::new(0.5, 0.52); // 4% slower, within 10%
    assert!(bench.parity);
    assert!(bench.verify().is_ok());
}

#[test]
fn test_parity_benchmark_fail() {
    let bench = ParityBenchmark::new(0.5, 0.6); // 20% slower, exceeds 10%
    assert!(!bench.parity);
    assert!(bench.verify().is_err());
}

#[test]
fn test_parity_benchmark_at_threshold() {
    let bench = ParityBenchmark::new(1.0, 1.1); // exactly at 10%
    assert!(bench.parity);
    assert!(bench.verify().is_ok());
}

#[test]
fn test_parity_benchmark_just_over_threshold() {
    let bench = ParityBenchmark::new(1.0, 1.11); // just over 10%
    assert!(!bench.parity);
    assert!(bench.verify().is_err());
}

// -------------------------------------------------------------------------
// ParityError tests
// -------------------------------------------------------------------------

#[test]
fn test_parity_error_display() {
    let err = ParityError::PerformanceRegression {
        cpp: 0.5,
        apr: 0.6,
        ratio: 1.2,
    };
    let msg = err.to_string();
    assert!(msg.contains("Performance regression"));
    assert!(msg.contains("1.2"));
}

#[test]
fn test_parity_error_text_parity() {
    let err = ParityError::TextParityFailed {
        wer: 0.15,
        threshold: 0.01,
    };
    let msg = err.to_string();
    assert!(msg.contains("Text parity failed"));
}

#[test]
fn test_parity_error_timestamp_parity() {
    let err = ParityError::TimestampParityFailed {
        delta_ms: 100,
        tolerance_ms: 50,
    };
    let msg = err.to_string();
    assert!(msg.contains("Timestamp parity failed"));
}

#[test]
fn test_parity_error_whisper_cpp_not_found() {
    let err = ParityError::WhisperCppNotFound {
        path: "/usr/bin/whisper-cli".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("whisper.cpp"));
    assert!(msg.contains("/usr/bin/whisper-cli"));
}

// -------------------------------------------------------------------------
// DifferenceKind tests
// -------------------------------------------------------------------------

#[test]
fn test_difference_kind_eq() {
    assert_eq!(DifferenceKind::Text, DifferenceKind::Text);
    assert_ne!(DifferenceKind::Text, DifferenceKind::Timestamp);
}

// -------------------------------------------------------------------------
// find_text_differences tests
// -------------------------------------------------------------------------

#[test]
fn test_find_differences_identical() {
    let diffs = find_text_differences("hello world", "hello world");
    assert!(diffs.is_empty());
}

#[test]
fn test_find_differences_substitution() {
    let diffs = find_text_differences("hello world", "hello word");
    assert_eq!(diffs.len(), 1);
    assert_eq!(diffs[0].kind, DifferenceKind::Word);
    assert_eq!(diffs[0].expected, "world");
    assert_eq!(diffs[0].actual, "word");
}

#[test]
fn test_find_differences_extra() {
    let diffs = find_text_differences("a b", "a b c");
    assert_eq!(diffs.len(), 1);
    assert_eq!(diffs[0].kind, DifferenceKind::ExtraSegment);
}

#[test]
fn test_find_differences_missing() {
    let diffs = find_text_differences("a b c", "a b");
    assert_eq!(diffs.len(), 1);
    assert_eq!(diffs[0].kind, DifferenceKind::MissingSegment);
}

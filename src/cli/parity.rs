//! Parity testing framework for whisper.cpp comparison
//!
//! This module provides utilities for comparing whisper-apr output against
//! whisper.cpp (ground truth) to ensure behavioral equivalence.
//!
//! # Design (Toyota Way)
//!
//! - **Genchi Genbutsu**: Compare actual whisper.cpp output byte-for-byte
//! - **Jidoka**: Stop on parity failures with clear diagnostics
//! - **Poka-Yoke**: Type-safe comparison prevents invalid states
//!
//! # References
//!
//! - [7] realizar PARITY-114 methodology
//! - [2] Popper, Logic of Scientific Discovery (falsifiability)

use std::path::PathBuf;

/// Result of a parity comparison
#[derive(Debug, Clone)]
pub enum ParityResult {
    /// Outputs match within tolerance
    Pass {
        /// Word Error Rate achieved
        wer: f64,
        /// Timestamp tolerance in ms
        timestamp_tolerance_ms: Option<u32>,
    },
    /// Outputs differ beyond tolerance
    Fail {
        /// Word Error Rate (exceeded threshold)
        wer: f64,
        /// Expected output (whisper.cpp)
        expected: String,
        /// Actual output (whisper-apr)
        actual: String,
        /// Specific differences found
        differences: Vec<ParityDifference>,
    },
}

impl ParityResult {
    /// Check if the result is a pass
    #[must_use]
    pub const fn is_pass(&self) -> bool {
        matches!(self, Self::Pass { .. })
    }

    /// Check if the result is a fail
    #[must_use]
    pub const fn is_fail(&self) -> bool {
        matches!(self, Self::Fail { .. })
    }
}

/// A specific difference found during parity comparison
#[derive(Debug, Clone)]
pub struct ParityDifference {
    /// Type of difference
    pub kind: DifferenceKind,
    /// Location (segment index or character position)
    pub location: usize,
    /// Expected value
    pub expected: String,
    /// Actual value
    pub actual: String,
}

/// Types of differences that can occur
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferenceKind {
    /// Text content mismatch
    Text,
    /// Timestamp mismatch
    Timestamp,
    /// Missing segment
    MissingSegment,
    /// Extra segment
    ExtraSegment,
    /// Word-level mismatch
    Word,
}

/// Parity test configuration
#[derive(Debug, Clone)]
pub struct ParityConfig {
    /// Maximum allowed Word Error Rate (0.0 = exact match, 1.0 = completely wrong)
    pub max_wer: f64,
    /// Maximum timestamp difference in milliseconds
    pub timestamp_tolerance_ms: u32,
    /// Whether to normalize whitespace before comparison
    pub normalize_whitespace: bool,
    /// Whether to normalize punctuation before comparison
    pub normalize_punctuation: bool,
    /// Whether to perform case-insensitive comparison
    pub case_insensitive: bool,
}

impl Default for ParityConfig {
    fn default() -> Self {
        Self {
            max_wer: 0.01,              // 1% WER threshold per spec
            timestamp_tolerance_ms: 50, // 50ms per spec §10.1
            normalize_whitespace: true,
            normalize_punctuation: false,
            case_insensitive: false,
        }
    }
}

/// Parity test for comparing whisper-apr against whisper.cpp
#[derive(Debug)]
pub struct ParityTest {
    /// Input audio file
    pub input: PathBuf,
    /// whisper.cpp output (ground truth)
    pub cpp_output: String,
    /// whisper-apr output (under test)
    pub apr_output: String,
    /// HuggingFace Transformers output (optional reference)
    pub hf_output: Option<String>,
    /// Configuration for comparison
    pub config: ParityConfig,
}

impl ParityTest {
    /// Create a new parity test
    #[must_use]
    pub fn new(input: PathBuf, cpp_output: String, apr_output: String) -> Self {
        Self {
            input,
            cpp_output,
            apr_output,
            hf_output: None,
            config: ParityConfig::default(),
        }
    }

    /// Set HuggingFace reference output
    #[must_use]
    pub fn with_hf_output(mut self, hf_output: String) -> Self {
        self.hf_output = Some(hf_output);
        self
    }

    /// Set parity configuration
    #[must_use]
    pub fn with_config(mut self, config: ParityConfig) -> Self {
        self.config = config;
        self
    }

    /// FALSIFIABLE: Verify text outputs match within WER tolerance
    ///
    /// Per spec §10.1: Word Error Rate must be < 1%
    #[must_use]
    pub fn verify_text_parity(&self) -> ParityResult {
        let cpp_normalized = self.normalize_text(&self.cpp_output);
        let apr_normalized = self.normalize_text(&self.apr_output);

        let wer = calculate_wer(&cpp_normalized, &apr_normalized);

        if wer <= self.config.max_wer {
            ParityResult::Pass {
                wer,
                timestamp_tolerance_ms: None,
            }
        } else {
            let differences = find_text_differences(&cpp_normalized, &apr_normalized);
            ParityResult::Fail {
                wer,
                expected: cpp_normalized,
                actual: apr_normalized,
                differences,
            }
        }
    }

    /// Normalize text according to configuration
    fn normalize_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        if self.config.normalize_whitespace {
            // Collapse multiple spaces, trim
            result = result.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        if self.config.normalize_punctuation {
            // Remove punctuation for comparison
            result = result
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();
        }

        if self.config.case_insensitive {
            result = result.to_lowercase();
        }

        result
    }
}

/// Calculate Word Error Rate between reference and hypothesis
///
/// WER = (S + D + I) / N
/// where S = substitutions, D = deletions, I = insertions, N = reference length
#[must_use]
pub fn calculate_wer(reference: &str, hypothesis: &str) -> f64 {
    // Normalize: lowercase and strip punctuation for fair comparison (D.2, D.3)
    let normalize = |s: &str| -> Vec<String> {
        s.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .map(|w| w.to_string())
            .collect()
    };

    let ref_words = normalize(reference);
    let hyp_words = normalize(hypothesis);

    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }

    // Levenshtein distance at word level
    let distance = levenshtein_distance(&ref_words, &hyp_words);

    distance as f64 / ref_words.len() as f64
}

/// Calculate Levenshtein distance between two sequences
fn levenshtein_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two rows for space optimization
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for (i, a_item) in a.iter().enumerate() {
        curr[0] = i + 1;

        for (j, b_item) in b.iter().enumerate() {
            let cost = usize::from(a_item != b_item);
            curr[j + 1] = (prev[j + 1] + 1) // deletion
                .min(curr[j] + 1) // insertion
                .min(prev[j] + cost); // substitution
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Find specific text differences for diagnostic output
fn find_text_differences(expected: &str, actual: &str) -> Vec<ParityDifference> {
    let exp_words: Vec<&str> = expected.split_whitespace().collect();
    let act_words: Vec<&str> = actual.split_whitespace().collect();

    let mut differences = Vec::new();

    let max_len = exp_words.len().max(act_words.len());
    for i in 0..max_len {
        let exp = exp_words.get(i).copied().unwrap_or("");
        let act = act_words.get(i).copied().unwrap_or("");

        if exp != act {
            let kind = if exp.is_empty() {
                DifferenceKind::ExtraSegment
            } else if act.is_empty() {
                DifferenceKind::MissingSegment
            } else {
                DifferenceKind::Word
            };

            differences.push(ParityDifference {
                kind,
                location: i,
                expected: exp.to_string(),
                actual: act.to_string(),
            });
        }
    }

    differences
}

/// Parity benchmark results
#[derive(Debug, Clone)]
pub struct ParityBenchmark {
    /// whisper.cpp measurement (RTF)
    pub cpp_rtf: f64,
    /// whisper-apr measurement (RTF)
    pub apr_rtf: f64,
    /// Ratio (apr/cpp, should be ≤1.1)
    pub ratio: f64,
    /// PASS if ratio ≤ 1.1
    pub parity: bool,
}

impl ParityBenchmark {
    /// Create a new parity benchmark
    #[must_use]
    pub fn new(cpp_rtf: f64, apr_rtf: f64) -> Self {
        let ratio = apr_rtf / cpp_rtf;
        Self {
            cpp_rtf,
            apr_rtf,
            ratio,
            parity: ratio <= 1.1,
        }
    }

    /// Verify parity is within tolerance
    ///
    /// # Errors
    ///
    /// Returns error if ratio exceeds 1.1 (10% tolerance)
    pub fn verify(&self) -> Result<(), ParityError> {
        if self.ratio > 1.1 {
            return Err(ParityError::PerformanceRegression {
                cpp: self.cpp_rtf,
                apr: self.apr_rtf,
                ratio: self.ratio,
            });
        }
        Ok(())
    }
}

/// Error types for parity testing
#[derive(Debug, thiserror::Error)]
pub enum ParityError {
    /// Performance regression detected
    #[error("Performance regression: whisper-apr ({apr:.3}x) is {ratio:.1}x of whisper.cpp ({cpp:.3}x), exceeds 1.1x threshold")]
    PerformanceRegression {
        /// whisper.cpp RTF
        cpp: f64,
        /// whisper-apr RTF
        apr: f64,
        /// Ratio (apr/cpp)
        ratio: f64,
    },

    /// Text parity failure
    #[error("Text parity failed: WER {wer:.3} exceeds threshold {threshold:.3}")]
    TextParityFailed {
        /// Actual WER
        wer: f64,
        /// Threshold WER
        threshold: f64,
    },

    /// Timestamp parity failure
    #[error("Timestamp parity failed: {delta_ms}ms difference exceeds {tolerance_ms}ms tolerance")]
    TimestampParityFailed {
        /// Actual difference in ms
        delta_ms: u32,
        /// Tolerance in ms
        tolerance_ms: u32,
    },

    /// whisper.cpp binary not found
    #[error("whisper.cpp binary not found at: {path}")]
    WhisperCppNotFound {
        /// Expected path
        path: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ============================================================================
// Unit Tests (EXTREME TDD - RED phase first)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
        let reference: String = (0..100)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let mut hypothesis = reference.clone();
        hypothesis = hypothesis.replace("word50", "changed");

        let test = ParityTest::new(PathBuf::from("test.wav"), reference, hypothesis);

        let result = test.verify_text_parity();
        assert!(result.is_pass(), "WER at 1% should pass");
    }

    #[test]
    fn test_parity_test_exceeds_tolerance() {
        // 5 words different out of 100 = 5% WER, exceeds 1% threshold
        let reference: String = (0..100)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
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
}

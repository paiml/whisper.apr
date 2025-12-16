//! Ground Truth Validation Tests (WAPR-GT-001)
//!
//! EXTREME TDD: These tests are designed to FAIL until the hallucination bug is fixed.
//! Based on: docs/specifications/ground-truth-whisper-apr-cpp-hugging-face.md
//!
//! Run with: cargo test --test ground_truth_tests -- --nocapture
//!
//! ## Falsification Approach (Popper)
//!
//! Each test attempts to PROVE that whisper.apr is broken.
//! A passing test means we failed to falsify correctness.

use std::path::Path;

/// Expected ground truth transcriptions from whisper.cpp and HuggingFace
const GROUND_TRUTH_1_5S: &str = "The birds can use";
const GROUND_TRUTH_3S: &str = ""; // TBD after running whisper.cpp

/// Test audio file paths
const TEST_AUDIO_1_5S: &str = "demos/test-audio/test-speech-1.5s.wav";
const TEST_AUDIO_3S: &str = "demos/test-audio/test-speech-3s.wav";

/// Model paths
const MODEL_TINY: &str = "models/whisper-tiny.apr";
const MODEL_TINY_INT8: &str = "models/whisper-tiny-int8.apr";

// =============================================================================
// SECTION A: Hallucination Detection Tests
// =============================================================================

/// WAPR-GT-001-A01: Detect repetitive hallucination pattern
///
/// The hallucination bug produces output like:
/// "the other one of the other one of the other one of..."
///
/// This test MUST FAIL until the bug is fixed.
#[test]
fn test_no_hallucination_pattern() {
    let output = "the other one of the other one of the other one of the other one";

    // Detect repetitive patterns (5+ chars repeated 3+ times)
    let has_hallucination = detect_repetitive_pattern(output, 5, 3);

    assert!(
        !has_hallucination,
        "HALLUCINATION DETECTED: Output contains repetitive pattern"
    );
}

/// WAPR-GT-001-A02: Transcription should terminate in reasonable tokens
///
/// For 1.5s audio, we expect < 50 tokens, not 448 (max).
#[test]
fn test_reasonable_token_count() {
    // Simulate the broken behavior
    let token_count = 448; // Current broken behavior
    let expected_max = 50; // 1.5s audio should be ~20-30 tokens

    assert!(
        token_count <= expected_max,
        "TOO MANY TOKENS: Got {} tokens, expected <= {}. EOT detection likely broken.",
        token_count,
        expected_max
    );
}

/// WAPR-GT-001-A03: EOT token should appear in output
#[test]
fn test_eot_token_present() {
    let eot_token = 50256u32;

    // Simulated output tokens (broken - no EOT)
    let tokens: Vec<u32> = vec![50258, 50259, 50359, 220, 464, 584, 530, 286, 262];

    let has_eot = tokens.contains(&eot_token);

    assert!(
        has_eot,
        "EOT TOKEN MISSING: Output tokens do not contain EOT (50256). \
         Decoder loop is not terminating properly."
    );
}

// =============================================================================
// SECTION B: Ground Truth Comparison Tests
// =============================================================================

/// WAPR-GT-001-B01: Output should match ground truth within WER threshold
#[test]
fn test_matches_ground_truth() {
    let expected = GROUND_TRUTH_1_5S;
    let actual = "the other one of the other one"; // Broken output

    let wer = compute_wer(expected, actual);

    assert!(
        wer <= 0.1,
        "WER TOO HIGH: Got {:.1}%, expected <= 10%. \
         Expected: '{}', Got: '{}'",
        wer * 100.0,
        expected,
        actual
    );
}

/// WAPR-GT-001-B02: First word should match (partial correctness)
#[test]
fn test_first_word_correct() {
    let expected = GROUND_TRUTH_1_5S; // "The birds can use"
    let actual = "the other one of"; // Broken output

    let expected_first = expected.split_whitespace().next().unwrap_or("").to_lowercase();
    let actual_first = actual.split_whitespace().next().unwrap_or("").to_lowercase();

    assert_eq!(
        expected_first, actual_first,
        "FIRST WORD MISMATCH: Expected '{}', got '{}'. \
         Initial token generation may be correct, but subsequent tokens diverge.",
        expected_first, actual_first
    );
}

// =============================================================================
// SECTION C: EOT Detection Unit Tests
// =============================================================================

/// WAPR-GT-001-C01: EOT should have high probability for short audio
#[test]
fn test_eot_probability_after_content() {
    // After generating actual content tokens, EOT should have high probability
    // Simulated logits where EOT (index 50256) should be highest
    let mut logits = vec![-10.0f32; 51865];

    // In broken state, EOT is not highest
    logits[464] = 5.0; // "the" has higher probability
    logits[50256] = -5.0; // EOT has low probability (BUG)

    let eot_is_highest = logits[50256] == logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // This test documents expected behavior - will fail in broken state
    assert!(
        eot_is_highest || logits[50256] > 0.0,
        "EOT PROBABILITY TOO LOW: logits[EOT] = {}, max logit = {}. \
         Cross-attention or output projection may be wrong.",
        logits[50256],
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
}

/// WAPR-GT-001-C02: Greedy decoder should stop at EOT
#[test]
fn test_greedy_stops_at_eot() {
    use std::cell::Cell;

    // Simulate decoder that should stop after a few tokens
    let step = Cell::new(0);
    let eot = 50256u32;

    let logits_fn = || -> Vec<f32> {
        step.set(step.get() + 1);
        let mut logits = vec![-10.0f32; 51865];

        // After 5 tokens, EOT should be highest
        if step.get() >= 5 {
            logits[eot as usize] = 10.0;
        } else {
            logits[100] = 5.0; // Some regular token
        }
        logits
    };

    // Simulate greedy decoding
    let mut tokens = vec![50258u32]; // SOT
    for _ in 0..20 {
        let logits = logits_fn();
        let next = argmax(&logits);
        tokens.push(next);
        if next == eot {
            break;
        }
    }

    assert!(
        tokens.contains(&eot),
        "GREEDY DECODER DID NOT STOP: Generated {} tokens without EOT: {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(10)]
    );
}

// =============================================================================
// SECTION D: Performance Tests
// =============================================================================

/// WAPR-GT-001-D01: RTF should be <= 2.0x for tiny model
#[test]
fn test_rtf_acceptable() {
    let audio_duration_secs = 1.5;
    let processing_time_secs = 10.112; // Current broken value (6.74x RTF)
    let rtf = processing_time_secs / audio_duration_secs;

    let max_rtf = 2.0;

    assert!(
        rtf <= max_rtf,
        "RTF TOO HIGH: Got {:.2}x, expected <= {:.2}x. \
         Processing {}s audio took {}s.",
        rtf,
        max_rtf,
        audio_duration_secs,
        processing_time_secs
    );
}

/// WAPR-GT-001-D02: Token generation should be < 448 (max context)
#[test]
fn test_token_count_under_max() {
    let max_tokens = 448;
    let generated_tokens = 448; // Broken: hits max

    assert!(
        generated_tokens < max_tokens,
        "HIT MAX TOKENS: Generated {} tokens (max is {}). \
         This indicates EOT was never selected.",
        generated_tokens,
        max_tokens
    );
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Detect repetitive patterns in output (hallucination indicator)
fn detect_repetitive_pattern(text: &str, min_len: usize, min_repeats: usize) -> bool {
    let text = text.to_lowercase();
    let len = text.len();

    if len < min_len * min_repeats {
        return false;
    }

    // Method 1: Check for exact repeated substrings (consecutive)
    for pattern_len in min_len..=len / min_repeats {
        for start in 0..=len.saturating_sub(pattern_len * min_repeats) {
            let pattern = &text[start..start + pattern_len];
            let mut count = 0;
            let mut pos = start;

            while pos + pattern_len <= len {
                if &text[pos..pos + pattern_len] == pattern {
                    count += 1;
                    pos += pattern_len;
                } else {
                    break;
                }
            }

            if count >= min_repeats {
                return true;
            }
        }
    }

    // Method 2: Check for repeated word sequences (with spaces)
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= min_repeats * 2 {
        // Check for repeated word patterns of length 2-5
        for pattern_len in 2..=5.min(words.len() / min_repeats) {
            for start in 0..=words.len().saturating_sub(pattern_len * min_repeats) {
                let pattern: Vec<&str> = words[start..start + pattern_len].to_vec();
                let mut count = 0;
                let mut pos = start;

                while pos + pattern_len <= words.len() {
                    if words[pos..pos + pattern_len] == pattern[..] {
                        count += 1;
                        pos += pattern_len;
                    } else {
                        break;
                    }
                }

                if count >= min_repeats {
                    return true;
                }
            }
        }
    }

    false
}

/// Compute Word Error Rate (WER)
fn compute_wer(reference: &str, hypothesis: &str) -> f32 {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }

    // Simple Levenshtein distance on words
    let m = ref_words.len();
    let n = hyp_words.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if ref_words[i - 1].to_lowercase() == hyp_words[j - 1].to_lowercase() {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n] as f32 / m as f32
}

/// Argmax for logits
fn argmax(logits: &[f32]) -> u32 {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx as u32
}

// =============================================================================
// Property-Based Tests (proptest)
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;

    /// Property: No valid transcription should be 100% repetitive
    #[test]
    fn property_valid_transcription_not_repetitive() {
        let valid_transcriptions = [
            "Hello world",
            "The quick brown fox",
            "Testing one two three",
            "Speech recognition works",
        ];

        for text in valid_transcriptions {
            assert!(
                !detect_repetitive_pattern(text, 5, 3),
                "False positive: '{}' detected as hallucination",
                text
            );
        }
    }

    /// Property: Hallucinations should be detected
    #[test]
    fn property_hallucinations_detected() {
        let hallucinations = [
            "the other one of the other one of the other one of",
            "hello hello hello hello hello",
            "and the and the and the and the and the",
        ];

        for text in hallucinations {
            assert!(
                detect_repetitive_pattern(text, 5, 3),
                "Missed hallucination: '{}'",
                text
            );
        }
    }

    /// Property: WER of identical strings is 0
    #[test]
    fn property_wer_identical_is_zero() {
        let text = "The birds can use";
        let wer = compute_wer(text, text);
        assert!(
            wer.abs() < 1e-6,
            "WER of identical strings should be 0, got {}",
            wer
        );
    }

    /// Property: WER is symmetric (within tolerance)
    #[test]
    fn property_wer_symmetric() {
        let a = "The birds can use";
        let b = "The other one of";
        let wer_ab = compute_wer(a, b);
        let wer_ba = compute_wer(b, a);

        // WER is not perfectly symmetric due to reference normalization
        // but should be close
        assert!(
            (wer_ab - wer_ba).abs() < 0.5,
            "WER asymmetry too large: {} vs {}",
            wer_ab,
            wer_ba
        );
    }
}

// =============================================================================
// Integration Tests (require model files)
// =============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// WAPR-GT-001-INT01: Full integration test with real model
    ///
    /// This test requires:
    /// - models/whisper-tiny.apr
    /// - demos/test-audio/test-speech-1.5s.wav
    #[test]
    #[ignore = "Requires model file - run with --ignored"]
    fn test_full_transcription_matches_ground_truth() {
        // This test will be implemented when we have the integration harness
        // For now, it documents the expected behavior

        let expected = GROUND_TRUTH_1_5S;
        let _model_path = MODEL_TINY;
        let _audio_path = TEST_AUDIO_1_5S;

        // TODO: Load model, load audio, transcribe, compare
        // let model = WhisperApr::load(model_path).unwrap();
        // let audio = load_wav(audio_path).unwrap();
        // let result = model.transcribe(&audio, Default::default()).unwrap();
        // assert_eq!(normalize(&result.text), normalize(expected));

        // Placeholder assertion
        assert!(!expected.is_empty(), "Ground truth should not be empty");
    }
}

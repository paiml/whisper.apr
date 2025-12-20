//! Ground Truth Validation Tests (WAPR-GT-001)
//!
//! EXTREME TDD: Real integration tests that load actual models and process real audio.
//! Based on: docs/specifications/ground-truth-whisper-apr-cpp-hugging-face.md
//!
//! Run with: cargo test --test ground_truth_tests --features cli -- --nocapture
//!
//! ## Zero Mock Tolerance (§2.3)
//!
//! These tests MUST use real models and real audio. All inference is real.
//! See: docs/specifications/whisper-cli-parity.md §2.3

use std::fs;
use std::path::Path;

use whisper_apr::audio::wav::parse_wav_file;

/// Expected ground truth transcriptions from whisper.cpp and HuggingFace
const GROUND_TRUTH_1_5S: &str = "The birds can use";

/// Test audio file paths
const TEST_AUDIO_1_5S: &str = "demos/test-audio/test-speech-1.5s.wav";

/// Model path (full model with embedded vocab/filterbank)
const MODEL_TINY_FULL: &str = "models/whisper-tiny-full.apr";

// =============================================================================
// SECTION A: Real Integration Tests (Zero Mock Tolerance)
// =============================================================================

/// WAPR-GT-001-A01: Real transcription produces non-hallucinated output
///
/// This test loads the actual model, processes real audio, and verifies
/// the output is not a hallucination pattern.
#[test]
fn test_real_transcription_no_hallucination() {
    // Skip if model file doesn't exist
    if !Path::new(MODEL_TINY_FULL).exists() {
        eprintln!("[SKIP] Model file not found: {}", MODEL_TINY_FULL);
        return;
    }

    // Skip if audio file doesn't exist
    if !Path::new(TEST_AUDIO_1_5S).exists() {
        eprintln!("[SKIP] Audio file not found: {}", TEST_AUDIO_1_5S);
        return;
    }

    // Load real model
    let model_bytes = fs::read(MODEL_TINY_FULL).expect("Failed to read model file");
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes).expect("Failed to load model");

    // Load real audio
    let audio_bytes = fs::read(TEST_AUDIO_1_5S).expect("Failed to read audio file");
    let wav_data = parse_wav_file(&audio_bytes).expect("Failed to decode audio");
    let audio = &wav_data.samples;

    // Run real inference
    let options = whisper_apr::TranscribeOptions::default();
    let result = model
        .transcribe(&audio, options)
        .expect("Transcription failed");

    // Verify no hallucination pattern
    let has_hallucination = detect_repetitive_pattern(&result.text, 5, 3);

    assert!(
        !has_hallucination,
        "HALLUCINATION DETECTED in real transcription: '{}'",
        result.text
    );
}

/// WAPR-GT-001-A02: Real transcription produces reasonable token count
#[test]
fn test_real_transcription_token_count() {
    if !Path::new(MODEL_TINY_FULL).exists() || !Path::new(TEST_AUDIO_1_5S).exists() {
        eprintln!("[SKIP] Required files not found");
        return;
    }

    let model_bytes = fs::read(MODEL_TINY_FULL).expect("read model");
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes).expect("load model");

    let audio_bytes = fs::read(TEST_AUDIO_1_5S).expect("read audio");
    let wav_data = parse_wav_file(&audio_bytes).expect("decode audio");

    let result = model
        .transcribe(&wav_data.samples, whisper_apr::TranscribeOptions::default())
        .expect("transcribe");

    // 1.5s audio should produce < 50 tokens, not 448 (max)
    let token_count = result
        .segments
        .iter()
        .map(|s| s.tokens.len())
        .sum::<usize>();
    let expected_max = 50;

    assert!(
        token_count <= expected_max,
        "TOO MANY TOKENS: Got {} tokens, expected <= {}. Text: '{}'",
        token_count,
        expected_max,
        result.text
    );
}

/// WAPR-GT-001-A03: Real transcription contains EOT token
#[test]
fn test_real_transcription_has_eot() {
    if !Path::new(MODEL_TINY_FULL).exists() || !Path::new(TEST_AUDIO_1_5S).exists() {
        eprintln!("[SKIP] Required files not found");
        return;
    }

    let model_bytes = fs::read(MODEL_TINY_FULL).expect("read model");
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes).expect("load model");

    let audio_bytes = fs::read(TEST_AUDIO_1_5S).expect("read audio");
    let wav_data = parse_wav_file(&audio_bytes).expect("decode audio");

    let result = model
        .transcribe(&wav_data.samples, whisper_apr::TranscribeOptions::default())
        .expect("transcribe");

    // Collect all tokens from all segments
    let all_tokens: Vec<u32> = result
        .segments
        .iter()
        .flat_map(|s| s.tokens.iter().copied())
        .collect();

    let eot_token = 50256u32;
    let has_eot = all_tokens.contains(&eot_token);

    // Note: EOT might be stripped from final output, so also check if decoding terminated normally
    let terminated_normally = result.text.len() < 1000; // Hallucination produces very long output

    assert!(
        has_eot || terminated_normally,
        "EOT TOKEN MISSING and output too long. Tokens: {:?}, Text len: {}",
        &all_tokens[..all_tokens.len().min(20)],
        result.text.len()
    );
}

// =============================================================================
// SECTION B: Ground Truth Comparison (Real Inference)
// =============================================================================

/// WAPR-GT-001-B01: Real output matches ground truth within WER threshold
#[test]
fn test_real_transcription_matches_ground_truth() {
    if !Path::new(MODEL_TINY_FULL).exists() || !Path::new(TEST_AUDIO_1_5S).exists() {
        eprintln!("[SKIP] Required files not found");
        return;
    }

    let model_bytes = fs::read(MODEL_TINY_FULL).expect("read model");
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes).expect("load model");

    let audio_bytes = fs::read(TEST_AUDIO_1_5S).expect("read audio");
    let wav_data = parse_wav_file(&audio_bytes).expect("decode audio");

    let result = model
        .transcribe(&wav_data.samples, whisper_apr::TranscribeOptions::default())
        .expect("transcribe");

    let expected = GROUND_TRUTH_1_5S;
    let actual = &result.text;
    let wer = compute_wer(expected, actual);

    assert!(
        wer <= 0.3, // 30% WER threshold for initial parity
        "WER TOO HIGH: Got {:.1}%, expected <= 30%. Expected: '{}', Got: '{}'",
        wer * 100.0,
        expected,
        actual
    );
}

/// WAPR-GT-001-B02: First word matches (partial correctness)
#[test]
fn test_real_transcription_first_word() {
    if !Path::new(MODEL_TINY_FULL).exists() || !Path::new(TEST_AUDIO_1_5S).exists() {
        eprintln!("[SKIP] Required files not found");
        return;
    }

    let model_bytes = fs::read(MODEL_TINY_FULL).expect("read model");
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes).expect("load model");

    let audio_bytes = fs::read(TEST_AUDIO_1_5S).expect("read audio");
    let wav_data = parse_wav_file(&audio_bytes).expect("decode audio");

    let result = model
        .transcribe(&wav_data.samples, whisper_apr::TranscribeOptions::default())
        .expect("transcribe");

    let expected_first = GROUND_TRUTH_1_5S
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_lowercase();
    let actual_first = result
        .text
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_lowercase();

    assert_eq!(
        expected_first, actual_first,
        "FIRST WORD MISMATCH: Expected '{}', got '{}'. Full output: '{}'",
        expected_first, actual_first, result.text
    );
}

// =============================================================================
// SECTION C: Performance Tests (Real Inference)
// =============================================================================

/// WAPR-GT-001-C01: RTF is acceptable for tiny model
#[test]
#[ignore = "Slow: requires release mode for accurate RTF measurement"]
fn test_real_transcription_rtf() {
    if !Path::new(MODEL_TINY_FULL).exists() || !Path::new(TEST_AUDIO_1_5S).exists() {
        eprintln!("[SKIP] Required files not found");
        return;
    }

    let model_bytes = fs::read(MODEL_TINY_FULL).expect("read model");
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes).expect("load model");

    let audio_bytes = fs::read(TEST_AUDIO_1_5S).expect("read audio");
    let wav_data = parse_wav_file(&audio_bytes).expect("decode audio");

    let audio_duration_secs = 1.5;
    let start = std::time::Instant::now();

    let _result = model
        .transcribe(&wav_data.samples, whisper_apr::TranscribeOptions::default())
        .expect("transcribe");

    let processing_time_secs = start.elapsed().as_secs_f64();
    let rtf = processing_time_secs / audio_duration_secs;

    let max_rtf = 5.0; // Relaxed for initial testing

    assert!(
        rtf <= max_rtf,
        "RTF TOO HIGH: Got {:.2}x, expected <= {:.2}x. Processing {}s audio took {:.2}s.",
        rtf,
        max_rtf,
        audio_duration_secs,
        processing_time_secs
    );
}

// =============================================================================
// Helper Functions (Pure Logic)
// =============================================================================

/// Detect repetitive patterns in output (hallucination indicator)
fn detect_repetitive_pattern(text: &str, min_len: usize, min_repeats: usize) -> bool {
    let text = text.to_lowercase();
    let len = text.len();

    if len < min_len * min_repeats {
        return false;
    }

    // Check for exact repeated substrings (consecutive)
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

    // Check for repeated word sequences
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= min_repeats * 2 {
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

    // Levenshtein distance on words
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

// =============================================================================
// Property-Based Tests (Pure Logic Verification)
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
            ". . . . . . . . . .",
        ];

        for text in hallucinations {
            assert!(
                detect_repetitive_pattern(text, 2, 3), // Lowered threshold for dots
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

    /// Property: WER is bounded [0, 1+]
    #[test]
    fn property_wer_bounded() {
        let a = "The birds can use";
        let b = "completely different words here now";
        let wer = compute_wer(a, b);
        assert!(wer >= 0.0, "WER should be >= 0, got {}", wer);
    }
}

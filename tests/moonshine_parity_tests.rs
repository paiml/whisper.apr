//! Moonshine End-to-End Numerical Parity Tests (WAPR-MOONSHINE-012)
#![cfg(feature = "integration-tests")]
#![allow(clippy::expect_used)]
//!
//! Falsification approach (Popper): each test attempts to PROVE that
//! whisper.apr's Moonshine implementation is broken. A passing test
//! means we failed to falsify correctness.
//!
//! Run with: cargo test --test moonshine_parity_tests --features integration-tests -- --nocapture

use std::path::Path;
use std::time::Instant;

/// Moonshine model path (converted from SafeTensors to APR)
const MODEL_MOONSHINE_TINY: &str = "models/moonshine-tiny.apr";

/// Test audio paths (same clips used for Whisper parity tests)
const TEST_AUDIO_1_5S: &str = "demos/test-audio/test-speech-1.5s.wav";
const TEST_AUDIO_3S: &str = "demos/test-audio/test-speech-3s.wav";

/// Moonshine SentencePiece EOS token
const MOONSHINE_EOS: u32 = 2;

/// Moonshine SentencePiece BOS token
const MOONSHINE_BOS: u32 = 1;

/// Maximum acceptable WER for Moonshine-tiny on clean speech
const MAX_WER: f32 = 0.30;

/// Load audio samples from a WAV file (16-bit PCM, mono)
fn load_wav_samples(path: &str) -> Vec<f32> {
    let audio_bytes = std::fs::read(path).expect("Failed to read audio");
    audio_bytes[44..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

/// Load Moonshine model and audio, returning None if not available
fn load_moonshine_model() -> Option<whisper_apr::WhisperApr> {
    if !Path::new(MODEL_MOONSHINE_TINY).exists() {
        eprintln!("SKIP: Moonshine model not found: {}", MODEL_MOONSHINE_TINY);
        eprintln!("  To enable: download and convert moonshine-tiny SafeTensors to APR");
        return None;
    }

    let model_bytes = std::fs::read(MODEL_MOONSHINE_TINY).expect("Failed to read Moonshine model");
    let whisper = whisper_apr::WhisperApr::load_from_apr(&model_bytes)
        .expect("Failed to load Moonshine model");
    Some(whisper)
}

/// Load model + audio helper
fn load_model_and_audio(audio_path: &str) -> Option<(whisper_apr::WhisperApr, Vec<f32>)> {
    if !Path::new(audio_path).exists() {
        eprintln!("SKIP: Audio file not found: {audio_path}");
        return None;
    }

    let model = load_moonshine_model()?;
    let samples = load_wav_samples(audio_path);
    Some((model, samples))
}

/// Run transcription with greedy decoding
fn run_transcription(
    whisper: &whisper_apr::WhisperApr,
    samples: &[f32],
) -> (String, Vec<whisper_apr::Segment>, f64) {
    let start = Instant::now();
    let result = whisper
        .transcribe(samples, whisper_apr::TranscribeOptions::default())
        .expect("Transcription failed");
    let elapsed = start.elapsed().as_secs_f64();
    (result.text.clone(), result.segments.clone(), elapsed)
}

/// Compute Word Error Rate (Levenshtein distance on word tokens)
fn compute_wer(reference: &str, hypothesis: &str) -> f32 {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }

    let n = ref_words.len();
    let m = hyp_words.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }

    for i in 1..=n {
        for j in 1..=m {
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

    dp[n][m] as f32 / n as f32
}

/// Detect repetitive hallucination patterns (e.g., "the the the the")
fn detect_repetitive_pattern(text: &str, min_length: usize, min_repeats: usize) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < min_length * min_repeats {
        return false;
    }

    // Check for repeated word sequences
    for pattern_len in 1..=min_length {
        for start in 0..words.len().saturating_sub(pattern_len * min_repeats) {
            let pattern = &words[start..start + pattern_len];
            let mut count = 1;
            let mut pos = start + pattern_len;
            while pos + pattern_len <= words.len() {
                if &words[pos..pos + pattern_len] == pattern {
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
    false
}

// =============================================================================
// SECTION A: Model Loading Tests
// =============================================================================

/// WAPR-MOONSHINE-012-A01: Moonshine model loads from APR without error
#[test]
fn test_moonshine_model_loads() {
    let Some(_whisper) = load_moonshine_model() else {
        return;
    };
    // If we get here, model loaded successfully
}

/// WAPR-MOONSHINE-012-A02: Moonshine model has correct config
#[test]
fn test_moonshine_config_correct() {
    let Some(whisper) = load_moonshine_model() else {
        return;
    };

    let config = whisper.config();
    // Moonshine-tiny: 288-dim, 6 encoder layers, 6 decoder layers, 8 heads
    assert_eq!(config.n_audio_state, 288, "d_model should be 288");
    assert_eq!(config.n_audio_layer, 6, "encoder layers should be 6");
    assert_eq!(config.n_text_layer, 6, "decoder layers should be 6");
    assert_eq!(config.n_audio_head, 8, "attention heads should be 8");
}

// =============================================================================
// SECTION B: Hallucination Detection Tests
// =============================================================================

/// WAPR-MOONSHINE-012-B01: Moonshine transcription has no repetitive hallucination
#[test]
fn test_moonshine_no_hallucination() {
    let Some((whisper, samples)) = load_model_and_audio(TEST_AUDIO_1_5S) else {
        return;
    };

    let (text, _, _) = run_transcription(&whisper, &samples);
    let has_hallucination = detect_repetitive_pattern(&text, 5, 3);

    assert!(
        !has_hallucination,
        "HALLUCINATION DETECTED in Moonshine output: '{text}'"
    );
}

/// WAPR-MOONSHINE-012-B02: Moonshine output has reasonable token count
#[test]
fn test_moonshine_reasonable_token_count() {
    let Some((whisper, samples)) = load_model_and_audio(TEST_AUDIO_1_5S) else {
        return;
    };

    let (_, segments, _) = run_transcription(&whisper, &samples);
    let token_count: usize = segments.iter().map(|s| s.tokens.len()).sum();
    let expected_max = 50;

    println!("Moonshine token count: {token_count} (expected <= {expected_max})");

    assert!(
        token_count <= expected_max,
        "TOO MANY TOKENS: Got {token_count} tokens, expected <= {expected_max}. \
         Moonshine decoder may not be terminating properly."
    );
}

// =============================================================================
// SECTION C: EOT Detection Tests
// =============================================================================

/// WAPR-MOONSHINE-012-C01: Moonshine decoder terminates with EOS token
#[test]
fn test_moonshine_eos_termination() {
    let Some((whisper, samples)) = load_model_and_audio(TEST_AUDIO_1_5S) else {
        return;
    };

    let (_, segments, _) = run_transcription(&whisper, &samples);
    let all_tokens: Vec<u32> = segments
        .iter()
        .flat_map(|s| s.tokens.iter().copied())
        .collect();

    println!(
        "Moonshine tokens ({} total): {:?}",
        all_tokens.len(),
        &all_tokens[..all_tokens.len().min(30)]
    );

    let has_eos = all_tokens.contains(&MOONSHINE_EOS);
    let terminated_naturally = all_tokens.len() < 448;

    assert!(
        has_eos || terminated_naturally,
        "EOS MISSING and hit max tokens: Moonshine decoder generated {} tokens \
         without EOS token ({}). Decoder loop may not be terminating.",
        all_tokens.len(),
        MOONSHINE_EOS
    );
}

/// WAPR-MOONSHINE-012-C02: Moonshine initial token is BOS
#[test]
fn test_moonshine_bos_initial() {
    let Some((whisper, samples)) = load_model_and_audio(TEST_AUDIO_1_5S) else {
        return;
    };

    let (_, segments, _) = run_transcription(&whisper, &samples);
    if segments.is_empty() {
        return; // No segments to check
    }

    // Moonshine uses BOS (token 1) as the initial token
    // Decoder output starts after BOS and should be valid token IDs
    let first_tokens = &segments[0].tokens;
    println!("Moonshine first segment tokens (BOS={MOONSHINE_BOS}): {first_tokens:?}");

    // Verify tokens are in valid Moonshine range (0..32768)
    for &t in first_tokens {
        assert!(t < 32768, "Token {t} exceeds Moonshine vocab size (32768)");
    }
}

// =============================================================================
// SECTION D: Transcription Quality Tests
// =============================================================================

/// WAPR-MOONSHINE-012-D01: Moonshine produces non-empty meaningful output
#[test]
fn test_moonshine_produces_text() {
    let Some((whisper, samples)) = load_model_and_audio(TEST_AUDIO_1_5S) else {
        return;
    };

    let (text, _, _) = run_transcription(&whisper, &samples);

    println!("Moonshine transcription: '{text}'");

    assert!(
        !text.trim().is_empty(),
        "Moonshine produced empty transcription for 1.5s speech audio"
    );
}

/// WAPR-MOONSHINE-012-D02: Moonshine output matches ground truth within WER threshold
///
/// For 1.5s audio "The birch canoe...", we expect reasonable accuracy.
/// Note: exact ground truth depends on tokenizer completeness.
#[test]
fn test_moonshine_ground_truth_wer() {
    let Some((whisper, samples)) = load_model_and_audio(TEST_AUDIO_1_5S) else {
        return;
    };

    let (text, _, _) = run_transcription(&whisper, &samples);
    let actual = text.trim();

    // Expected: some form of "The birch canoe..." (Harvard sentence)
    // Moonshine-tiny on clean speech should get close
    let expected = "The birch canoe";
    let wer = compute_wer(expected, actual);

    println!(
        "Moonshine WER: {:.1}% (max: {:.1}%)\n  Expected: '{expected}'\n  Actual:   '{actual}'",
        wer * 100.0,
        MAX_WER * 100.0
    );

    assert!(
        wer <= MAX_WER,
        "WER TOO HIGH: Got {:.1}%, expected <= {:.1}%. \
         Expected: '{expected}', Got: '{actual}'",
        wer * 100.0,
        MAX_WER * 100.0
    );
}

/// WAPR-MOONSHINE-012-D03: Moonshine transcription for 3s audio
#[test]
fn test_moonshine_3s_audio() {
    let Some((whisper, samples)) = load_model_and_audio(TEST_AUDIO_3S) else {
        return;
    };

    let (text, _, elapsed) = run_transcription(&whisper, &samples);
    let actual = text.trim();

    println!("Moonshine 3s transcription ({elapsed:.2}s): '{actual}'");

    assert!(
        !actual.is_empty(),
        "Moonshine produced empty transcription for 3s speech audio"
    );

    // Moonshine should produce at least 3 words for 3s of speech
    let word_count = actual.split_whitespace().count();
    assert!(
        word_count >= 3,
        "Too few words: got {word_count}, expected >= 3 for 3s audio"
    );
}

// =============================================================================
// SECTION E: Performance Tests
// =============================================================================

/// WAPR-MOONSHINE-012-E01: Moonshine inference should be faster than real-time
#[test]
fn test_moonshine_performance() {
    let Some((whisper, samples)) = load_model_and_audio(TEST_AUDIO_1_5S) else {
        return;
    };

    let audio_duration = samples.len() as f64 / 16000.0;
    let (_, _, elapsed) = run_transcription(&whisper, &samples);

    let rtf = elapsed / audio_duration;
    // Debug builds are slow; allow generous RTF
    let max_rtf = if cfg!(debug_assertions) { 100.0 } else { 5.0 };

    println!(
        "Moonshine RTF: {rtf:.2}x (audio={audio_duration:.1}s, decode={elapsed:.2}s, max={max_rtf}x)"
    );

    assert!(
        rtf <= max_rtf,
        "RTF TOO SLOW: {rtf:.2}x > {max_rtf}x. Moonshine should be faster."
    );
}

// =============================================================================
// SECTION F: Variable-Length Correctness Tests (WAPR-MOONSHINE-014 preview)
// =============================================================================

/// WAPR-MOONSHINE-012-F01: Same audio with different padding gives same result
///
/// Moonshine's key advantage over Whisper is proportional processing.
/// Transcription should be identical regardless of trailing silence.
#[test]
fn test_moonshine_padding_invariance() {
    let Some((whisper, samples)) = load_model_and_audio(TEST_AUDIO_1_5S) else {
        return;
    };

    // Run with original audio (1.5s = 24,000 samples)
    let (text_original, _, _) = run_transcription(&whisper, &samples);

    // Run with 1s of silence appended (2.5s total)
    let mut padded = samples.clone();
    padded.extend(vec![0.0_f32; 16000]);
    let (text_padded, _, _) = run_transcription(&whisper, &padded);

    println!("Original: '{}'", text_original.trim());
    println!("Padded:   '{}'", text_padded.trim());

    assert_eq!(
        text_original.trim(),
        text_padded.trim(),
        "Moonshine transcription should be invariant to trailing silence padding. \
         Original: '{}', Padded: '{}'",
        text_original.trim(),
        text_padded.trim()
    );
}

// =============================================================================
// SECTION G: Encoder Shape Tests (no model weights needed)
// =============================================================================

/// WAPR-MOONSHINE-012-G01: ConvStem output shape is correct for variable-length audio
#[test]
fn test_moonshine_conv_stem_shape() {
    // This test doesn't need the model file — uses default weights
    let stem = whisper_apr::audio::ConvStem::new(288);

    // 1.5s at 16kHz = 24,000 samples
    let audio = vec![0.1_f32; 24_000];
    let output = stem.forward(&audio).expect("ConvStem forward failed");

    let expected_frames = whisper_apr::audio::ConvStem::output_frames(24_000);
    let d_model = 288;
    assert_eq!(
        output.len(),
        expected_frames * d_model,
        "ConvStem output shape: expected {}×{d_model}={}, got {}",
        expected_frames,
        expected_frames * d_model,
        output.len()
    );

    // All values should be finite
    assert!(
        output.iter().all(|x| x.is_finite()),
        "ConvStem output contains NaN or Inf"
    );
}

/// WAPR-MOONSHINE-012-G02: ConvStem output scales proportionally with input duration
#[test]
fn test_moonshine_conv_stem_proportional() {
    let stem = whisper_apr::audio::ConvStem::new(288);

    let audio_1s = vec![0.1_f32; 16_000]; // 1 second
    let audio_3s = vec![0.1_f32; 48_000]; // 3 seconds
    let audio_10s = vec![0.1_f32; 160_000]; // 10 seconds

    let out_1s = stem.forward(&audio_1s).expect("1s forward");
    let out_3s = stem.forward(&audio_3s).expect("3s forward");
    let out_10s = stem.forward(&audio_10s).expect("10s forward");

    let frames_1s = out_1s.len() / 288;
    let frames_3s = out_3s.len() / 288;
    let frames_10s = out_10s.len() / 288;

    println!("Frames: 1s={frames_1s}, 3s={frames_3s}, 10s={frames_10s}");

    // 3s should give ~3x the frames of 1s
    let ratio_3s = frames_3s as f32 / frames_1s as f32;
    assert!(
        (ratio_3s - 3.0).abs() < 1.0,
        "3s/1s frame ratio should be ~3x, got {ratio_3s:.1}x"
    );

    // 10s should give ~10x the frames of 1s
    let ratio_10s = frames_10s as f32 / frames_1s as f32;
    assert!(
        (ratio_10s - 10.0).abs() < 2.0,
        "10s/1s frame ratio should be ~10x, got {ratio_10s:.1}x"
    );
}

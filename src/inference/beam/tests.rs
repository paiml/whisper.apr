//! Tests for beam search decoding

use super::*;
use crate::tokenizer::special_tokens;

// =========================================================================
// Construction Tests
// =========================================================================

#[test]
fn test_beam_search_decoder_new() {
    let decoder = BeamSearchDecoder::new(5, 448);
    assert_eq!(decoder.beam_size(), 5);
    assert_eq!(decoder.max_tokens(), 448);
    assert!((decoder.temperature() - 0.0).abs() < f32::EPSILON);
    assert!((decoder.patience() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_beam_search_decoder_builder() {
    let decoder = BeamSearchDecoder::new(3, 256)
        .with_temperature(0.5)
        .with_patience(1.5)
        .with_length_penalty(0.8);
    assert_eq!(decoder.beam_size(), 3);
    assert!((decoder.temperature() - 0.5).abs() < f32::EPSILON);
    assert!((decoder.patience() - 1.5).abs() < f32::EPSILON);
    assert!((decoder.length_penalty() - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_beam_search_decoder_default() {
    let decoder = BeamSearchDecoder::default();
    assert_eq!(decoder.beam_size(), 5);
    assert_eq!(decoder.max_tokens(), 448);
}

// =========================================================================
// Hypothesis Tests
// =========================================================================

#[test]
fn test_hypothesis_new() {
    let hyp = Hypothesis::new(vec![1, 2, 3], -1.5);
    assert_eq!(hyp.tokens, vec![1, 2, 3]);
    assert!((hyp.score - (-1.5)).abs() < f32::EPSILON);
    assert!(!hyp.is_complete);
}

#[test]
fn test_hypothesis_normalized_score() {
    let hyp = Hypothesis::new(vec![1, 2, 3, 4], -4.0);

    // With length_penalty = 1.0, normalized = score / len = -4.0 / 4 = -1.0
    let norm = hyp.normalized_score(1.0);
    assert!((norm - (-1.0)).abs() < 1e-5);

    // With length_penalty = 0.5, normalized = score / len^0.5 = -4.0 / 2 = -2.0
    let norm = hyp.normalized_score(0.5);
    assert!((norm - (-2.0)).abs() < 1e-5);
}

// =========================================================================
// Log Softmax Tests
// =========================================================================

#[test]
fn test_log_softmax_sums_correctly() {
    let decoder = BeamSearchDecoder::new(3, 10);
    let logits = vec![1.0, 2.0, 3.0];
    let log_probs = decoder.log_softmax(&logits);

    // exp(log_probs) should sum to 1
    let sum: f32 = log_probs.iter().map(|&x| x.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_log_softmax_preserves_order() {
    let decoder = BeamSearchDecoder::new(3, 10);
    let logits = vec![1.0, 3.0, 2.0];
    let log_probs = decoder.log_softmax(&logits);

    assert!(log_probs[1] > log_probs[2]);
    assert!(log_probs[2] > log_probs[0]);
}

#[test]
fn test_log_softmax_with_temperature() {
    let decoder = BeamSearchDecoder::new(3, 10).with_temperature(0.5);
    let logits = vec![1.0, 2.0];
    let log_probs = decoder.log_softmax(&logits);

    // Higher temperature should soften distribution
    let decoder_hot = BeamSearchDecoder::new(3, 10).with_temperature(2.0);
    let log_probs_hot = decoder_hot.log_softmax(&logits);

    // With lower temp, difference should be larger
    let diff_cold = log_probs[1] - log_probs[0];
    let diff_hot = log_probs_hot[1] - log_probs_hot[0];
    assert!(diff_cold.abs() > diff_hot.abs());
}

#[test]
fn test_log_softmax_all_neg_inf_returns_uniform() {
    let decoder = BeamSearchDecoder::new(3, 10);
    let logits = vec![f32::NEG_INFINITY; 5];
    let log_probs = decoder.log_softmax(&logits);

    // Should return uniform distribution, not NaN
    assert_eq!(log_probs.len(), 5);
    assert!(log_probs.iter().all(|x| x.is_finite()));

    // All values equal (uniform)
    let expected = -(5.0_f32).ln();
    for &lp in &log_probs {
        assert!((lp - expected).abs() < 1e-5);
    }
}

#[test]
fn test_log_softmax_some_neg_inf() {
    let decoder = BeamSearchDecoder::new(3, 10);
    // Mix of finite and -inf (common after token suppression)
    let logits = vec![f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY, 3.0];
    let log_probs = decoder.log_softmax(&logits);

    // The finite logits should have valid log-prob, suppressed ones get -inf
    assert!(log_probs[1].is_finite());
    assert!(log_probs[3].is_finite());
    // Suppressed tokens get -inf log-prob (correct behavior)
    assert_eq!(log_probs[0], f32::NEG_INFINITY);
    assert_eq!(log_probs[2], f32::NEG_INFINITY);
    // Original order preserved among finite values
    assert!(log_probs[3] > log_probs[1]); // 3.0 > 2.0
}

// =========================================================================
// Top-K Tests
// =========================================================================

#[test]
fn test_top_k_indices() {
    let values = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let top_k = BeamSearchDecoder::top_k_indices(&values, 3);

    assert_eq!(top_k.len(), 3);
    assert_eq!(top_k[0].0, 3); // index of 0.9
    assert_eq!(top_k[1].0, 1); // index of 0.5
    assert_eq!(top_k[2].0, 2); // index of 0.3
}

#[test]
fn test_top_k_indices_k_larger_than_len() {
    let values = vec![0.1, 0.5];
    let top_k = BeamSearchDecoder::top_k_indices(&values, 5);

    assert_eq!(top_k.len(), 2);
}

#[test]
fn test_top_k_indices_single() {
    let values = vec![0.1, 0.5, 0.3];
    let top_k = BeamSearchDecoder::top_k_indices(&values, 1);

    assert_eq!(top_k.len(), 1);
    assert_eq!(top_k[0].0, 1); // index of 0.5
}

// =========================================================================
// Early Stopping Tests
// =========================================================================

#[test]
fn test_should_stop_early_no_completed() {
    let decoder = BeamSearchDecoder::new(3, 10);
    let completed: Vec<Hypothesis> = vec![];
    let candidates = vec![Hypothesis::new(vec![1], -1.0)];

    assert!(!decoder.should_stop_early(&completed, &candidates));
}

#[test]
fn test_should_stop_early_no_candidates() {
    let decoder = BeamSearchDecoder::new(3, 10);
    let completed = vec![Hypothesis::new(vec![1], -1.0)];
    let candidates: Vec<Hypothesis> = vec![];

    assert!(!decoder.should_stop_early(&completed, &candidates));
}

// =========================================================================
// Decode Tests
// =========================================================================

#[test]
fn test_beam_decode_stops_at_eot() {
    let decoder = BeamSearchDecoder::new(3, 100);
    let eot = special_tokens::EOT;

    // Always return EOT as highest probability - should stop immediately
    let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
        let mut logits = vec![-10.0_f32; 51865];
        logits[eot as usize] = 10.0;
        Ok(logits)
    };

    let result = decoder
        .decode(logits_fn, &[special_tokens::SOT], eot)
        .expect("beam decode should succeed");

    // Result should contain EOT (might have intermediate tokens too)
    assert!(result.len() >= 2, "result too short: {:?}", result);
    assert!(result.contains(&eot), "should contain EOT: {:?}", result);
}

#[test]
fn test_beam_decode_respects_max_tokens() {
    let decoder = BeamSearchDecoder::new(2, 3);

    let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
        let mut logits = vec![-10.0_f32; 51865];
        logits[100] = 10.0;
        Ok(logits)
    };

    let result = decoder
        .decode(logits_fn, &[special_tokens::SOT], special_tokens::EOT)
        .expect("beam decode should succeed");

    // Should stop at max_tokens
    assert!(result.len() <= 5); // initial + max_tokens
}

#[test]
fn test_beam_decode_explores_multiple_paths() {
    let decoder = BeamSearchDecoder::new(3, 5);

    // Give different tokens similar scores to force beam to explore
    let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
        let mut logits = vec![-10.0_f32; 51865];
        // Multiple tokens with similar scores
        logits[100] = 1.0;
        logits[101] = 0.9;
        logits[102] = 0.8;
        logits[special_tokens::EOT as usize] = 0.5;
        Ok(logits)
    };

    let result = decoder
        .decode(logits_fn, &[special_tokens::SOT], special_tokens::EOT)
        .expect("beam decode should succeed");

    // Should return a valid sequence
    assert!(!result.is_empty());
}

// =========================================================================
// N-Best Tests
// =========================================================================

#[test]
fn test_decode_nbest() {
    let decoder = BeamSearchDecoder::new(3, 5);
    let eot = special_tokens::EOT;

    let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
        let mut logits = vec![-10.0_f32; 51865];
        logits[100] = 1.0;
        logits[101] = 0.5;
        logits[eot as usize] = 0.9;
        Ok(logits)
    };

    let results = decoder
        .decode_nbest(logits_fn, &[special_tokens::SOT], eot, 2)
        .expect("decode_nbest should succeed");

    // Should return up to 2 results
    assert!(results.len() <= 2);
    assert!(!results.is_empty());
}

#[test]
fn test_decode_nbest_empty_initial() {
    let decoder = BeamSearchDecoder::new(2, 3);
    let eot = special_tokens::EOT;

    let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
        let mut logits = vec![-10.0_f32; 51865];
        logits[eot as usize] = 10.0;
        Ok(logits)
    };

    let results = decoder
        .decode_nbest(logits_fn, &[], eot, 3)
        .expect("decode_nbest should succeed");
    assert!(!results.is_empty());
}

// =========================================================================
// EXTREME TDD: Token Limit Invariant Tests
// =========================================================================

#[test]
fn test_beam_decode_total_tokens_never_exceeds_max() {
    // BUG: beam search loop runs `for _ in 0..max_tokens` generating
    // up to max_tokens NEW tokens, ignoring initial_tokens length.
    // This violates the invariant: output.len() <= max_tokens
    let decoder = BeamSearchDecoder::new(2, 10); // beam=2, max=10

    // Mock logits that never returns EOT
    let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
        let mut logits = vec![0.0_f32; 51865];
        logits[100] = 10.0; // Always pick token 100
        logits[101] = 9.0; // Second choice
        Ok(logits)
    };

    // Start with 5 initial tokens
    let initial = vec![1, 2, 3, 4, 5];
    let result = decoder
        .decode(logits_fn, &initial, special_tokens::EOT)
        .expect("decode should succeed");

    // INVARIANT: total tokens must never exceed max_tokens
    assert!(
        result.len() <= decoder.max_tokens(),
        "beam search: total tokens {} exceeds max_tokens {}",
        result.len(),
        decoder.max_tokens()
    );
}

// =========================================================================
// EXTREME TDD: O(n) Complexity Assertions
// =========================================================================

#[test]
fn test_beam_decode_is_on_not_on2() {
    // Performance test: decoding N tokens should take O(N) time, not O(N²)
    // We measure by counting logits_fn calls - should be O(N), not O(N²)
    use std::sync::atomic::{AtomicUsize, Ordering};

    let call_count = AtomicUsize::new(0);

    let decoder = BeamSearchDecoder::new(1, 50); // beam=1 for predictable calls
    let eot = special_tokens::EOT;

    let logits_fn = |tokens: &[u32]| -> WhisperResult<Vec<f32>> {
        call_count.fetch_add(1, Ordering::SeqCst);
        let mut logits = vec![-10.0_f32; 51865];
        // Return EOT after 20 tokens
        if tokens.len() >= 20 {
            logits[eot as usize] = 10.0;
        } else {
            logits[100] = 10.0;
        }
        Ok(logits)
    };

    let result = decoder
        .decode(logits_fn, &[special_tokens::SOT], eot)
        .expect("decode should succeed");

    let calls = call_count.load(Ordering::SeqCst);
    let tokens_generated = result.len() - 1; // minus initial SOT

    // O(n) means calls should be roughly equal to tokens_generated
    // O(n²) would mean calls ≈ tokens_generated² / 2
    // Allow 2x overhead for beam search bookkeeping
    let max_allowed_calls = tokens_generated * 3;

    assert!(
        calls <= max_allowed_calls,
        "O(n²) detected: {} calls for {} tokens (max allowed: {}). \
         Expected O(n) complexity.",
        calls,
        tokens_generated,
        max_allowed_calls
    );
}

// =========================================================================
// EXTREME TDD: Property-Based Tests for Beam Search
// =========================================================================

#[test]
fn property_beam_output_length_bounded_by_max_tokens() {
    // Property: For any initial_tokens and max_tokens,
    // output.len() <= max_tokens
    for max_tokens in [5, 10, 20, 50] {
        for initial_len in [0, 1, 3, max_tokens / 2, max_tokens - 1, max_tokens] {
            let decoder = BeamSearchDecoder::new(2, max_tokens);

            let logits_fn = |_: &[u32]| -> WhisperResult<Vec<f32>> {
                let mut logits = vec![0.0_f32; 51865];
                logits[100] = 10.0;
                logits[101] = 9.0;
                Ok(logits)
            };

            let initial: Vec<u32> = (0..initial_len).map(|i| i as u32).collect();
            let result = decoder
                .decode(logits_fn, &initial, special_tokens::EOT)
                .expect("decode should succeed");

            assert!(
                result.len() <= max_tokens,
                "Property violated: output.len()={} > max_tokens={} (initial_len={})",
                result.len(),
                max_tokens,
                initial_len
            );
        }
    }
}

#[test]
fn property_beam_initial_tokens_preserved() {
    // Property: output[0..initial.len()] == initial (prefix preserved)
    let decoder = BeamSearchDecoder::new(2, 100);
    let eot = special_tokens::EOT;

    for initial_len in [1, 3, 5, 10] {
        let logits_fn = |_: &[u32]| -> WhisperResult<Vec<f32>> {
            let mut logits = vec![0.0_f32; 51865];
            logits[eot as usize] = 10.0;
            Ok(logits)
        };

        let initial: Vec<u32> = (100..100 + initial_len).collect();
        let result = decoder
            .decode(logits_fn, &initial, eot)
            .expect("decode should succeed");

        assert_eq!(
            &result[..initial.len()],
            &initial[..],
            "Property violated: initial tokens not preserved (initial_len={})",
            initial_len
        );
    }
}

#[test]
fn property_beam_nbest_all_bounded() {
    // Property: All N-best results respect max_tokens limit
    let decoder = BeamSearchDecoder::new(3, 15);

    let logits_fn = |_: &[u32]| -> WhisperResult<Vec<f32>> {
        let mut logits = vec![0.0_f32; 51865];
        logits[100] = 10.0;
        logits[101] = 9.5;
        logits[102] = 9.0;
        Ok(logits)
    };

    let initial = vec![1, 2, 3, 4, 5]; // 5 initial tokens
    let results = decoder
        .decode_nbest(logits_fn, &initial, special_tokens::EOT, 3)
        .expect("decode_nbest should succeed");

    for (i, result) in results.iter().enumerate() {
        assert!(
            result.len() <= decoder.max_tokens(),
            "N-best[{}]: output.len()={} > max_tokens={}",
            i,
            result.len(),
            decoder.max_tokens()
        );
    }
}

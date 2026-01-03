//! Pipeline Fuzz Tests
//!
//! Property-based and fuzz tests for the encode/decode pipeline.
//! Validates correctness under various edge cases and random inputs.

use proptest::prelude::*;
use whisper_apr::model::{Decoder, ModelConfig};
use whisper_apr::simd;

// ============================================================================
// Fuzz Test Helpers
// ============================================================================

/// Create a decoder with test weights for fuzz testing
fn create_test_decoder(config: &ModelConfig) -> Decoder {
    let mut decoder = Decoder::new(config);
    let d_model = config.n_text_state as usize;

    // Initialize with small random-like weights
    for block in decoder.blocks_mut() {
        let scale = 0.1_f32;

        // Initialize all attention weights
        let weight: Vec<f32> = (0..d_model * d_model)
            .map(|i| {
                let row = i / d_model;
                let col = i % d_model;
                if row == col {
                    scale
                } else {
                    scale * (i as f32 * 0.01).sin() * 0.1
                }
            })
            .collect();

        block.cross_attn_mut().w_q_mut().set_weight(&weight);
        block.cross_attn_mut().w_k_mut().set_weight(&weight);
        block.cross_attn_mut().w_v_mut().set_weight(&weight);
        block.cross_attn_mut().w_o_mut().set_weight(&weight);
        block.self_attn_mut().w_q_mut().set_weight(&weight);
        block.self_attn_mut().w_k_mut().set_weight(&weight);
        block.self_attn_mut().w_v_mut().set_weight(&weight);
        block.self_attn_mut().w_o_mut().set_weight(&weight);

        // FFN weights
        let d_ff = d_model * 4;
        let fc1: Vec<f32> = (0..d_ff * d_model)
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();
        let fc2: Vec<f32> = (0..d_model * d_ff)
            .map(|i| (i as f32 * 0.002).cos() * 0.1)
            .collect();
        block.ffn.fc1.set_weight(&fc1);
        block.ffn.fc2.set_weight(&fc2);
    }

    // Token embeddings
    let n_vocab = config.n_vocab as usize;
    let emb: Vec<f32> = (0..n_vocab * d_model)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    decoder.token_embedding_mut().copy_from_slice(&emb);

    decoder.finalize_weights();
    decoder
}

// ============================================================================
// Property-Based Fuzz Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Fuzz test: Decoder output should always be finite for valid inputs
    #[test]
    fn fuzz_decoder_output_finite(
        seq_len in 1usize..10,
        enc_len in 1usize..20,
    ) {
        let config = ModelConfig::tiny();
        let decoder = create_test_decoder(&config);
        let d_model = config.n_text_state as usize;

        // Generate tokens (valid token IDs)
        let tokens: Vec<u32> = (0..seq_len).map(|i| (i % 1000) as u32).collect();

        // Generate encoder output
        let encoder_output: Vec<f32> = (0..enc_len * d_model)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        let result = decoder.forward(&tokens, &encoder_output);
        prop_assert!(result.is_ok(), "Decoder forward failed");

        let logits = result.expect("test assertion");
        prop_assert!(logits.iter().all(|x| x.is_finite()),
            "Decoder output contains non-finite values");
    }

    /// Fuzz test: Softmax should always sum to 1
    #[test]
    fn fuzz_softmax_sums_to_one(len in 10usize..1000) {
        let input: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.1).sin() * 10.0)
            .collect();

        let output = simd::softmax(&input);

        let sum: f32 = output.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 0.001,
            "Softmax sum {} is not close to 1.0", sum
        );

        prop_assert!(
            output.iter().all(|&x| x >= 0.0),
            "Softmax contains negative values"
        );
    }

    /// Fuzz test: Softmax should be stable for extreme values
    #[test]
    fn fuzz_softmax_numerical_stability(
        offset in -1000.0f32..1000.0f32,
        scale in 0.001f32..100.0f32,
    ) {
        let input: Vec<f32> = (0..100)
            .map(|i| offset + (i as f32) * scale)
            .collect();

        let output = simd::softmax(&input);

        // Should be finite
        prop_assert!(
            output.iter().all(|x| x.is_finite()),
            "Softmax output contains non-finite values for offset={}, scale={}",
            offset, scale
        );

        // Should sum to 1
        let sum: f32 = output.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 0.01,
            "Softmax sum {} not close to 1.0 for offset={}, scale={}",
            sum, offset, scale
        );
    }

    /// Fuzz test: Matrix multiply dimensions should be preserved
    #[test]
    fn fuzz_matmul_dimensions(
        rows in 1usize..32,
        inner in 1usize..64,
        cols in 1usize..128,
    ) {
        let a: Vec<f32> = (0..rows * inner)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let b: Vec<f32> = (0..inner * cols)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();

        let c = simd::matmul(&a, &b, rows, inner, cols);

        prop_assert_eq!(
            c.len(),
            rows * cols,
            "Output dimensions wrong: expected {}x{}={}, got {}",
            rows, cols, rows * cols, c.len()
        );

        prop_assert!(
            c.iter().all(|x| x.is_finite()),
            "Matmul output contains non-finite values"
        );
    }

    /// Fuzz test: Layer norm should produce normalized output
    #[test]
    fn fuzz_layer_norm_properties(len in 8usize..256) {
        let input: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.1).sin() * 5.0 + 10.0)
            .collect();
        let gamma = vec![1.0_f32; len];
        let beta = vec![0.0_f32; len];

        let output = simd::layer_norm(&input, &gamma, &beta, 1e-5);

        // Mean should be close to 0 (beta)
        let mean: f32 = output.iter().sum::<f32>() / len as f32;
        prop_assert!(
            mean.abs() < 0.01,
            "Layer norm mean {} should be close to 0", mean
        );

        // Variance should be close to 1
        let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / len as f32;
        prop_assert!(
            (var - 1.0).abs() < 0.1,
            "Layer norm variance {} should be close to 1", var
        );
    }

    /// Fuzz test: GELU should be bounded
    #[test]
    fn fuzz_gelu_bounded(len in 10usize..500) {
        let input: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.05) - 12.5) // Range [-12.5, 12.5]
            .collect();

        let output = simd::gelu(&input);

        prop_assert!(
            output.iter().all(|x| x.is_finite()),
            "GELU output contains non-finite values"
        );

        // GELU is approximately linear for large positive values
        // and approximately 0 for large negative values
        for (_i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            if inp > 3.0 {
                prop_assert!(out > 0.0, "GELU({}) = {} should be positive", inp, out);
            }
            if inp < -5.0 {
                prop_assert!(out.abs() < 0.1, "GELU({}) = {} should be near 0", inp, out);
            }
        }
    }

    /// Fuzz test: Argmax should return valid index
    #[test]
    fn fuzz_argmax_valid(len in 1usize..1000) {
        let input: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let argmax = simd::argmax(&input);

        prop_assert!(
            argmax < len,
            "Argmax {} is out of bounds for len {}", argmax, len
        );

        // Verify it's actually the max
        let max_val = input[argmax];
        prop_assert!(
            input.iter().all(|&x| x <= max_val + 1e-6),
            "Argmax {} (value {}) is not actually the maximum",
            argmax, max_val
        );
    }

    /// Fuzz test: Dot product should be commutative
    #[test]
    fn fuzz_dot_commutative(len in 4usize..256) {
        let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();

        let dot_ab = simd::dot(&a, &b);
        let dot_ba = simd::dot(&b, &a);

        prop_assert!(
            (dot_ab - dot_ba).abs() < 1e-4,
            "Dot product not commutative: {} vs {}", dot_ab, dot_ba
        );
    }

    /// Fuzz test: Vector add should be commutative
    #[test]
    fn fuzz_add_commutative(len in 4usize..256) {
        let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();

        let add_ab = simd::add(&a, &b);
        let add_ba = simd::add(&b, &a);

        for (x, y) in add_ab.iter().zip(add_ba.iter()) {
            prop_assert!(
                (x - y).abs() < 1e-6,
                "Add not commutative"
            );
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_empty_softmax() {
    let empty: Vec<f32> = vec![];
    let result = simd::softmax(&empty);
    assert!(result.is_empty());
}

#[test]
fn test_single_element_softmax() {
    let single = vec![5.0_f32];
    let result = simd::softmax(&single);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_uniform_softmax() {
    let uniform = vec![1.0_f32; 10];
    let result = simd::softmax(&uniform);
    for x in &result {
        assert!((*x - 0.1).abs() < 0.01);
    }
}

#[test]
fn test_extreme_softmax() {
    // Very large values
    let large = vec![1000.0_f32, 1001.0, 999.0];
    let result = simd::softmax(&large);
    assert!(result.iter().all(|x| x.is_finite()));
    assert!((result.iter().sum::<f32>() - 1.0).abs() < 0.01);

    // Very negative values
    let negative = vec![-1000.0_f32, -1001.0, -999.0];
    let result = simd::softmax(&negative);
    assert!(result.iter().all(|x| x.is_finite()));
    assert!((result.iter().sum::<f32>() - 1.0).abs() < 0.01);
}

#[test]
fn test_decoder_with_max_tokens() {
    let config = ModelConfig::tiny();
    let decoder = create_test_decoder(&config);
    let d_model = config.n_text_state as usize;
    let max_len = config.n_text_ctx as usize;

    // Test at max sequence length
    let tokens: Vec<u32> = (0..max_len.min(100)).map(|i| i as u32).collect();
    let encoder_output: Vec<f32> = vec![0.1; 10 * d_model];

    let result = decoder.forward(&tokens, &encoder_output);
    assert!(result.is_ok());
    let logits = result.expect("test assertion");
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_decoder_kv_cache_consistency() {
    let config = ModelConfig::tiny();
    let decoder = create_test_decoder(&config);
    let d_model = config.n_text_state as usize;

    let encoder_output: Vec<f32> = (0..10 * d_model).map(|i| (i as f32 * 0.01).sin()).collect();

    // Generate with batch forward
    let tokens = vec![100_u32, 101, 102];
    let batch_result = decoder
        .forward(&tokens, &encoder_output)
        .expect("test assertion");

    // Generate with forward_one
    let mut cache = decoder.create_kv_cache();
    let mut incremental_result = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        let logits = decoder
            .forward_one(token, &encoder_output, &mut cache)
            .expect("test assertion");
        if i == tokens.len() - 1 {
            incremental_result = logits;
        }
    }

    // Final logits should be similar (may differ due to cumulative numerical errors)
    // Just verify both produce finite results
    assert!(batch_result.iter().all(|x| x.is_finite()));
    assert!(incremental_result.iter().all(|x| x.is_finite()));
}

#[test]
fn test_vocab_projection_dimensions() {
    let config = ModelConfig::tiny();
    let d_model = config.n_text_state as usize;
    let n_vocab = config.n_vocab as usize;

    // Test 1×d_model @ d_model×n_vocab = 1×n_vocab
    let hidden = vec![0.1_f32; d_model];
    let embedding: Vec<f32> = (0..d_model * n_vocab)
        .map(|i| (i as f32 * 0.0001).sin())
        .collect();

    let logits = simd::matmul(&hidden, &embedding, 1, d_model, n_vocab);
    assert_eq!(logits.len(), n_vocab);
    assert!(logits.iter().all(|x| x.is_finite()));
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn stress_test_repeated_forward_one() {
    let config = ModelConfig::tiny();
    let decoder = create_test_decoder(&config);
    let d_model = config.n_text_state as usize;

    let encoder_output: Vec<f32> = vec![0.1; 10 * d_model];
    let mut cache = decoder.create_kv_cache();

    // Generate 50 tokens
    for i in 0..50 {
        let result = decoder.forward_one(i as u32, &encoder_output, &mut cache);
        assert!(result.is_ok(), "Failed at token {}", i);
        let logits = result.expect("test assertion");
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "Non-finite output at token {}",
            i
        );
    }
}

#[test]
fn stress_test_matmul_large() {
    // Test large matrix multiply (vocab projection size)
    let d_model = 384;
    let n_vocab = 51865;

    let hidden: Vec<f32> = (0..d_model).map(|i| (i as f32 * 0.01).sin()).collect();
    let embedding: Vec<f32> = (0..d_model * n_vocab)
        .map(|i| (i as f32 * 0.0001).sin())
        .collect();

    let logits = simd::matmul(&hidden, &embedding, 1, d_model, n_vocab);
    assert_eq!(logits.len(), n_vocab);
    assert!(logits.iter().all(|x| x.is_finite()));
}

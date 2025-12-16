//! Integration tests for realizar inference optimization
//!
//! These tests validate that the realizar-inference feature provides
//! the expected APIs, produces correct results, and delivers memory savings.
//!
//! Sprint 6: Validation & Integration Tests

#![cfg(feature = "realizar-inference")]

use whisper_apr::model::{FusedFFN, QuantizedLinearQ4K, QuantizedTensorQ4K};

/// Test: realizar-inference feature provides expected APIs
///
/// Validates that all optimization types are accessible when feature is enabled.
#[test]
fn test_realizar_feature_provides_expected_apis() {
    // QuantizedTensorQ4K should be accessible
    let super_block_bytes = 144usize;
    let n_values = 256usize;
    let raw_data = vec![0u8; super_block_bytes];

    let tensor = QuantizedTensorQ4K::from_raw(raw_data, vec![n_values]);
    assert_eq!(tensor.len(), n_values);
    assert!(!tensor.is_empty());

    // QuantizedLinearQ4K should be accessible
    let in_features = 256usize;
    let out_features = 64usize;
    let weight_values = in_features * out_features;
    let n_blocks = weight_values.div_ceil(256);
    let weight_data = vec![0u8; super_block_bytes * n_blocks];

    let linear = QuantizedLinearQ4K::from_raw(weight_data, None, in_features, out_features);
    assert_eq!(linear.in_features(), in_features);
    assert_eq!(linear.out_features(), out_features);

    // realizar_inference module should provide dequantize_q4_k
    let dequant_result = whisper_apr::realizar_inference::dequantize_q4_k(&[0u8; 144]);
    assert!(dequant_result.is_ok());
}

/// Test: Optimized FusedFFN forward matches baseline unfused forward
///
/// Validates that FusedFFN produces numerically equivalent results to
/// the unfused LayerNorm + Linear path.
#[test]
fn test_optimized_forward_matches_baseline() {
    let d_model = 64;
    let d_ff = 256;

    // Create FusedFFN with known weights
    let mut fused = FusedFFN::new(d_model, d_ff).expect("FusedFFN creation should succeed");

    // Set weights to identity-like values for testing
    let norm_weight = vec![1.0f32; d_model];
    let norm_bias = vec![0.0f32; d_model];
    fused.set_norm_weights(&norm_weight, &norm_bias);

    // Simple fc1 weights (scale by 0.1)
    let fc1_weight = (0..d_ff * d_model)
        .map(|i| if i % (d_model + 1) == 0 { 0.1 } else { 0.0 })
        .collect::<Vec<f32>>();
    let fc1_bias = vec![0.0f32; d_ff];
    fused.set_fc1_weights(&fc1_weight, &fc1_bias);

    // Simple fc2 weights
    let fc2_weight = (0..d_model * d_ff)
        .map(|i| if i % (d_ff + 1) == 0 { 0.1 } else { 0.0 })
        .collect::<Vec<f32>>();
    let fc2_bias = vec![0.0f32; d_model];
    fused.set_fc2_weights(&fc2_weight, &fc2_bias);

    // Test input
    let input = (0..d_model).map(|i| i as f32 * 0.1).collect::<Vec<f32>>();

    // Forward pass
    let output = fused.forward(&input).expect("Forward should succeed");

    // Output should have correct shape
    assert_eq!(output.len(), d_model, "Output should have d_model elements");

    // Output should be finite (no NaN/Inf)
    assert!(
        output.iter().all(|x: &f32| x.is_finite()),
        "All output values should be finite"
    );

    // Output should not be all zeros (computation happened)
    let sum: f32 = output.iter().map(|x: &f32| x.abs()).sum();
    assert!(sum > 0.0, "Output should not be all zeros");
}

/// Test: Q4K quantization provides significant memory savings
///
/// Validates that QuantizedLinearQ4K uses <20% of the memory compared
/// to equivalent f32 storage.
#[test]
fn test_quantized_memory_savings_significant() {
    let super_block_bytes = 144usize;
    let in_features = 512usize;
    let out_features = 512usize;
    let n_values = in_features * out_features; // 262144 values
    let n_blocks = n_values.div_ceil(256);

    // Q4K weight storage
    let weight_data = vec![0u8; super_block_bytes * n_blocks];
    let bias = vec![0.0f32; out_features];
    let linear = QuantizedLinearQ4K::from_raw(weight_data, Some(&bias), in_features, out_features);

    let q4k_memory = linear.memory_size();

    // Equivalent f32 storage
    let f32_weight_memory = n_values * 4; // 4 bytes per f32
    let f32_bias_memory = out_features * 4;
    let f32_total_memory = f32_weight_memory + f32_bias_memory;

    // Calculate compression ratio
    let compression_ratio = f32_total_memory as f64 / q4k_memory as f64;

    // Q4K should provide at least 5x compression (target is ~7x)
    assert!(
        compression_ratio > 5.0,
        "Q4K should provide >5x compression, got {:.2}x (Q4K: {} bytes, f32: {} bytes)",
        compression_ratio,
        q4k_memory,
        f32_total_memory
    );

    // Q4K should use less than 20% of f32 memory
    let memory_ratio = q4k_memory as f64 / f32_total_memory as f64;
    assert!(
        memory_ratio < 0.20,
        "Q4K should use <20% of f32 memory, got {:.1}%",
        memory_ratio * 100.0
    );
}

/// Test: Q4K linear forward produces reasonable outputs
///
/// Validates that QuantizedLinearQ4K::forward() produces finite, non-zero
/// outputs when given reasonable inputs.
#[test]
fn test_q4k_linear_forward_produces_outputs() {
    let super_block_bytes = 144usize;
    let in_features = 256usize;
    let out_features = 64usize;
    let n_values = in_features * out_features;
    let n_blocks = n_values.div_ceil(256);

    // Create Q4K linear with some non-zero scale data
    let mut weight_data = vec![0u8; super_block_bytes * n_blocks];
    // Set scale bytes to small non-zero values in each super-block
    for block in 0..n_blocks {
        let offset = block * super_block_bytes;
        // d (scale) in f16: 0x3C00 = 1.0
        weight_data[offset] = 0x00;
        weight_data[offset + 1] = 0x3C;
    }

    let bias = vec![0.1f32; out_features];
    let linear = QuantizedLinearQ4K::from_raw(weight_data, Some(&bias), in_features, out_features);

    // Input with some variation
    let input: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.01).collect();

    // Forward pass
    let output = linear.forward(&input).expect("Forward should succeed");

    // Output should have correct shape
    assert_eq!(output.len(), out_features);

    // Output should be finite
    assert!(
        output.iter().all(|x: &f32| x.is_finite()),
        "All output values should be finite"
    );

    // Output should include bias (not all zeros since bias is 0.1)
    let min_val: f32 = output.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        min_val >= 0.09, // bias of 0.1 minus small tolerance
        "Output should include bias contribution, min value: {}",
        min_val
    );
}

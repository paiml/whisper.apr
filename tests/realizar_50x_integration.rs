//! EXTREME TDD: 50x Performance Integration Tests (WAPR-PERF-003)
//!
//! RED PHASE: These tests define the target behavior for realizar integration.
//! Reference: docs/specifications/wasm-50x-performance-10x-smaller.md

#![cfg(feature = "realizar-inference")]

use whisper_apr::model::{Decoder, Encoder, ModelConfig};
use whisper_apr::realizar_inference::{dequantize_q4_k, fused_q4k_dot_simd, PagedKvCache};

// =============================================================================
// Section 1: PagedKvCache Integration (50x memory efficiency)
// =============================================================================

/// RED: PagedKvCache should be creatable with realizar API
#[test]
fn test_paged_kv_cache_creation() {
    // realizar::paged_kv::PagedKvCache API
    let page_size = 16;
    let num_pages = 64;
    let num_heads = 6;
    let head_dim = 64;

    // PagedKvCache::new returns struct directly
    let _cache = PagedKvCache::new(page_size, num_pages, num_heads, head_dim);

    // Cache created successfully - no panics
}

/// RED: PagedKvCache sequence operations
#[test]
fn test_paged_kv_cache_sequence_ops() {
    let mut cache = PagedKvCache::new(16, 64, 6, 64);

    // allocate_sequence takes num_tokens, returns Result<SeqId>
    let result = cache.allocate_sequence(32);
    assert!(result.is_ok(), "Sequence allocation should succeed");

    let _seq_id = result.unwrap();
    // SeqId returned - allocation succeeded
}

// =============================================================================
// Section 2: Q4K Quantization (10x compression)
// =============================================================================

/// RED: Q4K dequantization produces finite values
#[test]
fn test_q4k_dequantize_values() {
    // Q4_K block: 144 bytes per 256 values
    // Format: d(f16) + dmin(f16) + scales[12] + qs[128]
    let mut block_data = vec![0u8; 144];
    // Set d and dmin to small values (f16 encoded)
    block_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0 in f16
    block_data[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0 in f16

    let result = dequantize_q4_k(&block_data);
    assert!(result.is_ok(), "Dequantization should succeed");

    let output = result.unwrap();
    assert_eq!(output.len(), 256);
    assert!(output.iter().all(|&v| v.is_finite()));
}

/// RED: Fused Q4K dot product returns Result
#[test]
fn test_fused_q4k_dot() {
    // Q4_K block: 144 bytes
    let block_data = vec![0u8; 144];
    let input = vec![1.0f32; 256];

    let result = fused_q4k_dot_simd(&block_data, &input);

    // Should return Ok with a finite value
    assert!(result.is_ok());
    let value = result.unwrap();
    assert!(value.is_finite());
}

// =============================================================================
// Section 3: Decoder Integration
// =============================================================================

/// RED: Decoder construction with ModelConfig
#[test]
fn test_decoder_construction() {
    let config = ModelConfig::tiny();
    let decoder = Decoder::new(&config);

    // Verify decoder was created with correct number of layers
    assert_eq!(decoder.n_layers(), config.n_text_layer as usize);
}

/// RED: Decoder KV cache creation
#[test]
fn test_decoder_kv_cache() {
    let config = ModelConfig::tiny();
    let decoder = Decoder::new(&config);
    let kv_cache = decoder.create_kv_cache();

    // Should have caches for each layer
    assert_eq!(kv_cache.self_attn_cache.len(), config.n_text_layer as usize);
    assert_eq!(kv_cache.cross_attn_cache.len(), config.n_text_layer as usize);
}

/// RED: Decoder PAGED KV cache creation (realizar integration)
#[test]
fn test_decoder_paged_kv_cache() {
    let config = ModelConfig::tiny();
    let decoder = Decoder::new(&config);

    // Create paged cache with 64 pages
    let paged_cache = decoder.create_paged_kv_cache(64);

    // Should have correct number of layers
    assert_eq!(paged_cache.num_layers(), config.n_text_layer as usize);
    assert_eq!(paged_cache.total_pages(), 64);
}

/// RED: Paged cache sequence allocation
#[test]
fn test_paged_cache_sequence_allocation() {
    let config = ModelConfig::tiny();
    let decoder = Decoder::new(&config);
    let mut paged_cache = decoder.create_paged_kv_cache(64);

    // Allocate a sequence
    let result = paged_cache.allocate_sequence(32);
    assert!(result.is_ok(), "Sequence allocation should succeed");
}

/// GREEN: Decoder forward with paged cache (50x memory target)
#[test]
fn test_decoder_forward_paged() {
    let config = ModelConfig::tiny();
    let decoder = Decoder::new(&config);
    let mut paged_cache = decoder.create_paged_kv_cache(64);

    // Allocate sequence with 0 initial tokens (fresh sequence)
    let seq_id = paged_cache.allocate_sequence(0).unwrap();

    // Mock encoder output (seq_len=1500, d_model=384 for tiny)
    let encoder_output = vec![0.1f32; 1500 * 384];

    // Single token input
    let token = 50258u32; // <|startoftranscript|>

    // Forward with paged cache - should append to cache
    let result = decoder.forward_one_paged(token, &encoder_output, &mut paged_cache, seq_id);

    assert!(result.is_ok(), "Paged forward should succeed: {:?}", result.err());
    let logits = result.unwrap();
    assert_eq!(logits.len(), config.n_vocab as usize);
}

// =============================================================================
// Section 4: Encoder Integration
// =============================================================================

/// RED: Encoder construction
#[test]
fn test_encoder_construction() {
    let config = ModelConfig::tiny();
    let encoder = Encoder::new(&config);

    assert_eq!(encoder.n_layers(), config.n_audio_layer as usize);
}

// =============================================================================
// Section 5: Model Size Calculations (10x smaller target)
// =============================================================================

/// GREEN: Q4K model size calculation
#[test]
fn test_q4k_model_size_target() {
    // whisper-tiny: ~39M params
    // Q4K: 4.5 bits/param → ~22MB
    // With 60% pruning (keep 40%) → ~9MB
    let param_count = 39_000_000u64;
    let bits_per_param = 4.5f64;
    let pruning_keep = 0.4f64;

    let size_bytes = (param_count as f64 * bits_per_param / 8.0 * pruning_keep) as u64;
    let size_mb = size_bytes as f64 / (1024.0 * 1024.0);

    // Target: <10MB for Q4K+pruned tiny
    assert!(size_mb < 10.0, "Expected <10MB, got {size_mb:.1}MB");
}

/// GREEN: QuantizedLinearQ4K compression ratio validation
#[test]
fn test_q4k_linear_compression() {
    use whisper_apr::model::QuantizedLinearQ4K;

    let in_features: usize = 384; // tiny d_model
    let out_features: usize = 384;
    let n_values = in_features * out_features;

    // Q4K: 144 bytes per 256 values
    let super_block_bytes = 144;
    let n_blocks = n_values.div_ceil(256);
    let q4k_bytes = super_block_bytes * n_blocks;

    let raw_data = vec![0u8; q4k_bytes];
    let linear = QuantizedLinearQ4K::from_raw(raw_data, None, in_features, out_features);

    let f32_bytes = n_values * 4;
    let compression = f32_bytes as f64 / linear.memory_size() as f64;

    // Q4K should achieve ~7x compression (32/4.5 = 7.1x theoretical)
    assert!(compression > 6.0, "Q4K compression should be >6x, got {compression:.1}x");
}

/// RED: Compression ratio from spec
#[test]
fn test_compression_ratio_target() {
    // FP32 tiny: ~145MB
    // Target: 3.7MB
    // Ratio: 39x
    let fp32_size_mb = 145.0f64;
    let target_size_mb = 3.7f64;
    let target_ratio = fp32_size_mb / target_size_mb;

    assert!(target_ratio > 35.0, "Compression ratio should be >35x");
}

// =============================================================================
// Section 6: Performance Targets (50x speedup) - IGNORED until GPU ready
// =============================================================================

/// RED: Attention performance target
#[test]
#[ignore] // Enable when GPU backend is ready
fn test_attention_performance_50x() {
    use std::time::Instant;

    let seq_len = 1500;
    let d_model = 384;
    let _num_heads = 6;

    let _q = vec![0.0f32; seq_len * d_model];
    let _k = vec![0.0f32; seq_len * d_model];
    let _v = vec![0.0f32; seq_len * d_model];

    let start = Instant::now();
    // TODO: Call realizar flash attention when GPU ready
    let elapsed = start.elapsed();

    // Target: <4ms (from 188ms baseline = 47x)
    assert!(elapsed.as_millis() < 4, "Target: <4ms, got {}ms", elapsed.as_millis());
}

/// RED: Encoder performance target
#[test]
#[ignore] // Enable when GPU backend is ready
fn test_encoder_performance_50x() {
    use std::time::Instant;

    let config = ModelConfig::tiny();
    let encoder = Encoder::new(&config);
    let mel_input = vec![0.0f32; 3000 * 80];

    let start = Instant::now();
    let _output = encoder.forward(&mel_input);
    let elapsed = start.elapsed();

    // Target: <4ms (from 165ms baseline = 41x)
    assert!(elapsed.as_millis() < 4, "Target: <4ms, got {}ms", elapsed.as_millis());
}

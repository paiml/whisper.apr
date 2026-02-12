//! Tests for LFM2 model implementation

use super::*;
use crate::format::apr2::LayerType;
use crate::model::lfm2::wasm_config::{Lfm2WasmConfig, WasmMemoryEstimate, WasmQuantization};

#[test]
fn test_lfm2_new_small() {
    // Small config for testing
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 16;
    config.num_layers = 3;
    config.num_q_heads = 4;
    config.num_kv_heads = 2;
    config.intermediate_size = 32;
    config.vocab_size = 100;
    config.max_seq_len = 64;
    config.layer_types = vec![
        LayerType::Convolution {
            kernel_size: 4,
            cache_len: 3,
        },
        LayerType::Convolution {
            kernel_size: 4,
            cache_len: 3,
        },
        LayerType::Attention { use_gqa: true },
    ];

    let model = Lfm2::new(config).expect("should create model");
    assert_eq!(model.layers.len(), 3);
    assert!(model.layers[0].conv.is_some());
    assert!(model.layers[1].conv.is_some());
    assert!(model.layers[2].attention.is_some());
}

#[test]
fn test_lfm2_forward_small() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 2;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![
        LayerType::Convolution {
            kernel_size: 2,
            cache_len: 1,
        },
        LayerType::Attention { use_gqa: true },
    ];

    let model = Lfm2::new(config).expect("should create model");

    // Forward with small input
    let input_ids = vec![1u32, 2, 3, 4];
    let logits = model
        .forward(&input_ids, None)
        .expect("forward should succeed");

    assert_eq!(logits.len(), 4 * 50); // seq_len * vocab_size
}

#[test]
fn test_rmsnorm() {
    let norm = RmsNorm::new(4);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 positions

    let output = norm.forward(&input, 2).expect("should normalize");
    assert_eq!(output.len(), 8);

    // Check that output is normalized (RMS ≈ 1 after scaling)
    let rms_out: f32 = output[0..4].iter().map(|x| x * x).sum::<f32>() / 4.0;
    assert!(rms_out.sqrt() > 0.1); // Not zero
}

#[test]
fn test_lfm2_layer_conv() {
    let layer = Lfm2Layer::new(
        0,
        LayerType::Convolution {
            kernel_size: 3,
            cache_len: 2,
        },
        8,  // hidden_size
        16, // intermediate_size
        2,  // num_q_heads
        1,  // num_kv_heads
    )
    .expect("should create layer");

    assert!(layer.conv.is_some());
    assert!(layer.attention.is_none());
}

#[test]
fn test_lfm2_layer_attention() {
    let layer = Lfm2Layer::new(0, LayerType::Attention { use_gqa: true }, 8, 16, 2, 1)
        .expect("should create layer");

    assert!(layer.attention.is_some());
    assert!(layer.conv.is_none());
}

#[test]
fn test_argmax() {
    let x = vec![1.0, 3.0, 2.0, 0.5];
    assert_eq!(argmax(&x), 1);

    let y = vec![-1.0, -2.0, -0.5];
    assert_eq!(argmax(&y), 2);
}

#[test]
fn test_lfm2_generate() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 2;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![
        LayerType::Convolution {
            kernel_size: 2,
            cache_len: 1,
        },
        LayerType::Attention { use_gqa: true },
    ];

    let model = Lfm2::new(config).expect("should create model");

    // Generate tokens
    let prompt = vec![1u32, 2, 3];
    let output = model
        .generate(&prompt, 5, 1.0)
        .expect("generate should succeed");

    // Should have at least prompt length + some generated
    assert!(output.len() >= prompt.len());
    assert!(output.len() <= prompt.len() + 5);
}

#[test]
fn test_lfm2_load_stats_display() {
    let stats = LoadStats {
        tensors_loaded: 10,
        params_loaded: 1000,
    };

    let display = format!("{}", stats);
    assert!(display.contains("10 tensors"));
    assert!(display.contains("1000 params"));
}

#[test]
fn test_lfm2_config_default() {
    // Create default LFM2 config
    let config = Lfm2Config::lfm2_2_6b();

    // Verify LFM2-2.6B config values
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_layers, 30);
    assert_eq!(config.num_q_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    // GQA ratio = Q/KV = 32/8 = 4
    assert_eq!(config.num_q_heads / config.num_kv_heads, 4);
}

#[test]
fn test_lfm2_roundtrip_small() {
    // Create small model
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 4;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 8;
    config.vocab_size = 10;
    config.max_seq_len = 16;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config.clone()).expect("should create model");

    // Forward pass to verify it works
    let input_ids = vec![1u32, 2, 3];
    let logits = model
        .forward(&input_ids, None)
        .expect("forward should succeed");

    // logits should have seq_len * vocab_size elements
    assert_eq!(logits.len(), 3 * 10);
}

// =========================================================================
// WASM Configuration Tests (Section 18.7)
// =========================================================================

#[test]
fn test_wasm_quantization_bytes_per_param() {
    assert_eq!(WasmQuantization::Fp16.bytes_per_param(), 2.0);
    assert_eq!(WasmQuantization::Int8.bytes_per_param(), 1.0);
    assert_eq!(WasmQuantization::Int4Awq.bytes_per_param(), 0.5);
    assert_eq!(WasmQuantization::Int4Gptq.bytes_per_param(), 0.5);
}

#[test]
fn test_wasm_quantization_display() {
    assert_eq!(format!("{}", WasmQuantization::Fp16), "fp16");
    assert_eq!(format!("{}", WasmQuantization::Int8), "int8");
    assert_eq!(format!("{}", WasmQuantization::Int4Awq), "int4-awq");
    assert_eq!(format!("{}", WasmQuantization::Int4Gptq), "int4-gptq");
}

#[test]
fn test_wasm_quantization_viability() {
    let lfm2_params: u64 = 2_600_000_000;

    // fp16: 2.6B * 2 = 5.2GB - NOT viable
    assert!(!WasmQuantization::Fp16.is_wasm_viable(lfm2_params));

    // int8: 2.6B * 1 = 2.6GB - NOT viable (exceeds 2GB)
    assert!(!WasmQuantization::Int8.is_wasm_viable(lfm2_params));

    // int4: 2.6B * 0.5 = 1.3GB - Viable
    assert!(WasmQuantization::Int4Awq.is_wasm_viable(lfm2_params));
    assert!(WasmQuantization::Int4Gptq.is_wasm_viable(lfm2_params));
}

#[test]
fn test_lfm2_wasm_config_default() {
    let config = Lfm2WasmConfig::default();

    assert_eq!(config.quantization, WasmQuantization::Int4Awq);
    assert_eq!(config.max_context, 4096);
    assert_eq!(config.sliding_window, Some(2048));
    assert!(config.use_webgpu);
    assert!(config.streaming);
}

#[test]
fn test_lfm2_wasm_config_lfm2_2_6b() {
    let config = Lfm2WasmConfig::lfm2_2_6b();

    // Should be same as default
    assert_eq!(config.quantization, WasmQuantization::Int4Awq);
    assert_eq!(config.max_context, 4096);
}

#[test]
fn test_lfm2_wasm_config_full_attention() {
    let config = Lfm2WasmConfig::full_attention();

    assert!(config.sliding_window.is_none());
    assert_eq!(config.quantization, WasmQuantization::Int4Awq);
}

#[test]
fn test_lfm2_wasm_config_low_memory() {
    let config = Lfm2WasmConfig::low_memory();

    assert_eq!(config.max_context, 2048);
    assert_eq!(config.sliding_window, Some(1024));
}

#[test]
fn test_wasm_memory_estimate_int4() {
    let model_config = Lfm2Config::lfm2_2_6b();
    let wasm_config = Lfm2WasmConfig::default();

    let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);

    // Model bytes: 2.6B * 0.5 = 1.3GB
    assert_eq!(estimate.model_bytes, 1_300_000_000);

    // KV cache: depends on sliding window (2048 tokens)
    // Per token: 2 * 30 * 8 * 64 * 2 = 61440 bytes
    // Total: 61440 * 2048 = ~125MB
    assert!(estimate.kv_cache_bytes > 100_000_000); // > 100MB
    assert!(estimate.kv_cache_bytes < 200_000_000); // < 200MB

    // Total should be around 1.5-1.7GB
    assert!(estimate.total_bytes > 1_500_000_000);
    assert!(estimate.total_bytes < 2_000_000_000);

    // int4 with sliding window should be viable
    assert!(estimate.is_viable);
}

#[test]
fn test_wasm_memory_estimate_fp16_not_viable() {
    let model_config = Lfm2Config::lfm2_2_6b();
    let wasm_config = Lfm2WasmConfig {
        quantization: WasmQuantization::Fp16,
        ..Lfm2WasmConfig::default()
    };

    let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);

    // fp16 should not be viable
    assert!(!estimate.is_viable);
    assert!(!estimate.warnings.is_empty());
    assert!(estimate.warnings.iter().any(|w| w.contains("fp16")));
}

#[test]
fn test_wasm_memory_estimate_large_context_warning() {
    let model_config = Lfm2Config::lfm2_2_6b();
    let wasm_config = Lfm2WasmConfig {
        max_context: 16384,
        sliding_window: None,
        ..Lfm2WasmConfig::default()
    };

    let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);

    // Should have warning about large context
    assert!(estimate
        .warnings
        .iter()
        .any(|w| w.contains("16384") || w.contains("context")));
}

#[test]
fn test_wasm_memory_estimate_summary() {
    let model_config = Lfm2Config::lfm2_2_6b();
    let wasm_config = Lfm2WasmConfig::default();

    let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);
    let summary = estimate.summary();

    // Should contain key information
    assert!(summary.contains("Model:"));
    assert!(summary.contains("KV Cache:"));
    assert!(summary.contains("Overhead:"));
    assert!(summary.contains("Total:"));
    assert!(summary.contains("GB"));
}

#[test]
fn test_wasm_memory_estimate_display() {
    let model_config = Lfm2Config::lfm2_2_6b();
    let wasm_config = Lfm2WasmConfig::default();

    let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);
    let display = format!("{}", estimate);

    // Display should be same as summary
    assert_eq!(display, estimate.summary());
}

#[test]
fn test_wasm_memory_estimate_low_memory_config() {
    let model_config = Lfm2Config::lfm2_2_6b();
    let wasm_config = Lfm2WasmConfig::low_memory();

    let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);

    // Low memory config should use less KV cache
    let default_estimate = WasmMemoryEstimate::calculate(&model_config, &Lfm2WasmConfig::default());

    assert!(estimate.kv_cache_bytes < default_estimate.kv_cache_bytes);
    assert!(estimate.total_bytes < default_estimate.total_bytes);
}

// =========================================================================
// End-to-End Inference Tests (WAPR-LFM2-013)
// =========================================================================

#[test]
fn test_lfm2_tokenizer_model_roundtrip() {
    use crate::model::lfm2::tokenizer::ByteLevelTokenizer;

    // Create small model
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 512; // Need room for byte tokens
    config.max_seq_len = 32;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");
    let tokenizer = ByteLevelTokenizer::new();

    // Encode text
    let text = "Hello";
    let tokens = tokenizer.encode_without_special(text);
    assert!(!tokens.is_empty());

    // Forward pass
    let logits = model
        .forward(&tokens, None)
        .expect("forward should succeed");
    assert_eq!(logits.len(), tokens.len() * 512);

    // Verify logits are finite
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_lfm2_generate_with_eos() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 100;
    config.max_seq_len = 64;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");

    // Generate with max_tokens limit
    let prompt = vec![1u32, 2, 3];
    let output = model
        .generate(&prompt, 10, 0.0) // temperature 0 = deterministic
        .expect("generate should succeed");

    // Should respect max_tokens
    assert!(output.len() <= prompt.len() + 10);
}

#[test]
fn test_lfm2_forward_position_ids() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");

    let input_ids = vec![1u32, 2, 3, 4];

    // Without position_ids (default)
    let logits1 = model
        .forward(&input_ids, None)
        .expect("forward should succeed");

    // With explicit position_ids
    let position_ids = vec![0usize, 1, 2, 3];
    let logits2 = model
        .forward(&input_ids, Some(&position_ids))
        .expect("forward should succeed");

    // Both should produce same output for sequential positions
    assert_eq!(logits1.len(), logits2.len());
}

#[test]
fn test_lfm2_multi_layer_inference() {
    // Test with multiple layer types
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 16;
    config.num_layers = 4;
    config.num_q_heads = 4;
    config.num_kv_heads = 2;
    config.intermediate_size = 32;
    config.vocab_size = 100;
    config.max_seq_len = 32;
    config.layer_types = vec![
        LayerType::Convolution {
            kernel_size: 3,
            cache_len: 2,
        },
        LayerType::Convolution {
            kernel_size: 3,
            cache_len: 2,
        },
        LayerType::Attention { use_gqa: true },
        LayerType::Attention { use_gqa: true },
    ];

    let model = Lfm2::new(config).expect("should create model");

    // Verify layer structure
    assert_eq!(model.layers.len(), 4);
    assert!(model.layers[0].conv.is_some());
    assert!(model.layers[1].conv.is_some());
    assert!(model.layers[2].attention.is_some());
    assert!(model.layers[3].attention.is_some());

    // Forward pass
    let input_ids = vec![10u32, 20, 30];
    let logits = model
        .forward(&input_ids, None)
        .expect("forward should succeed");
    assert_eq!(logits.len(), 3 * 100);
}

#[test]
fn test_lfm2_longer_sequence() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 2;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 100;
    config.max_seq_len = 128;
    config.layer_types = vec![
        LayerType::Convolution {
            kernel_size: 2,
            cache_len: 1,
        },
        LayerType::Attention { use_gqa: true },
    ];

    let model = Lfm2::new(config).expect("should create model");

    // Test with longer sequence
    let seq_len = 64;
    let input_ids: Vec<u32> = (0..seq_len as u32).map(|i| i % 99 + 1).collect();
    let logits = model
        .forward(&input_ids, None)
        .expect("forward should succeed");

    assert_eq!(logits.len(), seq_len * 100);
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_lfm2_batch_inference_sequential() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");

    // Process multiple sequences sequentially
    let sequences = vec![vec![1u32, 2, 3], vec![4u32, 5], vec![6u32, 7, 8, 9]];

    for seq in &sequences {
        let logits = model.forward(seq, None).expect("forward should succeed");
        assert_eq!(logits.len(), seq.len() * 50);
    }
}

#[test]
fn test_lfm2_memory_bound_check() {
    // Verify model respects memory constraints
    let config = Lfm2Config::lfm2_2_6b();

    // Calculate expected parameter count
    let embedding_params = config.vocab_size as u64 * config.hidden_size as u64;
    let layer_params_approx = config.num_layers as u64
        * (4 * config.hidden_size as u64 * config.hidden_size as u64 // QKV + O
        + 3 * config.hidden_size as u64 * config.intermediate_size as u64); // FFN

    let total_params = embedding_params + layer_params_approx;

    // LFM2-2.6B should have approximately 2.6B params
    assert!(total_params > 2_000_000_000);
    assert!(total_params < 4_000_000_000);
}

#[test]
fn test_lfm2_generate_temperature_scaling() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");
    let prompt = vec![1u32, 2, 3];

    // Low temperature = more deterministic
    let output_low = model
        .generate(&prompt, 3, 0.1)
        .expect("generate should succeed");

    // High temperature = more random (but still valid)
    let output_high = model
        .generate(&prompt, 3, 2.0)
        .expect("generate should succeed");

    // Both should produce valid token sequences
    assert!(output_low.iter().all(|&t| t < 50));
    assert!(output_high.iter().all(|&t| t < 50));
}

#[test]
fn test_lfm2_empty_input() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");

    // Empty input should return empty output
    let input_ids: Vec<u32> = vec![];
    let logits = model
        .forward(&input_ids, None)
        .expect("forward should succeed");
    assert!(logits.is_empty());
}

#[test]
fn test_lfm2_single_token() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 2;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![
        LayerType::Convolution {
            kernel_size: 2,
            cache_len: 1,
        },
        LayerType::Attention { use_gqa: true },
    ];

    let model = Lfm2::new(config).expect("should create model");

    // Single token
    let input_ids = vec![10u32];
    let logits = model
        .forward(&input_ids, None)
        .expect("forward should succeed");
    assert_eq!(logits.len(), 50);
}

#[test]
fn test_lfm2_out_of_vocab_error() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");

    // Token ID > vocab_size should fail
    let input_ids = vec![1u32, 100]; // 100 > vocab_size (50)
    let result = model.forward(&input_ids, None);
    assert!(result.is_err());
}

// =========================================================================
// Streaming Generation Tests (WAPR-LFM2-014)
// =========================================================================

#[test]
fn test_lfm2_generate_streaming() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");
    let prompt = vec![1u32, 2, 3];

    // Track tokens via callback
    let mut streamed_tokens = Vec::new();
    let output = model
        .generate_streaming(&prompt, 5, 0.0, |token, _idx| {
            streamed_tokens.push(token);
            true // continue
        })
        .expect("streaming should succeed");

    // Should have streamed tokens
    assert!(!streamed_tokens.is_empty());
    // Output should include prompt + generated
    assert!(output.len() >= prompt.len());
    // Streamed tokens should match generated portion
    let generated = &output[prompt.len()..];
    assert_eq!(streamed_tokens.len(), generated.len());
    for (i, &token) in streamed_tokens.iter().enumerate() {
        assert_eq!(token, generated[i]);
    }
}

#[test]
fn test_lfm2_generate_streaming_early_stop() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");
    let prompt = vec![1u32, 2, 3];

    // Stop after 2 tokens
    let mut count = 0;
    let output = model
        .generate_streaming(&prompt, 10, 0.0, |_token, _idx| {
            count += 1;
            count < 2 // stop after 2
        })
        .expect("streaming should succeed");

    // Should have stopped early
    assert_eq!(output.len(), prompt.len() + 2);
}

#[test]
fn test_lfm2_generate_with_stats() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");
    let prompt = vec![1u32, 2, 3];

    // Without callback
    let (output, stats) = model
        .generate_with_stats::<fn(u32, usize) -> bool>(&prompt, 3, 0.0, None)
        .expect("generate should succeed");

    assert!(output.len() >= prompt.len());
    assert!(stats.tokens_generated > 0 || output.len() == prompt.len());
    assert!(stats.total_ms >= 0.0);
    // Display impl should work
    let display = format!("{}", stats);
    assert!(display.contains("tokens"));
}

#[test]
fn test_lfm2_generate_with_stats_and_callback() {
    let mut config = Lfm2Config::lfm2_2_6b();
    config.hidden_size = 8;
    config.num_layers = 1;
    config.num_q_heads = 2;
    config.num_kv_heads = 1;
    config.intermediate_size = 16;
    config.vocab_size = 50;
    config.max_seq_len = 32;
    config.layer_types = vec![LayerType::Attention { use_gqa: true }];

    let model = Lfm2::new(config).expect("should create model");
    let prompt = vec![1u32, 2, 3];

    let mut tokens_seen = 0;
    let (output, stats) = model
        .generate_with_stats(
            &prompt,
            3,
            0.0,
            Some(|_token: u32, _idx: usize| {
                tokens_seen += 1;
                true
            }),
        )
        .expect("generate should succeed");

    assert!(output.len() >= prompt.len());
    assert_eq!(tokens_seen, stats.tokens_generated);
}

#[test]
fn test_generation_stats_display() {
    let stats = GenerationStats {
        tokens_generated: 42,
        ms_per_token: 15.5,
        total_ms: 651.0,
        tokens_per_sec: 64.5,
        hit_eos: true,
    };

    let display = format!("{}", stats);
    assert!(display.contains("42 tokens"));
    assert!(display.contains("651.0ms"));
    assert!(display.contains("64.5 tok/s"));
}

#[test]
fn test_generation_stats_default() {
    let stats = GenerationStats::default();
    assert_eq!(stats.tokens_generated, 0);
    assert_eq!(stats.ms_per_token, 0.0);
    assert_eq!(stats.total_ms, 0.0);
    assert_eq!(stats.tokens_per_sec, 0.0);
    assert!(!stats.hit_eos);
}

//! Component-level timing analysis
//!
//! Profiles each decoder component to identify bottlenecks.

use std::fs;
use std::time::Instant;

fn main() {
    let data = fs::read("models/whisper-tiny-int8.apr").expect("read");
    let mut model = whisper_apr::WhisperApr::load_from_apr(&data).expect("load");

    // Get config info
    let config = model.config();
    let d_model = config.n_text_state as usize;
    let n_vocab = config.n_vocab as usize;

    println!("Model config:");
    println!("  d_model: {d_model}");
    println!("  n_vocab: {n_vocab}");
    println!("  n_layers: {}", config.n_text_layer);

    // Create encoder output
    let audio: Vec<f32> = (0..16000)
        .map(|i| ((i as f32) * 0.01).sin() * 0.3)
        .collect();
    let mel = model.compute_mel(&audio).expect("mel");
    let encoder_output = model.encoder_mut().forward_mel(&mel).expect("encode");

    let enc_len = encoder_output.len() / d_model;
    println!("  encoder_len: {enc_len}");

    // Time vocab projection (isolated)
    let hidden = vec![0.1_f32; d_model]; // 1 token hidden state
    let embedding_t = model.decoder_mut().token_embedding().to_vec();

    // Warmup
    for _ in 0..3 {
        let _ = whisper_apr::simd::matmul(&hidden, &embedding_t, 1, d_model, n_vocab);
    }

    // Time vocab projection
    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = whisper_apr::simd::matmul(&hidden, &embedding_t, 1, d_model, n_vocab);
    }
    let vocab_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("\n=== COMPONENT TIMING ===");
    println!("Vocab projection (1 × {d_model} @ {d_model} × {n_vocab}): {vocab_time:.1}ms");

    // Time full decoder forward vs forward_one
    let tokens = vec![50258_u32];

    // Warmup batch forward
    for _ in 0..3 {
        let _ = model.decoder_mut().forward(&tokens, &encoder_output);
    }

    // Time batch forward
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model
            .decoder_mut()
            .forward(&tokens, &encoder_output)
            .expect("dec");
    }
    let batch_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("Batch forward (1 token): {batch_time:.1}ms");

    // Time forward_one with fresh cache
    let mut cache = model.decoder_mut().create_kv_cache();

    // Warmup forward_one
    for _ in 0..3 {
        cache.clear();
        let _ = model
            .decoder_mut()
            .forward_one(50258, &encoder_output, &mut cache);
    }

    // Time forward_one first token
    cache.clear();
    let start = Instant::now();
    let _ = model
        .decoder_mut()
        .forward_one(50258, &encoder_output, &mut cache)
        .expect("dec");
    let first_token_time = start.elapsed().as_secs_f64() * 1000.0;
    println!("forward_one (first token): {first_token_time:.1}ms");

    // Time subsequent tokens (cached)
    let start = Instant::now();
    for _ in 0..(iterations - 1) {
        let _ = model
            .decoder_mut()
            .forward_one(50259, &encoder_output, &mut cache)
            .expect("dec");
    }
    let subsequent_time = start.elapsed().as_secs_f64() * 1000.0 / (iterations - 1) as f64;
    println!("forward_one (subsequent): {subsequent_time:.1}ms");

    // Analysis
    println!("\n=== ANALYSIS ===");
    println!(
        "Vocab projection: {:.1}% of batch forward",
        vocab_time / batch_time * 100.0
    );
    println!(
        "Cross-attn cache benefit: {:.1}ms saved ({:.1}x)",
        first_token_time - subsequent_time,
        first_token_time / subsequent_time
    );
}

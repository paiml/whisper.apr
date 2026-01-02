//! Single token decode benchmark
//!
//! Measures per-token latency with and without KV caching.

use std::fs;
use std::time::Instant;

fn main() {
    let model_path = "models/whisper-tiny-int8.apr";

    println!("Loading model...");
    let data = fs::read(model_path).expect("read model");
    let mut model = whisper_apr::WhisperApr::load_from_apr(&data).expect("load model");

    // Create encoder output (simulate 1s audio)
    let audio: Vec<f32> = (0..16000)
        .map(|i| ((i as f32) * 0.01).sin() * 0.3)
        .collect();
    let mel = model.compute_mel(&audio).expect("mel");

    let enc_start = Instant::now();
    let encoder_output = model.encoder_mut().forward_mel(&mel).expect("encode");
    let enc_time = enc_start.elapsed();
    println!("Encoder time: {:.1}ms", enc_time.as_secs_f64() * 1000.0);

    // Single token benchmark (batch forward)
    let single_token = vec![50258_u32];

    // Warmup
    for _ in 0..3 {
        let _ = model.decoder_mut().forward(&single_token, &encoder_output);
    }

    // Measure batch forward (1 token)
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model
            .decoder_mut()
            .forward(&single_token, &encoder_output)
            .expect("decode");
    }
    let total = start.elapsed();
    let avg = total.as_secs_f64() * 1000.0 / iterations as f64;

    println!("\n=== SINGLE TOKEN DECODE (batch forward) ===");
    println!("Iterations: {iterations}");
    println!("Total time: {:.1}ms", total.as_secs_f64() * 1000.0);
    println!("Average per token: {:.1}ms", avg);

    // Also measure with KV cache (forward_one)
    let mut cache = model.decoder_mut().create_kv_cache();

    // Warmup
    for _ in 0..3 {
        cache.clear();
        let _ = model
            .decoder_mut()
            .forward_one(50258, &encoder_output, &mut cache);
    }

    // Measure forward_one (first token)
    cache.clear();
    let start = Instant::now();
    let _ = model
        .decoder_mut()
        .forward_one(50258, &encoder_output, &mut cache)
        .expect("decode");
    let first_token_time = start.elapsed();

    // Measure forward_one (subsequent tokens)
    let start = Instant::now();
    for _ in 1..iterations {
        let _ = model
            .decoder_mut()
            .forward_one(50259, &encoder_output, &mut cache)
            .expect("decode");
    }
    let subsequent_total = start.elapsed();
    let subsequent_avg = subsequent_total.as_secs_f64() * 1000.0 / (iterations - 1) as f64;

    println!("\n=== WITH KV CACHE (forward_one) ===");
    println!(
        "First token: {:.1}ms",
        first_token_time.as_secs_f64() * 1000.0
    );
    println!("Subsequent tokens avg: {:.1}ms", subsequent_avg);

    println!("\n=== FIRST TOKEN LATENCY ===");
    let first_token_latency =
        enc_time.as_secs_f64() * 1000.0 + first_token_time.as_secs_f64() * 1000.0;
    println!("Encoder: {:.1}ms", enc_time.as_secs_f64() * 1000.0);
    println!(
        "First decode: {:.1}ms",
        first_token_time.as_secs_f64() * 1000.0
    );
    println!("TOTAL: {:.1}ms", first_token_latency);

    println!("\n=== SPEEDUP ===");
    println!("KV cache speedup: {:.2}x", avg / subsequent_avg);
}

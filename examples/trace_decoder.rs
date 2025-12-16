#![allow(clippy::unwrap_used)]
//! Trace decoder internals step by step
//!
//! Dumps intermediate values at each decoder layer to identify divergence.
//!
//! Usage:
//!   cargo run --example trace_decoder

use std::cell::RefCell;
use whisper_apr::model::DecoderKVCache;
use whisper_apr::tokenizer::special_tokens;
use whisper_apr::WhisperApr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   DECODER INTERNALS TRACE                                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Load model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    // Load audio and compute mel
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    // Compute mel
    let mel = model.compute_mel(&samples)?;
    println!("=== Mel Spectrogram ===");
    println!("  Shape: {} values ({} frames x 80 mels)", mel.len(), mel.len() / 80);
    let mel_stats = compute_stats(&mel);
    println!("  Stats: min={:.4}, max={:.4}, mean={:.6}, std={:.4}",
             mel_stats.0, mel_stats.1, mel_stats.2, mel_stats.3);

    // Encode
    let encoded = model.encode(&mel)?;
    println!("\n=== Encoder Output ===");
    println!("  Shape: {} values ({} frames x 384 dim)", encoded.len(), encoded.len() / 384);
    let enc_stats = compute_stats(&encoded);
    println!("  Stats: min={:.4}, max={:.4}, mean={:.6}, std={:.4}",
             enc_stats.0, enc_stats.1, enc_stats.2, enc_stats.3);
    println!("  First 10: {:?}", &encoded[..10.min(encoded.len())]);

    // Check encoder for issues
    let enc_nan = encoded.iter().filter(|x| x.is_nan()).count();
    let enc_inf = encoded.iter().filter(|x| x.is_infinite()).count();
    let enc_zero = encoded.iter().filter(|&&x| x == 0.0).count();
    println!("  NaN: {}, Inf: {}, Zero: {}", enc_nan, enc_inf, enc_zero);

    // Now trace decoder
    println!("\n=== Decoder Trace ===");

    let n_vocab = 51865;
    let max_tokens = 448;
    let d_model = 384;
    let n_layers = 4;

    // Initial tokens
    let initial_tokens = vec![
        special_tokens::SOT,           // 50257
        special_tokens::LANG_BASE,     // 50258 (English)
        special_tokens::TRANSCRIBE,    // 50358
        special_tokens::NO_TIMESTAMPS, // 50362
    ];

    println!("\n  Initial tokens: {:?}", initial_tokens);

    let cache = RefCell::new(DecoderKVCache::new(n_layers, d_model, max_tokens));

    // Process each initial token and trace
    for (idx, &token) in initial_tokens.iter().enumerate() {
        println!("\n  --- Token {} (id={}) ---", idx, token);

        let logits = model
            .decoder_mut()
            .forward_one(token, &encoded, &mut cache.borrow_mut())?;

        let logit_stats = compute_stats(&logits);
        println!("  Logits: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
                 logit_stats.0, logit_stats.1, logit_stats.2, logit_stats.3);

        // Find top tokens
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("  Top 5: {:?}",
                 indexed.iter().take(5).map(|(t, l)| (t, format!("{:.2}", l))).collect::<Vec<_>>());
    }

    // Now generate one more token after initial sequence
    println!("\n  --- After initial sequence, next token prediction ---");

    let logits = model
        .decoder_mut()
        .forward_one(initial_tokens[initial_tokens.len() - 1], &encoded, &mut cache.borrow_mut())?;

    println!("\n  Full logits analysis:");
    let logit_stats = compute_stats(&logits);
    println!("  Stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
             logit_stats.0, logit_stats.1, logit_stats.2, logit_stats.3);

    // Check expected tokens
    let expected_tokens = [
        (440, " The"),
        (464, "The"),
        (220, " "),
        (383, " the"),
    ];

    println!("\n  Expected token positions:");
    for (token, name) in &expected_tokens {
        println!("    Token {} '{}': logit = {:.4}", token, name, logits[*token]);
    }

    // Distribution analysis
    println!("\n  Logit distribution:");
    let mut sorted_logits: Vec<f32> = logits.clone();
    sorted_logits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!("    Percentiles:");
    println!("      1%:   {:.4}", sorted_logits[n_vocab / 100]);
    println!("      10%:  {:.4}", sorted_logits[n_vocab / 10]);
    println!("      50%:  {:.4}", sorted_logits[n_vocab / 2]);
    println!("      90%:  {:.4}", sorted_logits[n_vocab * 9 / 10]);
    println!("      99%:  {:.4}", sorted_logits[n_vocab * 99 / 100]);

    // Top 20 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n  Top 20 tokens:");
    for (i, (token, logit)) in indexed.iter().take(20).enumerate() {
        let token_type = categorize_token(*token as u32);
        println!("    {:2}. token {:5} ({:12}) = {:.4}", i + 1, token, token_type, logit);
    }

    // Check for anomalies
    println!("\n=== Anomaly Detection ===");

    // 1. Are all logits positive?
    let all_positive = sorted_logits[0] > 0.0;
    println!("  All logits positive: {} (min = {:.4})",
             if all_positive { "YES - ANOMALY!" } else { "no" }, sorted_logits[0]);

    // 2. Is variance too low?
    if logit_stats.3 < 2.0 {
        println!("  Low variance: YES - std={:.4} (expected > 2.0)", logit_stats.3);
    }

    // 3. Is mean too high?
    if logit_stats.2.abs() > 10.0 {
        println!("  High mean: YES - mean={:.4} (expected ~0)", logit_stats.2);
    }

    // Check token embeddings
    println!("\n=== Token Embedding Check ===");

    let decoder = model.decoder_mut();
    let emb_len = decoder.d_model() * decoder.n_vocab();
    println!("  Expected embedding size: {} x {} = {}", decoder.n_vocab(), decoder.d_model(), emb_len);

    // Sample some token embeddings
    let tokens_to_check = [0u32, 220, 440, 464, 50257];
    println!("\n  Sample token embeddings (first 5 values):");

    for &token in &tokens_to_check {
        let start = (token as usize) * d_model;
        let emb = decoder.token_embedding();
        let sample: Vec<f32> = emb[start..start + 5].to_vec();
        let emb_slice = &emb[start..start + d_model];
        let stats = compute_stats(emb_slice);
        println!("    Token {:5}: {:?} ... (min={:.4}, max={:.4}, mean={:.4})",
                 token, sample.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>(),
                 stats.0, stats.1, stats.2);
    }

    Ok(())
}

fn compute_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
    (min, max, mean, std)
}

fn categorize_token(token: u32) -> &'static str {
    match token {
        0..=255 => "byte",
        256..=50256 => "BPE",
        50257 => "SOT",
        50258..=50357 => "lang",
        50358 => "transcribe",
        50359 => "translate",
        50360 => "prev",
        50361 => "no_speech",
        50362 => "no_timestamps",
        50363 => "speaker_turn",
        50364..=51864 => "timestamp",
        _ => "unknown",
    }
}

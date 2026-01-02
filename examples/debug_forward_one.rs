//! Debug: Compare forward() vs forward_one() outputs

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FORWARD VS FORWARD_ONE DEBUG ===\n");

    let model_path = Path::new("models/whisper-tiny-int8.apr");
    let model_bytes = std::fs::read(model_path)?;

    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load a test audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    // Get mel and encoder output
    let mel = model.compute_mel(&samples)?;
    let encoded = model.encode(&mel)?;

    use whisper_apr::tokenizer::special_tokens;
    let lang_en = special_tokens::LANG_BASE;
    let initial_tokens = vec![
        special_tokens::SOT,
        lang_en,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    println!("Initial tokens: {:?}\n", initial_tokens);

    // Test 1: Use forward() (batch mode)
    println!("=== TEST 1: forward() (batch mode) ===\n");
    let batch_logits = model.decoder_mut().forward(&initial_tokens, &encoded, None)?;
    let n_vocab = 51865;
    let seq_len = initial_tokens.len();
    let last_logits_start = (seq_len - 1) * n_vocab;
    let batch_last_logits = &batch_logits[last_logits_start..last_logits_start + n_vocab];

    // Find argmax
    let batch_argmax = batch_last_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("assertion"))
        .map(|(idx, val)| (idx, *val))
        .expect("assertion");
    println!(
        "Batch forward argmax: token {} (logit {:.4})",
        batch_argmax.0, batch_argmax.1
    );

    // Test 2: Use forward_one() with KV cache
    println!("\n=== TEST 2: forward_one() with KV cache ===\n");

    let mut cache = model.decoder_mut().create_kv_cache();
    let mut final_logits: Option<Vec<f32>> = None;

    for (i, &token) in initial_tokens.iter().enumerate() {
        println!("  Processing token {}: {}", i, token);
        let logits = model
            .decoder_mut()
            .forward_one(token, &encoded, &mut cache, None)?;

        // Check logits stats
        let sum: f32 = logits.iter().sum();
        let mean = sum / logits.len() as f32;
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let nans = logits.iter().filter(|&&x| x.is_nan()).count();
        let infs = logits.iter().filter(|&&x| x.is_infinite()).count();

        // Find argmax
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("assertion"))
            .map(|(idx, val)| (idx, *val))
            .expect("assertion");

        println!(
            "    Logits: len={}, mean={:.4}, min={:.4}, max={:.4}, NaNs={}, Infs={}",
            logits.len(),
            mean,
            min,
            max,
            nans,
            infs
        );
        println!("    Argmax: token {} (logit {:.4})", argmax.0, argmax.1);
        println!("    Token 11 logit: {:.4}", logits[11]);

        final_logits = Some(logits);
    }

    // Compare final logits
    println!("\n=== COMPARISON ===\n");

    let cache_last_logits = final_logits.expect("assertion");
    let cache_argmax = cache_last_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("assertion"))
        .map(|(idx, val)| (idx, *val))
        .expect("assertion");

    println!(
        "Batch forward argmax: token {} (logit {:.4})",
        batch_argmax.0, batch_argmax.1
    );
    println!(
        "Cache forward_one argmax: token {} (logit {:.4})",
        cache_argmax.0, cache_argmax.1
    );

    // Check first 20 logits
    println!("\nFirst 20 logits comparison:");
    println!("  Batch:  {:?}", &batch_last_logits[..20]);
    println!("  Cache:  {:?}", &cache_last_logits[..20]);

    // Check token 11 specifically
    println!("\nToken 11:");
    println!("  Batch:  {:.4}", batch_last_logits[11]);
    println!("  Cache:  {:.4}", cache_last_logits[11]);

    // Check max diff
    let max_diff: f32 = batch_last_logits
        .iter()
        .zip(cache_last_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("\nMax difference between batch and cache: {:.6}", max_diff);

    Ok(())
}

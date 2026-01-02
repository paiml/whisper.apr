//! Trace hidden state through decoder layers
//!
//! Identifies where the positive bias is introduced.

use std::cell::RefCell;
use whisper_apr::model::DecoderKVCache;
use whisper_apr::tokenizer::special_tokens;
use whisper_apr::WhisperApr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HIDDEN STATE TRACE ===\n");

    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    // Load audio and compute encoder output
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples)?;
    let encoded = model.encode(&mel)?;

    println!(
        "Encoder output: mean={:.4}, std={:.4}",
        stats_mean(&encoded),
        stats_std(&encoded)
    );

    let n_vocab = 51865;
    let max_tokens = 448;
    let d_model = 384;
    let n_layers = 4;

    // Process initial tokens to populate cache
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    let cache = RefCell::new(DecoderKVCache::new(n_layers, d_model, max_tokens));

    for &token in &initial_tokens {
        let _ = model
            .decoder_mut()
            .forward_one(token, &encoded, &mut cache.borrow_mut(), None)?;
    }

    // Now trace what happens with token 50362 (the last initial token)
    // by manually stepping through the decoder

    let decoder = model.decoder_mut();

    // 1. Get token embedding
    let token = 50362u32;
    let pos = cache.borrow().seq_len();

    let emb_start = (token as usize) * d_model;
    let token_emb = &decoder.token_embedding()[emb_start..emb_start + d_model];
    println!("\n1. Token embedding (token={}):", token);
    println!(
        "   mean={:.4}, min={:.4}, max={:.4}, L2={:.4}",
        stats_mean(token_emb),
        stats_min(token_emb),
        stats_max(token_emb),
        l2_norm(token_emb)
    );

    // 2. Get positional embedding
    let pos_emb = decoder.positional_embedding();
    let pos_start = pos * d_model;
    let pos_slice = &pos_emb[pos_start..pos_start + d_model];
    println!("\n2. Positional embedding (pos={}):", pos);
    println!(
        "   mean={:.4}, min={:.4}, max={:.4}, L2={:.4}",
        stats_mean(pos_slice),
        stats_min(pos_slice),
        stats_max(pos_slice),
        l2_norm(pos_slice)
    );

    // 3. Sum: x = token_emb + pos_emb
    let mut x: Vec<f32> = token_emb
        .iter()
        .zip(pos_slice.iter())
        .map(|(t, p)| t + p)
        .collect();
    println!("\n3. After token + positional:");
    println!(
        "   mean={:.4}, min={:.4}, max={:.4}, L2={:.4}",
        stats_mean(&x),
        stats_min(&x),
        stats_max(&x),
        l2_norm(&x)
    );

    // The decoder blocks are private, so we can't trace through them directly
    // But we can look at what the forward function produces
    println!("\n4. Decoder block outputs (not directly accessible)");
    println!("   Would need to add tracing to decoder.rs");

    // 5. Check final layer norm weights
    println!("\n5. Final layer norm weights:");
    // These are also private, but we can infer from behavior

    // Let's compute what x would need to be before ln_post to produce
    // a hidden state with mean ~4 (which would give logit mean ~23)
    //
    // LayerNorm: y = gamma * (x - mean) / std + beta
    // If gamma ≈ 1 and beta ≈ 0, then y ≈ (x - mean) / std
    // For y to have mean 4, we need... it depends on gamma/beta

    println!("\n=== Analysis ===\n");

    // The issue is likely in the decoder blocks or cross-attention
    // Let me check if the encoder-decoder cross-attention is working correctly

    println!("Key observation: Initial embedding has mean near 0, but");
    println!("after decoder blocks, hidden state has large positive mean.");
    println!();
    println!("Possible issues:");
    println!("1. Cross-attention adding constant offset");
    println!("2. FFN bias terms accumulating");
    println!("3. LayerNorm gamma/beta shifting values");
    println!("4. Residual connections not balanced");

    // Check: what is the sum of all FFN bias terms?
    // In transformer: residual + attn + residual + ffn = x
    // If FFN has positive bias, it accumulates through layers

    Ok(())
}

fn stats_mean(x: &[f32]) -> f32 {
    x.iter().sum::<f32>() / x.len() as f32
}

fn stats_min(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::INFINITY, f32::min)
}

fn stats_max(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

fn stats_std(x: &[f32]) -> f32 {
    let mean = stats_mean(x);
    (x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32).sqrt()
}

fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

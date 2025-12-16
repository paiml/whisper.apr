//! Debug: Capture actual cross-attention weights during inference
//!
//! This runs actual inference and inspects what the attention looks like

use std::path::Path;

fn stats(data: &[f32]) -> (f32, f32, f32, f32) {
    let n = data.len() as f32;
    let sum: f32 = data.iter().sum();
    let mean = sum / n;
    let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min, max)
}

fn entropy(probs: &[f32]) -> f32 {
    -probs.iter().filter(|&&p| p > 1e-10).map(|&p| p * p.ln()).sum::<f32>()
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LIVE CROSS-ATTENTION DEBUG ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    let model_bytes = std::fs::read(model_path)?;
    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load test audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    // Get encoder output
    let mel = model.compute_mel(&samples)?;
    let encoded = model.encode(&mel)?;

    let d_model = 384;
    let n_heads = 6;
    let d_head = d_model / n_heads;
    let enc_len = encoded.len() / d_model;

    println!("Encoder output: {} timesteps x {} dims\n", enc_len, d_model);

    // Get initial tokens
    use whisper_apr::tokenizer::special_tokens;
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE, // English
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    println!("Initial tokens: {:?}", initial_tokens);

    // Create KV cache and run one forward pass
    let mut cache = model.decoder_mut().create_kv_cache();

    // Process each initial token
    for &token in &initial_tokens {
        let _ = model.decoder_mut().forward_one(token, &encoded, &mut cache)?;
    }

    println!("\nAfter processing {} initial tokens:", initial_tokens.len());

    // Now let's manually compute cross-attention for the next token
    // Get the decoder hidden state after the last token
    let decoder = model.decoder_mut();
    let blocks = decoder.blocks();
    let block0 = &blocks[0];

    // Get the W_q, W_k projections
    let w_q = &block0.cross_attn.w_q().weight;
    let w_k = &block0.cross_attn.w_k().weight;
    let scale = block0.cross_attn.scale();

    // Compute K = encoder @ W_k for all encoder positions
    let mut k_all = vec![0.0f32; enc_len * d_model];
    for t in 0..enc_len {
        let enc_t = &encoded[t * d_model..(t + 1) * d_model];
        for out_idx in 0..d_model {
            for in_idx in 0..d_model {
                k_all[t * d_model + out_idx] += enc_t[in_idx] * w_k[out_idx * d_model + in_idx];
            }
        }
    }

    // Check K statistics
    let (k_mean, k_std, k_min, k_max) = stats(&k_all);
    println!("\n[K PROJECTION (from encoder)]");
    println!("  mean={:.4}  std={:.4}  min={:.4}  max={:.4}", k_mean, k_std, k_min, k_max);

    // Sample K at different timesteps
    println!("\n  K at different timesteps (head 0 only, first 4 dims):");
    for t in [0, enc_len/4, enc_len/2, 3*enc_len/4, enc_len-1] {
        let k_t = &k_all[t * d_model..t * d_model + 4];
        println!("    t={:4}: [{:.3}, {:.3}, {:.3}, {:.3}]", t, k_t[0], k_t[1], k_t[2], k_t[3]);
    }

    // Now let's check what Q looks like
    // We need to get the decoder hidden state, which requires running through self-attention
    // For now, let's use the token embedding as an approximation
    let _vocab_size = 51865;
    let token_embed = decoder.token_embedding();
    let last_token = initial_tokens[initial_tokens.len() - 1] as usize;

    let decoder_hidden: Vec<f32> = token_embed[last_token * d_model..(last_token + 1) * d_model].to_vec();

    // Compute Q = decoder_hidden @ W_q
    let mut q = vec![0.0f32; d_model];
    for out_idx in 0..d_model {
        for in_idx in 0..d_model {
            q[out_idx] += decoder_hidden[in_idx] * w_q[out_idx * d_model + in_idx];
        }
    }

    let (q_mean, q_std, q_min, q_max) = stats(&q);
    println!("\n[Q PROJECTION (from token embedding)]");
    println!("  mean={:.4}  std={:.4}  min={:.4}  max={:.4}", q_mean, q_std, q_min, q_max);

    // Compute attention scores for head 0
    let q_head0: Vec<f32> = q[0..d_head].to_vec();
    let mut scores = vec![0.0f32; enc_len];
    for t in 0..enc_len {
        let k_t_head0 = &k_all[t * d_model..t * d_model + d_head];
        let dot: f32 = q_head0.iter().zip(k_t_head0).map(|(a, b)| a * b).sum();
        scores[t] = dot * scale;
    }

    let (s_mean, s_std, s_min, s_max) = stats(&scores);
    println!("\n[ATTENTION SCORES (head 0)]");
    println!("  mean={:.4}  std={:.4}  min={:.4}  max={:.4}", s_mean, s_std, s_min, s_max);

    // Apply softmax
    let weights = softmax(&scores);
    let e = entropy(&weights);
    let max_e = (enc_len as f32).ln();

    println!("\n[ATTENTION WEIGHTS]");
    println!("  Entropy: {:.4} / {:.4} = {:.1}%", e, max_e, e/max_e * 100.0);

    // Key insight: compare Q and K variance
    println!("\n=== ROOT CAUSE ANALYSIS ===");
    println!("  Q std:     {:.4}", q_std);
    println!("  K std:     {:.4}", k_std);
    println!("  Score std: {:.4}", s_std);
    println!("  Score range: {:.4}", s_max - s_min);

    if s_std < 0.5 {
        println!("\n  DIAGNOSIS: Score variance too low ({:.4})", s_std);
        println!("  -> Q·K dot products are too similar");
        println!("  -> Softmax produces near-uniform weights");

        // Check if Q is the problem
        if q_std < 0.1 {
            println!("\n  SUSPECT: Q has low variance (std={:.4})", q_std);
            println!("  -> Decoder hidden state may be degenerate");
        }

        // Check if K is the problem
        let k_timestep_variance: f32 = (0..enc_len)
            .map(|t| {
                let k_t = &k_all[t * d_model..t * d_model + d_head];
                let m: f32 = k_t.iter().sum::<f32>() / d_head as f32;
                k_t.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / d_head as f32
            })
            .sum::<f32>() / enc_len as f32;

        println!("\n  K per-timestep variance: {:.4}", k_timestep_variance.sqrt());
    }

    Ok(())
}

//! H16: Compute actual attention entropy during forward pass
//!
//! Verify if attention weights are peaked or uniform when using correct Q/K

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

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn entropy(probs: &[f32]) -> f32 {
    -probs
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f32>()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== H16: ACTUAL ATTENTION ENTROPY ===\n");

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

    println!("Encoder: {} timesteps × {} dims\n", enc_len, d_model);

    // Process initial tokens
    use whisper_apr::tokenizer::special_tokens;
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    let mut cache = model.decoder_mut().create_kv_cache();

    for &token in &initial_tokens {
        let _ = model
            .decoder_mut()
            .forward_one(token, &encoded, &mut cache)?;
    }

    println!("Processed {} initial tokens\n", initial_tokens.len());

    // Now compute attention for the NEXT token position
    // We need to simulate what happens when we decode the first content token

    let decoder = model.decoder_mut();
    let blocks = decoder.blocks();
    let block0 = &blocks[0];

    // Get the hidden state that will be used for the next token
    // This requires simulating the embedding + positional for position 4
    let pos = 4; // Next position after initial tokens

    // Simulate the decoder input for position 4 (we don't know the token yet,
    // but let's use a common word like "The" = 440)
    let test_token = 440u32; // "The"
    let token_embed = decoder.token_embedding();
    let pos_embed = decoder.positional_embedding();

    let mut x: Vec<f32> = token_embed
        [(test_token as usize) * d_model..((test_token as usize) + 1) * d_model]
        .to_vec();
    for (i, x_val) in x.iter_mut().enumerate() {
        *x_val += pos_embed[pos * d_model + i];
    }

    // LayerNorm1
    let normed1 = block0.ln1.forward(&x)?;

    // Self-attention (simplified - use cached K/V)
    let q_self = block0.self_attn.w_q().forward_simd(&normed1, 1)?;

    // Get cached self-attention K/V
    let k_self_cached = cache.self_attn_cache[0].get_key();
    let v_self_cached = cache.self_attn_cache[0].get_value();

    // We need to append the new K/V and compute attention
    let k_new = block0.self_attn.w_k().forward_simd(&normed1, 1)?;
    let v_new = block0.self_attn.w_v().forward_simd(&normed1, 1)?;

    // Combine cached + new K/V
    let mut k_full: Vec<f32> = k_self_cached.to_vec();
    k_full.extend(&k_new);
    let mut v_full: Vec<f32> = v_self_cached.to_vec();
    v_full.extend(&v_new);

    let self_kv_len = k_full.len() / d_model;
    println!("[SELF-ATTENTION for position {}]", pos);
    println!("  Q: {} values (1 × {})", q_self.len(), d_model);
    println!(
        "  K: {} values ({} × {})",
        k_full.len(),
        self_kv_len,
        d_model
    );

    // Compute self-attention scores for head 0
    let scale = 1.0 / (d_head as f32).sqrt();
    let mut self_scores = Vec::with_capacity(self_kv_len);
    for t in 0..self_kv_len {
        let q_head0 = &q_self[0..d_head];
        let k_head0_start = t * d_model;
        let dot: f32 = q_head0
            .iter()
            .zip(&k_full[k_head0_start..k_head0_start + d_head])
            .map(|(a, b)| a * b)
            .sum();
        self_scores.push(dot * scale);
    }

    let self_weights = softmax(&self_scores);
    let self_entropy = entropy(&self_weights);
    let self_max_entropy = (self_kv_len as f32).ln();
    let self_entropy_ratio = self_entropy / self_max_entropy;

    let (ss_mean, ss_std, ss_min, ss_max) = stats(&self_scores);
    println!(
        "  Scores (head 0): mean={:.4}  std={:.4}  range=[{:.2}, {:.2}]",
        ss_mean, ss_std, ss_min, ss_max
    );
    println!(
        "  Entropy: {:.4} / {:.4} = {:.1}%",
        self_entropy,
        self_max_entropy,
        self_entropy_ratio * 100.0
    );

    // Now compute the residual after self-attention
    // Simplified: approximate with V (since softmax is mostly recent token for causal)
    let attn_out_self = block0.self_attn.w_o().forward_simd(&v_new, 1)?;
    let residual1: Vec<f32> = x
        .iter()
        .zip(attn_out_self.iter())
        .map(|(a, b)| a + b)
        .collect();

    // LayerNorm2
    let normed2 = block0.ln2.forward(&residual1)?;
    let (n2_mean, n2_std, _, _) = stats(&normed2);
    println!(
        "\n  LayerNorm2 output: mean={:+.4}  std={:.4}",
        n2_mean, n2_std
    );

    // Cross-attention Q projection
    let q_cross = block0.cross_attn.w_q().forward_simd(&normed2, 1)?;
    let (qc_mean, qc_std, _, _) = stats(&q_cross);
    println!("  Cross-attn Q: mean={:+.4}  std={:.4}", qc_mean, qc_std);

    // Get cached cross-attention K (from encoder)
    let k_cross = cache.cross_attn_cache[0].get_key();
    let (kc_mean, kc_std, _, _) = stats(k_cross);
    println!("  Cross-attn K: mean={:+.4}  std={:.4}", kc_mean, kc_std);

    // Compute cross-attention scores for head 0
    println!("\n[CROSS-ATTENTION for position {}]", pos);

    let mut cross_scores = Vec::with_capacity(enc_len);
    for t in 0..enc_len {
        let q_head0 = &q_cross[0..d_head];
        let k_head0_start = t * d_model;
        let dot: f32 = q_head0
            .iter()
            .zip(&k_cross[k_head0_start..k_head0_start + d_head])
            .map(|(a, b)| a * b)
            .sum();
        cross_scores.push(dot * scale);
    }

    let (cs_mean, cs_std, cs_min, cs_max) = stats(&cross_scores);
    println!(
        "  Scores (head 0): mean={:.4}  std={:.4}  range=[{:.2}, {:.2}]",
        cs_mean, cs_std, cs_min, cs_max
    );

    let cross_weights = softmax(&cross_scores);
    let cross_entropy = entropy(&cross_weights);
    let cross_max_entropy = (enc_len as f32).ln();
    let cross_entropy_ratio = cross_entropy / cross_max_entropy;

    println!(
        "  Entropy: {:.4} / {:.4} = {:.1}%",
        cross_entropy,
        cross_max_entropy,
        cross_entropy_ratio * 100.0
    );

    // Top attended positions
    let mut indexed: Vec<(usize, f32)> = cross_weights.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n  Top-5 attended encoder positions:");
    for (i, (pos, weight)) in indexed.iter().take(5).enumerate() {
        let time_ms = (*pos as f32) * 20.0; // 50 Hz = 20ms per frame
        println!(
            "    {}: pos={} ({:.0}ms) = {:.4}",
            i + 1,
            pos,
            time_ms,
            weight
        );
    }

    // Diagnosis
    println!("\n=== DIAGNOSIS ===");
    if cross_entropy_ratio > 0.95 {
        println!(
            "❌ UNIFORM: Cross-attention entropy = {:.1}% (near max)",
            cross_entropy_ratio * 100.0
        );
        println!("   Decoder is NOT using encoder information effectively");
    } else if cross_entropy_ratio > 0.80 {
        println!(
            "⚠️  WEAK: Cross-attention entropy = {:.1}%",
            cross_entropy_ratio * 100.0
        );
        println!("   Some differentiation but not strong peaks");
    } else {
        println!(
            "✅ PEAKED: Cross-attention entropy = {:.1}%",
            cross_entropy_ratio * 100.0
        );
        println!("   Decoder is attending to specific encoder positions");
    }

    // Check score range
    if cs_std < 0.5 {
        println!("\n   ISSUE: Score std={:.4} is too low", cs_std);
        println!(
            "   Even with healthy Q (std={:.4}) and K (std={:.4})",
            qc_std, kc_std
        );
        println!("   -> Q and K may be nearly orthogonal");
    }

    Ok(())
}

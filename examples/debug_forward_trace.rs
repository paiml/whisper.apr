//! H15: Trace actual forward_one values at each step
//!
//! Compare manual computation (healthy) vs actual forward pass (broken)

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== H15: FORWARD_ONE TRACE ===\n");

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
    let enc_len = encoded.len() / d_model;
    println!("Encoder output: {} timesteps × {} dims", enc_len, d_model);

    // Get initial tokens
    use whisper_apr::tokenizer::special_tokens;
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE, // English
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];
    println!("Initial tokens: {:?}\n", initial_tokens);

    // Create KV cache
    let mut cache = model.decoder_mut().create_kv_cache();

    // Process each initial token and observe the logits
    println!("[PROCESSING INITIAL TOKENS]");
    for (i, &token) in initial_tokens.iter().enumerate() {
        let logits = model
            .decoder_mut()
            .forward_one(token, &encoded, &mut cache)?;
        let (l_mean, l_std, l_min, l_max) = stats(&logits);
        println!(
            "  Token {}: logits mean={:+.4}  std={:.4}  range=[{:.2}, {:.2}]",
            i, l_mean, l_std, l_min, l_max
        );
    }

    // Now let's manually replicate what forward_one does for comparison
    println!("\n[MANUAL TRACE FOR COMPARISON]");

    let decoder = model.decoder_mut();
    let blocks = decoder.blocks();
    let block0 = &blocks[0];

    // Step 1: Get token embedding for last token (NO_TIMESTAMPS = 50362)
    let token = special_tokens::NO_TIMESTAMPS;
    let token_embed = decoder.token_embedding();
    let tok_emb: Vec<f32> =
        token_embed[(token as usize) * d_model..((token as usize) + 1) * d_model].to_vec();
    let (t_mean, t_std, _, _) = stats(&tok_emb);
    println!(
        "  1. Token embed:       mean={:+.6}  std={:.6}",
        t_mean, t_std
    );

    // Step 2: Add positional embedding (position 3, since we've processed 4 tokens)
    let pos = 3; // 0-indexed position of 4th token
    let pos_embed = decoder.positional_embedding();
    let x: Vec<f32> = tok_emb
        .iter()
        .zip(&pos_embed[pos * d_model..(pos + 1) * d_model])
        .map(|(t, p)| t + p)
        .collect();
    let (x_mean, x_std, _, _) = stats(&x);
    println!(
        "  2. + Positional:      mean={:+.6}  std={:.6}",
        x_mean, x_std
    );

    // Step 3: LayerNorm1
    let normed1 = block0.ln1.forward(&x)?;
    let (n1_mean, n1_std, _, _) = stats(&normed1);
    println!(
        "  3. LayerNorm1:        mean={:+.6}  std={:.6}",
        n1_mean, n1_std
    );

    // Step 4: Self-attention Q projection
    let q_self = block0.self_attn.w_q().forward_simd(&normed1, 1)?;
    let (qs_mean, qs_std, _, _) = stats(&q_self);
    println!(
        "  4. Self-attn Q:       mean={:+.6}  std={:.6}",
        qs_mean, qs_std
    );

    // Step 5: Self-attention K projection
    let k_self = block0.self_attn.w_k().forward_simd(&normed1, 1)?;
    let (ks_mean, ks_std, _, _) = stats(&k_self);
    println!(
        "  5. Self-attn K:       mean={:+.6}  std={:.6}",
        ks_mean, ks_std
    );

    // Step 6: Self-attention V projection
    let v_self = block0.self_attn.w_v().forward_simd(&normed1, 1)?;
    let (vs_mean, vs_std, _, _) = stats(&v_self);
    println!(
        "  6. Self-attn V:       mean={:+.6}  std={:.6}",
        vs_mean, vs_std
    );

    // Step 7: After self-attention + residual (approximation)
    // In actual forward, this uses the full cache. Let's simulate with just this token.
    // For single token self-attention, output = V (after softmax of single score)
    let w_o_self = &block0.self_attn.w_o().weight;
    let mut attn_out_self = vec![0.0f32; d_model];
    for out_idx in 0..d_model {
        for in_idx in 0..d_model {
            attn_out_self[out_idx] += v_self[in_idx] * w_o_self[out_idx * d_model + in_idx];
        }
    }
    let residual1: Vec<f32> = x
        .iter()
        .zip(attn_out_self.iter())
        .map(|(a, b)| a + b)
        .collect();
    let (r1_mean, r1_std, _, _) = stats(&residual1);
    println!(
        "  7. Self-attn + res:   mean={:+.6}  std={:.6}",
        r1_mean, r1_std
    );

    // Step 8: LayerNorm2
    let normed2 = block0.ln2.forward(&residual1)?;
    let (n2_mean, n2_std, _, _) = stats(&normed2);
    println!(
        "  8. LayerNorm2:        mean={:+.6}  std={:.6}",
        n2_mean, n2_std
    );

    // Step 9: Cross-attention Q projection - THIS IS THE KEY
    let q_cross = block0.cross_attn.w_q().forward_simd(&normed2, 1)?;
    let (qc_mean, qc_std, _, _) = stats(&q_cross);
    println!(
        "  9. Cross-attn Q:      mean={:+.6}  std={:.6}  <-- KEY",
        qc_mean, qc_std
    );

    // Step 10: Cross-attention K projection (from encoder)
    let k_cross = block0.cross_attn.w_k().forward_simd(&encoded, enc_len)?;
    let (kc_mean, kc_std, _, _) = stats(&k_cross);
    println!(
        " 10. Cross-attn K:      mean={:+.6}  std={:.6}",
        kc_mean, kc_std
    );

    // Compare: is cross-attn Q healthy now?
    println!("\n=== COMPARISON ===");
    println!("  debug_cross_attn_live Q std: 0.0115 (using raw token embed)");
    println!(
        "  Actual forward trace Q std:  {:.6} (using proper hidden state)",
        qc_std
    );

    if qc_std > 0.5 {
        println!(
            "\n  FINDING: Manual trace produces healthy Q (std={:.4})",
            qc_std
        );
        println!("  But live inference had Q std=0.0115");
        println!("  -> Possible KV cache bug or different code path");
    }

    // Let's check what the KV cache contains
    println!("\n[KV CACHE CONTENTS]");
    println!("  Cache seq_len: {}", cache.seq_len());

    // Look at cached K/V from self-attention
    let self_k_cached = cache.self_attn_cache[0].get_key();
    let self_v_cached = cache.self_attn_cache[0].get_value();
    let (sk_mean, sk_std, _, _) = stats(self_k_cached);
    let (sv_mean, sv_std, _, _) = stats(self_v_cached);
    println!(
        "  Self-attn K cache: {} values, mean={:+.4}  std={:.4}",
        self_k_cached.len(),
        sk_mean,
        sk_std
    );
    println!(
        "  Self-attn V cache: {} values, mean={:+.4}  std={:.4}",
        self_v_cached.len(),
        sv_mean,
        sv_std
    );

    // Look at cached K/V from cross-attention
    let cross_k_cached = cache.cross_attn_cache[0].get_key();
    let cross_v_cached = cache.cross_attn_cache[0].get_value();
    let (ck_mean, ck_std, _, _) = stats(cross_k_cached);
    let (cv_mean, cv_std, _, _) = stats(cross_v_cached);
    println!(
        "  Cross-attn K cache: {} values, mean={:+.4}  std={:.4}",
        cross_k_cached.len(),
        ck_mean,
        ck_std
    );
    println!(
        "  Cross-attn V cache: {} values, mean={:+.4}  std={:.4}",
        cross_v_cached.len(),
        cv_mean,
        cv_std
    );

    Ok(())
}

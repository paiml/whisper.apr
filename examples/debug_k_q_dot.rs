//! H19: Analyze Q·K dot products at different positions
//!
//! Why does Q prefer K at padding positions over K at content positions?

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

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if na > 1e-10 && nb > 1e-10 {
        dot / (na * nb)
    } else {
        0.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== H19: Q·K DOT PRODUCT ANALYSIS ===\n");

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
    let scale = 1.0 / (d_head as f32).sqrt();

    println!("Encoder: {} positions × {} dims\n", enc_len, d_model);

    // Process initial tokens to get actual decoder state
    use whisper_apr::tokenizer::special_tokens;
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    let mut cache = model.decoder_mut().create_kv_cache();
    for &token in &initial_tokens {
        let _ = model.decoder_mut().forward_one(token, &encoded, &mut cache)?;
    }

    // Get the cached cross-attention K
    let k_cached = cache.cross_attn_cache[0].get_key().to_vec();

    // Get decoder block for Q computation
    let decoder = model.decoder_mut();
    let blocks = decoder.blocks();
    let block0 = &blocks[0];

    // Compute Q for a new token (let's use "The" = 440)
    let test_token = 440u32;
    let pos = 4;

    let token_embed = decoder.token_embedding();
    let pos_embed = decoder.positional_embedding();

    let mut x: Vec<f32> = token_embed[(test_token as usize) * d_model..((test_token as usize) + 1) * d_model].to_vec();
    for (i, x_val) in x.iter_mut().enumerate() {
        *x_val += pos_embed[pos * d_model + i];
    }

    // Through ln1 and self-attention (simplified)
    let normed1 = block0.ln1.forward(&x)?;
    let q_self = block0.self_attn.w_q().forward_simd(&normed1, 1)?;
    let k_self = block0.self_attn.w_k().forward_simd(&normed1, 1)?;
    let v_self = block0.self_attn.w_v().forward_simd(&normed1, 1)?;

    let attn_out_self = block0.self_attn.w_o().forward_simd(&v_self, 1)?;
    let residual1: Vec<f32> = x.iter().zip(attn_out_self.iter()).map(|(a, b)| a + b).collect();

    // Through ln2 to get cross-attention Q input
    let normed2 = block0.ln2.forward(&residual1)?;
    let q_cross = block0.cross_attn.w_q().forward_simd(&normed2, 1)?;

    println!("[CROSS-ATTENTION Q VECTOR]");
    let (q_mean, q_std, q_min, q_max) = stats(&q_cross);
    println!("  mean={:.4}  std={:.4}  range=[{:.4}, {:.4}]", q_mean, q_std, q_min, q_max);
    let q_norm: f32 = q_cross.iter().map(|&x| x * x).sum::<f32>().sqrt();
    println!("  L2 norm: {:.4}\n", q_norm);

    // Analyze K vectors at different positions
    println!("[K VECTORS AT DIFFERENT POSITIONS]");
    let positions = [0, 37, 74, 100, 500, 750, 1000, 1312, 1400, 1487, 1494, 1499];

    for &enc_pos in &positions {
        if enc_pos < enc_len {
            let k_start = enc_pos * d_model;
            let k_pos = &k_cached[k_start..k_start + d_model];

            let (k_mean, k_std, _, _) = stats(k_pos);
            let k_norm: f32 = k_pos.iter().map(|&x| x * x).sum::<f32>().sqrt();

            // Compute Q·K score for head 0
            let q_head0 = &q_cross[0..d_head];
            let k_head0 = &k_pos[0..d_head];
            let dot: f32 = q_head0.iter().zip(k_head0).map(|(a, b)| a * b).sum();
            let score = dot * scale;

            let cos = cosine_sim(q_head0, k_head0);

            let label = if enc_pos <= 75 { "AUDIO" } else { "PADDING" };
            println!("  Pos {:4} [{}]: score={:+.4}  cos={:+.4}  K_norm={:.4}  K_std={:.4}",
                     enc_pos, label, score, cos, k_norm, k_std);
        }
    }

    // Find positions with highest and lowest scores
    println!("\n[SCORE ANALYSIS]");
    let q_head0 = &q_cross[0..d_head];

    let mut scores: Vec<(usize, f32)> = (0..enc_len)
        .map(|pos| {
            let k_start = pos * d_model;
            let k_head0 = &k_cached[k_start..k_start + d_head];
            let dot: f32 = q_head0.iter().zip(k_head0).map(|(a, b)| a * b).sum();
            (pos, dot * scale)
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top-10 highest scoring positions:");
    for (pos, score) in scores.iter().take(10) {
        let label = if *pos <= 75 { "AUDIO" } else { "PADDING" };
        let time_ms = (*pos as f32) * 20.0;
        println!("    Pos {:4} ({:5.0}ms) [{}]: score={:+.4}", pos, time_ms, label, score);
    }

    println!("\n  Bottom-10 lowest scoring positions:");
    for (pos, score) in scores.iter().rev().take(10) {
        let label = if *pos <= 75 { "AUDIO" } else { "PADDING" };
        let time_ms = (*pos as f32) * 20.0;
        println!("    Pos {:4} ({:5.0}ms) [{}]: score={:+.4}", pos, time_ms, label, score);
    }

    // Score distribution
    let audio_scores: Vec<f32> = scores.iter().filter(|(p, _)| *p <= 75).map(|(_, s)| *s).collect();
    let padding_scores: Vec<f32> = scores.iter().filter(|(p, _)| *p > 75).map(|(_, s)| *s).collect();

    let (a_mean, a_std, a_min, a_max) = stats(&audio_scores);
    let (p_mean, p_std, p_min, p_max) = stats(&padding_scores);

    println!("\n[SCORE DISTRIBUTION BY REGION]");
    println!("  Audio (0-75):     mean={:+.4}  std={:.4}  range=[{:.2}, {:.2}]", a_mean, a_std, a_min, a_max);
    println!("  Padding (76-1499): mean={:+.4}  std={:.4}  range=[{:.2}, {:.2}]", p_mean, p_std, p_min, p_max);

    println!("\n=== DIAGNOSIS ===");
    if p_mean > a_mean {
        println!("  ISSUE: Padding region has HIGHER average score ({:.4} vs {:.4})", p_mean, a_mean);
        println!("  The Q vector is more aligned with padding K than audio K");
    } else {
        println!("  Audio region has higher average score ({:.4} vs {:.4})", a_mean, p_mean);
    }

    Ok(())
}

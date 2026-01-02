//! Debug Decoder Cross-Attention
//!
//! The encoder output differentiates audio from padding (cos=-0.26).
//! But the decoder still hallucinates, suggesting cross-attention prefers padding.
//!
//! This probe checks:
//! 1. Decoder Query vectors
//! 2. Cross-attention scores (Q @ K^T)
//! 3. Attention weights after softmax
//!
//! Run: cargo run --example debug_decoder_cross_attn

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

fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn softmax(data: &mut [f32]) {
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in data.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in data.iter_mut() {
        *x /= sum;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           DECODER CROSS-ATTENTION ANALYSIS                               ║");
    println!("║  Goal: Understand why decoder prefers padding over audio                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Load model
    let model_path = Path::new("models/whisper-tiny-fb.apr");
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

    // Get mel and encoder output
    let mel = model.compute_mel(&samples)?;
    let encoder_output = model.encode(&mel)?;

    let d_model = 384;
    let n_heads = 6;
    let d_head = d_model / n_heads;
    let enc_len = encoder_output.len() / d_model;

    println!("[ENCODER OUTPUT]");
    println!("  Shape: {} positions × {} d_model", enc_len, d_model);

    // Analyze encoder output at key positions
    let audio_pos = 30;
    let padding_pos = 1000;

    let audio_vec = &encoder_output[audio_pos * d_model..(audio_pos + 1) * d_model];
    let padding_vec = &encoder_output[padding_pos * d_model..(padding_pos + 1) * d_model];

    let (a_mean, a_std, _, _) = stats(audio_vec);
    let (p_mean, p_std, _, _) = stats(padding_vec);
    let a_norm = l2_norm(audio_vec);
    let p_norm = l2_norm(padding_vec);

    println!(
        "  Audio pos {}: mean={:+.4}, std={:.4}, L2={:.2}",
        audio_pos, a_mean, a_std, a_norm
    );
    println!(
        "  Padding pos {}: mean={:+.4}, std={:.4}, L2={:.2}",
        padding_pos, p_mean, p_std, p_norm
    );

    // Check decoder's Key projection weights
    println!("\n[DECODER CROSS-ATTENTION WEIGHTS]");
    let decoder = model.decoder_mut();
    let blocks = decoder.blocks();

    if blocks.is_empty() {
        println!("  ERROR: No decoder blocks found!");
        return Ok(());
    }

    // Analyze first decoder block's cross-attention
    let block0 = &blocks[0];
    let cross_attn = &block0.cross_attn;

    // Get K projection weights (transforms encoder output to Keys)
    let k_weight = &cross_attn.w_k().weight;
    let k_bias = &cross_attn.w_k().bias;

    let (kw_mean, kw_std, kw_min, kw_max) = stats(k_weight);
    let (kb_mean, kb_std, kb_min, kb_max) = stats(k_bias);

    println!("\n  K projection (encoder → Keys):");
    println!(
        "    Weight: [{}, {}] mean={:+.5} std={:.5} range=[{:.4},{:.4}]",
        cross_attn.d_model(),
        cross_attn.d_model(),
        kw_mean,
        kw_std,
        kw_min,
        kw_max
    );
    println!(
        "    Bias: [{}] mean={:+.5} std={:.5} range=[{:.4},{:.4}]",
        k_bias.len(),
        kb_mean,
        kb_std,
        kb_min,
        kb_max
    );

    // Compute K for audio and padding positions
    println!("\n[COMPUTING KEYS]");

    // K = encoder_output @ K_weight.T + K_bias
    // For position i: K[i] = encoder_output[i * d_model : (i+1) * d_model] @ K_weight.T + K_bias

    // Audio position key
    let audio_key = compute_key(audio_vec, k_weight, k_bias, d_model);
    // Padding position key
    let padding_key = compute_key(padding_vec, k_weight, k_bias, d_model);

    let audio_key_norm = l2_norm(&audio_key);
    let padding_key_norm = l2_norm(&padding_key);

    println!(
        "  Audio Key (pos {}): L2 norm = {:.4}",
        audio_pos, audio_key_norm
    );
    println!(
        "  Padding Key (pos {}): L2 norm = {:.4}",
        padding_pos, padding_key_norm
    );

    // Now check Q projection and compute a sample Query
    println!("\n[COMPUTING QUERIES]");

    // Get Q projection weights
    let q_weight = &cross_attn.w_q().weight;
    let q_bias = &cross_attn.w_q().bias;

    let (qw_mean, qw_std, _, _) = stats(q_weight);
    let (qb_mean, qb_std, _, _) = stats(q_bias);

    println!("  Q projection (decoder state → Queries):");
    println!("    Weight: mean={:+.5} std={:.5}", qw_mean, qw_std);
    println!("    Bias: mean={:+.5} std={:.5}", qb_mean, qb_std);

    // Get decoder token embedding for SOT token (start of transcript)
    let sot_token = 50258u32;
    let token_embeddings = decoder.token_embedding();
    let sot_embed_start = sot_token as usize * d_model;
    let sot_embed = &token_embeddings[sot_embed_start..sot_embed_start + d_model];

    println!(
        "  SOT token ({}) embedding L2: {:.4}",
        sot_token,
        l2_norm(sot_embed)
    );

    // Compute Query from SOT embedding
    let sot_query = compute_key(sot_embed, q_weight, q_bias, d_model); // Same math as K

    let sot_query_norm = l2_norm(&sot_query);
    println!("  SOT Query L2 norm: {:.4}", sot_query_norm);

    // Compute attention scores: Q @ K^T
    println!("\n[ATTENTION SCORES (Q @ K^T / sqrt(d_head))]");

    let scale = (d_head as f32).sqrt();

    // Score for audio position
    let audio_score: f32 = sot_query
        .iter()
        .zip(audio_key.iter())
        .map(|(q, k)| q * k)
        .sum::<f32>()
        / scale;

    // Score for padding position
    let padding_score: f32 = sot_query
        .iter()
        .zip(padding_key.iter())
        .map(|(q, k)| q * k)
        .sum::<f32>()
        / scale;

    println!("  Score(Q_sot, K_audio[{}]): {:.4}", audio_pos, audio_score);
    println!(
        "  Score(Q_sot, K_padding[{}]): {:.4}",
        padding_pos, padding_score
    );

    // Compute scores for a range of positions
    println!("\n[ATTENTION SCORE DISTRIBUTION]");

    let mut all_scores = Vec::with_capacity(enc_len);
    for pos in 0..enc_len {
        let enc_slice = &encoder_output[pos * d_model..(pos + 1) * d_model];
        let key = compute_key(enc_slice, k_weight, k_bias, d_model);
        let score: f32 = sot_query
            .iter()
            .zip(key.iter())
            .map(|(q, k)| q * k)
            .sum::<f32>()
            / scale;
        all_scores.push(score);
    }

    // Statistics for audio region (0-75) and padding region (1000-1500)
    let audio_scores: Vec<f32> = all_scores[..75].to_vec();
    let padding_scores: Vec<f32> = all_scores[1000..].to_vec();

    let (as_mean, as_std, as_min, as_max) = stats(&audio_scores);
    let (ps_mean, ps_std, ps_min, ps_max) = stats(&padding_scores);

    println!("  Audio region (0-75):");
    println!(
        "    Scores: mean={:.4} std={:.4} range=[{:.4},{:.4}]",
        as_mean, as_std, as_min, as_max
    );

    println!("  Padding region (1000-1500):");
    println!(
        "    Scores: mean={:.4} std={:.4} range=[{:.4},{:.4}]",
        ps_mean, ps_std, ps_min, ps_max
    );

    // Apply softmax to see attention weights
    println!("\n[ATTENTION WEIGHTS (softmax)]");

    let mut weights = all_scores.clone();
    softmax(&mut weights);

    let audio_weight_sum: f32 = weights[..75].iter().sum();
    let padding_weight_sum: f32 = weights[75..].iter().sum();

    println!(
        "  Total attention to audio region (0-75):    {:.4} ({:.1}%)",
        audio_weight_sum,
        audio_weight_sum * 100.0
    );
    println!(
        "  Total attention to padding region (75+):   {:.4} ({:.1}%)",
        padding_weight_sum,
        padding_weight_sum * 100.0
    );

    // Find top-5 attended positions
    let mut indexed_weights: Vec<(usize, f32)> = weights.iter().cloned().enumerate().collect();
    indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n  Top-5 attended positions:");
    for (pos, weight) in indexed_weights.iter().take(5) {
        let region = if *pos < 75 { "AUDIO" } else { "PADDING" };
        println!("    Position {}: {:.4}% [{}]", pos, weight * 100.0, region);
    }

    // Diagnosis
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         DIAGNOSIS                                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    if padding_weight_sum > audio_weight_sum {
        println!("  ✗ CROSS-ATTENTION PREFERS PADDING");
        println!("    Audio attention: {:.1}%", audio_weight_sum * 100.0);
        println!("    Padding attention: {:.1}%", padding_weight_sum * 100.0);
        println!("");
        println!("    Possible causes:");
        println!("    1. Padding scores are systematically higher");
        println!("    2. More padding positions → more total weight");
        println!("    3. Softmax temperature/scaling issue");
    } else {
        println!("  ✓ CROSS-ATTENTION PREFERS AUDIO");
        println!("    Audio attention: {:.1}%", audio_weight_sum * 100.0);
        println!("    Padding attention: {:.1}%", padding_weight_sum * 100.0);
    }

    Ok(())
}

/// Compute linear projection: out = input @ weight.T + bias
fn compute_key(input: &[f32], weight: &[f32], bias: &[f32], d_model: usize) -> Vec<f32> {
    let mut out = bias.to_vec();
    // weight is [d_model, d_model] stored row-major
    // out[i] = sum_j(input[j] * weight[i * d_model + j]) + bias[i]
    for i in 0..d_model {
        for j in 0..d_model {
            out[i] += input[j] * weight[i * d_model + j];
        }
    }
    out
}

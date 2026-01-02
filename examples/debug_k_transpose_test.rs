//! H34: Transpose Inversion Hypothesis Test
//!
//! Tests if transposing the K projection weights fixes the attention pattern.
//!
//! Run: cargo run --example debug_k_transpose_test

use std::path::Path;

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

/// Compute linear projection: out = input @ weight.T + bias
/// Weight is [out_features, in_features] (row-major)
fn linear_normal(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut out = bias.to_vec();
    for o in 0..d_out {
        for i in 0..d_in {
            out[o] += input[i] * weight[o * d_in + i];
        }
    }
    out
}

/// Compute linear projection with TRANSPOSED weight
/// If original weight is [out, in], treat it as [in, out] instead
/// This means: out[o] = bias[o] + sum_i(input[i] * weight[i * d_out + o])
fn linear_transposed(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut out = bias.to_vec();
    for o in 0..d_out {
        for i in 0..d_in {
            // Transposed access: instead of weight[o,i] use weight[i,o]
            out[o] += input[i] * weight[i * d_out + o];
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     H34: TRANSPOSE INVERSION HYPOTHESIS TEST                     ║");
    println!("║     Does transposing K weights fix attention?                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

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

    let mel = model.compute_mel(&samples)?;
    let encoder_output = model.encode(&mel)?;

    let d_model = 384;
    let d_head = 64;
    let enc_len = encoder_output.len() / d_model;
    let scale = (d_head as f32).sqrt();

    // Get decoder cross-attention weights
    let decoder = model.decoder_mut();
    let token_embed = decoder.token_embedding().to_vec();
    let blocks = decoder.blocks();
    let cross_attn = &blocks[0].cross_attn;

    let q_weight = cross_attn.w_q().weight.clone();
    let q_bias = cross_attn.w_q().bias.clone();
    let k_weight = cross_attn.w_k().weight.clone();
    let k_bias = cross_attn.w_k().bias.clone();

    // SOT token embedding
    let sot_token = 50258u32;
    let sot_embed_start = sot_token as usize * d_model;
    let sot_embed = &token_embed[sot_embed_start..sot_embed_start + d_model];

    // Compute Query (same for both tests)
    let query = linear_normal(sot_embed, &q_weight, &q_bias, d_model, d_model);

    println!("[QUERY]");
    println!("  Query L2: {:.4}", l2_norm(&query));

    // ================================================================
    // TEST 1: Normal K projection (current behavior)
    // ================================================================
    println!("\n[TEST 1: NORMAL K PROJECTION (current)]");

    let mut normal_scores = Vec::with_capacity(enc_len);
    for pos in 0..enc_len {
        let enc_slice = &encoder_output[pos * d_model..(pos + 1) * d_model];
        let key = linear_normal(enc_slice, &k_weight, &k_bias, d_model, d_model);
        let score: f32 = query
            .iter()
            .zip(key.iter())
            .map(|(q, k)| q * k)
            .sum::<f32>()
            / scale;
        normal_scores.push(score);
    }

    let mut normal_weights = normal_scores.clone();
    softmax(&mut normal_weights);

    let normal_audio_attn: f32 = normal_weights[..75].iter().sum();
    let normal_padding_attn: f32 = normal_weights[75..].iter().sum();

    println!(
        "  Audio attention (0-75):   {:.4} ({:.1}%)",
        normal_audio_attn,
        normal_audio_attn * 100.0
    );
    println!(
        "  Padding attention (75+):  {:.4} ({:.1}%)",
        normal_padding_attn,
        normal_padding_attn * 100.0
    );

    // Top positions
    let mut indexed: Vec<(usize, f32)> = normal_weights.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("  Top-3 positions:");
    for (pos, weight) in indexed.iter().take(3) {
        let region = if *pos < 75 { "AUDIO" } else { "PADDING" };
        println!("    Position {}: {:.4}% [{}]", pos, weight * 100.0, region);
    }

    // ================================================================
    // TEST 2: Transposed K projection (H34 hypothesis)
    // ================================================================
    println!("\n[TEST 2: TRANSPOSED K PROJECTION (H34)]");

    let mut transposed_scores = Vec::with_capacity(enc_len);
    for pos in 0..enc_len {
        let enc_slice = &encoder_output[pos * d_model..(pos + 1) * d_model];
        let key = linear_transposed(enc_slice, &k_weight, &k_bias, d_model, d_model);
        let score: f32 = query
            .iter()
            .zip(key.iter())
            .map(|(q, k)| q * k)
            .sum::<f32>()
            / scale;
        transposed_scores.push(score);
    }

    let mut transposed_weights = transposed_scores.clone();
    softmax(&mut transposed_weights);

    let trans_audio_attn: f32 = transposed_weights[..75].iter().sum();
    let trans_padding_attn: f32 = transposed_weights[75..].iter().sum();

    println!(
        "  Audio attention (0-75):   {:.4} ({:.1}%)",
        trans_audio_attn,
        trans_audio_attn * 100.0
    );
    println!(
        "  Padding attention (75+):  {:.4} ({:.1}%)",
        trans_padding_attn,
        trans_padding_attn * 100.0
    );

    // Top positions
    let mut indexed: Vec<(usize, f32)> = transposed_weights.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("  Top-3 positions:");
    for (pos, weight) in indexed.iter().take(3) {
        let region = if *pos < 75 { "AUDIO" } else { "PADDING" };
        println!("    Position {}: {:.4}% [{}]", pos, weight * 100.0, region);
    }

    // ================================================================
    // VERDICT
    // ================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         H34 VERDICT                              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let normal_correct = normal_audio_attn > normal_padding_attn;
    let transposed_correct = trans_audio_attn > trans_padding_attn;

    if !normal_correct && transposed_correct {
        println!("  ✓ H34 CONFIRMED: Transposing K weights FIXES attention!");
        println!(
            "    Normal:     Audio={:.1}%, Padding={:.1}% (WRONG)",
            normal_audio_attn * 100.0,
            normal_padding_attn * 100.0
        );
        println!(
            "    Transposed: Audio={:.1}%, Padding={:.1}% (CORRECT)",
            trans_audio_attn * 100.0,
            trans_padding_attn * 100.0
        );
        println!("\n  ROOT CAUSE: K projection weights are stored WITHOUT transpose.");
        println!("  FIX: Transpose K weights during loading or change forward() indexing.");
    } else if normal_correct && !transposed_correct {
        println!("  ✗ H34 FALSIFIED: Normal K weights are correct!");
        println!(
            "    Normal:     Audio={:.1}%, Padding={:.1}%",
            normal_audio_attn * 100.0,
            normal_padding_attn * 100.0
        );
        println!(
            "    Transposed: Audio={:.1}%, Padding={:.1}%",
            trans_audio_attn * 100.0,
            trans_padding_attn * 100.0
        );
        println!("\n  Issue is NOT weight transpose. Look elsewhere.");
    } else if !normal_correct && !transposed_correct {
        println!("  ⚠ H34 INCONCLUSIVE: Neither version has correct attention");
        println!(
            "    Normal:     Audio={:.1}%, Padding={:.1}%",
            normal_audio_attn * 100.0,
            normal_padding_attn * 100.0
        );
        println!(
            "    Transposed: Audio={:.1}%, Padding={:.1}%",
            trans_audio_attn * 100.0,
            trans_padding_attn * 100.0
        );
        println!("\n  Problem is deeper than weight transpose.");
    } else {
        println!("  ? UNEXPECTED: Both versions have correct attention");
        println!(
            "    Normal:     Audio={:.1}%, Padding={:.1}%",
            normal_audio_attn * 100.0,
            normal_padding_attn * 100.0
        );
        println!(
            "    Transposed: Audio={:.1}%, Padding={:.1}%",
            trans_audio_attn * 100.0,
            trans_padding_attn * 100.0
        );
    }

    // Also check score statistics
    println!("\n[SCORE STATISTICS]");
    let normal_audio_scores: Vec<f32> = normal_scores[..75].to_vec();
    let normal_padding_scores: Vec<f32> = normal_scores[75..].to_vec();
    let trans_audio_scores: Vec<f32> = transposed_scores[..75].to_vec();
    let trans_padding_scores: Vec<f32> = transposed_scores[75..].to_vec();

    let nas_mean: f32 = normal_audio_scores.iter().sum::<f32>() / 75.0;
    let nps_mean: f32 =
        normal_padding_scores.iter().sum::<f32>() / normal_padding_scores.len() as f32;
    let tas_mean: f32 = trans_audio_scores.iter().sum::<f32>() / 75.0;
    let tps_mean: f32 =
        trans_padding_scores.iter().sum::<f32>() / trans_padding_scores.len() as f32;

    println!(
        "  Normal scores:     Audio mean={:+.2}, Padding mean={:+.2}",
        nas_mean, nps_mean
    );
    println!(
        "  Transposed scores: Audio mean={:+.2}, Padding mean={:+.2}",
        tas_mean, tps_mean
    );

    Ok(())
}

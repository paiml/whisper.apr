//! H35: Positional Singularity Hypothesis
//!
//! Decompose the attention score at position 1494 into:
//!   Score = Q · (C_t @ K_w) + Q · (P_t @ K_w) + Q · K_bias
//!
//! where E_t = C_t + P_t (Encoder output = Conv output + Positional embedding)
//!
//! Run: cargo run --example debug_h35_positional_singularity

use std::path::Path;

fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Compute linear projection: out = input @ weight.T + bias
fn linear(input: &[f32], weight: &[f32], bias: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
    let mut out = bias.to_vec();
    for o in 0..d_out {
        for i in 0..d_in {
            out[o] += input[i] * weight[o * d_in + i];
        }
    }
    out
}

/// Compute linear projection WITHOUT bias: out = input @ weight.T
fn linear_no_bias(input: &[f32], weight: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d_out];
    for o in 0..d_out {
        for i in 0..d_in {
            out[o] += input[i] * weight[o * d_in + i];
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     H35: POSITIONAL SINGULARITY HYPOTHESIS                       ║");
    println!("║     Is the attractor in PosEmbed or ConvOutput?                  ║");
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

    // Get conv output and positional embeddings SEPARATELY
    let (conv_output, pos_embed) = {
        let encoder = model.encoder_mut();
        let conv_out = encoder.conv_frontend().forward(&mel)?;
        let pe = encoder.positional_embedding().to_vec();
        (conv_out, pe)
    };

    let d_model = 384;
    let d_head = 64;
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

    // Compute Query
    let query = linear(sot_embed, &q_weight, &q_bias, d_model, d_model);

    println!("[QUERY ANALYSIS]");
    println!("  SOT embedding L2: {:.4}", l2_norm(sot_embed));
    println!("  Query L2: {:.4}", l2_norm(&query));
    println!("  Q_bias L2: {:.4}", l2_norm(&q_bias));

    // Compute Q · K_bias (constant offset for all positions)
    let q_dot_kbias = dot_product(&query, &k_bias);
    println!("  Q · K_bias: {:.4} (constant offset)", q_dot_kbias);

    // Test positions: attractor positions and control positions
    let test_positions = [
        (1494, "ATTRACTOR (end)"),
        (1493, "ATTRACTOR"),
        (1487, "ATTRACTOR"),
        (1000, "MID-PADDING"),
        (100, "EARLY-PADDING"),
        (30, "AUDIO"),
        (0, "AUDIO (start)"),
    ];

    println!("\n[SCORE DECOMPOSITION]");
    println!("  Score = Q · (E @ K_w) / scale");
    println!("        = Q · ((C + P) @ K_w) / scale");
    println!("        = [Q · (C @ K_w) + Q · (P @ K_w) + Q · K_bias] / scale\n");

    println!(
        "  {:>6} | {:>12} | {:>12} | {:>12} | {:>12} | {:>15}",
        "Pos", "Conv Contrib", "PE Contrib", "Bias", "Total Score", "Region"
    );
    println!("  {}", "-".repeat(85));

    for (pos, region) in test_positions {
        let start = pos * d_model;
        let end = start + d_model;

        let conv_vec = &conv_output[start..end];
        let pe_vec = &pos_embed[start..end];

        // Compute K projections for each component separately (without bias)
        let conv_k = linear_no_bias(conv_vec, &k_weight, d_model, d_model);
        let pe_k = linear_no_bias(pe_vec, &k_weight, d_model, d_model);

        // Compute contributions
        let conv_contrib = dot_product(&query, &conv_k);
        let pe_contrib = dot_product(&query, &pe_k);

        // Total score (should equal using full encoder output)
        let total_unscaled = conv_contrib + pe_contrib + q_dot_kbias;
        let total_score = total_unscaled / scale;

        println!(
            "  {:>6} | {:>12.2} | {:>12.2} | {:>12.2} | {:>12.4} | {:>15}",
            pos, conv_contrib, pe_contrib, q_dot_kbias, total_score, region
        );
    }

    // ================================================================
    // VERDICT
    // ================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         H35 ANALYSIS                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Check position 1494 specifically
    let pos = 1494;
    let start = pos * d_model;
    let end = start + d_model;
    let conv_vec = &conv_output[start..end];
    let pe_vec = &pos_embed[start..end];

    let conv_k = linear_no_bias(conv_vec, &k_weight, d_model, d_model);
    let pe_k = linear_no_bias(pe_vec, &k_weight, d_model, d_model);
    let conv_contrib = dot_product(&query, &conv_k);
    let pe_contrib = dot_product(&query, &pe_k);

    // Also check audio position
    let audio_pos = 30;
    let audio_start = audio_pos * d_model;
    let audio_conv_vec = &conv_output[audio_start..audio_start + d_model];
    let audio_pe_vec = &pos_embed[audio_start..audio_start + d_model];
    let audio_conv_k = linear_no_bias(audio_conv_vec, &k_weight, d_model, d_model);
    let audio_pe_k = linear_no_bias(audio_pe_vec, &k_weight, d_model, d_model);
    let audio_conv_contrib = dot_product(&query, &audio_conv_k);
    let audio_pe_contrib = dot_product(&query, &audio_pe_k);

    println!("  Position 1494 (ATTRACTOR):");
    println!("    Conv contribution:  {:+.2}", conv_contrib);
    println!("    PE contribution:    {:+.2}", pe_contrib);
    println!("    Bias contribution:  {:+.2}", q_dot_kbias);

    println!("\n  Position 30 (AUDIO):");
    println!("    Conv contribution:  {:+.2}", audio_conv_contrib);
    println!("    PE contribution:    {:+.2}", audio_pe_contrib);
    println!("    Bias contribution:  {:+.2}", q_dot_kbias);

    println!("\n  [COMPARISON]");
    println!(
        "    1494 Total: {:+.2}",
        conv_contrib + pe_contrib + q_dot_kbias
    );
    println!(
        "    30 Total:   {:+.2}",
        audio_conv_contrib + audio_pe_contrib + q_dot_kbias
    );
    println!(
        "    Difference: {:+.2}",
        (conv_contrib + pe_contrib) - (audio_conv_contrib + audio_pe_contrib)
    );

    // Determine source of attractor
    let pe_diff = pe_contrib - audio_pe_contrib;
    let conv_diff = conv_contrib - audio_conv_contrib;

    println!("\n  [SOURCE OF ATTRACTOR]");
    println!("    PE difference (1494 vs 30):   {:+.2}", pe_diff);
    println!("    Conv difference (1494 vs 30): {:+.2}", conv_diff);

    if pe_diff.abs() > conv_diff.abs() * 2.0 {
        println!("\n  ✓ H35 CONFIRMED: Positional Embedding is the DOMINANT attractor source");
        println!("    The decoder is biased to look at END-OF-SEQUENCE positions.");
        println!("    FIX: Apply attention mask to exclude padding positions.");
    } else if conv_diff.abs() > pe_diff.abs() * 2.0 {
        println!("\n  ⚠ H35 PARTIAL: Convolutional output is the DOMINANT attractor source");
        println!("    Edge padding in convolution creates high-scoring positions.");
        println!("    FIX: Mask padding region OR fix conv edge handling.");
    } else {
        println!("\n  ? H35 MIXED: Both PE and Conv contribute significantly");
        println!("    PE contribution diff:   {:+.2}", pe_diff);
        println!("    Conv contribution diff: {:+.2}", conv_diff);
        println!("    BOTH components need attention/masking fix.");
    }

    // Additional: Check raw vectors
    println!("\n[RAW VECTOR ANALYSIS]");
    println!("  Position 1494:");
    println!(
        "    Conv L2: {:.4}, PE L2: {:.4}",
        l2_norm(conv_vec),
        l2_norm(pe_vec)
    );
    println!(
        "    Conv_K L2: {:.4}, PE_K L2: {:.4}",
        l2_norm(&conv_k),
        l2_norm(&pe_k)
    );
    println!("  Position 30:");
    println!(
        "    Conv L2: {:.4}, PE L2: {:.4}",
        l2_norm(audio_conv_vec),
        l2_norm(audio_pe_vec)
    );
    println!(
        "    Conv_K L2: {:.4}, PE_K L2: {:.4}",
        l2_norm(&audio_conv_k),
        l2_norm(&audio_pe_k)
    );

    Ok(())
}

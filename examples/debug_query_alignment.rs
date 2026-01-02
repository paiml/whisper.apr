//! Debug Decoder Query Alignment
//!
//! Why does decoder Query align with padding encoder output more than audio?
//!
//! Run: cargo run --example debug_query_alignment

use std::path::Path;

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

fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn stats(data: &[f32]) -> (f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Compute linear projection: out = input @ weight.T + bias
fn linear(input: &[f32], weight: &[f32], bias: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
    let mut out = bias.to_vec();
    for i in 0..d_out {
        for j in 0..d_in {
            out[i] += input[j] * weight[i * d_in + j];
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     DECODER QUERY ALIGNMENT ANALYSIS                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    let model_bytes = std::fs::read(model_path)?;
    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load and encode audio
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
    let enc_len = encoder_output.len() / d_model;

    // Get decoder components
    let decoder = model.decoder_mut();
    let token_embed = decoder.token_embedding().to_vec();
    let blocks = decoder.blocks();

    // SOT token embedding
    let sot_token = 50258u32;
    let sot_embed_start = sot_token as usize * d_model;
    let sot_embed = &token_embed[sot_embed_start..sot_embed_start + d_model];

    println!("[SOT TOKEN EMBEDDING]");
    let (sot_mean, sot_std) = stats(sot_embed);
    println!(
        "  SOT ({}) embedding: L2={:.4}, mean={:+.4}, std={:.4}",
        sot_token,
        l2_norm(sot_embed),
        sot_mean,
        sot_std
    );

    // Get cross-attention weights from block 0
    let cross_attn = &blocks[0].cross_attn;
    let q_weight = &cross_attn.w_q().weight;
    let q_bias = &cross_attn.w_q().bias;
    let k_weight = &cross_attn.w_k().weight;
    let k_bias = &cross_attn.w_k().bias;

    println!("\n[CROSS-ATTENTION PROJECTIONS]");
    let (qw_mean, qw_std) = stats(q_weight);
    let (qb_mean, qb_std) = stats(q_bias);
    let (kw_mean, kw_std) = stats(k_weight);
    let (kb_mean, kb_std) = stats(k_bias);

    println!("  Q_weight: mean={:+.5}, std={:.5}", qw_mean, qw_std);
    println!(
        "  Q_bias:   mean={:+.5}, std={:.5}, L2={:.4}",
        qb_mean,
        qb_std,
        l2_norm(q_bias)
    );
    println!("  K_weight: mean={:+.5}, std={:.5}", kw_mean, kw_std);
    println!(
        "  K_bias:   mean={:+.5}, std={:.5}, L2={:.4}",
        kb_mean,
        kb_std,
        l2_norm(k_bias)
    );

    // Compute Query from SOT embedding (simplified - ignoring decoder pos embed and LN for now)
    let query = linear(sot_embed, q_weight, q_bias, d_model, d_model);
    println!("\n[QUERY FROM SOT]");
    let (q_mean, q_std) = stats(&query);
    println!(
        "  Query: L2={:.4}, mean={:+.4}, std={:.4}",
        l2_norm(&query),
        q_mean,
        q_std
    );

    // Sample encoder output positions
    let audio_positions = [0, 10, 20, 30, 40, 50];
    let padding_positions = [1000, 1200, 1400, 1487, 1493, 1494];

    println!("\n[QUERY-KEY ALIGNMENT ANALYSIS]\n");
    println!(
        "  {:>6} | {:>12} | {:>12} | {:>12} | {:>8}",
        "Pos", "cos(Q,K)", "cos(Q,Enc)", "L2(K)", "Region"
    );
    println!("  {}", "-".repeat(65));

    // For each position, compute Key and check alignment with Query
    for &pos in audio_positions.iter().chain(padding_positions.iter()) {
        if pos < enc_len {
            let start = pos * d_model;
            let end = start + d_model;
            let enc_vec = &encoder_output[start..end];

            // Compute Key from encoder output
            let key = linear(enc_vec, k_weight, k_bias, d_model, d_model);

            let cos_q_k = cosine_sim(&query, &key);
            let cos_q_enc = cosine_sim(&query, enc_vec);
            let k_norm = l2_norm(&key);
            let region = if pos < 75 { "AUDIO" } else { "PADDING" };

            println!(
                "  {:>6} | {:>12.4} | {:>12.4} | {:>12.4} | {:>8}",
                pos, cos_q_k, cos_q_enc, k_norm, region
            );
        }
    }

    // Check what direction the Query is pointing
    println!("\n[QUERY DIRECTION ANALYSIS]");

    // Compute average encoder output for audio and padding
    let audio_avg: Vec<f32> = (0..d_model)
        .map(|d| {
            (0..75)
                .filter(|&p| p < enc_len)
                .map(|p| encoder_output[p * d_model + d])
                .sum::<f32>()
                / 75.0
        })
        .collect();

    let padding_avg: Vec<f32> = (0..d_model)
        .map(|d| {
            (1000..1400)
                .filter(|&p| p < enc_len)
                .map(|p| encoder_output[p * d_model + d])
                .sum::<f32>()
                / 400.0
        })
        .collect();

    println!(
        "  cos(Query, Avg Audio Encoder): {:.4}",
        cosine_sim(&query, &audio_avg)
    );
    println!(
        "  cos(Query, Avg Padding Encoder): {:.4}",
        cosine_sim(&query, &padding_avg)
    );

    // Check alignment with Key-projected versions
    let audio_key_avg = linear(&audio_avg, k_weight, k_bias, d_model, d_model);
    let padding_key_avg = linear(&padding_avg, k_weight, k_bias, d_model, d_model);

    println!(
        "  cos(Query, Avg Audio Key): {:.4}",
        cosine_sim(&query, &audio_key_avg)
    );
    println!(
        "  cos(Query, Avg Padding Key): {:.4}",
        cosine_sim(&query, &padding_key_avg)
    );

    // The smoking gun: check if Q and K projections are inverting the relationship
    println!("\n[PROJECTION EFFECT]");
    println!("  Before K projection:");
    println!("    Audio L2: {:.4}", l2_norm(&audio_avg));
    println!("    Padding L2: {:.4}", l2_norm(&padding_avg));
    println!("  After K projection:");
    println!("    Audio Key L2: {:.4}", l2_norm(&audio_key_avg));
    println!("    Padding Key L2: {:.4}", l2_norm(&padding_key_avg));

    Ok(())
}

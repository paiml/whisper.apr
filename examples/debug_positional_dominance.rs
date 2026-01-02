//! H31: Positional Dominance Hypothesis
//!
//! At padding positions, if ConvOut is weak/constant, then:
//!   EncoderOut ≈ PosEmbed
//!
//! This would prove the decoder is attending to WHERE, not WHAT.
//!
//! Run: cargo run --example debug_positional_dominance

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     H31: POSITIONAL DOMINANCE HYPOTHESIS                         ║");
    println!("║     Is EncoderOut ≈ PosEmbed at padding positions?               ║");
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

    // Get conv output and positional embeddings (BEFORE transformer blocks)
    let (conv_output, pos_embed) = {
        let encoder = model.encoder_mut();
        let conv_out = encoder.conv_frontend().forward(&mel)?;
        let pe = encoder.positional_embedding().to_vec();
        (conv_out, pe)
    };

    // Get full encoder output (AFTER transformer blocks)
    let encoder_output = model.encode(&mel)?;

    let d_model = 384;
    let enc_len = encoder_output.len() / d_model;

    println!("[COMPARISON: EncoderOut vs PosEmbed vs ConvOut]\n");
    println!(
        "  {:>6} | {:>12} | {:>12} | {:>12} | {:>12} | {:>8}",
        "Pos", "cos(Enc,PE)", "cos(Conv,PE)", "L2(Enc)", "L2(Conv)", "Region"
    );
    println!("  {}", "-".repeat(85));

    let test_positions = [
        // Audio region
        0, 10, 20, 30, 40, 50, 60, 70, // Padding region
        100, 500, 1000, 1200, 1400, // High-attention positions
        1486, 1487, 1493, 1494, // End positions
        1498, 1499,
    ];

    for &pos in &test_positions {
        if pos < enc_len {
            let start = pos * d_model;
            let end = start + d_model;

            let enc_vec = &encoder_output[start..end];
            let pe_vec = &pos_embed[start..end];
            let conv_vec = &conv_output[start..end];

            let cos_enc_pe = cosine_sim(enc_vec, pe_vec);
            let cos_conv_pe = cosine_sim(conv_vec, pe_vec);
            let enc_norm = l2_norm(enc_vec);
            let conv_norm = l2_norm(conv_vec);

            let region = if pos < 75 { "AUDIO" } else { "PADDING" };

            println!(
                "  {:>6} | {:>12.4} | {:>12.4} | {:>12.4} | {:>12.4} | {:>8}",
                pos, cos_enc_pe, cos_conv_pe, enc_norm, conv_norm, region
            );
        }
    }

    // Summary statistics
    println!("\n[SUMMARY]\n");

    // Audio region average
    let audio_cos: Vec<f32> = (0..75)
        .filter(|&p| p < enc_len)
        .map(|p| {
            let start = p * d_model;
            let end = start + d_model;
            cosine_sim(&encoder_output[start..end], &pos_embed[start..end])
        })
        .collect();

    let audio_cos_avg: f32 = audio_cos.iter().sum::<f32>() / audio_cos.len() as f32;

    // Padding region average (positions 200-1400, avoiding boundary effects)
    let padding_cos: Vec<f32> = (200..1400)
        .filter(|&p| p < enc_len)
        .map(|p| {
            let start = p * d_model;
            let end = start + d_model;
            cosine_sim(&encoder_output[start..end], &pos_embed[start..end])
        })
        .collect();

    let padding_cos_avg: f32 = padding_cos.iter().sum::<f32>() / padding_cos.len() as f32;

    println!(
        "  Audio region (0-75) avg cos(EncoderOut, PosEmbed):   {:.4}",
        audio_cos_avg
    );
    println!(
        "  Padding region (200-1400) avg cos(EncoderOut, PosEmbed): {:.4}",
        padding_cos_avg
    );

    // Verdict
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      H31 VERDICT                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    if padding_cos_avg.abs() > 0.8 {
        println!("  ✓ H31 VERIFIED: Padding EncoderOut ≈ PosEmbed");
        println!(
            "    Cosine similarity {:.2} indicates positional dominance.",
            padding_cos_avg
        );
        println!("    The 'content' signal is effectively NULL in padding region.");
        println!("    Decoder is attending to WHERE, not WHAT.");
    } else if padding_cos_avg.abs() > 0.5 {
        println!("  ⚠ H31 PARTIAL: Moderate positional correlation");
        println!(
            "    Cosine similarity {:.2} shows some positional influence.",
            padding_cos_avg
        );
    } else {
        println!("  ✗ H31 FALSIFIED: EncoderOut ≠ PosEmbed at padding");
        println!(
            "    Cosine similarity {:.2} indicates transformer blocks add signal.",
            padding_cos_avg
        );
    }

    // Check the high-attention positions specifically
    println!("\n[HIGH-ATTENTION POSITIONS CHECK]\n");
    for &pos in &[1493, 1494] {
        let start = pos * d_model;
        let end = start + d_model;

        let enc_vec = &encoder_output[start..end];
        let pe_vec = &pos_embed[start..end];

        let cos = cosine_sim(enc_vec, pe_vec);
        println!("  Position {}: cos(EncoderOut, PosEmbed) = {:.4}", pos, cos);
    }

    Ok(())
}

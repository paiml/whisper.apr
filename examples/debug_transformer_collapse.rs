//! Transformer Block Collapse Tracer
//!
//! Since H25 (conv stem) and H26 (positional embeddings) are falsified,
//! the "Blind Encoder" bug is in the transformer blocks.
//!
//! This probe traces through each transformer block to find WHERE
//! the audio/padding differentiation collapses.
//!
//! Run: cargo run --example debug_transformer_collapse

use std::path::Path;
use whisper_apr::model::{EncoderBlock, LayerNorm};

fn stats(data: &[f32]) -> (f32, f32) {
    let n = data.len() as f32;
    let sum: f32 = data.iter().sum();
    let mean = sum / n;
    let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, variance.sqrt())
}

fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|&x| x * x).sum::<f32>().sqrt()
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

fn analyze_layer(name: &str, data: &[f32], d_model: usize, audio_pos: usize, padding_pos: usize) {
    let audio_start = audio_pos * d_model;
    let audio_end = audio_start + d_model;
    let padding_start = padding_pos * d_model;
    let padding_end = padding_start + d_model;

    let audio_vec = &data[audio_start..audio_end];
    let padding_vec = &data[padding_start..padding_end];

    let (a_mean, a_std) = stats(audio_vec);
    let (p_mean, p_std) = stats(padding_vec);
    let a_norm = l2_norm(audio_vec);
    let p_norm = l2_norm(padding_vec);
    let sim = cosine_sim(audio_vec, padding_vec);

    println!(
        "  {:12} | Audio({}): mean={:+.4} std={:.4} L2={:.2} | Pad({}): mean={:+.4} std={:.4} L2={:.2} | cos={:.4}",
        name, audio_pos, a_mean, a_std, a_norm, padding_pos, p_mean, p_std, p_norm, sim
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           TRANSFORMER BLOCK COLLAPSE TRACER                              ║");
    println!("║  Goal: Find where audio/padding differentiation collapses                ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Load model
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

    // Compute mel
    let mel = model.compute_mel(&samples)?;
    let n_mels = 80;
    let n_frames = mel.len() / n_mels;
    let audio_frames = (samples.len() as f32 / 16000.0 * 100.0).ceil() as usize;

    println!("[INPUT]");
    println!(
        "  Audio: {:.2}s ({} samples)",
        samples.len() as f32 / 16000.0,
        samples.len()
    );
    println!("  Mel: {} frames × {} mels", n_frames, n_mels);
    println!("  Audio region: frames 0..{}", audio_frames);

    // Get encoder
    let encoder = model.encoder_mut();
    let d_model = encoder.d_model();
    let n_layers = encoder.n_layers();
    let max_len = encoder.max_len();

    println!(
        "  Encoder: {} layers, d_model={}, max_len={}",
        n_layers, d_model, max_len
    );

    // Step 1: Conv frontend
    let conv_output = encoder.conv_frontend().forward(&mel)?;
    let seq_len = conv_output.len() / d_model;
    let audio_pos_conv = audio_frames / 2; // After stride-2 conv

    println!(
        "\n  Conv output: {} positions × {} d_model",
        seq_len, d_model
    );
    println!("  Audio ends at position ~{}", audio_pos_conv);

    // Test positions: one in audio region, one in padding region
    let audio_pos = 30; // Well within audio region
    let padding_pos = 1000; // Well into padding region

    println!("\n[LAYER-BY-LAYER TRACE]");
    println!(
        "  Comparing position {} (audio) vs {} (padding)\n",
        audio_pos, padding_pos
    );
    println!(
        "  {:12} | {:45} | {:45} | Similarity",
        "Stage", "Audio", "Padding"
    );
    println!("  {}", "-".repeat(120));

    // Analyze conv output
    analyze_layer("Conv Output", &conv_output, d_model, audio_pos, padding_pos);

    // Step 2: Add positional embeddings
    let mut x = conv_output.clone();
    let pos_embed = encoder.positional_embedding();
    for pos in 0..seq_len {
        for d in 0..d_model {
            x[pos * d_model + d] += pos_embed[pos * d_model + d];
        }
    }
    analyze_layer("+ Pos Embed", &x, d_model, audio_pos, padding_pos);

    // Step 3: Trace through each transformer block
    let blocks = encoder.blocks();
    for (i, block) in blocks.iter().enumerate() {
        // The forward pass is: x + Attention(LN(x)) then x + FFN(LN(x))
        // Let's trace each sub-operation

        // Pre-attention LayerNorm
        let normed = block.ln1.forward(&x)?;
        analyze_layer(
            &format!("Block{} LN1", i),
            &normed,
            d_model,
            audio_pos,
            padding_pos,
        );

        // Self-attention
        let attn_out = block.self_attn.forward(&normed, None)?;
        analyze_layer(
            &format!("Block{} Attn", i),
            &attn_out,
            d_model,
            audio_pos,
            padding_pos,
        );

        // Residual after attention
        let mut residual: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();
        analyze_layer(
            &format!("Block{} +Res1", i),
            &residual,
            d_model,
            audio_pos,
            padding_pos,
        );

        // Pre-FFN LayerNorm
        let normed2 = block.ln2.forward(&residual)?;
        analyze_layer(
            &format!("Block{} LN2", i),
            &normed2,
            d_model,
            audio_pos,
            padding_pos,
        );

        // FFN
        let ffn_out = block.ffn.forward(&normed2)?;
        analyze_layer(
            &format!("Block{} FFN", i),
            &ffn_out,
            d_model,
            audio_pos,
            padding_pos,
        );

        // Residual after FFN
        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }
        analyze_layer(
            &format!("Block{} +Res2", i),
            &residual,
            d_model,
            audio_pos,
            padding_pos,
        );

        x = residual;
        println!("  {}", "-".repeat(120));
    }

    // Final LayerNorm
    let final_output = encoder.ln_post().forward(&x)?;
    analyze_layer("Final LN", &final_output, d_model, audio_pos, padding_pos);

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         ANALYSIS SUMMARY                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Check where similarity jumps
    let conv_audio = &conv_output[audio_pos * d_model..(audio_pos + 1) * d_model];
    let conv_padding = &conv_output[padding_pos * d_model..(padding_pos + 1) * d_model];
    let final_audio = &final_output[audio_pos * d_model..(audio_pos + 1) * d_model];
    let final_padding = &final_output[padding_pos * d_model..(padding_pos + 1) * d_model];

    let conv_sim = cosine_sim(conv_audio, conv_padding);
    let final_sim = cosine_sim(final_audio, final_padding);

    println!("  Conv output similarity (audio↔padding):  {:.4}", conv_sim);
    println!(
        "  Final output similarity (audio↔padding): {:.4}",
        final_sim
    );

    if final_sim > 0.99 && conv_sim < 0.99 {
        println!("\n  ✗ COLLAPSE DETECTED!");
        println!("    Audio and padding became nearly identical through transformer blocks.");
        println!("    Look at the trace above to see where cosine similarity jumped to ~1.0");
    } else if final_sim < 0.95 {
        println!("\n  ✓ NO COLLAPSE");
        println!("    Audio and padding remain differentiated through the encoder.");
    }

    Ok(())
}

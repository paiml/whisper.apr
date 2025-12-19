//! H18: Check encoder positional embedding
//!
//! Verify if positional embeddings create bias toward certain positions

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
    println!("=== H18: ENCODER POSITIONAL EMBEDDING ANALYSIS ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    let model_bytes = std::fs::read(model_path)?;
    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    let d_model = 384;

    // Get encoder positional embedding
    let pos_embed = model.encoder_mut().positional_embedding();
    let n_positions = pos_embed.len() / d_model;

    println!("[ENCODER POSITIONAL EMBEDDING]");
    println!(
        "  Shape: {} positions × {} dims = {} values",
        n_positions,
        d_model,
        pos_embed.len()
    );

    // Check overall stats
    let (pe_mean, pe_std, pe_min, pe_max) = stats(pos_embed);
    println!(
        "  Overall: mean={:.6}  std={:.6}  range=[{:.4}, {:.4}]",
        pe_mean, pe_std, pe_min, pe_max
    );

    // Check per-position stats
    println!("\n  Per-position norms:");
    let positions = [0, 37, 74, 100, 500, 750, 1000, 1312, 1400, 1487, 1499];

    for &pos in &positions {
        if pos < n_positions {
            let start = pos * d_model;
            let end = start + d_model;
            let pos_vec = &pos_embed[start..end];

            // L2 norm
            let norm: f32 = pos_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let (mean, std, _, _) = stats(pos_vec);

            let label = if pos <= 75 { "AUDIO" } else { "PADDING" };
            println!(
                "    Pos {:4} [{}]: L2={:.4}  mean={:+.4}  std={:.4}",
                pos, label, norm, mean, std
            );
        }
    }

    // Check if there's a trend in positional embedding norms
    println!("\n  Norm trend across positions:");
    let mut norms: Vec<(usize, f32)> = (0..n_positions)
        .map(|pos| {
            let start = pos * d_model;
            let end = start + d_model;
            let norm: f32 = pos_embed[start..end]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
                .sqrt();
            (pos, norm)
        })
        .collect();

    // Find positions with highest norms
    norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n  Top-10 highest norm positions:");
    for (pos, norm) in norms.iter().take(10) {
        let label = if *pos <= 75 { "AUDIO" } else { "PADDING" };
        println!("    Pos {:4} [{}]: L2={:.4}", pos, label, norm);
    }

    println!("\n  Bottom-10 lowest norm positions:");
    for (pos, norm) in norms.iter().rev().take(10) {
        let label = if *pos <= 75 { "AUDIO" } else { "PADDING" };
        println!("    Pos {:4} [{}]: L2={:.4}", pos, label, norm);
    }

    // Check similarity between positions
    println!("\n[POSITIONAL EMBEDDING SIMILARITY]");

    // Compute average embedding for audio region vs padding region
    let audio_avg: Vec<f32> = (0..d_model)
        .map(|d| (0..75).map(|pos| pos_embed[pos * d_model + d]).sum::<f32>() / 75.0)
        .collect();

    let padding_avg: Vec<f32> = (0..d_model)
        .map(|d| {
            (1312..1500)
                .map(|pos| pos_embed[pos * d_model + d])
                .sum::<f32>()
                / 188.0
        })
        .collect();

    // Cosine similarity
    let dot: f32 = audio_avg.iter().zip(&padding_avg).map(|(a, b)| a * b).sum();
    let norm_a: f32 = audio_avg.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = padding_avg.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let cos_sim = dot / (norm_a * norm_b);

    println!("  Audio region (0-74) avg norm: {:.4}", norm_a);
    println!("  Padding region (1312-1499) avg norm: {:.4}", norm_b);
    println!("  Cosine similarity: {:.4}", cos_sim);

    // Check consecutive position similarity
    println!("\n  Consecutive position cosine similarities:");
    for pos in [0, 74, 500, 1000, 1400] {
        if pos + 1 < n_positions {
            let v1 = &pos_embed[pos * d_model..(pos + 1) * d_model];
            let v2 = &pos_embed[(pos + 1) * d_model..(pos + 2) * d_model];

            let dot: f32 = v1.iter().zip(v2).map(|(a, b)| a * b).sum();
            let n1: f32 = v1.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let n2: f32 = v2.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let sim = dot / (n1 * n2);

            println!("    Pos {} vs {}: {:.4}", pos, pos + 1, sim);
        }
    }

    Ok(())
}

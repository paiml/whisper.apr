//! H21: Encoder Output Verification (Post Slaney Fix)
//!
//! Verify that after mel fix, encoder differentiates audio from padding.
//! The "Padding Attractor" should be resolved.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== H21: ENCODER VERIFICATION (POST-FIX) ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    let model_bytes = std::fs::read(model_path)?;
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load test audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();

    // Compute mel and encode
    let mel = model.compute_mel(&samples)?;
    let encoded = model.encode(&mel)?;

    let d_model = 384;
    let n_positions = encoded.len() / d_model;

    // Audio region: positions 0-75 (where actual speech is)
    // Padding region: positions 1400+ (silence/padding)
    let audio_end = 75.min(n_positions);
    let padding_start = 1400.min(n_positions.saturating_sub(100));

    let audio_region: Vec<f32> = (0..audio_end)
        .flat_map(|p| encoded[p * d_model..(p + 1) * d_model].to_vec())
        .collect();

    let padding_region: Vec<f32> = (padding_start..n_positions.min(padding_start + 100))
        .flat_map(|p| encoded[p * d_model..(p + 1) * d_model].to_vec())
        .collect();

    // Stats
    let audio_mean = audio_region.iter().sum::<f32>() / audio_region.len() as f32;
    let audio_var = audio_region
        .iter()
        .map(|x| (x - audio_mean).powi(2))
        .sum::<f32>()
        / audio_region.len() as f32;
    let audio_std = audio_var.sqrt();

    let pad_mean = padding_region.iter().sum::<f32>() / padding_region.len() as f32;
    let pad_var = padding_region
        .iter()
        .map(|x| (x - pad_mean).powi(2))
        .sum::<f32>()
        / padding_region.len() as f32;
    let pad_std = pad_var.sqrt();

    println!(
        "Encoder output (d_model={}, positions={}):",
        d_model, n_positions
    );
    println!("\nAudio region (0-{}):", audio_end);
    println!("  mean: {:+.6}", audio_mean);
    println!("  std:  {:.6}", audio_std);

    println!(
        "\nPadding region ({}-{}):",
        padding_start,
        n_positions.min(padding_start + 100)
    );
    println!("  mean: {:+.6}", pad_mean);
    println!("  std:  {:.6}", pad_std);

    // Key test: std should differ significantly
    let std_diff = (audio_std - pad_std).abs();
    let mean_diff = (audio_mean - pad_mean).abs();

    println!("\n=== DIFFERENTIATION TEST ===");
    println!("Std difference:  {:.6}", std_diff);
    println!("Mean difference: {:.6}", mean_diff);

    if std_diff > 0.05 || mean_diff > 0.05 {
        println!("\n✅ PASS: Encoder differentiates audio from padding");
        println!("   The 'Padding Attractor' hypothesis (H19) should be RESOLVED.");
    } else {
        println!("\n❌ FAIL: Encoder outputs similar for audio and padding");
        println!("   'Padding Attractor' may still be present - investigate encoder weights.");
    }

    // Cosine similarity between average audio vector and average padding vector
    let audio_avg: Vec<f32> = (0..d_model)
        .map(|d| audio_region.iter().skip(d).step_by(d_model).sum::<f32>() / audio_end as f32)
        .collect();
    let pad_avg: Vec<f32> = (0..d_model)
        .map(|d| padding_region.iter().skip(d).step_by(d_model).sum::<f32>() / 100.0)
        .collect();

    let dot: f32 = audio_avg.iter().zip(&pad_avg).map(|(a, b)| a * b).sum();
    let norm_a = audio_avg.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b = pad_avg.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let cosine = if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    };

    println!("\nCosine similarity (audio vs padding): {:.4}", cosine);
    if cosine < 0.9 {
        println!("  → Vectors are distinguishable (good!)");
    } else {
        println!("  → Vectors too similar - encoder may not be learning");
    }

    Ok(())
}

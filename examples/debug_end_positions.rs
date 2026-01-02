//! Investigate anomalously high attention at end positions
//!
//! The cross-attention shows positions 1494, 1493, 1487 getting most attention.
//! This probe investigates what's special about these end positions.

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     INVESTIGATING END POSITION ATTENTION ANOMALY                 ║");
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
    let enc_len = encoder_output.len() / d_model;

    println!("[ENCODER OUTPUT SHAPE]");
    println!("  {} positions × {} d_model\n", enc_len, d_model);

    // Check the positions that get high attention
    let high_attn_positions = [1494, 1493, 1487, 1486, 1312, 1000, 100, 30, 0];

    println!("[ENCODER OUTPUT AT KEY POSITIONS]");
    println!(
        "  {:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Pos", "L2 Norm", "Mean", "Std", "Min", "Max"
    );
    println!("  {}", "-".repeat(75));

    for &pos in &high_attn_positions {
        if pos < enc_len {
            let start = pos * d_model;
            let end = start + d_model;
            let vec = &encoder_output[start..end];
            let norm = l2_norm(vec);
            let (mean, std, min, max) = stats(vec);
            let region = if pos < 75 { "AUDIO" } else { "PADDING" };
            println!(
                "  {:>6} | {:>10.4} | {:>10.4} | {:>10.4} | {:>10.4} | {:>10.4} [{}]",
                pos, norm, mean, std, min, max, region
            );
        }
    }

    // Check the very last few positions
    println!("\n[VERY LAST POSITIONS (potential boundary issue)]");
    for pos in (enc_len - 10)..enc_len {
        let start = pos * d_model;
        let end = start + d_model;
        let vec = &encoder_output[start..end];
        let norm = l2_norm(vec);
        let (mean, std, min, max) = stats(vec);
        println!(
            "  Pos {:>4}: L2={:>10.4}, mean={:>+10.4}, std={:>10.4}, range=[{:>.3},{:>.3}]",
            pos, norm, mean, std, min, max
        );
    }

    // Check the mel spectrogram at these positions (before stride-2, so pos*2 in mel)
    println!("\n[MEL SPECTROGRAM AT CORRESPONDING POSITIONS]");
    let n_mels = 80;
    let mel_frames = mel.len() / n_mels;

    for &enc_pos in &[1494, 1493, 1487, 100, 30, 0] {
        let mel_pos = enc_pos * 2; // Accounting for stride-2 in conv
        if mel_pos < mel_frames {
            let mel_start = mel_pos * n_mels;
            let mel_end = mel_start + n_mels;
            let mel_vec = &mel[mel_start..mel_end];
            let (mean, std, min, max) = stats(mel_vec);
            let norm = l2_norm(mel_vec);
            println!("  Enc pos {} → Mel pos {}: mean={:+.4}, std={:.4}, norm={:.4}, range=[{:.3},{:.3}]",
                     enc_pos, mel_pos, mean, std, norm, min, max);
        } else {
            println!("  Enc pos {} → Mel pos {} (BEYOND MEL)!", enc_pos, mel_pos);
        }
    }

    // Check positional embeddings at these positions
    println!("\n[POSITIONAL EMBEDDINGS AT KEY POSITIONS]");
    let pos_embed = model.encoder_mut().positional_embedding();

    for &pos in &[1494, 1493, 1487, 100, 30, 0] {
        if pos < enc_len {
            let start = pos * d_model;
            let end = start + d_model;
            let pe = &pos_embed[start..end];
            let (mean, std, min, max) = stats(pe);
            let norm = l2_norm(pe);
            println!(
                "  Pos {:>4}: L2={:>8.4}, mean={:>+8.4}, std={:.4}, range=[{:.3},{:.3}]",
                pos, norm, mean, std, min, max
            );
        }
    }

    Ok(())
}

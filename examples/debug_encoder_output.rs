//! H7 Falsification: Check encoder output variance
//!
//! If std < 0.01: Encoder is degenerate (bug upstream)
//! If std > 0.1: Encoder is healthy (bug downstream in decoder)

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
    println!("=== H7 FALSIFICATION: ENCODER OUTPUT VARIANCE ===\n");

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

    println!("Audio samples: {} ({:.2}s @ 16kHz)", samples.len(), samples.len() as f32 / 16000.0);

    // Step 1: Mel spectrogram
    let mel = model.compute_mel(&samples)?;
    let (mel_mean, mel_std, mel_min, mel_max) = stats(&mel);
    println!("\n[MEL SPECTROGRAM]");
    println!("  shape: {} values", mel.len());
    println!("  mean={:.6}  std={:.6}  min={:.6}  max={:.6}", mel_mean, mel_std, mel_min, mel_max);

    // Step 2: Encoder output
    let encoded = model.encode(&mel)?;
    let (enc_mean, enc_std, enc_min, enc_max) = stats(&encoded);
    println!("\n[ENCODER OUTPUT]");
    println!("  shape: {} values", encoded.len());
    println!("  mean={:.6}  std={:.6}  min={:.6}  max={:.6}", enc_mean, enc_std, enc_min, enc_max);

    // H7 Verdict
    println!("\n=== H7 VERDICT ===");
    if enc_std < 0.01 {
        println!("CONFIRMED: Encoder output is DEGENERATE (std={:.6} < 0.01)", enc_std);
        println!("  -> Bug is UPSTREAM in encoder or mel computation");
        println!("  -> Cross-attention receives garbage K/V regardless of weights");
    } else if enc_std > 0.1 {
        println!("FALSIFIED: Encoder output is HEALTHY (std={:.6} > 0.1)", enc_std);
        println!("  -> Bug is DOWNSTREAM in decoder cross-attention");
        println!("  -> Proceed to H8 (K/V projection) or H9/H10 (attention math)");
    } else {
        println!("INCONCLUSIVE: Encoder std={:.6} (borderline)", enc_std);
        println!("  -> Compare against whisper.cpp encoder output");
    }

    // Additional: Check per-timestep variance
    println!("\n[PER-TIMESTEP ANALYSIS]");
    let d_model = 384; // whisper-tiny dimension
    let n_timesteps = encoded.len() / d_model;
    println!("  {} timesteps x {} dims", n_timesteps, d_model);

    for t in [0, n_timesteps/2, n_timesteps-1].iter().filter(|&&t| t < n_timesteps) {
        let start = t * d_model;
        let end = start + d_model;
        let slice = &encoded[start..end];
        let (m, s, _, _) = stats(slice);
        println!("  t={:3}: mean={:+.4}  std={:.4}", t, m, s);
    }

    // Check if all timesteps are identical (degenerate)
    if n_timesteps > 1 {
        let t0 = &encoded[0..d_model];
        let t1 = &encoded[d_model..2*d_model];
        let diff: f32 = t0.iter().zip(t1.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / d_model as f32;
        println!("\n  L1 distance t0 vs t1: {:.6}", diff);
        if diff < 0.001 {
            println!("  WARNING: Timesteps are nearly identical!");
        }
    }

    Ok(())
}

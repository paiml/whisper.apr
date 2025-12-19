//! H17: Check where actual audio content is in mel spectrogram
//!
//! If model attends to pos 1487 but audio is only 1.5s (75 frames),
//! either mel is wrong or attention is looking at wrong region

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
    println!("=== H17: MEL SPECTROGRAM CONTENT ANALYSIS ===\n");

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

    println!("[AUDIO INPUT]");
    println!(
        "  Samples: {} ({:.2}s @ 16kHz)",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    // Check audio energy distribution
    let chunk_size = 1600; // 100ms chunks
    let n_chunks = samples.len() / chunk_size;
    println!("  Energy per 100ms chunk:");

    let mut nonzero_chunks = 0;
    for i in 0..n_chunks.min(30) {
        let start = i * chunk_size;
        let end = start + chunk_size;
        let rms: f32 =
            (samples[start..end].iter().map(|&x| x * x).sum::<f32>() / chunk_size as f32).sqrt();
        if rms > 0.01 {
            nonzero_chunks += 1;
            println!(
                "    Chunk {:2} ({:.1}s - {:.1}s): RMS = {:.4}",
                i,
                start as f32 / 16000.0,
                end as f32 / 16000.0,
                rms
            );
        }
    }
    println!(
        "  Non-zero chunks (RMS>0.01): {}/{}",
        nonzero_chunks, n_chunks
    );

    // Compute mel spectrogram
    let mel = model.compute_mel(&samples)?;
    let n_mels = 80;
    let mel_len = mel.len() / n_mels;

    println!("\n[MEL SPECTROGRAM]");
    println!("  Shape: {} × {} = {} values", mel_len, n_mels, mel.len());
    println!(
        "  Time coverage: {} frames × 20ms = {:.1}s",
        mel_len,
        mel_len as f32 * 0.02
    );

    // Check mel energy distribution
    println!("\n  Energy per frame region:");

    let regions = [
        (0, 75, "0-1.5s (actual audio)"),
        (75, 150, "1.5-3s"),
        (150, 750, "3-15s"),
        (750, 1500, "15-30s (padding)"),
    ];

    for (start, end, label) in regions.iter() {
        let start_idx = start * n_mels;
        let end_idx = ((*end).min(mel_len)) * n_mels;
        if start_idx < mel.len() && end_idx > start_idx {
            let region = &mel[start_idx..end_idx];
            let (mean, std, min, max) = stats(region);
            println!(
                "    {} (frames {}-{}): mean={:.4}  std={:.4}  range=[{:.2}, {:.2}]",
                label,
                start,
                end.min(&mel_len),
                mean,
                std,
                min,
                max
            );
        }
    }

    // Check specific frames
    println!("\n  Per-frame analysis (every 100 frames):");
    for frame in (0..mel_len).step_by(100) {
        let start = frame * n_mels;
        let end = start + n_mels;
        let (mean, std, _, _) = stats(&mel[start..end]);
        let time_s = frame as f32 * 0.02;
        println!(
            "    Frame {:4} ({:5.1}s): mean={:+.4}  std={:.4}",
            frame, time_s, mean, std
        );
    }

    // Check the encoder output at attended positions
    println!("\n[ENCODER OUTPUT AT ATTENDED POSITIONS]");
    let encoded = model.encode(&mel)?;
    let d_model = 384;
    let enc_len = encoded.len() / d_model;

    println!("  Encoder: {} timesteps × {} dims", enc_len, d_model);

    // Check positions the model attended to (1487, 1494, etc)
    let attended_positions = [1487, 1494, 1493, 1486, 1312, 0, 37, 74]; // Top + expected

    for &pos in &attended_positions {
        if pos < enc_len {
            let start = pos * d_model;
            let end = start + d_model;
            let (mean, std, _, _) = stats(&encoded[start..end]);
            let time_s = pos as f32 * 0.02;
            let label = if pos < 75 { "ACTUAL AUDIO" } else { "PADDING" };
            println!(
                "    Pos {:4} ({:5.1}s) [{}]: mean={:+.4}  std={:.4}",
                pos, time_s, label, mean, std
            );
        }
    }

    // The key question: why does high-energy audio (pos 0-74) not get attended?
    println!("\n=== DIAGNOSIS ===");
    println!(
        "  Audio duration: {:.2}s = ~{} mel frames",
        samples.len() as f32 / 16000.0,
        samples.len() / 320
    );
    println!(
        "  Expected attention region: frames 0-{}",
        samples.len() / 320
    );
    println!("  Actual attended region: frames 1312-1494 (padding)");
    println!("\n  ISSUE: Model is attending to PADDING instead of CONTENT");

    Ok(())
}

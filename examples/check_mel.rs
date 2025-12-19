//! Check mel and encoder dimensions

use whisper_apr::WhisperApr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let model = WhisperApr::load_from_apr(&model_bytes)?;

    // Load test audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    println!("Audio samples: {}", samples.len());
    println!("Audio duration: {:.2}s", samples.len() as f32 / 16000.0);

    // Compute mel - this now pads to 30s
    let mel = model.compute_mel(&samples)?;

    let n_mels = 80;
    let n_frames = mel.len() / n_mels;
    println!("\nMel spectrogram:");
    println!("  Total values: {}", mel.len());
    println!("  N_mels: {}", n_mels);
    println!("  N_frames: {}", n_frames);

    // Expected: 3000 frames for 30s audio
    println!("\nExpected: 3000 frames");
    println!("Difference: {}", n_frames as i32 - 3000);

    // Encode
    let encoder_output = model.encode(&mel)?;
    let hidden_dim = 384; // tiny
    let n_positions = encoder_output.len() / hidden_dim;

    println!("\nEncoder output:");
    println!("  Total values: {}", encoder_output.len());
    println!("  Hidden dim: {}", hidden_dim);
    println!("  N_positions: {}", n_positions);
    println!("\nExpected: 1500 positions");
    println!("Difference: {}", n_positions as i32 - 1500);

    // Check mel stats
    let mel_min = mel.iter().cloned().fold(f32::INFINITY, f32::min);
    let mel_max = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mel_mean = mel.iter().sum::<f32>() / mel.len() as f32;

    println!("\nMel stats:");
    println!("  Min: {:.4}", mel_min);
    println!("  Max: {:.4}", mel_max);
    println!("  Mean: {:.4}", mel_mean);

    // First and last 10 frames
    println!("\nFirst frame mel values (first 10):");
    for i in 0..10.min(n_mels) {
        print!("{:.3} ", mel[i]);
    }
    println!();

    println!("\nLast frame mel values (first 10):");
    let last_frame_start = (n_frames - 1) * n_mels;
    for i in 0..10.min(n_mels) {
        print!("{:.3} ", mel[last_frame_start + i]);
    }
    println!();

    Ok(())
}

//! Debug mel spectrogram at each computation stage

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MEL COMPUTATION STAGE DEBUG ===\n");

    // Load test audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    println!("Audio: {} samples", samples.len());
    println!(
        "  mean: {:.6}",
        samples.iter().sum::<f32>() / samples.len() as f32
    );
    println!(
        "  std:  {:.6}",
        (samples.iter().map(|x| x.powi(2)).sum::<f32>() / samples.len() as f32).sqrt()
    );

    // Parameters (Whisper defaults)
    let n_fft = 400;
    let hop_length = 160;
    let n_mels = 80;
    let n_freqs = n_fft / 2 + 1; // 201

    // Load filterbank from model
    let model_bytes = std::fs::read("models/whisper-tiny.apr")?;
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // We'll compute just the first few frames manually
    let n_frames = 10;

    // Create Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Compute manually for first few frames
    let mut raw_mel_energies: Vec<f32> = Vec::new();
    let mut log_mel_values: Vec<f32> = Vec::new();

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;

        // Apply window and prepare FFT input
        let mut fft_input: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| {
                let sample = if start + i < samples.len() {
                    samples[start + i]
                } else {
                    0.0
                };
                Complex::new(sample * window[i], 0.0)
            })
            .collect();

        // Compute FFT
        fft.process(&mut fft_input);

        // Power spectrum
        let power_spec: Vec<f32> = fft_input
            .iter()
            .take(n_freqs)
            .map(|c| c.norm_sqr())
            .collect();

        if frame_idx == 0 {
            println!("\n=== Frame 0 Power Spectrum ===");
            println!("  First 5 bins: {:?}", &power_spec[..5]);
            let ps_mean = power_spec.iter().sum::<f32>() / power_spec.len() as f32;
            let ps_max = power_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("  mean: {:.6}, max: {:.6}", ps_mean, ps_max);
        }

        // For each mel bin, we'd normally apply filterbank
        // Here just store raw values for first frame
        if frame_idx == 0 {
            for mel_idx in 0..n_mels.min(5) {
                // Sum over frequency bins (would use filterbank weights)
                let mel_energy: f32 = power_spec.iter().sum::<f32>() / n_freqs as f32;
                raw_mel_energies.push(mel_energy);

                let log_mel = (mel_energy.max(1e-10)).log10();
                log_mel_values.push(log_mel);
            }
        }
    }

    println!("\n=== Raw Mel Energies (first 5 bins of frame 0) ===");
    for (i, &e) in raw_mel_energies.iter().take(5).enumerate() {
        println!("  mel[{}] energy: {:.6}", i, e);
    }

    println!("\n=== Log10(mel) Before Normalization ===");
    for (i, &l) in log_mel_values.iter().take(5).enumerate() {
        println!("  log10(mel[{}]): {:+.6}", i, l);
    }

    // Now compute using the model's method and show result
    let mel = model.compute_mel(&samples)?;
    let mel_mean = mel.iter().take(80 * 148).sum::<f32>() / (80 * 148) as f32;
    let mel_min = mel
        .iter()
        .take(80 * 148)
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let mel_max = mel
        .iter()
        .take(80 * 148)
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    println!("\n=== Model's compute_mel() Output (first 148 frames) ===");
    println!("  mean: {:+.6}", mel_mean);
    println!("  min:  {:+.6}", mel_min);
    println!("  max:  {:+.6}", mel_max);
    println!("\n  First 5 values: {:?}", &mel[..5]);

    println!("\n=== Expected Ground Truth ===");
    println!("  mean: {:+.6}", -0.214805);
    println!("  min:  {:+.6}", -0.765833);
    println!("  max:  {:+.6}", 1.234167);

    // The key insight: if our log10 values before normalization are different,
    // that's the issue. If they're the same but result after norm is different,
    // the normalization formula is wrong.

    println!("\n=== Normalization Formula Check ===");
    // Simulate: (x + 4.0) / 4.0
    // For x = -2.0: result = 2.0/4.0 = 0.5
    // For x = -4.0: result = 0.0/4.0 = 0.0
    // For x = 0.0: result = 4.0/4.0 = 1.0
    println!("  Formula: (x + 4.0) / 4.0");
    println!("  x=-4 -> {:.2}", (-4.0 + 4.0) / 4.0);
    println!("  x=-2 -> {:.2}", (-2.0 + 4.0) / 4.0);
    println!("  x=0  -> {:.2}", (0.0 + 4.0) / 4.0);

    // GT mean of -0.215 implies pre-norm value of: -0.215 * 4 - 4 = -4.86
    // But log10(0.000014) ≈ -4.86, which would mean extremely low mel energy
    // That's plausible for typical audio

    println!("\n  Reverse engineering GT:");
    println!(
        "  GT mean -0.215 implies pre-norm: {:.2}",
        -0.215 * 4.0 - 4.0
    );
    println!("  Our mean +0.18 implies pre-norm: {:.2}", 0.18 * 4.0 - 4.0);

    Ok(())
}

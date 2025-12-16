#![allow(clippy::unwrap_used)]
//! Mel spectrogram computation example
//!
//! Demonstrates the mel filterbank and spectrogram computation.
//!
//! Run with: `cargo run --example mel_spectrogram`

use whisper_apr::audio::MelFilterbank;

fn main() {
    println!("=== Whisper.apr Mel Spectrogram Example ===\n");

    // Whisper's standard parameters
    let n_mels = 80; // Number of mel frequency bins
    let n_fft = 400; // FFT window size (25ms at 16kHz)
    let sample_rate: usize = 16000; // Whisper's native sample rate
    let hop_length = 160; // Hop size (10ms at 16kHz)

    println!("Mel filterbank parameters:");
    println!("  Mel bins: {}", n_mels);
    println!(
        "  FFT size: {} ({:.1}ms window)",
        n_fft,
        n_fft as f32 / sample_rate as f32 * 1000.0
    );
    println!(
        "  Hop length: {} ({:.1}ms)",
        hop_length,
        hop_length as f32 / sample_rate as f32 * 1000.0
    );
    println!("  Sample rate: {} Hz", sample_rate);
    println!();

    // Create mel filterbank
    let mel = MelFilterbank::new(n_mels, n_fft, sample_rate as u32);

    // Demonstrate mel scale conversion
    println!("Mel scale examples:");
    for freq in [100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0] {
        let mel_val = MelFilterbank::hz_to_mel(freq);
        let back = MelFilterbank::mel_to_hz(mel_val);
        println!("  {} Hz -> {:.2} mel -> {:.2} Hz", freq, mel_val, back);
    }
    println!();

    // Generate test audio signals
    println!("=== Signal Analysis ===\n");

    // 1. Pure silence
    let silence: Vec<f32> = vec![0.0; sample_rate]; // 1 second
    analyze_signal(&mel, "Silence", &silence, hop_length);

    // 2. Low frequency tone (200 Hz)
    let low_tone: Vec<f32> = (0..sample_rate)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.5
        })
        .collect();
    analyze_signal(&mel, "200 Hz tone", &low_tone, hop_length);

    // 3. Mid frequency tone (1000 Hz)
    let mid_tone: Vec<f32> = (0..sample_rate)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.5
        })
        .collect();
    analyze_signal(&mel, "1000 Hz tone", &mid_tone, hop_length);

    // 4. High frequency tone (4000 Hz)
    let high_tone: Vec<f32> = (0..sample_rate)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 4000.0 * t).sin() * 0.5
        })
        .collect();
    analyze_signal(&mel, "4000 Hz tone", &high_tone, hop_length);

    // 5. White noise (random)
    let noise: Vec<f32> = (0..sample_rate)
        .map(|i| {
            // Simple pseudo-random using sine
            ((i as f32 * 12345.6789).sin() * 43758.5453).fract() * 2.0 - 1.0
        })
        .map(|x| x * 0.3) // Scale down
        .collect();
    analyze_signal(&mel, "White noise", &noise, hop_length);

    // 6. Speech-like (multiple harmonics)
    let speech_like: Vec<f32> = (0..sample_rate)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let fundamental = 150.0; // Typical male voice fundamental
            let f1 = (2.0 * std::f32::consts::PI * fundamental * t).sin();
            let f2 = (2.0 * std::f32::consts::PI * fundamental * 2.0 * t).sin() * 0.5;
            let f3 = (2.0 * std::f32::consts::PI * fundamental * 3.0 * t).sin() * 0.25;
            let f4 = (2.0 * std::f32::consts::PI * fundamental * 4.0 * t).sin() * 0.125;
            (f1 + f2 + f3 + f4) * 0.3
        })
        .collect();
    analyze_signal(&mel, "Speech-like (harmonics)", &speech_like, hop_length);

    println!("=== Example Complete ===");
}

fn analyze_signal(mel: &MelFilterbank, name: &str, audio: &[f32], hop_length: usize) {
    println!("{} ({} samples):", name, audio.len());

    match mel.compute(audio, hop_length) {
        Ok(mel_spec) => {
            let n_frames = mel_spec.len() / 80;
            println!("  Frames: {}", n_frames);

            // Calculate energy per mel bin (average across frames)
            let mut bin_energy = vec![0.0f32; 80];
            for frame in 0..n_frames {
                for bin in 0..80 {
                    bin_energy[bin] += mel_spec[frame * 80 + bin].abs();
                }
            }
            for e in &mut bin_energy {
                *e /= n_frames as f32;
            }

            // Find peak mel bin
            let (peak_bin, peak_energy) = bin_energy
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, &e)| (i, e))
                .unwrap_or((0, 0.0));

            // Total energy
            let total_energy: f32 = bin_energy.iter().sum();

            println!("  Peak mel bin: {} (energy: {:.4})", peak_bin, peak_energy);
            println!("  Total energy: {:.4}", total_energy);

            // Show energy distribution (simplified histogram)
            if total_energy > 0.0 {
                let low: f32 = bin_energy[0..27].iter().sum();
                let mid: f32 = bin_energy[27..54].iter().sum();
                let high: f32 = bin_energy[54..80].iter().sum();
                println!(
                    "  Distribution: low={:.1}% mid={:.1}% high={:.1}%",
                    low / total_energy * 100.0,
                    mid / total_energy * 100.0,
                    high / total_energy * 100.0
                );
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();
}

//! Debug decoder output
//!
//! Shows what tokens are being generated to diagnose the whitespace issue.

use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DECODER DEBUG ===\n");

    // Load model
    let model_path = Path::new("models/whisper-tiny-int8.apr");
    if !model_path.exists() {
        eprintln!("Model not found: {}", model_path.display());
        return Ok(());
    }

    println!("Loading model...");
    let model_bytes = std::fs::read(model_path)?;
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;
    println!("Model loaded.\n");

    // Load very short audio
    let audio_path = Path::new("demos/test-audio/test-speech-1.5s.wav");
    if !audio_path.exists() {
        eprintln!("Audio not found");
        return Ok(());
    }

    let audio_bytes = std::fs::read(audio_path)?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();
    println!(
        "Audio: {} samples ({:.2}s)\n",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    // Step 1: Compute mel
    println!("Computing mel spectrogram...");
    let t0 = Instant::now();
    let mel = model.compute_mel(&samples)?;
    println!(
        "  Mel: {} values ({} frames) in {:?}\n",
        mel.len(),
        mel.len() / 80,
        t0.elapsed()
    );

    // Check mel values
    let mel_min = mel.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let mel_max = mel.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mel_mean: f32 = mel.iter().sum::<f32>() / mel.len() as f32;
    println!(
        "  Mel stats: min={:.2}, max={:.2}, mean={:.2}",
        mel_min, mel_max, mel_mean
    );

    // Step 2: Encode
    println!("\nEncoding...");
    let t0 = Instant::now();
    let encoded = model.encode(&mel)?;
    println!(
        "  Encoded: {} features in {:?}",
        encoded.len(),
        t0.elapsed()
    );

    // Check encoded values
    let enc_min = encoded.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let enc_max = encoded.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let enc_mean: f32 = encoded.iter().sum::<f32>() / encoded.len() as f32;
    println!(
        "  Encoded stats: min={:.2}, max={:.2}, mean={:.2}",
        enc_min, enc_max, enc_mean
    );

    // Check if encoder output looks reasonable
    let non_zero_count = encoded.iter().filter(|&&x| x.abs() > 1e-6).count();
    println!(
        "  Non-zero values: {} ({:.1}%)",
        non_zero_count,
        non_zero_count as f32 / encoded.len() as f32 * 100.0
    );

    // Step 3: Limited transcribe (max 10 tokens for speed)
    println!("\nTranscribing (limited to see first tokens)...");

    // Use the model's transcribe but with very limited options
    let options = whisper_apr::TranscribeOptions {
        language: Some("en".to_string()),
        task: whisper_apr::Task::Transcribe,
        strategy: whisper_apr::DecodingStrategy::Greedy,
        word_timestamps: false,
    };

    let t0 = Instant::now();
    let result = model.transcribe(&samples, options)?;
    let elapsed = t0.elapsed();

    println!("\n=== RESULTS ===");
    println!("Time: {:?}", elapsed);
    println!("Text length: {} chars", result.text.len());
    println!(
        "Text (first 200 chars): {:?}",
        &result.text[..result.text.len().min(200)]
    );
    println!("Segments: {}", result.segments.len());

    // Check the actual bytes
    println!(
        "\nText as bytes (first 50): {:?}",
        &result.text.as_bytes()[..result.text.len().min(50)]
    );

    // Check what the whitespace is
    let unique_chars: std::collections::HashSet<_> = result.text.chars().collect();
    println!("Unique characters: {:?}", unique_chars);

    Ok(())
}

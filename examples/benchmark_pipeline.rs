//! Full Pipeline Benchmark (WAPR-BENCH-001)
//!
//! Comprehensive benchmark measuring every step of the whisper.apr pipeline
//! from audio loading through transcription.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example benchmark_pipeline --release
//! ```
//!
//! # Requirements
//!
//! - Model file: models/whisper-tiny-int8.apr
//! - Test audio: demos/test-audio/test-speech-1.5s.wav

use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║    WHISPER.APR FULL PIPELINE BENCHMARK (WAPR-BENCH-001)      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // PHASE 1: Model Loading
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 1: MODEL LOADING                                      │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let model_path = Path::new("models/whisper-tiny-int8.apr");
    if !model_path.exists() {
        eprintln!("❌ Model not found: {}", model_path.display());
        eprintln!("\nTo download the model, run:");
        eprintln!("  cargo run --features converter --release --bin convert_model -- tiny --quantize int8");
        return Ok(());
    }

    println!("[1] Loading model file...");
    let t0 = Instant::now();
    let model_bytes = std::fs::read(model_path)?;
    let file_load_time = t0.elapsed();
    println!(
        "    File load: {:>8.2}ms ({:.2} MB)",
        file_load_time.as_secs_f64() * 1000.0,
        model_bytes.len() as f64 / 1_000_000.0
    );

    println!("[2] Parsing APR format and loading weights...");
    let t0 = Instant::now();
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;
    let parse_time = t0.elapsed();
    println!(
        "    APR parse: {:>8.2}ms",
        parse_time.as_secs_f64() * 1000.0
    );

    let total_model_load = file_load_time + parse_time;
    println!("\n    ══════════════════════════════════════════════════════════");
    println!(
        "    TOTAL MODEL LOAD: {:>8.2}ms",
        total_model_load.as_secs_f64() * 1000.0
    );
    println!("    ══════════════════════════════════════════════════════════\n");

    // =========================================================================
    // PHASE 2: Audio Preprocessing
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 2: AUDIO PREPROCESSING                                │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let test_files = [
        ("1.5s", "demos/test-audio/test-speech-1.5s.wav"),
        ("3s", "demos/test-audio/test-speech-3s.wav"),
    ];

    for (name, path) in test_files {
        let audio_path = Path::new(path);
        if !audio_path.exists() {
            println!("⚠️  {} not found, skipping\n", path);
            continue;
        }

        println!("📁 Audio: {} ({})", name, path);
        println!("────────────────────────────────────────────────────────────");

        // Step B: Load WAV bytes
        let t0 = Instant::now();
        let audio_bytes = std::fs::read(audio_path)?;
        let load_time = t0.elapsed();
        println!(
            "[B] WAV Load:      {:>8.2}ms ({} bytes)",
            load_time.as_secs_f64() * 1000.0,
            audio_bytes.len()
        );

        // Step C: Parse PCM
        let t0 = Instant::now();
        let samples: Vec<f32> = audio_bytes[44..]
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();
        let parse_time = t0.elapsed();
        let audio_duration = samples.len() as f32 / 16000.0;
        println!(
            "[C] PCM Parse:     {:>8.2}ms ({} samples, {:.2}s)",
            parse_time.as_secs_f64() * 1000.0,
            samples.len(),
            audio_duration
        );

        // Step F: Mel spectrogram
        let t0 = Instant::now();
        let mel = model.compute_mel(&samples)?;
        let mel_time = t0.elapsed();
        let mel_frames = mel.len() / 80;
        println!(
            "[F] Mel Compute:   {:>8.2}ms ({} frames)",
            mel_time.as_secs_f64() * 1000.0,
            mel_frames
        );

        // Step G: Encoder
        let t0 = Instant::now();
        let encoded = model.encode(&mel)?;
        let encode_time = t0.elapsed();
        let encoded_len = encoded.len();
        println!(
            "[G] Encode:        {:>8.2}ms ({} features)",
            encode_time.as_secs_f64() * 1000.0,
            encoded_len
        );

        // Step H: Full transcribe (includes decode)
        let t0 = Instant::now();
        let result = model.transcribe(&samples, whisper_apr::TranscribeOptions::default())?;
        let full_transcribe_time = t0.elapsed();

        // Estimate decode time by subtracting mel + encode
        let decode_time_est = full_transcribe_time - mel_time - encode_time;
        println!(
            "[H] Decode (est):  {:>8.2}ms",
            decode_time_est.as_secs_f64() * 1000.0
        );

        println!("────────────────────────────────────────────────────────────");

        // Summary
        let preprocess_time = load_time + parse_time + mel_time;
        let total_time = full_transcribe_time;

        println!(
            "    Preprocess:    {:>8.2}ms",
            preprocess_time.as_secs_f64() * 1000.0
        );
        println!(
            "    Encode:        {:>8.2}ms ({:.1}% of total)",
            encode_time.as_secs_f64() * 1000.0,
            encode_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        println!(
            "    Decode:        {:>8.2}ms ({:.1}% of total)",
            decode_time_est.as_secs_f64() * 1000.0,
            decode_time_est.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        println!("    ════════════════════════════════════════════════════════");
        println!(
            "    TOTAL:         {:>8.2}ms",
            total_time.as_secs_f64() * 1000.0
        );

        // RTF calculation
        let rtf = total_time.as_secs_f64() / audio_duration as f64;
        let rtf_status = if rtf <= 1.0 {
            "✅ REAL-TIME"
        } else if rtf <= 2.0 {
            "✅ TARGET MET"
        } else if rtf <= 3.0 {
            "⚠️  ACCEPTABLE"
        } else {
            "❌ TOO SLOW"
        };
        println!("    RTF:           {:>8.2}x {}", rtf, rtf_status);

        // Output text
        println!("\n    Output: \"{}\"", result.text.trim());
        println!("\n");
    }

    // =========================================================================
    // PHASE 3: Performance Analysis
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 3: PERFORMANCE ANALYSIS                               │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("Per-Component Breakdown (typical for 1.5s audio):");
    println!("────────────────────────────────────────────────────────────");
    println!("  Component    │ Expected   │ Actual*    │ Status");
    println!("  ────────────│────────────│────────────│────────────");
    println!("  WAV Load     │ <5ms       │ <1ms       │ ✅");
    println!("  PCM Parse    │ <1ms       │ <1ms       │ ✅");
    println!("  Mel FFT      │ <10ms      │ ~2ms       │ ✅");
    println!("  Encoder      │ <500ms     │ ~50-200ms  │ ✅");
    println!("  Decoder      │ <2000ms    │ MEASURING  │ ⏳");
    println!("  ────────────│────────────│────────────│────────────");
    println!("  TOTAL E2E    │ <3000ms    │ MEASURING  │ ⏳");
    println!("\n  * Actual values depend on hardware and model.\n");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK COMPLETE                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

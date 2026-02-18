//! fp16 vs f32 Weight Storage Benchmark
//!
//! Measures the actual speedup from fp16 weight storage across:
//! 1. Isolated LinearWeights::forward_simd (single-token, the hot path)
//! 2. End-to-end transcription (f32 model vs fp16 model)
//! 3. Output correctness verification (transcription text must match)
//!
//! # Usage
//! ```bash
//! cargo run --example benchmark_fp16 --release
//! ```

use std::path::Path;
use std::time::Instant;
use whisper_apr::audio::wav::parse_wav_file;
use whisper_apr::TranscribeOptions;

/// Number of iterations for micro-benchmarks
const MICRO_ITERS: usize = 1000;
/// Number of iterations for e2e benchmarks
const E2E_ITERS: usize = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║    fp16 vs f32 WEIGHT STORAGE BENCHMARK                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // PART 1: Isolated LinearWeights micro-benchmark
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 1: LinearWeights::forward_simd (single-token decode)   │");
    println!("│         This is the memory-bandwidth-bound hot path.        │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    micro_benchmark_linear_weights();

    // =========================================================================
    // PART 2: End-to-end transcription benchmark
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 2: End-to-end transcription (f32 vs fp16 model)        │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let f32_model_path = Path::new("models/whisper-tiny-fb.apr");
    let f16_model_path = Path::new("models/whisper-tiny-f16.apr");
    let audio_path = Path::new("demos/test-audio/test-speech-1.5s.wav");

    if !f32_model_path.exists() {
        eprintln!("  Model not found: {}", f32_model_path.display());
        eprintln!("  Trying alternate path...");
    }
    // Try alternate f32 model
    let f32_model_path = if f32_model_path.exists() {
        f32_model_path.to_path_buf()
    } else {
        let alt = Path::new("models/whisper-tiny.apr");
        if !alt.exists() {
            eprintln!("  No f32 model found. Skipping e2e benchmark.");
            return Ok(());
        }
        alt.to_path_buf()
    };

    if !f16_model_path.exists() {
        eprintln!("  fp16 model not found: {}. Skipping e2e benchmark.", f16_model_path.display());
        return Ok(());
    }
    if !audio_path.exists() {
        eprintln!("  Audio not found: {}. Skipping e2e benchmark.", audio_path.display());
        return Ok(());
    }

    e2e_benchmark(&f32_model_path, f16_model_path, audio_path)?;

    // =========================================================================
    // PART 3: Summary and claim verification
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 3: Claim Verification                                  │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("  Claims from plan:");
    println!("  - fp16 model file ~50%% smaller than f32     [CHECK file sizes above]");
    println!("  - ~2x speedup on single-token decode         [CHECK Part 1 above]");
    println!("  - Output identical/near-identical to f32     [CHECK Part 2 above]");
    println!("  - Memory bandwidth is the bottleneck         [CHECK: if speedup ~2x, confirmed]");
    println!();

    Ok(())
}

fn micro_benchmark_linear_weights() {
    use whisper_apr::model::LinearWeights;

    // Test various sizes representative of Whisper layers
    let configs: Vec<(usize, usize, &str)> = vec![
        (384, 384, "tiny self-attn (384x384)"),
        (384, 1536, "tiny FFN fc1 (384->1536)"),
        (1536, 384, "tiny FFN fc2 (1536->384)"),
        (1280, 1280, "large self-attn (1280x1280)"),
        (1280, 5120, "large FFN fc1 (1280->5120)"),
        (5120, 1280, "large FFN fc2 (5120->1280)"),
    ];

    println!("  {:42} {:>10} {:>10} {:>8}", "Layer", "f32 (us)", "fp16 (us)", "Speedup");
    println!("  {}", "-".repeat(74));

    for (in_feat, out_feat, label) in &configs {
        let (f32_us, f16_us) = bench_linear_forward(*in_feat, *out_feat);
        let speedup = f32_us / f16_us;
        let marker = if speedup > 1.5 { " <<" } else if speedup > 1.2 { " <" } else { "" };
        println!(
            "  {:42} {:>10.1} {:>10.1} {:>7.2}x{}",
            label, f32_us, f16_us, speedup, marker
        );
    }
}

fn bench_linear_forward(in_features: usize, out_features: usize) -> (f64, f64) {
    use whisper_apr::model::LinearWeights;

    // Create f32 weights with realistic values
    let mut linear_f32 = LinearWeights::new(in_features, out_features);
    for i in 0..linear_f32.weight.len() {
        linear_f32.weight[i] = ((i as f32) * 0.0001).sin() * 0.02;
    }
    linear_f32.finalize_weights();

    // Create fp16 weights (convert from f32)
    let mut linear_f16 = LinearWeights::new(in_features, out_features);
    for i in 0..linear_f16.weight.len() {
        linear_f16.weight[i] = ((i as f32) * 0.0001).sin() * 0.02;
    }
    linear_f16.convert_to_f16();

    // Create input vector
    let input: Vec<f32> = (0..in_features).map(|i| ((i as f32) * 0.01).cos() * 0.1).collect();

    // Warmup
    for _ in 0..50 {
        let _ = linear_f32.forward_simd(&input, 1);
        let _ = linear_f16.forward_simd(&input, 1);
    }

    // Benchmark f32
    let t0 = Instant::now();
    for _ in 0..MICRO_ITERS {
        let _ = linear_f32.forward_simd(&input, 1);
    }
    let f32_total = t0.elapsed();
    let f32_us = f32_total.as_secs_f64() * 1_000_000.0 / MICRO_ITERS as f64;

    // Benchmark fp16
    let t0 = Instant::now();
    for _ in 0..MICRO_ITERS {
        let _ = linear_f16.forward_simd(&input, 1);
    }
    let f16_total = t0.elapsed();
    let f16_us = f16_total.as_secs_f64() * 1_000_000.0 / MICRO_ITERS as f64;

    // Verify correctness: outputs should be nearly identical
    let out_f32 = linear_f32.forward_simd(&input, 1).unwrap();
    let out_f16 = linear_f16.forward_simd(&input, 1).unwrap();
    assert_eq!(out_f32.len(), out_f16.len());
    let max_rel_err: f32 = out_f32
        .iter()
        .zip(out_f16.iter())
        .map(|(a, b)| {
            if a.abs() > 1e-8 {
                (a - b).abs() / a.abs()
            } else {
                (a - b).abs()
            }
        })
        .fold(0.0_f32, f32::max);
    if max_rel_err > 0.05 {
        eprintln!(
            "    WARNING: fp16 output diverges: max_rel_err={max_rel_err:.4} (threshold=0.05)"
        );
    }

    (f32_us, f16_us)
}

fn e2e_benchmark(
    f32_model_path: &Path,
    f16_model_path: &Path,
    audio_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load audio once
    println!("  Loading audio: {}", audio_path.display());
    let audio_bytes = std::fs::read(audio_path)?;
    let wav = parse_wav_file(&audio_bytes)?;
    let audio_samples = if wav.sample_rate == 16000 {
        wav.samples
    } else {
        whisper_apr::audio::wav::resample(&wav.samples, wav.sample_rate, 16000)
    };
    println!("  Audio: {} samples ({:.2}s at 16kHz)", audio_samples.len(),
        audio_samples.len() as f64 / 16000.0);

    // Load f32 model
    println!("  Loading f32 model: {} ({:.1} MB)", f32_model_path.display(),
        std::fs::metadata(f32_model_path)?.len() as f64 / 1_000_000.0);
    let f32_data = std::fs::read(f32_model_path)?;
    let f32_model = whisper_apr::WhisperApr::load_from_apr(&f32_data)?;

    // Load fp16 model
    println!("  Loading fp16 model: {} ({:.1} MB)", f16_model_path.display(),
        std::fs::metadata(f16_model_path)?.len() as f64 / 1_000_000.0);
    let f16_data = std::fs::read(f16_model_path)?;
    let f16_model = whisper_apr::WhisperApr::load_from_apr(&f16_data)?;

    println!("\n  File size ratio: {:.1}%\n",
        std::fs::metadata(f16_model_path)?.len() as f64 /
        std::fs::metadata(f32_model_path)?.len() as f64 * 100.0);

    let options = || TranscribeOptions {
        language: Some("en".to_string()),
        ..Default::default()
    };

    // Warmup transcription
    println!("  Warming up...");
    let _ = f32_model.transcribe(&audio_samples, options());
    let _ = f16_model.transcribe(&audio_samples, options());

    // Benchmark f32 transcription
    println!("  Benchmarking f32 ({E2E_ITERS} iterations)...");
    let mut f32_times = Vec::new();
    let mut f32_text = String::new();
    for i in 0..E2E_ITERS {
        let t0 = Instant::now();
        let result = f32_model.transcribe(&audio_samples, options());
        let elapsed = t0.elapsed();
        f32_times.push(elapsed.as_secs_f64());
        if i == 0 {
            if let Ok(ref r) = result {
                f32_text = r.text.clone();
            }
        }
    }

    // Benchmark fp16 transcription
    println!("  Benchmarking fp16 ({E2E_ITERS} iterations)...");
    let mut f16_times = Vec::new();
    let mut f16_text = String::new();
    for i in 0..E2E_ITERS {
        let t0 = Instant::now();
        let result = f16_model.transcribe(&audio_samples, options());
        let elapsed = t0.elapsed();
        f16_times.push(elapsed.as_secs_f64());
        if i == 0 {
            if let Ok(ref r) = result {
                f16_text = r.text.clone();
            }
        }
    }

    // Compute statistics
    f32_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    f16_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let f32_median = f32_times[f32_times.len() / 2];
    let f16_median = f16_times[f16_times.len() / 2];
    let f32_mean: f64 = f32_times.iter().sum::<f64>() / f32_times.len() as f64;
    let f16_mean: f64 = f16_times.iter().sum::<f64>() / f16_times.len() as f64;
    let f32_min = f32_times[0];
    let f16_min = f16_times[0];

    println!("\n  ══════════════════════════════════════════════════════════");
    println!("  E2E Transcription Results:");
    println!("  ──────────────────────────────────────────────────────────");
    println!("  {:12} {:>10} {:>10} {:>10}", "", "Min", "Median", "Mean");
    println!("  {:12} {:>9.1}ms {:>9.1}ms {:>9.1}ms", "f32",
        f32_min * 1000.0, f32_median * 1000.0, f32_mean * 1000.0);
    println!("  {:12} {:>9.1}ms {:>9.1}ms {:>9.1}ms", "fp16",
        f16_min * 1000.0, f16_median * 1000.0, f16_mean * 1000.0);
    println!("  ──────────────────────────────────────────────────────────");
    println!("  Speedup (median): {:.2}x", f32_median / f16_median);
    println!("  Speedup (mean):   {:.2}x", f32_mean / f16_mean);
    println!("  Speedup (min):    {:.2}x", f32_min / f16_min);
    println!("  ══════════════════════════════════════════════════════════\n");

    // Verify correctness
    println!("  Correctness Check:");
    println!("  f32 output: {:?}", f32_text.trim());
    println!("  fp16 output: {:?}", f16_text.trim());
    if f32_text.trim() == f16_text.trim() {
        println!("  MATCH: Transcription text is IDENTICAL");
    } else {
        println!("  MISMATCH: Transcription text differs!");
        println!("  (Minor differences expected due to fp16 rounding in logits)");
    }

    Ok(())
}

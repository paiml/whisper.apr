//! fp16 vs f32 Weight Storage Benchmark
//!
//! Measures the actual speedup from fp16 weight storage across:
//! 1. Isolated LinearWeights::forward_simd (single-token, the hot path)
//! 2. End-to-end transcription at multiple audio durations (1.5s, 30s, 60s)
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║    fp16 vs f32 WEIGHT STORAGE + THREADING BENCHMARK         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Print system info
    println!(
        "  CPUs: {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    );
    println!("  SIMD: {}", whisper_apr::simd::backend_name());
    println!();

    // =========================================================================
    // PART 1: Isolated LinearWeights micro-benchmark
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 1: LinearWeights::forward_simd (single-token decode)   │");
    println!("│         This is the memory-bandwidth-bound hot path.        │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    micro_benchmark_linear_weights();

    // =========================================================================
    // PART 2: End-to-end transcription across audio durations
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 2: End-to-end transcription (f32 vs fp16 model)        │");
    println!("│         Multiple audio durations to isolate decoder impact.  │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    run_e2e_benchmark()?;

    // =========================================================================
    // PART 3: Summary
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PART 3: Analysis                                            │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("  Key insight: fp16 speedup comes from decoder (memory-bound matvec).");
    println!("  Short audio is encoder-dominated; longer audio exercises the decoder more.");
    println!("  Threading (rayon) parallelizes Q/K/V projections and matvec rows.");
    println!("  Larger models (base/large) benefit more from both optimizations.");
    println!();

    Ok(())
}

/// Load f32 and fp16 models, returning both or an early-exit error.
fn load_models(
) -> Result<Option<(whisper_apr::WhisperApr, whisper_apr::WhisperApr)>, Box<dyn std::error::Error>>
{
    let f32_model_path = resolve_f32_model()?;
    let f16_model_path = Path::new("models/whisper-tiny-f16.apr");

    if !f16_model_path.exists() {
        eprintln!(
            "  fp16 model not found: {}. Skipping e2e benchmark.",
            f16_model_path.display()
        );
        return Ok(None);
    }

    println!(
        "  Loading f32 model: {} ({:.1} MB)",
        f32_model_path.display(),
        std::fs::metadata(&f32_model_path)?.len() as f64 / 1_000_000.0
    );
    let f32_data = std::fs::read(&f32_model_path)?;
    let f32_model = whisper_apr::WhisperApr::load_from_apr(&f32_data)?;

    println!(
        "  Loading fp16 model: {} ({:.1} MB)",
        f16_model_path.display(),
        std::fs::metadata(f16_model_path)?.len() as f64 / 1_000_000.0
    );
    let f16_data = std::fs::read(f16_model_path)?;
    let f16_model = whisper_apr::WhisperApr::load_from_apr(&f16_data)?;

    println!(
        "  File size ratio: {:.1}%\n",
        std::fs::metadata(f16_model_path)?.len() as f64
            / std::fs::metadata(&f32_model_path)?.len() as f64
            * 100.0
    );

    Ok(Some((f32_model, f16_model)))
}

/// Run end-to-end transcription benchmarks across multiple audio durations.
fn run_e2e_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let models = load_models()?;
    let (f32_model, f16_model) = match models {
        Some(pair) => pair,
        None => return Ok(()),
    };

    let audio_files: Vec<(&str, &str, usize)> = vec![
        ("demos/test-audio/test-speech-1.5s.wav", "1.5s speech", 5),
        ("demos/test-audio/test-speech-3s.wav", "3s speech", 5),
        ("demos/test-audio/test-speech-full.wav", "~35s speech", 3),
        ("demos/test-audio/test-30s.wav", "30s mixed", 3),
        ("demos/test-audio/test-60s.wav", "60s mixed", 2),
    ];

    println!("  ══════════════════════════════════════════════════════════════════════");
    println!(
        "  {:20} {:>9} {:>9} {:>8} {:>8}  {}",
        "Audio", "f32 (ms)", "fp16 (ms)", "Speedup", "Tokens", "Match?"
    );
    println!("  ──────────────────────────────────────────────────────────────────────");

    for (path, label, iters) in &audio_files {
        bench_single_audio(&f32_model, &f16_model, path, label, *iters)?;
    }
    println!("  ══════════════════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Benchmark a single audio file with both f32 and fp16 models.
fn bench_single_audio(
    f32_model: &whisper_apr::WhisperApr,
    f16_model: &whisper_apr::WhisperApr,
    path: &str,
    label: &str,
    iters: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let audio_path = Path::new(path);
    if !audio_path.exists() {
        println!("  {:20} [not found, skipping]", label);
        return Ok(());
    }

    let audio_samples = load_audio_samples(audio_path)?;
    let duration_s = audio_samples.len() as f64 / 16000.0;

    let options = || TranscribeOptions {
        language: Some("en".to_string()),
        ..Default::default()
    };

    // Warmup
    let _ = f32_model.transcribe(&audio_samples, options());
    let _ = f16_model.transcribe(&audio_samples, options());

    // Benchmark both models
    let (f32_times, f32_text, f32_tokens) =
        run_transcription_iters(f32_model, &audio_samples, &options, iters);
    let (f16_times, f16_text, _) =
        run_transcription_iters(f16_model, &audio_samples, &options, iters);

    let f32_median = median(&f32_times);
    let f16_median = median(&f16_times);
    let speedup = f32_median / f16_median;
    let matched = f32_text.trim() == f16_text.trim();

    let marker = speedup_marker(speedup);
    println!(
        "  {:20} {:>8.1} {:>9.1} {:>7.2}x{:3} {:>6}  {}",
        format!("{label} ({duration_s:.1}s)"),
        f32_median * 1000.0,
        f16_median * 1000.0,
        speedup,
        marker,
        f32_tokens,
        if matched { "YES" } else { "DIFF" }
    );

    if label.contains("speech") && !matched {
        println!("    f32: {:?}", f32_text.trim());
        println!("    f16: {:?}", f16_text.trim());
    }

    Ok(())
}

/// Load and resample audio to 16 kHz mono.
fn load_audio_samples(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let audio_bytes = std::fs::read(path)?;
    let wav = parse_wav_file(&audio_bytes)?;
    let samples = if wav.sample_rate == 16000 {
        wav.samples
    } else {
        whisper_apr::audio::wav::resample(&wav.samples, wav.sample_rate, 16000)
    };
    Ok(samples)
}

/// Run transcription `iters` times, collecting timings and first-run text/token count.
fn run_transcription_iters(
    model: &whisper_apr::WhisperApr,
    audio_samples: &[f32],
    options: &dyn Fn() -> TranscribeOptions,
    iters: usize,
) -> (Vec<f64>, String, usize) {
    let mut times = Vec::with_capacity(iters);
    let mut text = String::new();
    let mut tokens = 0usize;
    for i in 0..iters {
        let t0 = Instant::now();
        let result = model.transcribe(audio_samples, options());
        times.push(t0.elapsed().as_secs_f64());
        if i == 0 {
            if let Ok(ref r) = result {
                text = r.text.clone();
                tokens = r.text.split_whitespace().count();
            }
        }
    }
    (times, text, tokens)
}

/// Compute median of a slice (sorts a copy).
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("NaN in benchmark timings"));
    sorted[sorted.len() / 2]
}

/// Return a visual marker for speedup significance.
fn speedup_marker(speedup: f64) -> &'static str {
    if speedup > 1.5 {
        " <<"
    } else if speedup > 1.2 {
        " <"
    } else {
        ""
    }
}

fn resolve_f32_model() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let primary = Path::new("models/whisper-tiny-fb.apr");
    if primary.exists() {
        return Ok(primary.to_path_buf());
    }
    let alt = Path::new("models/whisper-tiny.apr");
    if alt.exists() {
        return Ok(alt.to_path_buf());
    }
    Err("No f32 model found (tried whisper-tiny-fb.apr, whisper-tiny.apr)".into())
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

    println!(
        "  {:42} {:>10} {:>10} {:>8}",
        "Layer", "f32 (us)", "fp16 (us)", "Speedup"
    );
    println!("  {}", "-".repeat(74));

    for (in_feat, out_feat, label) in &configs {
        let (f32_us, f16_us) = bench_linear_forward(*in_feat, *out_feat);
        let speedup = f32_us / f16_us;
        let marker = speedup_marker(speedup);
        println!(
            "  {:42} {:>10.1} {:>10.1} {:>7.2}x{}",
            label, f32_us, f16_us, speedup, marker
        );
    }
}

fn bench_linear_forward(in_features: usize, out_features: usize) -> (f64, f64) {
    use whisper_apr::model::LinearWeights;

    let mut linear_f32 = LinearWeights::new(in_features, out_features);
    for i in 0..linear_f32.weight.len() {
        linear_f32.weight[i] = ((i as f32) * 0.0001).sin() * 0.02;
    }
    linear_f32.finalize_weights();

    let mut linear_f16 = LinearWeights::new(in_features, out_features);
    for i in 0..linear_f16.weight.len() {
        linear_f16.weight[i] = ((i as f32) * 0.0001).sin() * 0.02;
    }
    linear_f16.convert_to_f16();

    let input: Vec<f32> = (0..in_features)
        .map(|i| ((i as f32) * 0.01).cos() * 0.1)
        .collect();

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
    let f32_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / MICRO_ITERS as f64;

    // Benchmark fp16
    let t0 = Instant::now();
    for _ in 0..MICRO_ITERS {
        let _ = linear_f16.forward_simd(&input, 1);
    }
    let f16_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / MICRO_ITERS as f64;

    // Verify correctness
    let out_f32 = linear_f32.forward_simd(&input, 1).expect("f32 forward_simd failed");
    let out_f16 = linear_f16.forward_simd(&input, 1).expect("f16 forward_simd failed");
    let atol = 1e-4;
    let rtol = 0.02;
    let mut violations = 0usize;
    for (a, b) in out_f32.iter().zip(out_f16.iter()) {
        let abs_err = (a - b).abs();
        if a.abs() > atol && abs_err / a.abs() > rtol {
            violations += 1;
        }
    }
    if violations > 0 {
        eprintln!(
            "    WARNING: {violations}/{} elements exceed {rtol:.0}% rel error",
            out_f32.len()
        );
    }

    (f32_us, f16_us)
}

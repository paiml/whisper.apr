//! Pipeline Step Benchmark (WAPR-BENCH-001)
//!
//! Measures each step of the whisper.apr transcription pipeline using
//! real test audio files from demos/test-audio/.
//!
//! # Steps Measured
//!
//! - B: WAV file load (disk I/O)
//! - C: PCM parse (i16 → f32)
//! - F: Mel spectrogram computation
//! - G: Encoder forward pass
//! - H: Decoder forward pass (per-token)
//! - E2E: End-to-end transcription
//!
//! # Running
//!
//! ```bash
//! cargo bench --bench pipeline
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

/// Load WAV file bytes (Step B)
fn load_wav_bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("Failed to read WAV file")
}

/// Parse PCM from WAV bytes (Step C)
fn parse_pcm(bytes: &[u8]) -> Vec<f32> {
    // Skip 44-byte WAV header
    let pcm_data = &bytes[44..];

    // Convert i16 PCM to f32 normalized
    pcm_data
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}

/// Benchmark WAV loading (Step B)
fn bench_step_b_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_b_load");

    let files = [
        ("1.5s", "demos/test-audio/test-speech-1.5s.wav", 48078),
        ("3s", "demos/test-audio/test-speech-3s.wav", 96078),
        ("full", "demos/test-audio/test-speech-full.wav", 1076018),
    ];

    for (name, path, size) in files {
        let path = Path::new(path);
        if !path.exists() {
            continue;
        }

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("wav", name), &path, |b, path| {
            b.iter(|| black_box(load_wav_bytes(path)));
        });
    }

    group.finish();
}

/// Benchmark PCM parsing (Step C)
fn bench_step_c_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_c_parse");

    let files = [
        ("1.5s", "demos/test-audio/test-speech-1.5s.wav"),
        ("3s", "demos/test-audio/test-speech-3s.wav"),
        ("full", "demos/test-audio/test-speech-full.wav"),
    ];

    for (name, path) in files {
        let path = Path::new(path);
        if !path.exists() {
            continue;
        }

        let bytes = load_wav_bytes(path);
        let sample_count = (bytes.len() - 44) / 2;

        group.throughput(Throughput::Elements(sample_count as u64));

        group.bench_with_input(BenchmarkId::new("pcm", name), &bytes, |b, bytes| {
            b.iter(|| black_box(parse_pcm(bytes)));
        });
    }

    group.finish();
}

/// Benchmark mel spectrogram computation (Step F)
fn bench_step_f_mel(c: &mut Criterion) {
    use whisper_apr::audio::MelFilterbank;

    let mut group = c.benchmark_group("step_f_mel");

    let mel = MelFilterbank::new(80, 400, 16000);

    let files = [
        ("1.5s", "demos/test-audio/test-speech-1.5s.wav"),
        ("3s", "demos/test-audio/test-speech-3s.wav"),
    ];

    for (name, path) in files {
        let path = Path::new(path);
        if !path.exists() {
            continue;
        }

        let bytes = load_wav_bytes(path);
        let samples = parse_pcm(&bytes);
        let duration_ms = (samples.len() as f32 / 16.0) as u64;

        group.throughput(Throughput::Elements(duration_ms));

        // Scalar version
        group.bench_with_input(BenchmarkId::new("scalar", name), &samples, |b, samples| {
            b.iter(|| black_box(mel.compute(samples, 160)));
        });

        // SIMD version
        group.bench_with_input(BenchmarkId::new("simd", name), &samples, |b, samples| {
            b.iter(|| black_box(mel.compute_simd(samples, 160)));
        });
    }

    group.finish();
}

/// Standalone timing report for full pipeline
///
/// This runs outside criterion to provide a human-readable timing breakdown.
fn bench_pipeline_timing_report(c: &mut Criterion) {
    use whisper_apr::audio::MelFilterbank;

    let mut group = c.benchmark_group("pipeline_timing");
    group.sample_size(10); // Fewer samples for slow operations

    let path = Path::new("demos/test-audio/test-speech-1.5s.wav");
    if !path.exists() {
        eprintln!("Test audio file not found: {}", path.display());
        return;
    }

    // Pre-load everything
    let bytes = load_wav_bytes(path);
    let samples = parse_pcm(&bytes);
    let mel_filterbank = MelFilterbank::new(80, 400, 16000);

    // Benchmark just mel computation for the 1.5s file
    let duration_ms = (samples.len() as f32 / 16.0) as u64;
    group.throughput(Throughput::Elements(duration_ms));

    group.bench_function("mel_1.5s", |b| {
        b.iter(|| black_box(mel_filterbank.compute(&samples, 160)));
    });

    group.finish();
}

/// Benchmark encoder forward pass (Step G)
/// Requires model file: models/whisper-tiny-int8.apr
#[allow(unused_imports)]
fn bench_step_g_encoder(c: &mut Criterion) {
    use whisper_apr::audio::MelFilterbank;

    let model_path = Path::new("models/whisper-tiny-int8.apr");
    if !model_path.exists() {
        eprintln!(
            "⚠️  Model not found: {}. Skipping encoder benchmark.",
            model_path.display()
        );
        return;
    }

    let audio_path = Path::new("demos/test-audio/test-speech-1.5s.wav");
    if !audio_path.exists() {
        eprintln!(
            "⚠️  Audio not found: {}. Skipping encoder benchmark.",
            audio_path.display()
        );
        return;
    }

    // Load model
    let model_bytes = std::fs::read(model_path).expect("read model");
    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes).expect("load model");

    // Prepare audio
    let wav_bytes = load_wav_bytes(audio_path);
    let samples = parse_pcm(&wav_bytes);

    // Compute mel
    let mel = model.compute_mel(&samples).expect("compute mel");

    let mut group = c.benchmark_group("step_g_encoder");
    group.sample_size(10); // Encoder is slow

    group.bench_function("tiny_int8_1.5s", |b| {
        b.iter(|| {
            let result = model.encoder_mut().forward_mel(black_box(&mel));
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark decoder forward pass - single token (Step H)
/// Requires model file: models/whisper-tiny-int8.apr
fn bench_step_h_decoder(c: &mut Criterion) {
    let model_path = Path::new("models/whisper-tiny-int8.apr");
    if !model_path.exists() {
        eprintln!(
            "⚠️  Model not found: {}. Skipping decoder benchmark.",
            model_path.display()
        );
        return;
    }

    let audio_path = Path::new("demos/test-audio/test-speech-1.5s.wav");
    if !audio_path.exists() {
        eprintln!(
            "⚠️  Audio not found: {}. Skipping decoder benchmark.",
            audio_path.display()
        );
        return;
    }

    // Load model
    let model_bytes = std::fs::read(model_path).expect("read model");
    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes).expect("load model");

    // Prepare audio and encode
    let wav_bytes = load_wav_bytes(audio_path);
    let samples = parse_pcm(&wav_bytes);
    let mel = model.compute_mel(&samples).expect("compute mel");
    let encoder_output = model.encoder_mut().forward_mel(&mel).expect("encode");

    // Initial tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    let initial_tokens = vec![50258_u32, 50259, 50359, 50363];

    let mut group = c.benchmark_group("step_h_decoder");
    group.sample_size(10); // Decoder is slow

    group.bench_function("tiny_int8_single_step", |b| {
        b.iter(|| {
            let result = model
                .decoder_mut()
                .forward(black_box(&initial_tokens), black_box(&encoder_output), None);
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark end-to-end transcription (E2E)
/// Requires model file: models/whisper-tiny-int8.apr
fn bench_e2e_transcription(c: &mut Criterion) {
    let model_path = Path::new("models/whisper-tiny-int8.apr");
    if !model_path.exists() {
        eprintln!(
            "⚠️  Model not found: {}. Skipping E2E benchmark.",
            model_path.display()
        );
        return;
    }

    let audio_path = Path::new("demos/test-audio/test-speech-1.5s.wav");
    if !audio_path.exists() {
        eprintln!(
            "⚠️  Audio not found: {}. Skipping E2E benchmark.",
            audio_path.display()
        );
        return;
    }

    // Load model once
    let model_bytes = std::fs::read(model_path).expect("read model");
    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes).expect("load model");

    // Prepare audio
    let wav_bytes = load_wav_bytes(audio_path);
    let samples = parse_pcm(&wav_bytes);

    let mut group = c.benchmark_group("e2e_transcription");
    group.sample_size(10); // E2E is slow

    // Measure first token latency
    group.bench_function("first_token_1.5s", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();

                // Step F: Mel
                let mel = model.compute_mel(black_box(&samples)).expect("mel");

                // Step G: Encode
                let encoder_output = model.encoder_mut().forward_mel(&mel).expect("encode");

                // Step H: First decode step
                let initial_tokens = vec![50258_u32, 50259, 50359, 50363];
                let _ = model
                    .decoder_mut()
                    .forward(&initial_tokens, &encoder_output, None);

                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

/// Print detailed timing breakdown (non-criterion)
pub fn print_timing_breakdown() {
    use whisper_apr::audio::MelFilterbank;

    println!("\n========================================");
    println!(" WHISPER.APR PIPELINE TIMING BREAKDOWN");
    println!("========================================\n");

    let files = [
        ("1.5s", "demos/test-audio/test-speech-1.5s.wav"),
        ("3s", "demos/test-audio/test-speech-3s.wav"),
    ];

    for (name, path) in files {
        let path = Path::new(path);
        if !path.exists() {
            println!("⚠️  {} not found, skipping\n", path.display());
            continue;
        }

        println!("📁 Audio: {} ({})", name, path.display());
        println!("─────────────────────────────────────");

        // Step B: Load WAV
        let t0 = Instant::now();
        let bytes = load_wav_bytes(path);
        let load_time = t0.elapsed();
        println!(
            "[B] WAV Load:      {:>8.2}ms ({} bytes)",
            load_time.as_secs_f64() * 1000.0,
            bytes.len()
        );

        // Step C: Parse PCM
        let t0 = Instant::now();
        let samples = parse_pcm(&bytes);
        let parse_time = t0.elapsed();
        let duration_secs = samples.len() as f32 / 16000.0;
        println!(
            "[C] PCM Parse:     {:>8.2}ms ({} samples, {:.2}s)",
            parse_time.as_secs_f64() * 1000.0,
            samples.len(),
            duration_secs
        );

        // Step F: Mel spectrogram
        let mel_filterbank = MelFilterbank::new(80, 400, 16000);

        let t0 = Instant::now();
        let mel_result = mel_filterbank.compute(&samples, 160);
        let mel_time = t0.elapsed();

        match &mel_result {
            Ok(mel) => {
                let frames = mel.len() / 80;
                println!(
                    "[F] Mel Compute:   {:>8.2}ms ({} frames)",
                    mel_time.as_secs_f64() * 1000.0,
                    frames
                );
            }
            Err(e) => {
                println!("[F] Mel Compute:   FAILED - {:?}", e);
            }
        }

        // Step F (SIMD): Mel spectrogram with SIMD
        let t0 = Instant::now();
        let mel_simd_result = mel_filterbank.compute_simd(&samples, 160);
        let mel_simd_time = t0.elapsed();

        match &mel_simd_result {
            Ok(mel) => {
                let frames = mel.len() / 80;
                let speedup = mel_time.as_secs_f64() / mel_simd_time.as_secs_f64();
                println!(
                    "[F] Mel SIMD:      {:>8.2}ms ({} frames, {:.1}x speedup)",
                    mel_simd_time.as_secs_f64() * 1000.0,
                    frames,
                    speedup
                );
            }
            Err(e) => {
                println!("[F] Mel SIMD:      FAILED - {:?}", e);
            }
        }

        // Total preprocessing time
        let preprocess_total = load_time + parse_time + mel_time;
        println!("────────────────────────────────────");
        println!(
            "    Preprocess:    {:>8.2}ms",
            preprocess_total.as_secs_f64() * 1000.0
        );

        // RTF for preprocessing only
        let preprocess_rtf = preprocess_total.as_secs_f64() / duration_secs as f64;
        println!("    Preprocess RTF: {:>7.3}x", preprocess_rtf);
        println!();
    }

    println!("========================================");
    println!("Note: For encoder/decoder timing with model:");
    println!("  cargo run --example format_comparison --release");
    println!("========================================\n");
}

criterion_group!(
    benches,
    bench_step_b_load,
    bench_step_c_parse,
    bench_step_f_mel,
    bench_step_g_encoder,
    bench_step_h_decoder,
    bench_e2e_transcription,
    bench_pipeline_timing_report,
);

criterion_main!(benches);

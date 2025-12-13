//! Benchmarks for Whisper.apr inference performance
//!
//! This benchmark suite measures end-to-end transcription performance
//! and component-level latency for optimization.
//!
//! # Benchmark Methodology
//!
//! - Tests multiple audio durations: 5s, 15s, 30s
//! - Compares model sizes: tiny, base
//! - Measures Real-Time Factor (RTF)
//! - Uses Criterion for statistical analysis
//!
//! # Performance Targets
//!
//! | Model | Target RTF | Memory Peak |
//! |-------|------------|-------------|
//! | tiny  | ≤2.0x      | ≤150MB      |
//! | base  | ≤2.5x      | ≤350MB      |
//!
//! RTF = processing_time / audio_duration
//! (Lower is better, 1.0 = real-time)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use whisper_apr::audio::MelFilterbank;

/// Generate synthetic audio samples (16kHz mono f32)
fn generate_audio(duration_secs: f32) -> Vec<f32> {
    let sample_rate = 16000;
    let num_samples = (duration_secs * sample_rate as f32) as usize;

    // Generate simple sine wave for benchmarking
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (440.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.5
        })
        .collect()
}

/// Benchmark mel spectrogram computation
fn bench_mel_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_spectrogram");

    let mel = MelFilterbank::new(80, 400, 16000);

    for duration in [5.0, 15.0, 30.0] {
        let audio = generate_audio(duration);
        let audio_duration_ms = (duration * 1000.0) as u64;

        group.throughput(Throughput::Elements(audio_duration_ms));

        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{duration}s")),
            &audio,
            |bencher, audio| {
                bencher.iter(|| {
                    let mel_spec = mel.compute(audio, 160).ok();
                    black_box(mel_spec);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd", format!("{duration}s")),
            &audio,
            |bencher, audio| {
                bencher.iter(|| {
                    let mel_spec = mel.compute_simd(audio, 160).ok();
                    black_box(mel_spec);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark encoder forward pass
fn bench_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoder");

    // Simulated mel spectrogram dimensions
    // 30s audio @ 16kHz with hop=160 = 3000 frames
    for (name, frames) in [("5s", 500), ("15s", 1500), ("30s", 3000)] {
        group.throughput(Throughput::Elements(frames as u64));

        group.bench_with_input(
            BenchmarkId::new("forward", name),
            &frames,
            |bencher, &frames| {
                // Placeholder: simulate encoder computation
                let input = vec![0.0f32; frames * 80]; // frames x n_mels

                bencher.iter(|| {
                    black_box(&input);
                    // Actual encoder forward pass will go here
                });
            },
        );
    }

    group.finish();
}

/// Benchmark decoder with greedy decoding
fn bench_decoder_greedy(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder_greedy");

    // Typical output lengths for different audio durations
    // ~4 tokens per second of audio
    for (name, max_tokens) in [("5s", 20), ("15s", 60), ("30s", 120)] {
        group.throughput(Throughput::Elements(max_tokens as u64));

        group.bench_with_input(
            BenchmarkId::new("decode", name),
            &max_tokens,
            |bencher, &max_tokens| {
                bencher.iter(|| {
                    // Placeholder: simulate greedy decoding
                    let tokens: Vec<u32> = (0..max_tokens as u32).collect();
                    black_box(tokens);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark decoder with beam search
fn bench_decoder_beam(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder_beam");
    group.sample_size(50); // Beam search is slower, reduce samples

    for beam_size in [3, 5, 10] {
        for (name, max_tokens) in [("5s", 20), ("15s", 60)] {
            group.throughput(Throughput::Elements(max_tokens as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("beam_{beam_size}"), name),
                &(beam_size, max_tokens),
                |bencher, &(beam_size, max_tokens)| {
                    bencher.iter(|| {
                        // Placeholder: simulate beam search
                        let beams: Vec<Vec<u32>> = (0..beam_size)
                            .map(|_| (0..max_tokens as u32).collect())
                            .collect();
                        black_box(beams);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark end-to-end transcription (Real-Time Factor)
fn bench_transcribe_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcribe_e2e");
    group.sample_size(20); // Full pipeline is slow

    for duration in [5.0, 15.0, 30.0] {
        let audio = generate_audio(duration);
        let audio_duration_ms = (duration * 1000.0) as u64;

        group.throughput(Throughput::Elements(audio_duration_ms));

        group.bench_with_input(
            BenchmarkId::new("tiny", format!("{duration}s")),
            &audio,
            |bencher, audio| {
                bencher.iter(|| {
                    // Placeholder: full transcription pipeline
                    black_box(audio.len());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark tokenizer encode/decode
fn bench_tokenizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer");

    let texts = [
        ("short", "Hello world"),
        ("medium", "The quick brown fox jumps over the lazy dog."),
        (
            "long",
            "This is a longer piece of text that contains multiple sentences. \
             It should test the tokenizer's performance on realistic transcription output.",
        ),
    ];

    for (name, text) in texts {
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::new("encode", name), &text, |bencher, &text| {
            bencher.iter(|| {
                // Placeholder: actual BPE encoding
                let tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
                black_box(tokens);
            });
        });

        group.bench_with_input(BenchmarkId::new("decode", name), &text, |bencher, &text| {
            let tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
            bencher.iter(|| {
                // Placeholder: actual BPE decoding
                let decoded: String = tokens.iter().map(|&t| t as u8 as char).collect();
                black_box(decoded);
            });
        });
    }

    group.finish();
}

/// Benchmark attention computation (SIMD comparison)
fn bench_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");

    // Whisper attention dimensions
    // tiny: d_model=384, n_heads=6, head_dim=64
    // base: d_model=512, n_heads=8, head_dim=64
    for (name, seq_len, d_model) in [
        ("tiny_short", 100, 384),
        ("tiny_long", 1500, 384),
        ("base_short", 100, 512),
        ("base_long", 1500, 512),
    ] {
        let elements = seq_len * d_model;
        group.throughput(Throughput::Elements(elements as u64));

        group.bench_with_input(
            BenchmarkId::new("forward", name),
            &(seq_len, d_model),
            |bencher, &(seq_len, d_model)| {
                let q = vec![0.0f32; seq_len * d_model];
                let k = vec![0.0f32; seq_len * d_model];
                let v = vec![0.0f32; seq_len * d_model];

                bencher.iter(|| {
                    // Placeholder: actual attention computation
                    black_box((&q, &k, &v));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_mel_spectrogram,
    bench_encoder,
    bench_decoder_greedy,
    bench_decoder_beam,
    bench_transcribe_e2e,
    bench_tokenizer,
    bench_attention,
);

criterion_main!(benches);

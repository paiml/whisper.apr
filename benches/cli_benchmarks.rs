//! Benchmarks for whisper-apr-cli components
//!
//! Run with: `cargo bench --bench cli_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use whisper_apr::audio::wav::{parse_wav, resample};

/// Create a test WAV file with given duration in seconds
fn create_test_wav(duration_secs: f32, sample_rate: u32) -> Vec<u8> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let data_size = (num_samples * 2) as u32;
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + num_samples * 2);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());

    // Generate sine wave samples
    let freq = 440.0f32;
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * freq * t).sin();
        let sample_i16 = (sample * 32767.0) as i16;
        wav.extend_from_slice(&sample_i16.to_le_bytes());
    }

    wav
}

// =============================================================================
// BENCHMARKS
// =============================================================================

fn bench_wav_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("wav_parsing");

    for duration in [1.0, 10.0, 60.0] {
        let wav = create_test_wav(duration, 16000);
        let _samples = (duration * 16000.0) as usize;

        group.throughput(Throughput::Bytes(wav.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("parse", format!("{}s", duration)),
            &wav,
            |b, wav| {
                b.iter(|| parse_wav(black_box(wav)));
            },
        );
    }

    group.finish();
}

fn bench_resample(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample");

    // 48kHz to 16kHz (common case)
    let samples_48k: Vec<f32> = (0..48000).map(|i| (i as f32 / 48000.0).sin()).collect();
    group.throughput(Throughput::Elements(samples_48k.len() as u64));
    group.bench_function("48k_to_16k_1s", |b| {
        b.iter(|| resample(black_box(&samples_48k), 48000, 16000));
    });

    // 8kHz to 16kHz (upsampling)
    let samples_8k: Vec<f32> = (0..8000).map(|i| (i as f32 / 8000.0).sin()).collect();
    group.throughput(Throughput::Elements(samples_8k.len() as u64));
    group.bench_function("8k_to_16k_1s", |b| {
        b.iter(|| resample(black_box(&samples_8k), 8000, 16000));
    });

    // 44.1kHz to 16kHz
    let samples_44k: Vec<f32> = (0..44100).map(|i| (i as f32 / 44100.0).sin()).collect();
    group.throughput(Throughput::Elements(samples_44k.len() as u64));
    group.bench_function("44.1k_to_16k_1s", |b| {
        b.iter(|| resample(black_box(&samples_44k), 44100, 16000));
    });

    group.finish();
}

fn bench_wav_parsing_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("wav_throughput");

    // Measure samples per second parsing throughput
    for duration in [0.1, 1.0, 5.0] {
        let wav = create_test_wav(duration, 16000);
        let expected_samples = (duration * 16000.0) as u64;

        group.throughput(Throughput::Elements(expected_samples));
        group.bench_with_input(
            BenchmarkId::new("samples_per_sec", format!("{}s", duration)),
            &wav,
            |b, wav| {
                b.iter(|| {
                    let wav_data = parse_wav(black_box(wav)).expect("valid wav");
                    wav_data.samples.len()
                });
            },
        );
    }

    group.finish();
}

fn bench_resample_various_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample_ratios");

    let _samples: Vec<f32> = (0..16000).map(|i| (i as f32 / 16000.0).sin()).collect();

    // Test various resampling ratios
    let ratios = [
        (48000u32, 16000u32, "3:1"),
        (44100, 16000, "2.76:1"),
        (22050, 16000, "1.38:1"),
        (16000, 16000, "1:1"),
        (8000, 16000, "1:2"),
    ];

    for (src, dst, label) in ratios {
        // Adjust sample count for source rate
        let adjusted_samples: Vec<f32> = (0..(src as usize))
            .map(|i| (i as f32 / src as f32).sin())
            .collect();

        group.throughput(Throughput::Elements(adjusted_samples.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("ratio", label),
            &adjusted_samples,
            |b, samples| {
                b.iter(|| resample(black_box(samples), src, dst));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_wav_parsing,
    bench_resample,
    bench_wav_parsing_throughput,
    bench_resample_various_ratios,
);

criterion_main!(benches);

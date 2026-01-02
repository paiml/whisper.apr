//! Benchmarks comparing model format efficiency
//!
//! Compares GGML, APR (f32/int8), and SafeTensors formats for:
//! - File size (compression ratio)
//! - Loading time (parse + deserialize)
//! - Memory usage (peak allocation)
//! - First-token latency
//!
//! This is the core value proposition of whisper.apr:
//! Efficient model delivery for WASM environments.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

/// Model format metadata
#[derive(Debug, Clone)]
struct FormatInfo {
    name: &'static str,
    path: &'static str,
    file_size: u64,
}

fn get_format_infos() -> Vec<FormatInfo> {
    let formats = [
        (
            "SafeTensors",
            "/home/noah/.cache/whisper-apr/whisper-tiny.safetensors",
        ),
        ("GGML", "/home/noah/.cache/whisper-apr/ggml-tiny.bin"),
        ("APR-f32", "models/whisper-tiny.apr"),
        ("APR-int8", "models/whisper-tiny-int8.apr"),
    ];

    formats
        .iter()
        .filter_map(|(name, path)| {
            if Path::new(path).exists() {
                let file_size = fs::metadata(path).ok()?.len();
                Some(FormatInfo {
                    name,
                    path,
                    file_size,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Benchmark: File read (I/O only)
fn bench_file_read(c: &mut Criterion) {
    let formats = get_format_infos();
    if formats.is_empty() {
        eprintln!("No model files found for benchmarking");
        return;
    }

    let mut group = c.benchmark_group("format_file_read");

    for format in &formats {
        group.throughput(Throughput::Bytes(format.file_size));
        group.bench_with_input(
            BenchmarkId::new("read", format.name),
            &format.path,
            |b, path| {
                b.iter(|| {
                    let data = fs::read(path).expect("read file");
                    black_box(data.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: APR parsing (header + index)
fn bench_apr_parse(c: &mut Criterion) {
    let apr_formats: Vec<_> = get_format_infos()
        .into_iter()
        .filter(|f| f.name.starts_with("APR"))
        .collect();

    if apr_formats.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("apr_parse");

    for format in &apr_formats {
        let data = fs::read(format.path).expect("read APR");
        let data_len = data.len();

        group.throughput(Throughput::Bytes(data_len as u64));
        group.bench_with_input(BenchmarkId::new("parse", format.name), &data, |b, data| {
            b.iter(|| {
                let reader = whisper_apr::format::AprReader::new(data.clone()).expect("parse APR");
                black_box(reader.n_tensors())
            });
        });
    }

    group.finish();
}

/// Benchmark: Full model load (APR only - our target format)
fn bench_model_load(c: &mut Criterion) {
    let apr_formats: Vec<_> = get_format_infos()
        .into_iter()
        .filter(|f| f.name.starts_with("APR"))
        .collect();

    if apr_formats.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("model_load");
    group.sample_size(10); // Model loading is slow

    for format in &apr_formats {
        let data = fs::read(format.path).expect("read APR");

        group.throughput(Throughput::Bytes(format.file_size));
        group.bench_with_input(BenchmarkId::new("load", format.name), &data, |b, data| {
            b.iter(|| {
                let model = whisper_apr::WhisperApr::load_from_apr(data).expect("load model");
                black_box(model.config().n_vocab)
            });
        });
    }

    group.finish();
}

/// Benchmark: Single tensor load (dequantization cost)
fn bench_tensor_load(c: &mut Criterion) {
    let apr_formats: Vec<_> = get_format_infos()
        .into_iter()
        .filter(|f| f.name.starts_with("APR"))
        .collect();

    if apr_formats.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("tensor_load");

    // Test loading a medium-sized tensor (attention weights)
    let tensor_name = "decoder.layers.0.self_attn.q_proj.weight";

    for format in &apr_formats {
        let data = fs::read(format.path).expect("read APR");
        let reader = whisper_apr::format::AprReader::new(data).expect("parse APR");

        group.bench_with_input(
            BenchmarkId::new("q_proj_weight", format.name),
            &reader,
            |b, reader| {
                b.iter(|| {
                    let tensor = reader.load_tensor(tensor_name).expect("load tensor");
                    black_box(tensor.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: First token latency (encoder + 1 decoder step)
fn bench_first_token(c: &mut Criterion) {
    // Only test with int8 model for speed
    let model_path = "models/whisper-tiny-int8.apr";
    if !Path::new(model_path).exists() {
        return;
    }

    let data = fs::read(model_path).expect("read model");
    let mut model = whisper_apr::WhisperApr::load_from_apr(&data).expect("load model");

    // Generate 1 second of audio
    let sample_rate = 16000;
    let audio: Vec<f32> = (0..sample_rate)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (440.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.3
        })
        .collect();

    // Pre-compute mel spectrogram
    let mel = model.compute_mel(&audio).expect("compute mel");

    let mut group = c.benchmark_group("first_token_latency");
    group.sample_size(20);

    // Benchmark encoder
    group.bench_function("encoder", |b| {
        b.iter(|| {
            let output = model.encoder_mut().forward_mel(&mel).expect("encode");
            black_box(output.len())
        });
    });

    // Benchmark decoder (1 step)
    let encoder_output = model.encoder_mut().forward_mel(&mel).expect("encode");
    let initial_tokens = vec![50258_u32, 50259, 50359, 50363]; // SOT, en, transcribe, notimestamps

    group.bench_function("decoder_1step", |b| {
        b.iter(|| {
            let logits = model
                .decoder_mut()
                .forward(&initial_tokens, &encoder_output, None)
                .expect("decode");
            black_box(logits.len())
        });
    });

    group.finish();
}

/// Print format comparison summary
fn print_format_summary() {
    println!("\n=== Model Format Comparison ===\n");

    let formats = get_format_infos();
    let safetensors_size = formats
        .iter()
        .find(|f| f.name == "SafeTensors")
        .map(|f| f.file_size)
        .unwrap_or(1);

    println!("{:<15} {:>12} {:>12}", "Format", "Size", "Ratio");
    println!("{:-<15} {:-<12} {:-<12}", "", "", "");

    for format in &formats {
        let ratio = format.file_size as f64 / safetensors_size as f64;
        let size_mb = format.file_size as f64 / 1_000_000.0;
        println!(
            "{:<15} {:>10.1} MB {:>11.1}%",
            format.name,
            size_mb,
            ratio * 100.0
        );
    }

    println!();
}

criterion_group!(
    benches,
    bench_file_read,
    bench_apr_parse,
    bench_tensor_load,
    bench_model_load,
    bench_first_token,
);

criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_summary() {
        print_format_summary();
    }
}

//! Benchmarks for WASM SIMD operations
//!
//! This benchmark suite compares scalar vs WASM SIMD 128-bit performance
//! for core Whisper operations.
//!
//! # Running WASM Benchmarks
//!
//! These benchmarks require running in a browser environment or
//! via wasm-pack with a headless browser.
//!
//! ```bash
//! # Build WASM benchmarks
//! wasm-pack build --target web --release
//!
//! # Run with wasm-bindgen-test
//! wasm-pack test --headless --chrome
//! ```
//!
//! # Performance Targets
//!
//! Expected SIMD speedup over scalar:
//! - Matrix multiply: 3-4x
//! - Element-wise ops: 3-4x
//! - Softmax: 2-3x
//! - Dot product: 3-4x

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use whisper_apr::simd;

/// Generate test matrix data
fn generate_matrix(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|i| (i as f32) * 0.001).collect()
}

/// Benchmark matrix multiplication (critical for attention)
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    // Typical attention matrix sizes
    // Q×K^T: [seq_len, head_dim] × [head_dim, seq_len]
    for (name, m, k, n) in [
        ("small", 64, 64, 64),
        ("attention_tiny", 100, 64, 100), // tiny model attention
        ("attention_base", 100, 64, 100), // base model attention
        ("encoder_ctx", 1500, 64, 1500),  // full context
    ] {
        let ops = m * k * n * 2; // multiply-add per element
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &(m, k, n),
            |bencher, &(m, k, n)| {
                let a = generate_matrix(m, k);
                let b = generate_matrix(k, n);
                let mut c = vec![0.0f32; m * n];

                bencher.iter(|| {
                    // Naive scalar matmul
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0f32;
                            for p in 0..k {
                                sum += a[i * k + p] * b[p * n + j];
                            }
                            c[i * n + j] = sum;
                        }
                    }
                    black_box(&c);
                });
            },
        );

        // SIMD version using actual SIMD matmul
        group.bench_with_input(
            BenchmarkId::new("simd", name),
            &(m, k, n),
            |bencher, &(m, k, n)| {
                let a = generate_matrix(m, k);
                let b = generate_matrix(k, n);

                bencher.iter(|| {
                    let c = simd::matmul(&a, &b, m, k, n);
                    black_box(c);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark softmax (used in attention)
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for size in [64, 256, 1024, 4096] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                // Scalar softmax with numerical stability
                let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = input.iter().map(|x| (x - max).exp()).sum();
                for (i, x) in input.iter().enumerate() {
                    output[i] = (x - max).exp() / sum;
                }
                black_box(&output);
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

            bencher.iter(|| {
                let output = simd::softmax(&input);
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark dot product (used in attention scores)
fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [64, 256, 1024, 4096, 16384] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bencher, &size| {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002).collect();

            bencher.iter(|| {
                let result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bencher, &size| {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002).collect();

            bencher.iter(|| {
                let result = simd::dot(&a, &b);
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GELU activation (used in FFN)
fn bench_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu");

    for size in [384, 512, 1536, 2048] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 5.0).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                for (i, &x) in input.iter().enumerate() {
                    // GELU approximation
                    output[i] =
                        0.5 * x * (1.0 + ((0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()));
                }
                black_box(&output);
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 5.0).collect();

            bencher.iter(|| {
                let output = simd::gelu(&input);
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark layer normalization
fn bench_layer_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm");

    for hidden_size in [384, 512, 768, 1024] {
        group.throughput(Throughput::Elements(hidden_size as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", hidden_size),
            &hidden_size,
            |bencher, &hidden_size| {
                let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.01).collect();
                let gamma: Vec<f32> = vec![1.0; hidden_size];
                let beta: Vec<f32> = vec![0.0; hidden_size];
                let mut output = vec![0.0f32; hidden_size];

                bencher.iter(|| {
                    let mean: f32 = input.iter().sum::<f32>() / hidden_size as f32;
                    let var: f32 =
                        input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_size as f32;
                    let std = (var + 1e-5).sqrt();

                    for i in 0..hidden_size {
                        output[i] = (input[i] - mean) / std * gamma[i] + beta[i];
                    }
                    black_box(&output);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd", hidden_size),
            &hidden_size,
            |bencher, &hidden_size| {
                let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.01).collect();
                let gamma: Vec<f32> = vec![1.0; hidden_size];
                let beta: Vec<f32> = vec![0.0; hidden_size];

                bencher.iter(|| {
                    let output = simd::layer_norm(&input, &gamma, &beta, 1e-5);
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_softmax,
    bench_dot_product,
    bench_gelu,
    bench_layer_norm,
);

criterion_main!(benches);

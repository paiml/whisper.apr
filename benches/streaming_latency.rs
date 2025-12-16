//! Streaming Latency Benchmarks (WAPR-112)
//!
//! Measures latency characteristics of the streaming inference pipeline:
//! - Chunk processing latency (time to process one audio chunk)
//! - State machine transition overhead
//! - KV cache operations
//! - End-to-end streaming latency
//!
//! # Latency Targets
//!
//! | Mode        | Chunk Size | Target Latency |
//! |-------------|------------|----------------|
//! | Standard    | 30s        | < 5000ms       |
//! | Low-latency | 500ms      | < 200ms        |
//! | Ultra-low   | 250ms      | < 100ms        |

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use whisper_apr::audio::{StreamingConfig, StreamingProcessor, LOW_LATENCY_CHUNK_DURATION};
use whisper_apr::model::{ModelConfig, StreamingCacheStats, StreamingKVCache};

/// Generate synthetic audio samples at specified sample rate
fn generate_audio(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Simple speech-like signal: fundamental + harmonics
            let f0 = 150.0; // typical speech fundamental
            let signal = (f0 * 2.0 * std::f32::consts::PI * t).sin() * 0.3
                + (f0 * 2.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.2
                + (f0 * 3.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.1;
            signal
        })
        .collect()
}

/// Benchmark streaming processor initialization
fn bench_processor_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_init");

    group.bench_function("standard_config", |b| {
        b.iter(|| {
            let config = StreamingConfig::default();
            black_box(StreamingProcessor::new(config))
        });
    });

    group.bench_function("low_latency_config", |b| {
        b.iter(|| {
            let config = StreamingConfig::low_latency();
            black_box(StreamingProcessor::new(config))
        });
    });

    group.bench_function("ultra_low_latency_config", |b| {
        b.iter(|| {
            let config = StreamingConfig::ultra_low_latency();
            black_box(StreamingProcessor::new(config))
        });
    });

    group.finish();
}

/// Benchmark chunk processing latency for different modes
fn bench_chunk_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_processing");
    group.sample_size(100);

    // Standard mode: 30s chunks
    {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            enable_vad: false,
            ..StreamingConfig::default()
        };
        let chunk_samples = config.chunk_samples();
        let audio = generate_audio(30.0, 16000);

        group.throughput(Throughput::Elements(chunk_samples as u64));
        group.bench_with_input(BenchmarkId::new("standard", "30s"), &audio, |b, audio| {
            b.iter(|| {
                let mut processor = StreamingProcessor::new(config.clone());
                processor.push_audio(audio);
                processor.process();
                black_box(processor.state())
            });
        });
    }

    // Low-latency mode: 500ms chunks
    {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            enable_vad: false,
            ..StreamingConfig::low_latency()
        };
        let chunk_samples = config.chunk_samples();
        let audio = generate_audio(LOW_LATENCY_CHUNK_DURATION, 16000);

        group.throughput(Throughput::Elements(chunk_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("low_latency", "500ms"),
            &audio,
            |b, audio| {
                b.iter(|| {
                    let mut processor = StreamingProcessor::new(config.clone());
                    processor.push_audio(audio);
                    processor.process();
                    black_box(processor.state())
                });
            },
        );
    }

    // Ultra-low latency: 250ms chunks
    {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            enable_vad: false,
            ..StreamingConfig::ultra_low_latency()
        };
        let chunk_samples = config.chunk_samples();
        let audio = generate_audio(0.25, 16000);

        group.throughput(Throughput::Elements(chunk_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("ultra_low", "250ms"),
            &audio,
            |b, audio| {
                b.iter(|| {
                    let mut processor = StreamingProcessor::new(config.clone());
                    processor.push_audio(audio);
                    processor.process();
                    black_box(processor.state())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark streaming KV cache operations
fn bench_kv_cache_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache");
    group.sample_size(100);

    let config = ModelConfig::tiny();
    let d_model = config.n_text_state as usize;
    let n_layers = config.n_text_layer as usize;

    // Cache creation
    group.bench_function("create_standard", |b| {
        b.iter(|| black_box(StreamingKVCache::standard(n_layers, d_model)));
    });

    group.bench_function("create_low_latency", |b| {
        b.iter(|| black_box(StreamingKVCache::low_latency(n_layers, d_model)));
    });

    group.bench_function("create_ultra_low", |b| {
        b.iter(|| black_box(StreamingKVCache::ultra_low_latency(n_layers, d_model)));
    });

    // Cache append (single token)
    {
        let key = vec![0.5_f32; d_model];
        let value = vec![0.5_f32; d_model];

        group.bench_function("append_single", |b| {
            let mut cache = StreamingKVCache::low_latency(n_layers, d_model);
            b.iter(|| {
                if cache.will_slide() {
                    cache.full_reset();
                }
                cache.append_with_slide(0, &key, &value).ok();
                black_box(cache.seq_len())
            });
        });
    }

    // Cache slide operation
    {
        group.bench_function("slide_window", |b| {
            b.iter_batched(
                || {
                    let mut cache = StreamingKVCache::new(n_layers, d_model, 64, 16);
                    // Fill cache
                    let kv = vec![0.5_f32; d_model];
                    for _ in 0..64 {
                        cache.inner_mut().self_attn_cache[0].append(&kv, &kv).ok();
                    }
                    cache
                },
                |mut cache| {
                    cache.slide_window().ok();
                    black_box(cache.seq_len())
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    // Cache warm-up
    {
        let keys = vec![0.5_f32; d_model * 16]; // 16 tokens of context
        let values = vec![0.5_f32; d_model * 16];

        group.bench_function("warm_up", |b| {
            b.iter(|| {
                let mut cache = StreamingKVCache::low_latency(n_layers, d_model);
                cache.warm_up(0, &keys, &values).ok();
                black_box(cache.seq_len())
            });
        });
    }

    // Cache reset
    group.bench_function("reset", |b| {
        b.iter_batched(
            || {
                let mut cache = StreamingKVCache::low_latency(n_layers, d_model);
                let kv = vec![0.5_f32; d_model];
                for _ in 0..32 {
                    cache.inner_mut().self_attn_cache[0].append(&kv, &kv).ok();
                }
                cache
            },
            |mut cache| {
                cache.reset();
                black_box(cache.is_empty())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark continuous streaming simulation
fn bench_continuous_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("continuous_streaming");
    group.sample_size(50);

    // Simulate 10 seconds of continuous streaming in different modes
    for (name, config, chunk_duration) in [
        ("standard", StreamingConfig::default(), 30.0_f32),
        ("low_latency", StreamingConfig::low_latency(), 0.5),
        ("ultra_low", StreamingConfig::ultra_low_latency(), 0.25),
    ] {
        let config = StreamingConfig {
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            enable_vad: false,
            ..config
        };

        // Calculate number of chunks for 10 seconds of audio
        let num_chunks = (10.0 / chunk_duration).ceil() as usize;
        let chunk_samples = (chunk_duration * 16000.0) as usize;

        group.throughput(Throughput::Elements((num_chunks * chunk_samples) as u64));

        group.bench_with_input(
            BenchmarkId::new(name, "10s"),
            &(config.clone(), chunk_duration),
            |b, (config, chunk_dur)| {
                let chunk_audio = generate_audio(*chunk_dur, 16000);

                b.iter(|| {
                    let mut processor = StreamingProcessor::new(config.clone());
                    let mut chunk_count = 0;

                    for _ in 0..num_chunks {
                        processor.push_audio(&chunk_audio);
                        processor.process();

                        if processor.has_chunk() {
                            let _ = processor.get_chunk();
                            chunk_count += 1;
                        }
                    }

                    black_box(chunk_count)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark state transitions
fn bench_state_transitions(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_transitions");
    group.sample_size(100);

    let config = StreamingConfig {
        input_sample_rate: 16000,
        output_sample_rate: 16000,
        min_speech_duration_ms: 0, // Immediate transition
        enable_vad: false,
        ..StreamingConfig::low_latency()
    };

    // Measure state transition overhead
    group.bench_function("transition_overhead", |b| {
        b.iter_batched(
            || {
                let processor = StreamingProcessor::new(config.clone());
                processor
            },
            |mut processor| {
                // Force state transitions
                let audio = vec![0.5_f32; 8000]; // 500ms
                processor.push_audio(&audio);
                processor.process();

                let events = processor.event_count();
                black_box(events)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Event processing overhead
    group.bench_function("event_drain", |b| {
        b.iter_batched(
            || {
                let mut processor = StreamingProcessor::new(config.clone());
                let audio = vec![0.5_f32; 8000];
                processor.push_audio(&audio);
                processor.process();
                processor
            },
            |mut processor| {
                let events = processor.drain_events();
                black_box(events.len())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark latency configuration APIs
fn bench_config_apis(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_apis");

    group.bench_function("expected_latency_ms", |b| {
        let config = StreamingConfig::low_latency();
        b.iter(|| black_box(config.expected_latency_ms()));
    });

    group.bench_function("chunk_samples", |b| {
        let config = StreamingConfig::low_latency();
        b.iter(|| black_box(config.chunk_samples()));
    });

    group.bench_function("overlap_samples", |b| {
        let config = StreamingConfig::low_latency();
        b.iter(|| black_box(config.overlap_samples()));
    });

    group.bench_function("is_low_latency", |b| {
        let config = StreamingConfig::low_latency();
        b.iter(|| black_box(config.is_low_latency()));
    });

    group.bench_function("latency_mode", |b| {
        let config = StreamingConfig::low_latency();
        b.iter(|| black_box(config.latency_mode()));
    });

    group.finish();
}

/// Benchmark streaming cache statistics
fn bench_cache_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_stats");

    let config = ModelConfig::tiny();
    let d_model = config.n_text_state as usize;
    let n_layers = config.n_text_layer as usize;

    group.bench_function("get_stats", |b| {
        let mut cache = StreamingKVCache::low_latency(n_layers, d_model);
        let kv = vec![0.5_f32; d_model];
        for _ in 0..32 {
            cache.inner_mut().self_attn_cache[0].append(&kv, &kv).ok();
        }

        b.iter(|| black_box(cache.stats()));
    });

    group.bench_function("utilization", |b| {
        let stats = StreamingCacheStats {
            seq_len: 32,
            total_tokens: 100,
            slide_count: 2,
            window_size: 64,
            context_overlap: 16,
            memory_bytes: 8192,
        };

        b.iter(|| black_box(stats.utilization()));
    });

    group.bench_function("tokens_per_slide", |b| {
        let stats = StreamingCacheStats {
            seq_len: 32,
            total_tokens: 100,
            slide_count: 2,
            window_size: 64,
            context_overlap: 16,
            memory_bytes: 8192,
        };

        b.iter(|| black_box(stats.tokens_per_slide()));
    });

    group.finish();
}

criterion_group!(
    streaming_benches,
    bench_processor_init,
    bench_chunk_processing,
    bench_kv_cache_ops,
    bench_continuous_streaming,
    bench_state_transitions,
    bench_config_apis,
    bench_cache_stats,
);

criterion_main!(streaming_benches);

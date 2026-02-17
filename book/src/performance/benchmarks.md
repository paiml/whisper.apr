# Benchmarks Overview

Whisper.apr includes comprehensive benchmarks to track performance and guide optimization.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench --bench inference
cargo bench --bench wasm_simd

# Run with HTML report
cargo bench -- --save-baseline main
```

## Benchmark Groups

### Inference Benchmarks (`benches/inference.rs`)

End-to-end transcription performance:

| Benchmark | Description |
|-----------|-------------|
| `mel_spectrogram` | Audio to mel spectrogram conversion |
| `encoder` | Encoder forward pass (various sequence lengths) |
| `decoder_greedy` | Greedy decoding performance |
| `decoder_beam` | Beam search with different beam sizes |
| `transcribe_e2e` | Full pipeline end-to-end |
| `tokenizer` | BPE encode/decode |
| `attention` | Multi-head attention computation |

### SIMD Benchmarks (`benches/wasm_simd.rs`)

Scalar vs SIMD performance comparison:

| Benchmark | Description | Expected Speedup |
|-----------|-------------|------------------|
| `matmul` | Matrix multiplication | 3-4x |
| `softmax` | Softmax activation | 2-3x |
| `dot_product` | Vector dot product | 3-4x |
| `gelu` | GELU activation | 2-3x |
| `layer_norm` | Layer normalization | 2-3x |

## Performance Targets

### Real-Time Factor (RTF)

RTF = processing_time / audio_duration

| Model | Target RTF | Status |
|-------|------------|--------|
| tiny | <= 2.0x | Exceeded (0.47x) |
| base | <= 2.5x | On track |
| small | <= 4.0x | On track |

### Memory Budget

| Model | Target | Status |
|-------|--------|--------|
| tiny | <= 150MB | Exceeded (90.45MB) |
| base | <= 350MB | On track |
| small | <= 800MB | On track |

## Key Optimizations (v0.2.4)

### Tiled MatVec (3.5x Speedup)

Single-token decoding uses a tiled_matvec fast path in matmul_raw, providing 3.5x speedup for the decoder's autoregressive step. This is the most impactful optimization for real-world transcription latency.

### Moonshine SIMD Routing

Moonshine GQA and MLP layers are routed through trueno SIMD matmul and SDPA for hardware-accelerated inference on all platforms.

### SIMD Vectorization

All matrix operations dispatch through trueno for automatic SIMD acceleration (4x typical speedup).

### KV-Cache Reuse

60% reduction in decoder compute through key-value caching across autoregressive steps.

### Quantized MatMul

Int4 compute with FP32 accumulation for 4x model size reduction with minimal accuracy loss.

## Achieved Performance

**Whisper-tiny Q4K on native (all 7/7 targets met):**

| Target | Goal | Achieved | Ratio |
|--------|------|----------|-------|
| RTF | < 2.0x | 0.47x | 4.26x better |
| ms_per_token | < 50ms | 47.17ms | 1.06x better |
| decoder_latency (1.5s) | < 1500ms | 707.55ms | 2.12x better |
| memory_peak | < 150MB | 90.45MB | 1.66x better |
| simd_speedup | > 2.0x | 2.12x | 1.06x better |
| q4k_weight_reduction | > 80% | 86% | 1.08x better |
| tokens_per_sec | > 20 | 21.2 | 1.06x better |

**Average achievement ratio: 1.76x**

## Interpreting Results

Criterion provides statistical analysis:

```
mel_spectrogram/compute/30s
                        time:   [12.345 ms 12.456 ms 12.567 ms]
                        thrpt:  [2.3891 Melem/s 2.4106 Melem/s 2.4321 Melem/s]
                 change: [-2.1234% -1.5678% -1.0123%] (p = 0.00 < 0.05)
                        Performance has improved.
```

- **time**: Mean execution time with confidence interval
- **thrpt**: Throughput (elements or bytes per second)
- **change**: Comparison to baseline (if available)

## Profiling Integration

Use Renacer for detailed profiling:

```bash
# Profile with source correlation
renacer --function-time --source -- cargo bench --bench inference

# Generate flamegraph
renacer --flamegraph -- cargo bench --bench inference > flame.svg
```

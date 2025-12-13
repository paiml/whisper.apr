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
| `mel_spectrogram` | Audio → mel spectrogram conversion |
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

| Model | Target RTF | Audio | Expected Time |
|-------|------------|-------|---------------|
| tiny  | ≤2.0x      | 60s   | ≤120s         |
| base  | ≤2.5x      | 60s   | ≤150s         |
| small | ≤4.0x      | 60s   | ≤240s         |

### Memory Budget

| Model | WASM Size | Peak Memory |
|-------|-----------|-------------|
| tiny  | ~40MB     | ≤150MB      |
| base  | ~75MB     | ≤350MB      |
| small | ~250MB    | ≤800MB      |

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

## Browser Benchmarks

For WASM performance in browsers:

```bash
# Build WASM
wasm-pack build --target web --release

# Run browser benchmarks
cd browser-bench
npm install
npm run bench
```

Browser benchmark results are saved to `benchmark-results/`.

## Continuous Benchmarking

CI runs benchmarks on every PR:

1. Benchmarks run against `main` baseline
2. Results compared for regressions
3. PR blocked if >10% performance regression
4. Results archived for historical tracking

## Profiling Integration

Use Renacer for detailed profiling:

```bash
# Profile with source correlation
renacer --function-time --source -- cargo bench --bench inference

# Generate flamegraph
renacer --flamegraph -- cargo bench --bench inference > flame.svg
```

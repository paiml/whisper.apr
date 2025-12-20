# Parallel Inference

Whisper.apr supports multi-threaded inference via the `parallel` feature, providing significant speedup on multi-core systems.

## Quick Start

### CLI Usage

```bash
# Use 4 threads for inference
whisper-apr transcribe -f audio.wav --threads 4

# Auto-detect optimal thread count (default)
whisper-apr transcribe -f audio.wav
```

### Cargo Build

```bash
# Enable parallel feature
cargo build --release --features parallel

# Run with parallel inference
cargo run --release --features parallel --bin whisper-apr-cli -- \
  transcribe -f audio.wav --threads 4
```

## Architecture

### Unified Design (CLI + WASM)

The parallel implementation uses a unified abstraction that works identically on:

- **CLI (native)**: Uses [rayon](https://crates.io/crates/rayon) for work-stealing parallelism
- **WASM (browser)**: Uses [wasm-bindgen-rayon](https://crates.io/crates/wasm-bindgen-rayon) with Web Workers

```
┌─────────────────────────────────────────────────────────────────┐
│                    parallel_map() Abstraction                    │
├─────────────────────────────────────────────────────────────────┤
│  #[cfg(feature = "parallel")]                                   │
│  fn parallel_map<T, F>(range, f) -> Vec<T>                      │
│  {                                                              │
│      #[cfg(target_arch = "wasm32")]                             │
│      { range.into_par_iter().map(f).collect() } // wasm-rayon   │
│                                                                 │
│      #[cfg(not(target_arch = "wasm32"))]                        │
│      { range.into_par_iter().map(f).collect() } // rayon        │
│  }                                                              │
│                                                                 │
│  #[cfg(not(feature = "parallel"))]                              │
│  fn parallel_map<T, F>(range, f) -> Vec<T>                      │
│  { range.map(f).collect() } // Sequential fallback              │
└─────────────────────────────────────────────────────────────────┘
```

### Parallelization Points

| Component | Parallel Dimension | Expected Speedup |
|-----------|-------------------|------------------|
| Multi-Head Attention | n_heads (6 for tiny) | Up to 6x |
| Encoder Blocks | Sequential (dependencies) | 1x |
| Decoder Blocks | Sequential (dependencies) | 1x |
| Mel Spectrogram | n_mels (80) | Future work |
| FFN | SIMD, not thread-parallel | Via trueno |

The primary parallelization target is **multi-head attention**, where each head computes independently:

```rust
// Before (sequential)
for head in 0..n_heads {
    let head_out = compute_attention_head(head, &q, &k, &v, mask)?;
    head_outputs.push(head_out);
}

// After (parallel)
let head_outputs = parallel_try_map(0..n_heads, |head| {
    compute_attention_head(head, &q, &k, &v, mask)
})?;
```

## Amdahl's Law Analysis

Maximum speedup is limited by the sequential fraction of the workload:

```
Speedup = 1 / (S + P/N)

Where:
  S = Sequential fraction (~75%)
  P = Parallel fraction (~25%)
  N = Number of threads
```

### Predicted vs Measured Speedup

| Threads | Predicted | Measured | RTF (1.5s audio) |
|---------|-----------|----------|------------------|
| 1 | 1.00x | 1.00x | 6.61x |
| 2 | 1.14x | 1.11x | 5.98x |
| 4 | 1.23x | 1.22x | 5.42x |
| 8 | 1.28x | - | - |

The measured speedup closely matches the Amdahl's Law prediction, validating the analysis.

### Why Not Higher Speedup?

The parallel fraction is only ~25% because:

1. **Linear projections (Q/K/V/O matmul)** remain sequential
2. **FFN layers** use SIMD but not thread parallelism
3. **LayerNorm and residual connections** are inherently sequential
4. **I/O and tokenization** cannot be parallelized

## WASM Considerations

### SharedArrayBuffer Requirements

WASM parallel inference requires browser support for SharedArrayBuffer:

```javascript
// Check if parallel WASM is supported
if (crossOriginIsolated) {
  // Initialize thread pool
  await initThreadPool(navigator.hardwareConcurrency - 1);
}
```

### Required HTTP Headers

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### Browser Support

| Browser | Version | Support |
|---------|---------|---------|
| Chrome | 68+ | ✅ |
| Firefox | 79+ | ✅ |
| Safari | 15.2+ | ✅ |
| Edge | 79+ | ✅ |

### Building for WASM

```bash
# Parallel WASM build (requires nightly + atomics)
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
  wasm-pack build --target web --features parallel -- -Z build-std=std,panic_abort

# Sequential fallback (works everywhere)
wasm-pack build --target web --features simd
```

## Configuration API

### Rust API

```rust
use whisper_apr::parallel::{
    configure_thread_pool,
    is_parallel_available,
    thread_count,
    parallel_map,
    parallel_try_map,
};

// Check availability
if is_parallel_available() {
    // Configure thread pool (call once at startup)
    configure_thread_pool(Some(4))?;

    println!("Using {} threads", thread_count());
}

// Use parallel operations
let results = parallel_map(0..n_heads, |head| {
    compute_head(head)
});
```

### JavaScript API (WASM)

```javascript
import init, {
  initThreadPool,
  isThreadedAvailable,
  optimalThreadCount
} from 'whisper-apr';

await init();

if (isThreadedAvailable()) {
  const threads = optimalThreadCount();
  await initThreadPool(threads);
  console.log(`Using ${threads} threads`);
}
```

## Benchmarking

Run the parallel inference example:

```bash
# Sequential
cargo run --release --example parallel_inference

# Parallel
cargo run --release --example parallel_inference --features parallel
```

Compare CLI performance:

```bash
# 1 thread
time whisper-apr transcribe -f audio.wav --threads 1

# 4 threads
time whisper-apr transcribe -f audio.wav --threads 4
```

## Specification Reference

See [§11.3 Performance Parity: Parallel Inference](../../../docs/specifications/whisper-cli-parity.md#113-performance-parity-parallel-inference-specification) for the complete specification including:

- Five-Whys root cause analysis
- Amdahl's Law derivation
- Popperian falsification checklist (Section P)
- Thread safety requirements

# Reproducibility Guide

## Overview

This document describes how to reproduce whisper.apr results for scientific verification
following Popperian falsificationism methodology.

## Environment Reproducibility

### Option 1: Nix (Recommended)

```bash
# Exact reproducible environment
nix develop

# Verify environment
rustc --version  # Expected: rustc 1.75+
wasm-pack --version
```

### Option 2: Docker

```dockerfile
FROM rust:1.75

RUN rustup target add wasm32-unknown-unknown
RUN cargo install wasm-pack cargo-nextest cargo-llvm-cov

WORKDIR /app
COPY . .
RUN cargo build --release
```

### Option 3: Manual

See [CONTRIBUTING.md](CONTRIBUTING.md) for manual setup instructions.

## Measurable Performance Thresholds

All performance claims are falsifiable with specific thresholds:

| Metric | Threshold | Measurement Method |
|--------|-----------|-------------------|
| RTF (tiny) | < 2.0x | `cargo bench --bench inference` |
| RTF (base) | < 2.5x | `cargo bench --bench inference` |
| Memory (tiny) | < 150 MB | `heaptrack cargo run --release --example rtf_benchmark` |
| Memory (base) | < 350 MB | Peak RSS measurement |
| SIMD speedup | > 2.0x | `cargo bench --bench wasm_simd` |
| First token | < 500 ms | Decoder timing |
| Token throughput | > 20 tok/s | `cargo bench` |

## Test Coverage Thresholds

| Metric | Threshold | Command |
|--------|-----------|---------|
| Line coverage | ≥ 95% | `make coverage` |
| Mutation score | ≥ 85% | `cargo mutants` |
| Branch coverage | ≥ 80% | `cargo llvm-cov` |

## Ground Truth Validation

### Reference Implementations

Results are validated against three independent implementations:

1. **whisper.cpp** (C++ reference)
   ```bash
   /home/noah/.local/bin/main -m models/ggml-tiny.bin -f audio.wav
   ```

2. **HuggingFace Transformers** (Python reference)
   ```bash
   uv run scripts/hf_transcribe.py audio.wav
   ```

3. **whisper.apr** (this implementation)
   ```bash
   cargo run --release --bin whisper-apr-cli -- transcribe audio.wav
   ```

### Expected Output

For `demos/test-audio/test-speech-1.5s.wav`:
- **Ground Truth**: "The birds can use"
- **WER Tolerance**: < 5% vs reference

## Random Seed Behavior

### Greedy Decoding (Default)

Greedy decoding is **fully deterministic**:
- No random sampling
- Same input → same output (bit-identical)
- Temperature = 0

### Beam Search

Beam search is deterministic given:
- Same beam width
- Same input audio
- Same model weights

No random seeds are used in inference.

## Model Reproducibility

### Weight Conversion

```bash
# Convert from official OpenAI weights
cargo run --bin whisper-apr-cli -- convert \
    --model tiny \
    --source huggingface \
    --output models/whisper-tiny.apr

# Verify checksum
sha256sum models/whisper-tiny.apr
```

### Mel Filterbank

The mel filterbank is embedded in the .apr model file for exact reproducibility:
- Source: librosa slaney-normalized filterbank
- Shape: 80 x 201 (n_mels x n_freqs)
- Normalization: Rows sum to ~0.025 (not 1.0)

## Benchmark Reproducibility

### Hardware Specifications

Document your hardware when reporting benchmarks:

```bash
# CPU info
lscpu | grep "Model name"
cat /proc/cpuinfo | grep "model name" | head -1

# Memory
free -h

# For WASM benchmarks
# Browser: Chrome 113+ / Firefox 121+
# Record browser version and OS
```

### Statistical Rigor

- Minimum 10 iterations per benchmark
- Report: mean, std, min, max
- 95% confidence intervals when applicable
- Warm-up iterations: 3 (excluded from stats)

```bash
# Run with statistical output
cargo bench --bench inference -- --confidence-level 0.95
```

## CI/CD Verification

All claims are verified in CI:

```yaml
# .github/workflows/ci.yml
- name: Run Tests
  run: cargo test

- name: Run Benchmarks
  run: cargo bench --bench inference

- name: Check Coverage
  run: make coverage
  env:
    COVERAGE_THRESHOLD: 95
```

## Falsification Protocol

To falsify any claim:

1. **Set up reproducible environment** (Nix/Docker/manual)
2. **Run the specific test/benchmark**
3. **Compare against documented threshold**
4. **Report discrepancy with full environment details**

Example:
```bash
# Falsify RTF < 2.0x claim
nix develop
cargo bench --bench inference -- rtf
# If reported RTF > 2.0x, the claim is falsified
```

## Contact

For reproducibility issues, open a GitHub issue with:
- Full environment details (`uname -a`, `rustc --version`)
- Exact commands run
- Observed vs expected results

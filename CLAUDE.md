# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Whisper.apr is a WASM-first automatic speech recognition (ASR) engine implementing OpenAI's Whisper architecture in pure Rust. The project targets `wasm32-unknown-unknown` (no Emscripten) with native WASM SIMD 128-bit intrinsics support.

**Core Dependencies:**
- `trueno` for SIMD-accelerated tensor operations
- `thiserror` for error handling
- `wasm-bindgen` for WASM bindings (optional)

## Build Commands

```bash
# Check compilation
cargo check

# Standard build
cargo build --release

# WASM build
cargo build --target wasm32-unknown-unknown --features wasm --release

# Run tests
cargo test                          # All tests
cargo test --lib                    # Unit tests only
cargo test test_name                # Single test
cargo test audio::                  # Module tests

# Lint (strict mode - zero warnings)
cargo clippy -- -D warnings

# Format
cargo fmt
cargo fmt --check                   # Verify formatting

# Run benchmarks
cargo bench --bench inference
```

## PMAT Compliance

```bash
# Check compliance status
pmat comply check

# Run quality gates
pmat quality-gate

# TDG scoring
pmat tdg .

# Rust project score
pmat rust-project-score

# Hooks status
pmat hooks status
```

### Tiered Quality Gates

```bash
# Tier 1: On-save (<1s)
cargo check --target wasm32-unknown-unknown && cargo fmt --check && cargo clippy -- -D warnings

# Tier 2: Pre-commit (<5s)
cargo test --lib

# Tier 3: Pre-push (1-5 min)
cargo test --all && wasm-pack test --headless --chrome

# Tier 4: CI/CD (5-60 min)
cargo mutants --no-times && pmat tdg . --include-components
```

## Architecture

### Module Structure

```
src/
├── lib.rs          # Public API: ModelType, DecodingStrategy, TranscribeOptions, TranscriptionResult
├── error.rs        # WhisperError enum, WhisperResult<T> alias
├── audio/          # Audio preprocessing pipeline
│   ├── mel.rs      # Mel spectrogram computation (80-mel filterbank)
│   └── resampler.rs # Sample rate conversion to 16kHz
├── model/          # Transformer architecture
│   ├── encoder.rs  # Audio encoder (convolutional + transformer layers)
│   ├── decoder.rs  # Text decoder with cross-attention
│   └── attention.rs # Multi-head attention (SIMD-optimized via trueno)
├── tokenizer/      # BPE tokenization
│   └── vocab.rs    # Vocabulary handling (51,865 tokens)
├── inference/      # Decoding strategies
│   ├── greedy.rs   # Fast greedy decoding
│   └── beam.rs     # Beam search with configurable width
└── wasm/           # WASM bindings (feature-gated)
```

### Data Flow

```
Audio (f32[]) → Resampler (16kHz) → MelFilterbank (80 bins) → Encoder → Decoder → Tokens → Text
```

### Model Configuration

Whisper model sizes are defined in `ModelConfig`:
- **tiny**: 39M params, 384-dim, 4 layers, 6 heads
- **base**: 74M params, 512-dim, 6 layers, 8 heads
- **small**: 244M params (post-1.0)

### .apr Model Format

Binary format optimized for WASM delivery:
- Magic: `APR1`
- LZ4-compressed weights in 64KB blocks (streaming decompression)
- Quantization support: fp32, fp16, int8

## Code Style Enforcements

**From `Cargo.toml` lints:**
- `unwrap_used = "deny"` - All errors must use `Result` types
- `expect_used = "warn"` - Prefer explicit error handling
- `panic = "warn"` - Avoid panics in library code
- `missing_docs = "warn"` - Document all public items

**Quality targets (from `.pmat-metrics.toml`):**
- 95% test coverage
- 85% mutation score
- ≤10 cyclomatic complexity per function
- Zero SATD comments (no TODO/FIXME/HACK)
- A+ TDG grade (≥95.0)

## Feature Flags

```toml
default = ["std"]
std = []           # Standard library support
wasm = [...]       # WASM bindings via wasm-bindgen
simd = []          # Explicit SIMD optimization paths
tracing = [...]    # Performance tracing via renacer
```

## Key Design Decisions

1. **#![no_std] compatible core** - The `std` feature gates standard library usage
2. **Zero unwrap()** - All error paths return `WhisperResult<T>`
3. **Trueno for SIMD** - Matrix operations dispatch through trueno for automatic SIMD acceleration
4. **Streaming model loading** - LZ4 block decompression enables progressive loading without full download

## Performance Targets

| Model | Target RTF | Memory Peak |
|-------|------------|-------------|
| tiny  | ≤2.0x      | ≤150MB      |
| base  | ≤2.5x      | ≤350MB      |
| small | ≤4.0x      | ≤800MB      |

RTF = Real-Time Factor (processing time / audio duration)

## Ticket Prefix

All work items use prefix `WAPR-`:
- `WAPR-XXX`: Core features
- `WAPR-PERF-XXX`: Performance optimization
- `WAPR-QA-XXX`: Quality/testing
- `WAPR-DOC-XXX`: Documentation

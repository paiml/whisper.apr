# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Whisper.apr is a WASM-first automatic speech recognition (ASR) engine implementing OpenAI's Whisper architecture in pure Rust. The project targets `wasm32-unknown-unknown` (no Emscripten) with native WASM SIMD 128-bit intrinsics support.

**Core Dependencies:**
- `trueno` for SIMD-accelerated tensor operations
- `thiserror` for error handling
- `wasm-bindgen` for WASM bindings (optional)

## ⚠️ CRITICAL: NO PYTHON - RUST ONLY

**This project is 100% Rust. Python is NEVER used.**

- ❌ **NEVER** create Python scripts for any purpose
- ❌ **NEVER** use `uv run`, `pip`, `torch`, `transformers`, or any Python tooling
- ❌ **NEVER** suggest "just use Python for ground truth extraction"
- ✅ **ALWAYS** use Rust for all tooling, extraction, and verification
- ✅ **ALWAYS** use existing JSON ground truth files in `test_data/*.json`
- ✅ **ALWAYS** load reference data via `aprender::verify::GroundTruth`

**Ground Truth is pre-extracted in:**
- `test_data/ref_c_mel_numpy.json` - Mel spectrogram reference (mean=-0.2148, std=0.4479)
- `test_data/ref_a_audio.json` - Audio input reference
- `test_data/reference_summary.json` - All stages summary

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

# Bash script linting (REQUIRED for all .sh files)
bashrs lint scripts/               # Use bashrs, NOT shellcheck

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

## Coverage and Testing Tools

**REQUIRED:** Use ONLY the Makefile targets for coverage:
```bash
# Run coverage (uses cargo-llvm-cov + cargo-nextest)
make coverage

# Coverage summary only
make coverage-summary

# Open HTML report
make coverage-open
```

**PROHIBITED tools - DO NOT install or use:**
- `cargo-tarpaulin` - Not compatible with WASM workflow
- Any coverage tool not in the Makefile

**Note:** llvm-cov doesn't generate profraw for `wasm32-unknown-unknown` targets. Coverage is measured on native test runs only. Browser integration tests are validated separately via probar.

## Debugging and Testing - MANDATORY TOOLING

**CRITICAL: NO AD-HOC DEBUGGING ("whack-a-mole")**

When debugging issues, you MUST use the proper tooling:

### For Browser/WASM Issues - Use Probar ONLY

```bash
# Run GUI tests
cd demos && probar test -v

# Run specific test
probar test test_name

# Browser E2E testing
cargo test --package whisper-apr-demo-tests

# Pixel regression tests
probar coverage

# Serve WASM for testing (probar handles COOP/COEP headers)
probar serve
```

**PROHIBITED - DO NOT USE:**
- `python -m http.server` or any Python HTTP server
- `npx serve` or any Node.js HTTP server
- Manual browser testing without probar tests
- Adding `console.log` / `info!()` statements and rebuilding

**Why probar only?**
- Automatically sets required COOP/COEP headers for SharedArrayBuffer
- Integrates with test framework for reproducible browser tests
- Provides coverage tracking and pixel regression testing

### For Performance/Tracing - Use Renacer

The `tracing` feature integrates with renacer for instrumentation:
```bash
# Trace execution (native)
renacer -s -- cargo test test_name

# Browser tracing is automatic via tracing_wasm
# Check browser console for structured logs with spans
```

### For Quality Issues - Use PMAT

```bash
# Track work
pmat work start WAPR-XXX
pmat work continue WAPR-XXX
pmat work complete WAPR-XXX

# Quality gates
pmat quality-gate

# Root cause analysis (Toyota Way)
pmat five-whys "description of issue"
```

### Test-First Debugging Flow

1. **Write a failing probar test** that reproduces the issue
2. **Run the test** to confirm it fails
3. **Fix the code** until the test passes
4. **Verify** with full test suite: `probar test && make test-fast`

**NEVER** fix bugs without a test that would have caught them.

## Shell Script Policy

**CRITICAL: All shell scripts MUST be validated by bashrs**

Shell scripts are written in Rust using bashrs (Rash) and transpiled to shell. This ensures:
- Type safety and compile-time error checking
- Deterministic, reproducible behavior
- Formal verification capability

**Workflow:**
```bash
# Write script in Rust (scripts/foo.rs)
# Transpile to shell
bashrs build scripts/foo.rs -o scripts/foo.sh

# Verify shell matches source
bashrs verify scripts/foo.rs scripts/foo.sh

# Check for compatibility issues
bashrs check scripts/foo.rs
```

**PROHIBITED - DO NOT USE:**
- Hand-written `.sh` files without bashrs source
- Shell scripts not validated by `bashrs verify`
- Bash-specific features not supported by bashrs

**Enforcement:**
```bash
# Lint all scripts in CI
for rs in scripts/*.rs; do
  bashrs check "$rs" || exit 1
done
```

## Python Usage Policy

**CRITICAL: uv-only transient execution**

Python packages are used ONLY via `uv run` for single-shot transient execution. No persistent package installation is permitted.

**ALLOWED:**
```bash
# Single-shot script execution with inline dependencies
uv run --with openai-whisper scripts/ground_truth.py

# Transient tool execution
uvx ruff check .
```

**PROHIBITED - DO NOT USE:**
- `pip install` - No persistent package installation
- `pip3 install` - No persistent package installation
- `conda install` - No conda environments
- `python -m venv` - No virtual environments
- Any persistent Python environment

**Rationale:** Transient execution ensures reproducibility, avoids dependency conflicts, and keeps the system clean. All Python dependencies are declared inline in scripts or via `--with` flags.

## Ground Truth Validation

For ASR accuracy validation, compare against reference implementations:

```bash
# 3-column comparison: whisper.apr vs whisper.cpp vs HuggingFace
./scripts/ground_truth_compare.sh demos/test-audio/test-speech-1.5s.wav

# whisper.cpp (C++ reference - GPU accelerated)
/home/noah/.local/bin/main -m /home/noah/src/whisper.cpp/models/ggml-tiny.bin -f audio.wav

# HuggingFace (Python reference - via uv)
uv run scripts/hf_transcribe.py audio.wav
```

**Full Specification:** `docs/specifications/ground-truth-whisper-apr-cpp-hugging-face.md`
- 100-point falsification checklist
- Toyota Way framework (Genchi Genbutsu, Five Whys, Jidoka, Kaizen)
- APR format vs custom code analysis
- 10 peer-reviewed citations
- Probar/TUI/renacer integration

## Ticket Prefix

All work items use prefix `WAPR-`:
- `WAPR-XXX`: Core features
- `WAPR-PERF-XXX`: Performance optimization
- `WAPR-QA-XXX`: Quality/testing
- `WAPR-DOC-XXX`: Documentation

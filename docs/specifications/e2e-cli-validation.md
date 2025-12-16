# WAPR-CLI-001: End-to-End CLI Validation Specification

**Version**: 1.0.0
**Status**: Active
**Author**: Claude Code
**Date**: 2025-12-15
**Depends On**: WAPR-BENCH-001, WAPR-BENCH-002

---

## Executive Summary

Before investing further in WASM integration, we must prove the whisper.apr architecture works end-to-end with real model weights and real audio. This specification defines a native CLI tool that validates the complete transcription pipeline.

**Core Principle**: "Prove it works natively before crossing the WASM boundary."

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [CLI Requirements](#2-cli-requirements)
3. [Architecture](#3-architecture)
4. [Implementation Plan](#4-implementation-plan)
5. [Test Audio Assets](#5-test-audio-assets)
6. [Acceptance Criteria](#6-acceptance-criteria)
7. [Usage Examples](#7-usage-examples)

---

## 1. Motivation

### 1.1 Why CLI Before WASM?

| Concern | Native CLI | WASM |
|---------|-----------|------|
| Debugging | Full gdb/lldb support | Limited (console only) |
| Profiling | renacer, perf, flamegraph | Browser DevTools only |
| Iteration speed | `cargo run` | Build → wasm-pack → reload |
| Error messages | Full backtraces | Cryptic JS errors |
| Model loading | Direct file I/O | Fetch + ArrayBuffer |

### 1.2 What We Need to Prove

1. **Model Loading**: `.apr` format loads correctly with all weights
2. **Audio Pipeline**: WAV → PCM → Resample → Mel works
3. **Encoder**: Produces valid encoder output shapes
4. **Decoder**: Generates coherent token sequences
5. **Detokenization**: Tokens → readable text
6. **End-to-End**: Real audio → real transcription

### 1.3 Risk Mitigation

If architecture issues exist, finding them in native code is 10x easier than in WASM:
- Numeric precision issues
- Memory layout problems
- Attention mask bugs
- Tokenizer mismatches

---

## 2. CLI Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F1 | Load `.apr` model files | P0 |
| F2 | Accept WAV audio input (16kHz mono) | P0 |
| F3 | Output transcription to stdout | P0 |
| F4 | Support `--verbose` for timing info | P1 |
| F5 | Support `--json` for structured output | P1 |
| F6 | Automatic sample rate conversion | P2 |
| F7 | Streaming mode for long audio | P2 |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| N1 | RTF for tiny model | < 2.0x |
| N2 | Memory usage (tiny-int8) | < 200MB peak |
| N3 | Startup time (model load) | < 5s |
| N4 | First token latency | < 500ms |

### 2.3 CLI Interface

```
whisper-apr-cli 0.1.0
Native CLI for whisper.apr speech recognition

USAGE:
    whisper-apr-cli [OPTIONS] --model <MODEL> <AUDIO>

ARGS:
    <AUDIO>    Path to WAV audio file

OPTIONS:
    -m, --model <MODEL>      Path to .apr model file
    -l, --language <LANG>    Language code (default: auto)
    -v, --verbose            Show timing and debug info
    -j, --json               Output as JSON
    -t, --timestamps         Include word timestamps
    -h, --help               Print help information
    -V, --version            Print version information
```

---

## 3. Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      whisper-apr-cli                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   CLI Args   │───►│  Model Load  │───►│   Validate   │       │
│  │   (clap)     │    │  (.apr file) │    │   Weights    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Audio Load  │───►│   Resample   │───►│     Mel      │       │
│  │  (WAV/PCM)   │    │  (to 16kHz)  │    │ Spectrogram  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Encoder    │───►│   Decoder    │───►│  Detokenize  │       │
│  │   Forward    │    │  (greedy)    │    │   (BPE)      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                                          ┌──────────────┐       │
│                                          │   Output     │       │
│                                          │ (text/JSON)  │       │
│                                          └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Structure

```
src/bin/
└── whisper-apr-cli.rs    # Main CLI entry point

src/
├── lib.rs                # WhisperApr public API
├── audio/
│   ├── wav.rs           # WAV file parsing
│   └── resample.rs      # Sample rate conversion
├── model/
│   ├── encoder.rs       # Audio encoder
│   └── decoder.rs       # Text decoder
├── format/
│   └── apr.rs           # .apr model format
└── vocabulary/
    └── bpe.rs           # BPE tokenizer
```

### 3.3 Data Flow

```
Input: test-speech.wav (16kHz, mono, 3.0s)
       │
       ▼
┌─────────────────────────────────────────────┐
│ 1. Load WAV                                  │
│    48,000 samples (3.0s × 16kHz)            │
│    Memory: 192KB (f32)                       │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ 2. Mel Spectrogram                          │
│    80 bins × 188 frames                     │
│    Memory: 60KB (f32)                        │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ 3. Encoder                                   │
│    Input: (1, 80, 188) mel features         │
│    Output: (1, 188, 384) encoder states     │
│    Time: ~100ms (tiny model)                │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ 4. Decoder (autoregressive)                  │
│    Generate ~20 tokens                       │
│    Time: ~100ms (with KV cache)             │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ 5. Detokenize                                │
│    Tokens: [50258, 50259, ..., 50257]       │
│    Output: "Hello, how are you today?"      │
└─────────────────────────────────────────────┘

Total: ~200ms for 3s audio = 0.067x RTF
```

---

## 4. Implementation Plan

### Phase 1: Core CLI (This PR)

1. **Create CLI binary** (`src/bin/whisper-apr-cli.rs`)
   - Argument parsing with clap
   - Model loading
   - Audio loading
   - Basic transcription

2. **Add WAV parsing** (extend `src/audio/`)
   - Read WAV header
   - Parse PCM samples
   - Handle 16-bit signed int

3. **Wire up pipeline**
   - Load model → compute mel → encode → decode → detokenize

### Phase 2: Polish

1. Add `--verbose` timing output
2. Add `--json` structured output
3. Add error handling improvements
4. Add sample rate conversion

### Phase 3: Validation

1. Test with real Whisper tiny model
2. Compare output with reference implementation
3. Measure WER on test dataset

---

## 5. Test Audio Assets

### 5.1 Required Test Files

Location: `test-audio/`

| File | Duration | Content | Purpose |
|------|----------|---------|---------|
| `hello.wav` | 1.0s | "Hello" | Minimal test |
| `numbers.wav` | 3.0s | "One two three four five" | Multi-word |
| `sentence.wav` | 5.0s | Full sentence | Realistic test |
| `silence.wav` | 2.0s | Silence | Edge case |

### 5.2 Audio Requirements

- Format: WAV (RIFF)
- Sample rate: 16kHz (preferred) or auto-convert
- Channels: Mono
- Bit depth: 16-bit signed PCM
- Duration: 0.5s - 30s

### 5.3 Creating Test Audio

```bash
# Record with Sox
sox -d -r 16000 -c 1 -b 16 test-audio/hello.wav trim 0 1

# Convert existing file
ffmpeg -i input.mp3 -ar 16000 -ac 1 -acodec pcm_s16le test-audio/output.wav

# Generate test tone (for debugging)
sox -n -r 16000 -c 1 test-audio/tone.wav synth 2 sine 440
```

---

## 6. Acceptance Criteria

### 6.1 Must Pass (P0)

| # | Criterion | Verification |
|---|-----------|--------------|
| 1 | CLI compiles without warnings | `cargo build --release` |
| 2 | CLI runs without panic | `./target/release/whisper-apr-cli --help` |
| 3 | Model loads successfully | Load tiny-int8.apr |
| 4 | Audio loads successfully | Load test WAV file |
| 5 | Transcription produces text | Non-empty output |
| 6 | Transcription is meaningful | Not garbage tokens |

### 6.2 Should Pass (P1)

| # | Criterion | Verification |
|---|-----------|--------------|
| 7 | RTF < 2.0x | Timing output |
| 8 | Memory < 200MB | `/usr/bin/time -v` |
| 9 | "Hello" transcribes correctly | WER = 0% |
| 10 | Numbers transcribe correctly | WER < 20% |

### 6.3 Nice to Have (P2)

| # | Criterion | Verification |
|---|-----------|--------------|
| 11 | Sample rate conversion works | 48kHz input |
| 12 | JSON output valid | `jq .` |
| 13 | Timestamps accurate | ±0.5s |

---

## 7. Usage Examples

### 7.1 Basic Transcription

```bash
# Simple usage
./whisper-apr-cli -m models/whisper-tiny-int8.apr test-audio/hello.wav
# Output: Hello

# With verbose timing
./whisper-apr-cli -m models/whisper-tiny-int8.apr -v test-audio/hello.wav
# Output:
# [INFO] Model loaded in 1.2s
# [INFO] Audio: 1.0s, 16000 samples
# [INFO] Mel: 0.5ms
# [INFO] Encode: 45ms
# [INFO] Decode: 80ms (15 tokens)
# [INFO] RTF: 0.13x
# Hello
```

### 7.2 JSON Output

```bash
./whisper-apr-cli -m models/whisper-tiny-int8.apr -j test-audio/hello.wav
```

```json
{
  "text": "Hello",
  "language": "en",
  "duration_seconds": 1.0,
  "processing_time_ms": 125,
  "rtf": 0.125,
  "segments": [
    {
      "start": 0.0,
      "end": 0.8,
      "text": "Hello"
    }
  ]
}
```

### 7.3 Integration Testing

```bash
# Run CLI tests
cargo test --bin whisper-apr-cli

# Benchmark CLI
hyperfine './whisper-apr-cli -m models/tiny.apr test-audio/3s.wav'

# Memory profiling
/usr/bin/time -v ./whisper-apr-cli -m models/tiny.apr test-audio/3s.wav
```

---

## 8. Testing Requirements

### 8.1 Coverage Requirements

| Metric | Target | Tool |
|--------|--------|------|
| Line Coverage | ≥95% | `cargo-llvm-cov` |
| Branch Coverage | ≥90% | `cargo-llvm-cov` |
| Mutation Score | ≥80% | `cargo-mutants` |

### 8.2 Unit Tests

```rust
// Required test categories for whisper-apr-cli:

// WAV Parsing Tests
- test_parse_wav_16bit_mono
- test_parse_wav_16bit_stereo
- test_parse_wav_8bit
- test_parse_wav_24bit
- test_parse_wav_32bit_float
- test_parse_wav_invalid_header
- test_parse_wav_truncated
- test_parse_wav_no_data_chunk

// Resampling Tests
- test_resample_no_change
- test_resample_downsample_48k_to_16k
- test_resample_upsample_8k_to_16k
- test_resample_edge_cases

// CLI Argument Tests
- test_args_required_model
- test_args_required_audio
- test_args_verbose_flag
- test_args_json_flag
- test_args_language_default

// JSON Output Tests
- test_json_output_structure
- test_json_output_with_timestamps
- test_json_output_serialization
```

### 8.3 Property-Based Tests (proptest)

```rust
// Property tests for robustness:

proptest! {
    // WAV parsing never panics on arbitrary input
    #[test]
    fn fuzz_wav_parsing(data: Vec<u8>) {
        let _ = parse_wav(&data); // Must not panic
    }

    // Resampling preserves approximate duration
    #[test]
    fn prop_resample_duration(
        samples in prop::collection::vec(any::<f32>(), 100..10000),
        src_rate in 8000u32..96000,
        dst_rate in 8000u32..96000,
    ) {
        let resampled = resample(&samples, src_rate, dst_rate);
        let expected_len = (samples.len() as f64 * dst_rate as f64 / src_rate as f64) as usize;
        prop_assert!((resampled.len() as i64 - expected_len as i64).abs() <= 1);
    }

    // Resampling output is bounded
    #[test]
    fn prop_resample_bounded(samples in prop::collection::vec(-1.0f32..1.0, 100..1000)) {
        let resampled = resample(&samples, 48000, 16000);
        for &s in &resampled {
            prop_assert!(s >= -1.5 && s <= 1.5);
        }
    }
}
```

### 8.4 Renacer Tracing Integration

The CLI must support full tracing via renacer for performance analysis:

```rust
// Required trace spans:
#[instrument(level = "info", skip(data))]
fn parse_wav(data: &[u8]) -> Result<...>

#[instrument(level = "info", skip(samples))]
fn resample(samples: &[f32], src: u32, dst: u32) -> Vec<f32>

#[instrument(level = "info", skip(model_data))]
fn load_model(model_data: &[u8]) -> Result<WhisperApr>

#[instrument(level = "info", skip(whisper, samples))]
fn transcribe(whisper: &WhisperApr, samples: &[f32]) -> Result<...>
```

Enable with: `cargo run --features cli,tracing --bin whisper-apr-cli`

### 8.5 Benchmark Suite

Location: `benches/cli_benchmarks.rs`

```rust
// Required benchmarks:
criterion_group!(
    cli_benches,
    bench_wav_parsing_1s,
    bench_wav_parsing_10s,
    bench_wav_parsing_60s,
    bench_resample_48k_to_16k,
    bench_resample_8k_to_16k,
    bench_model_loading_tiny_int8,
    bench_model_loading_tiny_fp32,
    bench_transcribe_1s,
    bench_transcribe_3s,
    bench_transcribe_10s,
    bench_end_to_end_1s,
    bench_end_to_end_10s,
);
```

Run with: `cargo bench --bench cli_benchmarks --features cli`

---

## 9. Success Metrics

### 9.1 Definition of Done

- [ ] CLI binary builds and runs
- [ ] Can load real .apr model file
- [ ] Can transcribe real WAV audio
- [ ] Output is meaningful English text
- [ ] RTF < 2.0x for tiny model
- [ ] All P0 acceptance criteria pass
- [ ] **≥95% test coverage**
- [ ] **≥80% mutation score**
- [ ] **All property tests pass**
- [ ] **Benchmarks documented**
- [ ] **Renacer tracing works**

### 9.2 Test Commands

```bash
# Run all CLI tests
cargo test --features cli --bin whisper-apr-cli

# Run with coverage
cargo llvm-cov --features cli --bin whisper-apr-cli --html

# Run mutation tests
cargo mutants --features cli --bin whisper-apr-cli -- --no-times

# Run benchmarks
cargo bench --features cli --bench cli_benchmarks

# Run with tracing
RUST_LOG=info cargo run --features cli,tracing --bin whisper-apr-cli -- \
    -m models/whisper-tiny-int8.apr -v test.wav
```

### 9.3 Future Work (Post-CLI)

Once CLI validates the architecture:

1. **WASM Revival**: Port proven pipeline to wasm32
2. **Streaming**: Add chunked processing
3. **GPU**: Add WebGPU acceleration
4. **Models**: Support base/small sizes

---

## References

1. OpenAI Whisper: https://github.com/openai/whisper
2. whisper.cpp: https://github.com/ggerganov/whisper.cpp
3. Rust clap: https://docs.rs/clap
4. cargo-mutants: https://mutants.rs
5. proptest: https://docs.rs/proptest
6. renacer: https://github.com/paiml/renacer

---

*Document version 1.1.0 - Added comprehensive testing requirements*

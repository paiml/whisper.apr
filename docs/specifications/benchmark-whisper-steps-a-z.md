# WAPR-BENCH-001: Whisper Pipeline Benchmark Specification

**Version**: 1.1.0
**Status**: Draft
**Author**: Claude Code
**Date**: 2025-12-15

---

## Executive Summary

This specification defines a systematic benchmark methodology for the whisper.apr transcription pipeline. Each step from audio loading to text output is isolated, measured, and validated using on-disk assets and renacer tracing instrumentation.

**Core Principle**: "What gets measured gets managed." [1]

---

## Table of Contents

1. [Pipeline Architecture](#1-pipeline-architecture)
2. [Benchmark Assets](#2-benchmark-assets)
3. [Step-by-Step Breakdown (A-Z)](#3-step-by-step-breakdown-a-z)
4. [Renacer Instrumentation](#4-renacer-instrumentation)
5. [Performance Targets](#5-performance-targets)
6. [Implementation Plan](#6-implementation-plan)
7. [Peer-Reviewed Annotations](#7-peer-reviewed-annotations)
8. [100-Point Falsification QA](#8-100-point-falsification-qa)
9. [Appendix A: Benchmark Harness Template](#appendix-a-benchmark-harness-template)
10. [Appendix B: Expected Trace Output](#appendix-b-expected-trace-output)
11. [Appendix C: Aprender Infrastructure Reference](#appendix-c-aprender-benchmarking-infrastructure-reference-implementation)
12. [Appendix D: Probar TUI Performance Simulation](#appendix-d-probar-tui-performance-simulation)

---

## 1. Pipeline Architecture

### 1.1 End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WHISPER.APR TRANSCRIPTION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [A] Audio Source                                                           │
│       │                                                                     │
│       ▼                                                                     │
│  [B] WAV File Load ──────────────────────────────────────────► Disk I/O     │
│       │                                                                     │
│       ▼                                                                     │
│  [C] PCM Parse (i16 → f32) ──────────────────────────────────► CPU          │
│       │                                                                     │
│       ▼                                                                     │
│  [D] Resample (48kHz → 16kHz) ───────────────────────────────► CPU/SIMD     │
│       │                                                                     │
│       ▼                                                                     │
│  [E] Chunking (1.5s windows) ────────────────────────────────► Memory       │
│       │                                                                     │
│       ▼                                                                     │
│  [F] Mel Spectrogram ────────────────────────────────────────► CPU/SIMD     │
│       │    └─ FFT (400 samples)                                             │
│       │    └─ Mel filterbank (80 bins)                                      │
│       │    └─ Log compression                                               │
│       ▼                                                                     │
│  [G] Encoder Forward Pass ───────────────────────────────────► CPU/SIMD     │
│       │    └─ Conv1 + Conv2 (frontend)                                      │
│       │    └─ Positional embedding                                          │
│       │    └─ N transformer blocks                                          │
│       ▼                                                                     │
│  [H] Decoder Forward Pass (per token) ───────────────────────► CPU/SIMD     │
│       │    └─ Token embedding                                               │
│       │    └─ Positional embedding                                          │
│       │    └─ Self-attention (with KV cache)                                │
│       │    └─ Cross-attention                                               │
│       │    └─ FFN                                                           │
│       ▼                                                                     │
│  [I] Token Selection ────────────────────────────────────────► CPU          │
│       │    └─ Greedy: argmax                                                │
│       │    └─ Beam: top-k + scoring                                         │
│       ▼                                                                     │
│  [J] Token Detokenization ───────────────────────────────────► CPU          │
│       │    └─ BPE decode                                                    │
│       │    └─ Unicode normalization                                         │
│       ▼                                                                     │
│  [K] Result Assembly ────────────────────────────────────────► Memory       │
│       │    └─ Timestamp extraction                                          │
│       │    └─ Segment merging                                               │
│       ▼                                                                     │
│  [L] Output                                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Critical Path Analysis

| Step | Typical % of Total Time | Parallelizable | SIMD Accelerated |
|------|------------------------|----------------|------------------|
| B-C  | <1%                    | No             | No               |
| D    | 1-2%                   | Yes            | Yes              |
| F    | 2-5%                   | Yes            | Yes (FFT)        |
| G    | 20-30%                 | Partial        | Yes (matmul)     |
| H    | 60-70%                 | No (sequential)| Yes (matmul)     |
| I-K  | <1%                    | No             | No               |

**Bottleneck**: Decoder forward pass (H) dominates due to autoregressive nature.

---

## 2. Benchmark Assets

### 2.1 Audio Test Files

Location: `demos/test-audio/`

| File | Duration | Samples | Size | Purpose |
|------|----------|---------|------|---------|
| `test-speech-1.5s.wav` | 1.5s | 24,000 | 48KB | Single chunk test |
| `test-speech-3s.wav` | 3.0s | 48,000 | 96KB | Standard benchmark |
| `test-speech-full.wav` | 33.6s | 537,970 | 1MB | Sustained load test |
| `test-tone-2s.wav` | 2.0s | 32,000 | 64KB | Synthetic baseline |

### 2.2 Model Files

Location: `models/`

| File | Size | Quantization | Purpose |
|------|------|--------------|---------|
| `whisper-tiny.apr` | 151MB | fp32 | Reference baseline |
| `whisper-tiny-int8.apr` | 38MB | int8 | Production target |

### 2.3 Expected Outputs

Location: `demos/test-audio/expected/`

| Audio File | Expected Transcription | Tolerance |
|------------|----------------------|-----------|
| `test-speech-1.5s.wav` | (varies by sample) | WER < 20% |
| `test-speech-3s.wav` | (varies by sample) | WER < 15% |
| `test-tone-2s.wav` | "" (silence/noise) | Empty or noise tokens |

---

## 3. Step-by-Step Breakdown (A-Z)

### Step A: Audio Source Identification

**Input**: File path or audio stream
**Output**: Audio source handle
**Measurement**: Path validation time (μs)

```rust
#[tracing::instrument(level = "debug")]
fn identify_audio_source(path: &Path) -> Result<AudioSource> {
    let _span = renacer::span!("step_a_identify_source");
    // Validate path exists
    // Check file extension
    // Return source type
}
```

**Benchmark Command**:
```bash
renacer -s -- cargo bench --bench pipeline -- step_a
```

---

### Step B: WAV File Load

**Input**: File path
**Output**: Raw bytes (`Vec<u8>`)
**Measurement**: Disk read throughput (MB/s)

```rust
#[tracing::instrument(level = "debug")]
fn load_wav_bytes(path: &Path) -> Result<Vec<u8>> {
    let _span = renacer::span!("step_b_load_wav");
    std::fs::read(path)
}
```

**Expected Performance**:
| File Size | SSD Time | HDD Time |
|-----------|----------|----------|
| 48KB | <1ms | <5ms |
| 1MB | <5ms | <50ms |

---

### Step C: PCM Parse

**Input**: Raw WAV bytes
**Output**: Normalized f32 samples (`Vec<f32>`)
**Measurement**: Samples/second throughput

```rust
#[tracing::instrument(level = "debug", skip(bytes))]
fn parse_pcm(bytes: &[u8]) -> Result<Vec<f32>> {
    let _span = renacer::span!("step_c_parse_pcm");

    // Skip 44-byte WAV header
    let pcm_data = &bytes[44..];

    // Convert i16 PCM to f32 normalized
    let samples: Vec<f32> = pcm_data
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    Ok(samples)
}
```

**Expected Performance**:
- Target: >100M samples/second
- 48,000 samples: <0.5ms

---

### Step D: Resample

**Input**: f32 samples at source rate (e.g., 48kHz)
**Output**: f32 samples at 16kHz
**Measurement**: Resample ratio, latency

```rust
#[tracing::instrument(level = "debug", skip(samples))]
fn resample_to_16k(samples: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    let _span = renacer::span!("step_d_resample");

    if source_rate == 16000 {
        return Ok(samples.to_vec());
    }

    let resampler = SincResampler::new(source_rate, 16000)?;
    resampler.resample(samples)
}
```

**Expected Performance**:
| Source Rate | Ratio | Time (1s audio) |
|-------------|-------|-----------------|
| 16kHz | 1:1 (passthrough) | <0.1ms |
| 44.1kHz | 2.76:1 | ~5ms |
| 48kHz | 3:1 | ~5ms |

---

### Step E: Chunking

**Input**: Full audio samples
**Output**: Overlapping 1.5s chunks
**Measurement**: Memory allocation overhead

```rust
#[tracing::instrument(level = "debug", skip(samples))]
fn chunk_audio(samples: &[f32], chunk_duration: f32) -> Vec<AudioChunk> {
    let _span = renacer::span!("step_e_chunk");

    let chunk_samples = (16000.0 * chunk_duration) as usize;
    let hop_samples = chunk_samples / 2; // 50% overlap

    samples
        .chunks(hop_samples)
        .enumerate()
        .map(|(i, _)| {
            let start = i * hop_samples;
            let end = (start + chunk_samples).min(samples.len());
            AudioChunk {
                samples: samples[start..end].to_vec(),
                start_time: start as f32 / 16000.0,
            }
        })
        .collect()
}
```

---

### Step F: Mel Spectrogram

**Input**: Audio chunk (f32 samples)
**Output**: Mel features (80 bins × T frames)
**Measurement**: FFT time, filterbank time, total

```rust
#[tracing::instrument(level = "debug", skip(samples))]
fn compute_mel(samples: &[f32]) -> Result<Vec<f32>> {
    let _span = renacer::span!("step_f_mel");

    // Sub-spans for detailed profiling
    let fft_span = renacer::span!("step_f1_fft");
    let stft = compute_stft(samples, N_FFT, HOP_LENGTH);
    drop(fft_span);

    let filterbank_span = renacer::span!("step_f2_filterbank");
    let mel = apply_mel_filterbank(&stft, N_MELS);
    drop(filterbank_span);

    let log_span = renacer::span!("step_f3_log");
    let log_mel = mel.iter().map(|x| (x.max(1e-10)).ln()).collect();
    drop(log_span);

    Ok(log_mel)
}
```

**Expected Performance** (1.5s audio = 94 frames):
| Sub-step | Time (SIMD) | Time (scalar) |
|----------|-------------|---------------|
| STFT | ~3ms | ~15ms |
| Filterbank | ~1ms | ~5ms |
| Log | <0.5ms | <1ms |
| **Total** | **~5ms** | **~20ms** |

---

### Step G: Encoder Forward Pass

**Input**: Mel features
**Output**: Encoded audio features
**Measurement**: Conv time, attention time, total

```rust
#[tracing::instrument(level = "debug", skip(mel))]
fn encode(mel: &[f32]) -> Result<Vec<f32>> {
    let _span = renacer::span!("step_g_encode");

    // Conv frontend
    let conv_span = renacer::span!("step_g1_conv");
    let x = conv1(mel);
    let x = gelu(&x);
    let x = conv2(&x);
    let x = gelu(&x);
    drop(conv_span);

    // Add positional embedding
    let pos_span = renacer::span!("step_g2_pos_embed");
    let x = add_positional_embedding(&x);
    drop(pos_span);

    // Transformer blocks
    for i in 0..N_ENCODER_LAYERS {
        let block_span = renacer::span!("step_g3_block", layer = i);
        x = encoder_block(&x, i);
        drop(block_span);
    }

    // Layer norm
    let ln_span = renacer::span!("step_g4_ln");
    let x = layer_norm(&x);
    drop(ln_span);

    Ok(x)
}
```

**Expected Performance** (whisper-tiny, 1.5s audio):
| Sub-step | Time |
|----------|------|
| Conv frontend | ~50ms |
| Positional embed | <1ms |
| 4 transformer blocks | ~300ms |
| Layer norm | <1ms |
| **Total** | **~350ms** |

---

### Step H: Decoder Forward Pass (Per Token)

**Input**: Encoder output, previous tokens, KV cache
**Output**: Logits for next token
**Measurement**: Per-token latency, cache efficiency

```rust
#[tracing::instrument(level = "debug", skip(encoder_out, cache))]
fn decode_one_token(
    token: u32,
    encoder_out: &[f32],
    cache: &mut DecoderKVCache,
) -> Result<Vec<f32>> {
    let _span = renacer::span!("step_h_decode_token", token = token);

    // Token embedding
    let embed_span = renacer::span!("step_h1_embed");
    let x = token_embed(token);
    let x = add_positional_embedding(&x, cache.seq_len());
    drop(embed_span);

    // Decoder blocks
    for i in 0..N_DECODER_LAYERS {
        let block_span = renacer::span!("step_h2_block", layer = i);

        // Self-attention with KV cache
        let self_attn_span = renacer::span!("step_h2a_self_attn");
        x = self_attention(&x, cache, i);
        drop(self_attn_span);

        // Cross-attention
        let cross_attn_span = renacer::span!("step_h2b_cross_attn");
        x = cross_attention(&x, encoder_out, i);
        drop(cross_attn_span);

        // FFN
        let ffn_span = renacer::span!("step_h2c_ffn");
        x = ffn(&x, i);
        drop(ffn_span);

        drop(block_span);
    }

    // Final layer norm + projection to logits
    let proj_span = renacer::span!("step_h3_proj");
    let logits = project_to_vocab(&x);
    drop(proj_span);

    Ok(logits)
}
```

**Expected Performance** (whisper-tiny, per token):
| Sub-step | Time (with KV cache) | Time (no cache) |
|----------|---------------------|-----------------|
| Embedding | <1ms | <1ms |
| Self-attention | ~20ms | ~50ms |
| Cross-attention | ~30ms | ~30ms |
| FFN | ~20ms | ~20ms |
| Projection | ~5ms | ~5ms |
| **Total** | **~80ms** | **~110ms** |

**Critical**: For 20 tokens, total decode = 20 × 80ms = **1.6s**

---

### Step I: Token Selection

**Input**: Logits (vocab_size)
**Output**: Selected token ID
**Measurement**: Selection time, strategy overhead

```rust
#[tracing::instrument(level = "debug", skip(logits))]
fn select_token(logits: &[f32], strategy: DecodingStrategy) -> u32 {
    let _span = renacer::span!("step_i_select");

    match strategy {
        DecodingStrategy::Greedy => {
            let _greedy_span = renacer::span!("step_i1_greedy");
            argmax(logits)
        }
        DecodingStrategy::BeamSearch { beam_size, .. } => {
            let _beam_span = renacer::span!("step_i2_beam");
            beam_select(logits, beam_size)
        }
    }
}
```

**Expected Performance**:
| Strategy | Time |
|----------|------|
| Greedy (argmax) | <0.1ms |
| Beam (k=5) | ~1ms |

---

### Step J: Detokenization

**Input**: Token IDs
**Output**: UTF-8 text
**Measurement**: BPE decode time

```rust
#[tracing::instrument(level = "debug", skip(tokens))]
fn detokenize(tokens: &[u32]) -> Result<String> {
    let _span = renacer::span!("step_j_detokenize");

    let pieces: Vec<&str> = tokens
        .iter()
        .filter_map(|&t| vocab.get(t))
        .collect();

    let text = pieces.join("");
    let normalized = unicode_normalize(&text);

    Ok(normalized)
}
```

**Expected Performance**: <1ms for typical transcription

---

### Step K: Result Assembly

**Input**: Text, token timings
**Output**: `TranscriptionResult`
**Measurement**: Timestamp extraction time

```rust
#[tracing::instrument(level = "debug")]
fn assemble_result(text: String, tokens: &[u32]) -> TranscriptionResult {
    let _span = renacer::span!("step_k_assemble");

    let segments = extract_timestamp_segments(tokens);

    TranscriptionResult {
        text,
        language: "en".to_string(),
        segments,
    }
}
```

---

### Step L: Output

**Input**: `TranscriptionResult`
**Output**: Serialized response (JSON/text)
**Measurement**: Serialization time

---

## 4. Renacer Instrumentation

### 4.1 Trace Configuration

```toml
# renacer.toml
[tracing]
level = "debug"
output = "chrome"  # Chrome trace format for visualization
file = "whisper-benchmark.trace.json"

[spans]
# Enable all whisper pipeline spans
include = ["step_*"]

[metrics]
# Collect timing histograms
histograms = true
percentiles = [50, 90, 95, 99]
```

### 4.2 Running Benchmarks with Tracing

```bash
# Full pipeline trace
renacer -s -- cargo bench --bench pipeline

# Specific step trace
renacer -s --filter "step_h*" -- cargo bench --bench decode

# Export Chrome trace
renacer export chrome whisper-benchmark.trace.json
```

### 4.3 Visualization

Open `whisper-benchmark.trace.json` in:
- Chrome: `chrome://tracing`
- Perfetto: https://ui.perfetto.dev

---

## 5. Performance Targets

### 5.1 Latency Targets (whisper-tiny-int8, 1.5s audio)

| Step | Target | Stretch Goal |
|------|--------|--------------|
| B: Load | <5ms | <1ms |
| C: Parse | <1ms | <0.5ms |
| D: Resample | <10ms | <5ms |
| F: Mel | <10ms | <5ms |
| G: Encode | <500ms | <300ms |
| H: Decode (total) | <2s | <1s |
| **End-to-End** | **<3s** | **<1.5s** |

### 5.2 RTF Targets

| Audio Duration | Max Processing Time | RTF Target |
|---------------|---------------------|------------|
| 1.5s | 3.0s | 2.0x |
| 3.0s | 4.5s | 1.5x |
| 30s | 30s | 1.0x |

### 5.3 Memory Targets

| Model | Peak Memory | Sustained |
|-------|-------------|-----------|
| tiny-int8 | <150MB | <100MB |
| tiny-fp32 | <300MB | <200MB |

---

## 6. Implementation Plan

### Phase 1: Instrumentation (Week 1)
1. Add renacer spans to all pipeline functions
2. Create benchmark harness with test audio files
3. Establish baseline measurements

### Phase 2: Profiling (Week 2)
1. Run full pipeline traces
2. Identify hotspots (expected: decoder)
3. Document bottleneck analysis

### Phase 3: Optimization (Weeks 3-4)
1. SIMD optimization for identified hotspots
2. Memory layout improvements
3. KV cache efficiency

### Phase 4: Validation (Week 5)
1. Run 100-point QA falsification
2. Regression testing
3. Documentation

---

## 7. Peer-Reviewed Annotations

### [1] Deming, W. Edwards (1986)

> "In God we trust; all others must bring data."

*Out of the Crisis*, MIT Press. ISBN 978-0-911379-01-0.

**Application**: Every performance claim in this benchmark must be backed by measured data, not assumptions. The renacer instrumentation provides this data foundation.

---

### [2] Radford, A., et al. (2022)

> "Whisper's encoder processes audio in 30-second chunks using a log-Mel spectrogram with 80 channels... The decoder uses learned position embeddings and tied input-output token embeddings."

*Robust Speech Recognition via Large-Scale Weak Supervision*, OpenAI. arXiv:2212.04356.

**Application**: Our benchmark validates conformance to the original Whisper architecture. The mel spectrogram (Step F) must produce 80-channel output, and decoder timing (Step H) is expected to dominate due to autoregressive generation.

---

### [3] Hennessy, J. & Patterson, D. (2017)

> "Amdahl's Law: The performance improvement gained from using some faster mode of execution is limited by the fraction of time the faster mode can be used."

*Computer Architecture: A Quantitative Approach*, 6th Edition, Morgan Kaufmann. ISBN 978-0-12-811905-1.

**Application**: Since decoder (Step H) consumes 60-70% of runtime, optimizing other steps yields diminishing returns. Focus optimization effort on decoder attention mechanisms.

---

### [4] Gregg, B. (2020)

> "The USE Method: For every resource, check Utilization, Saturation, and Errors."

*Systems Performance: Enterprise and the Cloud*, 2nd Edition, Pearson. ISBN 978-0-13-682015-4.

**Application**: Each benchmark step should report:
- **Utilization**: CPU/SIMD lane usage
- **Saturation**: Queue depth, cache pressure
- **Errors**: Failed allocations, numeric overflow

---

### [5] Mytkowicz, T., et al. (2009)

> "Measurement bias is prevalent in computer systems... seemingly innocuous aspects of experimental setup can have significant impact on performance."

*Producing Wrong Data Without Doing Anything Obviously Wrong*, ASPLOS '09. DOI: 10.1145/1508244.1508275.

**Application**: Benchmark methodology must control for:
- Warmup runs (JIT, cache priming)
- System load variance
- Memory allocator state
- File system caching

Our benchmark runs 5 warmup iterations before 20 measured iterations, with statistical outlier removal.

---

## 8. 100-Point Falsification QA

### Methodology

Following Popperian falsificationism [Popper, 1959], we attempt to **disprove** that our benchmarks are accurate rather than confirm they are correct.

### Test Categories

| Category | Points | Focus |
|----------|--------|-------|
| Asset Validity | 10 | Test files are correct |
| Timing Accuracy | 20 | Measurements are precise |
| Reproducibility | 20 | Results are consistent |
| Coverage | 20 | All steps measured |
| Correctness | 20 | Output matches expected |
| Edge Cases | 10 | Boundary conditions |

---

### Section 1: Asset Validity (10 points)

| # | Falsification Attempt | Command | Expected | Points |
|---|----------------------|---------|----------|--------|
| 1 | Test audio file exists | `test -f demos/test-audio/test-speech-3s.wav` | Exit 0 | 1 |
| 2 | Audio is valid WAV | `file demos/test-audio/test-speech-3s.wav \| grep "WAVE audio"` | Match | 1 |
| 3 | Audio is 16kHz | `ffprobe -show_entries stream=sample_rate ... \| grep 16000` | Match | 1 |
| 4 | Audio is mono | `ffprobe -show_entries stream=channels ... \| grep 1` | Match | 1 |
| 5 | Audio duration correct | `ffprobe -show_entries format=duration ... \| grep "3.0"` | Match | 1 |
| 6 | Model file exists | `test -f models/whisper-tiny-int8.apr` | Exit 0 | 1 |
| 7 | Model loads without error | `cargo test --release test_model_load` | Pass | 1 |
| 8 | Model size reasonable | `stat --printf="%s" models/whisper-tiny-int8.apr \| awk '$1 > 30000000'` | True | 1 |
| 9 | Expected output exists | `test -f demos/test-audio/expected/test-speech-3s.txt` | Exit 0 | 1 |
| 10 | Test harness compiles | `cargo build --release --bench pipeline` | Exit 0 | 1 |

---

### Section 2: Timing Accuracy (20 points)

| # | Falsification Attempt | Command | Expected | Points |
|---|----------------------|---------|----------|--------|
| 11 | Timer resolution < 1μs | `cargo test test_timer_resolution` | < 1000ns | 2 |
| 12 | Warmup runs excluded | Check benchmark output excludes first 5 | Excluded | 2 |
| 13 | Multiple iterations | Benchmark runs >= 20 iterations | >= 20 | 2 |
| 14 | Std dev reported | Output includes standard deviation | Present | 2 |
| 15 | Outliers marked | Statistical outlier detection enabled | Enabled | 2 |
| 16 | No timer overflow | 30s audio benchmark completes | No overflow | 2 |
| 17 | Renacer spans recorded | Trace file contains step_* spans | Present | 2 |
| 18 | Span nesting correct | Parent-child relationships valid | Valid | 2 |
| 19 | Timestamps monotonic | All span start < end | True | 2 |
| 20 | Total = sum of parts | End-to-end ≈ Σ(steps) ± 5% | Within 5% | 2 |

---

### Section 3: Reproducibility (20 points)

| # | Falsification Attempt | Command | Expected | Points |
|---|----------------------|---------|----------|--------|
| 21 | Same input → same output | Run 3x, compare transcriptions | Identical | 2 |
| 22 | Timing variance < 20% | CoV of 20 runs < 0.2 | CoV < 0.2 | 2 |
| 23 | Cross-run consistency | Run on different days | < 10% diff | 2 |
| 24 | Memory stable | No memory growth over runs | Stable | 2 |
| 25 | No order dependence | Shuffle test order, same results | Same | 2 |
| 26 | Cache warmup works | Run 2 vs run 6 timing similar | < 5% diff | 2 |
| 27 | Parallel runs isolated | Run 2 benchmarks simultaneously | No interference | 2 |
| 28 | File system cached | Second read faster | Faster | 2 |
| 29 | Deterministic decode | Same tokens every run | Identical | 2 |
| 30 | Seed reproducibility | Same RNG seed → same beam results | Identical | 2 |

---

### Section 4: Coverage (20 points)

| # | Falsification Attempt | Command | Expected | Points |
|---|----------------------|---------|----------|--------|
| 31 | Step A traced | `grep "step_a" trace.json` | Present | 2 |
| 32 | Step B traced | `grep "step_b" trace.json` | Present | 2 |
| 33 | Step C traced | `grep "step_c" trace.json` | Present | 2 |
| 34 | Step D traced | `grep "step_d" trace.json` | Present | 2 |
| 35 | Step F traced | `grep "step_f" trace.json` | Present | 2 |
| 36 | Step G traced | `grep "step_g" trace.json` | Present | 2 |
| 37 | Step H traced | `grep "step_h" trace.json` | Present | 2 |
| 38 | Sub-steps traced | step_f1, step_f2, etc. present | Present | 2 |
| 39 | Per-token decode | Each token has span | All tokens | 2 |
| 40 | Full pipeline span | Root span covers all | Covers all | 2 |

---

### Section 5: Correctness (20 points)

| # | Falsification Attempt | Command | Expected | Points |
|---|----------------------|---------|----------|--------|
| 41 | Transcription non-empty | Output text length > 0 | > 0 | 2 |
| 42 | Language detected | Language field = "en" | "en" | 2 |
| 43 | WER acceptable | WER vs expected < 20% | < 20% | 2 |
| 44 | No garbage tokens | No repeated token runs > 5 | No repeats | 2 |
| 45 | Proper EOT | Token sequence ends with EOT | Ends EOT | 2 |
| 46 | Valid UTF-8 | Output is valid UTF-8 | Valid | 2 |
| 47 | Mel shape correct | (80, T) where T = samples/160 | Correct | 2 |
| 48 | Encoder output shape | (1500, d_model) | Correct | 2 |
| 49 | Logits shape | (vocab_size,) per step | Correct | 2 |
| 50 | RTF reasonable | RTF < 5.0x for tiny model | < 5.0x | 2 |

---

### Section 6: Edge Cases (10 points)

| # | Falsification Attempt | Command | Expected | Points |
|---|----------------------|---------|----------|--------|
| 51 | Empty audio | Transcribe 0-length | Empty or error | 1 |
| 52 | Very short (0.1s) | Transcribe 1600 samples | Handles | 1 |
| 53 | Very long (5min) | Transcribe 300s audio | Completes | 1 |
| 54 | Silence only | Transcribe test-tone.wav (no speech) | Minimal output | 1 |
| 55 | Max tokens | Decode hits max_tokens limit | Truncates | 1 |
| 56 | Wrong sample rate | Feed 48kHz without resample | Error or auto-detect | 1 |
| 57 | Stereo input | Feed stereo WAV | Error or auto-convert | 1 |
| 58 | Corrupt WAV | Truncated WAV file | Graceful error | 1 |
| 59 | Concurrent transcribe | 2 transcriptions parallel | Both complete | 1 |
| 60 | Memory pressure | Transcribe with low memory | Graceful handling | 1 |

---

### Automated Test Runner

```bash
#!/bin/bash
# run-benchmark-qa.sh

set -euo pipefail

PASS=0
FAIL=0
TOTAL=60

log_result() {
    local num=$1
    local desc=$2
    local result=$3

    if [ "$result" = "PASS" ]; then
        echo "[PASS] #$num: $desc"
        ((PASS++))
    else
        echo "[FAIL] #$num: $desc"
        ((FAIL++))
    fi
}

# Section 1: Asset Validity
echo "=== Section 1: Asset Validity ==="

# Test 1: Audio file exists
if test -f demos/test-audio/test-speech-3s.wav; then
    log_result 1 "Test audio file exists" "PASS"
else
    log_result 1 "Test audio file exists" "FAIL"
fi

# ... (remaining tests)

echo ""
echo "================================"
echo "RESULTS: $PASS / $TOTAL passed"
echo "================================"

if [ $PASS -ge 57 ]; then
    echo "VERDICT: Benchmark methodology VALIDATED (95%+)"
    exit 0
elif [ $PASS -ge 48 ]; then
    echo "VERDICT: Minor issues (80%+)"
    exit 1
else
    echo "VERDICT: Significant defects (<80%)"
    exit 2
fi
```

---

## Appendix A: Benchmark Harness Template

```rust
// benches/pipeline.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

fn benchmark_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("whisper_pipeline");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);

    // Load assets once
    let model = load_model("models/whisper-tiny-int8.apr").unwrap();
    let audio_1_5s = load_audio("demos/test-audio/test-speech-1.5s.wav").unwrap();
    let audio_3s = load_audio("demos/test-audio/test-speech-3s.wav").unwrap();

    // Step F: Mel spectrogram
    group.bench_with_input(
        BenchmarkId::new("step_f_mel", "1.5s"),
        &audio_1_5s,
        |b, audio| b.iter(|| model.compute_mel(audio)),
    );

    // Step G: Encode
    let mel_1_5s = model.compute_mel(&audio_1_5s).unwrap();
    group.bench_with_input(
        BenchmarkId::new("step_g_encode", "1.5s"),
        &mel_1_5s,
        |b, mel| b.iter(|| model.encode(mel)),
    );

    // Step H: Full decode
    group.bench_with_input(
        BenchmarkId::new("step_h_decode", "1.5s"),
        &audio_1_5s,
        |b, audio| b.iter(|| model.transcribe(audio, Default::default())),
    );

    // End-to-end
    group.bench_with_input(
        BenchmarkId::new("e2e", "3s"),
        &audio_3s,
        |b, audio| b.iter(|| model.transcribe(audio, Default::default())),
    );

    group.finish();
}

criterion_group!(benches, benchmark_pipeline);
criterion_main!(benches);
```

---

## Appendix B: Expected Trace Output

```json
{
  "traceEvents": [
    {"name": "transcribe", "ph": "B", "ts": 0, "pid": 1, "tid": 1},
    {"name": "step_f_mel", "ph": "B", "ts": 100, "pid": 1, "tid": 1},
    {"name": "step_f1_fft", "ph": "B", "ts": 110, "pid": 1, "tid": 1},
    {"name": "step_f1_fft", "ph": "E", "ts": 3100, "pid": 1, "tid": 1},
    {"name": "step_f2_filterbank", "ph": "B", "ts": 3110, "pid": 1, "tid": 1},
    {"name": "step_f2_filterbank", "ph": "E", "ts": 4100, "pid": 1, "tid": 1},
    {"name": "step_f_mel", "ph": "E", "ts": 4500, "pid": 1, "tid": 1},
    {"name": "step_g_encode", "ph": "B", "ts": 4600, "pid": 1, "tid": 1},
    {"name": "step_g_encode", "ph": "E", "ts": 354600, "pid": 1, "tid": 1},
    {"name": "step_h_decode_token", "ph": "B", "ts": 354700, "args": {"token": 50258}},
    {"name": "step_h_decode_token", "ph": "E", "ts": 434700},
    // ... more tokens
    {"name": "transcribe", "ph": "E", "ts": 2500000, "pid": 1, "tid": 1}
  ]
}
```

---

## Appendix C: Aprender Benchmarking Infrastructure (Reference Implementation)

The sibling project `aprender` (../aprender) has production-grade benchmarking infrastructure that whisper.apr should adopt.

### C.1 Infrastructure Comparison

| Feature | aprender | whisper.apr |
|---------|----------|-------------|
| **renacer.toml** | ✅ 5 performance assertions | ❌ Spec only |
| **Golden traces** | ✅ 7 files + automation script | ❌ Not implemented |
| **CI benchmark workflow** | ✅ Automated with PR comments | ⏳ Partial |
| **Makefile targets** | `bench`, `profile`, `chaos-test` | `bench` only |
| **Step-by-step pipeline** | ⏳ Indirect (via examples) | ✅ A-Z steps defined |
| **RTF targets** | ❌ | ✅ 2.0x for tiny |
| **Quantization benchmarks** | ❌ | ✅ fp32 vs int8 |

### C.2 Aprender's renacer.toml (Port This)

Location: `../aprender/renacer.toml`

```toml
[[assertion]]
name = "critical_path_latency"
max_duration_ms = 1000

[[assertion]]
name = "syscall_budget"
max_syscalls = 3000

[[assertion]]
name = "memory_allocation"
max_bytes = 1073741824  # 1GB

[[assertion]]
name = "god_process_detection"
pattern = "GodProcess"
action = "warn"

[[assertion]]
name = "pcie_bottleneck"
pattern = "PCIeBottleneck"
action = "warn"
```

**Whisper.apr equivalent** (to create):

```toml
# renacer.toml for whisper.apr

[[assertion]]
name = "encoder_latency"
max_duration_ms = 500
pattern = "step_g_encode"

[[assertion]]
name = "decoder_per_token"
max_duration_ms = 100
pattern = "step_h_decode_token"

[[assertion]]
name = "mel_compute"
max_duration_ms = 20
pattern = "step_f_mel"

[[assertion]]
name = "first_token_latency"
max_duration_ms = 800
pattern = "first_token"

[[assertion]]
name = "memory_budget"
max_bytes = 157286400  # 150MB for tiny-int8
```

### C.3 Golden Traces Infrastructure

Aprender maintains golden traces in `../aprender/golden_traces/`:

| Trace File | Duration | Syscalls | Purpose |
|------------|----------|----------|---------|
| `iris_clustering.json` | 0.842ms | 97 | K-means baseline |
| `dataframe_basics.json` | 0.855ms | 96 | DataFrame ops |
| `graph_algorithms_comprehensive.json` | 1.734ms | 191 | 11 algorithms |

**Whisper.apr golden traces** (to create in `golden_traces/`):

| Trace File | Expected Duration | Purpose |
|------------|-------------------|---------|
| `mel_compute_1.5s.json` | <10ms | Mel spectrogram baseline |
| `encoder_tiny.json` | <350ms | Encoder forward pass |
| `decoder_20_tokens.json` | <1.6s | 20-token decode sequence |
| `e2e_3s_audio.json` | <3s | Full pipeline |

### C.4 Automation Script (Port This)

Aprender's `scripts/capture_golden_traces.sh`:

```bash
#!/bin/bash
# Capture golden traces for regression testing

set -euo pipefail

TRACES_DIR="golden_traces"
mkdir -p "$TRACES_DIR"

# Capture with renacer
renacer --format json -- cargo run --release --example benchmark_pipeline \
    > "$TRACES_DIR/pipeline_baseline.json" 2>&1

# Generate summary
renacer analyze "$TRACES_DIR/pipeline_baseline.json" \
    --output "$TRACES_DIR/ANALYSIS.md"

echo "Golden traces captured to $TRACES_DIR/"
```

**Whisper.apr version** (to create as `scripts/capture_golden_traces.sh`):

```bash
#!/bin/bash
# Capture golden traces for whisper.apr pipeline

set -euo pipefail

TRACES_DIR="golden_traces"
MODELS_DIR="models"

mkdir -p "$TRACES_DIR"

echo "=== Capturing Mel Spectrogram Baseline ==="
renacer --format json -- cargo run --release --example mel_spectrogram \
    > "$TRACES_DIR/mel_compute_1.5s.json" 2>&1

echo "=== Capturing Encoder Baseline ==="
renacer --format json -- cargo bench --bench inference -- encoder \
    > "$TRACES_DIR/encoder_tiny.json" 2>&1

echo "=== Capturing Full Pipeline ==="
renacer --format json -- cargo run --release --example format_comparison \
    > "$TRACES_DIR/e2e_baseline.json" 2>&1

# Generate analysis
echo "=== Generating Analysis ==="
cat > "$TRACES_DIR/ANALYSIS.md" << 'EOF'
# Golden Trace Analysis

## Baselines (whisper-tiny-int8)

| Component | Target | Measured | Status |
|-----------|--------|----------|--------|
| Mel (1.5s) | <10ms | TBD | ⏳ |
| Encoder | <350ms | TBD | ⏳ |
| Decoder/token | <80ms | TBD | ⏳ |
| E2E (3s audio) | <3s | TBD | ⏳ |

## Regression Thresholds

- Any component >20% slower than baseline: FAIL
- Total pipeline >10% slower: WARN
- Memory >150MB peak: FAIL
EOF

echo "Golden traces captured to $TRACES_DIR/"
```

### C.5 Makefile Targets (Port This)

Aprender's Makefile benchmark targets:

```makefile
.PHONY: bench profile chaos-test

bench:
	cargo bench --all-features

profile:
	renacer -s -- cargo run --release --example benchmark_pipeline

chaos-test:
	renacer --chaos-mode -- cargo test --release
```

**Whisper.apr additions** (add to Makefile):

```makefile
.PHONY: bench-pipeline profile golden-traces

bench-pipeline:
	cargo bench --bench pipeline
	cargo bench --bench format_comparison

profile:
	renacer -s -- cargo run --release --example format_comparison

golden-traces:
	./scripts/capture_golden_traces.sh

bench-regression:
	renacer diff golden_traces/e2e_baseline.json --threshold 20
```

### C.6 CI/CD Integration

Aprender's `.github/workflows/benchmark.yml` features:

- Manual trigger (`workflow_dispatch`)
- Automatic on PR (src/**/*.rs, benches/**/*.rs changes)
- Weekly scheduled runs (Sundays 2 AM UTC)
- Results stored as artifacts (90 days retention)
- Automatic PR comments with benchmark summaries

**Action item**: Port this workflow to whisper.apr for automated regression detection.

### C.7 Implementation Checklist

- [x] Create `renacer.toml` with pipeline assertions
- [x] Create `golden_traces/` directory
- [x] Add `scripts/capture_golden_traces.sh`
- [x] Capture initial baselines for tiny-int8
- [x] Add Makefile targets: `profile`, `golden-traces`, `bench-regression`
- [x] Port `.github/workflows/benchmark.yml`
- [x] Add renacer spans to pipeline functions (Steps F, G, H)
- [x] Document baselines in `golden_traces/ANALYSIS.md`

---

## References

1. Deming, W. E. (1986). *Out of the Crisis*. MIT Press.
2. Radford, A., et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. OpenAI. arXiv:2212.04356.
3. Hennessy, J. & Patterson, D. (2017). *Computer Architecture: A Quantitative Approach*, 6th Ed.
4. Gregg, B. (2020). *Systems Performance: Enterprise and the Cloud*, 2nd Ed.
5. Mytkowicz, T., et al. (2009). *Producing Wrong Data Without Doing Anything Obviously Wrong*. ASPLOS '09.
6. Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.

---

*Document version 1.2.0 - Added Appendix D: Probar TUI Performance Simulation*

---

## Appendix D: Probar TUI Performance Simulation

This appendix defines a complete TUI-based performance simulation using probar's ratatui integration. The simulation provides real-time, step-by-step visualization of the Whisper pipeline with live performance metrics.

### D.1 TUI Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    WHISPER.APR BENCHMARK TUI                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ ┌──────────────────────────────────┐ │
│ │         PIPELINE PROGRESS           │ │       LIVE METRICS               │ │
│ │  ════════════════════════════════   │ │                                  │ │
│ │  [A] Source     ████████████ 100%   │ │  RTF:      ████░░░░ 1.2x        │ │
│ │  [B] Load       ████████████ 100%   │ │  Memory:   ██████░░ 78MB/150MB  │ │
│ │  [C] Parse      ████████████ 100%   │ │  CPU:      ████████ 95%         │ │
│ │  [D] Resample   ████████████ 100%   │ │  Tokens:   12/~30               │ │
│ │  [F] Mel        ████████████ 100%   │ │                                  │ │
│ │  [G] Encode     ████████░░░░  67%   │ │  ─────────────────────────────  │ │
│ │  [H] Decode     ░░░░░░░░░░░░   0%   │ │  Elapsed:  1.234s               │ │
│ │  [I] Select     ░░░░░░░░░░░░   0%   │ │  ETA:      0.892s               │ │
│ │  [J] Detok      ░░░░░░░░░░░░   0%   │ │                                  │ │
│ │  [K] Assemble   ░░░░░░░░░░░░   0%   │ │                                  │ │
│ └─────────────────────────────────────┘ └──────────────────────────────────┘ │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │                          STEP TIMING BREAKDOWN                           │ │
│ │                                                                          │ │
│ │  Load ▏ 2ms                                                              │ │
│ │  Parse▏ 0.4ms                                                            │ │
│ │  Resam▏ 4ms                                                              │ │
│ │  Mel  ▏▏▏▏▏ 8ms                                                          │ │
│ │  Encod▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏ 289ms ← CURRENT               │ │
│ │  Decod▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏ ~800ms (est)    │ │
│ │                                                                          │ │
│ │  Target: ═══════════════════════════════════════════════════════ 1500ms  │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │  STATUS: Encoding audio chunk 1/1                                        │ │
│ │  TRANSCRIPT: (waiting for decode...)                                     │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  [q] Quit  [p] Pause  [r] Reset  [s] Step  [+/-] Speed                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### D.2 Probar TUI Playbook

Location: `demos/playbooks/benchmark-tui.yaml`

```yaml
version: "1.0"
name: "Whisper Pipeline Benchmark TUI"
description: "WAPR-BENCH-TUI: Step-by-step performance visualization"

# State machine for pipeline steps
machine:
  id: "benchmark_pipeline"
  initial: "initializing"

  states:
    initializing:
      id: "initializing"
      invariants:
        - description: "TUI renders correctly"
          condition: "frame_matches_snapshot('init')"
        - description: "All progress bars at 0%"
          condition: "all_progress_zero()"

    loading_model:
      id: "loading_model"
      invariants:
        - description: "Loading indicator visible"
          condition: "has_widget('model_loading_spinner')"
        - description: "Memory gauge updating"
          condition: "memory_gauge_increasing()"

    ready:
      id: "ready"
      invariants:
        - description: "Start button enabled"
          condition: "button_enabled('start_benchmark')"
        - description: "Model info displayed"
          condition: "has_text('whisper-tiny')"

    # Pipeline step states
    step_a_source:
      id: "step_a_source"
      invariants:
        - description: "Step A progress bar highlighted"
          condition: "step_highlighted('A')"

    step_b_load:
      id: "step_b_load"
      invariants:
        - description: "Step B progress updating"
          condition: "step_progress_increasing('B')"
        - description: "Disk I/O metric visible"
          condition: "has_metric('disk_throughput')"

    step_c_parse:
      id: "step_c_parse"
      invariants:
        - description: "Sample count updating"
          condition: "metric_increasing('samples_parsed')"

    step_d_resample:
      id: "step_d_resample"
      invariants:
        - description: "Resample ratio displayed"
          condition: "has_text('48kHz → 16kHz') || has_text('16kHz (passthrough)')"

    step_f_mel:
      id: "step_f_mel"
      invariants:
        - description: "FFT progress visible"
          condition: "has_substep('fft')"
        - description: "Mel bins count shown"
          condition: "has_text('80 mel bins')"

    step_g_encode:
      id: "step_g_encode"
      invariants:
        - description: "Encoder layer progress"
          condition: "has_substep('layer_')"
        - description: "Attention visualization updating"
          condition: "attention_vis_active()"

    step_h_decode:
      id: "step_h_decode"
      invariants:
        - description: "Token counter incrementing"
          condition: "metric_increasing('tokens_generated')"
        - description: "Per-token timing shown"
          condition: "has_metric('token_latency_ms')"
        - description: "Partial transcript updating"
          condition: "transcript_updating()"

    step_i_select:
      id: "step_i_select"
      invariants:
        - description: "Selection strategy shown"
          condition: "has_text('greedy') || has_text('beam')"

    step_j_detok:
      id: "step_j_detok"
      invariants:
        - description: "BPE decode active"
          condition: "has_substep('bpe')"

    step_k_assemble:
      id: "step_k_assemble"
      invariants:
        - description: "Timestamp extraction shown"
          condition: "has_substep('timestamps')"

    completed:
      id: "completed"
      invariants:
        - description: "All progress bars at 100%"
          condition: "all_progress_complete()"
        - description: "Final transcript shown"
          condition: "transcript_length() > 0"
        - description: "RTF metric calculated"
          condition: "has_metric('final_rtf')"

    error:
      id: "error"
      invariants:
        - description: "Error message visible"
          condition: "has_error_widget()"

  transitions:
    # Startup sequence
    - id: "init_to_loading"
      from: "initializing"
      to: "loading_model"
      event: "wasm_ready"

    - id: "loading_to_ready"
      from: "loading_model"
      to: "ready"
      event: "model_loaded"

    # Pipeline execution (happy path)
    - id: "start_pipeline"
      from: "ready"
      to: "step_a_source"
      event: "start_clicked"

    - id: "a_to_b"
      from: "step_a_source"
      to: "step_b_load"
      event: "step_a_complete"
      assertions:
        - type: timing
          max_ms: 10

    - id: "b_to_c"
      from: "step_b_load"
      to: "step_c_parse"
      event: "step_b_complete"
      assertions:
        - type: timing
          max_ms: 50

    - id: "c_to_d"
      from: "step_c_parse"
      to: "step_d_resample"
      event: "step_c_complete"
      assertions:
        - type: timing
          max_ms: 10

    - id: "d_to_f"
      from: "step_d_resample"
      to: "step_f_mel"
      event: "step_d_complete"
      assertions:
        - type: timing
          max_ms: 100

    - id: "f_to_g"
      from: "step_f_mel"
      to: "step_g_encode"
      event: "step_f_complete"
      assertions:
        - type: timing
          max_ms: 50

    - id: "g_to_h"
      from: "step_g_encode"
      to: "step_h_decode"
      event: "step_g_complete"
      assertions:
        - type: timing
          max_ms: 500

    - id: "h_to_i"
      from: "step_h_decode"
      to: "step_i_select"
      event: "step_h_complete"
      assertions:
        - type: timing
          max_ms: 2000

    - id: "i_to_j"
      from: "step_i_select"
      to: "step_j_detok"
      event: "step_i_complete"

    - id: "j_to_k"
      from: "step_j_detok"
      to: "step_k_assemble"
      event: "step_j_complete"

    - id: "k_to_complete"
      from: "step_k_assemble"
      to: "completed"
      event: "step_k_complete"

    # Reset from completed
    - id: "reset_from_complete"
      from: "completed"
      to: "ready"
      event: "reset_clicked"

    # Error transitions from any step
    - id: "any_to_error"
      from: "*"
      to: "error"
      event: "pipeline_error"

    - id: "error_to_ready"
      from: "error"
      to: "ready"
      event: "dismiss_error"

  forbidden:
    - from: "step_h_decode"
      to: "step_f_mel"
      reason: "Cannot go backwards in pipeline"

    - from: "completed"
      to: "step_a_source"
      reason: "Must reset before restarting"

# Performance visualization configuration
visualization:
  # Main widgets
  widgets:
    pipeline_progress:
      type: "multi_progress"
      layout: { x: 0, y: 0, width: 50%, height: 40% }
      steps:
        - { id: "A", label: "Source", color: "cyan" }
        - { id: "B", label: "Load", color: "blue" }
        - { id: "C", label: "Parse", color: "green" }
        - { id: "D", label: "Resample", color: "yellow" }
        - { id: "F", label: "Mel", color: "magenta" }
        - { id: "G", label: "Encode", color: "red" }
        - { id: "H", label: "Decode", color: "red", style: "bold" }
        - { id: "I", label: "Select", color: "cyan" }
        - { id: "J", label: "Detok", color: "blue" }
        - { id: "K", label: "Assemble", color: "green" }

    live_metrics:
      type: "gauge_group"
      layout: { x: 50%, y: 0, width: 50%, height: 40% }
      gauges:
        - { id: "rtf", label: "RTF", max: 5.0, warning: 2.0, critical: 4.0 }
        - { id: "memory", label: "Memory", max: 150, unit: "MB" }
        - { id: "cpu", label: "CPU", max: 100, unit: "%" }
        - { id: "tokens", label: "Tokens", dynamic_max: true }

    timing_breakdown:
      type: "horizontal_bar"
      layout: { x: 0, y: 40%, width: 100%, height: 35% }
      scale: "logarithmic"
      target_line: 1500  # Target total time in ms
      colors:
        under_target: "green"
        over_target: "red"
        current: "yellow"

    status_bar:
      type: "text_block"
      layout: { x: 0, y: 75%, width: 100%, height: 15% }
      fields:
        - { id: "status", label: "STATUS" }
        - { id: "transcript", label: "TRANSCRIPT", scrollable: true }

    controls:
      type: "key_hints"
      layout: { x: 0, y: 90%, width: 100%, height: 10% }
      keys:
        - { key: "q", action: "Quit" }
        - { key: "p", action: "Pause" }
        - { key: "r", action: "Reset" }
        - { key: "s", action: "Step" }
        - { key: "+/-", action: "Speed" }

  # Animation settings
  animation:
    frame_rate: 30
    progress_smoothing: true
    highlight_duration_ms: 500

  # Snapshot points for regression testing
  snapshots:
    - id: "init"
      state: "initializing"
      description: "Initial TUI state"

    - id: "ready"
      state: "ready"
      description: "Ready to start benchmark"

    - id: "encoding"
      state: "step_g_encode"
      description: "Encoder in progress"

    - id: "decoding"
      state: "step_h_decode"
      description: "Decoder generating tokens"

    - id: "complete"
      state: "completed"
      description: "Benchmark complete with results"

# Performance assertions for the TUI
performance:
  max_duration_ms: 5000  # Total benchmark time
  max_memory_mb: 200

  step_budgets:
    step_a_source: 10
    step_b_load: 50
    step_c_parse: 10
    step_d_resample: 100
    step_f_mel: 50
    step_g_encode: 500
    step_h_decode: 2000
    step_i_select: 10
    step_j_detok: 10
    step_k_assemble: 10

  rtf_target: 2.0
  rtf_critical: 4.0

performance_assertions:
  - name: "end_to_end_time"
    condition: "total_time_ms() <= 3000"
    critical: "total_time_ms() <= 5000"
    failure_reason: "Pipeline exceeded time budget"

  - name: "rtf_target"
    condition: "rtf() <= 2.0"
    critical: "rtf() <= 4.0"
    failure_reason: "Real-time factor too high"

  - name: "memory_budget"
    condition: "peak_memory_mb() <= 150"
    critical: "peak_memory_mb() <= 200"
    failure_reason: "Memory usage exceeded budget"

  - name: "no_frame_drops"
    condition: "tui_frame_drops() == 0"
    failure_reason: "TUI animation dropped frames"

  - name: "decoder_dominance"
    condition: "step_h_percentage() >= 50"
    failure_reason: "Decoder should dominate runtime (Amdahl's Law validation)"
```

### D.3 TUI Implementation (Rust)

Location: `examples/benchmark_tui.rs`

```rust
//! Benchmark TUI - Interactive pipeline performance visualization
//!
//! Run: cargo run --release --example benchmark_tui -- --model models/whisper-tiny-int8.apr

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline, BarChart},
    Frame, Terminal,
};
use std::{
    io,
    sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use whisper_apr::{WhisperApr, TranscribeOptions};

/// Pipeline step with timing and progress
#[derive(Clone, Debug)]
struct PipelineStep {
    id: char,
    name: &'static str,
    progress: f64,
    elapsed_ms: f64,
    target_ms: f64,
    status: StepStatus,
}

#[derive(Clone, Debug, PartialEq)]
enum StepStatus {
    Pending,
    Running,
    Complete,
    Error(String),
}

/// Application state
struct App {
    steps: Vec<PipelineStep>,
    current_step: usize,
    rtf: f64,
    memory_mb: f64,
    cpu_percent: f64,
    tokens_generated: usize,
    tokens_expected: usize,
    transcript: String,
    status_message: String,
    start_time: Option<Instant>,
    running: bool,
    paused: bool,
    timing_history: Vec<(char, f64)>,
}

impl App {
    fn new() -> Self {
        Self {
            steps: vec![
                PipelineStep { id: 'A', name: "Source", progress: 0.0, elapsed_ms: 0.0, target_ms: 10.0, status: StepStatus::Pending },
                PipelineStep { id: 'B', name: "Load", progress: 0.0, elapsed_ms: 0.0, target_ms: 50.0, status: StepStatus::Pending },
                PipelineStep { id: 'C', name: "Parse", progress: 0.0, elapsed_ms: 0.0, target_ms: 10.0, status: StepStatus::Pending },
                PipelineStep { id: 'D', name: "Resample", progress: 0.0, elapsed_ms: 0.0, target_ms: 100.0, status: StepStatus::Pending },
                PipelineStep { id: 'F', name: "Mel", progress: 0.0, elapsed_ms: 0.0, target_ms: 50.0, status: StepStatus::Pending },
                PipelineStep { id: 'G', name: "Encode", progress: 0.0, elapsed_ms: 0.0, target_ms: 500.0, status: StepStatus::Pending },
                PipelineStep { id: 'H', name: "Decode", progress: 0.0, elapsed_ms: 0.0, target_ms: 2000.0, status: StepStatus::Pending },
                PipelineStep { id: 'I', name: "Select", progress: 0.0, elapsed_ms: 0.0, target_ms: 10.0, status: StepStatus::Pending },
                PipelineStep { id: 'J', name: "Detok", progress: 0.0, elapsed_ms: 0.0, target_ms: 10.0, status: StepStatus::Pending },
                PipelineStep { id: 'K', name: "Assemble", progress: 0.0, elapsed_ms: 0.0, target_ms: 10.0, status: StepStatus::Pending },
            ],
            current_step: 0,
            rtf: 0.0,
            memory_mb: 0.0,
            cpu_percent: 0.0,
            tokens_generated: 0,
            tokens_expected: 30,
            transcript: String::new(),
            status_message: "Press [s] to start benchmark".to_string(),
            start_time: None,
            running: false,
            paused: false,
            timing_history: Vec::new(),
        }
    }

    fn total_elapsed_ms(&self) -> f64 {
        self.steps.iter().map(|s| s.elapsed_ms).sum()
    }

    fn total_target_ms(&self) -> f64 {
        self.steps.iter().map(|s| s.target_ms).sum()
    }
}

fn main() -> Result<(), io::Error> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let app = Arc::new(Mutex::new(App::new()));

    // Run app
    let res = run_app(&mut terminal, app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: Arc<Mutex<App>>,
) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, &app.lock().expect("lock")))?;

        if event::poll(Duration::from_millis(33))? {
            if let Event::Key(key) = event::read()? {
                let mut app = app.lock().expect("lock");
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('p') => app.paused = !app.paused,
                    KeyCode::Char('r') => *app = App::new(),
                    KeyCode::Char('s') => {
                        if !app.running {
                            app.running = true;
                            app.start_time = Some(Instant::now());
                            app.status_message = "Running benchmark...".to_string();
                        }
                    }
                    _ => {}
                }
            }
        }

        // Simulate progress if running
        {
            let mut app = app.lock().expect("lock");
            if app.running && !app.paused && app.current_step < app.steps.len() {
                let step = &mut app.steps[app.current_step];
                step.status = StepStatus::Running;
                step.progress += 0.05; // Simulate progress
                step.elapsed_ms += 3.0; // Simulate time

                if step.progress >= 1.0 {
                    step.progress = 1.0;
                    step.status = StepStatus::Complete;
                    app.timing_history.push((step.id, step.elapsed_ms));
                    app.current_step += 1;

                    if app.current_step >= app.steps.len() {
                        app.running = false;
                        app.status_message = "Benchmark complete!".to_string();
                        app.rtf = app.total_elapsed_ms() / 1500.0; // 1.5s audio
                    }
                }

                // Update metrics
                app.memory_mb = 78.0 + (app.current_step as f64 * 5.0);
                app.cpu_percent = 85.0 + (rand::random::<f64>() * 10.0);
                if app.current_step >= 6 {
                    app.tokens_generated = ((app.steps[6].progress * 20.0) as usize).min(20);
                    if app.tokens_generated > 0 {
                        app.transcript = "The quick brown fox...".to_string();
                    }
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &App) {
    // Create layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Percentage(40),  // Progress + Metrics
            Constraint::Percentage(35),  // Timing breakdown
            Constraint::Percentage(15),  // Status
            Constraint::Percentage(10),  // Controls
        ])
        .split(f.area());

    // Top section: Progress and Metrics side by side
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(chunks[0]);

    // Pipeline progress
    render_pipeline_progress(f, top_chunks[0], app);

    // Live metrics
    render_live_metrics(f, top_chunks[1], app);

    // Timing breakdown
    render_timing_breakdown(f, chunks[1], app);

    // Status bar
    render_status(f, chunks[2], app);

    // Controls
    render_controls(f, chunks[3]);
}

fn render_pipeline_progress(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" PIPELINE PROGRESS ")
        .borders(Borders::ALL);

    let inner = block.inner(area);
    f.render_widget(block, area);

    let step_height = (inner.height as usize / app.steps.len()).max(1);

    for (i, step) in app.steps.iter().enumerate() {
        let step_area = Rect {
            x: inner.x,
            y: inner.y + (i * step_height) as u16,
            width: inner.width,
            height: step_height as u16,
        };

        let (color, modifier) = match step.status {
            StepStatus::Pending => (Color::DarkGray, Modifier::empty()),
            StepStatus::Running => (Color::Yellow, Modifier::BOLD),
            StepStatus::Complete => (Color::Green, Modifier::empty()),
            StepStatus::Error(_) => (Color::Red, Modifier::BOLD),
        };

        let gauge = Gauge::default()
            .block(Block::default())
            .gauge_style(Style::default().fg(color).add_modifier(modifier))
            .percent((step.progress * 100.0) as u16)
            .label(format!("[{}] {} {:.0}%", step.id, step.name, step.progress * 100.0));

        f.render_widget(gauge, step_area);
    }
}

fn render_live_metrics(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" LIVE METRICS ")
        .borders(Borders::ALL);

    let inner = block.inner(area);
    f.render_widget(block, area);

    let metrics_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .split(inner);

    // RTF gauge
    let rtf_color = if app.rtf < 2.0 { Color::Green } else if app.rtf < 4.0 { Color::Yellow } else { Color::Red };
    let rtf_gauge = Gauge::default()
        .gauge_style(Style::default().fg(rtf_color))
        .percent(((app.rtf / 5.0) * 100.0).min(100.0) as u16)
        .label(format!("RTF: {:.2}x", app.rtf));
    f.render_widget(rtf_gauge, metrics_layout[0]);

    // Memory gauge
    let mem_gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent(((app.memory_mb / 150.0) * 100.0) as u16)
        .label(format!("Memory: {:.0}MB / 150MB", app.memory_mb));
    f.render_widget(mem_gauge, metrics_layout[1]);

    // CPU gauge
    let cpu_gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Magenta))
        .percent(app.cpu_percent as u16)
        .label(format!("CPU: {:.0}%", app.cpu_percent));
    f.render_widget(cpu_gauge, metrics_layout[2]);

    // Tokens counter
    let tokens = Paragraph::new(format!("Tokens: {} / ~{}", app.tokens_generated, app.tokens_expected))
        .style(Style::default().fg(Color::White));
    f.render_widget(tokens, metrics_layout[3]);
}

fn render_timing_breakdown(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" STEP TIMING BREAKDOWN ")
        .borders(Borders::ALL);

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Create bar chart data
    let data: Vec<(&str, u64)> = app.steps
        .iter()
        .map(|s| (s.name, s.elapsed_ms as u64))
        .collect();

    let bar_chart = BarChart::default()
        .bar_width(8)
        .bar_gap(1)
        .group_gap(2)
        .bar_style(Style::default().fg(Color::Cyan))
        .value_style(Style::default().fg(Color::White))
        .data(&data)
        .max(500);

    f.render_widget(bar_chart, inner);
}

fn render_status(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" STATUS ")
        .borders(Borders::ALL);

    let text = format!(
        "STATUS: {}\nTRANSCRIPT: {}",
        app.status_message,
        if app.transcript.is_empty() { "(waiting...)" } else { &app.transcript }
    );

    let para = Paragraph::new(text)
        .block(block)
        .style(Style::default().fg(Color::White));

    f.render_widget(para, area);
}

fn render_controls(f: &mut Frame, area: Rect) {
    let controls = Paragraph::new(" [q] Quit  [p] Pause  [r] Reset  [s] Start  ")
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::TOP));

    f.render_widget(controls, area);
}
```

### D.4 Probar TUI Test Suite

Location: `demos/tests/src/benchmark_tui_tests.rs`

```rust
//! Probar TUI tests for benchmark visualization
//!
//! These tests validate the TUI renders correctly and responds to events.

use probar::tui::{TuiTestBackend, TuiSnapshot, expect_frame, FrameAssertion};
use probar::playbook::{PlaybookRunner, Playbook};

/// Test initial TUI state renders correctly
#[test]
fn test_benchmark_tui_initial_state() {
    let backend = TuiTestBackend::new(120, 40);
    let mut app = BenchmarkApp::new();

    // Render initial frame
    let frame = backend.render(|f| app.ui(f));

    // Verify initial state
    expect_frame(&frame)
        .contains_text("PIPELINE PROGRESS")
        .contains_text("LIVE METRICS")
        .contains_text("[A] Source")
        .contains_text("RTF:")
        .contains_text("[q] Quit")
        .all_gauges_at(0);
}

/// Test pipeline progress updates correctly
#[test]
fn test_pipeline_progress_animation() {
    let backend = TuiTestBackend::new(120, 40);
    let mut app = BenchmarkApp::new();

    // Start benchmark
    app.handle_key(KeyCode::Char('s'));
    assert!(app.running);

    // Simulate progress
    for _ in 0..20 {
        app.tick();
    }

    let frame = backend.render(|f| app.ui(f));

    // Verify progress
    expect_frame(&frame)
        .gauge_progress("A", 100)  // Step A complete
        .gauge_progress("B", |p| p > 0);  // Step B started
}

/// Test step transitions follow state machine
#[test]
fn test_step_transitions() {
    let playbook = Playbook::from_file("demos/playbooks/benchmark-tui.yaml")
        .expect("load playbook");

    let runner = PlaybookRunner::new(playbook);

    // Verify state machine is valid
    let validation = runner.validate_state_machine();
    assert!(validation.is_valid, "State machine validation failed: {:?}", validation.issues);

    // Verify no unreachable states
    assert!(validation.reachability.unreachable_states.is_empty());

    // Verify determinism
    assert!(validation.determinism.is_deterministic);
}

/// Test forbidden transitions are rejected
#[test]
fn test_forbidden_transitions_rejected() {
    let backend = TuiTestBackend::new(120, 40);
    let mut app = BenchmarkApp::new();

    // Try to go backwards (should be rejected)
    app.current_step = 6;  // At decode
    app.try_transition_to(4);  // Try to go to mel

    // Should still be at decode
    assert_eq!(app.current_step, 6);
}

/// Test performance assertions
#[test]
fn test_performance_assertions() {
    let playbook = Playbook::from_file("demos/playbooks/benchmark-tui.yaml")
        .expect("load playbook");

    let runner = PlaybookRunner::new(playbook);

    // Run with simulated timing
    let result = runner.run_with_metrics(SimulatedMetrics {
        total_time_ms: 2500.0,  // Under 3000ms target
        rtf: 1.7,               // Under 2.0 target
        peak_memory_mb: 120.0,  // Under 150MB target
        step_h_percentage: 65.0, // Decoder dominance
    });

    assert!(result.all_passed(), "Performance assertions failed: {:?}", result.failures);
}

/// Test TUI snapshot regression
#[test]
fn test_tui_snapshot_regression() {
    let backend = TuiTestBackend::new(120, 40);
    let mut app = BenchmarkApp::new();

    // Capture initial state
    let frame = backend.render(|f| app.ui(f));
    let snapshot = TuiSnapshot::capture(&frame);

    // Compare against golden snapshot
    snapshot.assert_matches_golden("snapshots/benchmark_tui_init.snap")
        .with_tolerance(0.01);  // 1% pixel difference tolerance
}

/// Test decode step shows token progress
#[test]
fn test_decode_token_progress() {
    let backend = TuiTestBackend::new(120, 40);
    let mut app = BenchmarkApp::new();

    // Advance to decode step
    app.advance_to_step(6);

    // Simulate token generation
    for i in 0..5 {
        app.generate_token(format!("token_{}", i));
        let frame = backend.render(|f| app.ui(f));

        expect_frame(&frame)
            .contains_text(&format!("Tokens: {}", i + 1))
            .metric_increasing("tokens_generated");
    }
}

/// Test error state rendering
#[test]
fn test_error_state_display() {
    let backend = TuiTestBackend::new(120, 40);
    let mut app = BenchmarkApp::new();

    // Trigger error
    app.trigger_error("Model load failed: out of memory");

    let frame = backend.render(|f| app.ui(f));

    expect_frame(&frame)
        .contains_text("ERROR")
        .contains_text("out of memory")
        .has_style(Color::Red);
}
```

### D.5 Running the TUI Simulation

```bash
# Build and run the TUI benchmark
cargo run --release --example benchmark_tui -- \
    --model models/whisper-tiny-int8.apr \
    --audio demos/test-audio/test-speech-3s.wav

# Run with probar playbook validation
probar playbook demos/playbooks/benchmark-tui.yaml --validate

# Run TUI tests
cargo test --package whisper-apr-demo-tests benchmark_tui

# Capture golden snapshots
cargo test --package whisper-apr-demo-tests benchmark_tui -- --update-snapshots

# Run full probar test suite with TUI
probar test --tui --playbook demos/playbooks/benchmark-tui.yaml
```

### D.6 Integration with Renacer Tracing

The TUI integrates with renacer for real-time trace visualization:

```rust
// In benchmark_tui.rs, add renacer integration
use renacer::{Tracer, SpanEvent};

impl App {
    /// Subscribe to renacer trace events for live updates
    fn subscribe_to_traces(&mut self, tracer: &Tracer) {
        tracer.on_span_enter(|span| {
            if span.name.starts_with("step_") {
                let step_id = span.name.chars().nth(5).unwrap_or('?');
                self.set_step_running(step_id);
            }
        });

        tracer.on_span_exit(|span, duration| {
            if span.name.starts_with("step_") {
                let step_id = span.name.chars().nth(5).unwrap_or('?');
                self.set_step_complete(step_id, duration.as_secs_f64() * 1000.0);
            }
        });
    }
}
```

### D.7 Makefile Integration

Add to `Makefile`:

```makefile
# TUI Benchmark
.PHONY: bench-tui bench-tui-test bench-tui-snapshot

bench-tui: ## Run interactive benchmark TUI
	cargo run --release --example benchmark_tui -- \
		--model models/whisper-tiny-int8.apr \
		--audio demos/test-audio/test-speech-3s.wav

bench-tui-test: ## Run probar TUI tests
	cargo test --package whisper-apr-demo-tests benchmark_tui
	probar playbook demos/playbooks/benchmark-tui.yaml --validate

bench-tui-snapshot: ## Update TUI golden snapshots
	cargo test --package whisper-apr-demo-tests benchmark_tui -- --update-snapshots

bench-tui-record: ## Record TUI session for documentation
	probar record --tui --output docs/benchmark-tui-recording.gif \
		cargo run --release --example benchmark_tui
```

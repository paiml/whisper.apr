# Transcribe Folder: 2x whisper.cpp Performance Specification

**WAPR-PERF-004: Beating whisper.cpp by 2x on Tiny/Small Models**

| Field | Value |
|-------|-------|
| Status | PLANNING |
| Author | Claude Code |
| Created | 2026-01-20 |
| PMAT Roadmap ID | `WAPR-PERF-004` |
| Toyota Way Phase | Kaizen (改善) - Performance Breakthrough |
| Batuta Stack | trueno 0.13.0, aprender 0.24.1, realizar 0.6.3 |
| Target Models | whisper-tiny (39M), whisper-small (244M) |
| Performance Goal | **2x faster** than whisper.cpp (CPU and GPU) |

---

## Executive Summary

This specification defines the systematic approach to achieve **2x performance improvement** over whisper.cpp for the Whisper tiny and small models on both CPU and GPU backends. The approach leverages the batuta stack's advanced primitives (trueno SIMD/PTX, realizar inference engine, aprender format) combined with rigorous profiling via renacer.

### Current State vs Target

**Whisper Tiny (39M params):**

| Implementation | Backend | RTF | tok/s | Target RTF | Speedup |
|----------------|---------|-----|-------|------------|---------|
| whisper.cpp | CPU (AVX2) | 0.08x | ~200 | - | baseline |
| whisper.cpp | GPU (CUDA) | 0.02x | ~800 | - | baseline |
| **whisper.apr** | **CPU (AVX2)** | **TBD** | **TBD** | **0.04x** | **2x** |
| **whisper.apr** | **GPU (CUDA)** | **TBD** | **TBD** | **0.01x** | **2x** |

**Whisper Small (244M params):**

| Implementation | Backend | RTF | tok/s | Target RTF | Speedup |
|----------------|---------|-----|-------|------------|---------|
| whisper.cpp | CPU (AVX2) | 0.25x | ~65 | - | baseline |
| whisper.cpp | GPU (CUDA) | 0.06x | ~270 | - | baseline |
| **whisper.apr** | **CPU (AVX2)** | **TBD** | **TBD** | **0.125x** | **2x** |
| **whisper.apr** | **GPU (CUDA)** | **TBD** | **TBD** | **0.03x** | **2x** |

### Key Technologies

1. **Fused Kernels** - trueno's fused_layernorm_linear, fused attention (eliminate memory bandwidth)
2. **Flash Attention** - realizar's O(N) memory attention (Dao et al., 2022)
3. **Speculative Decoding** - realizar's draft-verify pipeline (Leviathan et al., 2023)
4. **KV Cache Compression** - PagedKVCache with ZRAM (Kwon et al., 2023)
5. **INT8 Acceleration** - Native INT8 matmul on RTX 4090 (8.9 compute)
6. **Chunked Streaming** - Pipeline parallelism for long audio

---

## 1. Architecture Overview

### 1.1 Component Responsibility Matrix

| Responsibility | whisper.apr | realizar | trueno | renacer |
|---------------|-------------|----------|--------|---------|
| **Audio Pipeline** | ✅ Primary | ❌ | ❌ | Trace |
| **Mel Spectrogram** | ✅ Primary | ❌ | SIMD kernels | Trace |
| **Encoder** | ✅ Structure | Attention | MatMul/GELU | Profile |
| **Decoder** | ✅ Structure | KV Cache | MatMul | Profile |
| **Quantization** | Format | Dequant | Kernels | ❌ |
| **GPU Dispatch** | Config | ✅ Primary | CUDA PTX | GPU Metrics |
| **Profiling** | ❌ | Trace points | ❌ | ✅ Primary |

### 1.2 Data Flow (Optimized Pipeline)

```
Audio Input (f32[])
      │
      ▼ ──────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Audio Preprocessing (trueno SIMD)                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                │
│  │ Resample    │──▶│ MelFilter   │──▶│ Normalize   │                │
│  │ (16kHz)     │   │ (80 bins)   │   │ (log-mel)   │                │
│  └─────────────┘   └─────────────┘   └─────────────┘                │
│  AVX2/AVX512: Vectorized FFT, filterbank multiplication             │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼ mel[80, 3000]
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Encoder (realizar + trueno)                               │
│  ┌─────────────┐   ┌─────────────────────────────────────────────┐  │
│  │ Conv1d ×2   │──▶│ Transformer Blocks (n_layers)               │  │
│  │ (GELU)      │   │ ┌─────────────────────────────────────────┐ │  │
│  └─────────────┘   │ │ FusedLayerNormLinear → Flash Attention  │ │  │
│                    │ │ FusedLayerNormLinear → FFN (SwiGLU)     │ │  │
│                    │ └─────────────────────────────────────────┘ │  │
│                    └─────────────────────────────────────────────┘  │
│  GPU: FlashAttention-2 with 8x memory reduction                     │
│  CPU: Blocked attention with cache-aware tiling                     │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼ encoder_output[1, 1500, d_model]
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Decoder (realizar + trueno)                               │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Autoregressive Generation with KV Cache                         ││
│  │ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            ││
│  │ │ Self-Attn    │─▶│ Cross-Attn   │─▶│ FFN          │            ││
│  │ │ (KV cached)  │  │ (encoder KV) │  │ (SwiGLU)     │            ││
│  │ └──────────────┘  └──────────────┘  └──────────────┘            ││
│  │                                                                  ││
│  │ PagedKVCache: O(1) memory per token, ZRAM compression           ││
│  └─────────────────────────────────────────────────────────────────┘│
│  Speculative Decoding: 3-5x speedup via draft model                 │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼ tokens[]
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Token Decoding (whisper.apr)                              │
│  BPE detokenization → UTF-8 text                                    │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
   Transcription Result
```

### 1.3 Folder Processing Strategy

To support batched validation and bulk transcription, strict path handling logic is required.

**CLI Command:**
```bash
whisper-apr-cli transcribe-folder \
    --input-dir ./raw_audio \
    --output-dir ./transcripts \
    --format json \
    --recursive \
    --workers 4
```

**Path Resolution Logic:**

| Input Structure | Output Structure | Notes |
|----------------|------------------|-------|
| `./raw/a.wav` | `./trans/a.json` | Flat mapping if no subdirs |
| `./raw/sub/b.mp3` | `./trans/sub/b.json` | **Structure Mirroring** (Required) |
| `./raw/c.wav` | `./trans/c.json` | Extension replacement |
| `./raw/d.wav` | *Skip* | If `./trans/d.json` exists (Resumable) |

**Conflict Resolution:**
1.  **Mirroring:** Output directory must verify/create subdirectories to match input tree.
2.  **Atomicity:** Write to `${filename}.tmp` then rename to target to prevent partial writes on crash.
3.  **Determinism:** Parallel workers must deterministically pick files (e.g., sorted list) to ensure reproducible logs.

---

## 2. Performance Analysis Framework

### 2.1 Profiling Toolchain (renacer + trueno-explain)

```bash
# Full system trace with source correlation
renacer -s -- whisper-apr-cli transcribe --file audio.wav -v

# GPU kernel profiling
trueno-explain profile whisper-apr-cli transcribe --file audio.wav --gpu

# Flame graph generation
renacer --flamegraph -o profile.svg -- whisper-apr-cli transcribe --file audio.wav

# Comparative benchmark
./scripts/benchmark-vs-whisper-cpp.sh --model tiny --iterations 100
```

### 2.2 QA Script Architecture (aprender Showcase Style)

```bash
#!/usr/bin/env bash
# scripts/perf-qa-2x-whisper-cpp.sh
#
# Automated QA validation: whisper.apr must be 2x faster than whisper.cpp
# Exit code 0 = PASS, non-zero = FAIL (Jidoka principle)

set -euo pipefail

# Configuration
MODELS=("tiny" "small")
BACKENDS=("cpu" "gpu")
TEST_AUDIO="demos/test-audio/test-speech-30s.wav"
ITERATIONS=10
SPEEDUP_TARGET=2.0

# Baseline measurement (whisper.cpp)
measure_whisper_cpp() {
    # Mandate optimized build check
    if ! ldd /home/noah/.local/bin/main | grep -q "libcublas"; then
        echo "WARNING: whisper.cpp might be unoptimized!" >&2
    fi
    # ... (measurement logic same as previous)
}
# ... (rest of script)
```

---

## 3. Optimization Strategies

### 3.1 CPU Optimizations (trueno 0.13.0)

| Optimization | Expected Speedup | Implementation |
|--------------|------------------|----------------|
| **AVX-512 SIMD** | 2x over AVX2 | `trueno::simd::avx512` |
| **Fused LayerNorm+Linear** | 1.5x | `trueno::ops::fused_layernorm_linear` |
| **Cache-Blocked MatMul** | 1.3x | `trueno::ops::matmul_blocked` |
| **Parallel FFT** | 1.2x | `rayon` + `rustfft` |
| **INT8 VNNI** | 2x on Ice Lake+ | `trueno::simd::vnni_i8` |
| **Memory Prefetch** | 1.1x | `_mm_prefetch` hints |

**Combined theoretical speedup:** 2x × 1.5x × 1.3x × 1.2x × 2x × 1.1x = **10.3x**
**Conservative estimate:** **2-3x** (accounting for Amdahl's law)

### 3.2 GPU Optimizations (trueno-gpu 0.4.8)

| Optimization | Expected Speedup | Implementation |
|--------------|------------------|----------------|
| **FlashAttention-2** | 3x memory, 2x speed | `realizar::layers::FlashAttention` |
| **Tensor Cores (FP16)** | 4x over FP32 | `trueno_gpu::tensor_core_gemm` |
| **INT8 Tensor Cores** | 2x over FP16 | `trueno_gpu::int8_gemm` |
| **Fused Kernels** | 1.5x | Custom PTX via trueno-gpu |
| **Async Copy** | 1.2x | `cudaMemcpyAsync` |
| **Persistent Kernels** | 1.3x | Reduce launch overhead |

**Combined theoretical speedup:** 2x × 4x × 2x × 1.5x × 1.2x × 1.3x = **37.4x**
**Conservative estimate:** **2-4x** (memory bandwidth limited)

### 3.3 Speculative Decoding (realizar 0.6.3)

```rust
use realizar::speculative::{SpeculativeConfig, DraftModel};

let config = SpeculativeConfig {
    draft_model: DraftModel::Tiny,  // Use tiny as draft for small
    target_model: Model::Small,
    lookahead: 4,                   // Predict 4 tokens ahead
    acceptance_threshold: 0.9,
};

// Expected speedup: 3-5x for decoder phase
let result = speculative_decode(&encoder_output, &config)?;
```

### 3.4 Chunked Streaming (Long Audio)

```rust
// Process long audio in 30s chunks with overlap
const CHUNK_SIZE: usize = 30 * 16000;  // 30 seconds
const OVERLAP: usize = 2 * 16000;       // 2 second overlap

fn transcribe_chunked(audio: &[f32]) -> Vec<Segment> {
    let chunks: Vec<_> = audio
        .windows(CHUNK_SIZE + OVERLAP)
        .step_by(CHUNK_SIZE)
        .collect();

    // Process chunks in parallel (pipeline parallelism)
    chunks
        .par_iter()
        .map(|chunk| transcribe_chunk(chunk))
        .flatten()
        .merge_overlapping()
        .collect()
}
```

---

## 4. Peer-Reviewed Citations

*(Same as previous version)*

---

## 5. Toyota Way Framework

### 5.1 Genchi Genbutsu (現地現物) - Go and See

**Direct Measurement Protocol:**

```bash
# Step 1: Baseline whisper.cpp measurement
/home/noah/.local/bin/main \
    -m /home/noah/src/whisper.cpp/models/ggml-tiny.bin \
    -f test-audio.wav \
    --no-timestamps 2>&1 | grep "total time"

# Step 2: whisper.apr measurement with same audio
whisper-apr-cli transcribe --file test-audio.wav --model tiny -v 2>&1 | grep "Total:"

# Step 3: renacer system trace for bottleneck identification
renacer -s --flamegraph -o baseline.svg -- whisper-apr-cli transcribe --file test-audio.wav

# Step 4: trueno-explain GPU kernel profiling
trueno-explain profile whisper-apr-cli transcribe --file test-audio.wav --gpu
```

### 5.2 Five Whys Analysis

**Why is whisper.apr slower than whisper.cpp?**

| Level | Question | Answer |
|-------|----------|--------|
| Why 1 | Why is RTF higher than whisper.cpp? | Decoder step is slower |
| Why 2 | Why is decoder slower? | KV cache operations are memory-bound |
| Why 3 | Why is KV cache memory-bound? | No cache-aware tiling or compression |
| Why 4 | Why no cache optimization? | Using basic Vec<f32> storage |
| Why 5 | Root cause? | **Need PagedKVCache with ZRAM compression** |

| Level | Question | Answer |
|-------|----------|--------|
| Why 1 | Why is attention slow? | O(N²) memory access pattern |
| Why 2 | Why O(N²) memory? | Standard attention reads full K,V |
| Why 3 | Why read full K,V? | No FlashAttention implementation |
| Why 4 | Why no FlashAttention? | Not integrated from realizar |
| Why 5 | Root cause? | **Need realizar::layers::FlashAttention integration** |

### 5.3 Jidoka (自働化) - Stop on Defect

```yaml
# .pmat-metrics.toml quality gates
[performance_gates]
# Stop CI if performance regresses
tiny_cpu_rtf_max = 0.04    # 2x faster than whisper.cpp (0.08x)
tiny_gpu_rtf_max = 0.01    # 2x faster than whisper.cpp (0.02x)
small_cpu_rtf_max = 0.125  # 2x faster than whisper.cpp (0.25x)
small_gpu_rtf_max = 0.03   # 2x faster than whisper.cpp (0.06x)

# Jidoka: Automatic failure on regression
[quality_gates.jidoka]
stop_on_regression = true
regression_threshold_percent = 5.0
```

### 5.4 Kaizen (改善) - Continuous Improvement

**Incremental Optimization Roadmap:**

| Phase | Focus | Expected Gain | Cumulative |
|-------|-------|---------------|------------|
| 1 | Chunked streaming (fix truncation bug) | - | - |
| 2 | FlashAttention integration | 2x | 2x |
| 3 | Fused LayerNorm+Linear | 1.3x | 2.6x |
| 4 | PagedKVCache | 1.2x | 3.1x |
| 5 | Speculative decoding | 1.5x | 4.7x |
| 6 | INT8 Tensor Cores | 1.5x | 7.0x |

---

## 6. 100-Point Popperian Falsification Checklist

The scientific method requires attempting to **falsify** hypotheses, not confirm them. Each checkpoint attempts to prove whisper.apr cannot achieve 2x performance.

### Section A: Baseline Measurement (Points 1-15)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 1 | whisper.cpp baseline invalid | Re-run 10x, check variance | σ < 5% of mean |
| 2 | Audio file corrupt | ffprobe validation | Valid WAV/MP3/MP4 |
| 3 | Model mismatch | Compare vocab size, dims | Exact match |
| 4 | RTF calculation wrong | Manual: wall_time / audio_len | Formula correct |
| 5 | GPU not used (whisper.cpp) | nvidia-smi during run | GPU utilization > 80% |
| 6 | CPU throttling | Check frequency during run | Stable boost clock |
| 7 | Memory pressure | Check swap usage | No swapping |
| 8 | Disk I/O bottleneck | Monitor read latency | < 1ms average |
| 9 | Model not cached | Second run same speed | No load time diff |
| 10 | Wrong audio sample rate | Check 16kHz conversion | Exact 16000 Hz |
| 11 | Audio truncation | Check output length | Full audio processed |
| 12 | **Weak Baseline Configuration** | Check `whisper.cpp` flags | **MUST** have AVX2/FMA/F16C/BLAS |
| 13 | CUDA version mismatch | Check nvcc version | CUDA 12.x |
| 14 | CPU model different | lscpu comparison | Same hardware |
| 15 | Thermal throttling | Monitor temps | < 90°C sustained |

### Section B: whisper.apr Correctness (Points 16-30)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 16 | Output different from whisper.cpp | WER comparison | WER < 5% |
| 17 | **Hallucination Detection** | **N-gram Analysis** | **Any 5-gram repeated >3x in 10s = FAIL** |
| 18 | EOT token not detected | Check termination | Stops at EOT |
| 19 | Language detection wrong | Compare detected lang | Matches audio |
| 20 | Timestamps misaligned | Compare segment times | < 100ms drift |
| 21 | UTF-8 encoding wrong | Check for mojibake | Valid UTF-8 |
| 22 | Special characters lost | Test punctuation | Preserved |
| 23 | Numerals wrong | "2+2=4" test | Correct digits |
| 24 | Multilingual broken | Test non-English | Correct output |
| 25 | Long audio fails | 10+ minute test | Full transcription |
| 26 | Short audio fails | 1 second test | Valid output |
| 27 | Silence handling wrong | Test silent audio | Empty or silence marker |
| 28 | Noise handling wrong | Test noisy audio | Reasonable output |
| 29 | Multiple speakers wrong | Test conversation | Distinguishes turns |
| 30 | Streaming broken | Test chunked input | Consistent output |

### Section C: CPU Performance (Points 31-50)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 31 | AVX2 not used | Check disassembly | AVX2 instructions present |
| 32 | AVX-512 not used (if available) | Check CPU flags | Uses if available |
| 33 | SIMD width suboptimal | Profile vector ops | 256/512-bit ops |
| 34 | Cache misses high | perf stat | L3 miss < 5% |
| 35 | Branch mispredictions | perf stat | < 1% misprediction |
| 36 | Memory bandwidth saturated | mbw benchmark | < 80% theoretical |
| 37 | Thread scaling poor | Test 1,2,4,8 threads | Near-linear to 4 |
| 38 | **Optimization Interference** | **Test A+B vs A, B** | **Combined Speed > Individual Speed** |
| 39 | Allocation pressure | Count mallocs | < 1000/inference |
| 40 | Copy overhead | Profile memcpy | < 5% of total |
| 41 | FFT not vectorized | Profile mel computation | Uses SIMD FFT |
| 42 | MatMul not tiled | Check blocking factor | Cache-aware tiles |
| 43 | LayerNorm not fused | Check kernel fusion | Single pass |
| 44 | Attention not blocked | Check attention impl | Blocked K,V access |
| 45 | Softmax unstable | Check for overflow | No NaN/Inf |
| 46 | GELU approximation wrong | Compare to reference | < 1e-5 error |
| 47 | INT8 VNNI not used | Check instruction mix | VNNI if available |
| 48 | Prefetch missing | Check memory access | Prefetch hints |
| 49 | tiny model < 0.04x RTF | Benchmark 10x | Median < 0.04x |
| 50 | small model < 0.125x RTF | Benchmark 10x | Median < 0.125x |

### Section D: GPU Performance (Points 51-70)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 51 | GPU not detected | Check device enum | RTX 4090 found |
| 52 | Wrong compute capability | Check CC | 8.9 for RTX 4090 |
| 53 | Tensor Cores not used | NSight profile | TC instructions |
| 54 | FP16 not used | Check precision | FP16 accumulate |
| 55 | INT8 not used | Check quantized path | INT8 matmul |
| 56 | Memory transfer dominates | Profile H2D/D2H | < 10% of total |
| 57 | Kernel launch overhead | Count launches | < 100/inference |
| 58 | Occupancy too low | NSight metrics | > 50% occupancy |
| 59 | Shared memory unused | Check kernel config | Uses shared mem |
| 60 | Register spilling | Check spill count | No spilling |
| 61 | Warp divergence | Check efficiency | > 90% efficiency |
| 62 | Memory coalescing poor | Check transactions | Coalesced access |
| 63 | FlashAttention not used | Profile attention | O(N) memory |
| 64 | Async copy not used | Check memcpy async | Uses cudaMemcpyAsync |
| 65 | Stream parallelism poor | Check concurrent ops | Multiple streams |
| 66 | KV cache on CPU | Check allocation | GPU-resident |
| 67 | Fused kernels missing | Check kernel count | Fused ops |
| 68 | PTX not optimized | Check generated PTX | Optimized code |
| 69 | tiny model < 0.01x RTF | Benchmark 10x | Median < 0.01x |
| 70 | small model < 0.03x RTF | Benchmark 10x | Median < 0.03x |

### Section E: Memory Efficiency (Points 71-85)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 71 | Peak memory too high | Monitor RSS | < 1.5x model size |
| 72 | Memory leak detected | Run 100x, check growth | No growth |
| 73 | KV cache unbounded | Check cache size | Bounded by max_tokens |
| 74 | Activation memory high | Profile peak usage | < 2x batch memory |
| 75 | Weight duplication | Check tensor aliasing | No duplicates |
| 76 | Intermediate tensors leak | Track allocations | All freed |
| 77 | PagedKVCache fragmentation | Check page utilization | > 90% utilization |
| 78 | ZRAM compression low | Check compression ratio | > 2x compression |
| 79 | GPU memory leak | nvidia-smi monitoring | Stable VRAM |
| 80 | Pinned memory misuse | Check allocation type | Appropriate pinned |
| 81 | Batch memory scales linearly | Test batch 1,2,4,8 | O(batch) memory |
| 82 | Streaming memory bounded | Test long audio | Constant memory |
| 83 | Model loading copies | Check mmap usage | Zero-copy if possible |
| 84 | Tokenizer memory leak | Run 1000x tokenize | No growth |
| 85 | Audio buffer leak | Process 1000 files | No growth |

### Section F: Latency Breakdown (Points 86-95)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 86 | Model load dominates | Profile cold start | < 500ms load |
| 87 | Mel spectrogram slow | Profile audio pipeline | < 5% of total |
| 88 | Encoder dominates | Profile encoder | < 30% of total |
| 89 | Decoder dominates | Profile decoder | < 60% of total |
| 90 | Tokenization slow | Profile BPE | < 1% of total |
| 91 | First token latency high | Measure TTFT | < 50ms (GPU) |
| 92 | Inter-token latency high | Measure ITL | < 5ms (GPU) |
| 93 | Batch latency scales | Test batch sizes | Sublinear scaling |
| 94 | Prefill/decode ratio wrong | Profile phases | Prefill < decode |
| 95 | Warmup required | Compare cold/warm | < 2x difference |

### Section G: Comparative Analysis (Points 96-100)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 96 | whisper.apr not 2x faster (CPU tiny) | Benchmark | RTF < 0.04x |
| 97 | whisper.apr not 2x faster (GPU tiny) | Benchmark | RTF < 0.01x |
| 98 | whisper.apr not 2x faster (CPU small) | Benchmark | RTF < 0.125x |
| 99 | whisper.apr not 2x faster (GPU small) | Benchmark | RTF < 0.03x |
| 100 | Regression after optimization | CI benchmark | No regression > 5% |

### Section H: Folder & Path Determinism (Points 101-110)

| # | Falsification Test | Method | Pass Criteria |
|---|-------------------|--------|---------------|
| 101 | **Structure not mirrored** | Deeply nested test | `./out/a/b/c.json` exists |
| 102 | **Format extension mismatch** | `--format json` check | `.json`, not `.txt` |
| 103 | **Atomicity violation** | Kill process mid-write | No partial files |
| 104 | **Overwrite behavior** | Run twice | Timestamp updates (or skips if configured) |
| 105 | **Relative path failure** | Use `../` in args | Resolves correctly |
| 106 | **Missing parent dirs** | Output to non-existent | Creates parents |
| 107 | **Log determinism** | Run 5x parallel | Log order sorted/tagged |
| 108 | **Hidden file leakage** | Input has `.git` | Ignores hidden files |
| 109 | **Symlink loops** | Create circular link | Errors or halts safely |
| 110 | **Space in path failure** | `My Documents/audio` | Handles spaces correctly |

---

## 7. Implementation Roadmap

### Phase 1: Fix Chunked Streaming (Week 1)
1. Implement 30-second chunking with overlap
2. Fix audio truncation bug in `compute_mel`
3. Add proper segment merging for long audio
4. Validate full transcription accuracy

### Phase 2: FlashAttention Integration (Week 2)
1. Integrate `realizar::layers::FlashAttention`
2. Add O(N) memory attention path
3. Validate numerical equivalence
4. Benchmark 2x memory reduction

### Phase 3: Fused Kernels (Week 3)
1. Integrate `trueno::ops::fused_layernorm_linear`
2. Fuse QKV projection with LayerNorm
3. Fuse FFN with activation
4. Benchmark 1.3x speedup

### Phase 4: KV Cache Optimization (Week 4)
1. Integrate `realizar::cache::PagedKVCache`
2. Add ZRAM compression via `trueno-zram-adaptive`
3. Implement cache-aware attention
4. Benchmark memory reduction

### Phase 5: Speculative Decoding (Week 5)
1. Integrate `realizar::speculative`
2. Use tiny model as draft for small
3. Tune lookahead and acceptance threshold
4. Benchmark 1.5x decoder speedup

### Phase 6: INT8 Acceleration (Week 6)
1. Enable INT8 Tensor Core path on RTX 4090
2. Integrate `trueno_gpu::int8_gemm`
3. Validate accuracy preservation
4. Benchmark 1.5x additional speedup

---

## 8. Definition of Done (Provisional Corroboration)

*Note: In the Popperian framework, a theory is never "done," only "provisionally corroborated" until falsified by a new test.*

1. `scripts/perf-qa-2x-whisper-cpp.sh` exits 0
2. **All 110 falsification points pass** (including new Folder/Path and Hallucination checks)
3. **tiny CPU: RTF < 0.04x** (2x faster than whisper.cpp)
4. **tiny GPU: RTF < 0.01x** (2x faster than whisper.cpp)
5. **small CPU: RTF < 0.125x** (2x faster than whisper.cpp)
6. **small GPU: RTF < 0.03x** (2x faster than whisper.cpp)
7. No accuracy regression (WER within 1% of whisper.cpp)
8. Memory usage < 1.5x model size
9. Long audio (10+ minutes) fully transcribed
10. CI performance gate prevents regressions

---

## 9. Failure Conditions

- Any model/backend combination slower than whisper.cpp = **FAIL**
- WER > 5% compared to whisper.cpp = **FAIL**
- Memory usage > 2x model size = **FAIL**
- **Any 5-gram repeated >3 times (Hallucination)** = **FAIL**
- CI benchmark regression > 5% = **FAIL**
- Chunked streaming loses audio = **FAIL**
- Folder output structure fails to mirror input = **FAIL**

---

## Appendix A: whisper.cpp Benchmark Commands

```bash
# Tiny model, CPU
/home/noah/.local/bin/main \
    -m /home/noah/src/whisper.cpp/models/ggml-tiny.bin \
    -f audio.wav \
    --no-timestamps \
    -ng \
    -t 8

# Tiny model, GPU
/home/noah/.local/bin/main \
    -m /home/noah/src/whisper.cpp/models/ggml-tiny.bin \
    -f audio.wav \
    --no-timestamps

# Small model, CPU
/home/noah/.local/bin/main \
    -m /home/noah/src/whisper.cpp/models/ggml-small.bin \
    -f audio.wav \
    --no-timestamps \
    -ng \
    -t 8

# Small model, GPU
/home/noah/.local/bin/main \
    -m /home/noah/src/whisper.cpp/models/ggml-small.bin \
    -f audio.wav \
    --no-timestamps
```

## Appendix B: renacer Profiling Commands

```bash
# Full system trace with source correlation
renacer -s -- whisper-apr-cli transcribe --file audio.wav

# Flamegraph output
renacer --flamegraph -o profile.svg -- whisper-apr-cli transcribe --file audio.wav

# GPU kernel breakdown
renacer --gpu -- whisper-apr-cli transcribe --file audio.wav --gpu

# Compare two runs
renacer diff baseline.trace optimized.trace
```

## Appendix C: trueno-explain Profiling

```bash
# Profile SIMD utilization
trueno-explain simd whisper-apr-cli transcribe --file audio.wav

# Profile GPU kernels
trueno-explain profile whisper-apr-cli transcribe --file audio.wav --gpu

# Generate roofline model
trueno-explain roofline whisper-apr-cli transcribe --file audio.wav
```

---

*This specification follows the Toyota Way principles and Popperian falsification methodology to systematically achieve and validate 2x performance improvement over whisper.cpp.*
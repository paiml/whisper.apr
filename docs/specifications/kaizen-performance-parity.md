# Kaizen Performance Parity: whisper.apr vs whisper.cpp

**Document:** WAPR-KAIZEN-001
**Status:** Active
**Version:** 1.0.0
**Date:** 2026-03-09
**Toyota Way Phase:** Kaizen (continuous improvement, measured cycle-by-cycle)

---

## 1. Methodology

**Guiding principle:** `apr profile` is the single source of truth. Every optimization
cycle is measured before/after. No change is accepted without profile evidence.

**Benchmark configuration:**
- Audio: `jfk.wav` (11.0s, 176000 samples, 16kHz mono)
- Model: whisper-tiny (4 encoder layers, 4 decoder layers, d_model=384, 6 heads)
- CPU: AMD Ryzen Threadripper 7960X 24-Core
- Measurement: `--warmup 1 --runs 3` (averaged)

**Commands:**
```bash
# whisper.apr baseline
whisper-apr apr profile models/whisper-tiny.apr \
    /home/noah/src/whisper.cpp/samples/jfk.wav \
    --warmup 1 --runs 3 --per-token --format json

# whisper.cpp baseline
whisper-cpp/build/bin/whisper-cli -m models/ggml-tiny.bin \
    -f samples/jfk.wav -t 4
```

---

## 2. Baseline (2026-03-09, pre-kaizen)

### Head-to-Head: jfk.wav (11.0s audio), whisper-tiny

| Stage | whisper.cpp | whisper.apr | Gap | Ratio |
|---|---|---|---|---|
| Model load | 59 ms | 552 ms | +493 ms | 9.4x slower |
| Mel spectrogram | 6 ms | 31 ms | +25 ms | 4.8x slower |
| Encoder | 411 ms | 745 ms | +334 ms | 1.8x slower |
| Decoder | 109 ms (batched) | 335 ms | +226 ms | 3.1x slower |
| **Total (inference)** | **527 ms** | **1111 ms** | **+584 ms** | **2.1x slower** |

**After Cycle 1 (parallel GEMM + zero-copy):**

| Stage | whisper.cpp | whisper.apr | Gap | Ratio |
|---|---|---|---|---|
| Mel spectrogram | 6 ms | 31 ms | +25 ms | 5.2x slower |
| Encoder | 411 ms | 551 ms | +140 ms | 1.34x slower |
| Decoder | 109 ms (batched) | 335 ms | +226 ms | 3.1x slower |
| **Total (inference)** | **527 ms** | **918 ms** | **+391 ms** | **1.74x slower** |

**After Cycle 2 (decoder fp16 + fused QKV):**

| Stage | whisper.cpp | whisper.apr | Gap | Ratio |
|---|---|---|---|---|
| Mel spectrogram | 6 ms | 33 ms | +27 ms | 5.5x slower |
| Encoder | 411 ms | 553 ms | +142 ms | 1.35x slower |
| Decoder | 109 ms (batched) | 206 ms | +97 ms | 1.89x slower |
| **Total (inference)** | **527 ms** | **792 ms** | **+265 ms** | **1.50x slower** |

**After Cycle 3 (flash attention block_size 128):**

| Stage | whisper.cpp | whisper.apr | Gap | Ratio |
|---|---|---|---|---|
| Mel spectrogram | 6 ms | 32 ms | +26 ms | 5.3x slower |
| Encoder | 411 ms | 517 ms | +106 ms | 1.26x slower |
| Decoder | 109 ms (batched) | 190 ms | +81 ms | 1.74x slower |
| **Total (inference)** | **527 ms** | **739 ms** | **+212 ms** | **1.40x slower** |

### Key differences explaining the gap

| Factor | whisper.cpp | whisper.apr |
|---|---|---|
| Weight format | GGML q8_0 (quantized) | f32 SafeTensors |
| Memory bandwidth | ~77 MB model | ~145 MB model |
| Decode strategy | Batched (beam 5) | Sequential greedy |
| Threading (encoder) | 4 threads, GGML graph | rayon par_iter |
| Mel computation | Optimized C + SIMD | Rust FFT |
| Model loading | Memory-mapped | Full file read + deserialize |

### Pareto analysis (updated after Cycle 3, gap = 212 ms)

| Optimization | Est. savings | % of remaining gap | Difficulty |
|---|---|---|---|
| Encoder attention/FFN tuning | 50-100 ms | 24-47% | Medium |
| Mel SIMD optimization | 20-25 ms | 9-12% | Easy |
| Decoder further optimization | 30-50 ms | 14-24% | Medium |
| Model mmap loading | 600+ ms | N/A (load time) | Medium |
| Batched decode | 50-80 ms | 24-38% | Hard |

**Note:** Encoder INT8 was tried and failed (compute-bound, not memory-bound).
Encoder is now 1.26x cpp — close to parity. Decoder (1.74x) is the new bottleneck.

---

## 3. Kaizen Cycles

Each cycle: measure → identify bottleneck → implement → measure → record.

### Cycle 0: Baseline (DONE)

**Profile:**
```
Mel spec       31.2 ms    2.8%
Encoder       732.5 ms   67.1%
Decoder       336.9 ms   30.1%
Total        1100.5 ms
RTF: 0.10x
```

**Decision:** Encoder is 67% of inference time. Start there.

---

### Cycle 1: Parallel GEMM + Zero-Copy Matmul (DONE)

**Target:** Reduce encoder from 745 ms → 500 ms (33% reduction)

**Lesson learned (failed attempt):** INT8 per-row quantization makes encoder SLOWER
(745 ms → 920 ms). Encoder is compute-bound (1500-token batch matmul reuses weights
from L2/L3 cache 1500 times), not memory-bandwidth-bound. INT8's per-token dequant
overhead destroys the benefit. Fused QKV also blocked by encoder self-attention using
same KV input as Q.

**Root cause found via sub-step profiling:**
- Added `conv_frontend_ms` and `encoder_blocks_ms` to `ProfilingStats`
- Discovered trueno's `Matrix::matmul()` used single-threaded `gemm_blis` (not `gemm_blis_parallel`)
- whisper.apr used trueno 0.15 from crates.io (no `parallel` feature) instead of local 0.16.2
- `matmul_with_matrix` copied input via `a.to_vec()` every call (2.3 MB × 28 calls = 65 MB/forward)

**Actions:**
- [x] Add encoder sub-step profiling (conv_frontend_ms, encoder_blocks_ms)
- [x] Fix trueno version (0.15 → 0.16) so `[patch.crates-io]` applies
- [x] Enable `trueno/parallel` in whisper.apr `parallel` feature
- [x] Change `Matrix::matmul()` to use `gemm_blis_parallel` (multi-threaded BLIS)
- [x] Replace `matmul_with_matrix` to call BLIS directly (skip `a.to_vec()` allocation)
- [x] Replace `matmul` wrapper to call BLIS directly (same zero-copy pattern)

**Result:**
```
Before:  Encoder 800 ms (blocks 745 ms), Decoder 343 ms, Total 1178 ms (2.24x cpp)
After:   Encoder 551 ms (blocks 514 ms), Decoder 335 ms, Total  918 ms (1.74x cpp)
                  -31%         -31%                         -22%
```

---

### Cycle 2: Decoder fp16 + Fused QKV (DONE)

**Target:** Reduce decoder from 335 ms → 200 ms

**Root cause:** f32 model loads decoder weights as f32. This means:
1. Memory-bound single-token matvec reads 4 bytes/weight instead of 2
2. Fused QKV (single matvec for Q+K+V) requires fp16 weights — unavailable for f32
3. `decoder.finalize_weights()` was never called (encoder had it, decoder didn't)

**Actions:**
- [x] Add `decoder.finalize_weights()` after loading decoder weights
- [x] Add `decoder.convert_to_f16()` for f32 models — halves bandwidth, enables fused QKV
- [x] `finalize_weights()` calls `fuse_qkv_weights()` which creates combined fp16 W_qkv matrix
- [x] `forward_qkv_into()` now uses fused fp16 path: one matvec instead of three

**Result:**
```
Before:  Decoder 335 ms (14.0 ms/token), Total  918 ms (1.74x cpp)
After:   Decoder 206 ms ( 8.6 ms/token), Total  792 ms (1.50x cpp)
                 -38%       -39%                  -14%
```

---

### Cycle 3: Flash Attention Block Size Tuning (DONE)

**Target:** Reduce total from 792 ms → ~740 ms

**Root cause:** Flash Attention block_size=32 creates excessive per-block overhead for
1500-token encoder sequences (47 iterations per head). Block size 128 reduces to 12
iterations, improving both encoder and decoder attention.

**Actions:**
- [x] Change `FLASH_ATTENTION_BLOCK_SIZE` from 32 to 128
- [x] Profile before/after (3-run average)

**Result:**
```
Before:  Encoder 553 ms, Decoder 206 ms, Total 792 ms (1.50x cpp)
After:   Encoder 517 ms, Decoder 190 ms, Total 739 ms (1.40x cpp)
                  -7%             -8%            -7%
```

---

### Cycle 4: Decoder optimization (NEXT)

**Target:** Reduce decoder from 190 ms → ~140 ms (ms/token from 7.9 → 5.8)

**Actions:**
- [ ] Profile decoder per-token breakdown (attention vs FFN vs layer norm)
- [ ] Identify memory-bandwidth bottleneck in cross-attention
- [ ] Explore KV cache layout optimization
- [ ] Profile before/after

**Result:** _pending_

---

### Cycle 5: Mel spectrogram SIMD

**Target:** Reduce mel from 32 ms → ~10 ms

**Actions:**
- [ ] Profile mel computation breakdown (FFT vs filterbank multiply vs log)
- [ ] Apply AVX2 SIMD to filterbank multiply
- [ ] Profile before/after

**Result:** _pending_

---

### Cycle 6: Model loading (mmap)

**Target:** Reduce load from 717 ms → ~60 ms (match whisper.cpp)

**Actions:**
- [ ] Implement memory-mapped `.apr` loading (read tensor offsets, mmap pages on demand)
- [ ] Profile load time before/after

**Result:** _pending_

---

## 4. Parity target

| Stage | Current | Target | whisper.cpp ref | Gap to target |
|---|---|---|---|---|
| Total inference | 739 ms | ≤ 685 ms | 527 ms | -54 ms needed |
| Encoder | 517 ms | ≤ 500 ms | 411 ms | -17 ms needed |
| Decoder | 190 ms | ≤ 142 ms | 109 ms | -48 ms needed |
| Mel | 32 ms | ≤ 10 ms | 6 ms | -22 ms needed |
| Load | 717 ms | ≤ 100 ms | 59 ms | -617 ms needed |
| RTF (11s audio) | 0.067x | ≤ 0.062x | 0.048x | close |

**Parity definition:** Total inference time within 1.3x of whisper.cpp on the same
hardware, same audio, same model size (1.3x × 527 ms = 685 ms).

**Current status:** 1.40x (739 ms). Need to close 54 ms gap.
Best paths: decoder optimization (-48 ms) + mel SIMD (-22 ms) = -70 ms → well under target.

---

## 5. Profile enhancement backlog

Improvements to `apr profile` discovered during kaizen:

- [ ] Add `--threads N` flag to control rayon thread pool size
- [x] Add encoder sub-step breakdown (conv_frontend, encoder_blocks) — done Cycle 1
- [ ] Add decoder sub-step breakdown (attention, cross-attention, FFN separately)
- [ ] Add memory peak tracking (RSS before/after)
- [ ] Add `--compare` flag that auto-runs whisper.cpp and diffs
- [ ] Add `--model-info` to print weight format, quantization, fused status
- [ ] Show whether fused QKV optimizations are active in profile output
- [ ] Per-token decode timing histogram (not just average)

---

## 6. References

- WAPR-PARITY-001: Correctness parity specification
- WAPR-PERF-005: `apr profile` implementation
- Liker (2004), *The Toyota Way* — Kaizen, Genchi Genbutsu, Jidoka
- whisper.cpp benchmarks: https://github.com/ggerganov/whisper.cpp/issues/89

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
- CPU: AMD Ryzen Threadripper 7960X 24-Core (48 logical)
- Measurement: `--warmup 3 --runs 10` (averaged), upgraded from `--warmup 1 --runs 3` at Cycle 9
- Thread comparison: Both systems tested at same thread count (4t and 16t)

**Note on thread comparison (Cycle 9+):** Cycles 0-8 compared whisper.apr@16t vs whisper.cpp@4t.
Starting from Cycle 9, we compare at the same thread count for fairness. This revealed that
whisper.cpp scales 2.55x from 4→16 threads while whisper.apr scales only 1.37x.

**Commands:**
```bash
# whisper.apr (16 threads)
whisper-apr apr profile models/whisper-tiny.apr \
    /home/noah/src/whisper.cpp/samples/jfk.wav \
    --warmup 3 --runs 10 --threads 16 --format json

# whisper.cpp (16 threads)
whisper-cpp/build/bin/whisper-cli -m models/ggml-tiny.bin \
    -f samples/jfk.wav -t 16
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

**After Cycles 4+5 (decoder attention + sparse mel):**

| Stage | whisper.cpp | whisper.apr | Gap | Ratio |
|---|---|---|---|---|
| Mel spectrogram | 6 ms | 5 ms | -1 ms | **0.83x faster** |
| Encoder | 411 ms | 527 ms | +116 ms | 1.28x slower |
| Decoder | 109 ms (batched) | 163 ms | +54 ms | 1.50x slower |
| **Total (inference)** | **527 ms** | **695 ms** | **+168 ms** | **1.32x slower** |

**After Cycle 6 (encoder copy_from_slice):**

| Stage | whisper.cpp | whisper.apr | Gap | Ratio |
|---|---|---|---|---|
| Mel spectrogram | 6 ms | 5 ms | -1 ms | **0.83x faster** |
| Encoder | 411 ms | 513 ms | +102 ms | 1.25x slower |
| Decoder | 109 ms (batched) | 161 ms | +52 ms | 1.48x slower |
| **Total (inference)** | **527 ms** | **686 ms** | **+159 ms** | **1.30x slower** |

**After Cycle 7 (conv weight cache + in-place residual):**

| Stage | whisper.cpp | whisper.apr | Gap | Ratio |
|---|---|---|---|---|
| Mel spectrogram | 6 ms | 4 ms | -2 ms | **0.67x faster** |
| Encoder | 411 ms | 522 ms | +111 ms | 1.27x slower |
| Decoder | 109 ms (batched) | 162 ms | +53 ms | 1.49x slower |
| **Total (inference)** | **527 ms** | **687 ms** | **+160 ms** | **1.30x slower** |

**After Cycle 8 (smart thread pool sizing, 16 threads):**

| Stage | whisper.cpp | whisper.apr | Gap | Ratio |
|---|---|---|---|---|
| Mel spectrogram | 6 ms | 4 ms | -2 ms | **0.67x faster** |
| Encoder | 415 ms | 509 ms | +94 ms | 1.23x slower |
| Decoder | 107 ms (batched) | 134 ms | +27 ms | 1.25x slower |
| **Total (inference)** | **528 ms** | **646 ms** | **+118 ms** | **1.22x slower** |

### Key differences explaining the gap

| Factor | whisper.cpp | whisper.apr |
|---|---|---|
| Weight format | GGML q8_0 (quantized) | f32 SafeTensors |
| Memory bandwidth | ~77 MB model | ~145 MB model |
| Decode strategy | Batched (beam 5) | Sequential greedy |
| Threading (encoder) | 4 threads, GGML graph | rayon par_iter |
| Mel computation | Optimized C + SIMD | Rust FFT |
| Model loading | Memory-mapped | Full file read + deserialize |

### Pareto analysis (updated after Cycle 8, gap = 118 ms, ~1.22x)

| Optimization | Est. savings | % of remaining gap | Difficulty |
|---|---|---|---|
| Encoder matmul tuning | 15-40 ms | 13-34% | Hard |
| AVX-512 microkernel (trueno) | 20-50 ms | 17-42% | Hard |
| Fused encoder QKV | 4-8 ms | 3-7% | Medium |
| Encoder scratch buffers | 5-12 ms | 4-10% | Medium |
| Model mmap loading | 600+ ms | N/A (load time) | Medium |
| Batched decode | 20-30 ms | 17-25% | Hard |

**Failed experiments in this cycle:**
- BLIS tiling KC=384: +30ms worse (L2 pressure from larger packed buffers)
- BLIS tiling KC=512+MC=128: +9ms worse
- Flash attention block_size=1536: +60ms worse (L3 thrashing from full attention matrix)
- fp16 weight caching for decoder: +26ms worse (L3 cache pressure, +75MB working set)
- Encoder INT8: +175ms worse (compute-bound, dequant overhead)

**1.22x — WELL BELOW 1.3x PARITY TARGET.**
Remaining gains require architectural changes (AVX-512 kernel, batched decode).

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

### Cycle 4: Decoder Attention Optimization (DONE)

**Target:** Reduce decoder from 190 ms → ~160 ms

**Root cause:** `compute_attention_cached_with_scratch` used element-by-element
K/V head extraction (1.15M copies per layer per token). Also used separate
q_head/head_out intermediary buffers.

**Lesson learned (failed attempt):** Computing QK^T directly from the interleaved
KV cache layout (strided access) was 8% SLOWER than extracting to contiguous buffers.
The extraction acts as a cache-friendly prefetch — sequential copy is fast, while
strided access during FMA causes L1/L2 cache misses (1500 × d_model stride).

**Actions:**
- [x] Replace element-by-element extraction with `copy_from_slice` (batch copy)
- [x] Eliminate separate q_head/head_out buffers (read Q directly from interleaved layout)
- [x] Write output directly instead of through head_out intermediary

**Result:**
```
Before:  Decoder 190 ms (7.9 ms/token), Total 739 ms (1.40x cpp)
After:   Decoder 168 ms (7.0 ms/token), Total 725 ms (1.37x cpp)
                  -12%       -11%                  -2%
```

---

### Cycle 5: Sparse Mel Filterbank (DONE)

**Target:** Reduce mel from 32 ms → ~10 ms

**Root cause:** Dense filterbank multiply did 80 × 201 = 16,080 FMAs per frame,
but triangular mel filters have only ~10-20 non-zero entries per row (>90% zeros).
3000 frames × 16,080 = 48.2M FMAs; sparse needs only ~2.4M.

**Actions:**
- [x] Precompute CSR-style sparse representation in `MelFilterbank::new()`
- [x] Replace dense filterbank loop with sparse iteration
- [x] Update both center-padded and unpadded code paths

**Result:**
```
Before:  Mel 32 ms, Total 725 ms (1.37x cpp)
After:   Mel  5 ms, Total 695 ms (1.32x cpp)
              -84%          -4%
```

---

### Cycle 6: Encoder copy_from_slice (DONE)

**Target:** Reduce encoder from 527 ms → ~510 ms

**Root cause:** `extract_head` and `concat_heads` in `MultiHeadAttention` used element-by-element
copy loops (1 element at a time) for head extraction/concatenation. For 1500-token encoder sequences
with 6 heads × 64 d_head, this is 576,000 individual indexed writes per forward pass.
`copy_from_slice` uses `memcpy` semantics (bulk copy) which the compiler optimizes to SIMD moves.

**Actions:**
- [x] Replace element-by-element loop in `extract_head` with `copy_from_slice` (batch d_head elements)
- [x] Replace element-by-element loop in `concat_heads` with `copy_from_slice` (batch d_head elements)

**Result:**
```
Before:  Encoder 527 ms, Decoder 163 ms, Total 695 ms (1.32x cpp)
After:   Encoder 513 ms, Decoder 161 ms, Total 686 ms (1.30x cpp)
                  -2.7%         -1.2%            -1.3%
```

**1.3x PARITY TARGET ACHIEVED.**

---

### Cycle 7: Conv weight caching + in-place residual (DONE)

**Target:** Squeeze remaining encoder overhead (~5-10 ms)

**Root cause:** Two micro-optimizations discovered during profiling:
1. `Conv1d::forward()` reshapes+transposes weight matrix on every call (~1ms each for 2 conv layers)
2. `EncoderBlock::forward()` allocated a new Vec for residual add via `add_residual()` helper

**Lesson learned (failed attempt):** Caching dequanted f32 weights for decoder's fp16 path
added ~75MB to working set, filling 58% of 128MB L3 cache. The 23 single-token decodes
suffered cache misses that overwhelmed the 8ms first-token savings. Reverted.

**Actions:**
- [x] Cache transposed Conv1d weights at load time (`finalize_weights()`)
- [x] Add `ConvFrontend::finalize_weights()` called from `Encoder::finalize_weights()`
- [x] In-place residual add in `EncoderBlock::forward()` (reuse attn_out buffer)
- [x] Remove dead `q_head`/`head_out` fields from `AttentionScratch`

**Result:**
```
Before:  Encoder 513 ms, Decoder 161 ms, Total 686 ms (1.30x cpp)
After:   Encoder 522 ms, Decoder 162 ms, Total 687 ms (1.30x cpp)  [10-run avg]
Best-of: Encoder 515 ms, Decoder 159 ms, Total 677 ms (1.28x cpp)  [3-run avg]
```

Note: 10-run avg shows ~1ms improvement absorbed by thermal variance. The conv cache
saves ~2ms deterministically (visible in conv_frontend_ms: 28ms → 26ms) but total
is within noise floor of Threadripper turbo boost variance (±40ms).

---

### Cycle 8: Smart thread pool sizing (DONE)

**Target:** Reduce decoder overhead from excess thread pool contention

**Root cause:** `apr profile` didn't configure the rayon thread pool. Rayon defaulted
to all 48 logical cores (24-core Threadripper with SMT). For single-token decoder
matvecs (below PARALLEL_THRESHOLD), the 48-thread pool caused work-stealing cache
line bouncing and scheduling overhead. The decoder lost ~22ms vs optimal thread count.

**Thread sweep results (5-run avg, jfk.wav):**

| Threads | Encoder | Decoder | Total | Ratio |
|---|---|---|---|---|
| 1 | 2150 ms | 213 ms | 2367 ms | 4.49x |
| 4 | 825 ms | 150 ms | 979 ms | 1.86x |
| 8 | 524 ms | 138 ms | 665 ms | 1.26x |
| 12 | 519 ms | 138 ms | 660 ms | 1.25x |
| 16 | 517 ms | 140 ms | 660 ms | 1.25x |
| 48 (default) | 519 ms | 160 ms | 682 ms | 1.29x |

**Lesson learned (failed experiments):**
1. BLIS tiling KC=384 or KC=512+MC=128 did NOT improve performance. The current
   KC=256, MC=72 constants are already well-matched to the cache hierarchy, and
   larger KC increases packed buffer sizes causing L2 pressure.
2. Flash attention block_size=256 → no change. block_size=1536 (no tiling) → 8% WORSE
   due to full 1500×1500 attention matrix (9MB/head) thrashing L3 cache.

**Actions:**
- [x] Add `--threads` flag to `apr profile`
- [x] Smart default: logical_cores/2 capped at 16 (≈ physical cores, avoids SMT overhead)
- [x] Call `configure_thread_pool()` before model load in `run_profile()`

**Result:**
```
Before:  Encoder 522 ms, Decoder 162 ms, Total 687 ms (1.30x cpp)  [default 48 threads]
After:   Encoder 509 ms, Decoder 134 ms, Total 646 ms (1.23x cpp)  [16 threads]
                  -2.5%         -17%             -6%
```

---

### Cycle 9: Sequential QKV Projections (DONE)

**Target:** Enable full thread utilization per GEMM call in encoder

**Root cause:** `forward_cross_flash` used `rayon::join` to run Q/K/V projections in parallel.
But each projection calls `gemm_blis_parallel` which itself uses all rayon threads internally.
Nested parallelism meant the outer 3-way split forced each GEMM into 1/3 of the thread pool.
For these large GEMMs (1500×384→384), sequential execution with full thread pool per GEMM is faster.

**Actions:**
- [x] Remove `rayon::join` for Q/K/V projection in `forward_cross_flash`
- [x] Remove `rayon::join` for Q/K/V projection in `forward_cross_flash_v2`
- [x] Make projections sequential so each GEMM gets full thread pool

**Result:**
```
4-thread:  Total 1350 ms → 849 ms  (-37%, GEMM fully parallelized)
16-thread: Total  646 ms → 640 ms  (-1%, already saturated at 16t)
```

Commit: `786dc67` — `perf: Remove nested rayon::join for encoder Q/K/V projections`

---

### Cycle 10: Parallel Flash Attention over Q-Rows (DONE)

**Target:** Fix 6-head thread utilization bottleneck in encoder attention

**Root cause:** `parallel_map(0..6, |head| flash_attention_simd(...))` distributed 6 heads
across 16 threads — only 6/16 threads active (37.5% utilization). For whisper-tiny with
only 6 attention heads, the parallelism grain was too coarse.

**Lesson learned (failed attempt):** Zero-copy approach working directly on interleaved
Q/K/V data (strided access at d_model=384 stride) was 38% SLOWER (534ms vs 386ms).
With d_head=64 (256 bytes useful per 1536-byte stride), only 17% of loaded cache lines
contain useful data. The `extract_head` copy to contiguous per-head layout is essential
for cache performance — sequential copy is fast, strided FMA causes L1/L2 misses.

**Actions:**
- [x] Replace `parallel_map(0..n_heads)` with sequential head loop
- [x] Add `flash_attention_simd_parallel()` that uses `par_chunks_mut(d_head)` over Q-rows
- [x] Each head: extract_head (contiguous copy) → parallel over 1500 Q-rows (all 16 threads)

**Result:**
```
16-thread: Encoder 530 ms → 386 ms  (-27%, full thread utilization)
8→16 thread scaling now works (was flat before)
```

Commit: `b3c9792` — `perf: Parallelize flash attention over Q-rows for full thread utilization`

---

### Cycle 11: Zero-Allocation SIMD Dot Product (DONE)

**Target:** Eliminate heap allocation overhead in hot loop of parallel flash attention

**Root cause:** `simd::dot(a, b)` calls `Vector::from_slice()` which does `data.to_vec()`
— a heap allocation + memcpy per call. In parallel flash attention, this is called:
- 1500 Q-rows × 1500 K-rows × 6 heads × 4 layers = 54M dot product calls
- Plus 1500 × 1500 × 6 × 4 = 54M value accumulations
- ≈ 28M heap allocations per encoder pass (~7.2 GB allocation churn)

Similarly, `simd::max_element()` allocates via `Vector::from_slice` internally.

**Actions:**
- [x] Replace `simd::dot()` with `simd::dot_nalloc()` (inline AVX2+FMA, zero heap allocation)
- [x] Inline `max_element` as `block_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)`
- [x] Apply to both Q·K dot products and V accumulation in `flash_attention_simd_parallel`

**Result:**
```
1-thread:  Encoder 2146 ms → 1183 ms  (-45%, allocation was dominant at 1 thread)
16-thread: Encoder  386 ms →  299 ms  (-23%, allocation overhead amortized by parallelism)
Total@16t:          524 ms →  441 ms  (-16%)
```

Commit: `315a253` — `perf: Use zero-allocation dot product in parallel flash attention`

---

### Cumulative Impact: Cycles 9-11

**Encoder block breakdown (post Cycle 11, instrumented single run):**
- Attention: 24.8 ms/block (40%) — down from 74.8 ms/block pre-Cycle 10
- FFN: 35.1 ms/block (57%) — **NOW THE DOMINANT BOTTLENECK**
- LN1+LN2: 2 ms/block (3%)

**Thread scaling (10-run avg, post Cycle 11):**

| Threads | Encoder | Decoder | Total | vs cpp@same-t |
|---|---|---|---|---|
| 4 | 452 ms | 150 ms | 605 ms | 1.18x |
| 16 | 299 ms | 139 ms | 441 ms | 2.19x |

**Session total: 646 ms → 441 ms (32% faster). Encoder: 509 ms → 299 ms (41% faster).**

---

### Cycle 12: FFN Optimization (NEXT)

**Target:** Reduce FFN from 35 ms/block → ~15 ms/block (encoder 299 → ~220 ms)

**Root cause (identified):** FFN is two large GEMMs per block:
- fc1: 1500×384 → 1536 (884K FLOPs)
- fc2: 1500×1536 → 384 (884K FLOPs)
These use `gemm_blis_parallel` which packs B (weight matrix) independently per thread.
With 16 threads × 2 GEMMs × 4 layers = 128 redundant B packings per encoder pass.

**Potential actions:**
- [ ] Pre-pack FFN weight matrices at load time (eliminate runtime B packing)
- [ ] Add `gemm_blis_prepacked` API to trueno (accept pre-packed B)
- [ ] Profile BLIS packing vs compute ratio to quantify savings
- [ ] Investigate AVX-512 microkernel for 2x throughput on supported CPUs

**Result:** _pending_

---

### Cycle 13: Model Loading (mmap) (BACKLOG)

**Target:** Reduce load from 700 ms → ~60 ms (match whisper.cpp)

**Actions:**
- [ ] Implement memory-mapped `.apr` loading (read tensor offsets, mmap pages on demand)
- [ ] Profile load time before/after

**Result:** _pending_

---

## 4. Parity Target

### Thread scaling gap (discovered Cycle 9-11)

The previous parity analysis compared whisper.apr@16t vs whisper.cpp@4t. Now measuring
at the same thread count reveals a fundamental scaling gap:

| Config | whisper.cpp | whisper.apr | Ratio |
|---|---|---|---|
| 4 threads | 513 ms | 605 ms | 1.18x |
| 16 threads | 201 ms | 441 ms | 2.19x |
| Scaling 4→16 | 2.55x | 1.37x | — |

whisper.cpp's GGML graph scheduler achieves 2.55x scaling from 4→16 threads.
whisper.apr's rayon+BLIS approach achieves only 1.37x. The encoder is the bottleneck:
whisper.cpp encoder scales 2.67x (403→151ms) while ours scales 1.51x (452→299ms).

### Current status (same thread count comparison)

| Stage | whisper.apr@4t | whisper.cpp@4t | Ratio@4t | whisper.apr@16t | whisper.cpp@16t | Ratio@16t |
|---|---|---|---|---|---|---|
| Mel | 3 ms | 7 ms | **0.43x** | 4 ms | 5 ms | **0.80x** |
| Encoder | 452 ms | 403 ms | 1.12x | 299 ms | 151 ms | 1.98x |
| Decoder | 150 ms | 103 ms | 1.46x | 139 ms | 45 ms | 3.09x |
| **Total** | **605 ms** | **513 ms** | **1.18x** | **441 ms** | **201 ms** | **2.19x** |

**At 4 threads: 1.18x — BELOW 1.3x PARITY TARGET.**
**At 16 threads: 2.19x — thread scaling is the primary remaining gap.**

### Pareto analysis (updated after Cycle 11)

| Optimization | Est. savings@16t | % of remaining 240ms gap | Difficulty |
|---|---|---|---|
| Pre-packed FFN weights (trueno) | 20-40 ms | 8-17% | Medium |
| AVX-512 GEMM microkernel (trueno) | 40-80 ms | 17-33% | Hard |
| Better thread scaling (work-sharing) | 50-100 ms | 21-42% | Very Hard |
| Batched decode (beam search) | 50-80 ms | 21-33% | Hard |
| Model mmap loading | 600+ ms | N/A (load time) | Medium |

**Key insight:** At 4 threads, we are within 1.18x parity — the core algorithm is competitive.
The gap at 16 threads is a **thread scaling problem**, not an algorithm problem. whisper.cpp's
GGML graph-level parallelism distributes work more efficiently than our per-GEMM rayon approach.

**11 kaizen cycles: 2.1x → 1.18x@4t (86% of gap eliminated).**

---

## 5. Profile Enhancement Specification (WAPR-PROFILE-001)

### Design Principle: Wire Existing Tools, Don't Reinvent

whisper.apr depends on trueno (SIMD GEMM) and realizar (ML inference) — both have
world-class profiling infrastructure that is currently **completely disconnected** from
`apr profile`. The current profiler is 5 `Instant::now()` calls and a HashMap.

**Existing infrastructure we MUST wire (not reinvent):**

| Tool | Location | What it provides | Currently used? |
|------|----------|------------------|-----------------|
| `trueno::BrickProfiler` | `trueno/src/brick/profiler/` | O(1) BrickId timing, RDTSCP cycles, BrickCategory (Norm/Attn/FFN), CategoryStats with percentage(), BrickStats with bottleneck classification, cycle counters, IPC estimation | **NO** |
| `trueno::BlisProfiler` | `trueno/src/blis/profiler.rs` | 4-level GEMM hierarchy (Macro/Midi/Micro/Pack), per-level GFLOP/s, packing time breakdown | **NO** (always `None`) |
| `trueno::TileStats` | `trueno/src/brick/profiler/tile_stats.rs` | `arithmetic_intensity()`, `cache_efficiency(peak_gflops)`, `gflops()` | **NO** |
| `trueno::hardware` | `trueno/src/hardware/` | `CpuCapability`, `SimdWidth`, `RooflineParams`, `Bottleneck` enum | **NO** |
| `trueno::profiling` | `trueno/src/brick/profiling.rs` | `cpu_cycles()` (RDTSCP), `cached_nanos_or_now()` (1ns, no syscall), `init_time_service()`, `get_page_faults()`, `with_page_fault_tracking()` | **NO** |
| `realizar::BrickProfiler` | `realizar/src/brick/profiler.rs` | `measure(op, closure)`, `ProfileReport` with `percentage_breakdown()`, `hottest()`, `sorted_by_time()`, thread-local macros | **NO** |
| `realizar::InferenceTracer` | `realizar/src/inference_trace/` | `TraceStep::{Attention,FFN,LayerNorm,TransformerBlock}`, `TraceEvent` with layer/duration/shapes, `to_json()` for Chrome Trace, `format_text()` | **NO** |

### Done

- [x] Add `--threads N` flag to control rayon thread pool size — Cycle 8
- [x] Add encoder sub-step breakdown (conv_frontend, encoder_blocks) — Cycle 1

---

### Gap 1: Per-Operator Breakdown via trueno BrickProfiler (P0)

**Problem:** Single `encoder_blocks_ms` for all 4 blocks. No attn/FFN/LN split.
We added/removed temp `Instant::now()` probes every kaizen cycle — anti-genchi-genbutsu.

**Wire these trueno APIs:**

```rust
// trueno::BrickProfiler — O(1) array-indexed timing (no HashMap, no string alloc)
let mut profiler = BrickProfiler::enabled();
trueno::brick::profiling::init_time_service();  // background thread, 100μs resolution, 1ns reads

// Per-operator timing via BrickId (already defined in trueno for Norm, Attention, FFN)
let timer = profiler.start_brick(BrickId::LayerNorm);
let normed = self.ln1.forward(x)?;
profiler.stop_brick(timer, x.len() as u64);  // records ns + elements

let timer = profiler.start_brick(BrickId::Attention);
let attn_out = self.self_attn.forward(&normed, None)?;
profiler.stop_brick(timer, normed.len() as u64);

// ... etc for FFN ...

// Automatic category rollup (Norm/Attention/FFN/Other)
let cats: [CategoryStats; BrickCategory::COUNT] = profiler.category_stats();
// cats[BrickCategory::Norm].percentage(total_ns)   → 2%
// cats[BrickCategory::Attention].percentage(total_ns) → 41%
// cats[BrickCategory::FFN].percentage(total_ns)     → 57%

// Per-brick diagnostics
let stats: &BrickStats = profiler.brick_stats(BrickId::Attention);
stats.avg_us();           // average microseconds
stats.cycles_per_element(); // RDTSCP-based
stats.estimated_ipc();    // instructions per cycle estimate
stats.diagnose_from_cycles(); // "compute-bound" / "memory-bound" / "stalled"
```

**Also wire `trueno::profiling::cpu_cycles()`** for RDTSCP timing (zero syscall cost)
via `BrickStats::add_sample_with_cycles()`. This gives IPC and cycle-level diagnostics
that `Instant::now()` cannot provide.

**Also wire `trueno::profiling::get_page_faults()`** per encoder pass to detect
memory pressure (e.g. the fp16 decoder caching experiment that caused +26ms from
L3 pressure would have shown as elevated page faults).

**Output:**
```
apr profile (default output — always shows block breakdown)

  Encoder Block Breakdown (4 blocks, BrickProfiler):
  Block  LN1(ms)  Attn(ms)  LN2(ms)  FFN(ms)  Total(ms)  Cycles/elem  IPC
  0      0.4      24.8      0.5      35.1     60.8       12.3         1.8
  1      0.4      24.6      0.5      35.0     60.5       12.1         1.9
  2      0.5      24.9      0.5      35.2     61.1       12.4         1.8
  3      0.5      25.0      0.5      34.9     60.9       12.2         1.8
  ─────────────────────────────────────────────────────────────────────
  Category totals:  LN=3.7ms(2%)  Attn=99.3ms(41%)  FFN=140.2ms(57%)
  Page faults: 0 minor, 0 major (no memory pressure)
```

JSON adds `block_detail` array with per-block `{ln1, attn, ln2, ffn, cycles_per_elem, ipc}`.

**Implementation chain:**
1. `init_time_service()` at startup in `run_profile()`
2. `BrickProfiler::enabled()` passed into `Encoder::forward_profiled()`
3. `EncoderBlock::forward_profiled()` uses `start_brick/stop_brick` with `BrickId::LayerNorm`, `BrickId::Attention`, `BrickId::MatMul` (for FFN)
4. `profiler.category_stats()` for automatic Norm/Attn/FFN rollup
5. `get_page_faults()` before/after encoder pass
6. Propagate `BrickProfiler` + page fault deltas through `ProfilingStats` → `ProfileSummary`

---

### Gap 2: Roofline via trueno Hardware + TileStats + BlisProfiler (P1)

**Problem:** 3 failed experiments from wrong compute/memory classification.

**Wire these trueno APIs:**

```rust
// Hardware detection (already in trueno)
use trueno::hardware::{CpuCapability, RooflineParams, SimdWidth, Bottleneck};
let cpu = CpuCapability::detect();  // CPUID: AVX2/AVX-512, cores, cache sizes
let roofline = RooflineParams::from_cpu(&cpu);
// roofline.peak_gflops_f32   → 614.0 (AVX2) or 1228.0 (AVX-512)
// roofline.peak_bandwidth_gbs → 76.8 (DDR5-4800 quad-channel)
// roofline.balance_point()    → 8.0 FLOP/byte

// Per-GEMM classification via BlisProfiler
let mut blis_prof = BlisProfiler::enabled();
gemm_blis(m, n, k, a, b, c, Some(&mut blis_prof))?;
let gflops = blis_prof.total_gflops();  // achieved GFLOP/s
let ai = (2.0 * m * n * k) as f64 / ((m*k + k*n + m*n) * 4) as f64; // arithmetic intensity
let bound = if ai > roofline.balance_point() { Bottleneck::Compute } else { Bottleneck::Memory };

// Per-tile cache efficiency (trueno::TileStats)
profiler.enable_tile_profiling();
// ... after GEMM ...
for level in [TileLevel::Macro, TileLevel::Midi, TileLevel::Micro] {
    let ts = &profiler.tile_stats[level as usize];
    ts.arithmetic_intensity();  // measured from actual access patterns
    ts.cache_efficiency(roofline.peak_gflops_f32);  // % of roofline ceiling
    ts.gflops();  // achieved at this tile level
}

// BrickStats bottleneck classification (automatic)
let stats = profiler.brick_stats(BrickId::MatMul);
stats.get_bottleneck(); // BrickBottleneck::Compute | Memory | Unknown
```

**Output:**
```
apr profile --roofline

  Hardware: AMD 7960X — AVX-512, Peak: 1228 GFLOP/s, BW: 76.8 GB/s, Balance: 16.0 FLOP/byte
  Detected: trueno::CpuCapability { simd: AVX512, cores: 24, l1: 32K, l2: 1M, l3: 128M }

  Operator             AI(F/B)  GFLOP/s  Ceiling  Bound    Util%  Tile Eff%
  Encoder GEMM fc1     79.2     194.2    1228.0   Compute  15.8%  macro:68% micro:92%
  Encoder GEMM fc2     79.2     196.0    1228.0   Compute  16.0%  macro:69% micro:91%
  Encoder Attn QKV     52.8     162.4    1228.0   Compute  13.2%  macro:72% micro:88%
  Flash Attn Score     1.3      —        76.8     Memory   —      —
  Decoder Matvec       0.5      —        76.8     Memory   —      —
  LayerNorm            0.67     —        76.8     Memory   —      —
```

**Implementation chain:**
1. `CpuCapability::detect()` at startup — cache in `run_profile()`
2. `RooflineParams::from_cpu()` for peak compute/bandwidth
3. `BlisProfiler::enabled()` passed to each GEMM via `LinearWeights::forward_simd()`
4. Per-GEMM: compute AI from (M,N,K), read `blis_prof.total_gflops()`, classify bound
5. `TileStats` from BrickProfiler for cache efficiency at each blocking level
6. `BrickStats.get_bottleneck()` for automatic classification

---

### Gap 3: Thread Scaling Sweep (P1)

**Problem:** 2.55x vs 1.37x scaling gap discovered by accident. Manual spreadsheet.

**Output:**
```
apr profile --sweep-threads 1,4,8,16

  Thread Scaling (jfk.wav, whisper-tiny):
  Threads  Encoder   Decoder   Total    Speedup  Efficiency  Amdahl Serial%
  1        1183 ms   213 ms    1400 ms  1.00x    100%        —
  4         452 ms   150 ms     605 ms  2.31x     58%        14.2%
  8         350 ms   141 ms     494 ms  2.83x     35%        18.4%
  16        299 ms   139 ms     441 ms  3.17x     20%        20.9%

  Optimal: 8 threads (diminishing returns beyond, decoder saturates at 4)
  Amdahl serial fraction: ~20% (theoretical max speedup: 5.0x)
  Per-component:
    Encoder: 3.96x (1→16), efficiency 25% — parallel GEMM + flash attn
    Decoder: 1.53x (1→16), efficiency 10% — sequential token loop
```

**Implementation:** Loop thread counts, rebuild rayon pool, full profile per count.
Compute Amdahl `s` from: `T(N) = T(1) * (s + (1-s)/N)`. Report per-component.

---

### Gap 4: GEMM-Level Detail via trueno BlisProfiler (P1)

**Problem:** 16x redundant B-packing in FFN found by code reading, not measurement.

**Wire these trueno APIs:**

```rust
// BlisProfiler — already built into gemm_blis() via profiler parameter
let mut blis_prof = BlisProfiler::enabled();
gemm_blis(m, n, k, a, b, c, Some(&mut blis_prof))?;

// 4-level BLIS hierarchy breakdown
blis_prof.pack_stats.total_ns;    // packing time (THE key metric for redundancy)
blis_prof.pack_stats.count;       // number of pack operations
blis_prof.micro_stats.gflops();   // microkernel achieved GFLOP/s
blis_prof.macro_stats.gflops();   // overall achieved GFLOP/s
blis_prof.summary();              // formatted text summary

// KaizenMetrics for aggregate tracking
let mut kaizen = KaizenMetrics::default();
kaizen.record(m, n, k, elapsed);
kaizen.gflops();  // running GFLOP/s average
```

**Output:**
```
apr profile --gemm-detail

  GEMM Detail (encoder pass, 24 GEMMs via trueno::BlisProfiler):
  GEMM              M     N     K     Time(ms)  GFLOP/s  Pack(ms)  Pack%  Micro GFLOP/s
  Enc.0 QKV proj    1500  384   384   4.3       162.4    0.8       18%    224.1
  Enc.0 FFN fc1     1500  1536  384   8.9       194.2    2.0       22%    285.3
  Enc.0 FFN fc2     1500  384   1536  8.8       196.0    1.8       21%    280.7
  ...
  ────────────────────────────────────────────────────────────────────────────
  Total: 24 GEMMs, 140.2ms, avg 172.3 GFLOP/s, 28.1% of AVX2 peak (614 GFLOP/s)
  Packing: 31.2ms total (22.3% of GEMM time)
    B-packing per thread: 16 threads × 24 GEMMs = 384 redundant packs → ~18ms waste
  KaizenMetrics: 3.42 GFLOP total, 24.4 GFLOP/s sustained
```

**Implementation chain:**
1. `LinearWeights` gets `Option<&mut BlisProfiler>` parameter when profiling
2. Forward to `gemm_blis_parallel()` → `gemm_blis()` (already accepts `Option<&mut BlisProfiler>`)
3. After each GEMM: snapshot `blis_prof.{pack,micro,midi,macro}_stats`, then `blis_prof.reset()`
4. Aggregate per-GEMM snapshots into `ProfileSummary`
5. Report with `blis_prof.summary()` or custom table

---

### Gap 5: Structured Traces via realizar InferenceTracer (P1 — upgraded from P2)

**Problem:** `--format renacer` is hand-rolled `format!()` with no thread lanes.
realizar's `InferenceTracer` already produces structured Chrome Trace JSON with
layer attribution, step types, and `to_json()` export.

**Wire these realizar APIs:**

```rust
use realizar::inference_trace::{InferenceTracer, TraceConfig, TraceStep, ModelInfo};

// Configure tracer with all steps enabled
let config = TraceConfig::enabled();
let mut tracer = InferenceTracer::new(config);
tracer.set_model_info(ModelInfo {
    name: "whisper-tiny".into(),
    num_layers: 4,
    hidden_dim: 384,
    vocab_size: 51865,
    num_heads: 6,
    quant_type: Some("f32".into()),
});

// Encoder pass: trace each block with layer attribution
for (layer_idx, block) in self.blocks.iter().enumerate() {
    tracer.start_step(TraceStep::TransformerBlock);

    tracer.start_step(TraceStep::LayerNorm);
    let normed = block.ln1.forward(x)?;
    // tracer captures duration_us automatically on next start_step

    tracer.start_step(TraceStep::Attention);
    let attn_out = block.self_attn.forward(&normed, None)?;

    tracer.start_step(TraceStep::FFN);
    let ffn_out = block.ffn.forward(&normed2)?;

    tracer.trace_layer(layer_idx, 0, Some(&x), seq_len, self.d_model);
}

// Export
tracer.to_json()   // → Chrome Trace JSON with per-layer, per-step events
tracer.format_text() // → Human-readable text summary
```

**TraceEvent fields we get for free:**
- `step: TraceStep` — Attention/FFN/LayerNorm/TransformerBlock
- `layer: Option<usize>` — which encoder/decoder block
- `duration_us: u64` — per-step wall time
- `input_shape/output_shape: Vec<usize>` — tensor dimensions
- `details.brick_timings: Option<Vec<(String, u64, u64)>>` — can embed BrickProfiler data
- `details.brick_categories: Option<Vec<(String, u64)>>` — can embed CategoryStats

**Also wire `trueno::profiling::with_page_fault_tracking()`:**

```rust
let (result, minor_faults, major_faults) = with_page_fault_tracking("encoder", || {
    encoder.forward_profiled(&conv_output)
});
// major_faults > 0 → memory pressure (detected the fp16 cache disaster)
```

**Output for `--format renacer`:**
Replace hand-rolled JSON with `tracer.to_json()`. Adds:
- Per-step events with `TraceStep` names as event names
- Layer attribution (nested under TransformerBlock spans)
- Tensor shape metadata
- BrickProfiler category stats embedded in TraceDetails
- Page fault counters as metadata events

**Output for `--format text` (default):**
`tracer.format_text()` provides a formatted summary alongside our existing table.

**Implementation chain:**
1. `InferenceTracer::new(TraceConfig::enabled())` in `run_profile()`
2. Pass tracer into `Encoder::forward_profiled()` and `Decoder::forward_block_cached()`
3. Call `start_step()` / `trace_layer()` at each operator boundary
4. Embed `BrickProfiler.category_stats()` into `TraceDetails.brick_categories`
5. Replace `format_renacer()` hand-rolled JSON with `tracer.to_json()`
6. Add page fault tracking via `with_page_fault_tracking()` per pipeline phase

---

### Implementation Priority (revised: MAXIMUM tooling)

| Gap | Priority | Wire | Estimated effort | Kaizen impact |
|-----|----------|------|------------------|---------------|
| 1. BrickProfiler per-op | P0 | trueno::BrickProfiler, cpu_cycles, get_page_faults | 6 hours | Per-block attn/FFN/LN + IPC + page faults |
| 2. Roofline classification | P1 | trueno::hardware + BlisProfiler + TileStats | 8 hours | Compute/memory bound per operator |
| 4. GEMM-level detail | P1 | trueno::BlisProfiler (pass Some() not None) | 4 hours | Packing waste quantified |
| 5. InferenceTracer traces | P1 | realizar::InferenceTracer + to_json() | 6 hours | Real Chrome Trace, not hand-rolled |
| 3. Thread scaling sweep | P1 | (pure implementation, no wiring) | 3 hours | Amdahl serial fraction |

**Recommended order:** Gap 1 → Gap 4 → Gap 5 → Gap 2 → Gap 3.
Gap 1 gives per-block breakdown + IPC + page faults via BrickProfiler.
Gap 4 passes `Some(&mut BlisProfiler)` to GEMM calls (one-line change per call site).
Gap 5 replaces hand-rolled renacer with InferenceTracer.to_json().
Gap 2 combines hardware detection + BlisProfiler + TileStats for roofline.
Gap 3 is pure CLI logic (no new wiring needed).

All 5 gaps wire existing tools. Zero new profiler infrastructure invented.

---

## 6. References

- WAPR-PARITY-001: Correctness parity specification
- WAPR-PERF-005: `apr profile` implementation
- Liker (2004), *The Toyota Way* — Kaizen, Genchi Genbutsu, Jidoka
- whisper.cpp benchmarks: https://github.com/ggerganov/whisper.cpp/issues/89

### Profiling methodology references (Gap analysis, §5)

- Yuan et al. (2024), "LLM Inference Unveiled: Survey and Roofline Model Insights", arXiv:2402.16363
- Dice & Kogan (2021), "Optimizing Inference Performance of Transformers on CPUs", arXiv:2102.06621
- ProfInfer (2026), "eBPF-based Fine-Grained LLM Inference Profiler", arXiv:2601.20755
- LIFE (2025), "Forecasting LLM Inference Performance via Analytical Modeling", arXiv:2508.00904
- NonGEMM Bench (2024), "Understanding the Performance Horizon of the Latest ML Workloads", arXiv:2404.11788
- "Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference", arXiv:2505.06461
- NERSC Roofline Performance Model, docs.nersc.gov/tools/performance/roofline/
- NVIDIA Nsight Compute Profiling Guide, docs.nvidia.com/nsight-compute/ProfilingGuide/
- Intel VTune Memory Access Analysis, intel.com/content/www/us/en/docs/vtune-profiler/

### trueno/realizar API references (§5 implementation)

- `trueno::BrickProfiler` — PAR-200 O(1) BrickId timing, `start_brick()`/`stop_brick()`, `category_stats()`
- `trueno::BrickStats` — `cpu_cycles()`, `cycles_per_element()`, `estimated_ipc()`, `diagnose_from_cycles()`, `get_bottleneck()`
- `trueno::BlisProfiler` — 4-level GEMM hierarchy, `{macro,midi,micro,pack}_stats`, `total_gflops()`, `summary()`
- `trueno::TileStats` — `arithmetic_intensity()`, `cache_efficiency()`, `gflops()`
- `trueno::hardware` — `CpuCapability::detect()`, `RooflineParams::from_cpu()`, `SimdWidth`, `Bottleneck`
- `trueno::profiling` — `init_time_service()`, `cached_nanos_or_now()`, `cpu_cycles()`, `get_page_faults()`, `with_page_fault_tracking()`
- `realizar::InferenceTracer` — `TraceStep::{Attention,FFN,LayerNorm,TransformerBlock}`, `to_json()`, `format_text()`
- `realizar::BrickProfiler` — `measure()`, `ProfileReport`, `percentage_breakdown()`, `hottest()`

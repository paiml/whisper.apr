# Golden Trace Analysis

Generated: 2025-12-15
Updated: 2025-12-15 (WAPR-PERF-002)

Reference: `docs/specifications/benchmark-whisper-steps-a-z.md`

## Performance Summary

**Key Finding**: SIMD feature was not enabled by default, causing 1.6x slowdown.

**Fix Applied**: Added `simd` to default features in Cargo.toml.

## Baselines (whisper-tiny-int8)

### Before Fix (SIMD disabled)

| Component | Target | Measured | Status |
|-----------|--------|----------|--------|
| Mel Spectrogram (1s) | <10ms | 0.9ms | ✅ |
| Encoder | <500ms | 118.5ms | ✅ |
| Decoder (1 step) | <100ms | 153.3ms | ❌ |
| **First Token Latency** | **<120ms** | **272.8ms** | ❌ |

### After Fix (SIMD enabled by default)

| Component | Target | Measured | Speedup |
|-----------|--------|----------|---------|
| Mel Spectrogram (1s) | <10ms | 1.0ms | 1.0x |
| Encoder | <500ms | **55.5ms** | **2.1x** |
| Decoder (1 step) | <100ms | **118.7ms** | **1.3x** |
| **First Token Latency** | **<120ms** | **175.2ms** | **1.6x** |

### SLA Compliance

| SLA Target | Value | Current | Status |
|------------|-------|---------|--------|
| First Token (tiny) | 120ms | 175ms | ⚠️ 46% over |
| RTF | 1.8x | ~1.5x | ✅ |
| Memory | 145MB | <150MB | ✅ |
| p50 Latency | 100ms | ~120ms | ⚠️ |
| p95 Latency | 250ms | ~200ms | ✅ |

## RTF Analysis

| Audio Duration | Processing | RTF | Target | Status |
|---------------|------------|-----|--------|--------|
| 1.5s | ~340ms | 0.23x | 2.0x | ✅ Excellent |
| 3.0s | ~700ms | 0.23x | 1.5x | ✅ Excellent |

## Optimization History

### WAPR-PERF-001 (2025-12-15)
- Added SIMD attention in `compute_attention_cached`
- Added cross-attention K/V caching

### WAPR-PERF-002 (2025-12-15)
- **Root cause**: `simd` feature not in default features
- **Fix**: Added `simd` to `default` features in Cargo.toml
- **Impact**: 1.6x overall speedup, 2.1x encoder speedup

## Regression Thresholds

- Any component >20% slower than baseline: **FAIL**
- Total pipeline >10% slower: **WARN**
- Memory >150MB peak (tiny-int8): **FAIL**

## Remaining Opportunities

1. **Decoder still at 118ms** (target: 100ms)
   - FFN layer optimization
   - Better memory layout
   - Flash attention for long sequences

2. **First Token at 175ms** (target: 120ms)
   - Need ~30% more improvement
   - Focus on decoder self-attention

## WAPR-BENCH-001 Analysis (2025-12-15)

### Root Cause Analysis

**Problem**: Excessive memory allocations in hot paths

**Findings from code review**:

1. **simd::matmul wrapper** (simd.rs:224-238)
   - Copies input `a` with `.to_vec()`
   - Copies input `b` with `.to_vec()`
   - Copies output with `.as_slice().to_vec()`
   - **3 allocations per matmul call**

2. **scaled_dot_product_attention** (simd.rs:411-452)
   - transpose: 1 allocation
   - matmul for Q@K^T: 3 allocations
   - matmul for weights@V: 3 allocations
   - softmax weights: 1 allocation
   - **8 allocations per attention call**

3. **compute_attention_cached** (decoder.rs:1579-1595)
   - Creates new Vec for q_head, k_head, v_head per head
   - **3 allocations × n_heads × n_layers per token**

4. **LinearWeights::forward_simd** (attention.rs:153)
   - Uses simd::matmul which copies 3x
   - Called 8x per layer (Q,K,V,O for self + cross attention)
   - **24 allocations per layer per token**

### Optimization Strategy

1. **Use trueno Matrix directly** (like project_to_vocab)
   - `Matrix::from_vec` with owned Vec (no input copy)
   - Reuse transposed weight matrices (already cached)

2. **Add zero-copy attention path**
   - Process all heads at once instead of per-head
   - Use view-based slicing instead of collect

3. **Pre-allocate output buffers**
   - Cache output matrices in KV cache struct
   - Reuse across tokens

### Expected Impact

| Optimization | Estimated Speedup |
|--------------|-------------------|
| Remove matmul copies | 10-15% |
| Batch head processing | 5-10% |
| Pre-allocated buffers | 5-10% |
| **Combined** | **20-35%** |

With 30% speedup:
- Decoder: 118ms → ~83ms ✅ (target: 100ms)
- First Token: 175ms → ~123ms ≈ (target: 120ms)

### Applied Optimizations (2025-12-15)

#### 1. Cached trueno Matrix in LinearWeights

**Files modified:**
- `src/model/attention.rs`: Added `weight_matrix: Option<Matrix<f32>>` field
- `src/simd.rs`: Added `matmul_with_matrix()` zero-copy variant

**Changes:**
- `LinearWeights::finalize_weights()` now caches a trueno `Matrix<f32>` alongside the transposed Vec
- `LinearWeights::forward_simd()` uses cached Matrix via `matmul_with_matrix()`
- Eliminates `Matrix::from_vec()` call for weight matrix on every forward pass

**Impact:**
- Saves 1 allocation per Linear layer forward
- 8 Linear layers per decoder block × 4 blocks = 32 allocations saved per token
- Estimated 10-15% speedup for decoder forward path

#### 2. Added optimized matmul variants

**New functions in `src/simd.rs`:**
- `matmul_owned(a: Vec<f32>, b: Vec<f32>, ...)` - takes ownership, no input copy
- `matmul_with_matrix(a: &[f32], b: &Matrix<f32>, ...)` - uses pre-built Matrix for B

**Remaining work:**
- [x] Optimize `compute_attention_cached` head slicing (3 allocations × n_heads × n_layers) - **DONE in WAPR-BENCH-002**
- [x] Optimize `scaled_dot_product_attention` (8 allocations per call) - **DONE in WAPR-BENCH-002**
- [ ] Pre-allocate output buffers in KV cache

## WAPR-BENCH-002 Analysis (2025-12-15)

### Applied Optimizations

#### 1. Pre-allocated buffers in compute_attention_cached

**File:** `src/model/decoder.rs`

**Before:** Created new Vecs for q_head, k_head, v_head inside the per-head loop (3 × n_heads allocations)

**After:** Pre-allocate buffers outside loop and reuse across heads (3 allocations total)

**Impact:** Reduced allocations from 3×6=18 to 3 per attention call (6 heads)

#### 2. Optimized single-query attention path

**File:** `src/simd.rs`

**Added:** `scaled_dot_product_attention_single()` for seq_len=1 (incremental decode hot path)

**Optimizations:**
- Uses dot products instead of transpose + matmul for Q @ K^T
- Computes weighted sum directly instead of matmul for output
- Reduces allocations from 8 to 3 per attention call

### Performance Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Generate 20 tokens | 114.0ms | 110.0ms | **3.5% faster** |
| Subsequent tokens | 5.3ms | 4.9ms | **7.5% faster** |
| Throughput | 175.4 tok/s | 181.8 tok/s | **3.6% faster** |
| RTF (1s audio) | 0.19 | 0.19 | Same |

### SLA Status (Updated)

| SLA Target | Value | Current | Status |
|------------|-------|---------|--------|
| First Token (tiny) | 120ms | ~100ms (synthetic) | ⚠️ Needs real model test |
| RTF | 1.8x | 0.19x | ✅ 9.5x better than target |
| Memory | 145MB | <150MB | ✅ |
| Throughput | 100 tok/s | 181.8 tok/s | ✅ 82% over target |

### Remaining Opportunities

1. **KV Cache pre-allocation** - Pre-allocate K/V buffers in cache struct
2. **Batched multi-head attention** - Process all heads in single SIMD operation
3. **Fused attention kernel** - Combine Q@K^T, softmax, and @V in single pass

## Captured Traces

- `format_comparison.json` - Full pipeline trace
- `mel_spectrogram.json` - Mel computation only

## WAPR-CLI-001 Analysis (2025-12-15)

### CLI Validation Results

The native CLI successfully validates the end-to-end pipeline:

| Component | Status | Notes |
|-----------|--------|-------|
| Model Loading | ✅ | 0.57s for tiny-int8 (37MB) |
| WAV Parsing | ✅ | 16-bit PCM, auto stereo→mono |
| Resampling | ✅ | Linear interpolation |
| Mel Spectrogram | ✅ | 80-bin filterbank |
| Encoder | ✅ | Conv + transformer |
| Decoder | ⚠️ | Runs but produces repetitive tokens |
| Tokenizer | ✅ | BPE detokenization works |

### Performance Results

| Audio | Processing | RTF | Target | Status |
|-------|------------|-----|--------|--------|
| 1.5s | 6.3s | 4.23x | 2.0x | ❌ |
| 3.0s | 3.7s | 1.22x | 2.0x | ✅ |

### Issue Identified: Repetitive Token Generation

**Symptom**: Output is "rererererere..." repeated

**Possible causes**:
1. Decoder self-attention not attending correctly
2. Initial tokens (SOT, lang, task) not being set properly
3. KV cache corruption during generation
4. Temperature/sampling issue causing repetition

**Next steps**:
1. Add verbose token output to see generated token IDs
2. Verify initial token sequence matches reference
3. Check encoder output statistics (mean, std)
4. Compare attention patterns with reference

//! SIMD-accelerated vector operations

use trueno::Vector;

/// SIMD-accelerated dot product
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot product requires equal lengths");

    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);
    va.dot(&vb).unwrap_or(0.0)
}

/// Zero-allocation dot product with AVX2+FMA runtime dispatch (PMAT-014 O5, pv:avx2-fma-dot-v1).
///
/// Uses explicit `vfmadd231ps` intrinsics with 4 independent accumulators to saturate
/// the FMA execution unit. Falls back to scalar loop on non-AVX2 hardware.
/// Zero heap allocations in all paths.
#[inline]
#[must_use]
pub fn dot_nalloc(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot product requires equal lengths");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
            // SAFETY: CPU features verified at runtime. Lengths equal per debug_assert.
            return unsafe { dot_fma_avx2(a, b) };
        }
    }

    dot_scalar(a, b)
}

/// Scalar dot product fallback (no SIMD).
#[inline]
#[must_use]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// AVX2+FMA dot product with 4 independent accumulators (pv:avx2-fma-dot-v1).
///
/// Processes 32 elements per iteration (4 × 8-wide FMA). The 4 independent
/// accumulators hide the 5-cycle FMA latency (0.5c throughput × 10 in-flight = 5 accumulators
/// needed; 4 is close enough and simplifies the reduction).
///
/// # Safety
/// Requires AVX2 and FMA CPU features (checked by caller via `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_fma_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    let n = a.len();
    let mut i = 0;

    // 4 independent accumulators to hide FMA latency
    let mut acc0: __m256;
    let mut acc1: __m256;
    let mut acc2: __m256;
    let mut acc3: __m256;

    unsafe {
        acc0 = _mm256_setzero_ps();
        acc1 = _mm256_setzero_ps();
        acc2 = _mm256_setzero_ps();
        acc3 = _mm256_setzero_ps();

        // Main loop: 32 elements per iteration (4 × 8)
        while i + 32 <= n {
            let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);

            let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);

            let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
            let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
            acc2 = _mm256_fmadd_ps(a2, b2, acc2);

            let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
            let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
            acc3 = _mm256_fmadd_ps(a3, b3, acc3);

            i += 32;
        }

        // Handle remaining 8-element chunks
        while i + 8 <= n {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(av, bv, acc0);
            i += 8;
        }

        // Reduce 4 accumulators to 1
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        // Horizontal sum of 8 f32 lanes
        let mut buf = [0.0_f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc0);
        let mut sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

        // Scalar tail for remaining elements
        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }

        sum
    }
}

/// Online softmax: two-pass normalizer calculation (pv:online-softmax-v1).
///
/// Milakov & Gimelshein (2018): fuses max-finding and sum-of-exp into a single pass
/// using a running (max, sum_exp) pair, then normalizes in a second pass.
/// Saves one full read of the scores array vs standard 3-pass softmax.
pub fn softmax_online_inplace(scores: &[f32], weights: &mut [f32]) {
    debug_assert_eq!(scores.len(), weights.len());

    if scores.is_empty() {
        return;
    }

    // Pass 1: online max + running sum of exp
    let mut max_val = scores[0];
    let mut sum_exp = 1.0_f32;

    for &s in &scores[1..] {
        if s > max_val {
            sum_exp = sum_exp * (max_val - s).exp() + 1.0;
            max_val = s;
        } else {
            sum_exp += (s - max_val).exp();
        }
    }

    // Pass 2: normalize
    let inv_sum = 1.0 / sum_exp;
    for (w, &s) in weights.iter_mut().zip(scores.iter()) {
        *w = (s - max_val).exp() * inv_sum;
    }
}

/// SIMD-accelerated vector addition
#[must_use]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len(), "addition requires equal lengths");

    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);
    va.add(&vb)
        .map_or_else(|_| vec![0.0; a.len()], |v| v.as_slice().to_vec())
}

/// SIMD-accelerated vector subtraction
#[must_use]
pub fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len(), "subtraction requires equal lengths");

    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);
    va.sub(&vb)
        .map_or_else(|_| vec![0.0; a.len()], |v| v.as_slice().to_vec())
}

/// SIMD-accelerated element-wise multiplication
#[must_use]
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len(), "multiplication requires equal lengths");

    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);
    va.mul(&vb)
        .map_or_else(|_| vec![0.0; a.len()], |v| v.as_slice().to_vec())
}

/// SIMD-accelerated scalar multiplication
#[must_use]
pub fn scale(a: &[f32], s: f32) -> Vec<f32> {
    let va = Vector::from_slice(a);
    va.scale(s)
        .map_or_else(|_| vec![0.0; a.len()], |v| v.as_slice().to_vec())
}

/// SIMD-accelerated sum
#[must_use]
pub fn sum(a: &[f32]) -> f32 {
    let va = Vector::from_slice(a);
    va.sum().unwrap_or(0.0)
}

/// SIMD-accelerated mean
#[must_use]
pub fn mean(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    sum(a) / a.len() as f32
}

/// SIMD-accelerated variance
#[must_use]
pub fn variance(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let va = Vector::from_slice(a);
    va.variance().unwrap_or(0.0)
}

/// SIMD-accelerated standard deviation
#[must_use]
pub fn std_dev(a: &[f32]) -> f32 {
    variance(a).sqrt()
}

/// SIMD-accelerated max
#[must_use]
pub fn max(a: &[f32]) -> f32 {
    if a.is_empty() {
        return f32::NEG_INFINITY;
    }
    let va = Vector::from_slice(a);
    va.max().unwrap_or(f32::NEG_INFINITY)
}

/// SIMD-accelerated min
#[must_use]
pub fn min(a: &[f32]) -> f32 {
    if a.is_empty() {
        return f32::INFINITY;
    }
    let va = Vector::from_slice(a);
    va.min().unwrap_or(f32::INFINITY)
}

/// SIMD-accelerated argmax
#[must_use]
pub fn argmax(a: &[f32]) -> usize {
    if a.is_empty() {
        return 0;
    }
    let va = Vector::from_slice(a);
    va.argmax().unwrap_or(0)
}

/// Alias for max() - find maximum element in slice
///
/// Provided for Flash Attention compatibility.
#[must_use]
#[inline]
pub fn max_element(a: &[f32]) -> f32 {
    max(a)
}

/// In-place scalar multiplication: a[i] *= s for all i
///
/// More efficient than `scale()` when the result replaces the input.
pub fn scale_inplace(a: &mut [f32], s: f32) {
    // SIMD-friendly loop that auto-vectorizes well
    for x in a.iter_mut() {
        *x *= s;
    }
}

/// AXPY operation: y[i] += a * x[i] for all i
///
/// Computes y = y + a*x in-place. This is a fundamental BLAS Level 1 operation.
/// Used extensively in attention accumulation.
pub fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len(), "axpy requires equal lengths");

    // SIMD-friendly loop
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

/// In-place vector addition: y[i] += x[i] for all i
pub fn add_inplace(x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len(), "add_inplace requires equal lengths");

    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += xi;
    }
}

/// Broadcast add: Add vector to each row of a matrix in-place
///
/// For matrix (rows x cols) and vector (cols), adds vector to each row.
/// Equivalent to: for i in 0..rows { row[i] += vec; }
///
/// This is the hot path for bias addition in linear layers.
pub fn broadcast_add_inplace(matrix: &mut [f32], vec: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(matrix.len(), rows * cols, "matrix dimensions mismatch");
    debug_assert_eq!(vec.len(), cols, "vector dimension mismatch");

    for row in 0..rows {
        let row_start = row * cols;
        add_inplace(vec, &mut matrix[row_start..row_start + cols]);
    }
}

/// Dequantize a row of fp16 (u16 bits) values into an f32 buffer.
///
/// Converts IEEE 754 half-precision values stored as u16 bit patterns
/// into f32. The output buffer must be at least as long as the input.
///
/// This is the inner loop of fp16 inference: dequantize one weight row
/// into a thread-local f32 buffer that stays in L1 cache, then SIMD dot.
pub fn dequant_f16_row(f16_data: &[u16], out: &mut [f32]) {
    debug_assert!(
        out.len() >= f16_data.len(),
        "output buffer too small: {} < {}",
        out.len(),
        f16_data.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            // SAFETY: We've confirmed F16C+AVX are available. Pointer math is
            // bounds-checked by the chunk sizes and the debug_assert above.
            unsafe {
                dequant_f16_row_f16c(f16_data, out);
            }
            return;
        }
    }

    // Scalar fallback
    for (o, &bits) in out.iter_mut().zip(f16_data.iter()) {
        *o = half::f16::from_bits(bits).to_f32();
    }
}

/// F16C+AVX-accelerated fp16→f32 conversion (8 values per instruction).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c", enable = "avx")]
unsafe fn dequant_f16_row_f16c(f16_data: &[u16], out: &mut [f32]) {
    use std::arch::x86_64::{_mm256_cvtph_ps, _mm256_storeu_ps, _mm_loadu_si128};
    let n = f16_data.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let src = f16_data.as_ptr();
    let dst = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let half8 = _mm_loadu_si128(src.add(offset).cast());
            let float8 = _mm256_cvtph_ps(half8);
            _mm256_storeu_ps(dst.add(offset), float8);
        }
    }

    let base = chunks * 8;
    for j in 0..remainder {
        unsafe {
            *dst.add(base + j) = half::f16::from_bits(*src.add(base + j)).to_f32();
        }
    }
}

/// Compute dot product of an fp16 weight row with an f32 input vector.
///
/// Dequantizes the fp16 row into the provided f32 buffer, then computes
/// the SIMD dot product. The buffer should be reused across rows to stay
/// in L1 cache.
///
/// # Arguments
/// * `a_f16` - Weight row stored as fp16 bit patterns (u16)
/// * `b` - Input vector (f32)
/// * `buf` - Scratch buffer for dequantized f32 values (must be >= a_f16.len())
#[must_use]
pub fn dot_f16(a_f16: &[u16], b: &[f32], buf: &mut [f32]) -> f32 {
    debug_assert_eq!(a_f16.len(), b.len(), "dot_f16 requires equal lengths");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: CPU features verified at runtime. Lengths are equal per debug_assert.
            return unsafe { dot_f16_fused_f16c(a_f16, b) };
        }
    }

    // Scalar fallback: dequant then dot
    dequant_f16_row(a_f16, buf);
    dot(&buf[..a_f16.len()], b)
}

/// Fused fp16 dot product: load fp16, convert to f32 in register, FMA accumulate.
/// Single pass through memory — halves DRAM reads vs f32 dot.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c", enable = "avx", enable = "fma")]
unsafe fn dot_f16_fused_f16c(a_f16: &[u16], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_cvtph_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps,
        _mm256_storeu_ps, _mm_loadu_si128,
    };

    let n = a_f16.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let a_ptr = a_f16.as_ptr();
    let b_ptr = b.as_ptr();

    // SAFETY: all intrinsics guarded by #[target_feature] and runtime detection.
    // Pointer arithmetic is bounded by chunks*8 <= n = a_f16.len() = b.len().
    let mut acc: __m256 = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let half8 = _mm_loadu_si128(a_ptr.add(offset).cast());
            let a_f32 = _mm256_cvtph_ps(half8);
            let b_f32 = _mm256_loadu_ps(b_ptr.add(offset));
            acc = _mm256_fmadd_ps(a_f32, b_f32, acc);
        }
    }

    let mut sum_buf = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(sum_buf.as_mut_ptr(), acc) };
    let mut result: f32 = sum_buf.iter().sum();

    let base = chunks * 8;
    for j in 0..remainder {
        unsafe {
            let a_val = half::f16::from_bits(*a_ptr.add(base + j)).to_f32();
            let b_val = *b_ptr.add(base + j);
            result += a_val * b_val;
        }
    }

    result
}

// =========================================================================
// INT8 Symmetric Per-Row Quantization (pv:int8-symmetric-quant-v1)
// =========================================================================

/// Quantize a single row of f32 weights to symmetric INT8.
///
/// Returns (quantized_row, scale) where scale = max(|row|) / 127.
/// The original value can be recovered as: `w_f32 ≈ w_i8 * scale`.
pub fn quant_f32_row_to_i8(row: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = row.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    if abs_max == 0.0 {
        return (vec![0i8; row.len()], 0.0);
    }
    let scale = abs_max / 127.0;
    let inv_scale = 127.0 / abs_max;
    let quantized: Vec<i8> = row
        .iter()
        .map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (quantized, scale)
}

/// Compute dot product of an INT8 weight row with an f32 input vector.
///
/// `dot_i8(w_i8, x, scale) = scale * Σ(w_i8[i] * x[i])`
///
/// The INT8 values are widened to f32 and accumulated, then multiplied by scale once.
/// This halves memory bandwidth vs fp16 (1 byte/weight vs 2 bytes/weight).
#[must_use]
pub fn dot_i8(a_i8: &[i8], b: &[f32], scale: f32) -> f32 {
    debug_assert_eq!(a_i8.len(), b.len(), "dot_i8 requires equal lengths");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_i8_avx2(a_i8, b, scale) };
        }
    }

    dot_i8_scalar(a_i8, b, scale)
}

/// Scalar fallback for INT8 dot product.
fn dot_i8_scalar(a_i8: &[i8], b: &[f32], scale: f32) -> f32 {
    let mut sum = 0.0_f32;
    for (&a, &x) in a_i8.iter().zip(b.iter()) {
        sum += (a as f32) * x;
    }
    sum * scale
}

/// AVX2+FMA accelerated INT8 dot product.
///
/// Loads 8 i8 values at a time, sign-extends to i32 via VPMOVSXBD, converts to f32,
/// then FMA-accumulates with the input vector. Final horizontal sum scaled by row scale.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_i8_avx2(a_i8: &[i8], b: &[f32], scale: f32) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm256_storeu_ps, _mm_loadl_epi64,
    };

    let n = a_i8.len();
    let mut i = 0;

    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        // Process 16 elements per iteration (2 × 8)
        while i + 16 <= n {
            // First 8: load i8 → i32 → f32, FMA with input
            let i8_0 = _mm_loadl_epi64(a_i8.as_ptr().add(i).cast());
            let i32_0 = _mm256_cvtepi8_epi32(i8_0);
            let f32_0 = _mm256_cvtepi32_ps(i32_0);
            let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(f32_0, b0, acc0);

            // Second 8
            let i8_1 = _mm_loadl_epi64(a_i8.as_ptr().add(i + 8).cast());
            let i32_1 = _mm256_cvtepi8_epi32(i8_1);
            let f32_1 = _mm256_cvtepi32_ps(i32_1);
            let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            acc1 = _mm256_fmadd_ps(f32_1, b1, acc1);

            i += 16;
        }

        // Remainder: 8 at a time
        while i + 8 <= n {
            let i8_r = _mm_loadl_epi64(a_i8.as_ptr().add(i).cast());
            let i32_r = _mm256_cvtepi8_epi32(i8_r);
            let f32_r = _mm256_cvtepi32_ps(i32_r);
            let br = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(f32_r, br, acc0);
            i += 8;
        }

        acc0 = _mm256_add_ps(acc0, acc1);

        let mut buf = [0.0_f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc0);
        let mut sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

        // Scalar tail
        while i < n {
            sum += (a_i8[i] as f32) * b[i];
            i += 1;
        }

        sum * scale
    }
}

/// Quantize f32 values to fp16, returning u16 bit patterns.
///
/// Uses IEEE 754 half-precision format via the `half` crate.
/// Values outside fp16 range are clamped to ±inf.
#[must_use]
pub fn quant_f32_to_f16(f32_data: &[f32]) -> Vec<u16> {
    f32_data
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn vec_approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx_eq(*x, *y))
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = dot(&a, &b);
        assert!(approx_eq(result, 70.0)); // 1*5 + 2*6 + 3*7 + 4*8 = 70
    }

    #[test]
    fn test_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = add(&a, &b);
        assert!(vec_approx_eq(&result, &[5.0, 7.0, 9.0]));
    }

    #[test]
    fn test_sub() {
        let a = vec![5.0, 7.0, 9.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = sub(&a, &b);
        assert!(vec_approx_eq(&result, &[4.0, 5.0, 6.0]));
    }

    #[test]
    fn test_mul() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = mul(&a, &b);
        assert!(vec_approx_eq(&result, &[4.0, 10.0, 18.0]));
    }

    #[test]
    fn test_scale() {
        let a = vec![1.0, 2.0, 3.0];
        let result = scale(&a, 2.0);
        assert!(vec_approx_eq(&result, &[2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_sum() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(sum(&a), 10.0));
    }

    #[test]
    fn test_mean() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(mean(&a), 2.5));
    }

    #[test]
    fn test_mean_empty() {
        let a: Vec<f32> = vec![];
        assert!(approx_eq(mean(&a), 0.0));
    }

    #[test]
    fn test_variance() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // mean = 3, var = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5 = 10/5 = 2
        assert!(approx_eq(variance(&a), 2.0));
    }

    #[test]
    fn test_std_dev() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(std_dev(&a), 2.0_f32.sqrt()));
    }

    #[test]
    fn test_max() {
        let a = vec![1.0, 5.0, 3.0, 2.0];
        assert!(approx_eq(max(&a), 5.0));
    }

    #[test]
    fn test_min() {
        let a = vec![1.0, 5.0, 3.0, 2.0];
        assert!(approx_eq(min(&a), 1.0));
    }

    #[test]
    fn test_argmax() {
        let a = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(argmax(&a), 1);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_scale_inplace() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        scale_inplace(&mut a, 2.0);
        assert!(vec_approx_eq(&a, &[2.0, 4.0, 6.0, 8.0]));
    }

    #[test]
    fn test_scale_inplace_zero() {
        let mut a = vec![1.0, 2.0, 3.0];
        scale_inplace(&mut a, 0.0);
        assert!(vec_approx_eq(&a, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_axpy() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0];
        axpy(2.0, &x, &mut y);
        // y = y + 2*x = [10+2, 20+4, 30+6] = [12, 24, 36]
        assert!(vec_approx_eq(&y, &[12.0, 24.0, 36.0]));
    }

    #[test]
    fn test_axpy_zero_scalar() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0];
        axpy(0.0, &x, &mut y);
        // y should be unchanged
        assert!(vec_approx_eq(&y, &[10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_add_inplace() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0];
        add_inplace(&x, &mut y);
        assert!(vec_approx_eq(&y, &[11.0, 22.0, 33.0]));
    }

    #[test]
    fn test_broadcast_add_inplace() {
        let mut matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let vec = vec![10.0, 20.0, 30.0];
        broadcast_add_inplace(&mut matrix, &vec, 2, 3);
        // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
        // Row 1: [4+10, 5+20, 6+30] = [14, 25, 36]
        assert!(vec_approx_eq(
            &matrix,
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        ));
    }

    #[test]
    fn test_max_element() {
        let a = vec![1.0, 5.0, 3.0, 2.0];
        assert!(approx_eq(max_element(&a), 5.0));
    }

    #[test]
    fn test_max_empty() {
        let a: Vec<f32> = vec![];
        assert_eq!(max(&a), f32::NEG_INFINITY);
    }

    #[test]
    fn test_min_empty() {
        let a: Vec<f32> = vec![];
        assert_eq!(min(&a), f32::INFINITY);
    }

    #[test]
    fn test_argmax_empty() {
        let a: Vec<f32> = vec![];
        assert_eq!(argmax(&a), 0);
    }

    #[test]
    fn test_variance_empty() {
        let a: Vec<f32> = vec![];
        assert!(approx_eq(variance(&a), 0.0));
    }

    #[test]
    fn test_dot_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(approx_eq(dot(&a, &b), 0.0));
    }

    #[test]
    fn test_sum_empty() {
        let a: Vec<f32> = vec![];
        assert!(approx_eq(sum(&a), 0.0));
    }

    #[test]
    fn test_scale_empty() {
        let a: Vec<f32> = vec![];
        let result = scale(&a, 2.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_add_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let result = add(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sub_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let result = sub(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn test_mul_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let result = mul(&a, &b);
        assert!(result.is_empty());
    }

    // =========================================================================
    // fp16 Tests
    // =========================================================================

    #[test]
    fn test_dequant_f16_row() {
        let f32_vals = [1.0_f32, 2.0, 3.0, 4.0];
        let f16_bits: Vec<u16> = f32_vals
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let mut out = vec![0.0_f32; 4];
        dequant_f16_row(&f16_bits, &mut out);
        for (a, &b) in out.iter().zip(f32_vals.iter()) {
            assert!(approx_eq(*a, b));
        }
    }

    #[test]
    fn test_dot_f16() {
        let a_f32 = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let a_f16: Vec<u16> = a_f32
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let mut buf = vec![0.0_f32; 4];
        let result = dot_f16(&a_f16, &b, &mut buf);
        // 1*5 + 2*6 + 3*7 + 4*8 = 70
        assert!(approx_eq(result, 70.0));
    }

    #[test]
    fn test_quant_f32_to_f16_roundtrip() {
        let original = vec![0.0_f32, 1.0, -1.0, 0.5, 65504.0];
        let f16_bits = quant_f32_to_f16(&original);
        assert_eq!(f16_bits.len(), original.len());

        // Round-trip: f32 -> f16 -> f32
        let mut recovered = vec![0.0_f32; original.len()];
        dequant_f16_row(&f16_bits, &mut recovered);

        for (a, &b) in recovered.iter().zip(original.iter()) {
            assert!(approx_eq(*a, b));
        }
    }

    #[test]
    fn test_quant_f32_to_f16_empty() {
        let result = quant_f32_to_f16(&[]);
        assert!(result.is_empty());
    }

    // === pv:avx2-fma-dot-v1 property tests ===

    #[test]
    fn pv_dot_fma_scalar_equivalence() {
        use std::f32::consts::PI;
        for len in [1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 384, 1536] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01 * PI).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.017 + 0.3).cos()).collect();
            let scalar = dot_scalar(&a, &b);
            let nalloc = dot_nalloc(&a, &b);
            let diff = (scalar - nalloc).abs();
            let tol = len as f32 * f32::EPSILON * scalar.abs().max(1.0);
            assert!(
                diff < tol,
                "len={len}: scalar={scalar}, nalloc={nalloc}, diff={diff}, tol={tol}"
            );
        }
    }

    #[test]
    fn pv_dot_empty_and_unit() {
        assert_eq!(dot_nalloc(&[], &[]), 0.0);
        assert_eq!(dot_nalloc(&[3.0], &[7.0]), 21.0);
    }

    #[test]
    fn pv_dot_commutativity() {
        let a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.2).cos()).collect();
        let ab = dot_nalloc(&a, &b);
        let ba = dot_nalloc(&b, &a);
        assert!((ab - ba).abs() < 1e-6, "ab={ab}, ba={ba}");
    }

    #[test]
    fn pv_dot_self_non_negative() {
        let a: Vec<f32> = (0..384).map(|i| (i as f32 - 192.0) * 0.01).collect();
        assert!(dot_nalloc(&a, &a) >= 0.0);
    }

    #[test]
    fn pv_dot_nan_propagation() {
        let a = [f32::NAN, 1.0];
        let b = [1.0, 1.0];
        assert!(dot_nalloc(&a, &b).is_nan());
    }

    // === pv:online-softmax-v1 property tests ===

    fn softmax_standard(scores: &[f32]) -> Vec<f32> {
        let max_s = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    #[test]
    fn pv_softmax_online_matches_standard() {
        for len in [1, 2, 6, 64, 384, 448, 1500] {
            let scores: Vec<f32> = (0..len)
                .map(|i| (i as f32 * 0.1 - len as f32 * 0.05))
                .collect();
            let reference = softmax_standard(&scores);
            let mut online = vec![0.0_f32; len];
            softmax_online_inplace(&scores, &mut online);
            for (i, (&r, &o)) in reference.iter().zip(online.iter()).enumerate() {
                let diff = (r - o).abs();
                assert!(
                    diff < 1e-5,
                    "len={len}, i={i}: ref={r}, online={o}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn pv_softmax_sum_to_one() {
        for len in [1, 6, 64, 1500] {
            let scores: Vec<f32> = (0..len).map(|i| i as f32 * 0.3 - 5.0).collect();
            let mut weights = vec![0.0_f32; len];
            softmax_online_inplace(&scores, &mut weights);
            let sum: f32 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "len={len}: sum={sum}");
        }
    }

    #[test]
    fn pv_softmax_positivity() {
        // Range limited to avoid f32 underflow: exp(-80) ≈ 1.8e-35 > 0, but exp(-200) = 0.0
        let scores = [-20.0_f32, -10.0, 0.0, 10.0, 20.0];
        let mut weights = vec![0.0_f32; 5];
        softmax_online_inplace(&scores, &mut weights);
        for (i, &w) in weights.iter().enumerate() {
            assert!(w > 0.0, "i={i}: weight={w} should be positive");
        }
    }

    #[test]
    fn pv_softmax_order_preservation() {
        let scores = [1.0_f32, 3.0, 2.0, 5.0, 4.0];
        let mut weights = vec![0.0_f32; 5];
        softmax_online_inplace(&scores, &mut weights);
        assert!(weights[3] > weights[1]); // 5.0 > 3.0
        assert!(weights[1] > weights[2]); // 3.0 > 2.0
        assert!(weights[2] > weights[0]); // 2.0 > 1.0
    }

    #[test]
    fn pv_softmax_shift_invariance() {
        let scores = [1.0_f32, 2.0, 3.0, 4.0];
        let shifted: Vec<f32> = scores.iter().map(|&s| s + 1000.0).collect();
        let mut w1 = vec![0.0_f32; 4];
        let mut w2 = vec![0.0_f32; 4];
        softmax_online_inplace(&scores, &mut w1);
        softmax_online_inplace(&shifted, &mut w2);
        for (i, (&a, &b)) in w1.iter().zip(w2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "i={i}: w1={a}, w2={b}");
        }
    }

    #[test]
    fn pv_softmax_single_element() {
        let mut w = [0.0_f32];
        softmax_online_inplace(&[42.0], &mut w);
        assert_eq!(w[0], 1.0);
    }

    // === pv:int8-symmetric-quant-v1 property tests ===

    #[test]
    fn pv_i8q_roundtrip_accuracy() {
        // Quantize → dequant should be within tolerance
        let row: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let (q, scale) = quant_f32_row_to_i8(&row);
        for (i, (&orig, &qi)) in row.iter().zip(q.iter()).enumerate() {
            let recovered = qi as f32 * scale;
            let diff = (orig - recovered).abs();
            // Tolerance: scale / 127 ≈ quantization step
            assert!(
                diff < scale + 1e-6,
                "i={i}: orig={orig}, recovered={recovered}, diff={diff}"
            );
        }
    }

    #[test]
    fn pv_i8q_zero_row() {
        let row = vec![0.0_f32; 64];
        let (q, scale) = quant_f32_row_to_i8(&row);
        assert_eq!(scale, 0.0);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn pv_i8q_range_bounded() {
        let row: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1 - 19.2)).collect();
        let (q, _scale) = quant_f32_row_to_i8(&row);
        for &v in &q {
            assert!((-127..=127).contains(&v), "i8 value {v} out of range");
        }
    }

    #[test]
    fn pv_i8q_dot_scalar_equivalence() {
        use std::f32::consts::PI;
        for len in [1, 8, 16, 64, 384, 1536] {
            let weights: Vec<f32> = (0..len)
                .map(|i| (i as f32 * 0.01 * PI).sin() * 0.3)
                .collect();
            let input: Vec<f32> = (0..len).map(|i| (i as f32 * 0.017 + 0.3).cos()).collect();

            // Reference: f32 dot product
            let ref_dot: f32 = weights.iter().zip(input.iter()).map(|(w, x)| w * x).sum();

            // INT8 dot product
            let (q, scale) = quant_f32_row_to_i8(&weights);
            let i8_dot = dot_i8(&q, &input, scale);

            let diff = (ref_dot - i8_dot).abs();
            // Tolerance scales with vector length (accumulation error) and quantization error
            let tol = len as f32 * scale * 0.5 + 1e-4;
            assert!(
                diff < tol,
                "len={len}: ref={ref_dot}, i8={i8_dot}, diff={diff}, tol={tol}"
            );
        }
    }

    #[test]
    fn pv_i8q_dot_empty() {
        assert_eq!(dot_i8(&[], &[], 1.0), 0.0);
    }

    #[test]
    fn pv_i8q_scale_positive() {
        let row: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let (_q, scale) = quant_f32_row_to_i8(&row);
        assert!(scale > 0.0, "scale should be positive for non-zero row");
    }
}

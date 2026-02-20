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
}

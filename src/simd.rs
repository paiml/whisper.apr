//! SIMD-accelerated operations via trueno
//!
//! Provides optimized implementations of common ML operations using
//! trueno's backend-agnostic SIMD acceleration.
//!
//! # Operations
//!
//! - Matrix multiplication (`matmul`)
//! - Softmax with numerical stability
//! - Layer normalization
//! - GELU activation
//! - Scaled dot-product attention
//!
//! # Backend Selection
//!
//! The module automatically selects the best available backend:
//! - WASM SIMD (128-bit) for browser deployment
//! - Native SIMD (AVX2/AVX-512) for server deployment
//! - Scalar fallback for maximum compatibility

use trueno::{Backend, Matrix, Vector};

/// Get the best available SIMD backend
#[must_use]
pub fn best_backend() -> Backend {
    trueno::select_best_available_backend()
}

/// Check if SIMD is available
#[must_use]
pub fn simd_available() -> bool {
    !matches!(best_backend(), Backend::Scalar)
}

/// Get backend name for debugging
#[must_use]
pub fn backend_name() -> &'static str {
    match best_backend() {
        Backend::Scalar => "Scalar",
        Backend::SSE2 => "SSE2",
        Backend::AVX => "AVX",
        Backend::AVX2 => "AVX2",
        Backend::AVX512 => "AVX512",
        Backend::NEON => "NEON",
        Backend::WasmSIMD => "WasmSIMD",
        Backend::GPU => "GPU",
        Backend::Auto => "Auto",
    }
}

// ============================================================================
// Vector Operations
// ============================================================================

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
    let m = mean(a);
    // Compute (x - mean)^2 and sum
    let mut sum_sq = 0.0;
    for &x in a {
        let diff = x - m;
        sum_sq += diff * diff;
    }
    sum_sq / a.len() as f32
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

// ============================================================================
// Matrix Operations
// ============================================================================

/// SIMD-accelerated matrix multiplication
///
/// Computes C = A @ B where A is (rows x inner) and B is (inner x cols)
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn matmul(a: &[f32], b: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), rows * inner, "A dimensions mismatch");
    debug_assert_eq!(b.len(), inner * cols, "B dimensions mismatch");

    let Ok(ma) = Matrix::from_vec(rows, inner, a.to_vec()) else {
        return vec![0.0; rows * cols];
    };
    let Ok(mb) = Matrix::from_vec(inner, cols, b.to_vec()) else {
        return vec![0.0; rows * cols];
    };
    ma.matmul(&mb)
        .map_or_else(|_| vec![0.0; rows * cols], |mc| mc.as_slice().to_vec())
}

/// SIMD-accelerated matrix-vector multiplication
///
/// Computes y = A @ x where A is (rows x cols) and x is (cols,)
#[must_use]
pub fn matvec(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), rows * cols, "A dimensions mismatch");
    debug_assert_eq!(x.len(), cols, "x dimension mismatch");

    let Ok(ma) = Matrix::from_vec(rows, cols, a.to_vec()) else {
        return vec![0.0; rows];
    };
    let vx = Vector::from_slice(x);
    ma.matvec(&vx)
        .map_or_else(|_| vec![0.0; rows], |v| v.as_slice().to_vec())
}

/// SIMD-accelerated matrix transpose
#[must_use]
pub fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), rows * cols, "dimensions mismatch");

    let Ok(ma) = Matrix::from_vec(rows, cols, a.to_vec()) else {
        return vec![0.0; rows * cols];
    };
    ma.transpose().as_slice().to_vec()
}

// ============================================================================
// Activation Functions
// ============================================================================

/// SIMD-accelerated softmax with numerical stability
///
/// Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
#[must_use]
pub fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }

    // Find max for numerical stability
    let max_val = max(x);

    // Compute exp(x - max)
    let mut exp_vals: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();

    // Normalize
    let sum_exp: f32 = exp_vals.iter().sum();
    if sum_exp > 0.0 {
        for v in &mut exp_vals {
            *v /= sum_exp;
        }
    }

    exp_vals
}

/// SIMD-accelerated log-softmax with numerical stability
#[must_use]
pub fn log_softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }

    let max_val = max(x);
    let shifted: Vec<f32> = x.iter().map(|&v| v - max_val).collect();
    let log_sum_exp = shifted.iter().map(|&v| v.exp()).sum::<f32>().ln();

    shifted.iter().map(|&v| v - log_sum_exp).collect()
}

/// SIMD-accelerated GELU activation
///
/// GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[must_use]
pub fn gelu(x: &[f32]) -> Vec<f32> {
    const SQRT_2_PI: f32 = 0.797_884_6; // sqrt(2/π)
    const COEF: f32 = 0.044_715;

    x.iter()
        .map(|&v| {
            let x3 = v * v * v;
            let inner = SQRT_2_PI * COEF.mul_add(x3, v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect()
}

/// SIMD-accelerated ReLU activation
#[must_use]
pub fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// SIMD-accelerated sigmoid activation
#[must_use]
pub fn sigmoid(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

/// SIMD-accelerated tanh activation
#[must_use]
pub fn tanh_activation(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.tanh()).collect()
}

// ============================================================================
// Layer Operations
// ============================================================================

/// SIMD-accelerated layer normalization
///
/// LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), gamma.len(), "gamma dimension mismatch");
    debug_assert_eq!(x.len(), beta.len(), "beta dimension mismatch");

    let m = mean(x);
    let v = variance(x);
    let std = (v + eps).sqrt();

    x.iter()
        .zip(gamma.iter())
        .zip(beta.iter())
        .map(|((&xi, &g), &b)| ((xi - m) / std).mul_add(g, b))
        .collect()
}

/// SIMD-accelerated batch layer normalization
///
/// Applies layer norm to each row of a (batch x features) matrix
pub fn batch_layer_norm(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch_size: usize,
    features: usize,
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), batch_size * features, "x dimensions mismatch");
    debug_assert_eq!(gamma.len(), features, "gamma dimension mismatch");
    debug_assert_eq!(beta.len(), features, "beta dimension mismatch");

    let mut output = Vec::with_capacity(x.len());

    for i in 0..batch_size {
        let start = i * features;
        let end = start + features;
        let row = &x[start..end];
        let normalized = layer_norm(row, gamma, beta, eps);
        output.extend(normalized);
    }

    output
}

// ============================================================================
// Attention Operations
// ============================================================================

/// SIMD-accelerated scaled dot-product attention
///
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// # Arguments
/// * `query` - Query tensor (seq_len x d_model)
/// * `key` - Key tensor (seq_len x d_model)
/// * `value` - Value tensor (seq_len x d_model)
/// * `seq_len` - Sequence length
/// * `d_model` - Model dimension
/// * `mask` - Optional attention mask (seq_len x seq_len)
pub fn scaled_dot_product_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq_len: usize,
    d_model: usize,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let scale = 1.0 / (d_model as f32).sqrt();

    // Q @ K^T
    let key_t = transpose(key, seq_len, d_model);
    let mut scores = matmul(query, &key_t, seq_len, d_model, seq_len);

    // Scale
    for s in &mut scores {
        *s *= scale;
    }

    // Apply mask (add large negative values to masked positions)
    if let Some(m) = mask {
        debug_assert_eq!(m.len(), seq_len * seq_len, "mask dimensions mismatch");
        for (i, &mask_val) in m.iter().enumerate() {
            if mask_val == 0.0 {
                scores[i] = f32::NEG_INFINITY;
            }
        }
    }

    // Row-wise softmax
    let mut weights = Vec::with_capacity(scores.len());
    for i in 0..seq_len {
        let start = i * seq_len;
        let end = start + seq_len;
        let row_softmax = softmax(&scores[start..end]);
        weights.extend(row_softmax);
    }

    // Weights @ V
    matmul(&weights, value, seq_len, seq_len, d_model)
}

// ============================================================================
// FFT Operations (for mel spectrogram)
// ============================================================================

/// Generate a Hann window
#[must_use]
pub fn hann_window(size: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    (0..size)
        .map(|i| {
            let x = (PI * i as f32) / (size - 1) as f32;
            x.sin().powi(2)
        })
        .collect()
}

/// SIMD-accelerated element-wise multiply-accumulate
///
/// Computes sum(a[i] * b[i]) - useful for convolutions
#[must_use]
pub fn multiply_accumulate(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

// ============================================================================
// Tests
// ============================================================================

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

    // =========================================================================
    // Backend Tests
    // =========================================================================

    #[test]
    fn test_backend_available() {
        let backend = best_backend();
        let name = backend_name();
        assert!(!name.is_empty());
        println!("Backend: {name:?} = {backend:?}");
    }

    #[test]
    fn test_simd_available() {
        // Just verify it doesn't panic
        let _ = simd_available();
    }

    // =========================================================================
    // Vector Operation Tests
    // =========================================================================

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
    // Matrix Operation Tests
    // =========================================================================

    #[test]
    fn test_matmul_identity() {
        // 2x2 identity matrix
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let identity = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let result = matmul(&a, &identity, 2, 2, 2);
        assert!(vec_approx_eq(&result, &a));
    }

    #[test]
    fn test_matmul_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let result = matmul(&a, &b, 2, 2, 2);
        // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
        assert!(vec_approx_eq(&result, &[19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn test_matvec() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let x = vec![5.0, 6.0]; // 2
        let result = matvec(&a, &x, 2, 2);
        // [1*5+2*6, 3*5+4*6] = [17, 39]
        assert!(vec_approx_eq(&result, &[17.0, 39.0]));
    }

    #[test]
    fn test_transpose() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let result = transpose(&a, 2, 3);
        // [1, 4, 2, 5, 3, 6] as 3x2
        assert!(vec_approx_eq(&result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    }

    // =========================================================================
    // Activation Function Tests
    // =========================================================================

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax(&x);
        assert_eq!(result.len(), 3);
        // Sum should be 1
        let total: f32 = result.iter().sum();
        assert!(approx_eq(total, 1.0));
        // Should be monotonically increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that could overflow without max subtraction
        let x = vec![1000.0, 1001.0, 1002.0];
        let result = softmax(&x);
        let total: f32 = result.iter().sum();
        assert!(approx_eq(total, 1.0));
        // All values should be finite
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_softmax_empty() {
        let x: Vec<f32> = vec![];
        let result = softmax(&x);
        assert!(result.is_empty());
    }

    #[test]
    fn test_log_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let result = log_softmax(&x);
        // exp(log_softmax) should equal softmax
        let softmax_result = softmax(&x);
        let exp_log_softmax: Vec<f32> = result.iter().map(|v| v.exp()).collect();
        assert!(vec_approx_eq(&exp_log_softmax, &softmax_result));
    }

    #[test]
    fn test_gelu() {
        let x = vec![-1.0, 0.0, 1.0];
        let result = gelu(&x);
        assert_eq!(result.len(), 3);
        // GELU(0) = 0
        assert!(approx_eq(result[1], 0.0));
        // GELU(x) > 0 for x > 0
        assert!(result[2] > 0.0);
    }

    #[test]
    fn test_relu() {
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        let result = relu(&x);
        assert!(vec_approx_eq(&result, &[0.0, 0.0, 1.0, 2.0]));
    }

    #[test]
    fn test_sigmoid() {
        let x = vec![-100.0, 0.0, 100.0];
        let result = sigmoid(&x);
        // sigmoid(-large) ≈ 0
        assert!(result[0] < 0.01);
        // sigmoid(0) = 0.5
        assert!(approx_eq(result[1], 0.5));
        // sigmoid(large) ≈ 1
        assert!(result[2] > 0.99);
    }

    #[test]
    fn test_tanh() {
        let x = vec![-100.0, 0.0, 100.0];
        let result = tanh_activation(&x);
        // tanh(-large) ≈ -1
        assert!(result[0] < -0.99);
        // tanh(0) = 0
        assert!(approx_eq(result[1], 0.0));
        // tanh(large) ≈ 1
        assert!(result[2] > 0.99);
    }

    // =========================================================================
    // Layer Operation Tests
    // =========================================================================

    #[test]
    fn test_layer_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        let result = layer_norm(&x, &gamma, &beta, 1e-5);

        // Mean should be ~0 after normalization
        assert!(approx_eq(mean(&result), 0.0));
        // Variance should be ~1 after normalization
        assert!((variance(&result) - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_layer_norm_with_params() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0, 2.0, 2.0, 2.0];
        let beta = vec![1.0, 1.0, 1.0, 1.0];
        let result = layer_norm(&x, &gamma, &beta, 1e-5);

        // Mean should be ~1 (beta) after normalization
        assert!((mean(&result) - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_batch_layer_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 batches x 3 features
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];
        let result = batch_layer_norm(&x, &gamma, &beta, 2, 3, 1e-5);

        assert_eq!(result.len(), 6);

        // Each row should have mean ~0
        assert!(approx_eq(mean(&result[0..3]), 0.0));
        assert!(approx_eq(mean(&result[3..6]), 0.0));
    }

    // =========================================================================
    // Attention Tests
    // =========================================================================

    #[test]
    fn test_scaled_dot_product_attention() {
        // Simple 2x4 test
        let query = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // 2x4
        let key = query.clone();
        let value = query.clone();

        let result = scaled_dot_product_attention(&query, &key, &value, 2, 4, None);
        assert_eq!(result.len(), 8);
        // All values should be finite
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_scaled_dot_product_attention_with_mask() {
        let query = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // 2x4
        let key = query.clone();
        let value = query.clone();

        // Causal mask: lower triangular
        let mask = vec![1.0, 0.0, 1.0, 1.0]; // 2x2

        let result = scaled_dot_product_attention(&query, &key, &value, 2, 4, Some(&mask));
        assert_eq!(result.len(), 8);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    // =========================================================================
    // FFT Operation Tests
    // =========================================================================

    #[test]
    fn test_hann_window() {
        let window = hann_window(4);
        assert_eq!(window.len(), 4);
        // Hann window is symmetric
        assert!(approx_eq(window[0], window[3]));
        assert!(approx_eq(window[1], window[2]));
        // Endpoints should be near 0
        assert!(window[0] < 0.1);
    }

    #[test]
    fn test_multiply_accumulate() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = multiply_accumulate(&a, &b);
        assert!(approx_eq(result, 32.0)); // 1*4 + 2*5 + 3*6 = 32
    }

    // =========================================================================
    // Property-Based Tests (WAPR-QA-002)
    // =========================================================================

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn property_dot_commutative(len in 4usize..256) {
                let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
                let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();

                let dot_ab = dot(&a, &b);
                let dot_ba = dot(&b, &a);

                prop_assert!((dot_ab - dot_ba).abs() < 1e-4, "dot product should be commutative");
            }

            #[test]
            fn property_softmax_sums_to_one(len in 4usize..128) {
                let input: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1) - 5.0).collect();
                let output = softmax(&input);

                let sum: f32 = output.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-5, "softmax sum {} should be 1.0", sum);
            }

            #[test]
            fn property_softmax_nonnegative(len in 4usize..128) {
                let input: Vec<f32> = (0..len).map(|i| (i as f32 * 0.3) - 10.0).collect();
                let output = softmax(&input);

                for val in &output {
                    prop_assert!(*val >= 0.0, "softmax output should be non-negative");
                }
            }

            #[test]
            fn property_gelu_bounded(len in 4usize..256) {
                let input: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2) - 10.0).collect();
                let output = gelu(&input);

                for (inp, out) in input.iter().zip(output.iter()) {
                    // GELU(x) is bounded: for x < 0, output < 0; for x > 0, output > 0
                    if *inp > 3.0 {
                        prop_assert!(*out > 0.0, "GELU of positive {} should be positive", inp);
                    }
                    if *inp < -3.0 {
                        prop_assert!(*out < 0.1, "GELU of negative {} should be small", inp);
                    }
                }
            }

            #[test]
            fn property_layer_norm_mean_zero(len in 8usize..256) {
                let input: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
                let gamma = vec![1.0; len];
                let beta = vec![0.0; len];

                let output = layer_norm(&input, &gamma, &beta, 1e-5);
                let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;

                prop_assert!(mean.abs() < 1e-4, "layer_norm mean {} should be ~0", mean);
            }

            #[test]
            fn property_matmul_output_shape(m in 2usize..16, k in 2usize..16, n in 2usize..16) {
                let a: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.01).sin()).collect();
                let b: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.02).cos()).collect();

                let c = matmul(&a, &b, m, k, n);
                prop_assert_eq!(c.len(), m * n, "matmul output shape should be m*n");
            }
        }
    }
}

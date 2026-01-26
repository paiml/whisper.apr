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
}

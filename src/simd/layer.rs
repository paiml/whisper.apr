//! SIMD-accelerated layer operations

use trueno::Vector;

/// SIMD-accelerated layer normalization
///
/// LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), gamma.len(), "gamma dimension mismatch");
    debug_assert_eq!(x.len(), beta.len(), "beta dimension mismatch");

    if x.is_empty() {
        return vec![];
    }

    let vx = Vector::from_slice(x);
    let vgamma = Vector::from_slice(gamma);
    let vbeta = Vector::from_slice(beta);

    vx.layer_norm(&vgamma, &vbeta, eps)
        .map_or_else(|_| vec![0.0; x.len()], |v| v.as_slice().to_vec())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{mean, variance};

    const EPSILON: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

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

    #[test]
    fn test_layer_norm_empty() {
        let x: Vec<f32> = vec![];
        let gamma: Vec<f32> = vec![];
        let beta: Vec<f32> = vec![];
        let result = layer_norm(&x, &gamma, &beta, 1e-5);
        assert!(result.is_empty());
    }
}

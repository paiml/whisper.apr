//! SIMD-accelerated activation functions

use trueno::Vector;

/// Apply a SIMD vector operation, handling empty input and errors.
fn apply_vector_op<E>(x: &[f32], op: impl FnOnce(&Vector<f32>) -> Result<Vector<f32>, E>) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    let vx = Vector::from_slice(x);
    op(&vx).map_or_else(|_| vec![0.0; x.len()], |v| v.as_slice().to_vec())
}

/// SIMD-accelerated softmax with numerical stability
///
/// Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
#[must_use]
pub fn softmax(x: &[f32]) -> Vec<f32> {
    apply_vector_op(x, Vector::softmax)
}

/// SIMD-accelerated log-softmax with numerical stability
#[must_use]
pub fn log_softmax(x: &[f32]) -> Vec<f32> {
    apply_vector_op(x, Vector::log_softmax)
}

/// SIMD-accelerated GELU activation
///
/// GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[must_use]
pub fn gelu(x: &[f32]) -> Vec<f32> {
    apply_vector_op(x, Vector::gelu)
}

/// SIMD-accelerated ReLU activation
#[must_use]
pub fn relu(x: &[f32]) -> Vec<f32> {
    apply_vector_op(x, Vector::relu)
}

/// SIMD-accelerated sigmoid activation
#[must_use]
pub fn sigmoid(x: &[f32]) -> Vec<f32> {
    apply_vector_op(x, Vector::sigmoid)
}

/// SIMD-accelerated tanh activation
#[must_use]
pub fn tanh_activation(x: &[f32]) -> Vec<f32> {
    apply_vector_op(x, Vector::tanh)
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
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_log_softmax_empty() {
        let x: Vec<f32> = vec![];
        let result = log_softmax(&x);
        assert!(result.is_empty());
    }

    #[test]
    fn test_gelu_empty() {
        let x: Vec<f32> = vec![];
        let result = gelu(&x);
        assert!(result.is_empty());
    }

    #[test]
    fn test_relu_empty() {
        let x: Vec<f32> = vec![];
        let result = relu(&x);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sigmoid_empty() {
        let x: Vec<f32> = vec![];
        let result = sigmoid(&x);
        assert!(result.is_empty());
    }

    #[test]
    fn test_tanh_empty() {
        let x: Vec<f32> = vec![];
        let result = tanh_activation(&x);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_single() {
        let x = vec![1.0];
        let result = softmax(&x);
        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0], 1.0)); // softmax of single element is 1
    }

    #[test]
    fn test_log_softmax_single() {
        let x = vec![1.0];
        let result = log_softmax(&x);
        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0], 0.0)); // log(1) = 0
    }

    #[test]
    fn test_gelu_positive() {
        let x = vec![0.5, 1.0, 2.0, 3.0];
        let result = gelu(&x);
        // GELU(x) ≈ x for large positive x
        assert!(result[3] > 2.9);
        // All should be positive
        assert!(result.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_gelu_negative() {
        let x = vec![-3.0, -2.0, -1.0, -0.5];
        let result = gelu(&x);
        // GELU is bounded below
        assert!(result.iter().all(|&v| v > -0.5));
    }

    #[test]
    fn test_relu_all_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = relu(&x);
        assert!(vec_approx_eq(&result, &x));
    }

    #[test]
    fn test_relu_all_negative() {
        let x = vec![-1.0, -2.0, -3.0, -4.0];
        let result = relu(&x);
        assert!(vec_approx_eq(&result, &[0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_sigmoid_gradient_region() {
        // Test values in the gradient-sensitive region
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = sigmoid(&x);
        // Should be strictly increasing
        for i in 1..result.len() {
            assert!(result[i] > result[i - 1]);
        }
    }

    #[test]
    fn test_tanh_symmetry() {
        let x = vec![-2.0, -1.0, 1.0, 2.0];
        let result = tanh_activation(&x);
        // tanh is odd: tanh(-x) = -tanh(x)
        assert!(approx_eq(result[0], -result[3]));
        assert!(approx_eq(result[1], -result[2]));
    }

    #[test]
    fn test_softmax_uniform() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let result = softmax(&x);
        // All should be equal (0.25)
        for &v in &result {
            assert!(approx_eq(v, 0.25));
        }
    }

    #[test]
    fn test_log_softmax_numerical_stability() {
        // Large values
        let x = vec![1000.0, 1001.0, 1002.0];
        let result = log_softmax(&x);
        // All should be finite
        assert!(result.iter().all(|&v| v.is_finite()));
        // Should be negative
        assert!(result.iter().all(|&v| v <= 0.0));
    }
}

//! Property-based tests for SIMD operations (WAPR-QA-002)

#[cfg(test)]
mod tests {
    use crate::simd::{dot, gelu, layer_norm, matmul, softmax};
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

//! FFT-related operations (for mel spectrogram)

use super::vector::dot;

/// Generate a Hann window
#[must_use]
pub fn hann_window(size: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    if size <= 1 {
        return vec![1.0; size];
    }
    let window: Vec<f32> = (0..size)
        .map(|i| {
            let x = (PI * i as f32) / (size - 1) as f32;
            x.sin().powi(2)
        })
        .collect();
    debug_assert_eq!(
        window.len(),
        size,
        "hann window length must match requested size"
    );
    debug_assert!(
        window.iter().all(|&x| (0.0..=1.0).contains(&x)),
        "all hann window values must be in [0, 1]"
    );
    window
}

/// SIMD-accelerated element-wise multiply-accumulate
///
/// Computes sum(a[i] * b[i]) - useful for convolutions
#[must_use]
pub fn multiply_accumulate(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

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

    #[test]
    fn test_hann_window_large() {
        let window = hann_window(256);
        assert_eq!(window.len(), 256);
        // Peak should be at center
        assert!(window[128] > window[0]);
        assert!(window[128] > window[255]);
    }
}

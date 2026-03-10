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

mod activation;
mod attention;
mod fft;
mod layer;
mod matrix;
mod optimized;
#[cfg(test)]
mod property_tests;
mod vector;

pub use activation::{gelu, log_softmax, relu, sigmoid, softmax, tanh_activation};
pub use attention::{scaled_dot_product_attention, scaled_dot_product_attention_single};
pub use fft::{hann_window, multiply_accumulate};
pub use layer::{batch_layer_norm, layer_norm};
pub use matrix::{
    enable_blis_profiling, matmul, matmul_owned, matmul_raw, matmul_with_matrix,
    matmul_with_prepacked, matvec, take_blis_profiler, transpose,
};
pub use optimized::{
    rms_norm, rms_norm_into, select_backend, tiled_matvec, tiled_matvec_f16, tiled_matvec_f16_into,
    tiled_matvec_i8_into, tiled_matvec_into, BackendCategory, GPU_THRESHOLD, PARALLEL_THRESHOLD,
    TILE_SIZE,
};
pub use vector::{
    add, add_inplace, argmax, axpy, broadcast_add_inplace, dequant_f16_row, dot, dot_f16, dot_i8,
    dot_nalloc, dot_scalar, max, max_element, mean, min, mul, quant_f32_row_to_i8,
    quant_f32_to_f16, scale, scale_inplace, softmax_online_inplace, std_dev, sub, sum, variance,
};

use trueno::Backend;

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
    backend_name_for(best_backend())
}

/// Get name string for a specific backend
#[must_use]
pub fn backend_name_for(backend: Backend) -> &'static str {
    match backend {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_backend_name_valid() {
        let name = backend_name();
        let valid_names = [
            "Scalar", "SSE2", "AVX", "AVX2", "AVX512", "NEON", "WasmSIMD", "GPU", "Auto",
        ];
        assert!(
            valid_names.contains(&name),
            "Backend name '{name}' not in valid list"
        );
    }

    #[test]
    fn test_best_backend_consistency() {
        // Multiple calls should return the same backend
        let b1 = best_backend();
        let b2 = best_backend();
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_simd_available_consistency() {
        let a1 = simd_available();
        let a2 = simd_available();
        assert_eq!(a1, a2);
    }

    #[test]
    fn test_re_exports() {
        // Verify key re-exports are accessible
        let _ = dot(&[1.0, 2.0], &[3.0, 4.0]);
        let _ = softmax(&[1.0, 2.0, 3.0]);
        let _ = gelu(&[0.0, 1.0]);
        let _ = relu(&[-1.0, 0.0, 1.0]);
        let _ = matmul(&[1.0, 0.0, 0.0, 1.0], &[1.0, 2.0, 3.0, 4.0], 2, 2, 2);
    }

    #[test]
    fn test_backend_name_for_all_variants() {
        assert_eq!(backend_name_for(Backend::Scalar), "Scalar");
        assert_eq!(backend_name_for(Backend::SSE2), "SSE2");
        assert_eq!(backend_name_for(Backend::AVX), "AVX");
        assert_eq!(backend_name_for(Backend::AVX2), "AVX2");
        assert_eq!(backend_name_for(Backend::AVX512), "AVX512");
        assert_eq!(backend_name_for(Backend::NEON), "NEON");
        assert_eq!(backend_name_for(Backend::WasmSIMD), "WasmSIMD");
        assert_eq!(backend_name_for(Backend::GPU), "GPU");
        assert_eq!(backend_name_for(Backend::Auto), "Auto");
    }
}

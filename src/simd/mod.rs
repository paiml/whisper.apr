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
pub use matrix::{matmul, matmul_owned, matmul_with_matrix, matvec, transpose};
pub use optimized::{
    rms_norm, rms_norm_into, select_backend, tiled_matvec, tiled_matvec_into, BackendCategory,
    GPU_THRESHOLD, PARALLEL_THRESHOLD, TILE_SIZE,
};
pub use vector::{
    add, add_inplace, argmax, axpy, broadcast_add_inplace, dot, max, max_element, mean, min, mul,
    scale, scale_inplace, std_dev, sub, sum, variance,
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
}

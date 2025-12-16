//! GPU compute operations (WAPR-130 to WAPR-133)
//!
//! Provides GPU-accelerated implementations of core operations used in
//! transformer inference.

mod gelu;
mod layernorm;
mod matmul;
mod softmax;

pub use gelu::{GeluApproximation, GeluConfig, GpuGelu};
pub use layernorm::{GpuLayerNorm, LayerNormConfig, LayerNormMode};
pub use matmul::{GpuMatMul, MatMulConfig, MatMulDimensions, TileSize};
pub use softmax::{GpuSoftmax, SoftmaxConfig, SoftmaxDimension};

/// Validate matrix dimensions for multiplication
#[must_use]
pub fn validate_matmul_dims(m: u32, k: u32, n: u32) -> bool {
    m > 0 && k > 0 && n > 0
}

/// Calculate optimal tile size for matrix multiplication
#[must_use]
pub fn optimal_tile_size(m: u32, n: u32, max_workgroup: u32) -> u32 {
    // Common tile sizes that work well on most GPUs
    let candidates = [32, 16, 8];

    for &tile in &candidates {
        if tile * tile <= max_workgroup && m >= tile && n >= tile {
            return tile;
        }
    }

    // Fallback to smallest
    8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_matmul_dims() {
        assert!(validate_matmul_dims(64, 128, 64));
        assert!(!validate_matmul_dims(0, 128, 64));
        assert!(!validate_matmul_dims(64, 0, 64));
        assert!(!validate_matmul_dims(64, 128, 0));
    }

    #[test]
    fn test_optimal_tile_size() {
        assert_eq!(optimal_tile_size(256, 256, 1024), 32);
        assert_eq!(optimal_tile_size(64, 64, 256), 16);
        assert_eq!(optimal_tile_size(16, 16, 64), 8);
    }
}

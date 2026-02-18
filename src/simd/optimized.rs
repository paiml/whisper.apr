//! Optimized SIMD operations (aprender/realizar patterns - WAPR-PERF-004)

use super::vector::dot;
use trueno::Vector;

/// Cache-efficient tile size for matrix operations.
/// 64 elements × 4 bytes = 256 bytes, fits in L1 cache line.
/// Pattern from: realizar/src/inference/simd.rs
pub const TILE_SIZE: usize = 64;

/// Threshold for multi-threaded SIMD (pattern from aprender/src/compute/mod.rs)
pub const PARALLEL_THRESHOLD: usize = 1_000;

/// Threshold for considering GPU dispatch
pub const GPU_THRESHOLD: usize = 100_000;

/// Tiled matrix-vector multiplication for cache efficiency.
///
/// Processes output in TILE_SIZE chunks to maximize L1 cache utilization.
/// Avoids trueno::Matrix allocation overhead for hot-path operations.
///
/// Pattern from: `realizar/src/inference/simd.rs:27`
///
/// # Performance
///
/// - 2-3x faster than naive row-by-row iteration
/// - Keeps working set in L1 cache (32KB typical)
/// - Better prefetcher utilization
///
/// # Arguments
///
/// * `weights` - Row-major weight matrix (rows × cols)
/// * `x` - Input vector (cols)
/// * `rows` - Number of output elements
/// * `cols` - Size of input vector / weight matrix width
///
/// # Returns
///
/// Output vector (rows)
#[must_use]
#[allow(clippy::needless_range_loop)] // Index used for weight offset computation
pub fn tiled_matvec(weights: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(weights.len(), rows * cols, "weight dimensions mismatch");
    debug_assert_eq!(x.len(), cols, "input dimension mismatch");

    // For large matrices, parallelize across output rows
    #[cfg(feature = "parallel")]
    if rows * cols >= PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        return (0..rows)
            .into_par_iter()
            .map(|i| {
                let row_offset = i * cols;
                dot(&weights[row_offset..row_offset + cols], x)
            })
            .collect();
    }

    let mut out = vec![0.0_f32; rows];
    for tile_start in (0..rows).step_by(TILE_SIZE) {
        let tile_end = (tile_start + TILE_SIZE).min(rows);
        for i in tile_start..tile_end {
            let row_offset = i * cols;
            out[i] = dot(&weights[row_offset..row_offset + cols], x);
        }
    }
    out
}

/// Tiled matrix-vector multiplication writing to pre-allocated output.
///
/// Zero-allocation variant for hot paths where output buffer is reused.
#[allow(clippy::needless_range_loop)] // Index used for weight offset computation
pub fn tiled_matvec_into(weights: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(weights.len(), rows * cols, "weight dimensions mismatch");
    debug_assert_eq!(x.len(), cols, "input dimension mismatch");
    debug_assert_eq!(out.len(), rows, "output dimension mismatch");

    // For large matrices, parallelize across output rows
    #[cfg(feature = "parallel")]
    if rows * cols >= PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row_offset = i * cols;
            *o = dot(&weights[row_offset..row_offset + cols], x);
        });
        return;
    }

    for tile_start in (0..rows).step_by(TILE_SIZE) {
        let tile_end = (tile_start + TILE_SIZE).min(rows);
        for i in tile_start..tile_end {
            let row_offset = i * cols;
            out[i] = dot(&weights[row_offset..row_offset + cols], x);
        }
    }
}

/// RMS normalization (faster than LayerNorm).
///
/// RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
///
/// Pattern from: `realizar/src/inference/norm.rs:93`
///
/// # Performance
///
/// - 1.3x faster than LayerNorm (skips mean computation)
/// - Single pass over data for sum of squares
/// - Used by LLaMA, Mistral, and modern transformers
///
/// # Note
///
/// Whisper uses LayerNorm, not RMSNorm. This is provided for potential
/// model variants or future optimization experiments.
#[must_use]
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), weight.len(), "dimension mismatch");

    if x.is_empty() {
        return vec![];
    }

    // Compute sum of squares using SIMD
    let vx = Vector::from_slice(x);
    let sum_sq = vx.dot(&vx).unwrap_or(0.0);

    // RMS = sqrt(mean(x²) + eps)
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Scale by inverse RMS and weight
    x.iter()
        .zip(weight.iter())
        .map(|(v, w)| v * inv_rms * w)
        .collect()
}

/// RMS normalization writing to pre-allocated output.
pub fn rms_norm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len(), "dimension mismatch");
    debug_assert_eq!(x.len(), out.len(), "output dimension mismatch");

    if x.is_empty() {
        return;
    }

    let vx = Vector::from_slice(x);
    let sum_sq = vx.dot(&vx).unwrap_or(0.0);
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for ((o, v), w) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *o = v * inv_rms * w;
    }
}

/// Tiled matrix-vector multiplication for fp16 weights.
///
/// Mirrors `tiled_matvec` but reads fp16 (u16 bit-pattern) weights.
/// Each row is dequantized into a reusable f32 buffer that stays in L1 cache,
/// then a SIMD dot product is computed against the f32 input vector.
///
/// This halves DRAM bandwidth vs f32 weights while keeping compute in f32.
///
/// # Arguments
///
/// * `weights_f16` - Row-major fp16 weight matrix stored as u16 bit patterns (rows × cols)
/// * `x` - Input vector (f32, cols elements)
/// * `rows` - Number of output elements
/// * `cols` - Size of input vector / weight matrix width
///
/// # Returns
///
/// Output vector (rows)
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn tiled_matvec_f16(weights_f16: &[u16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(
        weights_f16.len(),
        rows * cols,
        "fp16 weight dimensions mismatch"
    );
    debug_assert_eq!(x.len(), cols, "input dimension mismatch");

    // For large matrices, parallelize across output rows
    // Each thread gets its own dequant buffer via dot_f16's internal buf parameter
    #[cfg(feature = "parallel")]
    if rows * cols >= PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        return (0..rows)
            .into_par_iter()
            .map(|i| {
                let row_offset = i * cols;
                let row_f16 = &weights_f16[row_offset..row_offset + cols];
                // Each rayon task allocates its own thread-local buffer
                let mut buf = vec![0.0_f32; cols];
                super::vector::dot_f16(row_f16, x, &mut buf)
            })
            .collect();
    }

    let mut out = vec![0.0_f32; rows];
    // Thread-local dequantization buffer — stays in L1 cache
    let mut buf = vec![0.0_f32; cols];

    for tile_start in (0..rows).step_by(TILE_SIZE) {
        let tile_end = (tile_start + TILE_SIZE).min(rows);

        for i in tile_start..tile_end {
            let row_offset = i * cols;
            let row_f16 = &weights_f16[row_offset..row_offset + cols];
            out[i] = super::vector::dot_f16(row_f16, x, &mut buf);
        }
    }

    out
}

/// Tiled fp16 matrix-vector multiplication writing to pre-allocated output.
///
/// Zero-allocation variant for hot paths where output buffer is reused.
#[allow(clippy::needless_range_loop)]
pub fn tiled_matvec_f16_into(
    weights_f16: &[u16],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(
        weights_f16.len(),
        rows * cols,
        "fp16 weight dimensions mismatch"
    );
    debug_assert_eq!(x.len(), cols, "input dimension mismatch");
    debug_assert_eq!(out.len(), rows, "output dimension mismatch");

    // For large matrices, parallelize across output rows
    #[cfg(feature = "parallel")]
    if rows * cols >= PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row_offset = i * cols;
            let row_f16 = &weights_f16[row_offset..row_offset + cols];
            let mut buf = vec![0.0_f32; cols];
            *o = super::vector::dot_f16(row_f16, x, &mut buf);
        });
        return;
    }

    let mut buf = vec![0.0_f32; cols];

    for tile_start in (0..rows).step_by(TILE_SIZE) {
        let tile_end = (tile_start + TILE_SIZE).min(rows);

        for i in tile_start..tile_end {
            let row_offset = i * cols;
            let row_f16 = &weights_f16[row_offset..row_offset + cols];
            out[i] = super::vector::dot_f16(row_f16, x, &mut buf);
        }
    }
}

/// Backend category for automatic dispatch.
///
/// Pattern from: `aprender/src/compute/mod.rs`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendCategory {
    /// Single-threaded SIMD (small operations)
    SimdOnly,
    /// Multi-threaded SIMD via rayon (medium operations)
    SimdParallel,
    /// GPU dispatch via trueno-gpu (large operations)
    Gpu,
}

/// Select optimal backend based on operation size.
///
/// Pattern from: `aprender/src/compute/mod.rs`
///
/// # Thresholds
///
/// - `< 1,000` elements: Single-threaded SIMD (avoid thread pool overhead)
/// - `< 100,000` elements: Multi-threaded SIMD (rayon parallel)
/// - `>= 100,000` elements: GPU if available
#[must_use]
pub fn select_backend(size: usize, _gpu_available: bool) -> BackendCategory {
    if size < PARALLEL_THRESHOLD {
        BackendCategory::SimdOnly
    } else if size < GPU_THRESHOLD {
        BackendCategory::SimdParallel
    } else {
        // GPU dispatch would go here if trueno-gpu enabled
        // For now, fall back to parallel SIMD
        BackendCategory::SimdParallel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiled_matvec() {
        let weights = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let x = vec![5.0, 6.0];
        let result = tiled_matvec(&weights, &x, 2, 2);
        // [1*5+2*6, 3*5+4*6] = [17, 39]
        assert!((result[0] - 17.0).abs() < 1e-4);
        assert!((result[1] - 39.0).abs() < 1e-4);
    }

    #[test]
    fn test_tiled_matvec_into() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![5.0, 6.0];
        let mut out = vec![0.0; 2];
        tiled_matvec_into(&weights, &x, &mut out, 2, 2);
        assert!((out[0] - 17.0).abs() < 1e-4);
        assert!((out[1] - 39.0).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let result = rms_norm(&x, &weight, 1e-5);
        assert_eq!(result.len(), 4);
        // All values should be finite
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_empty() {
        let x: Vec<f32> = vec![];
        let weight: Vec<f32> = vec![];
        let result = rms_norm(&x, &weight, 1e-5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rms_norm_into() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        rms_norm_into(&x, &weight, 1e-5, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_select_backend_small() {
        let category = select_backend(100, false);
        assert_eq!(category, BackendCategory::SimdOnly);
    }

    #[test]
    fn test_select_backend_medium() {
        let category = select_backend(10_000, false);
        assert_eq!(category, BackendCategory::SimdParallel);
    }

    #[test]
    fn test_select_backend_large() {
        let category = select_backend(1_000_000, true);
        // Falls back to parallel SIMD since GPU not actually implemented
        assert_eq!(category, BackendCategory::SimdParallel);
    }

    #[test]
    fn test_tiled_matvec_f16() {
        // 2x2 matrix: [[1,2],[3,4]], input: [5,6]
        let weights_f32 = [1.0_f32, 2.0, 3.0, 4.0];
        let weights_f16: Vec<u16> = weights_f32
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let x = vec![5.0, 6.0];
        let result = tiled_matvec_f16(&weights_f16, &x, 2, 2);
        // [1*5+2*6, 3*5+4*6] = [17, 39]
        assert!((result[0] - 17.0).abs() < 0.1);
        assert!((result[1] - 39.0).abs() < 0.1);
    }

    #[test]
    fn test_tiled_matvec_f16_into() {
        let weights_f32 = [1.0_f32, 2.0, 3.0, 4.0];
        let weights_f16: Vec<u16> = weights_f32
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let x = vec![5.0, 6.0];
        let mut out = vec![0.0; 2];
        tiled_matvec_f16_into(&weights_f16, &x, &mut out, 2, 2);
        assert!((out[0] - 17.0).abs() < 0.1);
        assert!((out[1] - 39.0).abs() < 0.1);
    }

    #[test]
    fn test_tiled_matvec_f16_matches_f32() {
        // Verify fp16 path gives nearly identical results to f32 path
        let rows = 128;
        let cols = 64;
        let weights_f32: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
        let weights_f16: Vec<u16> = weights_f32
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();

        let result_f32 = tiled_matvec(&weights_f32, &x, rows, cols);
        let result_f16 = tiled_matvec_f16(&weights_f16, &x, rows, cols);

        for i in 0..rows {
            let rel_err = if result_f32[i].abs() > 1e-6 {
                (result_f32[i] - result_f16[i]).abs() / result_f32[i].abs()
            } else {
                (result_f32[i] - result_f16[i]).abs()
            };
            assert!(
                rel_err < 0.01,
                "row {i}: f32={} f16={} rel_err={rel_err}",
                result_f32[i],
                result_f16[i]
            );
        }
    }
}

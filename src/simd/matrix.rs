//! SIMD-accelerated matrix operations

use std::cell::RefCell;

use trueno::{Matrix, Vector};

// WAPR-PROFILE-001 Gap 4: Thread-local BlisProfiler for GEMM-level detail
// When enabled, matmul() routes through single-threaded gemm_blis with profiling.
thread_local! {
    static BLIS_PROFILER: RefCell<Option<trueno::blis::BlisProfiler>> = const { RefCell::new(None) };
}

/// Enable BLIS profiling for subsequent matmul calls on this thread.
///
/// WAPR-PROFILE-001 Gap 4: Routes GEMM through single-threaded gemm_blis
/// with BlisProfiler to capture pack/micro/macro hierarchy stats.
pub fn enable_blis_profiling() {
    BLIS_PROFILER.with(|cell| {
        *cell.borrow_mut() = Some(trueno::blis::BlisProfiler::enabled());
    });
}

/// Take the accumulated BlisProfiler stats and disable BLIS profiling.
///
/// Returns the profiler with BLIS hierarchy stats accumulated across all
/// matmul calls since `enable_blis_profiling()`.
pub fn take_blis_profiler() -> Option<trueno::blis::BlisProfiler> {
    BLIS_PROFILER.with(|cell| cell.borrow_mut().take())
}

/// Extract result matrix to Vec, falling back to zeros on error
fn result_to_vec(
    result: Result<Matrix<f32>, trueno::TruenoError>,
    fallback_size: usize,
) -> Vec<f32> {
    result.map_or_else(|_| vec![0.0; fallback_size], |m| m.as_slice().to_vec())
}

/// SIMD-accelerated matrix multiplication
///
/// Computes C = A @ B where A is (rows x inner) and B is (inner x cols).
/// Calls BLIS GEMM directly on raw slices to avoid allocation overhead.
///
/// WAPR-PROFILE-001 Gap 4: When BLIS profiling is enabled via `enable_blis_profiling()`,
/// routes through single-threaded `gemm_blis` with `BlisProfiler` to capture
/// pack/micro/macro hierarchy stats.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn matmul(a: &[f32], b: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * inner, "A dimensions mismatch");
    assert_eq!(b.len(), inner * cols, "B dimensions mismatch");

    let mut c = vec![0.0_f32; rows * cols];

    #[cfg(feature = "webgpu")]
    {
        if std::env::var("WHISPER_USE_WEBGPU").is_ok() {
            use crate::backend::{ComputeOp, MatMulOp};
            let op = MatMulOp::new(rows, inner, cols).with_data(a.to_vec(), b.to_vec());
            if let Ok(res) = op.execute_gpu() {
                return res;
            }
        }
    }

    // Gap 4: Check for active BLIS profiler (single-threaded for accurate profiling)
    let used_profiler = BLIS_PROFILER.with(|cell| {
        let mut opt = cell.borrow_mut();
        if let Some(ref mut profiler) = *opt {
            let _ = trueno::blis::gemm_blis(rows, cols, inner, a, b, &mut c, Some(profiler));
            true
        } else {
            false
        }
    });

    if !used_profiler
        && trueno::blis::parallel::gemm_blis_parallel(rows, cols, inner, a, b, &mut c).is_err()
    {
        return vec![0.0; rows * cols];
    }
    c
}

/// SIMD-accelerated matrix multiplication (zero-copy variant)
///
/// Takes ownership of input vectors to avoid allocation overhead.
/// Use this when you have owned Vecs and won't need them after.
///
/// Computes C = A @ B where A is (rows x inner) and B is (inner x cols)
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn matmul_owned(a: Vec<f32>, b: Vec<f32>, rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * inner, "A dimensions mismatch");
    assert_eq!(b.len(), inner * cols, "B dimensions mismatch");

    let Ok(ma) = Matrix::from_vec(rows, inner, a) else {
        return vec![0.0; rows * cols];
    };
    let Ok(mb) = Matrix::from_vec(inner, cols, b) else {
        return vec![0.0; rows * cols];
    };
    result_to_vec(ma.matmul(&mb), rows * cols)
}

/// SIMD-accelerated matrix multiplication with pre-constructed Matrix
///
/// Uses BLIS GEMM directly on raw slices to avoid copying input A.
/// B's data is accessed via `as_slice()` for zero-copy on the weight side.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn matmul_with_matrix(a: &[f32], b: &Matrix<f32>, rows: usize, inner: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * inner, "A dimensions mismatch");
    assert_eq!(b.rows(), inner, "B rows mismatch inner dimension");

    let cols = b.cols();
    let mut c = vec![0.0_f32; rows * cols];
    // Call BLIS GEMM directly — avoids a.to_vec() allocation for the input matrix
    if trueno::blis::parallel::gemm_blis_parallel(rows, cols, inner, a, b.as_slice(), &mut c)
        .is_err()
    {
        return vec![0.0; rows * cols];
    }
    c
}

/// SIMD-accelerated matrix multiplication with pre-packed B matrix.
///
/// Uses pre-packed B tiles shared across threads, eliminating redundant
/// B packing in parallel GEMM. The PrepackedB must match (inner × cols).
///
/// # WAPR-KAIZEN Cycle 12
///
/// For encoder FFN: 16 threads × 2 GEMMs × 4 layers = 128 B packings eliminated.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn matmul_with_prepacked(
    a: &[f32],
    prepacked_b: &trueno::blis::PrepackedB,
    rows: usize,
    inner: usize,
    cols: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), rows * inner, "A dimensions mismatch");
    assert_eq!(prepacked_b.k, inner, "PrepackedB K mismatch");
    assert_eq!(prepacked_b.n, cols, "PrepackedB N mismatch");

    let mut c = vec![0.0_f32; rows * cols];
    if trueno::blis::parallel::gemm_blis_parallel_with_prepacked_b(
        rows,
        cols,
        inner,
        a,
        prepacked_b,
        &mut c,
    )
    .is_err()
    {
        return vec![0.0; rows * cols];
    }
    c
}

/// SIMD-accelerated matrix-vector multiplication
///
/// Computes y = A @ x where A is (rows x cols) and x is (cols,)
#[must_use]
pub fn matvec(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * cols, "A dimensions mismatch");
    assert_eq!(x.len(), cols, "x dimension mismatch");

    let Ok(ma) = Matrix::from_vec(rows, cols, a.to_vec()) else {
        return vec![0.0; rows];
    };
    let vx = Vector::from_slice(x);
    ma.matvec(&vx)
        .map_or_else(|_| vec![0.0; rows], |v| v.as_slice().to_vec())
}

/// SIMD-accelerated linear projection for raw weight slices
///
/// Computes `output = input @ weight^T + bias` where weight is stored as
/// `[out_features, in_features]` row-major (standard PyTorch linear layer layout).
///
/// This is a convenience wrapper for modules that store weights as `Vec<f32>`
/// (e.g. GQA, MLP) rather than `LinearWeights`. For single-token inference
/// (seq_len=1), uses tiled matvec directly — no transpose or copy overhead.
/// For batch inference, delegates to trueno via transpose + matmul.
#[must_use]
pub fn matmul_raw(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    seq_len: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        seq_len * in_features,
        "input dimensions mismatch"
    );
    assert_eq!(
        weight.len(),
        out_features * in_features,
        "weight dimensions mismatch"
    );

    // Single-token fast path: use tiled_matvec (no transpose, no Matrix construction)
    // Weight is already [out_features, in_features] — perfect for matvec: y = W @ x
    if seq_len == 1 {
        let mut output = super::tiled_matvec(weight, input, out_features, in_features);
        if let Some(b) = bias {
            for (o, &bv) in output.iter_mut().zip(b.iter()) {
                *o += bv;
            }
        }
        return output;
    }

    // Batch path: transpose + matmul via trueno
    // Weight is [out_features, in_features], transpose to [in_features, out_features]
    let weight_t = transpose(weight, out_features, in_features);
    // input [seq_len, in_features] @ weight_t [in_features, out_features] = [seq_len, out_features]
    let mut output = matmul(input, &weight_t, seq_len, in_features, out_features);

    if let Some(b) = bias {
        for s in 0..seq_len {
            for o in 0..out_features {
                output[s * out_features + o] += b[o];
            }
        }
    }

    output
}

/// SIMD-accelerated matrix transpose
#[must_use]
pub fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * cols, "dimensions mismatch");

    let Ok(ma) = Matrix::from_vec(rows, cols, a.to_vec()) else {
        return vec![0.0; rows * cols];
    };
    ma.transpose().as_slice().to_vec()
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

    #[test]
    fn test_matmul_owned() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let result = matmul_owned(a, b, 2, 2, 2);
        assert!(vec_approx_eq(&result, &[19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn test_matmul_owned_rectangular() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let result = matmul_owned(a, b, 2, 3, 2);
        // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!(vec_approx_eq(&result, &[58.0, 64.0, 139.0, 154.0]));
    }

    #[test]
    fn test_matmul_with_matrix() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b_matrix = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0])
            .expect("2x2 matrix creation should succeed");
        let result = matmul_with_matrix(&a, &b_matrix, 2, 2);
        assert!(vec_approx_eq(&result, &[19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn test_matmul_with_matrix_rectangular() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b_matrix = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("3x2 matrix creation should succeed");
        let result = matmul_with_matrix(&a, &b_matrix, 2, 3);
        assert!(vec_approx_eq(&result, &[58.0, 64.0, 139.0, 154.0]));
    }

    #[test]
    fn test_matmul_larger() {
        // 3x4 @ 4x2 = 3x2
        let a: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let result = matmul(&a, &b, 3, 4, 2);
        assert_eq!(result.len(), 6);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_matvec_larger() {
        // 4x3 @ 3 = 4
        let a: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let x = vec![1.0, 2.0, 3.0];
        let result = matvec(&a, &x, 4, 3);
        assert_eq!(result.len(), 4);
        // Row 0: 1*1 + 2*2 + 3*3 = 14
        assert!(approx_eq(result[0], 14.0));
    }

    #[test]
    fn test_transpose_square() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let result = transpose(&a, 2, 2);
        assert!(vec_approx_eq(&result, &[1.0, 3.0, 2.0, 4.0]));
    }

    #[test]
    fn test_transpose_tall() {
        // 3x2 -> 2x3
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose(&a, 3, 2);
        assert!(vec_approx_eq(&result, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_matmul_raw_identity_weight() {
        // Weight = identity [2, 2], input = [1, 2, 3, 4] as 2x2
        // output = input @ I^T = input
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2] identity
        let result = matmul_raw(&input, &weight, None, 2, 2, 2);
        assert!(vec_approx_eq(&result, &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_matmul_raw_with_bias() {
        let input = vec![1.0, 0.0]; // [1, 2]
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2] identity
        let bias = vec![10.0, 20.0];
        let result = matmul_raw(&input, &weight, Some(&bias), 1, 2, 2);
        assert!(vec_approx_eq(&result, &[11.0, 20.0]));
    }

    #[test]
    fn test_matmul_raw_rectangular() {
        // input [2, 3], weight [2, 3] (out_features=2, in_features=3)
        // output = input @ weight^T = [2, 3] @ [3, 2] = [2, 2]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2x3: row0=[1,0,0], row1=[0,1,0]
        let result = matmul_raw(&input, &weight, None, 2, 3, 2);
        // Row 0: [1*1+2*0+3*0, 1*0+2*1+3*0] = [1, 2]
        // Row 1: [4*1+5*0+6*0, 4*0+5*1+6*0] = [4, 5]
        assert!(vec_approx_eq(&result, &[1.0, 2.0, 4.0, 5.0]));
    }

    #[test]
    fn test_matmul_raw_matches_scalar() {
        // Verify matmul_raw matches a scalar reference implementation
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let weight = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [2, 3]
        let bias = vec![0.5, -0.5];

        let result = matmul_raw(&input, &weight, Some(&bias), 2, 3, 2);

        // Scalar reference: output[i,j] = sum_k input[i,k]*weight[j,k] + bias[j]
        let mut expected = vec![0.0f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0f32;
                for k in 0..3 {
                    sum += input[i * 3 + k] * weight[j * 3 + k];
                }
                sum += bias[j];
                expected[i * 2 + j] = sum;
            }
        }

        assert!(vec_approx_eq(&result, &expected));
    }
}

//! Parity of the CUDA operations the GPU encoder is built from against plain Rust
//! reference code (whisper.apr#103). Row counts are deliberately not multiples of 16:
//! the encoder runs 1500 frames. Needs a CUDA device:
//! `TRUENO_FORCE_FP32_GEMM=1 cargo test --features realizar-gpu --test gpu_op_parity -- --ignored`.
#![cfg(feature = "realizar-gpu")]

use trueno_gpu::driver::CudaContext;
use trueno_gpu::memory::resident::{batched_multihead_attention_optimized, GpuResidentTensor};
use whisper_apr::cuda::{conv3_gelu_gpu, conv3_weight_matrix};

/// Deterministic pseudo-random values in [-scale, scale].
fn noise(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0 - 1.0) * scale
        })
        .collect()
}

/// The GELU the CPU encoder uses (`model/encoder/layers.rs`).
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

fn assert_close(op: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{op}: length");
    let (mut worst, mut at, mut bad) = (0.0f32, 0, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs() / w.abs().max(1.0);
        if d > tol || d.is_nan() {
            bad += 1;
        }
        if d > worst || d.is_nan() {
            (worst, at) = (d, i);
        }
    }
    assert!(
        bad == 0,
        "{op}: {bad}/{} values differ; worst {worst} at {at} (got {}, want {})",
        got.len(),
        got[at],
        want[at]
    );
}

/// A device context, for a process started with the switch `WhisperCuda` requires.
fn ctx() -> CudaContext {
    assert!(
        std::env::var("TRUENO_FORCE_FP32_GEMM").is_ok(),
        "run with TRUENO_FORCE_FP32_GEMM=1"
    );
    CudaContext::new(0).expect("CUDA device 0")
}

fn up(ctx: &CudaContext, v: &[f32]) -> GpuResidentTensor<f32> {
    GpuResidentTensor::from_host(ctx, v).expect("upload")
}

const M: usize = 124; // 124 % 16 == 12, like 1500
const K: usize = 128;
const N: usize = 80;

#[test]
#[ignore = "needs a CUDA device"]
fn linear_matches_reference() {
    let ctx = ctx();
    let (a, w, b) = (noise(M * K, 1, 1.0), noise(K * N, 2, 0.2), noise(N, 3, 0.5));
    let mut want = matmul(&a, &w, M, N, K);
    for (i, v) in want.iter_mut().enumerate() {
        *v += b[i % N];
    }
    for round in 0..3 {
        let got = up(&ctx, &a)
            .linear(
                &ctx,
                &up(&ctx, &w),
                Some(&up(&ctx, &b)),
                M as u32,
                K as u32,
                N as u32,
            )
            .and_then(|mut t| t.to_host())
            .expect("linear");
        assert_close(&format!("linear round {round}"), &got, &want, 5e-3);
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn fused_linear_gelu_matches_reference() {
    let ctx = ctx();
    let (a, w, b) = (noise(M * K, 4, 1.0), noise(K * N, 5, 0.2), noise(N, 6, 0.5));
    let mut want = matmul(&a, &w, M, N, K);
    for (i, v) in want.iter_mut().enumerate() {
        *v = gelu(*v + b[i % N]);
    }
    let got = up(&ctx, &a)
        .fused_linear_gelu(
            &ctx,
            &up(&ctx, &w),
            &up(&ctx, &b),
            M as u32,
            K as u32,
            N as u32,
        )
        .and_then(|mut t| t.to_host())
        .expect("fused");
    assert_close("fused_linear_gelu", &got, &want, 5e-3);
}

#[test]
#[ignore = "needs a CUDA device"]
fn layer_norm_matches_reference() {
    let ctx = ctx();
    let (x, g, b) = (noise(M * K, 7, 3.0), noise(K, 8, 1.5), noise(K, 9, 0.5));
    let mut want = x.clone();
    for row in want.chunks_mut(K) {
        let mean = row.iter().sum::<f32>() / K as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / K as f32;
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) / (var + 1e-5).sqrt() * g[i] + b[i];
        }
    }
    let got = up(&ctx, &x)
        .layer_norm(&ctx, &up(&ctx, &g), &up(&ctx, &b), K as u32, M as u32)
        .and_then(|mut t| t.to_host())
        .expect("layer_norm");
    assert_close("layer_norm", &got, &want, 5e-3);
}

#[test]
#[ignore = "needs a CUDA device"]
fn add_matches_reference() {
    let ctx = ctx();
    let (x, y) = (noise(M * K, 10, 2.0), noise(M * K, 11, 2.0));
    let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
    let got = up(&ctx, &x)
        .add(&ctx, &up(&ctx, &y))
        .and_then(|mut t| t.to_host())
        .expect("add");
    assert_close("add", &got, &want, 1e-5);
}

/// `softmax(Q K^T / sqrt(dim)) V` per head, over `[len, heads, dim]` buffers.
fn attention_reference(q: &[f32], k: &[f32], v: &[f32], heads: usize, dim: usize) -> Vec<f32> {
    let row = heads * dim;
    (0..q.len() / dim)
        .flat_map(|i| {
            // `i` counts (frame, head) pairs; `at` is that head's slice of a frame
            let (t, at) = (i / heads, (i % heads) * dim);
            let query = &q[t * row + at..t * row + at + dim];
            let scores: Vec<f32> = k
                .chunks(row)
                .map(|key| dot(query, &key[at..at + dim]) / (dim as f32).sqrt())
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            (0..dim).map(move |d| {
                let mixed = v.chunks(row).zip(&exp).map(|(val, e)| e * val[at + d]);
                mixed.sum::<f32>() / sum
            })
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
#[ignore = "needs a CUDA device"]
fn attention_matches_reference() {
    let ctx = ctx();
    let (heads, dim, len) = (4usize, 64usize, M);
    let size = len * heads * dim;
    let (q, k, v) = (
        noise(size, 12, 1.0),
        noise(size, 13, 1.0),
        noise(size, 14, 1.0),
    );
    let want = attention_reference(&q, &k, &v, heads, dim);
    for round in 0..6 {
        let got = batched_multihead_attention_optimized(
            &ctx,
            &up(&ctx, &q),
            &up(&ctx, &k),
            &up(&ctx, &v),
            heads as u32,
            dim as u32,
            len as u32,
        )
        .and_then(|mut t| t.to_host())
        .expect("attention");
        assert_close(&format!("attention round {round}"), &got, &want, 5e-3);
    }
}

/// Kernel 3, padding 1, then GELU, written out from the definition.
fn conv3_reference(x: &[f32], w: &[f32], b: &[f32], cin: usize, stride: usize) -> Vec<f32> {
    let (len, cout) = (x.len() / cin, b.len());
    let out_len = (len + 2 - 3) / stride + 1;
    (0..out_len * cout)
        .map(|i| {
            let (t, o) = (i / cout, i % cout);
            let taps = (0..cin * 3).map(|n| {
                let (c, tap) = (n / 3, n % 3);
                let src = (t * stride + tap).checked_sub(1).filter(|&s| s < len);
                src.map_or(0.0, |s| x[s * cin + c] * w[(o * cin + c) * 3 + tap])
            });
            gelu(b[o] + taps.sum::<f32>())
        })
        .collect()
}

#[test]
#[ignore = "needs a CUDA device"]
fn conv_frontend_matches_reference() {
    let ctx = ctx();
    // (channels in, channels out, frames, stride): conv1-like, then conv2-like.
    for (cin, cout, len, stride) in [(80, 96, 250, 1), (96, 128, 250, 2), (1280, 64, 100, 2)] {
        let (x, w, b) = (
            noise(len * cin, 15, 1.0),
            noise(cout * cin * 3, 16, 0.02),
            noise(cout, 17, 0.5),
        );
        let matrix = up(&ctx, &conv3_weight_matrix(&w, cin, cout));
        let got = conv3_gelu_gpu(&ctx, &x, &matrix, &up(&ctx, &b), cin, stride)
            .expect("conv")
            .to_host()
            .expect("download");
        let want = conv3_reference(&x, &w, &b, cin, stride);
        assert_close(&format!("conv {cin}x{cout}/{stride}"), &got, &want, 5e-3);
    }
}

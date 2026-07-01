/// Contract: AVX2+FMA dot product — zero-alloc, fused multiply-add for decoder matmul hot path v1.0.0
/// Paper: Intel 64 and IA-32 Architectures Optimization Reference Manual — Section 11.6 FMA
/// Paper: Agner Fog (2024) Instruction Tables — vfmadd231ps: 4-5c latency, 0.5c throughput
pub trait KernelContract {
    /// dot(a, b) = Σ_{i=0}^{n-1} a_i · b_i
    /// Domain: a, b ∈ ℝ^n (f32)
    /// Codomain: ℝ (f32)
    /// INVARIANT: dot(a, b) = dot(b, a) (commutativity)
    /// INVARIANT: dot(α·a, b) = α·dot(a, b) (linearity)
    /// INVARIANT: dot(a, a) ≥ 0 (non-negativity of self-dot)
    /// EQUIVALENCE (SIMD matches scalar): |dot_fma_avx2(a, b) - dot_scalar(a, b)| < tolerance
    /// INVARIANT (Zero-allocation): dot_fma_avx2 performs 0 heap allocations
    /// INVARIANT (Commutativity): |dot(a, b) - dot(b, a)| < ε
    /// BOUND (Self-dot non-negative): dot(a, a) ≥ 0 for all a
    /// INVARIANT (Empty input returns zero): dot([], []) = 0.0
    fn dot_product(&self, input: &[f32], output: &mut [f32]);
    /// acc_k = fma(a_k, b_k, acc_{k-1}) where fma(x,y,z) = RN(x·y+z)
    /// Domain: a_k, b_k ∈ f32, acc_0 = 0
    /// Codomain: f32
    /// INVARIANT: FMA rounds once (not twice as mul+add would)
    /// INVARIANT: 4 independent accumulators hide pipeline latency
    /// INVARIANT: |fma_dot - scalar_dot| ≤ n · ε_mach (different rounding, bounded error)
    /// EQUIVALENCE (SIMD matches scalar): |dot_fma_avx2(a, b) - dot_scalar(a, b)| < tolerance
    /// INVARIANT (Zero-allocation): dot_fma_avx2 performs 0 heap allocations
    /// INVARIANT (Commutativity): |dot(a, b) - dot(b, a)| < ε
    /// BOUND (Self-dot non-negative): dot(a, a) ≥ 0 for all a
    /// INVARIANT (Empty input returns zero): dot([], []) = 0.0
    fn fma_accumulation(&self, input: &[f32], output: &mut [f32]);
}

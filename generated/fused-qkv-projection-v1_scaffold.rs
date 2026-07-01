/// Contract: Fused QKV projection — concatenated weight matrix for single-matvec attention projection v1.0.0
/// Paper: Vaswani et al. (2017) Attention Is All You Need
/// Paper: Whisper decoder: pre-norm transformer with separate Q/K/V weight matrices
pub trait KernelContract {
    /// Fused (1 matvec with concatenated weights):
  W_qkv = [W_q; W_k; W_v]   ∈ ℝ^{3·d_model × d_model}
  b_qkv = [b_q; b_k; b_v]   ∈ ℝ^{3·d_model}
  normed = LayerNorm(x)
  qkv = W_qkv @ normed + b_qkv    (d_model → 3·d_model)
  q = qkv[0..d_model]
  k = qkv[d_model..2·d_model]
  v = qkv[2·d_model..3·d_model]

    /// Domain: x ∈ ℝ^{d_model}
    /// Codomain: q, k, v ∈ ℝ^{d_model} (sliced from ℝ^{3·d_model})
    /// INVARIANT: W_qkv rows [0..d) = W_q rows, [d..2d) = W_k rows, [2d..3d) = W_v rows
    /// INVARIANT: Contiguous memory layout for prefetch-friendly sequential access
    /// EQUIVALENCE (Fused matches separate QKV): |fused_qkv(x) - separate_qkv(x)| < ε element-wise
    /// INVARIANT (Output dimension correct): len(qkv) = 3 * d_model
    /// INVARIANT (Weight concatenation preserves values): W_qkv[i*d..(i+1)*d, :] = W_i for i ∈ {q,k,v}
    /// INVARIANT (Bias concatenation preserves values): b_qkv[i*d..(i+1)*d] = b_i for i ∈ {q,k,v}
    /// INVARIANT (Single matvec call): Exactly one tiled_matvec_f16_into call for Q+K+V combined
    fn fused_qkv(&self, input: &[f32], output: &mut [f32]);
    /// Standard (3 separate matvecs):
  normed = LayerNorm(x)
  q = W_q @ normed + b_q    (d_model → d_model)
  k = W_k @ normed + b_k    (d_model → d_model)
  v = W_v @ normed + b_v    (d_model → d_model)

    /// Domain: x ∈ ℝ^{d_model}
    /// Codomain: q, k, v ∈ ℝ^{d_model}
    /// EQUIVALENCE (Fused matches separate QKV): |fused_qkv(x) - separate_qkv(x)| < ε element-wise
    /// INVARIANT (Output dimension correct): len(qkv) = 3 * d_model
    /// INVARIANT (Weight concatenation preserves values): W_qkv[i*d..(i+1)*d, :] = W_i for i ∈ {q,k,v}
    /// INVARIANT (Bias concatenation preserves values): b_qkv[i*d..(i+1)*d] = b_i for i ∈ {q,k,v}
    /// INVARIANT (Single matvec call): Exactly one tiled_matvec_f16_into call for Q+K+V combined
    fn separate_qkv(&self, input: &[f32], output: &mut [f32]);
}

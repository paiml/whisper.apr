/// Contract: Online softmax — single-pass max+sum via running normalizer (Milakov & Gimelshein 2018) v1.0.0
/// Paper: Milakov & Gimelshein (2018) Online normalizer calculation for softmax
/// Paper: Rabe & Staats (2022) Self-attention Does Not Need O(n²) Memory
pub trait KernelContract {
    /// Online update rule (streaming max + sum_exp):
  Given running state (m_{i-1}, d_{i-1}) and new score x_i:
    m_i = max(m_{i-1}, x_i)
    d_i = d_{i-1} · exp(m_{i-1} - m_i) + exp(x_i - m_i)
Final: softmax(x)_j = exp(x_j - m_n) / d_n

    /// Domain: x ∈ ℝ^n, n ≥ 1
    /// Codomain: σ(x) ∈ (0,1)^n, Σ σ(x)_i = 1
    /// INVARIANT: d_i > 0 for all i (sum of positive exponentials)
    /// INVARIANT: m_i = max(x_1, ..., x_i)
    /// INVARIANT: d_i = Σ_{j=1}^{i} exp(x_j - m_i)
    /// EQUIVALENCE (Online matches standard softmax): |online_softmax(x) - standard_softmax(x)| < ε element-wise
    /// INVARIANT (Output sums to 1): |Σ σ(x)_i - 1.0| < ε
    /// INVARIANT (All outputs strictly positive): σ(x)_i > 0 for all i
    /// MONOTONICITY (Order preservation): x_i > x_j ⟹ σ(x)_i > σ(x)_j
    /// INVARIANT (Shift invariance): softmax(x + c) = softmax(x) for any scalar c
    /// INVARIANT (Two-pass (not three)): Reads scores array exactly twice: once for online max+sum, once for normalize
    fn online_normalizer(&self, input: &[f32], output: &mut [f32]);
    /// σ(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
    /// Domain: x ∈ ℝ^n
    /// Codomain: (0,1)^n
    /// EQUIVALENCE (Online matches standard softmax): |online_softmax(x) - standard_softmax(x)| < ε element-wise
    /// INVARIANT (Output sums to 1): |Σ σ(x)_i - 1.0| < ε
    /// INVARIANT (All outputs strictly positive): σ(x)_i > 0 for all i
    /// MONOTONICITY (Order preservation): x_i > x_j ⟹ σ(x)_i > σ(x)_j
    /// INVARIANT (Shift invariance): softmax(x + c) = softmax(x) for any scalar c
    /// INVARIANT (Two-pass (not three)): Reads scores array exactly twice: once for online max+sum, once for normalize
    fn standard_softmax(&self, input: &[f32], output: &mut [f32]);
}

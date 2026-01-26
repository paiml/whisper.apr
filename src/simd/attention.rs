//! SIMD-accelerated attention operations

use super::activation::softmax;
use super::matrix::{matmul, transpose};
use super::vector::{axpy, dot};

/// SIMD-accelerated scaled dot-product attention
///
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// # Arguments
/// * `query` - Query tensor (seq_len x d_model)
/// * `key` - Key tensor (seq_len x d_model)
/// * `value` - Value tensor (seq_len x d_model)
/// * `seq_len` - Sequence length
/// * `d_head` - Model dimension
/// * `mask` - Optional attention mask (seq_len x seq_len)
pub fn scaled_dot_product_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq_len: usize,
    d_head: usize,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    // WAPR-BENCH-002: Use optimized path for incremental decode (seq_len=1)
    // This is the hot path during token generation
    if seq_len == 1 {
        return scaled_dot_product_attention_single(query, key, value, d_head, mask);
    }

    // Compute kv_len from key dimensions (for cross-attention support)
    let kv_len = key.len() / d_head;
    let scale = 1.0 / (d_head as f32).sqrt();

    // Q @ K^T: (seq_len x d_head) @ (d_head x kv_len) = (seq_len x kv_len)
    let key_t = transpose(key, kv_len, d_head);
    let mut scores = matmul(query, &key_t, seq_len, d_head, kv_len);

    // Scale
    for s in &mut scores {
        *s *= scale;
    }

    // Apply mask by adding mask values to scores
    // Mask values: 0.0 for positions to attend to, NEG_INFINITY for masked positions
    if let Some(m) = mask {
        debug_assert_eq!(m.len(), seq_len * kv_len, "mask dimensions mismatch");
        for (i, &mask_val) in m.iter().enumerate() {
            scores[i] += mask_val;
        }
    }

    // Row-wise softmax
    let mut weights = Vec::with_capacity(scores.len());
    for i in 0..seq_len {
        let start = i * kv_len;
        let end = start + kv_len;
        let row_softmax = softmax(&scores[start..end]);
        weights.extend(row_softmax);
    }

    // Weights @ V: (seq_len x kv_len) @ (kv_len x d_head) = (seq_len x d_head)
    matmul(&weights, value, seq_len, kv_len, d_head)
}

/// Optimized attention for single query (seq_len=1)
///
/// WAPR-BENCH-002: This is the hot path for incremental decoding.
/// Reduces allocations by using dot products instead of matmul for single-row operations.
#[must_use]
pub fn scaled_dot_product_attention_single(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    d_head: usize,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let kv_len = key.len() / d_head;
    let scale = 1.0 / (d_head as f32).sqrt();

    // For seq_len=1, Q @ K^T reduces to dot products
    // Instead of transpose + matmul, compute scores directly
    let mut scores = Vec::with_capacity(kv_len);
    for pos in 0..kv_len {
        let k_start = pos * d_head;
        let score = dot(query, &key[k_start..k_start + d_head]) * scale;
        scores.push(score);
    }

    // Apply mask if present
    if let Some(m) = mask {
        for (i, &mask_val) in m.iter().take(kv_len).enumerate() {
            scores[i] += mask_val;
        }
    }

    // Softmax (single row - reuse scores buffer)
    let weights = softmax(&scores);

    // Weights @ V: for single query, this is weighted sum of value vectors
    // Use SIMD axpy: output += weight * V[pos]
    let mut output = vec![0.0_f32; d_head];
    for (pos, &weight) in weights.iter().enumerate() {
        let v_start = pos * d_head;
        axpy(weight, &value[v_start..v_start + d_head], &mut output);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_dot_product_attention() {
        // Simple 2x4 test
        let query = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // 2x4
        let key = query.clone();
        let value = query.clone();

        let result = scaled_dot_product_attention(&query, &key, &value, 2, 4, None);
        assert_eq!(result.len(), 8);
        // All values should be finite
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_scaled_dot_product_attention_with_mask() {
        let query = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // 2x4
        let key = query.clone();
        let value = query.clone();

        // Causal mask: lower triangular
        let mask = vec![1.0, 0.0, 1.0, 1.0]; // 2x2

        let result = scaled_dot_product_attention(&query, &key, &value, 2, 4, Some(&mask));
        assert_eq!(result.len(), 8);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_scaled_dot_product_attention_single() {
        // Single query test (seq_len=1)
        let query = vec![1.0, 0.0, 0.0, 1.0]; // 1x4
        let key = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // 2x4
        let value = key.clone();

        let result = scaled_dot_product_attention_single(&query, &key, &value, 4, None);
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_scaled_dot_product_attention_single_with_mask() {
        let query = vec![1.0, 0.0, 0.0, 1.0]; // 1x4
        let key = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // 2x4
        let value = key.clone();
        let mask = vec![0.0, f32::NEG_INFINITY]; // mask out second position

        let result = scaled_dot_product_attention_single(&query, &key, &value, 4, Some(&mask));
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&v| v.is_finite()));
    }
}

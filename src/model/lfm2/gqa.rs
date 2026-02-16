//! Grouped Query Attention (GQA) Implementation
//!
//! GQA is a memory-efficient attention variant where multiple query heads
//! share key-value heads. LFM2 uses 32 query heads with 8 KV heads (4:1 ratio).
//!
//! # Memory Savings
//!
//! With GQA ratio of 4:
//! - KV cache reduced by 4x compared to Multi-Head Attention
//! - Enables longer context windows in constrained memory (WASM)
//!
//! # References
//!
//! - "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
//!   https://arxiv.org/abs/2305.13245

use super::rope::RotaryEmbedding;
use crate::error::{WhisperError, WhisperResult};

/// Grouped Query Attention configuration
#[derive(Debug, Clone)]
pub struct GqaConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of query attention heads
    pub num_q_heads: usize,
    /// Number of key-value heads (shared across query head groups)
    pub num_kv_heads: usize,
    /// Head dimension (hidden_size / num_q_heads)
    pub head_dim: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Dropout probability (0.0 = no dropout)
    pub dropout: f32,
    /// Pad head_dim to next multiple of this value for aligned computation.
    /// HuggingFace Moonshine uses `pad_head_dim_to_multiple_of=8` (36→40).
    /// `None` means no padding.
    pub pad_head_dim_to: Option<usize>,
}

impl GqaConfig {
    /// Create config for LFM2-2.6B
    #[must_use]
    pub fn lfm2_2_6b() -> Self {
        Self {
            hidden_size: 2048,
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 64, // 2048 / 32
            causal: true,
            dropout: 0.0,
            pad_head_dim_to: None,
        }
    }

    /// GQA ratio (query heads per KV head)
    #[must_use]
    pub const fn gqa_ratio(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }

    /// Head dimension after padding (rounds up to next multiple of `pad_head_dim_to`)
    #[must_use]
    pub fn padded_head_dim(&self) -> usize {
        match self.pad_head_dim_to {
            Some(multiple) if multiple > 0 => self.head_dim.div_ceil(multiple) * multiple,
            _ => self.head_dim,
        }
    }

    /// Validate configuration
    ///
    /// # Errors
    /// Returns error if configuration is invalid
    pub fn validate(&self) -> WhisperResult<()> {
        if self.num_q_heads % self.num_kv_heads != 0 {
            return Err(WhisperError::Model(format!(
                "num_q_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_q_heads, self.num_kv_heads
            )));
        }
        if self.hidden_size % self.num_q_heads != 0 {
            return Err(WhisperError::Model(format!(
                "hidden_size ({}) must be divisible by num_q_heads ({})",
                self.hidden_size, self.num_q_heads
            )));
        }
        if self.head_dim != self.hidden_size / self.num_q_heads {
            return Err(WhisperError::Model(format!(
                "head_dim ({}) != hidden_size / num_q_heads ({})",
                self.head_dim,
                self.hidden_size / self.num_q_heads
            )));
        }
        Ok(())
    }
}

/// Grouped Query Attention layer
///
/// Implements efficient attention where multiple query heads share KV heads.
#[derive(Debug, Clone)]
pub struct GroupedQueryAttention {
    /// Configuration
    pub config: GqaConfig,
    /// Query projection weights [hidden_size, hidden_size]
    pub w_q: Vec<f32>,
    /// Key projection weights [hidden_size, kv_dim]
    pub w_k: Vec<f32>,
    /// Value projection weights [hidden_size, kv_dim]
    pub w_v: Vec<f32>,
    /// Output projection weights [hidden_size, hidden_size]
    pub w_o: Vec<f32>,
    /// Query bias (optional)
    pub b_q: Option<Vec<f32>>,
    /// Key bias (optional)
    pub b_k: Option<Vec<f32>>,
    /// Value bias (optional)
    pub b_v: Option<Vec<f32>>,
    /// Output bias (optional)
    pub b_o: Option<Vec<f32>>,
}

impl GroupedQueryAttention {
    /// Create new GQA layer with given config
    ///
    /// # Errors
    /// Returns error if config is invalid
    pub fn new(config: GqaConfig) -> WhisperResult<Self> {
        config.validate()?;

        let hidden_size = config.hidden_size;
        let kv_dim = config.num_kv_heads * config.head_dim;

        Ok(Self {
            config,
            w_q: vec![0.0; hidden_size * hidden_size],
            w_k: vec![0.0; hidden_size * kv_dim],
            w_v: vec![0.0; hidden_size * kv_dim],
            w_o: vec![0.0; hidden_size * hidden_size],
            b_q: None,
            b_k: None,
            b_v: None,
            b_o: None,
        })
    }

    /// KV dimension (num_kv_heads * head_dim)
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.config.num_kv_heads * self.config.head_dim
    }

    /// Forward pass through GQA
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [seq_len, hidden_size]
    /// * `seq_len` - Sequence length
    /// * `position_ids` - Position IDs for RoPE (optional)
    /// * `kv_cache` - Optional KV cache for incremental decoding
    /// * `rope` - Optional RoPE embedding for position encoding
    ///
    /// # Returns
    /// Output tensor [seq_len, hidden_size]
    ///
    /// # Errors
    /// Returns error if dimensions are invalid
    #[allow(clippy::too_many_lines)]
    pub fn forward(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
        _position_ids: Option<&[usize]>,
        _kv_cache: Option<&mut KvCache>,
    ) -> WhisperResult<Vec<f32>> {
        self.forward_with_rope(hidden_states, seq_len, None)
    }

    /// Forward pass through GQA with optional RoPE
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [seq_len, hidden_size]
    /// * `seq_len` - Sequence length
    /// * `rope` - Optional RoPE embedding for position encoding
    ///
    /// # Returns
    /// Output tensor [seq_len, hidden_size]
    ///
    /// # Errors
    /// Returns error if dimensions are invalid
    #[allow(clippy::too_many_lines)]
    pub fn forward_with_rope(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
        rope: Option<&RotaryEmbedding>,
    ) -> WhisperResult<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let num_q_heads = self.config.num_q_heads;
        let head_dim = self.config.head_dim;
        let padded_hd = self.config.padded_head_dim();

        // Validate input shape
        if hidden_states.len() != seq_len * hidden_size {
            return Err(WhisperError::Model(format!(
                "hidden_states length {} != seq_len * hidden_size ({})",
                hidden_states.len(),
                seq_len * hidden_size
            )));
        }

        // Project to Q, K, V
        // Q: [seq_len, hidden_size] -> [seq_len, num_q_heads * head_dim]
        let kv_dim = self.kv_dim();
        let (q_raw, k_raw, v_raw) = if cfg!(feature = "simd") {
            let q = crate::simd::matmul_raw(
                hidden_states,
                &self.w_q,
                self.b_q.as_deref(),
                seq_len,
                hidden_size,
                hidden_size,
            );
            let k = crate::simd::matmul_raw(
                hidden_states,
                &self.w_k,
                self.b_k.as_deref(),
                seq_len,
                hidden_size,
                kv_dim,
            );
            let v = crate::simd::matmul_raw(
                hidden_states,
                &self.w_v,
                self.b_v.as_deref(),
                seq_len,
                hidden_size,
                kv_dim,
            );
            (q, k, v)
        } else {
            let q = self.linear(hidden_states, &self.w_q, self.b_q.as_deref(), hidden_size);
            let k = self.linear(hidden_states, &self.w_k, self.b_k.as_deref(), kv_dim);
            let v = self.linear(hidden_states, &self.w_v, self.b_v.as_deref(), kv_dim);
            (q, k, v)
        };

        // K, V: [seq_len, hidden_size] -> [seq_len, num_kv_heads * head_dim]

        // Pad head_dim if needed (e.g., 36→40 for Moonshine)
        let (mut q, mut k, v, q_stride, kv_padded_dim) = if padded_hd == head_dim {
            (q_raw, k_raw, v_raw, hidden_size, kv_dim)
        } else {
            let q_padded = Self::pad_heads(&q_raw, seq_len, num_q_heads, head_dim, padded_hd);
            let k_padded = Self::pad_heads(
                &k_raw,
                seq_len,
                self.config.num_kv_heads,
                head_dim,
                padded_hd,
            );
            let v_padded = Self::pad_heads(
                &v_raw,
                seq_len,
                self.config.num_kv_heads,
                head_dim,
                padded_hd,
            );
            let q_s = num_q_heads * padded_hd;
            let kv_p = self.config.num_kv_heads * padded_hd;
            (q_padded, k_padded, v_padded, q_s, kv_p)
        };

        // Apply RoPE to Q and K if provided (operates on padded dim)
        if let Some(rope_emb) = rope {
            q = rope_emb.forward(&q, seq_len, num_q_heads, 0)?;
            k = rope_emb.forward(&k, seq_len, self.config.num_kv_heads, 0)?;
        }

        // Compute attention using shared GQA attention helper
        let attn_output = self.compute_gqa_attention(
            &q,
            &k,
            &v,
            seq_len,
            seq_len,
            q_stride,
            kv_padded_dim,
            padded_hd,
            head_dim,
            self.config.causal,
        );

        // Strip padding before output projection (padded → original head_dim)
        let attn_unpadded = if padded_hd == head_dim {
            attn_output
        } else {
            Self::unpad_heads(&attn_output, seq_len, num_q_heads, head_dim, padded_hd)
        };

        // Output projection
        let output = if cfg!(feature = "simd") {
            crate::simd::matmul_raw(
                &attn_unpadded,
                &self.w_o,
                self.b_o.as_deref(),
                seq_len,
                hidden_size,
                hidden_size,
            )
        } else {
            self.linear(&attn_unpadded, &self.w_o, self.b_o.as_deref(), hidden_size)
        };

        Ok(output)
    }

    /// Cross-attention forward pass (WAPR-MOONSHINE-003)
    ///
    /// Q is projected from decoder hidden states, K/V from encoder output.
    /// No causal masking. No RoPE (cross-attention spans different position spaces).
    ///
    /// # Arguments
    /// * `decoder_hidden` - Decoder hidden states [dec_seq_len, hidden_size]
    /// * `encoder_output` - Encoder output [enc_seq_len, hidden_size]
    /// * `dec_seq_len` - Decoder sequence length
    /// * `enc_seq_len` - Encoder sequence length
    ///
    /// # Returns
    /// Output tensor [dec_seq_len, hidden_size]
    ///
    /// # Errors
    /// Returns error if dimensions are invalid
    pub fn forward_cross_attention(
        &self,
        decoder_hidden: &[f32],
        encoder_output: &[f32],
        dec_seq_len: usize,
        enc_seq_len: usize,
    ) -> WhisperResult<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let num_q_heads = self.config.num_q_heads;
        let head_dim = self.config.head_dim;
        let padded_hd = self.config.padded_head_dim();
        let kv_dim = self.kv_dim();

        // Validate input shapes
        if decoder_hidden.len() != dec_seq_len * hidden_size {
            return Err(WhisperError::Model(format!(
                "decoder_hidden length {} != dec_seq_len * hidden_size ({})",
                decoder_hidden.len(),
                dec_seq_len * hidden_size
            )));
        }
        if encoder_output.len() != enc_seq_len * hidden_size {
            return Err(WhisperError::Model(format!(
                "encoder_output length {} != enc_seq_len * hidden_size ({})",
                encoder_output.len(),
                enc_seq_len * hidden_size
            )));
        }

        // Q from decoder, K/V from encoder
        let (q_raw, k_raw, v_raw) = if cfg!(feature = "simd") {
            let q = crate::simd::matmul_raw(
                decoder_hidden,
                &self.w_q,
                self.b_q.as_deref(),
                dec_seq_len,
                hidden_size,
                hidden_size,
            );
            let k = crate::simd::matmul_raw(
                encoder_output,
                &self.w_k,
                self.b_k.as_deref(),
                enc_seq_len,
                hidden_size,
                kv_dim,
            );
            let v = crate::simd::matmul_raw(
                encoder_output,
                &self.w_v,
                self.b_v.as_deref(),
                enc_seq_len,
                hidden_size,
                kv_dim,
            );
            (q, k, v)
        } else {
            let q = self.linear(decoder_hidden, &self.w_q, self.b_q.as_deref(), hidden_size);
            let k = self.linear_with_dim(
                encoder_output,
                &self.w_k,
                self.b_k.as_deref(),
                hidden_size,
                kv_dim,
            );
            let v = self.linear_with_dim(
                encoder_output,
                &self.w_v,
                self.b_v.as_deref(),
                hidden_size,
                kv_dim,
            );
            (q, k, v)
        };

        // Pad head_dim if needed
        let (q, k, v, q_stride, kv_padded_dim) = if padded_hd == head_dim {
            (q_raw, k_raw, v_raw, hidden_size, kv_dim)
        } else {
            let q_p = Self::pad_heads(&q_raw, dec_seq_len, num_q_heads, head_dim, padded_hd);
            let k_p = Self::pad_heads(
                &k_raw,
                enc_seq_len,
                self.config.num_kv_heads,
                head_dim,
                padded_hd,
            );
            let v_p = Self::pad_heads(
                &v_raw,
                enc_seq_len,
                self.config.num_kv_heads,
                head_dim,
                padded_hd,
            );
            (
                q_p,
                k_p,
                v_p,
                num_q_heads * padded_hd,
                self.config.num_kv_heads * padded_hd,
            )
        };

        // Compute attention using shared GQA attention helper (no causal mask)
        let attn_output = self.compute_gqa_attention(
            &q,
            &k,
            &v,
            dec_seq_len,
            enc_seq_len,
            q_stride,
            kv_padded_dim,
            padded_hd,
            head_dim,
            false,
        );

        // Strip padding before output projection
        let attn_unpadded = if padded_hd == head_dim {
            attn_output
        } else {
            Self::unpad_heads(&attn_output, dec_seq_len, num_q_heads, head_dim, padded_hd)
        };

        // Output projection
        let output = if cfg!(feature = "simd") {
            crate::simd::matmul_raw(
                &attn_unpadded,
                &self.w_o,
                self.b_o.as_deref(),
                dec_seq_len,
                hidden_size,
                hidden_size,
            )
        } else {
            self.linear(&attn_unpadded, &self.w_o, self.b_o.as_deref(), hidden_size)
        };
        Ok(output)
    }

    /// Linear projection helper
    fn linear(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        out_features: usize,
    ) -> Vec<f32> {
        let seq_len = input.len() / self.config.hidden_size;
        let in_features = self.config.hidden_size;
        let mut output = vec![0.0f32; seq_len * out_features];

        for i in 0..seq_len {
            for j in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..in_features {
                    // Weight layout: [out_features, in_features] (row-major)
                    sum += input[i * in_features + k] * weight[j * in_features + k];
                }
                if let Some(b) = bias {
                    sum += b[j];
                }
                output[i * out_features + j] = sum;
            }
        }

        output
    }

    /// Linear projection with explicit input dimension
    ///
    /// Used for cross-attention where K/V input dimension may differ from hidden_size.
    #[allow(clippy::unused_self)]
    fn linear_with_dim(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let seq_len = input.len() / in_features;
        let mut output = vec![0.0f32; seq_len * out_features];

        for i in 0..seq_len {
            for j in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..in_features {
                    sum += input[i * in_features + k] * weight[j * in_features + k];
                }
                if let Some(b) = bias {
                    sum += b[j];
                }
                output[i * out_features + j] = sum;
            }
        }

        output
    }

    /// Zero-pad each head's dimension from `head_dim` to `padded_hd`
    ///
    /// Input layout: `[seq_len, num_heads * head_dim]` (contiguous)
    /// Output layout: `[seq_len, num_heads * padded_hd]` (zero-filled gaps)
    fn pad_heads(
        data: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        padded_hd: usize,
    ) -> Vec<f32> {
        let in_stride = num_heads * head_dim;
        let out_stride = num_heads * padded_hd;
        let mut out = vec![0.0_f32; seq_len * out_stride];
        for s in 0..seq_len {
            for h in 0..num_heads {
                let src = s * in_stride + h * head_dim;
                let dst = s * out_stride + h * padded_hd;
                out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
                // dst + head_dim .. dst + padded_hd remains zero
            }
        }
        out
    }

    /// Strip padding from each head's dimension back to `head_dim`
    ///
    /// Input layout: `[seq_len, num_heads * padded_hd]`
    /// Output layout: `[seq_len, num_heads * head_dim]`
    fn unpad_heads(
        data: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        padded_hd: usize,
    ) -> Vec<f32> {
        let in_stride = num_heads * padded_hd;
        let out_stride = num_heads * head_dim;
        let mut out = vec![0.0_f32; seq_len * out_stride];
        for s in 0..seq_len {
            for h in 0..num_heads {
                let src = s * in_stride + h * padded_hd;
                let dst = s * out_stride + h * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
            }
        }
        out
    }

    /// Compute GQA attention over pre-projected Q, K, V tensors
    ///
    /// Handles both self-attention (q_seq_len == kv_seq_len, optional causal mask)
    /// and cross-attention (q_seq_len != kv_seq_len, no causal mask).
    /// Routes through SIMD SDPA when the `simd` feature is enabled, otherwise
    /// falls back to a scalar per-head per-position loop.
    #[allow(clippy::too_many_arguments)]
    fn compute_gqa_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_seq_len: usize,
        kv_seq_len: usize,
        q_stride: usize,
        kv_padded_dim: usize,
        padded_hd: usize,
        head_dim: usize,
        causal: bool,
    ) -> Vec<f32> {
        let num_q_heads = self.config.num_q_heads;
        let gqa_ratio = self.config.gqa_ratio();
        let mut attn_output = vec![0.0f32; q_seq_len * q_stride];

        if cfg!(feature = "simd") {
            self.attention_simd(
                q,
                k,
                v,
                &mut attn_output,
                q_seq_len,
                kv_seq_len,
                q_stride,
                kv_padded_dim,
                padded_hd,
                head_dim,
                num_q_heads,
                gqa_ratio,
                causal,
            );
        } else {
            Self::attention_scalar(
                q,
                k,
                v,
                &mut attn_output,
                q_seq_len,
                kv_seq_len,
                q_stride,
                kv_padded_dim,
                padded_hd,
                head_dim,
                num_q_heads,
                gqa_ratio,
                causal,
            );
        }

        attn_output
    }

    /// SIMD attention path: per-head scaled dot-product attention via trueno
    #[allow(clippy::too_many_arguments)]
    fn attention_simd(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        attn_output: &mut [f32],
        q_seq_len: usize,
        kv_seq_len: usize,
        q_stride: usize,
        kv_padded_dim: usize,
        padded_hd: usize,
        head_dim: usize,
        num_q_heads: usize,
        gqa_ratio: usize,
        causal: bool,
    ) {
        let mask = if causal {
            Some(Self::build_causal_mask(q_seq_len))
        } else {
            None
        };

        // Pre-extract K/V per KV head (shared across Q heads in same group)
        let k_heads: Vec<Vec<f32>> = (0..self.config.num_kv_heads)
            .map(|kv_head| Self::extract_head(k, kv_seq_len, kv_padded_dim, kv_head, padded_hd))
            .collect();
        let v_heads: Vec<Vec<f32>> = (0..self.config.num_kv_heads)
            .map(|kv_head| Self::extract_head(v, kv_seq_len, kv_padded_dim, kv_head, padded_hd))
            .collect();

        for q_head in 0..num_q_heads {
            let kv_head = q_head / gqa_ratio;
            let mut q_head_data = Self::extract_head(q, q_seq_len, q_stride, q_head, padded_hd);

            // Correct scaling: SDPA uses 1/sqrt(padded_hd) but we need 1/sqrt(head_dim)
            if padded_hd != head_dim {
                let correction = (padded_hd as f32 / head_dim as f32).sqrt();
                for val in &mut q_head_data {
                    *val *= correction;
                }
            }

            let head_out = crate::simd::scaled_dot_product_attention(
                &q_head_data,
                &k_heads[kv_head],
                &v_heads[kv_head],
                q_seq_len,
                padded_hd,
                mask.as_deref(),
            );

            Self::insert_head(
                attn_output,
                &head_out,
                q_seq_len,
                q_stride,
                q_head,
                padded_hd,
            );
        }
    }

    /// Scalar attention fallback: per-head per-position loop
    #[allow(clippy::too_many_arguments)]
    fn attention_scalar(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        attn_output: &mut [f32],
        q_seq_len: usize,
        kv_seq_len: usize,
        q_stride: usize,
        kv_padded_dim: usize,
        padded_hd: usize,
        head_dim: usize,
        num_q_heads: usize,
        gqa_ratio: usize,
        causal: bool,
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        for q_head in 0..num_q_heads {
            let kv_head = q_head / gqa_ratio;
            for seq_i in 0..q_seq_len {
                let q_offset = seq_i * q_stride + q_head * padded_hd;
                let q_vec = &q[q_offset..q_offset + padded_hd];

                let mut scores = Vec::with_capacity(kv_seq_len);
                for seq_j in 0..kv_seq_len {
                    if causal && seq_j > seq_i {
                        scores.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let k_offset = seq_j * kv_padded_dim + kv_head * padded_hd;
                    let k_vec = &k[k_offset..k_offset + padded_hd];
                    let score: f32 = q_vec.iter().zip(k_vec.iter()).map(|(qi, ki)| qi * ki).sum();
                    scores.push(score * scale);
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let attn_weights: Vec<f32> = if sum_exp > 0.0 {
                    exp_scores.iter().map(|e| e / sum_exp).collect()
                } else {
                    vec![1.0 / kv_seq_len as f32; kv_seq_len]
                };

                let out_offset = seq_i * q_stride + q_head * padded_hd;
                for (seq_j, &weight) in attn_weights.iter().enumerate() {
                    if weight.abs() < 1e-10 {
                        continue;
                    }
                    let v_offset = seq_j * kv_padded_dim + kv_head * padded_hd;
                    for d in 0..padded_hd {
                        attn_output[out_offset + d] += weight * v[v_offset + d];
                    }
                }
            }
        }
    }

    /// Extract contiguous per-head tensor from interleaved multi-head format
    ///
    /// Input layout: `[seq_len, num_heads * head_dim]` (interleaved)
    /// Output layout: `[seq_len, head_dim]` (contiguous for one head)
    fn extract_head(
        data: &[f32],
        seq_len: usize,
        stride: usize,
        head_idx: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(seq_len * head_dim);
        for s in 0..seq_len {
            let offset = s * stride + head_idx * head_dim;
            out.extend_from_slice(&data[offset..offset + head_dim]);
        }
        out
    }

    /// Insert per-head result back into interleaved multi-head format
    fn insert_head(
        output: &mut [f32],
        head_data: &[f32],
        seq_len: usize,
        stride: usize,
        head_idx: usize,
        head_dim: usize,
    ) {
        for s in 0..seq_len {
            let dst = s * stride + head_idx * head_dim;
            let src = s * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&head_data[src..src + head_dim]);
        }
    }

    /// Build causal attention mask `[seq_len, seq_len]`
    ///
    /// `mask[i][j] = 0.0` if j <= i (attend), `NEG_INFINITY` if j > i (masked)
    fn build_causal_mask(seq_len: usize) -> Vec<f32> {
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        mask
    }
}

impl GroupedQueryAttention {
    /// Project a single position to Q, K, V with optional RoPE (incremental decoding)
    ///
    /// For self-attention: applies RoPE to Q and K at the given position offset.
    /// For cross-attention: call with `rope = None`.
    ///
    /// # Arguments
    /// * `hidden` - Single hidden state `[hidden_size]`
    /// * `rope` - Optional RoPE embedding (for self-attention)
    /// * `position` - Position offset for RoPE
    ///
    /// # Returns
    /// `(Q, K, V)` where Q is `[num_q_heads * head_dim]`, K and V are `[kv_dim]`
    ///
    /// # Errors
    /// Returns error if dimensions are invalid or position exceeds RoPE max
    pub fn project_qkv_single(
        &self,
        hidden: &[f32],
        rope: Option<&RotaryEmbedding>,
        position: usize,
    ) -> WhisperResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let hidden_size = self.config.hidden_size;
        let head_dim = self.config.head_dim;
        let padded_hd = self.config.padded_head_dim();

        if hidden.len() != hidden_size {
            return Err(WhisperError::Model(format!(
                "hidden length {} != hidden_size {}",
                hidden.len(),
                hidden_size
            )));
        }

        let kv_dim = self.kv_dim();

        // Project with seq_len=1
        let (q_raw, k_raw, v_raw) = if cfg!(feature = "simd") {
            (
                crate::simd::matmul_raw(
                    hidden,
                    &self.w_q,
                    self.b_q.as_deref(),
                    1,
                    hidden_size,
                    hidden_size,
                ),
                crate::simd::matmul_raw(
                    hidden,
                    &self.w_k,
                    self.b_k.as_deref(),
                    1,
                    hidden_size,
                    kv_dim,
                ),
                crate::simd::matmul_raw(
                    hidden,
                    &self.w_v,
                    self.b_v.as_deref(),
                    1,
                    hidden_size,
                    kv_dim,
                ),
            )
        } else {
            (
                self.linear(hidden, &self.w_q, self.b_q.as_deref(), hidden_size),
                self.linear(hidden, &self.w_k, self.b_k.as_deref(), kv_dim),
                self.linear(hidden, &self.w_v, self.b_v.as_deref(), kv_dim),
            )
        };

        // Pad head_dim if needed
        let (mut q, mut k, v) = if padded_hd == head_dim {
            (q_raw, k_raw, v_raw)
        } else {
            let qp = Self::pad_heads(&q_raw, 1, self.config.num_q_heads, head_dim, padded_hd);
            let kp = Self::pad_heads(&k_raw, 1, self.config.num_kv_heads, head_dim, padded_hd);
            let vp = Self::pad_heads(&v_raw, 1, self.config.num_kv_heads, head_dim, padded_hd);
            (qp, kp, vp)
        };

        // Apply RoPE if provided (self-attention path) — operates on padded dims
        if let Some(rope_emb) = rope {
            q = rope_emb.forward(&q, 1, self.config.num_q_heads, position)?;
            k = rope_emb.forward(&k, 1, self.config.num_kv_heads, position)?;
        }

        Ok((q, k, v))
    }

    /// Compute attention for a single query against cached K/V with GQA expansion
    ///
    /// Each query head maps to a KV head via `kv_head = q_head / gqa_ratio`.
    /// No causal mask needed — single Q always attends to all cached positions.
    ///
    /// # Arguments
    /// * `q` - Query vector `[num_q_heads * head_dim]` (single position)
    /// * `k_cache` - Cached keys `[cache_len * kv_dim]`
    /// * `v_cache` - Cached values `[cache_len * kv_dim]`
    /// * `cache_len` - Number of positions in cache
    ///
    /// # Returns
    /// Attention output `[hidden_size]` (concatenated head outputs)
    ///
    /// # Errors
    /// Returns error if dimensions are invalid
    pub fn attention_cached(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        cache_len: usize,
    ) -> WhisperResult<Vec<f32>> {
        let num_q_heads = self.config.num_q_heads;
        let head_dim = self.config.head_dim;
        let padded_hd = self.config.padded_head_dim();
        let gqa_ratio = self.config.gqa_ratio();

        // Use padded dimensions for Q stride and KV stride
        let q_total = num_q_heads * padded_hd;
        let kv_padded_dim = self.config.num_kv_heads * padded_hd;

        if q.len() != q_total {
            return Err(WhisperError::Model(format!(
                "q length {} != num_q_heads * padded_head_dim ({})",
                q.len(),
                q_total
            )));
        }
        if k_cache.len() != cache_len * kv_padded_dim {
            return Err(WhisperError::Model(format!(
                "k_cache length {} != cache_len * kv_padded_dim ({})",
                k_cache.len(),
                cache_len * kv_padded_dim
            )));
        }

        let mut output = vec![0.0_f32; q_total];

        if cfg!(feature = "simd") {
            // Pre-extract K/V per KV head from cache into contiguous buffers
            let k_heads: Vec<Vec<f32>> = (0..self.config.num_kv_heads)
                .map(|kv_head| {
                    Self::extract_head(k_cache, cache_len, kv_padded_dim, kv_head, padded_hd)
                })
                .collect();
            let v_heads: Vec<Vec<f32>> = (0..self.config.num_kv_heads)
                .map(|kv_head| {
                    Self::extract_head(v_cache, cache_len, kv_padded_dim, kv_head, padded_hd)
                })
                .collect();

            // Scale correction for padded heads (precomputed once)
            let needs_correction = padded_hd != head_dim;
            let correction = if needs_correction {
                (padded_hd as f32 / head_dim as f32).sqrt()
            } else {
                1.0
            };

            for q_head in 0..num_q_heads {
                let kv_head = q_head / gqa_ratio;
                let q_offset = q_head * padded_hd;

                // SIMD single-query attention per head
                let head_out = if needs_correction {
                    let mut q_corrected = q[q_offset..q_offset + padded_hd].to_vec();
                    for val in &mut q_corrected {
                        *val *= correction;
                    }
                    crate::simd::scaled_dot_product_attention_single(
                        &q_corrected,
                        &k_heads[kv_head],
                        &v_heads[kv_head],
                        padded_hd,
                        None,
                    )
                } else {
                    crate::simd::scaled_dot_product_attention_single(
                        &q[q_offset..q_offset + padded_hd],
                        &k_heads[kv_head],
                        &v_heads[kv_head],
                        padded_hd,
                        None,
                    )
                };

                output[q_offset..q_offset + padded_hd].copy_from_slice(&head_out);
            }
        } else {
            // Scalar fallback
            let scale = 1.0 / (head_dim as f32).sqrt();
            for q_head in 0..num_q_heads {
                let kv_head = q_head / gqa_ratio;
                let q_offset = q_head * padded_hd;
                let q_vec = &q[q_offset..q_offset + padded_hd];

                let mut scores = Vec::with_capacity(cache_len);
                for pos in 0..cache_len {
                    let k_offset = pos * kv_padded_dim + kv_head * padded_hd;
                    let k_vec = &k_cache[k_offset..k_offset + padded_hd];
                    let score: f32 = q_vec.iter().zip(k_vec).map(|(qi, ki)| qi * ki).sum();
                    scores.push(score * scale);
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let attn_weights: Vec<f32> = if sum_exp > 0.0 {
                    exp_scores.iter().map(|e| e / sum_exp).collect()
                } else {
                    vec![1.0 / cache_len as f32; cache_len]
                };

                let out_offset = q_head * padded_hd;
                for (pos, &weight) in attn_weights.iter().enumerate() {
                    if weight.abs() < 1e-10 {
                        continue;
                    }
                    let v_offset = pos * kv_padded_dim + kv_head * padded_hd;
                    for d in 0..padded_hd {
                        output[out_offset + d] += weight * v_cache[v_offset + d];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Apply output projection W_o to attention output
    ///
    /// If head_dim padding is active, unpads before projecting.
    ///
    /// # Arguments
    /// * `attn_out` - Concatenated head output `[num_q_heads * padded_head_dim]`
    ///
    /// # Returns
    /// Projected output `[hidden_size]`
    pub fn output_projection(&self, attn_out: &[f32]) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let padded_hd = self.config.padded_head_dim();
        let hidden_size = self.config.hidden_size;
        let input = if padded_hd == head_dim {
            attn_out.to_vec()
        } else {
            Self::unpad_heads(attn_out, 1, self.config.num_q_heads, head_dim, padded_hd)
        };
        if cfg!(feature = "simd") {
            crate::simd::matmul_raw(
                &input,
                &self.w_o,
                self.b_o.as_deref(),
                1,
                hidden_size,
                hidden_size,
            )
        } else {
            self.linear(&input, &self.w_o, self.b_o.as_deref(), hidden_size)
        }
    }

    /// Project encoder output to K, V for cross-attention caching
    ///
    /// Called once per encoder output to populate cross-attention cache.
    /// If head_dim padding is active, returns padded K/V.
    ///
    /// # Arguments
    /// * `encoder_out` - Encoder hidden states `[enc_seq_len * hidden_size]`
    /// * `enc_seq_len` - Encoder sequence length
    ///
    /// # Returns
    /// `(K, V)` both `[enc_seq_len * padded_kv_dim]`
    pub fn project_kv(&self, encoder_out: &[f32], enc_seq_len: usize) -> (Vec<f32>, Vec<f32>) {
        let kv_dim = self.kv_dim();
        let head_dim = self.config.head_dim;
        let padded_hd = self.config.padded_head_dim();
        let hidden_size = self.config.hidden_size;

        let (k_raw, v_raw) = if cfg!(feature = "simd") {
            (
                crate::simd::matmul_raw(
                    encoder_out,
                    &self.w_k,
                    self.b_k.as_deref(),
                    enc_seq_len,
                    hidden_size,
                    kv_dim,
                ),
                crate::simd::matmul_raw(
                    encoder_out,
                    &self.w_v,
                    self.b_v.as_deref(),
                    enc_seq_len,
                    hidden_size,
                    kv_dim,
                ),
            )
        } else {
            (
                self.linear_with_dim(
                    encoder_out,
                    &self.w_k,
                    self.b_k.as_deref(),
                    hidden_size,
                    kv_dim,
                ),
                self.linear_with_dim(
                    encoder_out,
                    &self.w_v,
                    self.b_v.as_deref(),
                    hidden_size,
                    kv_dim,
                ),
            )
        };

        if padded_hd == head_dim {
            (k_raw, v_raw)
        } else {
            let k = Self::pad_heads(
                &k_raw,
                enc_seq_len,
                self.config.num_kv_heads,
                head_dim,
                padded_hd,
            );
            let v = Self::pad_heads(
                &v_raw,
                enc_seq_len,
                self.config.num_kv_heads,
                head_dim,
                padded_hd,
            );
            (k, v)
        }
    }

    /// Project decoder hidden state to Q only (for cross-attention)
    ///
    /// If head_dim padding is active, returns padded Q.
    ///
    /// # Arguments
    /// * `hidden` - Single decoder hidden state `[hidden_size]`
    ///
    /// # Returns
    /// Query vector `[num_q_heads * padded_head_dim]`
    pub fn project_q(&self, hidden: &[f32]) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let padded_hd = self.config.padded_head_dim();
        let hidden_size = self.config.hidden_size;
        let q_raw = if cfg!(feature = "simd") {
            crate::simd::matmul_raw(
                hidden,
                &self.w_q,
                self.b_q.as_deref(),
                1,
                hidden_size,
                hidden_size,
            )
        } else {
            self.linear(hidden, &self.w_q, self.b_q.as_deref(), hidden_size)
        };
        if padded_hd == head_dim {
            q_raw
        } else {
            Self::pad_heads(&q_raw, 1, self.config.num_q_heads, head_dim, padded_hd)
        }
    }
}

/// KV Cache for incremental decoding
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Cached key states [cache_len, num_kv_heads, head_dim]
    pub k_cache: Vec<f32>,
    /// Cached value states [cache_len, num_kv_heads, head_dim]
    pub v_cache: Vec<f32>,
    /// Current cache length
    pub cache_len: usize,
    /// Maximum cache length
    pub max_len: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl KvCache {
    /// Create new KV cache
    #[must_use]
    pub fn new(max_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let size = max_len * num_kv_heads * head_dim;
        Self {
            k_cache: vec![0.0; size],
            v_cache: vec![0.0; size],
            cache_len: 0,
            max_len,
            num_kv_heads,
            head_dim,
        }
    }

    /// Append new KV states to cache
    ///
    /// # Errors
    /// Returns error if cache is full
    pub fn append(&mut self, k: &[f32], v: &[f32], new_len: usize) -> WhisperResult<()> {
        if self.cache_len + new_len > self.max_len {
            return Err(WhisperError::Model(format!(
                "KV cache overflow: {} + {} > {}",
                self.cache_len, new_len, self.max_len
            )));
        }

        let kv_size = self.num_kv_heads * self.head_dim;
        let offset = self.cache_len * kv_size;

        self.k_cache[offset..offset + new_len * kv_size].copy_from_slice(k);
        self.v_cache[offset..offset + new_len * kv_size].copy_from_slice(v);
        self.cache_len += new_len;

        Ok(())
    }

    /// Reset cache to empty
    pub fn reset(&mut self) {
        self.cache_len = 0;
    }

    /// Memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        2 * self.k_cache.len() * std::mem::size_of::<f32>()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gqa_config_lfm2() {
        let config = GqaConfig::lfm2_2_6b();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_q_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.gqa_ratio(), 4);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gqa_config_validation() {
        // Invalid: Q heads not divisible by KV heads
        let config = GqaConfig {
            hidden_size: 2048,
            num_q_heads: 32,
            num_kv_heads: 7, // Not divisible
            head_dim: 64,
            causal: true,
            dropout: 0.0,
            pad_head_dim_to: None,
        };
        assert!(config.validate().is_err());

        // Invalid: hidden_size not divisible by Q heads
        let config = GqaConfig {
            hidden_size: 2000, // Not divisible by 32
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            causal: true,
            dropout: 0.0,
            pad_head_dim_to: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gqa_new() {
        let config = GqaConfig::lfm2_2_6b();
        let gqa = GroupedQueryAttention::new(config).expect("should create GQA");

        assert_eq!(gqa.kv_dim(), 8 * 64); // 512
        assert_eq!(gqa.w_q.len(), 2048 * 2048);
        assert_eq!(gqa.w_k.len(), 2048 * 512);
        assert_eq!(gqa.w_v.len(), 2048 * 512);
        assert_eq!(gqa.w_o.len(), 2048 * 2048);
    }

    #[test]
    fn test_gqa_forward_shape() {
        let config = GqaConfig::lfm2_2_6b();
        let gqa = GroupedQueryAttention::new(config).expect("should create GQA");

        let seq_len = 4;
        let hidden_size = 2048;
        let input = vec![0.1f32; seq_len * hidden_size];

        let output = gqa
            .forward(&input, seq_len, None, None)
            .expect("forward should succeed");

        assert_eq!(output.len(), seq_len * hidden_size);
    }

    #[test]
    fn test_gqa_forward_small() {
        // Small config for faster testing
        let config = GqaConfig {
            hidden_size: 16,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            causal: true,
            dropout: 0.0,
            pad_head_dim_to: None,
        };
        let mut gqa = GroupedQueryAttention::new(config).expect("should create GQA");

        // Initialize with small random-ish weights
        for (i, w) in gqa.w_q.iter_mut().enumerate() {
            *w = ((i % 7) as f32 - 3.0) * 0.1;
        }
        for (i, w) in gqa.w_k.iter_mut().enumerate() {
            *w = ((i % 5) as f32 - 2.0) * 0.1;
        }
        for (i, w) in gqa.w_v.iter_mut().enumerate() {
            *w = ((i % 11) as f32 - 5.0) * 0.1;
        }
        for (i, w) in gqa.w_o.iter_mut().enumerate() {
            *w = ((i % 3) as f32 - 1.0) * 0.1;
        }

        let seq_len = 3;
        let hidden_size = 16;
        let input: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        let output = gqa
            .forward(&input, seq_len, None, None)
            .expect("forward should succeed");

        assert_eq!(output.len(), seq_len * hidden_size);

        // Output should be different from input (transformation occurred)
        let diff: f32 = output
            .iter()
            .zip(input.iter())
            .map(|(o, i)| (o - i).abs())
            .sum();
        assert!(diff > 0.01, "Output should differ from input");
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = KvCache::new(100, 8, 64);
        assert_eq!(cache.cache_len, 0);
        assert_eq!(cache.max_len, 100);
        assert_eq!(cache.num_kv_heads, 8);
        assert_eq!(cache.head_dim, 64);
        assert_eq!(cache.k_cache.len(), 100 * 8 * 64);
    }

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KvCache::new(10, 2, 4);
        let kv_size = 2 * 4;

        let k = vec![1.0f32; kv_size * 3];
        let v = vec![2.0f32; kv_size * 3];

        cache.append(&k, &v, 3).expect("should append");
        assert_eq!(cache.cache_len, 3);

        // Append more
        let k2 = vec![3.0f32; kv_size * 2];
        let v2 = vec![4.0f32; kv_size * 2];
        cache.append(&k2, &v2, 2).expect("should append more");
        assert_eq!(cache.cache_len, 5);
    }

    #[test]
    fn test_kv_cache_overflow() {
        let mut cache = KvCache::new(5, 2, 4);
        let kv_size = 2 * 4;

        let k = vec![1.0f32; kv_size * 6]; // Too many
        let v = vec![2.0f32; kv_size * 6];

        assert!(cache.append(&k, &v, 6).is_err());
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KvCache::new(10, 2, 4);
        let kv_size = 2 * 4;

        let k = vec![1.0f32; kv_size * 5];
        let v = vec![2.0f32; kv_size * 5];
        cache.append(&k, &v, 5).expect("should append");

        cache.reset();
        assert_eq!(cache.cache_len, 0);
    }

    #[test]
    fn test_kv_cache_memory() {
        let cache = KvCache::new(100, 8, 64);
        let expected = 2 * 100 * 8 * 64 * 4; // 2 caches * elements * sizeof(f32)
        assert_eq!(cache.memory_bytes(), expected);
    }

    #[test]
    fn test_gqa_cross_attention_shape() {
        // Moonshine-tiny-like config
        let config = GqaConfig {
            hidden_size: 288,
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 36,
            causal: false,
            dropout: 0.0,
            pad_head_dim_to: None,
        };
        let gqa = GroupedQueryAttention::new(config).expect("should create GQA");

        let dec_seq_len = 3;
        let enc_seq_len = 7;
        let hidden_size = 288;

        let decoder_hidden = vec![0.1f32; dec_seq_len * hidden_size];
        let encoder_output = vec![0.2f32; enc_seq_len * hidden_size];

        let output = gqa
            .forward_cross_attention(&decoder_hidden, &encoder_output, dec_seq_len, enc_seq_len)
            .expect("cross-attention should succeed");

        assert_eq!(output.len(), dec_seq_len * hidden_size);
    }

    #[test]
    fn test_gqa_cross_attention_finite() {
        let config = GqaConfig {
            hidden_size: 16,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            causal: false,
            dropout: 0.0,
            pad_head_dim_to: None,
        };
        let gqa = GroupedQueryAttention::new(config).expect("should create GQA");

        let dec_seq_len = 2;
        let enc_seq_len = 5;
        let hidden_size = 16;

        let decoder_hidden: Vec<f32> = (0..dec_seq_len * hidden_size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let encoder_output: Vec<f32> = (0..enc_seq_len * hidden_size)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();

        let output = gqa
            .forward_cross_attention(&decoder_hidden, &encoder_output, dec_seq_len, enc_seq_len)
            .expect("cross-attention should succeed");

        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gqa_cross_attention_dimension_mismatch() {
        let config = GqaConfig {
            hidden_size: 16,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            causal: false,
            dropout: 0.0,
            pad_head_dim_to: None,
        };
        let gqa = GroupedQueryAttention::new(config).expect("should create GQA");

        // Wrong decoder hidden size
        let decoder_hidden = vec![0.1f32; 3 * 15]; // 15 != 16
        let encoder_output = vec![0.2f32; 5 * 16];

        let result = gqa.forward_cross_attention(&decoder_hidden, &encoder_output, 3, 5);
        assert!(result.is_err());
    }

    // =========================================================================
    // Cached projection method tests (WAPR-MOONSHINE-002)
    // =========================================================================

    /// Helper: create small GQA with deterministic weights for cached tests
    fn make_small_gqa() -> GroupedQueryAttention {
        let config = GqaConfig {
            hidden_size: 16,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            causal: true,
            dropout: 0.0,
            pad_head_dim_to: None,
        };
        let mut gqa = GroupedQueryAttention::new(config).expect("create GQA");
        for (i, w) in gqa.w_q.iter_mut().enumerate() {
            *w = ((i % 7) as f32 - 3.0) * 0.1;
        }
        for (i, w) in gqa.w_k.iter_mut().enumerate() {
            *w = ((i % 5) as f32 - 2.0) * 0.1;
        }
        for (i, w) in gqa.w_v.iter_mut().enumerate() {
            *w = ((i % 11) as f32 - 5.0) * 0.1;
        }
        for (i, w) in gqa.w_o.iter_mut().enumerate() {
            *w = ((i % 3) as f32 - 1.0) * 0.1;
        }
        gqa
    }

    #[test]
    fn test_project_qkv_single_shape() {
        let gqa = make_small_gqa();
        let hidden = vec![0.1_f32; 16];

        let (q, k, v) = gqa
            .project_qkv_single(&hidden, None, 0)
            .expect("project_qkv_single");

        // Q: num_q_heads * head_dim = 4 * 4 = 16
        assert_eq!(q.len(), 16);
        // K, V: num_kv_heads * head_dim = 2 * 4 = 8
        assert_eq!(k.len(), 8);
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_project_qkv_single_with_rope() {
        let gqa = make_small_gqa();
        let rope = RotaryEmbedding::new(crate::model::lfm2::rope::RopeConfig {
            head_dim: 4,
            base: 10000.0,
            max_seq_len: 100,
            rotary_dim: None,
        })
        .expect("rope");

        let hidden = vec![0.5_f32; 16];

        let (q0, k0, _v0) = gqa
            .project_qkv_single(&hidden, Some(&rope), 0)
            .expect("pos 0");
        let (q5, k5, _v5) = gqa
            .project_qkv_single(&hidden, Some(&rope), 5)
            .expect("pos 5");

        // Different positions should yield different Q and K (RoPE rotation)
        let q_diff: f32 = q0.iter().zip(q5.iter()).map(|(a, b)| (a - b).abs()).sum();
        let k_diff: f32 = k0.iter().zip(k5.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(q_diff > 1e-4, "RoPE should change Q across positions");
        assert!(k_diff > 1e-4, "RoPE should change K across positions");
    }

    #[test]
    fn test_project_qkv_single_wrong_size() {
        let gqa = make_small_gqa();
        let hidden = vec![0.1_f32; 15]; // wrong size
        assert!(gqa.project_qkv_single(&hidden, None, 0).is_err());
    }

    #[test]
    fn test_attention_cached_shape() {
        let gqa = make_small_gqa();

        let q = vec![0.1_f32; 16]; // hidden_size
        let kv_dim = 8; // num_kv_heads * head_dim
        let cache_len = 5;
        let k_cache = vec![0.2_f32; cache_len * kv_dim];
        let v_cache = vec![0.3_f32; cache_len * kv_dim];

        let out = gqa
            .attention_cached(&q, &k_cache, &v_cache, cache_len)
            .expect("attention_cached");

        assert_eq!(out.len(), 16); // hidden_size
    }

    #[test]
    fn test_attention_cached_finite() {
        let gqa = make_small_gqa();

        let q: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).sin()).collect();
        let kv_dim = 8;
        let cache_len = 3;
        let k_cache: Vec<f32> = (0..cache_len * kv_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v_cache: Vec<f32> = (0..cache_len * kv_dim)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();

        let out = gqa
            .attention_cached(&q, &k_cache, &v_cache, cache_len)
            .expect("attention_cached");

        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_attention_cached_wrong_q_size() {
        let gqa = make_small_gqa();
        let q = vec![0.1_f32; 15]; // wrong
        let k_cache = vec![0.2_f32; 8];
        let v_cache = vec![0.3_f32; 8];
        assert!(gqa.attention_cached(&q, &k_cache, &v_cache, 1).is_err());
    }

    #[test]
    fn test_output_projection_shape() {
        let gqa = make_small_gqa();
        let attn_out = vec![0.1_f32; 16];
        let projected = gqa.output_projection(&attn_out);
        assert_eq!(projected.len(), 16);
    }

    #[test]
    fn test_project_kv_shape() {
        let config = GqaConfig {
            hidden_size: 288,
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 36,
            causal: false,
            dropout: 0.0,
            pad_head_dim_to: None,
        };
        let gqa = GroupedQueryAttention::new(config).expect("create GQA");

        let enc_seq_len = 7;
        let encoder_out = vec![0.1_f32; enc_seq_len * 288];
        let (k, v) = gqa.project_kv(&encoder_out, enc_seq_len);

        // kv_dim = 2 * 36 = 72
        assert_eq!(k.len(), enc_seq_len * 72);
        assert_eq!(v.len(), enc_seq_len * 72);
    }

    #[test]
    fn test_project_q_shape() {
        let gqa = make_small_gqa();
        let hidden = vec![0.2_f32; 16];
        let q = gqa.project_q(&hidden);
        assert_eq!(q.len(), 16); // hidden_size = num_q_heads * head_dim
    }

    #[test]
    fn test_cached_vs_full_forward_consistency() {
        // Verify that cached single-token attention produces same result as
        // full forward for the same position
        let gqa = make_small_gqa();

        // Single token, no RoPE (position 0 with RoPE is identity anyway)
        let hidden = vec![0.3_f32; 16];

        // Full forward with seq_len=1 (no causal mask effect with 1 token)
        let full_out = gqa.forward(&hidden, 1, None, None).expect("full forward");

        // Cached path: project → attention_cached → output_projection
        let (q, k, v) = gqa.project_qkv_single(&hidden, None, 0).expect("project");
        let attn_out = gqa
            .attention_cached(&q, &k, &v, 1)
            .expect("attention_cached");
        let cached_out = gqa.output_projection(&attn_out);

        // Should match
        for (f, c) in full_out.iter().zip(cached_out.iter()) {
            assert!(
                (f - c).abs() < 1e-5,
                "cached vs full mismatch: {} vs {}",
                f,
                c
            );
        }
    }
}

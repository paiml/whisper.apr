//! Moonshine decoder block
//!
//! Masked MHA self-attention (RoPE) + cross-attention (no RoPE) + SiLU MLP FFN.
//! Pre-RMSNorm before each sub-layer with residual connections.

use crate::error::WhisperResult;
use crate::model::LayerKVCache;
use crate::model::lfm2::gqa::{GqaConfig, GroupedQueryAttention};
use crate::model::lfm2::layer::RmsNorm;
use crate::model::lfm2::mlp::{MlpActivation, MlpConfig, MlpFfn};
use crate::model::lfm2::rope::RotaryEmbedding;

/// Single Moonshine decoder transformer block
///
/// Architecture:
/// 1. Pre-RMSNorm → masked MHA self-attention (with RoPE) → residual
/// 2. Pre-RMSNorm → MHA cross-attention (Q from decoder, KV from encoder) → residual
/// 3. Pre-RMSNorm → SiLU MLP FFN → residual
#[derive(Debug, Clone)]
pub struct MoonshineDecoderBlock {
    /// Pre-self-attention RMS normalization
    pub ln1: RmsNorm,
    /// Masked MHA self-attention (causal, with RoPE)
    pub self_attn: GroupedQueryAttention,
    /// Pre-cross-attention RMS normalization
    pub ln_cross: RmsNorm,
    /// MHA cross-attention (Q from decoder, KV from encoder, no RoPE, no causal mask)
    pub cross_attn: GroupedQueryAttention,
    /// Pre-FFN RMS normalization
    pub ln2: RmsNorm,
    /// Standard MLP feed-forward network (fc1 → SiLU → fc2)
    pub ffn: MlpFfn,
}

impl MoonshineDecoderBlock {
    /// Create a new Moonshine decoder block
    ///
    /// # Arguments
    /// * `d_model` - Hidden dimension (288 for tiny, 416 for base)
    /// * `n_q_heads` - Number of query attention heads (8)
    /// * `n_kv_heads` - Number of key-value heads (8, MHA)
    /// * `intermediate_size` - FFN intermediate dimension (4x d_model)
    ///
    /// # Errors
    /// Returns error if attention or MLP config validation fails
    pub fn new(
        d_model: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        intermediate_size: usize,
    ) -> WhisperResult<Self> {
        let head_dim = d_model / n_q_heads;

        // Self-attention: causal (masked) with RoPE
        let self_attn_config = GqaConfig {
            hidden_size: d_model,
            num_q_heads: n_q_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            causal: true,
            dropout: 0.0,
        };

        // Cross-attention: not causal (decoder attends to all encoder positions)
        let cross_attn_config = GqaConfig {
            hidden_size: d_model,
            num_q_heads: n_q_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            causal: false,
            dropout: 0.0,
        };

        let mlp_config = MlpConfig {
            hidden_size: d_model,
            intermediate_size,
            bias: false,
            activation: MlpActivation::Silu,
        };

        Ok(Self {
            ln1: RmsNorm::new(d_model),
            self_attn: GroupedQueryAttention::new(self_attn_config)?,
            ln_cross: RmsNorm::new(d_model),
            cross_attn: GroupedQueryAttention::new(cross_attn_config)?,
            ln2: RmsNorm::new(d_model),
            ffn: MlpFfn::new(mlp_config)?,
        })
    }

    /// Incremental cached forward pass for a single token (WAPR-MOONSHINE-002)
    ///
    /// Processes only the new token embedding, appending K/V to cache and attending
    /// to all cached positions. O(n) per token instead of O(n²) full recompute.
    ///
    /// # Arguments
    /// * `x` - Single token hidden state `[d_model]`
    /// * `encoder_out` - Encoder output `[enc_seq_len * d_model]`
    /// * `enc_seq_len` - Encoder sequence length
    /// * `position` - Current decode position (for RoPE offset)
    /// * `rope` - Rotary position embedding
    /// * `self_attn_cache` - Self-attention KV cache for this layer (kv_dim width)
    /// * `cross_attn_cache` - Cross-attention KV cache for this layer (kv_dim width)
    /// * `cross_attn_cached` - Whether cross-attention K/V are already cached
    ///
    /// # Returns
    /// Output hidden state `[d_model]`
    ///
    /// # Errors
    /// Returns error if dimensions are invalid or cache overflows
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        x: &[f32],
        encoder_out: &[f32],
        enc_seq_len: usize,
        position: usize,
        rope: &RotaryEmbedding,
        self_attn_cache: &mut LayerKVCache,
        cross_attn_cache: &mut LayerKVCache,
        cross_attn_cached: bool,
    ) -> WhisperResult<Vec<f32>> {
        let d_model = self.self_attn.config.hidden_size;

        // 1. Pre-norm → self-attention with RoPE + KV cache
        let normed = self.ln1.forward(x, 1)?;
        let (q, k_new, v_new) =
            self.self_attn
                .project_qkv_single(&normed, Some(rope), position)?;

        // Append new K, V to self-attention cache
        self_attn_cache.append(&k_new, &v_new)?;

        // Attend to all cached K/V
        let k_full = self_attn_cache.get_key();
        let v_full = self_attn_cache.get_value();
        let cache_len = self_attn_cache.len();
        let attn_out = self.self_attn.attention_cached(&q, k_full, v_full, cache_len)?;
        let attn_out = self.self_attn.output_projection(&attn_out);

        // Residual connection
        let mut residual: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // 2. Pre-norm → cross-attention with cached encoder K/V
        let normed_cross = self.ln_cross.forward(&residual, 1)?;

        let cross_out = if !cross_attn_cached || cross_attn_cache.is_empty() {
            // First token: project encoder output to K/V and cache
            let (k_enc, v_enc) = self.cross_attn.project_kv(encoder_out, enc_seq_len);
            cross_attn_cache.append(&k_enc, &v_enc)?;

            let q_cross = self.cross_attn.project_q(&normed_cross);
            let attn_out = self.cross_attn.attention_cached(
                &q_cross,
                &k_enc,
                &v_enc,
                enc_seq_len,
            )?;
            self.cross_attn.output_projection(&attn_out)
        } else {
            // Subsequent tokens: reuse cached encoder K/V
            let k_cached = cross_attn_cache.get_key();
            let v_cached = cross_attn_cache.get_value();
            let cached_enc_len = cross_attn_cache.len();

            let q_cross = self.cross_attn.project_q(&normed_cross);
            let attn_out = self.cross_attn.attention_cached(
                &q_cross,
                k_cached,
                v_cached,
                cached_enc_len,
            )?;
            self.cross_attn.output_projection(&attn_out)
        };

        // Residual connection
        add_vectors_inplace(&mut residual, &cross_out);

        // 3. Pre-norm → SiLU MLP FFN → residual
        let normed2 = self.ln2.forward(&residual, 1)?;
        let ffn_out = self.ffn.forward(&normed2, 1)?;
        add_vectors_inplace(&mut residual, &ffn_out);

        debug_assert_eq!(residual.len(), d_model);
        Ok(residual)
    }

    /// Forward pass through decoder block
    ///
    /// # Arguments
    /// * `x` - Decoder input tensor [dec_seq_len, d_model]
    /// * `encoder_out` - Encoder output tensor [enc_seq_len, d_model]
    /// * `dec_seq_len` - Decoder sequence length
    /// * `enc_seq_len` - Encoder sequence length
    /// * `rope` - Rotary position embedding (for self-attention only)
    ///
    /// # Returns
    /// Output tensor [dec_seq_len, d_model]
    pub fn forward(
        &self,
        x: &[f32],
        encoder_out: &[f32],
        dec_seq_len: usize,
        enc_seq_len: usize,
        rope: &RotaryEmbedding,
    ) -> WhisperResult<Vec<f32>> {
        // 1. Masked self-attention with RoPE + residual
        let normed = self.ln1.forward(x, dec_seq_len)?;
        let self_attn_out =
            self.self_attn
                .forward_with_rope(&normed, dec_seq_len, Some(rope))?;
        let mut residual = add_vectors(x, &self_attn_out);

        // 2. Cross-attention (Q from decoder, KV from encoder) + residual
        let normed_cross = self.ln_cross.forward(&residual, dec_seq_len)?;
        let cross_attn_out = self.cross_attn.forward_cross_attention(
            &normed_cross,
            encoder_out,
            dec_seq_len,
            enc_seq_len,
        )?;
        add_vectors_inplace(&mut residual, &cross_attn_out);

        // 3. FFN + residual
        let normed2 = self.ln2.forward(&residual, dec_seq_len)?;
        let ffn_out = self.ffn.forward(&normed2, dec_seq_len)?;
        add_vectors_inplace(&mut residual, &ffn_out);

        Ok(residual)
    }
}

/// Element-wise vector addition
fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise in-place vector addition
fn add_vectors_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moonshine_decoder_block_new() {
        // Moonshine tiny: d=288, 8 Q heads, 8 KV heads (MHA), 4x intermediate
        let block = MoonshineDecoderBlock::new(288, 8, 8, 1152);
        assert!(block.is_ok());
    }

    #[test]
    fn test_moonshine_decoder_block_forward_shape() {
        let block = MoonshineDecoderBlock::new(288, 8, 8, 1152).expect("block creation");
        let rope = RotaryEmbedding::new(crate::model::lfm2::rope::RopeConfig {
            head_dim: 36, // 288 / 8
            base: 10000.0,
            max_seq_len: 2048,
        })
        .expect("rope creation");

        let d_model = 288;
        let dec_seq_len = 3;
        let enc_seq_len = 7;

        let decoder_input = vec![0.1_f32; dec_seq_len * d_model];
        let encoder_output = vec![0.2_f32; enc_seq_len * d_model];

        let output = block
            .forward(&decoder_input, &encoder_output, dec_seq_len, enc_seq_len, &rope)
            .expect("forward");
        assert_eq!(output.len(), dec_seq_len * d_model);
    }

    #[test]
    fn test_moonshine_decoder_block_forward_cached_shape() {
        let block = MoonshineDecoderBlock::new(288, 8, 8, 1152).expect("block creation");
        let rope = RotaryEmbedding::new(crate::model::lfm2::rope::RopeConfig {
            head_dim: 36, // 288 / 8
            base: 10000.0,
            max_seq_len: 2048,
        })
        .expect("rope creation");

        let d_model = 288;
        let kv_dim = 8 * 36; // num_kv_heads * head_dim = 288 (MHA: kv_dim == d_model)
        let enc_seq_len = 7;
        let max_tokens = 100;

        let encoder_output = vec![0.2_f32; enc_seq_len * d_model];

        // Simulate 3 decode steps
        let mut self_cache = LayerKVCache::new(kv_dim, max_tokens);
        let mut cross_cache = LayerKVCache::new(kv_dim, max_tokens);

        for pos in 0..3 {
            let x = vec![0.1_f32; d_model];
            let out = block
                .forward_cached(
                    &x,
                    &encoder_output,
                    enc_seq_len,
                    pos,
                    &rope,
                    &mut self_cache,
                    &mut cross_cache,
                    pos > 0,
                )
                .expect("forward_cached");
            assert_eq!(out.len(), d_model);
            assert!(out.iter().all(|v| v.is_finite()));
        }

        assert_eq!(self_cache.len(), 3);
        assert_eq!(cross_cache.len(), enc_seq_len);
    }

    #[test]
    fn test_moonshine_decoder_block_finite_output() {
        let block = MoonshineDecoderBlock::new(288, 8, 8, 1152).expect("block creation");
        let rope = RotaryEmbedding::new(crate::model::lfm2::rope::RopeConfig {
            head_dim: 36,
            base: 10000.0,
            max_seq_len: 2048,
        })
        .expect("rope creation");

        let d_model = 288;
        let dec_seq_len = 1;
        let enc_seq_len = 5;

        let decoder_input = vec![1.0_f32; dec_seq_len * d_model];
        let encoder_output = vec![0.5_f32; enc_seq_len * d_model];

        let output = block
            .forward(&decoder_input, &encoder_output, dec_seq_len, enc_seq_len, &rope)
            .expect("forward");
        assert!(output.iter().all(|v| v.is_finite()));
    }
}

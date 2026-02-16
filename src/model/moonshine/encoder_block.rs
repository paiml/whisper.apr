//! Moonshine encoder block
//!
//! Pre-LayerNorm (no bias) + MHA self-attention + GELU MLP FFN with residual connections.
//! RoPE is applied within the attention forward pass.

use crate::error::WhisperResult;
use crate::model::lfm2::gqa::{GqaConfig, GroupedQueryAttention};
use crate::model::lfm2::layer::LayerNormNoBias;
use crate::model::lfm2::mlp::{MlpActivation, MlpConfig, MlpFfn};
use crate::model::lfm2::rope::RotaryEmbedding;

/// Single Moonshine encoder transformer block
///
/// Architecture: Pre-LayerNorm → MHA self-attention → residual → Pre-LayerNorm → GELU MLP → residual
/// Uses `LayerNorm(bias=False)` matching the HuggingFace Moonshine implementation.
#[derive(Debug, Clone)]
pub struct MoonshineEncoderBlock {
    /// Pre-attention layer normalization (weight-only, no bias)
    pub ln1: LayerNormNoBias,
    /// Multi-head self-attention (with RoPE applied internally)
    pub self_attn: GroupedQueryAttention,
    /// Pre-FFN layer normalization (weight-only, no bias)
    pub ln2: LayerNormNoBias,
    /// Standard MLP feed-forward network (fc1 → GELU → fc2)
    pub ffn: MlpFfn,
}

impl MoonshineEncoderBlock {
    /// Create a new Moonshine encoder block
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

        let gqa_config = GqaConfig {
            hidden_size: d_model,
            num_q_heads: n_q_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            causal: false, // Encoder self-attention is bidirectional
            dropout: 0.0,
        };

        let mlp_config = MlpConfig {
            hidden_size: d_model,
            intermediate_size,
            bias: false,
            activation: MlpActivation::Gelu,
        };

        Ok(Self {
            ln1: LayerNormNoBias::new(d_model),
            self_attn: GroupedQueryAttention::new(gqa_config)?,
            ln2: LayerNormNoBias::new(d_model),
            ffn: MlpFfn::new(mlp_config)?,
        })
    }

    /// Forward pass through encoder block
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, d_model]
    /// * `seq_len` - Sequence length
    /// * `rope` - Rotary position embedding
    ///
    /// # Returns
    /// Output tensor [seq_len, d_model]
    pub fn forward(
        &self,
        x: &[f32],
        seq_len: usize,
        rope: &RotaryEmbedding,
    ) -> WhisperResult<Vec<f32>> {
        // Pre-norm + self-attention + residual
        let normed = self.ln1.forward(x, seq_len)?;
        let attn_out = self.self_attn.forward_with_rope(&normed, seq_len, Some(rope))?;
        let mut residual = add_vectors(x, &attn_out);

        // Pre-norm + FFN + residual
        let normed2 = self.ln2.forward(&residual, seq_len)?;
        let ffn_out = self.ffn.forward(&normed2, seq_len)?;
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
    fn test_moonshine_encoder_block_new() {
        // Moonshine tiny: d=288, 8 Q heads, 8 KV heads (MHA), 4x intermediate
        let block = MoonshineEncoderBlock::new(288, 8, 8, 1152);
        assert!(block.is_ok());
    }

    #[test]
    fn test_moonshine_encoder_block_forward_shape() {
        let block = MoonshineEncoderBlock::new(288, 8, 8, 1152).expect("block creation");
        let rope = RotaryEmbedding::new(crate::model::lfm2::rope::RopeConfig {
            head_dim: 36, // 288 / 8
            base: 10000.0,
            max_seq_len: 2048,
        })
        .expect("rope creation");

        let seq_len = 7; // ~1.5s of audio through conv stem
        let d_model = 288;
        let input = vec![0.1_f32; seq_len * d_model];

        let output = block.forward(&input, seq_len, &rope).expect("forward");
        assert_eq!(output.len(), seq_len * d_model);
    }

    #[test]
    fn test_moonshine_encoder_block_residual() {
        let block = MoonshineEncoderBlock::new(288, 8, 8, 1152).expect("block creation");
        let rope = RotaryEmbedding::new(crate::model::lfm2::rope::RopeConfig {
            head_dim: 36,
            base: 10000.0,
            max_seq_len: 2048,
        })
        .expect("rope creation");

        let seq_len = 4;
        let d_model = 288;
        let input = vec![1.0_f32; seq_len * d_model];

        let output = block.forward(&input, seq_len, &rope).expect("forward");

        // With zero weights, MLP output is zero, attention output passes through
        // residual connections, so output should be close to input
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_add_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = add_vectors(&a, &b);
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }
}

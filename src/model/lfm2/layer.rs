//! LFM2 Layer components
//!
//! Contains the layer-level building blocks for LFM2:
//! - `Lfm2Layer`: Single transformer layer (Conv or Attention)
//! - `RmsNorm`: RMS normalization
//! - `LoadStats`: Statistics for weight loading

use crate::error::{WhisperError, WhisperResult};
use crate::format::apr2::LayerType;

use super::conv::Conv1d;
use super::gqa::GroupedQueryAttention;
use super::rope::RotaryEmbedding;
use super::swiglu::SwiGluFfn;

/// Statistics from loading model weights
#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    /// Number of tensors successfully loaded
    pub tensors_loaded: usize,
    /// Total parameters loaded
    pub params_loaded: usize,
}

impl std::fmt::Display for LoadStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} tensors, {} params",
            self.tensors_loaded, self.params_loaded
        )
    }
}

/// Single LFM2 layer (Conv or Attention)
#[derive(Debug)]
pub struct Lfm2Layer {
    /// Layer index
    pub layer_idx: usize,
    /// Layer type
    pub layer_type: LayerType,
    /// Pre-attention/conv normalization
    pub input_norm: RmsNorm,
    /// Post-attention/conv normalization
    pub post_attn_norm: RmsNorm,
    /// Attention (if attention layer)
    pub attention: Option<GroupedQueryAttention>,
    /// Convolution (if conv layer)
    pub conv: Option<Conv1d>,
    /// Feed-forward network
    pub ffn: SwiGluFfn,
}

impl Lfm2Layer {
    /// Create new layer
    ///
    /// # Errors
    /// Returns error if layer creation fails
    pub fn new(
        layer_idx: usize,
        layer_type: LayerType,
        hidden_size: usize,
        intermediate_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) -> WhisperResult<Self> {
        let input_norm = RmsNorm::new(hidden_size);
        let post_attn_norm = RmsNorm::new(hidden_size);

        let (attention, conv) = match &layer_type {
            LayerType::Attention { use_gqa } => {
                let gqa_config = super::gqa::GqaConfig {
                    hidden_size,
                    num_q_heads,
                    num_kv_heads: if *use_gqa { num_kv_heads } else { num_q_heads },
                    head_dim: hidden_size / num_q_heads,
                    causal: true,
                    dropout: 0.0,
                };
                (Some(GroupedQueryAttention::new(gqa_config)?), None)
            }
            LayerType::Convolution {
                kernel_size,
                cache_len: _,
            } => {
                let conv_config = super::conv::Conv1dConfig {
                    channels: hidden_size,
                    kernel_size: *kernel_size as usize,
                    causal: true,
                    bias: false,
                };
                (None, Some(Conv1d::new_depthwise(conv_config)?))
            }
            LayerType::Ffn { activation: _ } => {
                // FFN-only layer (unusual but supported)
                (None, None)
            }
        };

        let ffn_config = super::swiglu::SwiGluConfig {
            hidden_size,
            intermediate_size,
            bias: false,
        };
        let ffn = SwiGluFfn::new(ffn_config)?;

        Ok(Self {
            layer_idx,
            layer_type,
            input_norm,
            post_attn_norm,
            attention,
            conv,
            ffn,
        })
    }

    /// Forward pass through layer
    ///
    /// # Errors
    /// Returns error if forward fails
    pub fn forward(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
        rope: &RotaryEmbedding,
        _position_ids: Option<&[usize]>,
    ) -> WhisperResult<Vec<f32>> {
        let _hidden_size = hidden_states.len() / seq_len;

        // Pre-norm
        let normed = self.input_norm.forward(hidden_states, seq_len)?;

        // Attention or Conv
        let attn_output = if let Some(ref attn) = self.attention {
            // Apply RoPE to Q, K in attention
            attn.forward_with_rope(&normed, seq_len, Some(rope))?
        } else if let Some(ref conv) = self.conv {
            conv.forward(&normed, seq_len, None)?
        } else {
            normed.clone()
        };

        // Residual connection
        let mut residual: Vec<f32> = hidden_states
            .iter()
            .zip(attn_output.iter())
            .map(|(h, a)| h + a)
            .collect();

        // Post-attention norm
        let normed2 = self.post_attn_norm.forward(&residual, seq_len)?;

        // FFN
        let ffn_output = self.ffn.forward(&normed2, seq_len)?;

        // Residual connection
        for (r, f) in residual.iter_mut().zip(ffn_output.iter()) {
            *r += f;
        }

        Ok(residual)
    }

    /// Number of parameters in this layer
    #[must_use]
    pub fn num_params(&self) -> usize {
        let norm_params = 2 * self.input_norm.weight.len();
        let attn_params = self
            .attention
            .as_ref()
            .map_or(0, |a| a.w_q.len() + a.w_k.len() + a.w_v.len() + a.w_o.len());
        let conv_params = self.conv.as_ref().map_or(0, Conv1d::num_params);
        let ffn_params = self.ffn.num_params();

        norm_params + attn_params + conv_params + ffn_params
    }

    /// Load weights for this layer from APR2 reader
    ///
    /// # Arguments
    /// * `reader` - APR2 reader
    /// * `layer_idx` - Layer index for tensor names
    ///
    /// # Errors
    /// Returns error if weight loading fails
    pub fn load_weights(
        &mut self,
        reader: &crate::format::Apr2Reader,
        layer_idx: usize,
    ) -> WhisperResult<LoadStats> {
        let mut stats = LoadStats::default();

        // Load input layer norm
        let ln1_name = format!("layers.{layer_idx}.ln1.weight");
        if let Ok(weight) = reader.load_tensor_f32(&ln1_name) {
            if weight.len() == self.input_norm.weight.len() {
                self.input_norm.weight = weight;
                stats.tensors_loaded += 1;
                stats.params_loaded += self.input_norm.weight.len();
            }
        }

        // Load post-attention layer norm
        let ln2_name = format!("layers.{layer_idx}.ln2.weight");
        if let Ok(weight) = reader.load_tensor_f32(&ln2_name) {
            if weight.len() == self.post_attn_norm.weight.len() {
                self.post_attn_norm.weight = weight;
                stats.tensors_loaded += 1;
                stats.params_loaded += self.post_attn_norm.weight.len();
            }
        }

        // Load attention weights (if attention layer)
        if let Some(ref mut attn) = self.attention {
            let prefix = format!("layers.{layer_idx}.attn");

            if let Ok(w) = reader.load_tensor_f32(&format!("{prefix}.q.weight")) {
                if w.len() == attn.w_q.len() {
                    attn.w_q = w;
                    stats.tensors_loaded += 1;
                    stats.params_loaded += attn.w_q.len();
                }
            }
            if let Ok(w) = reader.load_tensor_f32(&format!("{prefix}.k.weight")) {
                if w.len() == attn.w_k.len() {
                    attn.w_k = w;
                    stats.tensors_loaded += 1;
                    stats.params_loaded += attn.w_k.len();
                }
            }
            if let Ok(w) = reader.load_tensor_f32(&format!("{prefix}.v.weight")) {
                if w.len() == attn.w_v.len() {
                    attn.w_v = w;
                    stats.tensors_loaded += 1;
                    stats.params_loaded += attn.w_v.len();
                }
            }
            if let Ok(w) = reader.load_tensor_f32(&format!("{prefix}.o.weight")) {
                if w.len() == attn.w_o.len() {
                    attn.w_o = w;
                    stats.tensors_loaded += 1;
                    stats.params_loaded += attn.w_o.len();
                }
            }
        }

        // Load convolution weights (if conv layer)
        if let Some(ref mut conv) = self.conv {
            let conv_name = format!("layers.{layer_idx}.conv.weight");
            if let Ok(weight) = reader.load_tensor_f32(&conv_name) {
                if weight.len() == conv.weight.len() {
                    conv.weight = weight;
                    stats.tensors_loaded += 1;
                    stats.params_loaded += conv.weight.len();
                }
            }
        }

        // Load FFN weights
        let ffn_prefix = format!("layers.{layer_idx}.ffn");

        if let Ok(w) = reader.load_tensor_f32(&format!("{ffn_prefix}.gate.weight")) {
            if w.len() == self.ffn.w_gate.len() {
                self.ffn.w_gate = w;
                stats.tensors_loaded += 1;
                stats.params_loaded += self.ffn.w_gate.len();
            }
        }
        if let Ok(w) = reader.load_tensor_f32(&format!("{ffn_prefix}.up.weight")) {
            if w.len() == self.ffn.w_up.len() {
                self.ffn.w_up = w;
                stats.tensors_loaded += 1;
                stats.params_loaded += self.ffn.w_up.len();
            }
        }
        if let Ok(w) = reader.load_tensor_f32(&format!("{ffn_prefix}.down.weight")) {
            if w.len() == self.ffn.w_down.len() {
                self.ffn.w_down = w;
                stats.tensors_loaded += 1;
                stats.params_loaded += self.ffn.w_down.len();
            }
        }

        Ok(stats)
    }
}

/// RMS Normalization
#[derive(Debug)]
pub struct RmsNorm {
    /// Learnable scale parameter
    pub weight: Vec<f32>,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl RmsNorm {
    /// Create new RMSNorm layer
    #[must_use]
    pub fn new(hidden_size: usize) -> Self {
        Self {
            weight: vec![1.0; hidden_size],
            eps: 1e-5,
        }
    }

    /// Forward pass
    ///
    /// # Errors
    /// Returns error if dimensions are invalid
    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> WhisperResult<Vec<f32>> {
        let hidden_size = self.weight.len();

        if hidden_states.len() != seq_len * hidden_size {
            return Err(WhisperError::Model(format!(
                "hidden_states length {} != seq_len * hidden_size ({})",
                hidden_states.len(),
                seq_len * hidden_size
            )));
        }

        let mut output = vec![0.0f32; hidden_states.len()];

        for s in 0..seq_len {
            let start = s * hidden_size;
            let end = start + hidden_size;
            let x = &hidden_states[start..end];

            // Compute RMS
            let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / hidden_size as f32;
            let rms = (mean_sq + self.eps).sqrt();

            // Normalize and scale
            for (i, &w) in self.weight.iter().enumerate() {
                output[start + i] = (x[i] / rms) * w;
            }
        }

        Ok(output)
    }
}

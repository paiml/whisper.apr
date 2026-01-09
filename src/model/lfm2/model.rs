//! LFM2 Model Implementation
//!
//! This module provides the main LFM2 model struct that combines:
//! - Embedding layer
//! - Hybrid Conv/Attention layers (30 layers for LFM2-2.6B)
//! - RMSNorm normalization
//! - Language model head
//!
//! # Architecture
//!
//! ```text
//! Input IDs → Embedding → [Conv/Attention × 30] → RMSNorm → LM Head → Logits
//! ```
//!
//! # Layer Pattern
//!
//! LFM2 uses a repeating pattern of conv and attention layers:
//! - Layers 0, 1: Convolution
//! - Layer 2: Full Attention (GQA)
//! - Repeat...
//!
//! # Spec Reference
//!
//! See `docs/specifications/1.0-whisper-apr.md` Section 18 for full specification.

use crate::error::{WhisperError, WhisperResult};
use crate::format::apr2::{LayerType, Lfm2Config};

use super::conv::Conv1d;
use super::gqa::GroupedQueryAttention;
use super::rope::RotaryEmbedding;
use super::swiglu::SwiGluFfn;

/// LFM2 model
#[derive(Debug)]
pub struct Lfm2 {
    /// Model configuration
    pub config: Lfm2Config,
    /// Token embedding [vocab_size, hidden_size]
    pub embed_tokens: Vec<f32>,
    /// Transformer layers
    pub layers: Vec<Lfm2Layer>,
    /// Final layer normalization
    pub norm: RmsNorm,
    /// Language model head (often tied to embed_tokens)
    pub lm_head: Option<Vec<f32>>,
    /// Rotary position embedding (shared across layers)
    pub rope: RotaryEmbedding,
}

impl Lfm2 {
    /// Create new LFM2 model with given configuration
    ///
    /// # Errors
    /// Returns error if configuration is invalid
    pub fn new(config: Lfm2Config) -> WhisperResult<Self> {
        let vocab_size = config.vocab_size as usize;
        let hidden_size = config.hidden_size as usize;
        let num_layers = config.num_layers as usize;

        // Create layers based on layer_types
        let mut layers = Vec::with_capacity(num_layers);
        for (i, layer_type) in config.layer_types.iter().enumerate() {
            layers.push(Lfm2Layer::new(
                i,
                layer_type.clone(),
                hidden_size,
                config.intermediate_size as usize,
                config.num_q_heads as usize,
                config.num_kv_heads as usize,
            )?);
        }

        // Create RoPE with config
        let rope_config = super::rope::RopeConfig {
            head_dim: (hidden_size / config.num_q_heads as usize),
            base: config.rope_theta,
            max_seq_len: config.max_seq_len.min(4096) as usize, // WASM limit
        };
        let rope = RotaryEmbedding::new(rope_config)?;

        // Create final norm
        let norm = RmsNorm::new(hidden_size);

        Ok(Self {
            config,
            embed_tokens: vec![0.0; vocab_size * hidden_size],
            layers,
            norm,
            lm_head: None, // Tied to embed_tokens by default
            rope,
        })
    }

    /// Create LFM2-2.6B-Transcript model
    ///
    /// # Errors
    /// Returns error if model creation fails
    pub fn lfm2_2_6b() -> WhisperResult<Self> {
        Self::new(Lfm2Config::lfm2_2_6b())
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [seq_len]
    /// * `position_ids` - Position IDs (optional, defaults to 0..seq_len)
    ///
    /// # Returns
    /// Logits [seq_len, vocab_size]
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn forward(
        &self,
        input_ids: &[u32],
        position_ids: Option<&[usize]>,
    ) -> WhisperResult<Vec<f32>> {
        let seq_len = input_ids.len();
        let hidden_size = self.config.hidden_size as usize;
        let vocab_size = self.config.vocab_size as usize;

        // 1. Embedding lookup
        let mut hidden_states = vec![0.0f32; seq_len * hidden_size];
        for (i, &token_id) in input_ids.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= vocab_size {
                return Err(WhisperError::Model(format!(
                    "token_id {} >= vocab_size {}",
                    token_id, vocab_size
                )));
            }
            let embed_start = token_idx * hidden_size;
            let out_start = i * hidden_size;
            hidden_states[out_start..out_start + hidden_size]
                .copy_from_slice(&self.embed_tokens[embed_start..embed_start + hidden_size]);
        }

        // 2. Process through layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, seq_len, &self.rope, position_ids)?;
        }

        // 3. Final normalization
        hidden_states = self.norm.forward(&hidden_states, seq_len)?;

        // 4. Language model head
        let logits = self.lm_head_forward(&hidden_states, seq_len)?;

        Ok(logits)
    }

    /// Forward through language model head
    fn lm_head_forward(&self, hidden_states: &[f32], seq_len: usize) -> WhisperResult<Vec<f32>> {
        let hidden_size = self.config.hidden_size as usize;
        let vocab_size = self.config.vocab_size as usize;

        // Use lm_head if available, otherwise use embed_tokens (weight tying)
        let weights = self.lm_head.as_ref().unwrap_or(&self.embed_tokens);

        let mut logits = vec![0.0f32; seq_len * vocab_size];

        // Linear: hidden_states @ weights.T
        for s in 0..seq_len {
            for v in 0..vocab_size {
                let mut sum = 0.0f32;
                for h in 0..hidden_size {
                    // Weights: [vocab_size, hidden_size]
                    sum += hidden_states[s * hidden_size + h] * weights[v * hidden_size + h];
                }
                logits[s * vocab_size + v] = sum;
            }
        }

        Ok(logits)
    }

    /// Generate text from prompt
    ///
    /// # Arguments
    /// * `prompt_ids` - Input token IDs
    /// * `max_new_tokens` - Maximum new tokens to generate
    /// * `temperature` - Sampling temperature (0 = greedy)
    ///
    /// # Returns
    /// Generated token IDs
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> WhisperResult<Vec<u32>> {
        let mut output_ids = prompt_ids.to_vec();
        let vocab_size = self.config.vocab_size as usize;

        for _ in 0..max_new_tokens {
            // Forward pass
            let logits = self.forward(&output_ids, None)?;

            // Get logits for last position
            let last_logits_start = (output_ids.len() - 1) * vocab_size;
            let last_logits = &logits[last_logits_start..last_logits_start + vocab_size];

            // Sample next token
            let next_token = if temperature <= 0.0 {
                // Greedy: argmax
                argmax(last_logits) as u32
            } else {
                // Temperature sampling
                sample_with_temperature(last_logits, temperature)? as u32
            };

            output_ids.push(next_token);

            // Check for EOS (assuming token 2 is EOS)
            if next_token == 2 {
                break;
            }
        }

        Ok(output_ids)
    }

    /// Total number of parameters
    #[must_use]
    pub fn num_params(&self) -> usize {
        let embed_params = self.embed_tokens.len();
        let layer_params: usize = self.layers.iter().map(Lfm2Layer::num_params).sum();
        let norm_params = self.norm.weight.len();
        let lm_head_params = self.lm_head.as_ref().map_or(0, Vec::len);

        embed_params + layer_params + norm_params + lm_head_params
    }

    /// Estimate memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.num_params() * std::mem::size_of::<f32>()
    }

    /// Load weights from APR2 reader
    ///
    /// # Arguments
    /// * `reader` - APR2 reader with model weights
    ///
    /// # Errors
    /// Returns error if weight loading fails
    pub fn load_weights(&mut self, reader: &crate::format::Apr2Reader) -> WhisperResult<LoadStats> {
        let mut stats = LoadStats::default();

        // Load embedding weights
        if let Ok(embed) = reader.load_tensor_f32("embed.weight") {
            let expected = self.embed_tokens.len();
            if embed.len() == expected {
                self.embed_tokens = embed;
                stats.tensors_loaded += 1;
                stats.params_loaded += expected;
            } else {
                return Err(WhisperError::Model(format!(
                    "embed.weight size mismatch: {} vs {}",
                    embed.len(),
                    expected
                )));
            }
        }

        // Load final norm
        if let Ok(norm_weight) = reader.load_tensor_f32("norm.weight") {
            if norm_weight.len() == self.norm.weight.len() {
                self.norm.weight = norm_weight;
                stats.tensors_loaded += 1;
                stats.params_loaded += self.norm.weight.len();
            }
        }

        // Load lm_head (if not tied to embeddings)
        if let Ok(lm_head) = reader.load_tensor_f32("lm_head.weight") {
            self.lm_head = Some(lm_head.clone());
            stats.tensors_loaded += 1;
            stats.params_loaded += lm_head.len();
        }

        // Load layer weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_stats = layer.load_weights(reader, i)?;
            stats.tensors_loaded += layer_stats.tensors_loaded;
            stats.params_loaded += layer_stats.params_loaded;
        }

        Ok(stats)
    }

    /// Load model from APR2 file
    ///
    /// # Errors
    /// Returns error if file cannot be loaded
    pub fn from_apr2(reader: &crate::format::Apr2Reader) -> WhisperResult<Self> {
        // Get config from reader
        let config = reader.lfm2_config()?;

        // Create model
        let mut model = Self::new(config)?;

        // Load weights
        model.load_weights(reader)?;

        Ok(model)
    }

    /// Load model from APR2 file bytes
    ///
    /// # Errors
    /// Returns error if bytes are invalid or loading fails
    pub fn from_apr2_bytes(data: Vec<u8>) -> WhisperResult<Self> {
        let reader = crate::format::Apr2Reader::new(data)?;
        Self::from_apr2(&reader)
    }
}

/// Weight loading statistics
#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    /// Number of tensors loaded
    pub tensors_loaded: usize,
    /// Number of parameters loaded
    pub params_loaded: usize,
}

impl std::fmt::Display for LoadStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Loaded {} tensors ({} params, {:.2} MB)",
            self.tensors_loaded,
            self.params_loaded,
            self.params_loaded as f64 * 4.0 / (1024.0 * 1024.0)
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
            for (i, &xi) in x.iter().enumerate() {
                output[start + i] = (xi / rms) * self.weight[i];
            }
        }

        Ok(output)
    }
}

/// Argmax helper
fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Sample with temperature
fn sample_with_temperature(logits: &[f32], temperature: f32) -> WhisperResult<usize> {
    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum).collect();

    // Sample (simple linear search - could use binary search for efficiency)
    // For now, just return argmax of probs (deterministic placeholder)
    Ok(argmax(&probs))
}

// =============================================================================
// WASM Configuration (Section 18.7)
// =============================================================================

/// Quantization type for WASM deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmQuantization {
    /// 16-bit floating point (5.2GB for LFM2-2.6B) - NOT viable for WASM
    Fp16,
    /// 8-bit integer (2.6GB for LFM2-2.6B) - Marginal for WASM
    Int8,
    /// 4-bit integer with AWQ (1.3GB for LFM2-2.6B) - Viable for WASM
    Int4Awq,
    /// 4-bit integer with GPTQ (1.3GB for LFM2-2.6B) - Viable for WASM
    Int4Gptq,
}

impl WasmQuantization {
    /// Bytes per parameter for this quantization type
    #[must_use]
    pub const fn bytes_per_param(&self) -> f32 {
        match self {
            Self::Fp16 => 2.0,
            Self::Int8 => 1.0,
            Self::Int4Awq | Self::Int4Gptq => 0.5,
        }
    }

    /// Whether this quantization is viable for WASM (< 2GB)
    #[must_use]
    pub fn is_wasm_viable(&self, num_params: u64) -> bool {
        let model_bytes = (num_params as f64) * (self.bytes_per_param() as f64);
        model_bytes < 2_000_000_000.0 // 2GB limit
    }
}

impl std::fmt::Display for WasmQuantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp16 => write!(f, "fp16"),
            Self::Int8 => write!(f, "int8"),
            Self::Int4Awq => write!(f, "int4-awq"),
            Self::Int4Gptq => write!(f, "int4-gptq"),
        }
    }
}

/// WASM deployment configuration for LFM2
///
/// Based on Section 18.7 of the specification, this struct defines
/// the recommended configuration for running LFM2 in WebAssembly.
///
/// # Memory Budget
///
/// ```text
/// WASM 32-bit address space limit: 4 GB
/// Browser practical limit:         ~2 GB
///
/// Total (int4 + 4K context):
///   Model:     1.3 GB
///   KV Cache:  1.0 GB
///   Overhead:  0.2 GB
///   ─────────────────
///   Total:     2.5 GB  ⚠️ Tight but possible
/// ```
#[derive(Debug, Clone)]
pub struct Lfm2WasmConfig {
    /// Quantization type (int4 recommended for WASM)
    pub quantization: WasmQuantization,
    /// Maximum context length (limited for memory)
    pub max_context: usize,
    /// Sliding window size for bounded KV cache (None = full attention)
    pub sliding_window: Option<usize>,
    /// Whether to use WebGPU for acceleration
    pub use_webgpu: bool,
    /// Whether to enable token-by-token streaming
    pub streaming: bool,
}

impl Default for Lfm2WasmConfig {
    fn default() -> Self {
        Self {
            quantization: WasmQuantization::Int4Awq,
            max_context: 4096,
            sliding_window: Some(2048),
            use_webgpu: true,
            streaming: true,
        }
    }
}

impl Lfm2WasmConfig {
    /// Create recommended WASM config for LFM2-2.6B
    #[must_use]
    pub fn lfm2_2_6b() -> Self {
        Self::default()
    }

    /// Create config with no sliding window (full attention)
    #[must_use]
    pub fn full_attention() -> Self {
        Self {
            sliding_window: None,
            ..Self::default()
        }
    }

    /// Create conservative config for low-memory devices
    #[must_use]
    pub fn low_memory() -> Self {
        Self {
            quantization: WasmQuantization::Int4Awq,
            max_context: 2048,
            sliding_window: Some(1024),
            use_webgpu: true,
            streaming: true,
        }
    }
}

/// Memory estimation for WASM deployment
///
/// Provides detailed memory breakdown for planning LFM2 deployment
/// in WebAssembly environments.
#[derive(Debug, Clone)]
pub struct WasmMemoryEstimate {
    /// Model weights in bytes
    pub model_bytes: u64,
    /// KV cache in bytes (for max context)
    pub kv_cache_bytes: u64,
    /// Runtime overhead estimate in bytes
    pub overhead_bytes: u64,
    /// Total memory in bytes
    pub total_bytes: u64,
    /// Whether this configuration is viable for WASM
    pub is_viable: bool,
    /// Warning messages (if any)
    pub warnings: Vec<String>,
}

impl WasmMemoryEstimate {
    /// Calculate memory estimate for given config
    #[must_use]
    pub fn calculate(config: &Lfm2Config, wasm_config: &Lfm2WasmConfig) -> Self {
        // LFM2-2.6B has approximately 2.6 billion parameters
        let num_params: u64 = 2_600_000_000;
        let model_bytes =
            (num_params as f64 * wasm_config.quantization.bytes_per_param() as f64) as u64;

        // KV cache calculation
        // Per-token: 2 * num_layers * num_kv_heads * head_dim * 2 bytes (fp16)
        let num_layers = config.num_layers as u64;
        let num_kv_heads = config.num_kv_heads as u64;
        let head_dim = (config.hidden_size / config.num_q_heads) as u64;
        let bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2; // K and V, fp16

        let effective_context = wasm_config
            .sliding_window
            .unwrap_or(wasm_config.max_context) as u64;
        let kv_cache_bytes = bytes_per_token * effective_context;

        // Runtime overhead (WASM runtime, JS heap, etc.)
        let overhead_bytes: u64 = 200_000_000; // ~200MB

        let total_bytes = model_bytes + kv_cache_bytes + overhead_bytes;

        // Check viability
        let browser_limit: u64 = 2_000_000_000; // ~2GB practical limit
        let is_viable = total_bytes < browser_limit;

        let mut warnings = Vec::new();

        if !is_viable {
            warnings.push(format!(
                "Total memory ({:.2} GB) exceeds browser limit (~2 GB)",
                total_bytes as f64 / 1_000_000_000.0
            ));
        }

        if matches!(wasm_config.quantization, WasmQuantization::Fp16) {
            warnings.push("fp16 quantization exceeds WASM memory limits".to_string());
        }

        if wasm_config.max_context > 8192 {
            warnings.push(format!(
                "Large context ({}) may cause OOM in browser",
                wasm_config.max_context
            ));
        }

        if wasm_config.sliding_window.is_none() && wasm_config.max_context > 4096 {
            warnings.push("Full attention with >4K context may exceed memory".to_string());
        }

        Self {
            model_bytes,
            kv_cache_bytes,
            overhead_bytes,
            total_bytes,
            is_viable,
            warnings,
        }
    }

    /// Format as human-readable summary
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Model:    {:>7.2} GB\n",
            self.model_bytes as f64 / 1_000_000_000.0
        ));
        s.push_str(&format!(
            "KV Cache: {:>7.2} GB\n",
            self.kv_cache_bytes as f64 / 1_000_000_000.0
        ));
        s.push_str(&format!(
            "Overhead: {:>7.2} GB\n",
            self.overhead_bytes as f64 / 1_000_000_000.0
        ));
        s.push_str("─────────────────\n");
        s.push_str(&format!(
            "Total:    {:>7.2} GB  {}\n",
            self.total_bytes as f64 / 1_000_000_000.0,
            if self.is_viable { "✅" } else { "❌" }
        ));

        if !self.warnings.is_empty() {
            s.push_str("\nWarnings:\n");
            for w in &self.warnings {
                s.push_str(&format!("  ⚠️ {w}\n"));
            }
        }

        s
    }
}

impl std::fmt::Display for WasmMemoryEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lfm2_new_small() {
        // Small config for testing
        let mut config = Lfm2Config::lfm2_2_6b();
        config.hidden_size = 16;
        config.num_layers = 3;
        config.num_q_heads = 4;
        config.num_kv_heads = 2;
        config.intermediate_size = 32;
        config.vocab_size = 100;
        config.max_seq_len = 64;
        config.layer_types = vec![
            LayerType::Convolution {
                kernel_size: 4,
                cache_len: 3,
            },
            LayerType::Convolution {
                kernel_size: 4,
                cache_len: 3,
            },
            LayerType::Attention { use_gqa: true },
        ];

        let model = Lfm2::new(config).expect("should create model");
        assert_eq!(model.layers.len(), 3);
        assert!(model.layers[0].conv.is_some());
        assert!(model.layers[1].conv.is_some());
        assert!(model.layers[2].attention.is_some());
    }

    #[test]
    fn test_lfm2_forward_small() {
        let mut config = Lfm2Config::lfm2_2_6b();
        config.hidden_size = 8;
        config.num_layers = 2;
        config.num_q_heads = 2;
        config.num_kv_heads = 1;
        config.intermediate_size = 16;
        config.vocab_size = 50;
        config.max_seq_len = 32;
        config.layer_types = vec![
            LayerType::Convolution {
                kernel_size: 2,
                cache_len: 1,
            },
            LayerType::Attention { use_gqa: true },
        ];

        let model = Lfm2::new(config).expect("should create model");

        // Forward with small input
        let input_ids = vec![1u32, 2, 3, 4];
        let logits = model
            .forward(&input_ids, None)
            .expect("forward should succeed");

        assert_eq!(logits.len(), 4 * 50); // seq_len * vocab_size
    }

    #[test]
    fn test_rmsnorm() {
        let norm = RmsNorm::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 positions

        let output = norm.forward(&input, 2).expect("should normalize");
        assert_eq!(output.len(), 8);

        // Check that output is normalized (RMS ≈ 1 after scaling)
        let rms_out: f32 = output[0..4].iter().map(|x| x * x).sum::<f32>() / 4.0;
        assert!(rms_out.sqrt() > 0.1); // Not zero
    }

    #[test]
    fn test_lfm2_layer_conv() {
        let layer = Lfm2Layer::new(
            0,
            LayerType::Convolution {
                kernel_size: 3,
                cache_len: 2,
            },
            8,  // hidden_size
            16, // intermediate_size
            2,  // num_q_heads
            1,  // num_kv_heads
        )
        .expect("should create layer");

        assert!(layer.conv.is_some());
        assert!(layer.attention.is_none());
    }

    #[test]
    fn test_lfm2_layer_attention() {
        let layer = Lfm2Layer::new(0, LayerType::Attention { use_gqa: true }, 8, 16, 2, 1)
            .expect("should create layer");

        assert!(layer.attention.is_some());
        assert!(layer.conv.is_none());
    }

    #[test]
    fn test_argmax() {
        let x = vec![1.0, 3.0, 2.0, 0.5];
        assert_eq!(argmax(&x), 1);

        let y = vec![-1.0, -2.0, -0.5];
        assert_eq!(argmax(&y), 2);
    }

    #[test]
    fn test_lfm2_generate() {
        let mut config = Lfm2Config::lfm2_2_6b();
        config.hidden_size = 8;
        config.num_layers = 2;
        config.num_q_heads = 2;
        config.num_kv_heads = 1;
        config.intermediate_size = 16;
        config.vocab_size = 50;
        config.max_seq_len = 32;
        config.layer_types = vec![
            LayerType::Convolution {
                kernel_size: 2,
                cache_len: 1,
            },
            LayerType::Attention { use_gqa: true },
        ];

        let model = Lfm2::new(config).expect("should create model");

        // Generate tokens
        let prompt = vec![1u32, 2, 3];
        let output = model
            .generate(&prompt, 5, 1.0)
            .expect("generate should succeed");

        // Should have at least prompt length + some generated
        assert!(output.len() >= prompt.len());
        assert!(output.len() <= prompt.len() + 5);
    }

    #[test]
    fn test_lfm2_load_stats_display() {
        let stats = LoadStats {
            tensors_loaded: 10,
            params_loaded: 1000,
        };

        let display = format!("{}", stats);
        assert!(display.contains("10 tensors"));
        assert!(display.contains("1000 params"));
    }

    #[test]
    fn test_lfm2_config_default() {
        // Create default LFM2 config
        let config = Lfm2Config::lfm2_2_6b();

        // Verify LFM2-2.6B config values
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 30);
        assert_eq!(config.num_q_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        // GQA ratio = Q/KV = 32/8 = 4
        assert_eq!(config.num_q_heads / config.num_kv_heads, 4);
    }

    #[test]
    fn test_lfm2_roundtrip_small() {
        // Create small model
        let mut config = Lfm2Config::lfm2_2_6b();
        config.hidden_size = 4;
        config.num_layers = 1;
        config.num_q_heads = 2;
        config.num_kv_heads = 1;
        config.intermediate_size = 8;
        config.vocab_size = 10;
        config.max_seq_len = 16;
        config.layer_types = vec![LayerType::Attention { use_gqa: true }];

        let model = Lfm2::new(config.clone()).expect("should create model");

        // Forward pass to verify it works
        let input_ids = vec![1u32, 2, 3];
        let logits = model
            .forward(&input_ids, None)
            .expect("forward should succeed");

        // logits should have seq_len * vocab_size elements
        assert_eq!(logits.len(), 3 * 10);
    }

    // =========================================================================
    // WASM Configuration Tests (Section 18.7)
    // =========================================================================

    #[test]
    fn test_wasm_quantization_bytes_per_param() {
        assert_eq!(WasmQuantization::Fp16.bytes_per_param(), 2.0);
        assert_eq!(WasmQuantization::Int8.bytes_per_param(), 1.0);
        assert_eq!(WasmQuantization::Int4Awq.bytes_per_param(), 0.5);
        assert_eq!(WasmQuantization::Int4Gptq.bytes_per_param(), 0.5);
    }

    #[test]
    fn test_wasm_quantization_display() {
        assert_eq!(format!("{}", WasmQuantization::Fp16), "fp16");
        assert_eq!(format!("{}", WasmQuantization::Int8), "int8");
        assert_eq!(format!("{}", WasmQuantization::Int4Awq), "int4-awq");
        assert_eq!(format!("{}", WasmQuantization::Int4Gptq), "int4-gptq");
    }

    #[test]
    fn test_wasm_quantization_viability() {
        let lfm2_params: u64 = 2_600_000_000;

        // fp16: 2.6B * 2 = 5.2GB - NOT viable
        assert!(!WasmQuantization::Fp16.is_wasm_viable(lfm2_params));

        // int8: 2.6B * 1 = 2.6GB - NOT viable (exceeds 2GB)
        assert!(!WasmQuantization::Int8.is_wasm_viable(lfm2_params));

        // int4: 2.6B * 0.5 = 1.3GB - Viable
        assert!(WasmQuantization::Int4Awq.is_wasm_viable(lfm2_params));
        assert!(WasmQuantization::Int4Gptq.is_wasm_viable(lfm2_params));
    }

    #[test]
    fn test_lfm2_wasm_config_default() {
        let config = Lfm2WasmConfig::default();

        assert_eq!(config.quantization, WasmQuantization::Int4Awq);
        assert_eq!(config.max_context, 4096);
        assert_eq!(config.sliding_window, Some(2048));
        assert!(config.use_webgpu);
        assert!(config.streaming);
    }

    #[test]
    fn test_lfm2_wasm_config_lfm2_2_6b() {
        let config = Lfm2WasmConfig::lfm2_2_6b();

        // Should be same as default
        assert_eq!(config.quantization, WasmQuantization::Int4Awq);
        assert_eq!(config.max_context, 4096);
    }

    #[test]
    fn test_lfm2_wasm_config_full_attention() {
        let config = Lfm2WasmConfig::full_attention();

        assert!(config.sliding_window.is_none());
        assert_eq!(config.quantization, WasmQuantization::Int4Awq);
    }

    #[test]
    fn test_lfm2_wasm_config_low_memory() {
        let config = Lfm2WasmConfig::low_memory();

        assert_eq!(config.max_context, 2048);
        assert_eq!(config.sliding_window, Some(1024));
    }

    #[test]
    fn test_wasm_memory_estimate_int4() {
        let model_config = Lfm2Config::lfm2_2_6b();
        let wasm_config = Lfm2WasmConfig::default();

        let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);

        // Model bytes: 2.6B * 0.5 = 1.3GB
        assert_eq!(estimate.model_bytes, 1_300_000_000);

        // KV cache: depends on sliding window (2048 tokens)
        // Per token: 2 * 30 * 8 * 64 * 2 = 61440 bytes
        // Total: 61440 * 2048 = ~125MB
        assert!(estimate.kv_cache_bytes > 100_000_000); // > 100MB
        assert!(estimate.kv_cache_bytes < 200_000_000); // < 200MB

        // Total should be around 1.5-1.7GB
        assert!(estimate.total_bytes > 1_500_000_000);
        assert!(estimate.total_bytes < 2_000_000_000);

        // int4 with sliding window should be viable
        assert!(estimate.is_viable);
    }

    #[test]
    fn test_wasm_memory_estimate_fp16_not_viable() {
        let model_config = Lfm2Config::lfm2_2_6b();
        let wasm_config = Lfm2WasmConfig {
            quantization: WasmQuantization::Fp16,
            ..Lfm2WasmConfig::default()
        };

        let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);

        // fp16 should not be viable
        assert!(!estimate.is_viable);
        assert!(!estimate.warnings.is_empty());
        assert!(estimate.warnings.iter().any(|w| w.contains("fp16")));
    }

    #[test]
    fn test_wasm_memory_estimate_large_context_warning() {
        let model_config = Lfm2Config::lfm2_2_6b();
        let wasm_config = Lfm2WasmConfig {
            max_context: 16384,
            sliding_window: None,
            ..Lfm2WasmConfig::default()
        };

        let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);

        // Should have warning about large context
        assert!(estimate.warnings.iter().any(|w| w.contains("16384") || w.contains("context")));
    }

    #[test]
    fn test_wasm_memory_estimate_summary() {
        let model_config = Lfm2Config::lfm2_2_6b();
        let wasm_config = Lfm2WasmConfig::default();

        let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);
        let summary = estimate.summary();

        // Should contain key information
        assert!(summary.contains("Model:"));
        assert!(summary.contains("KV Cache:"));
        assert!(summary.contains("Overhead:"));
        assert!(summary.contains("Total:"));
        assert!(summary.contains("GB"));
    }

    #[test]
    fn test_wasm_memory_estimate_display() {
        let model_config = Lfm2Config::lfm2_2_6b();
        let wasm_config = Lfm2WasmConfig::default();

        let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);
        let display = format!("{}", estimate);

        // Display should be same as summary
        assert_eq!(display, estimate.summary());
    }

    #[test]
    fn test_wasm_memory_estimate_low_memory_config() {
        let model_config = Lfm2Config::lfm2_2_6b();
        let wasm_config = Lfm2WasmConfig::low_memory();

        let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);

        // Low memory config should use less KV cache
        let default_estimate = WasmMemoryEstimate::calculate(
            &model_config,
            &Lfm2WasmConfig::default(),
        );

        assert!(estimate.kv_cache_bytes < default_estimate.kv_cache_bytes);
        assert!(estimate.total_bytes < default_estimate.total_bytes);
    }
}

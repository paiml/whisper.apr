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
use crate::format::apr2::Lfm2Config;

use super::layer::{Lfm2Layer, LoadStats, RmsNorm};
use super::rope::RotaryEmbedding;

#[cfg(test)]
mod tests;

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

/// Generation statistics for streaming output
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Total tokens generated (excluding prompt)
    pub tokens_generated: usize,
    /// Time per token in milliseconds
    pub ms_per_token: f64,
    /// Total generation time in milliseconds
    pub total_ms: f64,
    /// Tokens per second
    pub tokens_per_sec: f64,
    /// Whether generation hit EOS
    pub hit_eos: bool,
}

impl std::fmt::Display for GenerationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} tokens in {:.1}ms ({:.1} tok/s, {:.1}ms/tok)",
            self.tokens_generated, self.total_ms, self.tokens_per_sec, self.ms_per_token
        )
    }
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
            rotary_dim: None,
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
        // Handle empty input
        if input_ids.is_empty() {
            return Ok(Vec::new());
        }

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

    /// Generate tokens with streaming callback
    ///
    /// This method calls the callback for each generated token, enabling
    /// real-time output display and early stopping.
    ///
    /// # Arguments
    /// * `prompt_ids` - Input token IDs
    /// * `max_new_tokens` - Maximum new tokens to generate
    /// * `temperature` - Sampling temperature (0 = greedy)
    /// * `callback` - Called with (token_id, token_index) for each generated token.
    ///   Return `false` to stop generation early
    ///
    /// # Returns
    /// Generated token IDs (including prompt)
    ///
    /// # Errors
    /// Returns error if generation fails
    ///
    /// # Example
    /// ```ignore
    /// model.generate_streaming(&prompt, 100, 0.7, |token, idx| {
    ///     print!("{}", tokenizer.decode(&[token]));
    ///     std::io::stdout().flush().ok();
    ///     true // continue generating
    /// })?;
    /// ```
    pub fn generate_streaming<F>(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        mut callback: F,
    ) -> WhisperResult<Vec<u32>>
    where
        F: FnMut(u32, usize) -> bool,
    {
        let mut output_ids = prompt_ids.to_vec();
        let vocab_size = self.config.vocab_size as usize;

        for i in 0..max_new_tokens {
            // Forward pass
            let logits = self.forward(&output_ids, None)?;

            // Get logits for last position
            let last_logits_start = (output_ids.len() - 1) * vocab_size;
            let last_logits = &logits[last_logits_start..last_logits_start + vocab_size];

            // Sample next token
            let next_token = if temperature <= 0.0 {
                argmax(last_logits) as u32
            } else {
                sample_with_temperature(last_logits, temperature)? as u32
            };

            output_ids.push(next_token);

            // Call callback with generated token
            if !callback(next_token, i) {
                break; // Early stop requested by callback
            }

            // Check for EOS
            if next_token == 2 {
                break;
            }
        }

        Ok(output_ids)
    }

    /// Generate with timing statistics
    ///
    /// # Arguments
    /// * `prompt_ids` - Input token IDs
    /// * `max_new_tokens` - Maximum new tokens to generate
    /// * `temperature` - Sampling temperature (0 = greedy)
    /// * `callback` - Optional streaming callback
    ///
    /// # Returns
    /// Tuple of (generated tokens, statistics)
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate_with_stats<F>(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        callback: Option<F>,
    ) -> WhisperResult<(Vec<u32>, GenerationStats)>
    where
        F: FnMut(u32, usize) -> bool,
    {
        let start = std::time::Instant::now();
        let prompt_len = prompt_ids.len();

        let output_ids = if let Some(cb) = callback {
            self.generate_streaming(prompt_ids, max_new_tokens, temperature, cb)?
        } else {
            self.generate(prompt_ids, max_new_tokens, temperature)?
        };

        let elapsed = start.elapsed();
        let tokens_generated = output_ids.len().saturating_sub(prompt_len);
        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let ms_per_token = if tokens_generated > 0 {
            total_ms / tokens_generated as f64
        } else {
            0.0
        };
        let tokens_per_sec = if total_ms > 0.0 {
            tokens_generated as f64 / (total_ms / 1000.0)
        } else {
            0.0
        };

        let stats = GenerationStats {
            tokens_generated,
            ms_per_token,
            total_ms,
            tokens_per_sec,
            hit_eos: output_ids.last() == Some(&2),
        };

        Ok((output_ids, stats))
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

/// Argmax helper
#[cfg(test)]
pub(crate) fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(not(test))]
fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
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

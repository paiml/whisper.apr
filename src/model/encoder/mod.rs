//! Transformer encoder
//!
//! Implements the audio encoder for both Whisper and Moonshine models.
//!
//! # Whisper Architecture
//! 1. Two 1D convolutions on mel spectrogram
//! 2. Sinusoidal positional encoding
//! 3. N transformer encoder blocks (MHA + GELU FFN)
//!
//! # Moonshine Architecture
//! 1. Learned conv stem (handled externally by `audio::ConvStem`)
//! 2. RoPE positional encoding (applied per-block, not additive)
//! 3. N transformer encoder blocks (MHA + GELU MLP FFN)

mod block;
mod conv;
mod layers;

pub use block::EncoderBlock;
pub use conv::{Conv1d, ConvFrontend};
pub use layers::{FeedForward, LayerNorm};

use super::moonshine::MoonshineEncoderBlock;
use super::{AttentionType, AudioFrontend, ModelConfig, PositionalEncoding};
use crate::error::{WhisperError, WhisperResult};
use crate::model::lfm2::layer::LayerNormNoBias;
use crate::model::lfm2::rope::{RopeConfig, RotaryEmbedding};

/// Transformer encoder for audio features
///
/// Supports both Whisper (mel + sinusoidal PE + MHA + GELU) and
/// Moonshine (conv stem + RoPE + MHA + GELU MLP) architectures.
#[derive(Debug, Clone)]
pub struct Encoder {
    /// Number of layers
    n_layers: usize,
    /// Hidden state dimension
    d_model: usize,
    /// Number of attention heads
    n_heads: usize,
    /// Number of mel channels
    n_mels: usize,
    /// Convolutional frontend (Whisper only; None for Moonshine)
    conv_frontend: Option<ConvFrontend>,
    /// Whisper encoder blocks (empty for Moonshine)
    blocks: Vec<EncoderBlock>,
    /// Moonshine encoder blocks (empty for Whisper)
    moonshine_blocks: Vec<MoonshineEncoderBlock>,
    /// Rotary position embedding (Moonshine only; None for Whisper)
    rope: Option<RotaryEmbedding>,
    /// Final layer norm (Whisper — learned affine LayerNorm)
    ln_post: LayerNorm,
    /// Final layer norm (Moonshine — LayerNorm without bias)
    ln_post_rms: Option<LayerNormNoBias>,
    /// Sinusoidal positional embeddings (Whisper only; empty for Moonshine/RoPE)
    positional_embedding: Vec<f32>,
    /// Maximum sequence length (0 = variable for Moonshine)
    max_len: usize,
    /// Audio frontend type
    audio_frontend: AudioFrontend,
    /// Positional encoding type
    positional_encoding: PositionalEncoding,
}

impl Encoder {
    /// Create a new encoder from model configuration
    ///
    /// Dispatches based on `config.audio_frontend` and `config.positional_encoding`:
    /// - Whisper: creates ConvFrontend + sinusoidal PE
    /// - Moonshine: no ConvFrontend (ConvStem is external), no sinusoidal PE (RoPE per-block)
    #[must_use]
    pub fn new(config: &ModelConfig) -> Self {
        let n_layers = config.n_audio_layer as usize;
        let d_model = config.n_audio_state as usize;
        let n_heads = config.n_audio_head as usize;
        let d_ff = d_model * 4;
        let max_len = config.n_audio_ctx as usize;
        let n_mels = config.n_mels as usize;

        // Whisper: conv frontend projects mel → d_model; Moonshine: ConvStem does this externally
        let conv_frontend = match config.audio_frontend {
            AudioFrontend::MelFilterbank => Some(ConvFrontend::new(n_mels, d_model)),
            AudioFrontend::LearnedConv => None,
        };

        // Dispatch block creation based on attention type
        let (blocks, moonshine_blocks, rope) = match config.attention_type {
            AttentionType::Mha => {
                // Whisper: standard MHA + GELU blocks
                let whisper_blocks: Vec<EncoderBlock> = (0..n_layers)
                    .map(|_| EncoderBlock::new(d_model, n_heads, d_ff))
                    .collect();
                (whisper_blocks, Vec::new(), None)
            }
            AttentionType::Gqa { kv_heads } => {
                // Moonshine: MHA + GELU MLP blocks with RoPE
                let head_dim = d_model / n_heads;
                // HF Moonshine config: intermediate_size = 4 * hidden_size
                let intermediate_size = d_model * 4;
                let mut moon_blocks = Vec::with_capacity(n_layers);
                for _ in 0..n_layers {
                    match MoonshineEncoderBlock::new(
                        d_model,
                        n_heads,
                        kv_heads as usize,
                        intermediate_size,
                    ) {
                        Ok(block) => moon_blocks.push(block),
                        Err(_) => return Self::fallback_encoder(config),
                    }
                }
                // Moonshine partial_rotary_factor=0.9: rotate first 32 of 36 head dims
                let rotary_dim = (head_dim as f64 * 0.9).floor() as usize;
                let rotary_dim = rotary_dim - (rotary_dim % 2);
                // RoPE uses padded head_dim (36→40) so it accepts padded Q/K
                let padded_hd = head_dim.div_ceil(8) * 8;
                let Ok(rope_emb) = RotaryEmbedding::new(RopeConfig {
                    head_dim: padded_hd,
                    base: 10000.0,
                    max_seq_len: 2048,
                    rotary_dim: Some(rotary_dim),
                }) else {
                    return Self::fallback_encoder(config);
                };
                (Vec::new(), moon_blocks, Some(rope_emb))
            }
        };

        // Whisper: fixed sinusoidal PE; Moonshine: RoPE applied within each attention layer
        let positional_embedding = match config.positional_encoding {
            PositionalEncoding::Sinusoidal => Self::create_positional_embedding(max_len, d_model),
            PositionalEncoding::Rotary => Vec::new(),
        };

        // Moonshine uses LayerNorm(bias=False) for final norm; Whisper uses LayerNorm
        let ln_post_rms = if rope.is_some() {
            Some(LayerNormNoBias::new(d_model))
        } else {
            None
        };

        Self {
            n_layers,
            d_model,
            n_heads,
            n_mels,
            conv_frontend,
            blocks,
            moonshine_blocks,
            rope,
            ln_post: LayerNorm::new(d_model),
            ln_post_rms,
            positional_embedding,
            max_len,
            audio_frontend: config.audio_frontend,
            positional_encoding: config.positional_encoding,
        }
    }

    /// Fallback encoder when Moonshine block creation fails (should not happen with valid config)
    fn fallback_encoder(config: &ModelConfig) -> Self {
        let d_model = config.n_audio_state as usize;
        let max_len = config.n_audio_ctx as usize;
        Self {
            n_layers: config.n_audio_layer as usize,
            d_model,
            n_heads: config.n_audio_head as usize,
            n_mels: config.n_mels as usize,
            conv_frontend: None,
            blocks: Vec::new(),
            moonshine_blocks: Vec::new(),
            rope: None,
            ln_post: LayerNorm::new(d_model),
            ln_post_rms: None,
            positional_embedding: Vec::new(),
            max_len,
            audio_frontend: config.audio_frontend,
            positional_encoding: config.positional_encoding,
        }
    }

    /// Create sinusoidal positional embeddings
    fn create_positional_embedding(max_len: usize, d_model: usize) -> Vec<f32> {
        let mut pe = vec![0.0_f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let angle = pos as f32 / 10000_f32.powf(2.0 * i as f32 / d_model as f32);
                pe[pos * d_model + 2 * i] = angle.sin();
                pe[pos * d_model + 2 * i + 1] = angle.cos();
            }
        }

        debug_assert_eq!(
            pe.len(),
            max_len * d_model,
            "positional embedding size must be max_len × d_model"
        );
        debug_assert!(
            pe.iter().all(|x| x.is_finite()),
            "all positional embedding values must be finite"
        );
        pe
    }

    /// Forward pass through encoder
    ///
    /// Input: features already projected to d_model dimension
    /// - Whisper: output of conv_frontend (mel → d_model)
    /// - Moonshine: output of ConvStem (raw audio → d_model)
    pub fn forward(&self, features: &[f32]) -> WhisperResult<Vec<f32>> {
        let seq_len = features.len() / self.d_model;

        if features.len() % self.d_model != 0 {
            return Err(WhisperError::Model("input size mismatch".into()));
        }
        // Whisper has fixed max_len; Moonshine has variable (max_len=0)
        if self.max_len > 0 && seq_len > self.max_len {
            return Err(WhisperError::Model(format!(
                "sequence length {} exceeds max {}",
                seq_len, self.max_len
            )));
        }

        let mut x = features.to_vec();

        // Dispatch based on model type
        if self.rope.is_some() {
            // Moonshine path: MHA + GELU MLP with RoPE per-block
            let rope = self
                .rope
                .as_ref()
                .ok_or_else(|| WhisperError::Model("Moonshine encoder requires RoPE".into()))?;
            for block in &self.moonshine_blocks {
                x = block.forward(&x, seq_len, rope)?;
            }
        } else {
            // Whisper path: sinusoidal PE + MHA + GELU
            if self.positional_encoding == PositionalEncoding::Sinusoidal {
                for pos in 0..seq_len {
                    for d in 0..self.d_model {
                        x[pos * self.d_model + d] +=
                            self.positional_embedding[pos * self.d_model + d];
                    }
                }
            }
            for block in &self.blocks {
                x = block.forward(&x)?;
            }
        }

        debug_assert!(
            x.iter().all(|v| v.is_finite()),
            "encoder output must be finite before final layer norm"
        );

        // Final layer norm: RmsNorm for Moonshine, LayerNorm for Whisper
        if let Some(ref rms) = self.ln_post_rms {
            rms.forward(&x, seq_len)
        } else {
            self.ln_post.forward(&x)
        }
    }

    /// Forward pass with BrickProfiler instrumentation (WAPR-PROFILE-001 Gap 1)
    ///
    /// Records per-operator timing and CPU cycles into the trueno `BrickProfiler`
    /// via O(1) `BrickId`-indexed arrays. Category breakdown (Norm/Attention/FFN)
    /// available via `profiler.category_stats()` after this call.
    #[cfg(feature = "realizar-inference")]
    pub fn forward_profiled(
        &self,
        features: &[f32],
        profiler: &mut trueno::BrickProfiler,
        tracer: Option<&mut realizar::InferenceTracer>,
    ) -> WhisperResult<Vec<f32>> {
        let seq_len = features.len() / self.d_model;

        if features.len() % self.d_model != 0 {
            return Err(WhisperError::Model("input size mismatch".into()));
        }
        if self.max_len > 0 && seq_len > self.max_len {
            return Err(WhisperError::Model(format!(
                "sequence length {} exceeds max {}",
                seq_len, self.max_len
            )));
        }

        let mut x = features.to_vec();

        // Whisper path only (Moonshine uses moonshine_blocks, not instrumented here)
        if self.positional_encoding == PositionalEncoding::Sinusoidal {
            for pos in 0..seq_len {
                for d in 0..self.d_model {
                    x[pos * self.d_model + d] += self.positional_embedding[pos * self.d_model + d];
                }
            }
        }

        // WAPR-PROFILE-001 Gap 5: Per-block InferenceTracer events
        if let Some(tracer) = tracer {
            for (layer_idx, block) in self.blocks.iter().enumerate() {
                tracer.start_step(realizar::TraceStep::TransformerBlock);
                x = block.forward_profiled(&x, profiler)?;
                tracer.trace_layer(layer_idx, 0, Some(&x), seq_len, self.d_model);
            }
        } else {
            for block in &self.blocks {
                x = block.forward_profiled(&x, profiler)?;
            }
        }

        debug_assert!(
            x.iter().all(|v| v.is_finite()),
            "encoder output must be finite before final layer norm"
        );

        self.ln_post.forward(&x)
    }

    /// Forward pass with BrickProfiler instrumentation (WAPR-PROFILE-001 Gap 1)
    ///
    /// Version without InferenceTracer when realizar-inference feature is disabled.
    #[cfg(not(feature = "realizar-inference"))]
    pub fn forward_profiled(
        &self,
        features: &[f32],
        profiler: &mut trueno::BrickProfiler,
    ) -> WhisperResult<Vec<f32>> {
        let seq_len = features.len() / self.d_model;

        if features.len() % self.d_model != 0 {
            return Err(WhisperError::Model("input size mismatch".into()));
        }
        if self.max_len > 0 && seq_len > self.max_len {
            return Err(WhisperError::Model(format!(
                "sequence length {} exceeds max {}",
                seq_len, self.max_len
            )));
        }

        let mut x = features.to_vec();

        if self.positional_encoding == PositionalEncoding::Sinusoidal {
            for pos in 0..seq_len {
                for d in 0..self.d_model {
                    x[pos * self.d_model + d] += self.positional_embedding[pos * self.d_model + d];
                }
            }
        }

        for block in &self.blocks {
            x = block.forward_profiled(&x, profiler)?;
        }

        debug_assert!(
            x.iter().all(|v| v.is_finite()),
            "encoder output must be finite before final layer norm"
        );

        self.ln_post.forward(&x)
    }

    /// Forward from mel spectrogram with BrickProfiler (WAPR-PROFILE-001 Gap 1)
    ///
    /// Records conv frontend timing via `BrickId::Embedding` (Other category),
    /// then delegates to `forward_profiled()` for encoder blocks.
    #[cfg(feature = "realizar-inference")]
    pub fn forward_mel_profiled(
        &self,
        mel: &[f32],
        profiler: &mut trueno::BrickProfiler,
        mut tracer: Option<&mut realizar::InferenceTracer>,
    ) -> WhisperResult<Vec<f32>> {
        if mel.len() % self.n_mels != 0 {
            return Err(WhisperError::Model(format!(
                "mel size {} not divisible by n_mels {}",
                mel.len(),
                self.n_mels
            )));
        }

        let frontend = self.conv_frontend.as_ref().ok_or_else(|| {
            WhisperError::Model(
                "forward_mel requires conv frontend (not available for Moonshine)".into(),
            )
        })?;

        // WAPR-PROFILE-001 Gap 5: Trace embed step (conv frontend)
        if let Some(ref mut t) = tracer {
            t.start_step(realizar::TraceStep::Embed);
        }

        // Profile conv_frontend via BrickId::Embedding (Other category)
        let c0 = trueno::brick::cpu_cycles();
        let timer = profiler.start_brick(trueno::BrickId::Embedding);
        let conv_output = frontend.forward(mel)?;
        let c1 = trueno::brick::cpu_cycles();
        let mel_frames = (mel.len() / self.n_mels) as u64;
        profiler.stop_brick(timer, mel_frames);
        let stats = profiler.brick_stats_mut(trueno::BrickId::Embedding);
        let cycles = c1.wrapping_sub(c0);
        stats.total_cycles += cycles;
        stats.min_cycles = stats.min_cycles.min(cycles);
        stats.max_cycles = stats.max_cycles.max(cycles);

        // Trace embed output
        if let Some(ref mut t) = tracer {
            let hidden_dim = self.d_model;
            let token_count = conv_output.len() / hidden_dim;
            t.trace_embed(token_count, hidden_dim, Some(&conv_output));
        }

        self.forward_profiled(&conv_output, profiler, tracer)
    }

    /// Forward from mel spectrogram with BrickProfiler (WAPR-PROFILE-001 Gap 1)
    ///
    /// Version without InferenceTracer when realizar-inference feature is disabled.
    #[cfg(not(feature = "realizar-inference"))]
    pub fn forward_mel_profiled(
        &self,
        mel: &[f32],
        profiler: &mut trueno::BrickProfiler,
    ) -> WhisperResult<Vec<f32>> {
        if mel.len() % self.n_mels != 0 {
            return Err(WhisperError::Model(format!(
                "mel size {} not divisible by n_mels {}",
                mel.len(),
                self.n_mels
            )));
        }

        let frontend = self.conv_frontend.as_ref().ok_or_else(|| {
            WhisperError::Model(
                "forward_mel requires conv frontend (not available for Moonshine)".into(),
            )
        })?;

        // Profile conv_frontend via BrickId::Embedding (Other category)
        let c0 = trueno::brick::cpu_cycles();
        let timer = profiler.start_brick(trueno::BrickId::Embedding);
        let conv_output = frontend.forward(mel)?;
        let c1 = trueno::brick::cpu_cycles();
        let mel_frames = (mel.len() / self.n_mels) as u64;
        profiler.stop_brick(timer, mel_frames);
        let stats = profiler.brick_stats_mut(trueno::BrickId::Embedding);
        let cycles = c1.wrapping_sub(c0);
        stats.total_cycles += cycles;
        stats.min_cycles = stats.min_cycles.min(cycles);
        stats.max_cycles = stats.max_cycles.max(cycles);

        self.forward_profiled(&conv_output, profiler)
    }

    /// Get number of layers
    #[must_use]
    pub const fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Get model dimension
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get number of attention heads
    #[must_use]
    pub const fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Get maximum sequence length
    #[must_use]
    pub const fn max_len(&self) -> usize {
        self.max_len
    }

    /// Get positional embedding reference
    #[must_use]
    pub fn positional_embedding(&self) -> &[f32] {
        &self.positional_embedding
    }

    /// Get mutable positional embedding reference
    pub fn positional_embedding_mut(&mut self) -> &mut [f32] {
        &mut self.positional_embedding
    }

    /// Get encoder blocks reference
    #[must_use]
    pub fn blocks(&self) -> &[EncoderBlock] {
        &self.blocks
    }

    /// Get mutable encoder blocks reference
    pub fn blocks_mut(&mut self) -> &mut [EncoderBlock] {
        &mut self.blocks
    }

    /// Get layer norm reference
    #[must_use]
    pub fn ln_post(&self) -> &LayerNorm {
        &self.ln_post
    }

    /// Get mutable layer norm reference
    pub fn ln_post_mut(&mut self) -> &mut LayerNorm {
        &mut self.ln_post
    }

    /// Get number of mel channels
    #[must_use]
    pub const fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Get convolutional frontend reference (Whisper only)
    #[must_use]
    pub fn conv_frontend(&self) -> Option<&ConvFrontend> {
        self.conv_frontend.as_ref()
    }

    /// Get mutable convolutional frontend reference (Whisper only)
    pub fn conv_frontend_mut(&mut self) -> Option<&mut ConvFrontend> {
        self.conv_frontend.as_mut()
    }

    /// Get Moonshine encoder blocks reference
    #[must_use]
    pub fn moonshine_blocks(&self) -> &[MoonshineEncoderBlock] {
        &self.moonshine_blocks
    }

    /// Get mutable Moonshine encoder blocks reference
    pub fn moonshine_blocks_mut(&mut self) -> &mut [MoonshineEncoderBlock] {
        &mut self.moonshine_blocks
    }

    /// Get RoPE embedding reference (Moonshine only)
    #[must_use]
    pub fn rope(&self) -> Option<&RotaryEmbedding> {
        self.rope.as_ref()
    }

    /// Get Moonshine final layer norm reference
    #[must_use]
    pub fn ln_post_rms(&self) -> Option<&LayerNormNoBias> {
        self.ln_post_rms.as_ref()
    }

    /// Get mutable Moonshine final layer norm reference (for loading weights)
    pub fn ln_post_rms_mut(&mut self) -> Option<&mut LayerNormNoBias> {
        self.ln_post_rms.as_mut()
    }

    /// Forward pass from raw mel spectrogram (Whisper path only)
    ///
    /// # Errors
    /// Returns error if mel size is invalid or conv frontend is not available (Moonshine)
    pub fn forward_mel(&self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        if mel.len() % self.n_mels != 0 {
            return Err(WhisperError::Model(format!(
                "mel size {} not divisible by n_mels {}",
                mel.len(),
                self.n_mels
            )));
        }

        let frontend = self.conv_frontend.as_ref().ok_or_else(|| {
            WhisperError::Model(
                "forward_mel requires conv frontend (not available for Moonshine)".into(),
            )
        })?;
        let conv_output = frontend.forward(mel)?;
        self.forward(&conv_output)
    }

    /// Get audio frontend type
    #[must_use]
    pub const fn audio_frontend(&self) -> AudioFrontend {
        self.audio_frontend
    }

    /// Get positional encoding type
    #[must_use]
    pub const fn positional_encoding(&self) -> PositionalEncoding {
        self.positional_encoding
    }

    /// Forward pass for a batch of feature sequences
    ///
    /// For Whisper: input is mel spectrograms (processed through conv frontend + encoder)
    /// For Moonshine: input is pre-projected features (processed through encoder blocks)
    pub fn forward_batch(&self, batch: &[Vec<f32>]) -> WhisperResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(batch.len());

        for features in batch {
            let encoded = if self.conv_frontend.is_some() {
                // Whisper: route through mel → conv frontend → encoder
                self.forward_mel(features)?
            } else {
                // Moonshine: features already projected to d_model, go straight through encoder blocks
                self.forward(features)?
            };
            results.push(encoded);
        }

        Ok(results)
    }

    /// Forward pass for batch with padding
    pub fn forward_batch_padded(&self, batch: &[Vec<f32>]) -> WhisperResult<BatchEncoderOutput> {
        let encoded = self.forward_batch(batch)?;

        let seq_lengths: Vec<usize> = encoded.iter().map(|e| e.len() / self.d_model).collect();
        let max_seq_len = seq_lengths.iter().copied().max().unwrap_or(0);

        let batch_size = encoded.len();
        let total_size = batch_size * max_seq_len * self.d_model;
        let mut padded = vec![0.0_f32; total_size];

        for (batch_idx, features) in encoded.iter().enumerate() {
            let seq_len = features.len() / self.d_model;
            for t in 0..seq_len {
                for d in 0..self.d_model {
                    let src_idx = t * self.d_model + d;
                    let dst_idx = batch_idx * max_seq_len * self.d_model + t * self.d_model + d;
                    padded[dst_idx] = features[src_idx];
                }
            }
        }

        Ok(BatchEncoderOutput {
            features: padded,
            seq_lengths,
            max_seq_len,
            batch_size,
            d_model: self.d_model,
        })
    }

    /// Forward pass with activation probing (Moonshine path only)
    ///
    /// Records activation snapshots at each encoder block boundary
    /// and the final layer norm output.
    ///
    /// # Errors
    /// Returns error on dimension mismatch or if RoPE is unavailable
    pub fn forward_probed(
        &self,
        features: &[f32],
        probe: &mut crate::probe::ActivationProbe,
    ) -> WhisperResult<Vec<f32>> {
        let seq_len = features.len() / self.d_model;

        if features.len() % self.d_model != 0 {
            return Err(WhisperError::Model("input size mismatch".into()));
        }
        if self.max_len > 0 && seq_len > self.max_len {
            return Err(WhisperError::Model(format!(
                "sequence length {} exceeds max {}",
                seq_len, self.max_len
            )));
        }

        let mut x = features.to_vec();

        if self.rope.is_some() {
            // Moonshine path: probed block-by-block
            let rope = self
                .rope
                .as_ref()
                .ok_or_else(|| WhisperError::Model("Moonshine encoder requires RoPE".into()))?;
            for (i, block) in self.moonshine_blocks.iter().enumerate() {
                x = block.forward_probed(&x, seq_len, rope, i, probe)?;
            }
        } else {
            // Whisper path: sinusoidal PE + blocks (no per-block probing)
            if self.positional_encoding == PositionalEncoding::Sinusoidal {
                for pos in 0..seq_len {
                    for d in 0..self.d_model {
                        x[pos * self.d_model + d] +=
                            self.positional_embedding[pos * self.d_model + d];
                    }
                }
            }
            for block in &self.blocks {
                x = block.forward(&x)?;
            }
        }

        // Final layer norm
        let result = if let Some(ref rms) = self.ln_post_rms {
            rms.forward(&x, seq_len)?
        } else {
            self.ln_post.forward(&x)?
        };
        probe.record("encoder.ln_post_out", &result, &[seq_len, self.d_model]);

        Ok(result)
    }

    /// Finalize all weights by caching transposed/pre-computed data
    pub fn finalize_weights(&mut self) {
        // Cache transposed conv frontend weights (avoid reshape+transpose per forward call)
        if let Some(ref mut frontend) = self.conv_frontend {
            frontend.finalize_weights();
        }
        for block in &mut self.blocks {
            block.finalize_weights();
        }
        // Moonshine blocks use MHA/MLP which don't have weight finalization;
        // no additional work needed for moonshine_blocks.
    }
}

/// Batched encoder output with padding information
#[derive(Debug, Clone)]
pub struct BatchEncoderOutput {
    /// Padded features (batch_size × max_seq_len × d_model) flattened
    pub features: Vec<f32>,
    /// Actual sequence length for each item
    pub seq_lengths: Vec<usize>,
    /// Maximum sequence length (for padding)
    pub max_seq_len: usize,
    /// Batch size
    pub batch_size: usize,
    /// Model dimension
    pub d_model: usize,
}

impl BatchEncoderOutput {
    /// Get features for a specific batch item (unpadded)
    #[must_use]
    pub fn get(&self, batch_idx: usize) -> Option<Vec<f32>> {
        if batch_idx >= self.batch_size {
            return None;
        }

        let seq_len = self.seq_lengths[batch_idx];
        let mut features = Vec::with_capacity(seq_len * self.d_model);

        for t in 0..seq_len {
            for d in 0..self.d_model {
                let idx = batch_idx * self.max_seq_len * self.d_model + t * self.d_model + d;
                features.push(self.features[idx]);
            }
        }

        Some(features)
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    /// Get the total number of tokens
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.seq_lengths.iter().sum()
    }
}

// Fused FFN for realizar-inference feature
#[cfg(feature = "realizar-inference")]
pub use fused::FusedFFN;

#[cfg(feature = "realizar-inference")]
mod fused {
    use crate::error::WhisperResult;

    /// Fused Feed-Forward Network
    ///
    /// Combines layer normalization with the first linear layer
    /// for improved performance when using fused kernels.
    #[derive(Debug, Clone)]
    pub struct FusedFFN {
        /// Fused weight matrix (LN + FC1)
        pub fused_weight: Vec<f32>,
        /// First linear layer bias
        pub fc1_bias: Vec<f32>,
        /// Second linear layer weight
        pub fc2_weight: Vec<f32>,
        /// Second linear layer bias
        pub fc2_bias: Vec<f32>,
        /// Model dimension
        pub d_model: usize,
        /// Feed-forward hidden dimension
        pub d_ff: usize,
    }

    impl FusedFFN {
        /// Create a new fused FFN layer
        pub fn new(d_model: usize, d_ff: usize) -> WhisperResult<Self> {
            Ok(Self {
                fused_weight: vec![0.0; d_model * d_ff],
                fc1_bias: vec![0.0; d_ff],
                fc2_weight: vec![0.0; d_ff * d_model],
                fc2_bias: vec![0.0; d_model],
                d_model,
                d_ff,
            })
        }

        /// Forward pass through fused FFN
        ///
        /// Computes: fc2(GELU(fused_weight @ input + fc1_bias)) + fc2_bias
        /// where fused_weight = fc1_weight * ln_weight (pre-fused)
        pub fn forward(&self, input: &[f32]) -> WhisperResult<Vec<f32>> {
            let seq_len = input.len() / self.d_model;
            let mut output = vec![0.0f32; seq_len * self.d_model];

            for s in 0..seq_len {
                let x = &input[s * self.d_model..(s + 1) * self.d_model];

                // FC1: fused_weight (d_ff x d_model) @ x (d_model) + fc1_bias -> hidden (d_ff)
                let mut hidden = vec![0.0f32; self.d_ff];
                for (i, h) in hidden.iter_mut().enumerate() {
                    let mut sum = self.fc1_bias[i];
                    for (j, &xj) in x.iter().enumerate() {
                        sum += self.fused_weight[i * self.d_model + j] * xj;
                    }
                    // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    let g = sum
                        * 0.5
                        * (1.0 + (0.797_884_6_f32 * (sum + 0.044715 * sum * sum * sum)).tanh());
                    *h = g;
                }

                // FC2: fc2_weight (d_model x d_ff) @ hidden (d_ff) + fc2_bias -> out (d_model)
                let out = &mut output[s * self.d_model..(s + 1) * self.d_model];
                for (i, o) in out.iter_mut().enumerate() {
                    let mut sum = self.fc2_bias[i];
                    for (j, &hj) in hidden.iter().enumerate() {
                        sum += self.fc2_weight[i * self.d_ff + j] * hj;
                    }
                    *o = sum;
                }
            }

            Ok(output)
        }

        /// Set fused weights combining layer norm and FC1
        pub fn set_fused_weights(
            &mut self,
            ln_weight: &[f32],
            ln_bias: &[f32],
            fc1_weight: &[f32],
        ) {
            for i in 0..self.d_ff {
                for j in 0..self.d_model {
                    self.fused_weight[i * self.d_model + j] =
                        fc1_weight[i * self.d_model + j] * ln_weight[j];
                }
            }
            let _ = ln_bias;
        }

        /// Set first linear layer weights and bias
        pub fn set_fc1_weights(&mut self, weight: &[f32], bias: &[f32]) {
            let len = weight.len().min(self.fused_weight.len());
            self.fused_weight[..len].copy_from_slice(&weight[..len]);
            let bias_len = bias.len().min(self.fc1_bias.len());
            self.fc1_bias[..bias_len].copy_from_slice(&bias[..bias_len]);
        }

        /// Set second linear layer weights and bias
        pub fn set_fc2_weights(&mut self, weight: &[f32], bias: &[f32]) {
            let len = weight.len().min(self.fc2_weight.len());
            self.fc2_weight[..len].copy_from_slice(&weight[..len]);
            let bias_len = bias.len().min(self.fc2_bias.len());
            self.fc2_bias[..bias_len].copy_from_slice(&bias[..bias_len]);
        }

        /// Set layer normalization weights (used in fusion)
        pub fn set_norm_weights(&mut self, _weight: &[f32], _bias: &[f32]) {
            // LayerNorm weights are used during fusion with FC1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_new() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        assert_eq!(encoder.n_layers(), 4);
        assert_eq!(encoder.d_model(), 384);
        assert_eq!(encoder.n_heads(), 6);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_positional_embedding_shape() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        assert_eq!(
            encoder.positional_embedding.len(),
            encoder.max_len * encoder.d_model
        );
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_forward() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let input = vec![0.0_f32; 10 * 384];
        let output = encoder.forward(&input).expect("forward should succeed");
        assert_eq!(output.len(), 10 * 384);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_forward_size_mismatch() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let input = vec![0.0_f32; 100];
        let result = encoder.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_forward_too_long() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let input = vec![0.0_f32; 2000 * 384];
        let result = encoder.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_positional_embedding_sinusoidal() {
        let pe = Encoder::create_positional_embedding(100, 64);
        assert!(pe[0].abs() < 1e-5, "sin(0) should be 0");
        assert!((pe[1] - 1.0).abs() < 1e-5, "cos(0) should be 1");
    }

    #[test]
    fn test_positional_embedding_different_positions() {
        let pe = Encoder::create_positional_embedding(100, 64);
        let pos0: Vec<f32> = pe[0..64].to_vec();
        let pos1: Vec<f32> = pe[64..128].to_vec();

        let diff: f32 = pos0
            .iter()
            .zip(pos1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_positional_embedding() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let pe = encoder.positional_embedding();
        assert_eq!(pe.len(), encoder.max_len() * encoder.d_model());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_positional_embedding_mut() {
        let config = ModelConfig::tiny();
        let mut encoder = Encoder::new(&config);
        encoder.positional_embedding_mut()[0] = 100.0;
        assert!((encoder.positional_embedding()[0] - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_blocks() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        assert_eq!(encoder.blocks().len(), 4);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_blocks_mut() {
        let config = ModelConfig::tiny();
        let mut encoder = Encoder::new(&config);
        let blocks = encoder.blocks_mut();
        assert_eq!(blocks.len(), 4);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_ln_post() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let ln = encoder.ln_post();
        assert_eq!(ln.normalized_shape, encoder.d_model());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_ln_post_mut() {
        let config = ModelConfig::tiny();
        let mut encoder = Encoder::new(&config);
        encoder.ln_post_mut().weight[0] = 2.0;
        assert!((encoder.ln_post().weight[0] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_n_mels() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        assert_eq!(encoder.n_mels(), 80);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_conv_frontend() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let frontend = encoder.conv_frontend().expect("Whisper has conv frontend");
        assert_eq!(frontend.n_mels, 80);
        assert_eq!(frontend.d_model, 384);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_conv_frontend_mut() {
        let config = ModelConfig::tiny();
        let mut encoder = Encoder::new(&config);
        encoder
            .conv_frontend_mut()
            .expect("Whisper has conv frontend")
            .conv1
            .bias_mut()[0] = 5.0;
        assert!(
            (encoder
                .conv_frontend()
                .expect("Whisper has conv frontend")
                .conv1
                .bias[0]
                - 5.0)
                .abs()
                < f32::EPSILON
        );
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_forward_mel() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let mel = vec![0.0_f32; 100 * 80];
        let output = encoder.forward_mel(&mel).expect("forward_mel");
        let expected_frames = encoder
            .conv_frontend()
            .expect("Whisper has conv frontend")
            .output_length(100);
        assert_eq!(output.len(), expected_frames * encoder.d_model());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_forward_mel_size_mismatch() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let mel = vec![0.0_f32; 123];
        let result = encoder.forward_mel(&mel);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_forward_batch_empty() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let batch: Vec<Vec<f32>> = Vec::new();
        let results = encoder.forward_batch(&batch).expect("forward_batch");
        assert!(results.is_empty());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_forward_batch_single() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let mel = vec![0.0_f32; 100 * 80];
        let batch = vec![mel];
        let results = encoder.forward_batch(&batch).expect("forward_batch");
        assert_eq!(results.len(), 1);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_forward_batch_padded_empty() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let batch: Vec<Vec<f32>> = Vec::new();
        let output = encoder.forward_batch_padded(&batch).expect("padded");
        assert!(output.is_empty());
        assert_eq!(output.batch_size, 0);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_batch_encoder_output_get() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let mel1 = vec![0.1_f32; 100 * 80];
        let mel2 = vec![0.2_f32; 50 * 80];
        let batch = vec![mel1, mel2];

        let output = encoder.forward_batch_padded(&batch).expect("padded");
        assert!(output.get(0).is_some());
        assert!(output.get(1).is_some());
        assert!(output.get(2).is_none());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_batch_encoder_output_total_tokens() {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let mel1 = vec![0.0_f32; 100 * 80];
        let mel2 = vec![0.0_f32; 50 * 80];
        let batch = vec![mel1, mel2];

        let output = encoder.forward_batch_padded(&batch).expect("padded");
        let total = output.total_tokens();
        let expected = output.seq_lengths.iter().sum::<usize>();
        assert_eq!(total, expected);
    }
}

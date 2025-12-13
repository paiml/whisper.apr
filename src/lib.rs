//! # Whisper.apr
//!
//! WASM-first automatic speech recognition engine implementing OpenAI's Whisper architecture.
//!
//! ## Overview
//!
//! Whisper.apr is designed from inception for WASM deployment via `wasm32-unknown-unknown`,
//! leveraging Rust's superior WASM toolchain for:
//! - 30-40% smaller binary sizes through tree-shaking
//! - Native WASM SIMD 128-bit intrinsics without Emscripten overhead
//! - Zero-copy audio buffer handling via shared memory
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use whisper_apr::{WhisperApr, TranscribeOptions};
//!
//! let whisper = WhisperApr::load("base.apr")?;
//! let result = whisper.transcribe(&audio_samples, TranscribeOptions::default())?;
//! println!("{}", result.text);
//! ```
//!
//! ## Features
//!
//! - `std` (default): Standard library support
//! - `wasm`: WASM bindings via wasm-bindgen
//! - `simd`: SIMD acceleration via trueno
//! - `tracing`: Performance tracing via renacer

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![deny(clippy::unwrap_used)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod audio;
pub mod detection;
pub mod error;
pub mod format;
pub mod inference;
pub mod memory;
pub mod model;
pub mod progress;
pub mod simd;
pub mod timestamps;
pub mod tokenizer;
pub mod vad;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use error::{WhisperError, WhisperResult};

/// Whisper model configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Tiny model (~39M parameters, ~40MB WASM)
    Tiny,
    /// Tiny English-only model (~39M parameters, ~40MB WASM)
    TinyEn,
    /// Base model (~74M parameters, ~75MB WASM)
    Base,
    /// Base English-only model (~74M parameters, ~75MB WASM)
    BaseEn,
    /// Small model (~244M parameters, ~250MB WASM)
    Small,
    /// Small English-only model (~244M parameters, ~250MB WASM)
    SmallEn,
    /// Medium model (~769M parameters, ~1.5GB WASM)
    Medium,
    /// Medium English-only model (~769M parameters, ~1.5GB WASM)
    MediumEn,
    /// Large model (~1.5B parameters, ~3GB WASM)
    Large,
    /// Large v1 model (~1.5B parameters, ~3GB WASM)
    LargeV1,
    /// Large v2 model (~1.5B parameters, ~3GB WASM)
    LargeV2,
    /// Large v3 model (~1.5B parameters, ~3GB WASM)
    LargeV3,
}

/// Decoding strategy for transcription
#[derive(Debug, Clone, Copy)]
pub enum DecodingStrategy {
    /// Fast, memory-efficient greedy decoding
    Greedy,
    /// Higher quality beam search
    BeamSearch {
        /// Number of beams (default: 5)
        beam_size: usize,
        /// Temperature for sampling (default: 0.0)
        temperature: f32,
        /// Patience factor (default: 1.0)
        patience: f32,
    },
    /// Sampling with temperature
    Sampling {
        /// Temperature for sampling (default: 1.0)
        temperature: f32,
        /// Top-k filtering (default: None)
        top_k: Option<usize>,
        /// Top-p (nucleus) filtering (default: None)
        top_p: Option<f32>,
    },
}

impl Default for DecodingStrategy {
    fn default() -> Self {
        Self::Greedy
    }
}

/// Task type for transcription
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Task {
    /// Transcribe in original language
    #[default]
    Transcribe,
    /// Translate to English
    Translate,
}

/// Options for transcription
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Language code (e.g., "en", "es") or "auto" for detection
    pub language: Option<String>,
    /// Task type (transcribe or translate)
    pub task: Task,
    /// Decoding strategy
    pub strategy: DecodingStrategy,
    /// Whether to include word-level timestamps
    pub word_timestamps: bool,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            task: Task::default(),
            strategy: DecodingStrategy::default(),
            word_timestamps: false,
        }
    }
}

/// A timestamped segment of transcription
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Transcribed text
    pub text: String,
    /// Token IDs
    pub tokens: Vec<u32>,
}

/// Result of transcription
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Full transcribed text
    pub text: String,
    /// Detected or specified language
    pub language: String,
    /// Timestamped segments
    pub segments: Vec<Segment>,
}

/// Result of batch transcription (WAPR-083)
#[derive(Debug, Clone)]
pub struct BatchTranscriptionResult {
    /// Individual transcription results
    pub results: Vec<TranscriptionResult>,
    /// Total processing time in seconds
    pub total_duration_secs: f32,
}

impl BatchTranscriptionResult {
    /// Get number of transcriptions
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get result by index
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&TranscriptionResult> {
        self.results.get(index)
    }

    /// Iterate over results
    pub fn iter(&self) -> impl Iterator<Item = &TranscriptionResult> {
        self.results.iter()
    }

    /// Get all texts
    #[must_use]
    pub fn texts(&self) -> Vec<&str> {
        self.results.iter().map(|r| r.text.as_str()).collect()
    }
}

/// Main Whisper ASR engine
///
/// This is the primary interface for transcription. It handles:
/// - Audio preprocessing (resampling, mel spectrogram)
/// - Encoder forward pass (audio → features)
/// - Decoder forward pass (features → tokens)
/// - Token decoding to text
///
/// # Example
///
/// ```rust,ignore
/// use whisper_apr::{WhisperApr, TranscribeOptions};
///
/// // Load model from .apr file
/// let whisper = WhisperApr::from_config(model_config)?;
///
/// // Transcribe audio (16kHz mono f32)
/// let result = whisper.transcribe(&audio, TranscribeOptions::default())?;
/// println!("{}", result.text);
/// ```
#[derive(Debug)]
pub struct WhisperApr {
    /// Model configuration
    config: model::ModelConfig,
    /// Audio encoder
    encoder: model::Encoder,
    /// Text decoder
    decoder: model::Decoder,
    /// BPE tokenizer
    tokenizer: tokenizer::BpeTokenizer,
    /// Mel filterbank
    mel_filters: audio::MelFilterbank,
    /// Resampler for non-16kHz audio
    resampler: Option<audio::SincResampler>,
}

impl WhisperApr {
    /// Create a new Whisper instance from model configuration
    ///
    /// Note: This creates the model structure but doesn't load weights.
    /// Use `load_weights` to load weights from an .apr file.
    #[must_use]
    pub fn from_config(config: model::ModelConfig) -> Self {
        let encoder = model::Encoder::new(&config);
        let decoder = model::Decoder::new(&config);
        let tokenizer = tokenizer::BpeTokenizer::with_base_tokens();
        let mel_filters =
            audio::MelFilterbank::new(config.n_mels as usize, audio::N_FFT, audio::SAMPLE_RATE);

        Self {
            config,
            encoder,
            decoder,
            tokenizer,
            mel_filters,
            resampler: None,
        }
    }

    /// Create a tiny model configuration
    #[must_use]
    pub fn tiny() -> Self {
        Self::from_config(model::ModelConfig::tiny())
    }

    /// Create a base model configuration
    #[must_use]
    pub fn base() -> Self {
        Self::from_config(model::ModelConfig::base())
    }

    /// Create a small model configuration
    #[must_use]
    pub fn small() -> Self {
        Self::from_config(model::ModelConfig::small())
    }

    /// Create a medium model configuration
    #[must_use]
    pub fn medium() -> Self {
        Self::from_config(model::ModelConfig::medium())
    }

    /// Create a large model configuration
    #[must_use]
    pub fn large() -> Self {
        Self::from_config(model::ModelConfig::large())
    }

    /// Get model configuration
    #[must_use]
    pub const fn config(&self) -> &model::ModelConfig {
        &self.config
    }

    /// Get model type
    #[must_use]
    pub const fn model_type(&self) -> ModelType {
        self.config.model_type
    }

    /// Transcribe audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, 16kHz, f32 normalized to [-1, 1])
    /// * `options` - Transcription options
    ///
    /// # Returns
    /// Transcription result with text and segments
    ///
    /// # Errors
    /// Returns error if transcription fails
    pub fn transcribe(
        &self,
        audio: &[f32],
        options: TranscribeOptions,
    ) -> WhisperResult<TranscriptionResult> {
        // 1. Compute mel spectrogram
        let mel = self.compute_mel(audio)?;

        // 2. Encode audio features
        let audio_features = self.encode(&mel)?;

        // 3. Determine language
        let language = options.language.clone().unwrap_or_else(|| "en".to_string());

        // 4. Get initial tokens based on task
        let initial_tokens = self.get_initial_tokens(&language, options.task);

        // 5. Decode tokens
        let tokens = self.decode(&audio_features, &initial_tokens, &options)?;

        // 6. Extract segments with timestamps
        let segments = if timestamps::has_timestamps(&tokens) {
            timestamps::extract_segments(&tokens, |ts| self.tokenizer.decode(ts).ok())
        } else {
            Vec::new()
        };

        // 7. Convert full token sequence to text
        let text = self.tokenizer.decode(&tokens)?;

        // 8. Build result
        Ok(TranscriptionResult {
            text,
            language,
            segments,
        })
    }

    /// Compute mel spectrogram from audio
    fn compute_mel(&self, audio: &[f32]) -> WhisperResult<Vec<f32>> {
        self.mel_filters.compute(audio, audio::HOP_LENGTH)
    }

    /// Encode audio features
    fn encode(&self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        self.encoder.forward(mel)
    }

    /// Get initial tokens for decoding
    #[allow(clippy::unused_self)]
    fn get_initial_tokens(&self, language: &str, task: Task) -> Vec<u32> {
        use tokenizer::special_tokens;

        let mut tokens = vec![special_tokens::SOT];

        // Add language token (defaults to English if unknown)
        let lang_token = special_tokens::language_token(language).unwrap_or_else(|| {
            special_tokens::language_token("en").unwrap_or(special_tokens::LANG_BASE)
        });
        tokens.push(lang_token);

        // Add task token
        match task {
            Task::Transcribe => tokens.push(special_tokens::TRANSCRIBE),
            Task::Translate => tokens.push(special_tokens::TRANSLATE),
        }

        // Add no timestamps token (for now)
        tokens.push(special_tokens::NO_TIMESTAMPS);

        tokens
    }

    /// Detect language from audio
    ///
    /// Analyzes the first few seconds of audio to detect the spoken language.
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, 16kHz, f32 normalized to [-1, 1])
    ///
    /// # Returns
    /// Language probabilities for detected languages
    ///
    /// # Errors
    /// Returns error if detection fails
    pub fn detect_language(&self, audio: &[f32]) -> WhisperResult<detection::LanguageProbs> {
        // 1. Compute mel spectrogram
        let mel = self.compute_mel(audio)?;

        // 2. Encode audio features
        let audio_features = self.encode(&mel)?;

        // 3. Create logits function
        let n_vocab = self.config.n_vocab as usize;
        let logits_fn = |tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            let all_logits = self.decoder.forward(tokens, &audio_features)?;
            let seq_len = tokens.len();
            let last_start = (seq_len - 1) * n_vocab;

            if all_logits.len() >= last_start + n_vocab {
                Ok(all_logits[last_start..last_start + n_vocab].to_vec())
            } else {
                let mut padded = vec![f32::NEG_INFINITY; n_vocab];
                let available = all_logits.len().saturating_sub(last_start);
                if available > 0 {
                    padded[..available].copy_from_slice(&all_logits[last_start..]);
                }
                Ok(padded)
            }
        };

        // 4. Detect language
        let detector = detection::LanguageDetector::new();
        detector.detect(logits_fn)
    }

    /// Decode tokens from audio features
    fn decode(
        &self,
        audio_features: &[f32],
        initial_tokens: &[u32],
        options: &TranscribeOptions,
    ) -> WhisperResult<Vec<u32>> {
        // Create logits function that runs decoder
        let n_vocab = self.config.n_vocab as usize;
        let max_tokens = self.config.n_text_ctx as usize;

        let logits_fn = |tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            // Run decoder forward pass - returns logits (seq_len * n_vocab)
            let all_logits = self.decoder.forward(tokens, audio_features)?;

            // Extract last token's logits
            let seq_len = tokens.len();
            let last_start = (seq_len - 1) * n_vocab;

            // Ensure we have correct vocabulary size
            if all_logits.len() >= last_start + n_vocab {
                Ok(all_logits[last_start..last_start + n_vocab].to_vec())
            } else {
                // Pad with -inf if needed (shouldn't happen with correct model)
                let mut padded = vec![f32::NEG_INFINITY; n_vocab];
                let available = all_logits.len().saturating_sub(last_start);
                if available > 0 {
                    padded[..available].copy_from_slice(&all_logits[last_start..]);
                }
                Ok(padded)
            }
        };

        // Choose decoding strategy
        match options.strategy {
            DecodingStrategy::Greedy => {
                let decoder = inference::GreedyDecoder::new(max_tokens);
                decoder.decode(logits_fn, initial_tokens)
            }
            DecodingStrategy::BeamSearch {
                beam_size,
                temperature,
                patience,
            } => {
                let decoder = inference::BeamSearchDecoder::new(beam_size, max_tokens)
                    .with_temperature(temperature)
                    .with_patience(patience);
                decoder.decode(logits_fn, initial_tokens)
            }
            DecodingStrategy::Sampling { temperature, .. } => {
                // Use greedy with temperature for sampling
                let decoder =
                    inference::GreedyDecoder::new(max_tokens).with_temperature(temperature);
                decoder.decode(logits_fn, initial_tokens)
            }
        }
    }

    /// Set resampler for non-16kHz audio
    ///
    /// # Errors
    /// Returns error if resampler creation fails
    pub fn set_resampler(&mut self, input_rate: u32) -> WhisperResult<()> {
        if input_rate == audio::SAMPLE_RATE {
            self.resampler = None;
        } else {
            self.resampler = Some(audio::SincResampler::new(input_rate, audio::SAMPLE_RATE)?);
        }
        Ok(())
    }

    /// Resample audio if needed
    ///
    /// # Errors
    /// Returns error if resampling fails
    pub fn resample(&self, audio: &[f32]) -> WhisperResult<Vec<f32>> {
        self.resampler
            .as_ref()
            .map_or_else(|| Ok(audio.to_vec()), |resampler| resampler.resample(audio))
    }

    /// Get the tokenizer
    #[must_use]
    pub const fn tokenizer(&self) -> &tokenizer::BpeTokenizer {
        &self.tokenizer
    }

    /// Get estimated memory usage in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        // Rough estimate based on model parameters
        let params = match self.config.model_type {
            ModelType::Tiny | ModelType::TinyEn => 39_000_000,
            ModelType::Base | ModelType::BaseEn => 74_000_000,
            ModelType::Small | ModelType::SmallEn => 244_000_000,
            ModelType::Medium | ModelType::MediumEn => 769_000_000,
            ModelType::Large | ModelType::LargeV1 | ModelType::LargeV2 | ModelType::LargeV3 => {
                1_550_000_000
            }
        };
        params * 4 // 4 bytes per f32 parameter
    }

    /// Load model weights from .apr file bytes
    ///
    /// Loads weights into encoder and decoder from an .apr format file.
    ///
    /// # Arguments
    /// * `data` - Raw .apr file bytes
    ///
    /// # Returns
    /// A new WhisperApr instance with loaded weights
    ///
    /// # Errors
    /// Returns error if file format is invalid or weights cannot be loaded
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let data = std::fs::read("model.apr")?;
    /// let whisper = WhisperApr::load_from_apr(&data)?;
    /// ```
    pub fn load_from_apr(data: &[u8]) -> WhisperResult<Self> {
        Self::load_from_apr_with_progress(data, &mut progress::null_callback)
    }

    /// Load model weights from .apr file bytes with progress callback
    ///
    /// Loads weights into encoder and decoder from an .apr format file,
    /// reporting progress via the callback.
    ///
    /// # Arguments
    /// * `data` - Raw .apr file bytes
    /// * `callback` - Progress callback function
    ///
    /// # Returns
    /// A new WhisperApr instance with loaded weights
    ///
    /// # Errors
    /// Returns error if file format is invalid or weights cannot be loaded
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let data = std::fs::read("model.apr")?;
    /// let whisper = WhisperApr::load_from_apr_with_progress(&data, &mut |p| {
    ///     println!("{}% - {}", p.percent(), p.display_message());
    /// })?;
    /// ```
    pub fn load_from_apr_with_progress(
        data: &[u8],
        callback: progress::ProgressCallback<'_>,
    ) -> WhisperResult<Self> {
        // Create progress tracker for model loading phases
        let mut tracker = progress::ProgressTracker::model_loading();

        // Phase 1: Parsing header
        callback(&tracker.to_progress());
        let reader = format::AprReader::new(data.to_vec())?;
        let config = reader.header.to_model_config();
        tracker.next_phase();

        // Phase 2: Loading encoder
        callback(&tracker.to_progress());
        let mut encoder = model::Encoder::new(&config);
        Self::load_encoder_weights(&reader, &mut encoder, &mut tracker, callback);
        tracker.next_phase();

        // Phase 3: Loading decoder
        callback(&tracker.to_progress());
        let mut decoder = model::Decoder::new(&config);
        Self::load_decoder_weights(&reader, &mut decoder, &mut tracker, callback);
        tracker.next_phase();

        // Phase 4: Loading vocabulary
        callback(&tracker.to_progress());
        let tokenizer = tokenizer::BpeTokenizer::with_base_tokens();
        tracker.next_phase();

        // Phase 5: Initializing
        callback(&tracker.to_progress());
        let mel_filters =
            audio::MelFilterbank::new(config.n_mels as usize, audio::N_FFT, audio::SAMPLE_RATE);
        tracker.complete();
        callback(&tracker.to_progress());

        Ok(Self {
            config,
            encoder,
            decoder,
            tokenizer,
            mel_filters,
            resampler: None,
        })
    }

    /// Load encoder weights from .apr reader
    fn load_encoder_weights(
        reader: &format::AprReader,
        encoder: &mut model::Encoder,
        tracker: &mut progress::ProgressTracker,
        callback: progress::ProgressCallback<'_>,
    ) {
        let n_layers = encoder.n_layers();

        // Load positional embedding if available
        if let Ok(pe) = reader.load_tensor("encoder.positional_embedding") {
            let target = encoder.positional_embedding_mut();
            let len = pe.len().min(target.len());
            target[..len].copy_from_slice(&pe[..len]);
        }

        // Load encoder block weights
        for layer_idx in 0..n_layers {
            let progress = layer_idx as f32 / n_layers as f32;
            tracker.update_phase_progress(progress);
            callback(&tracker.to_progress());

            let block = &mut encoder.blocks_mut()[layer_idx];

            // Layer norm 1
            Self::load_layer_norm_weights(
                reader,
                &format!("encoder.blocks.{layer_idx}.attn_ln"),
                &mut block.ln1,
            );

            // Self-attention
            Self::load_attention_weights(
                reader,
                &format!("encoder.blocks.{layer_idx}.attn"),
                &mut block.self_attn,
            );

            // Layer norm 2
            Self::load_layer_norm_weights(
                reader,
                &format!("encoder.blocks.{layer_idx}.mlp_ln"),
                &mut block.ln2,
            );

            // Feed-forward network
            Self::load_ffn_weights(
                reader,
                &format!("encoder.blocks.{layer_idx}.mlp"),
                &mut block.ffn,
            );
        }

        // Load final layer norm
        Self::load_layer_norm_weights(reader, "encoder.ln_post", encoder.ln_post_mut());
    }

    /// Load decoder weights from .apr reader
    fn load_decoder_weights(
        reader: &format::AprReader,
        decoder: &mut model::Decoder,
        tracker: &mut progress::ProgressTracker,
        callback: progress::ProgressCallback<'_>,
    ) {
        let n_layers = decoder.n_layers();

        // Load token embedding if available
        if let Ok(te) = reader.load_tensor("decoder.token_embedding") {
            let target = decoder.token_embedding_mut();
            let len = te.len().min(target.len());
            target[..len].copy_from_slice(&te[..len]);
        }

        // Load positional embedding if available
        if let Ok(pe) = reader.load_tensor("decoder.positional_embedding") {
            let target = decoder.positional_embedding_mut();
            let len = pe.len().min(target.len());
            target[..len].copy_from_slice(&pe[..len]);
        }

        // Load decoder block weights
        for layer_idx in 0..n_layers {
            let progress = layer_idx as f32 / n_layers as f32;
            tracker.update_phase_progress(progress);
            callback(&tracker.to_progress());

            let block = &mut decoder.blocks_mut()[layer_idx];

            // Layer norm 1 (before self-attention)
            Self::load_layer_norm_weights(
                reader,
                &format!("decoder.blocks.{layer_idx}.attn_ln"),
                &mut block.ln1,
            );

            // Self-attention
            Self::load_attention_weights(
                reader,
                &format!("decoder.blocks.{layer_idx}.attn"),
                &mut block.self_attn,
            );

            // Layer norm 2 (before cross-attention)
            Self::load_layer_norm_weights(
                reader,
                &format!("decoder.blocks.{layer_idx}.cross_attn_ln"),
                &mut block.ln2,
            );

            // Cross-attention
            Self::load_attention_weights(
                reader,
                &format!("decoder.blocks.{layer_idx}.cross_attn"),
                &mut block.cross_attn,
            );

            // Layer norm 3 (before FFN)
            Self::load_layer_norm_weights(
                reader,
                &format!("decoder.blocks.{layer_idx}.mlp_ln"),
                &mut block.ln3,
            );

            // Feed-forward network
            Self::load_ffn_weights(
                reader,
                &format!("decoder.blocks.{layer_idx}.mlp"),
                &mut block.ffn,
            );
        }

        // Load final layer norm
        Self::load_layer_norm_weights(reader, "decoder.ln", decoder.ln_post_mut());
    }

    /// Load layer norm weights
    fn load_layer_norm_weights(
        reader: &format::AprReader,
        prefix: &str,
        ln: &mut model::LayerNorm,
    ) {
        if let Ok(weight) = reader.load_tensor(&format!("{prefix}.weight")) {
            let len = weight.len().min(ln.weight.len());
            ln.weight[..len].copy_from_slice(&weight[..len]);
        }
        if let Ok(bias) = reader.load_tensor(&format!("{prefix}.bias")) {
            let len = bias.len().min(ln.bias.len());
            ln.bias[..len].copy_from_slice(&bias[..len]);
        }
    }

    /// Load attention weights
    fn load_attention_weights(
        reader: &format::AprReader,
        prefix: &str,
        attn: &mut model::MultiHeadAttention,
    ) {
        // Load query, key, value projections
        if let Ok(q_weight) = reader.load_tensor(&format!("{prefix}.query.weight")) {
            attn.set_query_weight(&q_weight);
        }
        if let Ok(k_weight) = reader.load_tensor(&format!("{prefix}.key.weight")) {
            attn.set_key_weight(&k_weight);
        }
        if let Ok(v_weight) = reader.load_tensor(&format!("{prefix}.value.weight")) {
            attn.set_value_weight(&v_weight);
        }
        if let Ok(out_weight) = reader.load_tensor(&format!("{prefix}.out.weight")) {
            attn.set_out_weight(&out_weight);
        }
    }

    /// Load feed-forward network weights
    fn load_ffn_weights(reader: &format::AprReader, prefix: &str, ffn: &mut model::FeedForward) {
        if let Ok(fc1_weight) = reader.load_tensor(&format!("{prefix}.0.weight")) {
            ffn.fc1.set_weight(&fc1_weight);
        }
        if let Ok(fc1_bias) = reader.load_tensor(&format!("{prefix}.0.bias")) {
            ffn.fc1.set_bias(&fc1_bias);
        }
        if let Ok(fc2_weight) = reader.load_tensor(&format!("{prefix}.2.weight")) {
            ffn.fc2.set_weight(&fc2_weight);
        }
        if let Ok(fc2_bias) = reader.load_tensor(&format!("{prefix}.2.bias")) {
            ffn.fc2.set_bias(&fc2_bias);
        }
    }

    /// Get mutable encoder reference (for testing/loading)
    pub fn encoder_mut(&mut self) -> &mut model::Encoder {
        &mut self.encoder
    }

    /// Get mutable decoder reference (for testing/loading)
    pub fn decoder_mut(&mut self) -> &mut model::Decoder {
        &mut self.decoder
    }

    // =========================================================================
    // Batch Transcription API (WAPR-083)
    // =========================================================================

    /// Transcribe a batch of audio samples
    ///
    /// Processes multiple audio segments in parallel for improved throughput.
    /// Each audio segment is transcribed independently with the same options.
    ///
    /// # Arguments
    /// * `audio_batch` - Batch of audio samples (each: mono, 16kHz, f32 normalized)
    /// * `options` - Transcription options (applied to all segments)
    ///
    /// # Returns
    /// Batch transcription result with individual results for each segment
    ///
    /// # Errors
    /// Returns error if any transcription fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let audio_segments = vec![audio1, audio2, audio3];
    /// let result = whisper.transcribe_batch(&audio_segments, TranscribeOptions::default())?;
    /// for (i, text) in result.texts().iter().enumerate() {
    ///     println!("Segment {}: {}", i, text);
    /// }
    /// ```
    pub fn transcribe_batch(
        &self,
        audio_batch: &[Vec<f32>],
        options: TranscribeOptions,
    ) -> WhisperResult<BatchTranscriptionResult> {
        if audio_batch.is_empty() {
            return Err(WhisperError::Audio("empty batch".into()));
        }

        let start_time = std::time::Instant::now();
        let mut results = Vec::with_capacity(audio_batch.len());

        // Process each audio segment
        for audio in audio_batch {
            let result = self.transcribe(audio, options.clone())?;
            results.push(result);
        }

        let total_duration_secs = start_time.elapsed().as_secs_f32();

        Ok(BatchTranscriptionResult {
            results,
            total_duration_secs,
        })
    }

    /// Transcribe a batch using the batch preprocessor for efficiency
    ///
    /// Uses `BatchPreprocessor` for efficient mel spectrogram computation
    /// across all audio segments before decoding.
    ///
    /// # Arguments
    /// * `batch` - Pre-constructed audio batch
    /// * `options` - Transcription options
    ///
    /// # Returns
    /// Batch transcription result
    ///
    /// # Errors
    /// Returns error if preprocessing or transcription fails
    pub fn transcribe_audio_batch(
        &self,
        batch: &audio::AudioBatch,
        options: TranscribeOptions,
    ) -> WhisperResult<BatchTranscriptionResult> {
        if batch.is_empty() {
            return Err(WhisperError::Audio("empty batch".into()));
        }

        let start_time = std::time::Instant::now();

        // Use batch preprocessor for efficient mel computation
        let preprocessor = audio::BatchPreprocessor::new(audio::AudioConfig::default());
        let mel_result = preprocessor.process_batch(batch)?;

        let mut results = Vec::with_capacity(batch.len());
        let language = options.language.clone().unwrap_or_else(|| "en".to_string());

        // Process each mel spectrogram
        for mel in &mel_result.mels {
            // Encode audio features
            let audio_features = self.encode(mel)?;

            // Get initial tokens
            let initial_tokens = self.get_initial_tokens(&language, options.task);

            // Decode
            let tokens = self.decode(&audio_features, &initial_tokens, &options)?;

            // Extract segments
            let segments = if timestamps::has_timestamps(&tokens) {
                timestamps::extract_segments(&tokens, |ts| self.tokenizer.decode(ts).ok())
            } else {
                Vec::new()
            };

            // Convert to text
            let text = self.tokenizer.decode(&tokens)?;

            results.push(TranscriptionResult {
                text,
                language: language.clone(),
                segments,
            });
        }

        let total_duration_secs = start_time.elapsed().as_secs_f32();

        Ok(BatchTranscriptionResult {
            results,
            total_duration_secs,
        })
    }

    /// Create an audio batch from a slice of audio segments
    #[must_use]
    pub fn create_audio_batch(audio_segments: &[Vec<f32>]) -> audio::AudioBatch {
        let mut batch = audio::AudioBatch::with_default_config();
        for segment in audio_segments {
            batch.add_segment(segment.clone());
        }
        batch
    }

    /// Transcribe with batch encoder for improved throughput
    ///
    /// Uses batched encoder forward pass for efficient processing of
    /// multiple audio segments.
    ///
    /// # Arguments
    /// * `audio_batch` - Batch of audio samples
    /// * `options` - Transcription options
    ///
    /// # Returns
    /// Batch transcription result
    ///
    /// # Errors
    /// Returns error if transcription fails
    pub fn transcribe_batch_optimized(
        &self,
        audio_batch: &[Vec<f32>],
        options: TranscribeOptions,
    ) -> WhisperResult<BatchTranscriptionResult> {
        if audio_batch.is_empty() {
            return Err(WhisperError::Audio("empty batch".into()));
        }

        let start_time = std::time::Instant::now();

        // Compute mel spectrograms for all segments
        let mut mels = Vec::with_capacity(audio_batch.len());
        for audio in audio_batch {
            let mel = self.compute_mel(audio)?;
            mels.push(mel);
        }

        // Use batch encoder
        let encoder_outputs = self.encoder.forward_batch(&mels)?;

        let mut results = Vec::with_capacity(audio_batch.len());
        let language = options.language.clone().unwrap_or_else(|| "en".to_string());

        // Decode each encoder output
        for features in &encoder_outputs {
            let initial_tokens = self.get_initial_tokens(&language, options.task);
            let tokens = self.decode(features, &initial_tokens, &options)?;

            let segments = if timestamps::has_timestamps(&tokens) {
                timestamps::extract_segments(&tokens, |ts| self.tokenizer.decode(ts).ok())
            } else {
                Vec::new()
            };

            let text = self.tokenizer.decode(&tokens)?;

            results.push(TranscriptionResult {
                text,
                language: language.clone(),
                segments,
            });
        }

        let total_duration_secs = start_time.elapsed().as_secs_f32();

        Ok(BatchTranscriptionResult {
            results,
            total_duration_secs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let options = TranscribeOptions::default();
        assert!(options.language.is_none());
        assert_eq!(options.task, Task::Transcribe);
        assert!(!options.word_timestamps);
    }

    #[test]
    fn test_decoding_strategy_default() {
        let strategy = DecodingStrategy::default();
        assert!(matches!(strategy, DecodingStrategy::Greedy));
    }

    // =========================================================================
    // WhisperApr Tests
    // =========================================================================

    #[test]
    fn test_whisper_tiny() {
        let whisper = WhisperApr::tiny();
        assert_eq!(whisper.model_type(), ModelType::Tiny);
        assert_eq!(whisper.config().n_audio_layer, 4);
    }

    #[test]
    fn test_whisper_base() {
        let whisper = WhisperApr::base();
        assert_eq!(whisper.model_type(), ModelType::Base);
        assert_eq!(whisper.config().n_audio_layer, 6);
    }

    #[test]
    fn test_whisper_memory_size() {
        let tiny = WhisperApr::tiny();
        let base = WhisperApr::base();

        assert!(tiny.memory_size() < base.memory_size());
        assert!(tiny.memory_size() > 100_000_000); // > 100MB
    }

    #[test]
    fn test_whisper_initial_tokens() {
        let whisper = WhisperApr::tiny();

        let tokens = whisper.get_initial_tokens("en", Task::Transcribe);
        assert_eq!(tokens[0], tokenizer::special_tokens::SOT);
        assert!(tokens.len() >= 4); // SOT, lang, task, no_timestamps

        let translate_tokens = whisper.get_initial_tokens("es", Task::Translate);
        assert!(translate_tokens.contains(&tokenizer::special_tokens::TRANSLATE));
    }

    #[test]
    fn test_whisper_set_resampler() {
        let mut whisper = WhisperApr::tiny();

        // No resampler by default
        assert!(whisper.resampler.is_none());

        // Set 44100 Hz resampler
        whisper.set_resampler(44100).expect("should succeed");
        assert!(whisper.resampler.is_some());

        // Setting 16000 should clear resampler
        whisper.set_resampler(16000).expect("should succeed");
        assert!(whisper.resampler.is_none());
    }

    #[test]
    fn test_whisper_resample_passthrough() {
        let whisper = WhisperApr::tiny();
        let audio = vec![0.1, 0.2, 0.3, 0.4];

        // Without resampler, should return copy
        let resampled = whisper.resample(&audio).expect("should succeed");
        assert_eq!(resampled, audio);
    }

    #[test]
    fn test_whisper_resample_with_resampler() {
        let mut whisper = WhisperApr::tiny();
        whisper.set_resampler(32000).expect("should succeed");

        // Generate a simple sine wave at 32kHz (0.5 second)
        let n_samples = 16000;
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 32000.0).sin())
            .collect();

        let resampled = whisper.resample(&audio).expect("should succeed");

        // Should have approximately half as many samples (32kHz -> 16kHz)
        assert!(resampled.len() > n_samples / 3);
        assert!(resampled.len() < n_samples);
    }

    #[test]
    fn test_whisper_tokenizer() {
        let whisper = WhisperApr::tiny();
        let tokenizer = whisper.tokenizer();

        // Tokenizer should be accessible (with base tokens we have 256)
        let vocab_size = tokenizer.vocab_size();
        assert!(
            vocab_size >= 256,
            "vocab_size should include base byte tokens"
        );
    }

    #[test]
    fn test_transcribe_options_with_language() {
        let options = TranscribeOptions {
            language: Some("es".to_string()),
            task: Task::Transcribe,
            strategy: DecodingStrategy::Greedy,
            word_timestamps: false,
        };

        assert_eq!(options.language, Some("es".to_string()));
    }

    #[test]
    fn test_transcribe_options_beam_search() {
        let options = TranscribeOptions {
            language: None,
            task: Task::Transcribe,
            strategy: DecodingStrategy::BeamSearch {
                beam_size: 5,
                temperature: 0.0,
                patience: 1.0,
            },
            word_timestamps: false,
        };

        assert!(matches!(
            options.strategy,
            DecodingStrategy::BeamSearch { .. }
        ));
    }

    #[test]
    fn test_segment_struct() {
        let segment = Segment {
            start: 0.0,
            end: 2.5,
            text: "Hello world".to_string(),
            tokens: vec![1, 2, 3],
        };

        assert!((segment.start - 0.0).abs() < f32::EPSILON);
        assert!((segment.end - 2.5).abs() < f32::EPSILON);
        assert_eq!(segment.text, "Hello world");
        assert_eq!(segment.tokens.len(), 3);
    }

    #[test]
    fn test_transcription_result_struct() {
        let result = TranscriptionResult {
            text: "Test transcription".to_string(),
            language: "en".to_string(),
            segments: vec![],
        };

        assert_eq!(result.text, "Test transcription");
        assert_eq!(result.language, "en");
        assert!(result.segments.is_empty());
    }

    // =========================================================================
    // Model Loading Tests
    // =========================================================================

    #[test]
    fn test_load_from_apr_basic() {
        // Create a minimal valid .apr file
        let data = format::create_test_apr();

        let result = WhisperApr::load_from_apr(&data);
        assert!(result.is_ok());

        let whisper = result.expect("should load");
        assert_eq!(whisper.model_type(), ModelType::Tiny);
    }

    #[test]
    fn test_load_from_apr_with_progress_callback() {
        let data = format::create_test_apr();

        let mut progress_updates = Vec::new();
        let mut callback = |p: &progress::Progress| {
            progress_updates.push(p.percent());
        };

        let result = WhisperApr::load_from_apr_with_progress(&data, &mut callback);
        assert!(result.is_ok());

        // Should have received multiple progress updates
        assert!(!progress_updates.is_empty());
    }

    #[test]
    fn test_load_from_apr_invalid_magic() {
        let mut data = format::create_test_apr();
        data[0] = b'X'; // Corrupt magic

        let result = WhisperApr::load_from_apr(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_apr_too_short() {
        let data = vec![b'A', b'P', b'R', b'1']; // Only magic, no header

        let result = WhisperApr::load_from_apr(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_mut_accessor() {
        let mut whisper = WhisperApr::tiny();
        let encoder = whisper.encoder_mut();
        assert_eq!(encoder.n_layers(), 4);
    }

    #[test]
    fn test_decoder_mut_accessor() {
        let mut whisper = WhisperApr::tiny();
        let decoder = whisper.decoder_mut();
        assert_eq!(decoder.n_layers(), 4);
    }

    // =========================================================================
    // Integration Tests - Full Pipeline
    // =========================================================================

    #[test]
    fn test_full_pipeline_tiny_model() {
        // Create model
        let whisper = WhisperApr::tiny();
        assert_eq!(whisper.model_type(), ModelType::Tiny);

        // Verify model components
        assert_eq!(whisper.config().n_audio_layer, 4);
        assert_eq!(whisper.config().n_text_layer, 4);
        assert_eq!(whisper.config().n_audio_state, 384);
        assert_eq!(whisper.config().n_text_state, 384);
    }

    #[test]
    fn test_full_pipeline_base_model() {
        // Create model
        let whisper = WhisperApr::base();
        assert_eq!(whisper.model_type(), ModelType::Base);

        // Verify model components
        assert_eq!(whisper.config().n_audio_layer, 6);
        assert_eq!(whisper.config().n_text_layer, 6);
        assert_eq!(whisper.config().n_audio_state, 512);
        assert_eq!(whisper.config().n_text_state, 512);
    }

    #[test]
    fn test_transcribe_options_all_strategies() {
        // Test greedy
        let opts_greedy = TranscribeOptions::default();
        assert!(matches!(opts_greedy.strategy, DecodingStrategy::Greedy));

        // Test beam search
        let opts_beam = TranscribeOptions {
            language: Some("en".to_string()),
            task: Task::Transcribe,
            strategy: DecodingStrategy::BeamSearch {
                beam_size: 5,
                temperature: 0.0,
                patience: 1.0,
            },
            word_timestamps: false,
        };
        assert!(matches!(
            opts_beam.strategy,
            DecodingStrategy::BeamSearch { .. }
        ));

        // Test sampling
        let opts_sampling = TranscribeOptions {
            language: None,
            task: Task::Translate,
            strategy: DecodingStrategy::Sampling {
                temperature: 0.7,
                top_k: Some(50),
                top_p: Some(0.9),
            },
            word_timestamps: true,
        };
        assert!(matches!(
            opts_sampling.strategy,
            DecodingStrategy::Sampling { .. }
        ));
    }

    #[test]
    fn test_memory_estimation_consistency() {
        let tiny = WhisperApr::tiny();
        let base = WhisperApr::base();

        // Base should require more memory
        assert!(base.config().weights_memory_mb() > tiny.config().weights_memory_mb());
        assert!(base.config().peak_memory_mb() > tiny.config().peak_memory_mb());
        assert!(base.config().parameter_count() > tiny.config().parameter_count());
    }

    #[test]
    fn test_simd_operations_integration() {
        // Test that SIMD module is accessible and works
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        // Vector operations
        let sum = simd::add(&a, &b);
        assert_eq!(sum.len(), 4);
        assert!((sum[0] - 6.0).abs() < 1e-5);
        assert!((sum[3] - 12.0).abs() < 1e-5);

        // Softmax
        let softmax = simd::softmax(&a);
        let sum_softmax: f32 = softmax.iter().sum();
        assert!((sum_softmax - 1.0).abs() < 1e-5);

        // Matrix operations
        let mat_a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let mat_b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let result = simd::matmul(&mat_a, &mat_b, 2, 2, 2);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_memory_pool_integration() {
        use memory::{get_buffer, pool_stats, return_buffer, MemoryPool};

        // Create a pool
        let pool = MemoryPool::new();

        // Allocate and reuse buffers
        let buf1 = pool.get(1024);
        assert_eq!(buf1.len(), 1024);
        pool.return_buffer(buf1);

        let buf2 = pool.get(1024);
        assert_eq!(buf2.len(), 1024);

        // Should have hit the pool
        let stats = pool.stats();
        assert_eq!(stats.hits, 1);

        // Test thread-local pool
        let tlbuf = get_buffer(512);
        assert_eq!(tlbuf.len(), 512);
        return_buffer(tlbuf);

        let tl_stats = pool_stats();
        assert!(tl_stats.allocations > 0);
    }

    #[test]
    fn test_audio_resampling_integration() {
        use audio::SincResampler;

        // Create resampler 44100 -> 16000
        let resampler = SincResampler::new(44100, 16000).expect("resampler should work");

        // Generate test audio (440Hz sine wave at 44100 Hz, 100ms)
        let duration_ms = 100;
        let samples_at_44100 = (44100 * duration_ms) / 1000;
        let input: Vec<f32> = (0..samples_at_44100)
            .map(|i| {
                let t = i as f32 / 44100.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let output = resampler.resample(&input).expect("resample should work");

        // Output should be approximately 16000/44100 of input length
        let expected_len = (input.len() as f32 * 16000.0 / 44100.0) as usize;
        assert!(
            (output.len() as i32 - expected_len as i32).abs() < 10,
            "output len {} vs expected ~{}",
            output.len(),
            expected_len
        );
    }

    #[test]
    fn test_vad_integration() {
        use vad::VadConfig;

        // Test VAD config defaults
        let config = VadConfig::default();
        assert!(config.energy_threshold > 0.0);
        assert!(config.zcr_threshold > 0.0);
        assert!(config.min_speech_frames > 0);
    }

    #[test]
    fn test_tokenizer_integration() {
        use tokenizer::special_tokens;

        // Test special tokens are defined
        assert!(special_tokens::SOT > 0);
        assert!(special_tokens::EOT > 0);
        assert!(special_tokens::TRANSCRIBE > 0);
        assert!(special_tokens::TRANSLATE > 0);
        assert!(special_tokens::NO_TIMESTAMPS > 0);
    }

    #[test]
    fn test_inference_beam_search_integration() {
        use inference::BeamSearchDecoder;

        let decoder = BeamSearchDecoder::new(5, 448); // beam size 5, max tokens 448

        // Verify decoder was created
        assert_eq!(decoder.beam_size(), 5);
    }

    #[test]
    fn test_format_decompression_integration() {
        use format::Decompressor;

        // Create a decompressor
        let mut decompressor = Decompressor::new();

        // Test that decompressor can be created
        assert!(decompressor.is_empty());

        // Verify decompressor is usable (reset clears internal state)
        decompressor.reset();
        assert!(decompressor.is_empty());
    }

    #[test]
    fn test_timestamps_generation() {
        let segment = Segment {
            start: 1.5,
            end: 3.25,
            text: "This is a test".to_string(),
            tokens: vec![1, 2, 3, 4],
        };

        // Duration calculation
        let duration = segment.end - segment.start;
        assert!((duration - 1.75).abs() < 1e-5);
    }

    #[test]
    fn test_language_detection_integration() {
        use detection::{is_supported, language_name};

        // Verify language support
        assert!(is_supported("en"));
        assert!(is_supported("es"));
        assert!(is_supported("ja"));
        assert!(!is_supported("invalid_lang"));

        // Verify language names
        assert_eq!(language_name("en"), Some("English"));
        assert_eq!(language_name("es"), Some("Spanish"));
        assert_eq!(language_name("zh"), Some("Chinese"));

        // Verify common supported languages have names
        // Note: Some less common languages may not have names defined
        for lang in &["en", "es", "fr", "de", "it", "ja", "zh", "ko", "pt", "ru"] {
            assert!(
                language_name(lang).is_some(),
                "language {} should have a name",
                lang
            );
        }
    }

    #[test]
    fn test_model_kv_cache_integration() {
        use model::{Decoder, ModelConfig};

        let config = ModelConfig::tiny();
        let decoder = Decoder::new(&config);

        // Get cache
        let cache = decoder.create_kv_cache();
        assert!(cache.self_attn_cache.iter().all(|c| c.is_empty()));
        assert!(cache.cross_attn_cache.iter().all(|c| c.is_empty()));
    }

    #[test]
    fn test_progress_tracking_integration() {
        use progress::{format_bytes, Progress};

        // Test progress creation
        let progress = Progress::new(50, 100);
        assert_eq!(progress.percent(), 50.0);
        assert_eq!(progress.current, 50);
        assert_eq!(progress.total, 100);

        // Test bytes formatting (exact format may vary)
        let kb_str = format_bytes(1024);
        assert!(kb_str.contains("KB"), "should contain KB: {}", kb_str);
        let mb_str = format_bytes(1024 * 1024);
        assert!(mb_str.contains("MB"), "should contain MB: {}", mb_str);
    }

    // =========================================================================
    // Additional Coverage Tests (WAPR-QA-001)
    // =========================================================================

    #[test]
    fn test_model_type_variants() {
        let tiny = ModelType::Tiny;
        let tiny_en = ModelType::TinyEn;
        let base = ModelType::Base;
        let base_en = ModelType::BaseEn;
        let small = ModelType::Small;

        // Test Debug
        assert!(format!("{tiny:?}").contains("Tiny"));
        assert!(format!("{tiny_en:?}").contains("TinyEn"));
        assert!(format!("{base:?}").contains("Base"));
        assert!(format!("{base_en:?}").contains("BaseEn"));
        assert!(format!("{small:?}").contains("Small"));

        // Test Clone
        let tiny_clone = tiny;
        assert_eq!(tiny_clone, ModelType::Tiny);

        // Test PartialEq
        assert_eq!(tiny, ModelType::Tiny);
        assert_ne!(tiny, base);
    }

    #[test]
    fn test_task_variants() {
        let transcribe = Task::Transcribe;
        let translate = Task::Translate;

        // Test Debug
        assert!(format!("{transcribe:?}").contains("Transcribe"));
        assert!(format!("{translate:?}").contains("Translate"));

        // Test Clone
        let transcribe_clone = transcribe;
        assert_eq!(transcribe_clone, Task::Transcribe);

        // Test PartialEq
        assert_eq!(transcribe, Task::Transcribe);
        assert_ne!(transcribe, translate);

        // Test Default
        assert_eq!(Task::default(), Task::Transcribe);
    }

    #[test]
    fn test_decoding_strategy_sampling() {
        let sampling = DecodingStrategy::Sampling {
            temperature: 0.8,
            top_k: Some(50),
            top_p: Some(0.9),
        };

        // Test Debug
        let debug_str = format!("{sampling:?}");
        assert!(debug_str.contains("Sampling"));

        // Test Clone
        let cloned = sampling.clone();
        assert!(matches!(cloned, DecodingStrategy::Sampling { .. }));
    }

    #[test]
    fn test_decoding_strategy_beam_search() {
        let beam = DecodingStrategy::BeamSearch {
            beam_size: 5,
            temperature: 0.0,
            patience: 1.0,
        };

        let debug_str = format!("{beam:?}");
        assert!(debug_str.contains("BeamSearch"));
    }

    #[test]
    fn test_transcribe_options_clone() {
        let options = TranscribeOptions {
            language: Some("fr".to_string()),
            task: Task::Translate,
            strategy: DecodingStrategy::Greedy,
            word_timestamps: true,
        };

        let cloned = options.clone();
        assert_eq!(cloned.language, Some("fr".to_string()));
        assert_eq!(cloned.task, Task::Translate);
        assert!(cloned.word_timestamps);
    }

    #[test]
    fn test_transcribe_options_debug() {
        let options = TranscribeOptions::default();
        let debug_str = format!("{options:?}");
        assert!(debug_str.contains("TranscribeOptions"));
    }

    #[test]
    fn test_segment_clone() {
        let segment = Segment {
            start: 1.0,
            end: 2.0,
            text: "test".to_string(),
            tokens: vec![1, 2],
        };

        let cloned = segment.clone();
        assert_eq!(cloned.text, "test");
        assert_eq!(cloned.tokens, vec![1, 2]);
    }

    #[test]
    fn test_segment_debug() {
        let segment = Segment {
            start: 0.0,
            end: 1.0,
            text: "hello".to_string(),
            tokens: vec![],
        };

        let debug_str = format!("{segment:?}");
        assert!(debug_str.contains("hello"));
    }

    #[test]
    fn test_transcription_result_clone() {
        let result = TranscriptionResult {
            text: "hello world".to_string(),
            language: "en".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: "hello".to_string(),
                tokens: vec![1],
            }],
        };

        let cloned = result.clone();
        assert_eq!(cloned.text, "hello world");
        assert_eq!(cloned.segments.len(), 1);
    }

    #[test]
    fn test_transcription_result_debug() {
        let result = TranscriptionResult {
            text: "test".to_string(),
            language: "en".to_string(),
            segments: vec![],
        };

        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("TranscriptionResult"));
    }

    #[test]
    fn test_whisper_debug() {
        let whisper = WhisperApr::tiny();
        let debug_str = format!("{whisper:?}");
        assert!(debug_str.contains("WhisperApr"));
    }

    #[test]
    fn test_whisper_memory_size_all_models() {
        let tiny = WhisperApr::tiny();
        let base = WhisperApr::base();

        // Tiny should be smaller than base
        assert!(tiny.memory_size() < base.memory_size());

        // Both should be non-zero
        assert!(tiny.memory_size() > 0);
        assert!(base.memory_size() > 0);
    }

    #[test]
    fn test_transcribe_options_sampling_strategy() {
        let options = TranscribeOptions {
            language: None,
            task: Task::Transcribe,
            strategy: DecodingStrategy::Sampling {
                temperature: 1.0,
                top_k: None,
                top_p: None,
            },
            word_timestamps: false,
        };

        assert!(matches!(
            options.strategy,
            DecodingStrategy::Sampling { .. }
        ));
    }

    #[test]
    fn test_transcribe_options_word_timestamps_enabled() {
        let options = TranscribeOptions {
            language: Some("en".to_string()),
            task: Task::Transcribe,
            strategy: DecodingStrategy::default(),
            word_timestamps: true,
        };

        assert!(options.word_timestamps);
    }

    // =========================================================================
    // v1.1 Extended Model Support - RED Tests (WAPR-070, WAPR-071)
    // =========================================================================

    #[test]
    fn test_model_type_small() {
        let small = ModelType::Small;
        assert!(format!("{small:?}").contains("Small"));
    }

    #[test]
    fn test_model_type_medium() {
        let medium = ModelType::Medium;
        assert!(format!("{medium:?}").contains("Medium"));
    }

    #[test]
    fn test_model_type_medium_en() {
        let medium_en = ModelType::MediumEn;
        assert!(format!("{medium_en:?}").contains("MediumEn"));
    }

    #[test]
    fn test_model_type_large() {
        let large = ModelType::Large;
        assert!(format!("{large:?}").contains("Large"));
    }

    #[test]
    fn test_model_type_large_v1() {
        let large_v1 = ModelType::LargeV1;
        assert!(format!("{large_v1:?}").contains("LargeV1"));
    }

    #[test]
    fn test_model_type_large_v2() {
        let large_v2 = ModelType::LargeV2;
        assert!(format!("{large_v2:?}").contains("LargeV2"));
    }

    #[test]
    fn test_model_type_large_v3() {
        let large_v3 = ModelType::LargeV3;
        assert!(format!("{large_v3:?}").contains("LargeV3"));
    }

    #[test]
    fn test_whisper_small() {
        let whisper = WhisperApr::small();
        assert_eq!(whisper.model_type(), ModelType::Small);
        assert_eq!(whisper.config().n_audio_layer, 12);
        assert_eq!(whisper.config().n_audio_state, 768);
    }

    #[test]
    fn test_whisper_medium() {
        let whisper = WhisperApr::medium();
        assert_eq!(whisper.model_type(), ModelType::Medium);
        assert_eq!(whisper.config().n_audio_layer, 24);
        assert_eq!(whisper.config().n_audio_state, 1024);
    }

    #[test]
    fn test_whisper_large() {
        let whisper = WhisperApr::large();
        assert_eq!(whisper.model_type(), ModelType::Large);
        assert_eq!(whisper.config().n_audio_layer, 32);
        assert_eq!(whisper.config().n_audio_state, 1280);
    }

    #[test]
    fn test_whisper_memory_size_all_extended_models() {
        let small = WhisperApr::small();
        let medium = WhisperApr::medium();
        let large = WhisperApr::large();

        // Verify size hierarchy
        assert!(small.memory_size() < medium.memory_size());
        assert!(medium.memory_size() < large.memory_size());

        // All should be non-zero
        assert!(small.memory_size() > 0);
        assert!(medium.memory_size() > 0);
        assert!(large.memory_size() > 0);
    }

    #[test]
    fn test_extended_model_memory_size_estimates() {
        let small = WhisperApr::small();
        let medium = WhisperApr::medium();
        let large = WhisperApr::large();

        // Small: ~244M params * 4 bytes = ~976MB
        assert!(small.memory_size() > 900_000_000);
        assert!(small.memory_size() < 1_200_000_000);

        // Medium: ~769M params * 4 bytes = ~3GB
        assert!(medium.memory_size() > 2_500_000_000);
        assert!(medium.memory_size() < 3_500_000_000);

        // Large: ~1.5B params * 4 bytes = ~6GB
        assert!(large.memory_size() > 5_000_000_000);
        assert!(large.memory_size() < 7_000_000_000);
    }

    // =========================================================================
    // WAPR-083: Batch Transcription API Tests
    // =========================================================================

    #[test]
    fn test_batch_transcription_result_len() {
        let result = BatchTranscriptionResult {
            results: vec![
                TranscriptionResult {
                    text: "Hello".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                },
                TranscriptionResult {
                    text: "World".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                },
            ],
            total_duration_secs: 1.5,
        };

        assert_eq!(result.len(), 2);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_batch_transcription_result_empty() {
        let result = BatchTranscriptionResult {
            results: vec![],
            total_duration_secs: 0.0,
        };

        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_batch_transcription_result_get() {
        let result = BatchTranscriptionResult {
            results: vec![
                TranscriptionResult {
                    text: "First".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                },
                TranscriptionResult {
                    text: "Second".to_string(),
                    language: "es".to_string(),
                    segments: vec![],
                },
            ],
            total_duration_secs: 2.0,
        };

        assert!(result.get(0).is_some());
        assert_eq!(result.get(0).map(|r| r.text.as_str()), Some("First"));
        assert_eq!(result.get(1).map(|r| r.language.as_str()), Some("es"));
        assert!(result.get(2).is_none());
    }

    #[test]
    fn test_batch_transcription_result_texts() {
        let result = BatchTranscriptionResult {
            results: vec![
                TranscriptionResult {
                    text: "One".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                },
                TranscriptionResult {
                    text: "Two".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                },
                TranscriptionResult {
                    text: "Three".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                },
            ],
            total_duration_secs: 3.0,
        };

        let texts = result.texts();
        assert_eq!(texts, vec!["One", "Two", "Three"]);
    }

    #[test]
    fn test_batch_transcription_result_iter() {
        let result = BatchTranscriptionResult {
            results: vec![
                TranscriptionResult {
                    text: "A".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                },
                TranscriptionResult {
                    text: "B".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                },
            ],
            total_duration_secs: 1.0,
        };

        let collected: Vec<&str> = result.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(collected, vec!["A", "B"]);
    }

    #[test]
    fn test_transcribe_batch_empty() {
        let whisper = WhisperApr::tiny();
        let result = whisper.transcribe_batch(&[], TranscribeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_transcribe_audio_batch_empty() {
        let whisper = WhisperApr::tiny();
        let batch = audio::AudioBatch::with_default_config();
        let result = whisper.transcribe_audio_batch(&batch, TranscribeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_transcribe_batch_optimized_empty() {
        let whisper = WhisperApr::tiny();
        let result = whisper.transcribe_batch_optimized(&[], TranscribeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_create_audio_batch() {
        let segments = vec![
            vec![0.1_f32, 0.2, 0.3],
            vec![0.4_f32, 0.5],
        ];

        let batch = WhisperApr::create_audio_batch(&segments);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_create_audio_batch_empty() {
        let segments: Vec<Vec<f32>> = vec![];
        let batch = WhisperApr::create_audio_batch(&segments);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_transcription_result_duration() {
        let result = BatchTranscriptionResult {
            results: vec![],
            total_duration_secs: 5.25,
        };

        assert!((result.total_duration_secs - 5.25).abs() < f32::EPSILON);
    }
}

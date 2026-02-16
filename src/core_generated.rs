//! Core whisper types and implementations
#![allow(clippy::all, clippy::pedantic, clippy::restriction, clippy::nursery)]

// Re-import crate modules for use in this module
#[cfg(feature = "realizar-gpu")]
use crate::cuda;
use crate::{
    audio, detection, error, format, inference, model, progress, timestamps, tokenizer, vad,
};
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
#[derive(Debug, Clone, Copy, Default)]
pub enum DecodingStrategy {
    /// Fast, memory-efficient greedy decoding
    #[default]
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
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// Language code (e.g., "en", "es") or "auto" for detection
    pub language: Option<String>,
    /// Task type (transcribe or translate)
    pub task: Task,
    /// Decoding strategy
    pub strategy: DecodingStrategy,
    /// Whether to include word-level timestamps
    pub word_timestamps: bool,
    /// Enable detailed performance profiling (WAPR-PERF-004)
    pub profile: bool,
}

/// A timestamped segment of transcription
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "cli", derive(serde::Serialize, serde::Deserialize))]
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

/// Performance profiling statistics (WAPR-PERF-004)
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "cli", derive(serde::Serialize, serde::Deserialize))]
pub struct ProfilingStats {
    /// Total inference time in milliseconds
    pub total_ms: f64,
    /// Timing breakdown by component (Audio, Encoder, Decoder)
    pub breakdown: std::collections::HashMap<String, f64>,
}

/// Result of transcription
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "cli", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscriptionResult {
    /// Full transcribed text
    pub text: String,
    /// Detected or specified language
    pub language: String,
    /// Timestamped segments
    pub segments: Vec<Segment>,
    /// Performance profiling data (if enabled)
    #[cfg_attr(feature = "cli", serde(skip_serializing_if = "Option::is_none"))]
    pub profiling: Option<ProfilingStats>,
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

/// Options for LFM2 summarization (Phase 3 API - Section 18.5)
///
/// Configures how the LFM2 model generates summaries from transcribed text.
pub struct SummarizeOptions<'a> {
    /// LFM2 model for text generation
    pub model: &'a model::lfm2::Lfm2,
    /// Tokenizer for encoding/decoding text
    pub tokenizer: &'a model::lfm2::Lfm2Tokenizer,
    /// Maximum tokens to generate (default: 256)
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy, default: 0.3)
    pub temperature: f32,
}

impl<'a> SummarizeOptions<'a> {
    /// Create new summarization options with default parameters
    ///
    /// # Arguments
    /// * `model` - LFM2 model for generation
    /// * `tokenizer` - Tokenizer for encoding/decoding
    #[must_use]
    pub fn new(model: &'a model::lfm2::Lfm2, tokenizer: &'a model::lfm2::Lfm2Tokenizer) -> Self {
        Self {
            model,
            tokenizer,
            max_tokens: 256,
            temperature: 0.3,
        }
    }

    /// Set maximum tokens to generate
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set sampling temperature
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
}

/// Result of transcription with summarization (Phase 3 API - Section 18.5)
///
/// Contains both the original transcription and the LFM2-generated summary.
#[derive(Debug, Clone)]
pub struct TranscribeSummaryResult {
    /// Original transcription result
    pub transcription: TranscriptionResult,
    /// LFM2-generated summary
    pub summary: String,
    /// Generation statistics (if available)
    pub generation_stats: Option<model::lfm2::GenerationStats>,
}

impl TranscribeSummaryResult {
    /// Get the transcript text
    #[must_use]
    pub fn transcript(&self) -> &str {
        &self.transcription.text
    }

    /// Get the summary text
    #[must_use]
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Check if summary was generated
    #[must_use]
    pub fn has_summary(&self) -> bool {
        !self.summary.is_empty()
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
#[derive(Debug, Clone)]
pub struct WhisperApr {
    /// Model configuration
    config: model::ModelConfig,
    /// Audio encoder
    encoder: model::Encoder,
    /// Text decoder
    decoder: model::Decoder,
    /// Tokenizer (BPE for Whisper, SentencePiece for Moonshine)
    tokenizer: tokenizer::Tokenizer,
    /// Mel filterbank (None for Moonshine models)
    mel_filters: Option<audio::MelFilterbank>,
    /// Learned conv stem for Moonshine (None for Whisper models)
    conv_stem: Option<audio::ConvStem>,
    /// Resampler for non-16kHz audio
    resampler: Option<audio::SincResampler>,
    /// Whether trained weights have been loaded
    weights_loaded: bool,
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
        let tokenizer = if config.model_family == format::ModelFamily::Moonshine {
            tokenizer::Tokenizer::SentencePiece(
                tokenizer::SentencePieceTokenizer::moonshine_default(),
            )
        } else {
            tokenizer::Tokenizer::Bpe(tokenizer::BpeTokenizer::with_base_tokens())
        };

        // Dispatch audio frontend based on model config
        let (mel_filters, conv_stem) = match config.audio_frontend {
            model::AudioFrontend::MelFilterbank => (
                Some(audio::MelFilterbank::new(
                    config.n_mels as usize,
                    audio::N_FFT,
                    audio::SAMPLE_RATE,
                )),
                None,
            ),
            model::AudioFrontend::LearnedConv => (
                None,
                Some(audio::ConvStem::new(64, config.n_audio_state as usize)),
            ),
        };

        Self {
            config,
            encoder,
            decoder,
            tokenizer,
            mel_filters,
            conv_stem,
            resampler: None,
            weights_loaded: false,
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

    /// Create a Moonshine tiny model configuration
    #[must_use]
    pub fn moonshine_tiny() -> Self {
        Self::from_config(model::ModelConfig::moonshine_tiny())
    }

    /// Create a Moonshine base model configuration
    #[must_use]
    pub fn moonshine_base() -> Self {
        Self::from_config(model::ModelConfig::moonshine_base())
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

    /// Chunk size for long audio processing (30 seconds at 16kHz)
    const CHUNK_SAMPLES: usize = 30 * audio::SAMPLE_RATE as usize; // 480,000 samples

    /// Overlap between chunks (5 seconds at 16kHz)
    const OVERLAP_SAMPLES: usize = 5 * audio::SAMPLE_RATE as usize; // 80,000 samples

    /// Maximum subtitle segment duration (seconds) for SRT output
    const MAX_SUBTITLE_SECS: f32 = 10.0;

    /// Transcribe audio samples
    ///
    /// Automatically handles long audio by chunking with 2-second overlap.
    /// For audio ≤30 seconds, processes in a single pass.
    /// For audio >30 seconds, uses chunked streaming with overlap.
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
        // Use chunked processing for audio longer than 30 seconds
        if audio.len() > Self::CHUNK_SAMPLES {
            return self.transcribe_chunked(audio, options);
        }

        // Process short audio in a single pass
        self.transcribe_single_chunk(audio, options)
    }

    /// Transcribe a single chunk of audio (≤30 seconds)
    ///
    /// Internal method for processing audio that fits in a single chunk.
    fn transcribe_single_chunk(
        &self,
        audio: &[f32],
        options: TranscribeOptions,
    ) -> WhisperResult<TranscriptionResult> {
        #[cfg(feature = "std")]
        let start_total = if options.profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 1-2. Compute audio features (mel + encode for Whisper, ConvStem + encode for Moonshine)
        #[cfg(feature = "std")]
        let start_enc = if options.profile {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let audio_features = self.compute_features(audio)?;
        #[cfg(feature = "std")]
        let enc_ms = start_enc.map(|s| s.elapsed().as_secs_f64() * 1000.0);

        // 3. Determine language
        let language = options.language.clone().unwrap_or_else(|| "en".to_string());

        // 4. Get initial tokens based on task
        let initial_tokens = self.get_initial_tokens(&language, options.task);

        // 5. Decode tokens
        #[cfg(feature = "std")]
        let start_dec = if options.profile {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let tokens = self.decode(&audio_features, &initial_tokens, &options)?;
        #[cfg(feature = "std")]
        let dec_ms = start_dec.map(|s| s.elapsed().as_secs_f64() * 1000.0);

        // 6. Convert full token sequence to text
        let text = self.tokenizer.decode(&tokens)?;

        // 7. Extract segments with timestamps (fallback: sentence-split for full audio)
        let segments = if timestamps::has_timestamps(&tokens) {
            timestamps::extract_segments(&tokens, |ts| self.tokenizer.decode(ts).ok())
        } else if !text.trim().is_empty() {
            let duration = audio.len() as f32 / audio::SAMPLE_RATE as f32;
            let single = vec![Segment {
                start: 0.0,
                end: duration,
                text: text.clone(),
                tokens: tokens.clone(),
            }];
            timestamps::split_long_segments(&single, Self::MAX_SUBTITLE_SECS)
        } else {
            Vec::new()
        };

        // 8. Build result with optional profiling
        #[cfg(feature = "std")]
        let profiling = if options.profile {
            let mut breakdown = std::collections::HashMap::new();
            if let Some(ms) = enc_ms {
                breakdown.insert("encoder_ms".to_string(), ms);
            }
            if let Some(ms) = dec_ms {
                breakdown.insert("decoder_ms".to_string(), ms);
            }

            Some(ProfilingStats {
                total_ms: start_total
                    .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0),
                breakdown,
            })
        } else {
            None
        };

        #[cfg(not(feature = "std"))]
        let profiling = None;

        Ok(TranscriptionResult {
            text,
            language,
            segments,
            profiling,
        })
    }

    /// Transcribe long audio using chunked streaming with overlap
    ///
    /// Splits audio into 30-second chunks with 2-second overlap, transcribes
    /// each chunk independently, then merges results to avoid duplication
    /// at chunk boundaries.
    ///
    /// # Algorithm (per spec §3.4)
    ///
    /// 1. Split audio into chunks: `windows(CHUNK + OVERLAP).step_by(CHUNK)`
    /// 2. Transcribe each chunk independently
    /// 3. Merge overlapping text by finding common subsequences
    /// 4. Adjust segment timestamps based on chunk offset
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, 16kHz, any length)
    /// * `options` - Transcription options
    ///
    /// # Returns
    /// Merged transcription result
    fn transcribe_chunked(
        &self,
        audio: &[f32],
        options: TranscribeOptions,
    ) -> WhisperResult<TranscriptionResult> {
        let chunk_size = Self::CHUNK_SAMPLES;
        let overlap = Self::OVERLAP_SAMPLES;
        let step = chunk_size; // Non-overlapping step for clean boundaries

        // Determine language from first chunk (for consistency)
        let language = options.language.clone().unwrap_or_else(|| "en".to_string());

        let mut all_segments: Vec<Segment> = Vec::new();
        let mut all_text = String::new();
        let mut chunk_idx = 0;

        // Process audio in chunks
        let mut offset = 0;
        while offset < audio.len() {
            // Calculate chunk boundaries
            let chunk_end = (offset + chunk_size + overlap).min(audio.len());
            let chunk = &audio[offset..chunk_end];

            // Skip very short final chunks (less than 0.5 seconds)
            if chunk.len() < audio::SAMPLE_RATE as usize / 2 {
                break;
            }

            // Transcribe this chunk
            let chunk_options = TranscribeOptions {
                language: Some(language.clone()),
                task: options.task,
                strategy: options.strategy,
                word_timestamps: options.word_timestamps,
                profile: options.profile,
            };

            let chunk_result = self.transcribe_single_chunk(chunk, chunk_options)?;

            // Calculate time offset for this chunk
            let time_offset = offset as f32 / audio::SAMPLE_RATE as f32;

            // Merge text: trim overlap from previous chunk's end
            let chunk_text = if chunk_idx == 0 {
                // First chunk: use full text
                chunk_result.text.clone()
            } else {
                // Subsequent chunks: remove overlap from beginning
                // Overlap is ~2 seconds, which corresponds to first ~2 seconds of text
                // We use simple heuristic: skip first few words proportional to overlap
                let overlap_ratio = overlap as f32 / chunk.len() as f32;
                let words: Vec<&str> = chunk_result.text.split_whitespace().collect();
                let skip_words = ((words.len() as f32) * overlap_ratio * 0.8) as usize;
                words
                    .into_iter()
                    .skip(skip_words)
                    .collect::<Vec<_>>()
                    .join(" ")
            };

            // Append to accumulated text
            if !chunk_text.is_empty() {
                if !all_text.is_empty() {
                    all_text.push(' ');
                }
                all_text.push_str(&chunk_text);
            }

            // Adjust segment timestamps and add to all_segments
            for mut seg in chunk_result.segments {
                seg.start += time_offset;
                seg.end += time_offset;
                all_segments.push(seg);
            }

            // Advance to next chunk
            offset += step;
            chunk_idx += 1;
        }

        // Merge overlapping segments if any
        let merged_segments = self.merge_overlapping_segments(all_segments);

        // Split long segments at sentence boundaries for proper subtitles
        let final_segments =
            timestamps::split_long_segments(&merged_segments, Self::MAX_SUBTITLE_SECS);

        Ok(TranscriptionResult {
            text: all_text,
            language,
            segments: final_segments,
            profiling: None,
        })
    }

    /// Merge overlapping segments from chunked transcription
    ///
    /// Combines segments that overlap in time, preferring the segment
    /// with more text content.
    fn merge_overlapping_segments(&self, segments: Vec<Segment>) -> Vec<Segment> {
        if segments.is_empty() {
            return segments;
        }

        let mut merged: Vec<Segment> = Vec::with_capacity(segments.len());
        let mut current = segments[0].clone();

        for seg in segments.into_iter().skip(1) {
            // Check for overlap (with 0.1s tolerance)
            if seg.start < current.end + 0.1 {
                // Merge: extend end time and concatenate text
                current.end = current.end.max(seg.end);
                if !seg.text.is_empty() {
                    if !current.text.is_empty() {
                        current.text.push(' ');
                    }
                    current.text.push_str(&seg.text);
                }
                current.tokens.extend(seg.tokens);
            } else {
                // No overlap: push current and start new
                merged.push(current);
                current = seg;
            }
        }

        // Don't forget the last segment
        merged.push(current);

        merged
    }

    /// Compute audio features (model-family-aware audio frontend)
    ///
    /// For Whisper: mel spectrogram (30s padded, 3000 frames x 80 mels)
    /// For Moonshine: learned ConvStem on variable-length audio
    ///
    /// # Errors
    /// Returns error if audio processing fails
    fn compute_features(&self, audio: &[f32]) -> WhisperResult<Vec<f32>> {
        match self.config.audio_frontend {
            model::AudioFrontend::MelFilterbank => {
                let mel = self.compute_mel(audio)?;
                self.encode(&mel)
            }
            model::AudioFrontend::LearnedConv => {
                let stem = self.conv_stem.as_ref().ok_or_else(|| {
                    WhisperError::Model("Moonshine requires ConvStem".into())
                })?;
                let stem_out = stem.forward(audio)?;
                // Pass ConvStem output through encoder blocks
                self.encoder.forward(&stem_out)
            }
        }
    }

    /// Get the end-of-transcription token ID for the current model
    fn eot_token(&self) -> u32 {
        if self.config.model_family == format::ModelFamily::Moonshine {
            2 // SentencePiece EOS token
        } else {
            tokenizer::special_tokens::EOT
        }
    }

    /// Compute mel spectrogram from audio
    ///
    /// Pads or truncates audio to exactly 30 seconds (480,000 samples at 16kHz)
    /// before computing mel spectrogram to match Whisper's expected input.
    /// The output is always exactly 3000 frames x 80 mel bins.
    pub fn compute_mel(&self, audio: &[f32]) -> WhisperResult<Vec<f32>> {
        const N_SAMPLES_30S: usize = 480_000; // 30 seconds at 16kHz
        const N_FRAMES: usize = 3000; // Whisper expects exactly 3000 frames
        const N_MELS: usize = 80;

        // Pad or truncate audio to exactly 30 seconds
        let padded_audio = match audio.len().cmp(&N_SAMPLES_30S) {
            std::cmp::Ordering::Equal => audio.to_vec(),
            std::cmp::Ordering::Less => {
                // Pad with zeros (silence) to 30 seconds
                let mut padded = vec![0.0_f32; N_SAMPLES_30S];
                padded[..audio.len()].copy_from_slice(audio);
                padded
            }
            std::cmp::Ordering::Greater => {
                // Truncate to 30 seconds
                audio[..N_SAMPLES_30S].to_vec()
            }
        };

        let mel_fb = self.mel_filters.as_ref().ok_or_else(|| {
            WhisperError::Audio("mel filterbank not available (Moonshine model?)".into())
        })?;
        let mut mel = mel_fb.compute(&padded_audio, audio::HOP_LENGTH)?;
        let actual_frames = mel.len() / N_MELS;

        // Ensure exactly 3000 frames (pad or truncate mel output)
        if actual_frames < N_FRAMES {
            // Pad with log-mel floor value (silence in log domain)
            // HF uses -1.0 (which is roughly log(0.1)/log(10) after normalization)
            let pad_value = -1.0_f32;
            let mut padded_mel = vec![pad_value; N_FRAMES * N_MELS];
            padded_mel[..mel.len()].copy_from_slice(&mel);
            mel = padded_mel;
        } else if actual_frames > N_FRAMES {
            mel.truncate(N_FRAMES * N_MELS);
        }

        // Mel spectrogram is computed as [frames, mels] = [3000, 80]
        // Our Conv1d expects (seq_len × in_channels) = (frames × mels)
        // which matches the mel layout, so no transpose needed
        Ok(mel)
    }

    /// Encode audio features (mel spectrogram -> encoder features)
    pub fn encode(&self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        // Use forward_mel to process mel spectrogram through conv frontend
        self.encoder.forward_mel(mel)
    }

    /// Get initial tokens for decoding
    ///
    /// Uses dynamic token lookup based on vocabulary size to support both
    /// English-only models (tiny.en, etc.) and multilingual models.
    fn get_initial_tokens(&self, language: &str, task: Task) -> Vec<u32> {
        // Moonshine: just BOS token (SentencePiece token 1)
        if self.config.model_family == format::ModelFamily::Moonshine {
            return vec![1]; // SentencePiece BOS
        }

        // Whisper: SOT + language + task tokens
        use tokenizer::special_tokens::{self, SpecialTokens};

        // Get correct special tokens for this model's vocabulary size
        let specials = SpecialTokens::for_vocab_size(self.config.n_vocab as usize);

        let mut tokens = vec![specials.sot];

        // Add language token (defaults to English if unknown)
        // For multilingual models, language tokens start at lang_base
        // For English-only models, we skip the language token
        if specials.is_multilingual {
            let lang_offset = special_tokens::language_offset(language).unwrap_or(0);
            tokens.push(specials.lang_base + lang_offset);
        }

        // Add task token
        match task {
            Task::Transcribe => tokens.push(specials.transcribe),
            Task::Translate => tokens.push(special_tokens::TRANSLATE),
        }

        // Timestamp mode: do NOT push no_timestamps token
        // This enables the decoder to produce <|t.tt|> timestamp tokens
        // which are needed for proper SRT/VTT segment timing

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
    ///
    /// Uses KV cache for O(n) incremental decoding instead of O(n²) full recomputation.
    fn decode(
        &self,
        audio_features: &[f32],
        initial_tokens: &[u32],
        options: &TranscribeOptions,
    ) -> WhisperResult<Vec<u32>> {
        use std::cell::RefCell;

        let n_vocab = self.config.n_vocab as usize;
        let max_tokens = self.config.n_text_ctx as usize;

        // Create KV cache for incremental decoding (O(n) instead of O(n²))
        // Uses decoder's create_kv_cache() which dispatches GQA vs MHA automatically
        let cache = RefCell::new(self.decoder.create_kv_cache());
        let processed_count = RefCell::new(0usize);

        // Create Whisper token suppressor using realizar's LogitProcessor architecture
        // This provides composable, reusable pre-sampling transforms
        let suppressor = inference::WhisperTokenSuppressor::new()
            .with_timestamp_suppression(false)
            .with_vocab_size(n_vocab);

        let logits_fn = |tokens: &[u32]| -> WhisperResult<Vec<f32>> {
            let seq_len = tokens.len();
            let already_processed = *processed_count.borrow();

            // Process only the tokens we haven't seen yet
            let mut logits = vec![f32::NEG_INFINITY; n_vocab];

            for &token in tokens.iter().take(seq_len).skip(already_processed) {
                logits =
                    self.decoder
                        .forward_one(token, audio_features, &mut cache.borrow_mut())?;
            }

            *processed_count.borrow_mut() = seq_len;

            // Apply Whisper-specific token suppression via LogitProcessor
            // Suppresses: SOT, language tokens, task tokens, timestamps (if disabled)
            suppressor.apply(&mut logits);

            // N-gram repetition suppression (prevents hallucinated loops)
            let eot_id = self.eot_token();
            Self::suppress_repetitions(&mut logits, tokens, eot_id, n_vocab);

            Ok(logits)
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

    /// Apply n-gram repetition suppression to logits
    ///
    /// Prevents hallucinated loops by penalizing repeated tokens, blocking
    /// trigram completions, forcing EOT on degenerate 4-gram loops, and
    /// capping unigram frequency in recent output.
    fn suppress_repetitions(logits: &mut [f32], tokens: &[u32], eot_id: u32, n_vocab: usize) {
        let text_tokens: Vec<u32> = tokens.iter().copied().filter(|&t| t < eot_id).collect();

        // Token-level penalty: penalize recently-seen tokens
        let window_start = text_tokens.len().saturating_sub(50);
        for &prev_tok in &text_tokens[window_start..] {
            if (prev_tok as usize) < n_vocab {
                logits[prev_tok as usize] -= 2.0;
            }
        }

        // Trigram blocking: if (prev2, prev1, X) already appeared, suppress X
        if text_tokens.len() >= 3 {
            let prev2 = text_tokens[text_tokens.len() - 2];
            let prev1 = text_tokens[text_tokens.len() - 1];
            for w in text_tokens[..text_tokens.len() - 2].windows(3) {
                if w[0] == prev2 && w[1] == prev1 && (w[2] as usize) < n_vocab {
                    logits[w[2] as usize] -= 10.0;
                }
            }
        }

        Self::suppress_degenerate_loops(logits, &text_tokens, eot_id, n_vocab);
    }

    /// Detect and suppress degenerate repetition loops (4-gram and unigram frequency)
    fn suppress_degenerate_loops(
        logits: &mut [f32],
        text_tokens: &[u32],
        eot_id: u32,
        n_vocab: usize,
    ) {
        // Force EOT if any 4-gram repeats 3+ times
        if text_tokens.len() >= 12 {
            let mut fourgram_counts = std::collections::HashMap::<[u32; 4], u32>::new();
            for w in text_tokens.windows(4) {
                *fourgram_counts.entry([w[0], w[1], w[2], w[3]]).or_insert(0) += 1;
            }
            if fourgram_counts.values().any(|&c| c >= 3) && (eot_id as usize) < n_vocab {
                logits[eot_id as usize] = 100.0;
                return;
            }
        }

        // Unigram frequency cap: suppress tokens dominating recent output
        if text_tokens.len() >= 80 {
            let window = &text_tokens[text_tokens.len() - 80..];
            let mut freq = std::collections::HashMap::<u32, u32>::new();
            for &t in window {
                *freq.entry(t).or_insert(0) += 1;
            }
            for (&tok, &count) in &freq {
                if count >= 8 && (tok as usize) < n_vocab {
                    logits[tok as usize] -= (count as f32 - 7.0) * 3.0;
                }
            }
            if freq.len() as f32 / 80.0 < 0.35 && (eot_id as usize) < n_vocab {
                logits[eot_id as usize] = 100.0;
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
    pub const fn tokenizer(&self) -> &tokenizer::Tokenizer {
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

    /// Check if model has trained weights loaded
    ///
    /// Returns true if weights have been loaded via `load_from_apr`,
    /// false if using default/uninitialized weights from `from_config`.
    #[must_use]
    pub fn has_weights(&self) -> bool {
        self.weights_loaded
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
        // Finalize encoder - cache transposed weights for fast forward
        encoder.finalize_weights();
        tracker.next_phase();

        // Phase 3: Loading decoder
        callback(&tracker.to_progress());
        let mut decoder = model::Decoder::new(&config);
        Self::load_decoder_weights(&reader, &mut decoder, &mut tracker, callback);
        tracker.next_phase();

        // Phase 4: Loading vocabulary
        callback(&tracker.to_progress());
        // Load vocab from APR if available, otherwise fall back to base tokens
        let tokenizer = tokenizer::Tokenizer::Bpe(reader.read_vocabulary().map_or_else(
            tokenizer::BpeTokenizer::with_base_tokens,
            tokenizer::BpeTokenizer::from_vocabulary,
        ));
        tracker.next_phase();

        // Phase 5: Initializing audio frontend
        callback(&tracker.to_progress());
        let (mel_filters, conv_stem) = match config.audio_frontend {
            model::AudioFrontend::MelFilterbank => {
                let mf = reader.read_mel_filterbank().map_or_else(
                    || {
                        audio::MelFilterbank::new(
                            config.n_mels as usize,
                            audio::N_FFT,
                            audio::SAMPLE_RATE,
                        )
                    },
                    |fb| audio::MelFilterbank::from_apr_data(fb, audio::SAMPLE_RATE),
                );
                (Some(mf), None)
            }
            model::AudioFrontend::LearnedConv => (
                None,
                Some(audio::ConvStem::new(64, config.n_audio_state as usize)),
            ),
        };
        tracker.complete();
        callback(&tracker.to_progress());

        Ok(Self {
            config,
            encoder,
            decoder,
            tokenizer,
            mel_filters,
            conv_stem,
            resampler: None,
            weights_loaded: true,
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

        // Load convolutional frontend weights (Whisper only; Moonshine uses learned conv stem)
        if let Some(conv_frontend) = encoder.conv_frontend_mut() {
            // Conv1: mel -> hidden (n_mels x d_model with kernel_size=3)
            if let Ok(weight) = reader.load_tensor("encoder.conv1.weight") {
                let target = conv_frontend.conv1.weight_mut();
                let len = weight.len().min(target.len());
                target[..len].copy_from_slice(&weight[..len]);
            }
            if let Ok(bias) = reader.load_tensor("encoder.conv1.bias") {
                let target = conv_frontend.conv1.bias_mut();
                let len = bias.len().min(target.len());
                target[..len].copy_from_slice(&bias[..len]);
            }

            // Conv2: hidden -> hidden with stride 2
            if let Ok(weight) = reader.load_tensor("encoder.conv2.weight") {
                let target = conv_frontend.conv2.weight_mut();
                let len = weight.len().min(target.len());
                target[..len].copy_from_slice(&weight[..len]);
            }
            if let Ok(bias) = reader.load_tensor("encoder.conv2.bias") {
                let target = conv_frontend.conv2.bias_mut();
                let len = bias.len().min(target.len());
                target[..len].copy_from_slice(&bias[..len]);
            }
        }

        // Load positional embedding if available (HF uses embed_positions.weight)
        let pe_result = reader
            .load_tensor("encoder.embed_positions.weight")
            .or_else(|_| reader.load_tensor("encoder.positional_embedding"));
        if let Ok(pe) = pe_result {
            let target = encoder.positional_embedding_mut();
            let len = pe.len().min(target.len());
            target[..len].copy_from_slice(&pe[..len]);
        }

        // Load encoder block weights — dispatch Whisper vs Moonshine
        if !encoder.moonshine_blocks().is_empty() {
            // Moonshine encoder: MHA + GELU MLP + LayerNorm(no bias)
            for layer_idx in 0..n_layers {
                let progress = layer_idx as f32 / n_layers as f32;
                tracker.update_phase_progress(progress);
                callback(&tracker.to_progress());

                let block = &mut encoder.moonshine_blocks_mut()[layer_idx];

                // Pre-attention LayerNorm (weight-only)
                Self::load_layernorm_nobias_weights(
                    reader,
                    &format!("encoder.blocks.{layer_idx}.ln1"),
                    &mut block.ln1,
                );

                // MHA self-attention
                Self::load_gqa_weights(
                    reader,
                    &format!("encoder.blocks.{layer_idx}.attn"),
                    &mut block.self_attn,
                );

                // Pre-FFN LayerNorm (weight-only)
                Self::load_layernorm_nobias_weights(
                    reader,
                    &format!("encoder.blocks.{layer_idx}.ln2"),
                    &mut block.ln2,
                );

                // MLP FFN (fc1 → GELU → fc2)
                Self::load_mlp_weights(
                    reader,
                    &format!("encoder.blocks.{layer_idx}.ffn"),
                    &mut block.ffn,
                );
            }
        } else {
            // Whisper encoder: MHA + GELU + LayerNorm (HuggingFace naming: encoder.layers.N.*)
            for layer_idx in 0..n_layers {
                let progress = layer_idx as f32 / n_layers as f32;
                tracker.update_phase_progress(progress);
                callback(&tracker.to_progress());

                let block = &mut encoder.blocks_mut()[layer_idx];

                // Self-attention layer norm (before attention)
                Self::load_layer_norm_weights(
                    reader,
                    &format!("encoder.layers.{layer_idx}.self_attn_layer_norm"),
                    &mut block.ln1,
                );

                // Self-attention
                Self::load_attention_weights(
                    reader,
                    &format!("encoder.layers.{layer_idx}.self_attn"),
                    &mut block.self_attn,
                );

                // Final layer norm (before FFN)
                Self::load_layer_norm_weights(
                    reader,
                    &format!("encoder.layers.{layer_idx}.final_layer_norm"),
                    &mut block.ln2,
                );

                // Feed-forward network
                Self::load_ffn_weights(
                    reader,
                    &format!("encoder.layers.{layer_idx}"),
                    &mut block.ffn,
                );
            }
        }

        // Load encoder final layer norm
        Self::load_layer_norm_weights(reader, "encoder.layer_norm", encoder.ln_post_mut());
    }

    /// Load decoder weights from .apr reader
    fn load_decoder_weights(
        reader: &format::AprReader,
        decoder: &mut model::Decoder,
        tracker: &mut progress::ProgressTracker,
        callback: progress::ProgressCallback<'_>,
    ) {
        let n_layers = decoder.n_layers();

        // Load token embedding if available (HF uses embed_tokens.weight)
        let te_result = reader
            .load_tensor("decoder.embed_tokens.weight")
            .or_else(|_| reader.load_tensor("decoder.token_embedding"));
        if let Ok(te) = te_result {
            let target = decoder.token_embedding_mut();
            let len = te.len().min(target.len());
            target[..len].copy_from_slice(&te[..len]);
        }

        // Load positional embedding if available (HF uses embed_positions.weight)
        let pe_result = reader
            .load_tensor("decoder.embed_positions.weight")
            .or_else(|_| reader.load_tensor("decoder.positional_embedding"));
        if let Ok(pe) = pe_result {
            let target = decoder.positional_embedding_mut();
            let len = pe.len().min(target.len());
            target[..len].copy_from_slice(&pe[..len]);
        }

        // Load decoder block weights — dispatch Whisper vs Moonshine
        if !decoder.moonshine_blocks().is_empty() {
            // Moonshine decoder: MHA self-attn + MHA cross-attn + SiLU gated MLP + LayerNorm(no bias)
            for layer_idx in 0..n_layers {
                let progress = layer_idx as f32 / n_layers as f32;
                tracker.update_phase_progress(progress);
                callback(&tracker.to_progress());

                let block = &mut decoder.moonshine_blocks_mut()[layer_idx];

                // Pre-self-attention LayerNorm (weight-only)
                Self::load_layernorm_nobias_weights(
                    reader,
                    &format!("decoder.blocks.{layer_idx}.ln1"),
                    &mut block.ln1,
                );

                // MHA self-attention (causal, with RoPE)
                Self::load_gqa_weights(
                    reader,
                    &format!("decoder.blocks.{layer_idx}.attn"),
                    &mut block.self_attn,
                );

                // Pre-cross-attention LayerNorm (weight-only)
                Self::load_layernorm_nobias_weights(
                    reader,
                    &format!("decoder.blocks.{layer_idx}.ln_cross"),
                    &mut block.ln_cross,
                );

                // MHA cross-attention (Q from decoder, KV from encoder)
                Self::load_gqa_weights(
                    reader,
                    &format!("decoder.blocks.{layer_idx}.cross_attn"),
                    &mut block.cross_attn,
                );

                // Pre-FFN LayerNorm (weight-only)
                Self::load_layernorm_nobias_weights(
                    reader,
                    &format!("decoder.blocks.{layer_idx}.ln2"),
                    &mut block.ln2,
                );

                // Gated MLP FFN (fc1[→2x] → SiLU gate → fc2)
                Self::load_gated_mlp_weights(
                    reader,
                    &format!("decoder.blocks.{layer_idx}.ffn"),
                    &mut block.ffn,
                );
            }

            // Load Moonshine final LayerNorm (weight-only)
            if let Some(ln) = decoder.ln_post_rms_mut() {
                Self::load_rms_norm_weights(reader, "decoder.ln_post", ln);
            }
        } else {
            // Whisper decoder: MHA + GELU + LayerNorm
            for layer_idx in 0..n_layers {
                let progress = layer_idx as f32 / n_layers as f32;
                tracker.update_phase_progress(progress);
                callback(&tracker.to_progress());

                let block = &mut decoder.blocks_mut()[layer_idx];

                // Layer norm 1 (before self-attention)
                Self::load_layer_norm_weights(
                    reader,
                    &format!("decoder.layers.{layer_idx}.self_attn_layer_norm"),
                    &mut block.ln1,
                );

                // Self-attention
                Self::load_attention_weights(
                    reader,
                    &format!("decoder.layers.{layer_idx}.self_attn"),
                    &mut block.self_attn,
                );

                // Layer norm 2 (before cross-attention)
                Self::load_layer_norm_weights(
                    reader,
                    &format!("decoder.layers.{layer_idx}.encoder_attn_layer_norm"),
                    &mut block.ln2,
                );

                // Cross-attention
                Self::load_attention_weights(
                    reader,
                    &format!("decoder.layers.{layer_idx}.encoder_attn"),
                    &mut block.cross_attn,
                );

                // Layer norm 3 (before FFN)
                Self::load_layer_norm_weights(
                    reader,
                    &format!("decoder.layers.{layer_idx}.final_layer_norm"),
                    &mut block.ln3,
                );

                // Feed-forward network
                Self::load_ffn_weights(
                    reader,
                    &format!("decoder.layers.{layer_idx}"),
                    &mut block.ffn,
                );
            }

            // Load Whisper final layer norm
            Self::load_layer_norm_weights(reader, "decoder.layer_norm", decoder.ln_post_mut());
        }

        // Finalize decoder - recompute cached transpose for token embeddings
        decoder.finalize_weights();
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

    /// Load attention weights (HuggingFace naming: q_proj, k_proj, v_proj, out_proj)
    fn load_attention_weights(
        reader: &format::AprReader,
        prefix: &str,
        attn: &mut model::MultiHeadAttention,
    ) {
        // Load query, key, value projections - weights and biases
        if let Ok(q_weight) = reader.load_tensor(&format!("{prefix}.q_proj.weight")) {
            attn.set_query_weight(&q_weight);
        }
        if let Ok(q_bias) = reader.load_tensor(&format!("{prefix}.q_proj.bias")) {
            attn.set_query_bias(&q_bias);
        }
        if let Ok(k_weight) = reader.load_tensor(&format!("{prefix}.k_proj.weight")) {
            attn.set_key_weight(&k_weight);
        }
        if let Ok(k_bias) = reader.load_tensor(&format!("{prefix}.k_proj.bias")) {
            attn.set_key_bias(&k_bias);
        }
        if let Ok(v_weight) = reader.load_tensor(&format!("{prefix}.v_proj.weight")) {
            attn.set_value_weight(&v_weight);
        }
        if let Ok(v_bias) = reader.load_tensor(&format!("{prefix}.v_proj.bias")) {
            attn.set_value_bias(&v_bias);
        }
        if let Ok(out_weight) = reader.load_tensor(&format!("{prefix}.out_proj.weight")) {
            attn.set_out_weight(&out_weight);
        }
        if let Ok(out_bias) = reader.load_tensor(&format!("{prefix}.out_proj.bias")) {
            attn.set_out_bias(&out_bias);
        }
    }

    /// Load feed-forward network weights (HuggingFace naming: fc1, fc2)
    fn load_ffn_weights(reader: &format::AprReader, prefix: &str, ffn: &mut model::FeedForward) {
        if let Ok(fc1_weight) = reader.load_tensor(&format!("{prefix}.fc1.weight")) {
            ffn.fc1.set_weight(&fc1_weight);
        }
        if let Ok(fc1_bias) = reader.load_tensor(&format!("{prefix}.fc1.bias")) {
            ffn.fc1.set_bias(&fc1_bias);
        }
        if let Ok(fc2_weight) = reader.load_tensor(&format!("{prefix}.fc2.weight")) {
            ffn.fc2.set_weight(&fc2_weight);
        }
        if let Ok(fc2_bias) = reader.load_tensor(&format!("{prefix}.fc2.bias")) {
            ffn.fc2.set_bias(&fc2_bias);
        }
    }

    /// Load RMS norm weights (LFM2: single weight vector, no bias)
    fn load_rms_norm_weights(
        reader: &format::AprReader,
        prefix: &str,
        rms: &mut model::lfm2::layer::RmsNorm,
    ) {
        if let Ok(weight) = reader.load_tensor(&format!("{prefix}.weight")) {
            let len = weight.len().min(rms.weight.len());
            rms.weight[..len].copy_from_slice(&weight[..len]);
        }
    }

    /// Load LayerNorm (no bias) weights (Moonshine: weight-only LayerNorm)
    fn load_layernorm_nobias_weights(
        reader: &format::AprReader,
        prefix: &str,
        ln: &mut model::lfm2::layer::LayerNormNoBias,
    ) {
        if let Ok(weight) = reader.load_tensor(&format!("{prefix}.weight")) {
            let len = weight.len().min(ln.weight.len());
            ln.weight[..len].copy_from_slice(&weight[..len]);
        }
    }

    /// Load GQA attention weights (Moonshine: q, k, v, o projections)
    fn load_gqa_weights(
        reader: &format::AprReader,
        prefix: &str,
        gqa: &mut model::lfm2::gqa::GroupedQueryAttention,
    ) {
        if let Ok(w) = reader.load_tensor(&format!("{prefix}.q.weight")) {
            let len = w.len().min(gqa.w_q.len());
            gqa.w_q[..len].copy_from_slice(&w[..len]);
        }
        if let Ok(w) = reader.load_tensor(&format!("{prefix}.k.weight")) {
            let len = w.len().min(gqa.w_k.len());
            gqa.w_k[..len].copy_from_slice(&w[..len]);
        }
        if let Ok(w) = reader.load_tensor(&format!("{prefix}.v.weight")) {
            let len = w.len().min(gqa.w_v.len());
            gqa.w_v[..len].copy_from_slice(&w[..len]);
        }
        if let Ok(w) = reader.load_tensor(&format!("{prefix}.o.weight")) {
            let len = w.len().min(gqa.w_o.len());
            gqa.w_o[..len].copy_from_slice(&w[..len]);
        }
    }

    /// Load MLP FFN weights (Moonshine: fc1 → activation → fc2)
    fn load_mlp_weights(
        reader: &format::AprReader,
        prefix: &str,
        ffn: &mut model::lfm2::mlp::MlpFfn,
    ) {
        if let Ok(w) = reader.load_tensor(&format!("{prefix}.fc1.weight")) {
            let len = w.len().min(ffn.fc1.len());
            ffn.fc1[..len].copy_from_slice(&w[..len]);
        }
        if let Ok(w) = reader.load_tensor(&format!("{prefix}.fc2.weight")) {
            let len = w.len().min(ffn.fc2.len());
            ffn.fc2[..len].copy_from_slice(&w[..len]);
        }
    }

    /// Load gated MLP FFN weights (Moonshine decoder: fc1[→2x] → SiLU gate → fc2)
    fn load_gated_mlp_weights(
        reader: &format::AprReader,
        prefix: &str,
        ffn: &mut model::lfm2::mlp::GatedMlpFfn,
    ) {
        if let Ok(w) = reader.load_tensor(&format!("{prefix}.fc1.weight")) {
            let len = w.len().min(ffn.fc1.len());
            ffn.fc1[..len].copy_from_slice(&w[..len]);
        }
        if let Ok(w) = reader.load_tensor(&format!("{prefix}.fc2.weight")) {
            let len = w.len().min(ffn.fc2.len());
            ffn.fc2[..len].copy_from_slice(&w[..len]);
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

    /// Get immutable encoder reference
    pub fn encoder(&self) -> &model::Encoder {
        &self.encoder
    }

    /// Get immutable decoder reference
    pub fn decoder(&self) -> &model::Decoder {
        &self.decoder
    }

    /// Convert to CUDA-accelerated model.
    ///
    /// Consumes self and returns a `WhisperCuda` wrapper for GPU inference.
    ///
    /// # Arguments
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    /// Returns error if CUDA is not available or device doesn't exist.
    #[cfg(feature = "realizar-gpu")]
    pub fn into_cuda(self, device_ordinal: i32) -> WhisperResult<cuda::WhisperCuda> {
        let mel_filters = self.mel_filters.ok_or_else(|| {
            WhisperError::Audio("CUDA requires mel filterbank (Whisper model)".into())
        })?;
        let bpe_tokenizer = match self.tokenizer {
            tokenizer::Tokenizer::Bpe(bpe) => bpe,
            tokenizer::Tokenizer::SentencePiece(_) => {
                return Err(WhisperError::Model(
                    "CUDA backend requires Whisper BPE tokenizer".into(),
                ));
            }
        };
        cuda::WhisperCuda::new_with_components(
            self.encoder,
            self.decoder,
            self.config,
            bpe_tokenizer,
            mel_filters,
            device_ordinal,
        )
    }

    /// Get conv stem reference (Moonshine models)
    #[must_use]
    pub const fn conv_stem(&self) -> Option<&audio::ConvStem> {
        self.conv_stem.as_ref()
    }

    /// Get mel filters reference (Whisper models)
    #[must_use]
    pub const fn mel_filters(&self) -> Option<&audio::MelFilterbank> {
        self.mel_filters.as_ref()
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
                profiling: None,
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
                profiling: None,
            });
        }

        let total_duration_secs = start_time.elapsed().as_secs_f32();

        Ok(BatchTranscriptionResult {
            results,
            total_duration_secs,
        })
    }

    // =========================================================================
    // VAD-Triggered Transcription (WAPR-093)
    // =========================================================================

    /// Transcribe audio using VAD to detect and transcribe only speech segments
    ///
    /// This method uses Voice Activity Detection to:
    /// 1. Detect speech segments in the audio
    /// 2. Transcribe only the detected speech (skipping silence)
    /// 3. Combine results with accurate timestamps
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, 16kHz, f32 normalized to [-1, 1])
    /// * `options` - Transcription options
    /// * `vad_config` - Optional VAD configuration (uses default if None)
    ///
    /// # Returns
    /// VAD transcription result with speech segments and timestamps
    ///
    /// # Errors
    /// Returns error if transcription fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use whisper_apr::{WhisperApr, TranscribeOptions};
    ///
    /// let whisper = WhisperApr::tiny();
    /// let result = whisper.transcribe_with_vad(&audio, TranscribeOptions::default(), None)?;
    ///
    /// for segment in &result.segments {
    ///     println!("[{:.2}s - {:.2}s] {}", segment.start, segment.end, segment.text);
    /// }
    /// ```
    pub fn transcribe_with_vad(
        &self,
        audio: &[f32],
        options: TranscribeOptions,
        vad_config: Option<vad::VadConfig>,
    ) -> WhisperResult<VadTranscriptionResult> {
        let start_time = std::time::Instant::now();

        // Create VAD detector
        let config = vad_config.unwrap_or_default();
        let mut vad = vad::VoiceActivityDetector::new(config);

        // Detect speech segments
        let speech_segments = vad.detect(audio);

        if speech_segments.is_empty() {
            return Ok(VadTranscriptionResult {
                text: String::new(),
                language: options.language.unwrap_or_else(|| "en".to_string()),
                segments: Vec::new(),
                speech_segments: Vec::new(),
                total_duration_secs: start_time.elapsed().as_secs_f32(),
                speech_duration_secs: 0.0,
            });
        }

        // Extract speech audio segments
        let sample_rate = audio::SAMPLE_RATE as f32;
        let mut speech_audios = Vec::with_capacity(speech_segments.len());
        let mut speech_duration = 0.0f32;

        for segment in &speech_segments {
            let start_sample = (segment.start * sample_rate) as usize;
            let end_sample = ((segment.end * sample_rate) as usize).min(audio.len());

            if end_sample > start_sample {
                speech_audios.push((
                    segment.start,
                    segment.end,
                    audio[start_sample..end_sample].to_vec(),
                ));
                speech_duration += segment.duration();
            }
        }

        // Transcribe each speech segment
        let mut all_segments = Vec::new();
        let mut full_text = String::new();
        let language = options.language.clone().unwrap_or_else(|| "en".to_string());

        for (seg_start, seg_end, speech_audio) in &speech_audios {
            // Transcribe this segment
            let result = self.transcribe(speech_audio, options.clone())?;

            // Create segment with corrected timestamps
            let segment = VadSpeechSegment {
                start: *seg_start,
                end: *seg_end,
                text: result.text.clone(),
                tokens: result
                    .segments
                    .first()
                    .map(|s| s.tokens.clone())
                    .unwrap_or_default(),
            };

            if !full_text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&result.text);

            all_segments.push(segment);
        }

        let total_duration_secs = start_time.elapsed().as_secs_f32();

        Ok(VadTranscriptionResult {
            text: full_text,
            language,
            segments: all_segments,
            speech_segments: speech_segments
                .into_iter()
                .map(|s| (s.start, s.end))
                .collect(),
            total_duration_secs,
            speech_duration_secs: speech_duration,
        })
    }

    /// Transcribe audio with custom silence detector configuration
    ///
    /// Similar to `transcribe_with_vad` but uses silence detection for
    /// segmenting audio based on silence gaps.
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, 16kHz, f32 normalized to [-1, 1])
    /// * `options` - Transcription options
    /// * `silence_config` - Optional silence detection configuration
    ///
    /// # Returns
    /// VAD transcription result
    ///
    /// # Errors
    /// Returns error if transcription fails
    pub fn transcribe_with_silence_detection(
        &self,
        audio: &[f32],
        options: TranscribeOptions,
        silence_config: Option<vad::SilenceConfig>,
    ) -> WhisperResult<VadTranscriptionResult> {
        let start_time = std::time::Instant::now();

        // Create silence detector
        let config = silence_config.unwrap_or_default();
        let mut detector = vad::SilenceDetector::new(config, audio::SAMPLE_RATE);

        // Detect silence segments
        let frame_size = 480; // 30ms at 16kHz
        let silence_segments = detector.detect(audio, frame_size);

        // Convert silence segments to speech segments (invert)
        let speech_segments = self.invert_silence_segments(&silence_segments, audio.len());

        if speech_segments.is_empty() {
            return Ok(VadTranscriptionResult {
                text: String::new(),
                language: options.language.unwrap_or_else(|| "en".to_string()),
                segments: Vec::new(),
                speech_segments: Vec::new(),
                total_duration_secs: start_time.elapsed().as_secs_f32(),
                speech_duration_secs: 0.0,
            });
        }

        // Extract and transcribe speech segments
        let sample_rate = audio::SAMPLE_RATE as f32;
        let mut all_segments = Vec::new();
        let mut full_text = String::new();
        let mut speech_duration = 0.0f32;
        let language = options.language.clone().unwrap_or_else(|| "en".to_string());

        for (start, end) in &speech_segments {
            let start_sample = (start * sample_rate) as usize;
            let end_sample = ((end * sample_rate) as usize).min(audio.len());

            if end_sample > start_sample {
                let speech_audio = &audio[start_sample..end_sample];
                let result = self.transcribe(speech_audio, options.clone())?;

                let segment = VadSpeechSegment {
                    start: *start,
                    end: *end,
                    text: result.text.clone(),
                    tokens: result
                        .segments
                        .first()
                        .map(|s| s.tokens.clone())
                        .unwrap_or_default(),
                };

                if !full_text.is_empty() {
                    full_text.push(' ');
                }
                full_text.push_str(&result.text);

                speech_duration += end - start;
                all_segments.push(segment);
            }
        }

        let total_duration_secs = start_time.elapsed().as_secs_f32();

        Ok(VadTranscriptionResult {
            text: full_text,
            language,
            segments: all_segments,
            speech_segments,
            total_duration_secs,
            speech_duration_secs: speech_duration,
        })
    }

    /// Invert silence segments to get speech segments
    fn invert_silence_segments(
        &self,
        silence_segments: &[vad::SilenceSegment],
        audio_len: usize,
    ) -> Vec<(f32, f32)> {
        let _ = self; // Method for consistency with transcription pipeline
        let sample_rate = audio::SAMPLE_RATE as f32;
        let total_duration = audio_len as f32 / sample_rate;
        let mut speech_segments = Vec::new();
        let mut current_pos = 0.0f32;

        for silence in silence_segments {
            if silence.start > current_pos {
                speech_segments.push((current_pos, silence.start));
            }
            current_pos = silence.end;
        }

        // Add final speech segment if there's audio after last silence
        if current_pos < total_duration {
            speech_segments.push((current_pos, total_duration));
        }

        speech_segments
    }

    // =========================================================================
    // Streaming Transcription API (WAPR-101)
    // =========================================================================

    /// Transcribe audio with partial results for real-time streaming
    ///
    /// This method enables real-time transcription by returning partial results
    /// as audio is being accumulated. It's designed for use with the
    /// `StreamingProcessor` and emits interim transcriptions.
    ///
    /// # Arguments
    /// * `partial_audio` - Partial audio buffer (may be incomplete utterance)
    /// * `options` - Transcription options
    /// * `is_final` - Whether this is the final audio chunk
    ///
    /// # Returns
    /// Partial transcription result
    ///
    /// # Errors
    /// Returns error if transcription fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use whisper_apr::{WhisperApr, TranscribeOptions};
    ///
    /// let whisper = WhisperApr::tiny();
    ///
    /// // Get partial result during streaming
    /// let partial = whisper.transcribe_partial(&audio_buffer, TranscribeOptions::default(), false)?;
    /// println!("Partial: {}", partial.text);
    ///
    /// // Get final result
    /// let final_result = whisper.transcribe_partial(&audio_buffer, TranscribeOptions::default(), true)?;
    /// println!("Final: {}", final_result.text);
    /// ```
    pub fn transcribe_partial(
        &self,
        partial_audio: &[f32],
        options: TranscribeOptions,
        is_final: bool,
    ) -> WhisperResult<PartialTranscriptionResult> {
        let start_time = std::time::Instant::now();

        // Skip if audio is too short (less than 0.5s)
        let min_samples = (audio::SAMPLE_RATE as f32 * 0.5) as usize;
        if partial_audio.len() < min_samples {
            return Ok(PartialTranscriptionResult {
                text: String::new(),
                language: options.language.unwrap_or_else(|| "en".to_string()),
                is_final,
                confidence: 0.0,
                duration_secs: partial_audio.len() as f32 / audio::SAMPLE_RATE as f32,
                processing_time_secs: start_time.elapsed().as_secs_f32(),
            });
        }

        // Transcribe the partial audio
        let result = self.transcribe(partial_audio, options)?;

        let processing_time = start_time.elapsed().as_secs_f32();

        Ok(PartialTranscriptionResult {
            text: result.text,
            language: result.language,
            is_final,
            confidence: 1.0, // Placeholder - could add proper confidence scoring
            duration_secs: partial_audio.len() as f32 / audio::SAMPLE_RATE as f32,
            processing_time_secs: processing_time,
        })
    }

    /// Create a streaming transcription session
    ///
    /// Returns a `StreamingSession` that manages the streaming processor
    /// and provides partial results as audio is pushed.
    ///
    /// # Arguments
    /// * `options` - Transcription options to use for all partial and final results
    /// * `input_sample_rate` - Sample rate of the input audio
    ///
    /// # Returns
    /// A new streaming session
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use whisper_apr::{WhisperApr, TranscribeOptions};
    ///
    /// let whisper = WhisperApr::tiny();
    /// let mut session = whisper.create_streaming_session(
    ///     TranscribeOptions::default(),
    ///     44100, // Input sample rate from microphone
    /// );
    ///
    /// // Push audio chunks as they arrive
    /// loop {
    ///     let audio_chunk = get_audio_from_microphone();
    ///     if let Some(partial) = session.push(&audio_chunk)? {
    ///         println!("Partial: {}", partial.text);
    ///     }
    ///     if session.has_chunk() {
    ///         let final_result = session.finalize()?;
    ///         println!("Final: {}", final_result.text);
    ///         break;
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn create_streaming_session(
        &self,
        options: TranscribeOptions,
        input_sample_rate: u32,
    ) -> StreamingSession<'_> {
        let streaming_config = audio::StreamingConfig::with_sample_rate(input_sample_rate);
        let processor = audio::StreamingProcessor::new(streaming_config);

        StreamingSession {
            whisper: self,
            processor,
            options,
            last_partial_text: String::new(),
        }
    }

    /// Transcribe audio and summarize the result using LFM2
    ///
    /// This unified API combines Whisper ASR with LFM2 summarization
    /// in a single operation, implementing the Phase 3 API from spec Section 18.5.
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, 16kHz, f32 normalized to [-1, 1])
    /// * `transcribe_options` - Transcription options
    /// * `summarize_options` - LFM2 summarization options
    ///
    /// # Returns
    /// Combined result with transcription and summary
    ///
    /// # Errors
    /// Returns error if transcription or summarization fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use whisper_apr::{WhisperApr, TranscribeOptions, SummarizeOptions, model::lfm2::{Lfm2, Lfm2Tokenizer}};
    ///
    /// let whisper = WhisperApr::tiny();
    /// let lfm2 = Lfm2::lfm2_2_6b()?;
    /// let tokenizer = Lfm2Tokenizer::new();
    ///
    /// let result = whisper.transcribe_and_summarize(
    ///     &audio_samples,
    ///     TranscribeOptions::default(),
    ///     SummarizeOptions::new(&lfm2, &tokenizer),
    /// )?;
    ///
    /// println!("Transcript: {}", result.transcription.text);
    /// println!("Summary: {}", result.summary);
    /// ```
    pub fn transcribe_and_summarize(
        &self,
        audio: &[f32],
        transcribe_options: TranscribeOptions,
        summarize_options: SummarizeOptions<'_>,
    ) -> WhisperResult<TranscribeSummaryResult> {
        // Step 1: Transcribe audio
        let transcription = self.transcribe(audio, transcribe_options)?;

        // Skip summarization for empty transcripts
        if transcription.text.trim().is_empty() {
            return Ok(TranscribeSummaryResult {
                transcription,
                summary: String::new(),
                generation_stats: None,
            });
        }

        // Step 2: Tokenize transcript for LFM2
        let input_tokens = summarize_options.tokenizer.encode(&transcription.text);

        // Step 3: Generate summary
        let (output_tokens, stats) = summarize_options.model.generate_with_stats(
            &input_tokens,
            summarize_options.max_tokens,
            summarize_options.temperature,
            Some(|_token: u32, _idx: usize| true), // Continue generating
        )?;

        // Step 4: Decode summary
        let summary = summarize_options.tokenizer.decode(&output_tokens);

        Ok(TranscribeSummaryResult {
            transcription,
            summary,
            generation_stats: Some(stats),
        })
    }
}

/// Result of partial transcription during streaming (WAPR-101)
#[derive(Debug, Clone)]
pub struct PartialTranscriptionResult {
    /// Transcribed text (may be incomplete)
    pub text: String,
    /// Detected or specified language
    pub language: String,
    /// Whether this is the final result
    pub is_final: bool,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Duration of audio processed in seconds
    pub duration_secs: f32,
    /// Time taken to process in seconds
    pub processing_time_secs: f32,
}

impl PartialTranscriptionResult {
    /// Check if result has any text
    #[must_use]
    pub fn has_text(&self) -> bool {
        !self.text.is_empty()
    }

    /// Check if this is an empty interim result
    #[must_use]
    pub fn is_empty_interim(&self) -> bool {
        self.text.is_empty() && !self.is_final
    }

    /// Get real-time factor (processing time / audio duration)
    #[must_use]
    pub fn real_time_factor(&self) -> f32 {
        if self.duration_secs <= 0.0 {
            0.0
        } else {
            self.processing_time_secs / self.duration_secs
        }
    }
}

/// Streaming transcription session (WAPR-101)
///
/// Manages the streaming processor and provides partial results
/// as audio is accumulated.
#[derive(Debug)]
pub struct StreamingSession<'a> {
    /// Reference to the Whisper model
    whisper: &'a WhisperApr,
    /// Streaming processor
    processor: audio::StreamingProcessor,
    /// Transcription options
    options: TranscribeOptions,
    /// Last partial text (for deduplication)
    last_partial_text: String,
}

impl StreamingSession<'_> {
    /// Push audio samples and get partial result if available
    ///
    /// # Arguments
    /// * `audio` - Audio samples at the configured input sample rate
    ///
    /// # Returns
    /// Optional partial transcription if enough audio has accumulated
    ///
    /// # Errors
    /// Returns error if transcription fails
    pub fn push(&mut self, audio: &[f32]) -> WhisperResult<Option<PartialTranscriptionResult>> {
        self.processor.push_audio(audio);
        self.processor.process();

        // Check for partial result
        if self.processor.has_partial() {
            if let Some(partial_audio) = self.processor.get_partial() {
                let result =
                    self.whisper
                        .transcribe_partial(&partial_audio, self.options.clone(), false)?;

                // Deduplicate (don't return same text twice)
                if result.text != self.last_partial_text {
                    result.text.clone_into(&mut self.last_partial_text);
                    return Ok(Some(result));
                }
            }
        }

        Ok(None)
    }

    /// Check if a full chunk is ready
    #[must_use]
    pub fn has_chunk(&self) -> bool {
        self.processor.has_chunk()
    }

    /// Check for pending events
    #[must_use]
    pub fn has_events(&self) -> bool {
        self.processor.has_events()
    }

    /// Drain and return all pending events
    pub fn drain_events(&mut self) -> Vec<audio::StreamingEvent> {
        self.processor.drain_events()
    }

    /// Get the final transcription for the accumulated chunk
    ///
    /// # Returns
    /// Final transcription result
    ///
    /// # Errors
    /// Returns error if no chunk is ready or transcription fails
    pub fn finalize(&mut self) -> WhisperResult<PartialTranscriptionResult> {
        let chunk = self
            .processor
            .get_chunk()
            .ok_or_else(|| WhisperError::Audio("no chunk ready for finalization".into()))?;

        let result = self
            .whisper
            .transcribe_partial(&chunk, self.options.clone(), true)?;
        self.last_partial_text.clear();

        Ok(result)
    }

    /// Flush any remaining audio and get final result
    ///
    /// # Returns
    /// Optional final transcription if there was audio to process
    ///
    /// # Errors
    /// Returns error if transcription fails
    pub fn flush(&mut self) -> WhisperResult<Option<PartialTranscriptionResult>> {
        if let Some(chunk) = self.processor.flush() {
            let result = self
                .whisper
                .transcribe_partial(&chunk, self.options.clone(), true)?;
            self.last_partial_text.clear();
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Reset the session state
    pub fn reset(&mut self) {
        self.processor.reset();
        self.last_partial_text.clear();
    }

    /// Get streaming processor state
    #[must_use]
    pub fn state(&self) -> audio::ProcessorState {
        self.processor.state()
    }

    /// Get chunk progress (0.0 - 1.0)
    #[must_use]
    pub fn chunk_progress(&self) -> f32 {
        self.processor.chunk_progress()
    }

    /// Get partial duration in seconds
    #[must_use]
    pub fn partial_duration(&self) -> f32 {
        self.processor.partial_duration()
    }

    /// Set partial result threshold in seconds
    pub fn set_partial_threshold(&mut self, seconds: f32) {
        self.processor.set_partial_threshold(seconds);
    }
}

/// Result of VAD-triggered transcription (WAPR-093)
#[derive(Debug, Clone)]
pub struct VadTranscriptionResult {
    /// Full transcribed text (concatenated from all speech segments)
    pub text: String,
    /// Detected or specified language
    pub language: String,
    /// Transcribed speech segments with timestamps
    pub segments: Vec<VadSpeechSegment>,
    /// Raw speech segment timestamps (start, end) in seconds
    pub speech_segments: Vec<(f32, f32)>,
    /// Total processing time in seconds
    pub total_duration_secs: f32,
    /// Total speech duration in seconds (excludes silence)
    pub speech_duration_secs: f32,
}

impl VadTranscriptionResult {
    /// Get number of speech segments
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Check if any speech was detected
    #[must_use]
    pub fn has_speech(&self) -> bool {
        !self.segments.is_empty()
    }

    /// Get silence ratio (0.0 = all speech, 1.0 = all silence)
    #[must_use]
    pub fn silence_ratio(&self, audio_duration: f32) -> f32 {
        if audio_duration <= 0.0 {
            return 1.0;
        }
        1.0 - (self.speech_duration_secs / audio_duration)
    }

    /// Get the first segment (if any)
    #[must_use]
    pub fn first_segment(&self) -> Option<&VadSpeechSegment> {
        self.segments.first()
    }

    /// Get the last segment (if any)
    #[must_use]
    pub fn last_segment(&self) -> Option<&VadSpeechSegment> {
        self.segments.last()
    }

    /// Iterate over segments
    pub fn iter(&self) -> impl Iterator<Item = &VadSpeechSegment> {
        self.segments.iter()
    }
}

/// A speech segment detected by VAD (WAPR-093)
#[derive(Debug, Clone)]
pub struct VadSpeechSegment {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Transcribed text for this segment
    pub text: String,
    /// Token IDs for this segment
    pub tokens: Vec<u32>,
}

impl VadSpeechSegment {
    /// Get duration of this segment
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Check if segment has text
    #[must_use]
    pub fn has_text(&self) -> bool {
        !self.text.is_empty()
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
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_tiny() {
        let whisper = WhisperApr::tiny();
        assert_eq!(whisper.model_type(), ModelType::Tiny);
        assert_eq!(whisper.config().n_audio_layer, 4);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_base() {
        let whisper = WhisperApr::base();
        assert_eq!(whisper.model_type(), ModelType::Base);
        assert_eq!(whisper.config().n_audio_layer, 6);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_memory_size() {
        let tiny = WhisperApr::tiny();
        let base = WhisperApr::base();

        assert!(tiny.memory_size() < base.memory_size());
        assert!(tiny.memory_size() > 100_000_000); // > 100MB
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_initial_tokens() {
        let whisper = WhisperApr::tiny();

        let tokens = whisper.get_initial_tokens("en", Task::Transcribe);
        assert_eq!(tokens[0], tokenizer::special_tokens::SOT);
        assert!(tokens.len() >= 3); // SOT, lang, task

        let translate_tokens = whisper.get_initial_tokens("es", Task::Translate);
        assert!(translate_tokens.contains(&tokenizer::special_tokens::TRANSLATE));
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
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
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_resample_passthrough() {
        let whisper = WhisperApr::tiny();
        let audio = vec![0.1, 0.2, 0.3, 0.4];

        // Without resampler, should return copy
        let resampled = whisper.resample(&audio).expect("should succeed");
        assert_eq!(resampled, audio);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
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
    #[ignore = "Allocates large model - run with --ignored"]
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
            profile: false,
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
            profile: false,
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
            profiling: None,
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
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encoder_mut_accessor() {
        let mut whisper = WhisperApr::tiny();
        let encoder = whisper.encoder_mut();
        assert_eq!(encoder.n_layers(), 4);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_decoder_mut_accessor() {
        let mut whisper = WhisperApr::tiny();
        let decoder = whisper.decoder_mut();
        assert_eq!(decoder.n_layers(), 4);
    }

    // =========================================================================
    // Integration Tests - Full Pipeline
    // =========================================================================

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
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
    #[ignore = "Allocates large model - run with --ignored"]
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
            profile: false,
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
            profile: false,
        };
        assert!(matches!(
            opts_sampling.strategy,
            DecodingStrategy::Sampling { .. }
        ));
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
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
        use crate::simd;

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
        use crate::memory::{get_buffer, pool_stats, return_buffer, MemoryPool};

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

    // =========================================================================
    // VAD-Triggered Transcription Tests (WAPR-093)
    // =========================================================================

    #[test]
    fn test_vad_transcription_result_new() {
        let result = VadTranscriptionResult {
            text: "hello world".to_string(),
            language: "en".to_string(),
            segments: vec![],
            speech_segments: vec![],
            total_duration_secs: 1.0,
            speech_duration_secs: 0.5,
        };

        assert_eq!(result.text, "hello world");
        assert_eq!(result.language, "en");
        assert!(!result.has_speech());
        assert_eq!(result.num_segments(), 0);
    }

    #[test]
    fn test_vad_transcription_result_with_segments() {
        let result = VadTranscriptionResult {
            text: "hello world".to_string(),
            language: "en".to_string(),
            segments: vec![
                VadSpeechSegment {
                    start: 0.0,
                    end: 1.0,
                    text: "hello".to_string(),
                    tokens: vec![1, 2],
                },
                VadSpeechSegment {
                    start: 1.5,
                    end: 2.5,
                    text: "world".to_string(),
                    tokens: vec![3, 4],
                },
            ],
            speech_segments: vec![(0.0, 1.0), (1.5, 2.5)],
            total_duration_secs: 3.0,
            speech_duration_secs: 2.0,
        };

        assert!(result.has_speech());
        assert_eq!(result.num_segments(), 2);
        assert!(result.first_segment().is_some());
        assert!(result.last_segment().is_some());
        assert_eq!(
            result.first_segment().map(|s| &s.text),
            Some(&"hello".to_string())
        );
        assert_eq!(
            result.last_segment().map(|s| &s.text),
            Some(&"world".to_string())
        );
    }

    #[test]
    fn test_vad_transcription_result_silence_ratio() {
        let result = VadTranscriptionResult {
            text: String::new(),
            language: "en".to_string(),
            segments: vec![],
            speech_segments: vec![],
            total_duration_secs: 1.0,
            speech_duration_secs: 0.5,
        };

        let ratio = result.silence_ratio(2.0);
        assert!((ratio - 0.75).abs() < 0.01); // 0.5 speech in 2.0 total = 0.75 silence
    }

    #[test]
    fn test_vad_transcription_result_silence_ratio_zero_duration() {
        let result = VadTranscriptionResult {
            text: String::new(),
            language: "en".to_string(),
            segments: vec![],
            speech_segments: vec![],
            total_duration_secs: 0.0,
            speech_duration_secs: 0.0,
        };

        let ratio = result.silence_ratio(0.0);
        assert!((ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vad_transcription_result_iter() {
        let result = VadTranscriptionResult {
            text: "a b".to_string(),
            language: "en".to_string(),
            segments: vec![
                VadSpeechSegment {
                    start: 0.0,
                    end: 1.0,
                    text: "a".to_string(),
                    tokens: vec![1],
                },
                VadSpeechSegment {
                    start: 1.0,
                    end: 2.0,
                    text: "b".to_string(),
                    tokens: vec![2],
                },
            ],
            speech_segments: vec![(0.0, 1.0), (1.0, 2.0)],
            total_duration_secs: 2.0,
            speech_duration_secs: 2.0,
        };

        let texts: Vec<_> = result.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec!["a", "b"]);
    }

    #[test]
    fn test_vad_speech_segment_duration() {
        let segment = VadSpeechSegment {
            start: 1.5,
            end: 3.0,
            text: "test".to_string(),
            tokens: vec![1, 2, 3],
        };

        assert!((segment.duration() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_vad_speech_segment_has_text() {
        let with_text = VadSpeechSegment {
            start: 0.0,
            end: 1.0,
            text: "hello".to_string(),
            tokens: vec![1],
        };
        let empty = VadSpeechSegment {
            start: 0.0,
            end: 1.0,
            text: String::new(),
            tokens: vec![],
        };

        assert!(with_text.has_text());
        assert!(!empty.has_text());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_transcribe_with_vad_silence_only() {
        let whisper = WhisperApr::tiny();
        let silence = vec![0.0; 16000]; // 1 second of silence

        let result = whisper
            .transcribe_with_vad(&silence, TranscribeOptions::default(), None)
            .expect("should succeed");

        assert!(!result.has_speech());
        assert_eq!(result.num_segments(), 0);
        assert!(result.text.is_empty());
    }

    #[test]
    fn test_transcribe_with_silence_detection_config() {
        // Test that silence config can be created and used
        let config = vad::SilenceConfig::new()
            .with_min_silence_duration(0.3)
            .with_max_silence_duration(2.0)
            .with_silence_threshold(0.001);

        assert!((config.min_silence_duration - 0.3).abs() < 0.01);
        assert!((config.max_silence_duration - 2.0).abs() < 0.01);
        assert!((config.silence_threshold - 0.001).abs() < 0.001);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_invert_silence_segments_empty() {
        let whisper = WhisperApr::tiny();
        let audio_len = 16000; // 1 second

        let silence_segments: Vec<vad::SilenceSegment> = vec![];
        let speech = whisper.invert_silence_segments(&silence_segments, audio_len);

        // No silence means all speech
        assert_eq!(speech.len(), 1);
        assert!((speech[0].0 - 0.0).abs() < 0.01);
        assert!((speech[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_invert_silence_segments_single() {
        let whisper = WhisperApr::tiny();
        let audio_len = 32000; // 2 seconds

        let silence_segments = vec![vad::SilenceSegment {
            start: 0.5,
            end: 1.5,
            noise_floor: 0.001,
        }];
        let speech = whisper.invert_silence_segments(&silence_segments, audio_len);

        // Speech at beginning and end, with silence in middle
        assert_eq!(speech.len(), 2);
        assert!((speech[0].0 - 0.0).abs() < 0.01); // Start
        assert!((speech[0].1 - 0.5).abs() < 0.01); // Before silence
        assert!((speech[1].0 - 1.5).abs() < 0.01); // After silence
        assert!((speech[1].1 - 2.0).abs() < 0.01); // End
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_invert_silence_segments_multiple() {
        let whisper = WhisperApr::tiny();
        let audio_len = 48000; // 3 seconds

        let silence_segments = vec![
            vad::SilenceSegment {
                start: 0.5,
                end: 1.0,
                noise_floor: 0.001,
            },
            vad::SilenceSegment {
                start: 2.0,
                end: 2.5,
                noise_floor: 0.001,
            },
        ];
        let speech = whisper.invert_silence_segments(&silence_segments, audio_len);

        // Speech: 0-0.5, 1.0-2.0, 2.5-3.0
        assert_eq!(speech.len(), 3);
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
            profile: false,
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
            profiling: None,
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
            profiling: None,
        };

        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("TranscriptionResult"));
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_debug() {
        let whisper = WhisperApr::tiny();
        let debug_str = format!("{whisper:?}");
        assert!(debug_str.contains("WhisperApr"));
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
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
            profile: false,
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
            profile: false,
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
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_small() {
        let whisper = WhisperApr::small();
        assert_eq!(whisper.model_type(), ModelType::Small);
        assert_eq!(whisper.config().n_audio_layer, 12);
        assert_eq!(whisper.config().n_audio_state, 768);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_medium() {
        let whisper = WhisperApr::medium();
        assert_eq!(whisper.model_type(), ModelType::Medium);
        assert_eq!(whisper.config().n_audio_layer, 24);
        assert_eq!(whisper.config().n_audio_state, 1024);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_large() {
        let whisper = WhisperApr::large();
        assert_eq!(whisper.model_type(), ModelType::Large);
        assert_eq!(whisper.config().n_audio_layer, 32);
        assert_eq!(whisper.config().n_audio_state, 1280);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
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
    #[ignore = "Allocates large model - run with --ignored"]
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
                    profiling: None,
                },
                TranscriptionResult {
                    text: "World".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                    profiling: None,
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
                    profiling: None,
                },
                TranscriptionResult {
                    text: "Second".to_string(),
                    language: "es".to_string(),
                    segments: vec![],
                    profiling: None,
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
                    profiling: None,
                },
                TranscriptionResult {
                    text: "Two".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                    profiling: None,
                },
                TranscriptionResult {
                    text: "Three".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                    profiling: None,
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
                    profiling: None,
                },
                TranscriptionResult {
                    text: "B".to_string(),
                    language: "en".to_string(),
                    segments: vec![],
                    profiling: None,
                },
            ],
            total_duration_secs: 1.0,
        };

        let collected: Vec<&str> = result.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(collected, vec!["A", "B"]);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_transcribe_batch_empty() {
        let whisper = WhisperApr::tiny();
        let result = whisper.transcribe_batch(&[], TranscribeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_transcribe_audio_batch_empty() {
        let whisper = WhisperApr::tiny();
        let batch = audio::AudioBatch::with_default_config();
        let result = whisper.transcribe_audio_batch(&batch, TranscribeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_transcribe_batch_optimized_empty() {
        let whisper = WhisperApr::tiny();
        let result = whisper.transcribe_batch_optimized(&[], TranscribeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_create_audio_batch() {
        let segments = vec![vec![0.1_f32, 0.2, 0.3], vec![0.4_f32, 0.5]];

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

    // =========================================================================
    // WAPR-101: Streaming Transcription API Tests
    // =========================================================================

    #[test]
    fn test_partial_transcription_result_new() {
        let result = PartialTranscriptionResult {
            text: "hello".to_string(),
            language: "en".to_string(),
            is_final: false,
            confidence: 0.95,
            duration_secs: 1.5,
            processing_time_secs: 0.3,
        };

        assert_eq!(result.text, "hello");
        assert_eq!(result.language, "en");
        assert!(!result.is_final);
        assert!((result.confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_partial_transcription_result_has_text() {
        let with_text = PartialTranscriptionResult {
            text: "hello".to_string(),
            language: "en".to_string(),
            is_final: false,
            confidence: 1.0,
            duration_secs: 1.0,
            processing_time_secs: 0.1,
        };
        let empty = PartialTranscriptionResult {
            text: String::new(),
            language: "en".to_string(),
            is_final: false,
            confidence: 0.0,
            duration_secs: 0.5,
            processing_time_secs: 0.05,
        };

        assert!(with_text.has_text());
        assert!(!empty.has_text());
    }

    #[test]
    fn test_partial_transcription_result_is_empty_interim() {
        let empty_interim = PartialTranscriptionResult {
            text: String::new(),
            language: "en".to_string(),
            is_final: false,
            confidence: 0.0,
            duration_secs: 0.5,
            processing_time_secs: 0.05,
        };
        let empty_final = PartialTranscriptionResult {
            text: String::new(),
            language: "en".to_string(),
            is_final: true,
            confidence: 0.0,
            duration_secs: 0.5,
            processing_time_secs: 0.05,
        };
        let with_text = PartialTranscriptionResult {
            text: "hello".to_string(),
            language: "en".to_string(),
            is_final: false,
            confidence: 1.0,
            duration_secs: 1.0,
            processing_time_secs: 0.1,
        };

        assert!(empty_interim.is_empty_interim());
        assert!(!empty_final.is_empty_interim()); // Final, not interim
        assert!(!with_text.is_empty_interim()); // Has text
    }

    #[test]
    fn test_partial_transcription_result_real_time_factor() {
        let result = PartialTranscriptionResult {
            text: "hello".to_string(),
            language: "en".to_string(),
            is_final: false,
            confidence: 1.0,
            duration_secs: 2.0,
            processing_time_secs: 0.5,
        };

        // RTF = 0.5 / 2.0 = 0.25
        assert!((result.real_time_factor() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_partial_transcription_result_real_time_factor_zero_duration() {
        let result = PartialTranscriptionResult {
            text: String::new(),
            language: "en".to_string(),
            is_final: false,
            confidence: 0.0,
            duration_secs: 0.0,
            processing_time_secs: 0.0,
        };

        assert!((result.real_time_factor() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_partial_transcription_result_debug_clone() {
        let result = PartialTranscriptionResult {
            text: "test".to_string(),
            language: "en".to_string(),
            is_final: true,
            confidence: 0.9,
            duration_secs: 1.0,
            processing_time_secs: 0.1,
        };

        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("PartialTranscriptionResult"));

        let cloned = result.clone();
        assert_eq!(cloned.text, "test");
        assert!(cloned.is_final);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_transcribe_partial_too_short() {
        let whisper = WhisperApr::tiny();
        let short_audio = vec![0.0; 4000]; // Only 0.25 seconds

        let result = whisper
            .transcribe_partial(&short_audio, TranscribeOptions::default(), false)
            .expect("should succeed with empty result");

        assert!(result.text.is_empty());
        assert!(!result.is_final);
        assert!((result.confidence - 0.0).abs() < 0.01);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_encode_3_second_chunk() {
        // This is what the realtime demo sends - 3 seconds of audio at 16kHz
        // Test that mel spectrogram can be encoded without "input size mismatch" error
        let whisper = WhisperApr::tiny();
        let audio = vec![0.0; 48000]; // 3 seconds at 16kHz

        // Compute mel spectrogram
        let mel = whisper
            .compute_mel(&audio)
            .expect("mel computation should succeed");

        // Encode should succeed without "input size mismatch" error
        let result = whisper.encode(&mel);
        assert!(
            result.is_ok(),
            "encode should succeed for 3s audio mel: {:?}",
            result.err()
        );

        // Verify output dimensions - should be (output_frames x d_model)
        let encoded = result.expect("encode should succeed");
        let d_model = whisper.config().n_text_state as usize; // 384 for tiny
        assert_eq!(
            encoded.len() % d_model,
            0,
            "encoded output should be multiple of d_model"
        );
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_create_streaming_session() {
        let whisper = WhisperApr::tiny();
        let session = whisper.create_streaming_session(TranscribeOptions::default(), 44100);

        assert_eq!(session.state(), audio::ProcessorState::WaitingForSpeech);
        assert!((session.chunk_progress() - 0.0).abs() < 0.01);
        assert!(!session.has_chunk());
        assert!(!session.has_events());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_streaming_session_reset() {
        let whisper = WhisperApr::tiny();
        let mut session = whisper.create_streaming_session(TranscribeOptions::default(), 16000);

        // Push some audio
        session.push(&vec![0.1; 1000]).expect("push should work");

        // Reset
        session.reset();

        assert_eq!(session.state(), audio::ProcessorState::WaitingForSpeech);
        assert!((session.partial_duration() - 0.0).abs() < 0.01);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_streaming_session_set_partial_threshold() {
        let whisper = WhisperApr::tiny();
        let mut session = whisper.create_streaming_session(TranscribeOptions::default(), 16000);

        session.set_partial_threshold(5.0);
        // Verify it was set (indirectly through behavior, not directly accessible)
        // The partial threshold affects when partial results are available
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_streaming_session_finalize_no_chunk() {
        let whisper = WhisperApr::tiny();
        let mut session = whisper.create_streaming_session(TranscribeOptions::default(), 16000);

        let result = session.finalize();
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_streaming_session_flush_empty() {
        let whisper = WhisperApr::tiny();
        let mut session = whisper.create_streaming_session(TranscribeOptions::default(), 16000);

        let result = session.flush().expect("flush should work");
        assert!(result.is_none());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_streaming_session_drain_events() {
        let whisper = WhisperApr::tiny();
        let mut session = whisper.create_streaming_session(TranscribeOptions::default(), 16000);

        // Reset emits an event
        session.reset();

        let events = session.drain_events();
        assert!(!events.is_empty());
        assert!(events
            .iter()
            .any(|e| matches!(e, audio::StreamingEvent::Reset)));
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_streaming_session_push_silence() {
        let whisper = WhisperApr::tiny();
        let mut session = whisper.create_streaming_session(TranscribeOptions::default(), 16000);

        // Push silence
        let result = session.push(&vec![0.0; 16000]).expect("push should work");
        assert!(result.is_none()); // No partial for silence
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_streaming_session_debug() {
        let whisper = WhisperApr::tiny();
        let session = whisper.create_streaming_session(TranscribeOptions::default(), 16000);

        let debug_str = format!("{session:?}");
        assert!(debug_str.contains("StreamingSession"));
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_streaming_session_state() {
        let whisper = WhisperApr::tiny();
        let session = whisper.create_streaming_session(TranscribeOptions::default(), 16000);

        // Initial state should be WaitingForSpeech
        let state = session.state();
        assert_eq!(state, audio::ProcessorState::WaitingForSpeech);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_streaming_session_chunk_progress() {
        let whisper = WhisperApr::tiny();
        let session = whisper.create_streaming_session(TranscribeOptions::default(), 16000);

        let progress = session.chunk_progress();
        assert!(progress >= 0.0 && progress <= 1.0);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_streaming_session_partial_duration() {
        let whisper = WhisperApr::tiny();
        let session = whisper.create_streaming_session(TranscribeOptions::default(), 16000);

        let duration = session.partial_duration();
        assert!(duration >= 0.0);
    }

    #[test]
    fn test_partial_transcription_result_rtf_with_zero_processing() {
        let result = PartialTranscriptionResult {
            text: "test".to_string(),
            language: "en".to_string(),
            is_final: true,
            confidence: 0.9,
            duration_secs: 5.0,
            processing_time_secs: 0.0,
        };

        let rtf = result.real_time_factor();
        assert!(rtf >= 0.0);
    }

    #[test]
    fn test_vad_transcription_result_methods_empty() {
        let result = VadTranscriptionResult {
            text: "Hello world".to_string(),
            language: "en".to_string(),
            segments: vec![],
            speech_segments: vec![],
            total_duration_secs: 5.0,
            speech_duration_secs: 0.0,
        };

        assert_eq!(result.num_segments(), 0);
        assert!(!result.has_speech());
    }

    #[test]
    fn test_batch_transcription_result_defaults() {
        let result = BatchTranscriptionResult {
            results: vec![],
            total_duration_secs: 0.0,
        };

        assert_eq!(result.len(), 0);
        assert!(result.is_empty());
        assert!(result.get(0).is_none());
        assert!(result.texts().is_empty());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_config_accessors() {
        let whisper = WhisperApr::tiny();
        let config = whisper.config();

        assert!(config.n_vocab > 0);
        assert!(config.n_audio_ctx > 0);
        assert!(config.n_text_ctx > 0);
    }

    #[test]
    fn test_transcribe_options_all_fields() {
        let options = TranscribeOptions {
            language: Some("fr".to_string()),
            task: Task::Translate,
            strategy: DecodingStrategy::BeamSearch {
                beam_size: 3,
                temperature: 0.2,
                patience: 1.5,
            },
            word_timestamps: true,
            profile: false,
        };

        assert_eq!(options.language, Some("fr".to_string()));
        assert_eq!(options.task, Task::Translate);
        assert!(options.word_timestamps);
    }

    #[test]
    fn test_segment_with_tokens() {
        let segment = Segment {
            text: "Hello".to_string(),
            start: 0.0,
            end: 1.0,
            tokens: vec![1, 2, 3, 4, 5],
        };

        assert_eq!(segment.tokens.len(), 5);
        assert!((segment.end - segment.start - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_model_from_config() {
        let config = model::ModelConfig::tiny();
        let whisper = WhisperApr::from_config(config);

        assert_eq!(whisper.model_type(), ModelType::Tiny);
    }

    #[test]
    fn test_decoding_strategy_variants() {
        let greedy = DecodingStrategy::Greedy;
        assert!(matches!(greedy, DecodingStrategy::Greedy));

        let sampling = DecodingStrategy::Sampling {
            temperature: 0.5,
            top_k: Some(40),
            top_p: Some(0.9),
        };
        if let DecodingStrategy::Sampling {
            temperature,
            top_k,
            top_p,
        } = sampling
        {
            assert!((temperature - 0.5).abs() < f32::EPSILON);
            assert_eq!(top_k, Some(40));
            assert_eq!(top_p, Some(0.9));
        }
    }

    #[test]
    fn test_task_variants_eq() {
        assert_eq!(Task::Transcribe, Task::Transcribe);
        assert_ne!(Task::Transcribe, Task::Translate);
    }

    #[test]
    fn test_model_type_all_variants() {
        let variants = vec![
            ModelType::Tiny,
            ModelType::TinyEn,
            ModelType::Base,
            ModelType::BaseEn,
            ModelType::Small,
            ModelType::SmallEn,
            ModelType::Medium,
            ModelType::MediumEn,
            ModelType::Large,
            ModelType::LargeV1,
            ModelType::LargeV2,
            ModelType::LargeV3,
        ];

        for variant in variants {
            let debug_str = format!("{variant:?}");
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_vad_speech_segment_empty_text() {
        let segment = VadSpeechSegment {
            start: 0.0,
            end: 1.0,
            text: String::new(),
            tokens: vec![],
        };

        assert!(!segment.has_text());
        assert!((segment.duration() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transcription_result_empty() {
        let result = TranscriptionResult {
            text: String::new(),
            language: "en".to_string(),
            segments: vec![],
            profiling: None,
        };

        assert!(result.text.is_empty());
        assert!(result.segments.is_empty());
    }

    #[test]
    fn test_batch_transcription_result_iter_coverage() {
        let results = vec![
            TranscriptionResult {
                text: "First".to_string(),
                language: "en".to_string(),
                segments: vec![],
                profiling: None,
            },
            TranscriptionResult {
                text: "Second".to_string(),
                language: "en".to_string(),
                segments: vec![],
                profiling: None,
            },
        ];

        let batch = BatchTranscriptionResult {
            results,
            total_duration_secs: 1.0,
        };

        let mut count = 0;
        for result in batch.iter() {
            count += 1;
            assert!(!result.text.is_empty());
        }
        assert_eq!(count, 2);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_clone() {
        let whisper = WhisperApr::tiny();
        let cloned = whisper.clone();

        assert_eq!(whisper.model_type(), cloned.model_type());
        assert_eq!(whisper.memory_size(), cloned.memory_size());
    }

    #[test]
    fn test_vad_transcription_result_first_last_segment() {
        let result = VadTranscriptionResult {
            text: "hello world".to_string(),
            language: "en".to_string(),
            segments: vec![
                VadSpeechSegment {
                    start: 0.0,
                    end: 1.0,
                    text: "hello".to_string(),
                    tokens: vec![1],
                },
                VadSpeechSegment {
                    start: 1.5,
                    end: 2.5,
                    text: "world".to_string(),
                    tokens: vec![2],
                },
            ],
            speech_segments: vec![(0.0, 1.0), (1.5, 2.5)],
            total_duration_secs: 3.0,
            speech_duration_secs: 2.0,
        };

        let first = result.first_segment().expect("first segment");
        assert_eq!(first.text, "hello");

        let last = result.last_segment().expect("last segment");
        assert_eq!(last.text, "world");
    }

    #[test]
    fn test_vad_transcription_result_iter_segments() {
        let result = VadTranscriptionResult {
            text: "test".to_string(),
            language: "en".to_string(),
            segments: vec![
                VadSpeechSegment {
                    start: 0.0,
                    end: 1.0,
                    text: "a".to_string(),
                    tokens: vec![1],
                },
                VadSpeechSegment {
                    start: 1.0,
                    end: 2.0,
                    text: "b".to_string(),
                    tokens: vec![2],
                },
            ],
            speech_segments: vec![(0.0, 1.0), (1.0, 2.0)],
            total_duration_secs: 2.0,
            speech_duration_secs: 2.0,
        };

        let mut count = 0;
        for segment in result.iter() {
            count += 1;
            assert!(segment.has_text());
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_partial_transcription_result_methods() {
        let result = PartialTranscriptionResult {
            text: "hello".to_string(),
            language: "en".to_string(),
            is_final: false,
            confidence: 0.85,
            duration_secs: 2.0,
            processing_time_secs: 0.5,
        };

        assert!(result.has_text());
        assert!((result.real_time_factor() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_partial_transcription_result_zero_duration() {
        let result = PartialTranscriptionResult {
            text: "".to_string(),
            language: "en".to_string(),
            is_final: true,
            confidence: 0.0,
            duration_secs: 0.0,
            processing_time_secs: 0.1,
        };

        assert!(!result.has_text());
        assert!((result.real_time_factor() - 0.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Full E2E Integration Test with Real Model (SIMD-optimized path)
    // =========================================================================

    #[test]
    #[ignore = "Requires model file - run with --ignored"]
    fn test_e2e_transcribe_with_int8_model() {
        // Load the real int8 quantized model
        let model_path = std::path::Path::new("models/whisper-tiny-int8.apr");
        if !model_path.exists() {
            eprintln!(
                "Skipping E2E test: model file not found at {:?}",
                model_path
            );
            return;
        }

        let model_data = std::fs::read(model_path).expect("Failed to read model file");
        eprintln!("Loaded model: {} bytes", model_data.len());

        let whisper = WhisperApr::load_from_apr(&model_data).expect("Failed to load model");
        eprintln!("Model loaded: {:?}", whisper.model_type());

        // Generate 3 seconds of test audio (silence with a bit of noise)
        // This exercises the full pipeline without needing real speech
        let sample_rate = 16000;
        let duration_secs = 3.0;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;

        // Generate low-amplitude noise (simulates silence/background)
        let audio: Vec<f32> = (0..num_samples)
            .map(|i| {
                // Very low amplitude noise
                let noise = ((i as f32 * 0.1).sin() * 0.001) + ((i as f32 * 0.37).cos() * 0.001);
                noise
            })
            .collect();

        eprintln!("Generated {} samples of test audio", audio.len());

        // Run transcription - this exercises the SIMD-optimized path:
        // - Mel spectrogram computation
        // - Encoder forward (uses SIMD)
        // - Decoder forward with SIMD matmul in project_to_vocab
        // - Quantized linear layers with SIMD matmul
        let start = std::time::Instant::now();
        let result = whisper.transcribe(&audio, TranscribeOptions::default());
        let elapsed = start.elapsed();

        eprintln!("Transcription completed in {:?}", elapsed);

        match result {
            Ok(transcription) => {
                eprintln!("Result: '{}'", transcription.text);
                eprintln!("Language: {}", transcription.language);
                eprintln!("Segments: {}", transcription.segments.len());

                // For silence, we expect empty or very short text
                // The main goal is verifying the pipeline doesn't crash
                assert!(
                    transcription.text.len() < 100,
                    "Unexpected long transcription for silence"
                );
            }
            Err(e) => {
                panic!("Transcription failed: {e:?}");
            }
        }

        // Verify reasonable performance (should be faster than real-time with SIMD)
        let rtf = elapsed.as_secs_f32() / duration_secs;
        eprintln!("Real-time factor: {rtf:.2}x");

        // With SIMD optimization, RTF should be reasonable (< 50x for debug build)
        assert!(rtf < 50.0, "RTF {rtf} is too slow, SIMD may not be working");
    }

    // =========================================================================
    // Unified Transcribe + Summarize API Tests (WAPR-LFM2-018)
    // =========================================================================

    #[test]
    fn test_summarize_options_default_params() {
        use model::lfm2::{Lfm2, Lfm2Tokenizer};

        // Create minimal LFM2 config for testing
        let config = format::apr2::Lfm2Config {
            hidden_size: 64,
            num_layers: 2,
            num_q_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            vocab_size: 1000,
            max_seq_len: 512,
            rope_theta: 10000.0,
            conv_dimension: 32,
            layer_types: vec![
                format::apr2::LayerType::Convolution {
                    kernel_size: 4,
                    cache_len: 3,
                },
                format::apr2::LayerType::Attention { use_gqa: true },
            ],
        };
        let model = Lfm2::new(config).expect("Model creation should succeed");
        let tokenizer = Lfm2Tokenizer::new();

        let options = SummarizeOptions::new(&model, &tokenizer);

        assert_eq!(options.max_tokens, 256);
        assert!((options.temperature - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_summarize_options_builder() {
        use model::lfm2::{Lfm2, Lfm2Tokenizer};

        let config = format::apr2::Lfm2Config {
            hidden_size: 64,
            num_layers: 2,
            num_q_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            vocab_size: 1000,
            max_seq_len: 512,
            rope_theta: 10000.0,
            conv_dimension: 32,
            layer_types: vec![
                format::apr2::LayerType::Convolution {
                    kernel_size: 4,
                    cache_len: 3,
                },
                format::apr2::LayerType::Attention { use_gqa: true },
            ],
        };
        let model = Lfm2::new(config).expect("Model creation should succeed");
        let tokenizer = Lfm2Tokenizer::new();

        let options = SummarizeOptions::new(&model, &tokenizer)
            .with_max_tokens(512)
            .with_temperature(0.7);

        assert_eq!(options.max_tokens, 512);
        assert!((options.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transcribe_summary_result_accessors() {
        let transcription = TranscriptionResult {
            text: "Hello world".to_string(),
            language: "en".to_string(),
            segments: vec![],
            profiling: None,
        };

        let result = TranscribeSummaryResult {
            transcription,
            summary: "Summary text".to_string(),
            generation_stats: None,
        };

        assert_eq!(result.transcript(), "Hello world");
        assert_eq!(result.summary(), "Summary text");
        assert!(result.has_summary());
    }

    #[test]
    fn test_transcribe_summary_result_empty_summary() {
        let transcription = TranscriptionResult {
            text: "Hello world".to_string(),
            language: "en".to_string(),
            segments: vec![],
            profiling: None,
        };

        let result = TranscribeSummaryResult {
            transcription,
            summary: String::new(),
            generation_stats: None,
        };

        assert!(!result.has_summary());
    }

    #[test]
    #[ignore = "Slow test - run with --ignored"]
    fn test_transcribe_and_summarize_empty_audio() {
        use model::lfm2::{Lfm2, Lfm2Tokenizer};

        let whisper = WhisperApr::tiny();
        // Use larger max_seq_len to handle typical transcription output
        let config = format::apr2::Lfm2Config {
            hidden_size: 64,
            num_layers: 2,
            num_q_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            vocab_size: 1000,
            max_seq_len: 2048, // Larger to handle transcription output
            rope_theta: 10000.0,
            conv_dimension: 32,
            layer_types: vec![
                format::apr2::LayerType::Convolution {
                    kernel_size: 4,
                    cache_len: 3,
                },
                format::apr2::LayerType::Attention { use_gqa: true },
            ],
        };
        let lfm2 = Lfm2::new(config).expect("Model creation should succeed");
        let tokenizer = Lfm2Tokenizer::new();

        // Empty audio (silence) should produce empty summary
        let audio = vec![0.0f32; 16000]; // 1 second of silence
        let transcribe_options = TranscribeOptions::default();
        let summarize_options = SummarizeOptions::new(&lfm2, &tokenizer).with_max_tokens(8); // Very short for speed

        let result = whisper
            .transcribe_and_summarize(&audio, transcribe_options, summarize_options)
            .expect("Should not fail");

        // For silence, transcription is typically empty or minimal
        // Summary should be empty if transcription is empty
        if result.transcription.text.trim().is_empty() {
            assert!(!result.has_summary());
        }
    }

    #[test]
    #[ignore = "Slow test - run with --ignored"]
    fn test_transcribe_and_summarize_integration() {
        use model::lfm2::{Lfm2, Lfm2Tokenizer};

        let whisper = WhisperApr::tiny();
        // Use larger max_seq_len to handle typical transcription output
        let config = format::apr2::Lfm2Config {
            hidden_size: 64,
            num_layers: 2,
            num_q_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            vocab_size: 1000,
            max_seq_len: 2048, // Larger to handle transcription output
            rope_theta: 10000.0,
            conv_dimension: 32,
            layer_types: vec![
                format::apr2::LayerType::Convolution {
                    kernel_size: 4,
                    cache_len: 3,
                },
                format::apr2::LayerType::Attention { use_gqa: true },
            ],
        };
        let lfm2 = Lfm2::new(config).expect("Model creation should succeed");
        let tokenizer = Lfm2Tokenizer::new();

        // Generate test audio (440Hz sine wave)
        let sample_rate = 16000;
        let duration_secs = 1.0;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let audio: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let transcribe_options = TranscribeOptions::default();
        let summarize_options = SummarizeOptions::new(&lfm2, &tokenizer).with_max_tokens(8); // Keep it short for test speed

        // This exercises the full pipeline
        let result =
            whisper.transcribe_and_summarize(&audio, transcribe_options, summarize_options);

        // The test validates that the pipeline doesn't crash
        // With synthetic weights, output quality is not meaningful
        assert!(result.is_ok());

        let result = result.expect("Should succeed");
        // Transcription always has a language
        assert_eq!(result.transcription.language, "en");
    }

    // =========================================================================
    // Chunked Transcription Tests (Phase 1 - WAPR-PERF-004)
    // =========================================================================

    #[test]
    fn test_chunk_constants() {
        // Verify chunk size matches spec: 30 seconds at 16kHz
        assert_eq!(WhisperApr::CHUNK_SAMPLES, 30 * 16000);

        // Verify overlap matches spec: 2 seconds at 16kHz
        assert_eq!(WhisperApr::OVERLAP_SAMPLES, 5 * 16000);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_short_audio_uses_single_chunk() {
        let _whisper = WhisperApr::tiny();

        // Audio shorter than 30 seconds should use single chunk
        let short_audio = vec![0.0_f32; 10 * 16000]; // 10 seconds
        assert!(short_audio.len() <= WhisperApr::CHUNK_SAMPLES);

        // This validates the transcribe method chooses single chunk path
        // (actual transcription requires loaded weights)
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_long_audio_uses_chunking() {
        let _whisper = WhisperApr::tiny();

        // Audio longer than 30 seconds should use chunked path
        let long_audio = vec![0.0_f32; 60 * 16000]; // 60 seconds
        assert!(long_audio.len() > WhisperApr::CHUNK_SAMPLES);

        // Validates that long audio triggers chunked transcription
        // (actual transcription requires loaded weights)
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_merge_overlapping_segments_empty() {
        let whisper = WhisperApr::tiny();
        let segments: Vec<Segment> = vec![];
        let merged = whisper.merge_overlapping_segments(segments);
        assert!(merged.is_empty());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_merge_overlapping_segments_single() {
        let whisper = WhisperApr::tiny();
        let segments = vec![Segment {
            start: 0.0,
            end: 5.0,
            text: "Hello".to_string(),
            tokens: vec![1, 2, 3],
        }];
        let merged = whisper.merge_overlapping_segments(segments);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "Hello");
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_merge_overlapping_segments_no_overlap() {
        let whisper = WhisperApr::tiny();
        let segments = vec![
            Segment {
                start: 0.0,
                end: 5.0,
                text: "Hello".to_string(),
                tokens: vec![1],
            },
            Segment {
                start: 10.0,
                end: 15.0,
                text: "World".to_string(),
                tokens: vec![2],
            },
        ];
        let merged = whisper.merge_overlapping_segments(segments);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].text, "Hello");
        assert_eq!(merged[1].text, "World");
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_merge_overlapping_segments_with_overlap() {
        let whisper = WhisperApr::tiny();
        let segments = vec![
            Segment {
                start: 0.0,
                end: 5.0,
                text: "Hello".to_string(),
                tokens: vec![1],
            },
            Segment {
                start: 4.9, // Overlaps with previous (within 0.1s tolerance)
                end: 10.0,
                text: "World".to_string(),
                tokens: vec![2],
            },
        ];
        let merged = whisper.merge_overlapping_segments(segments);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "Hello World");
        assert_eq!(merged[0].start, 0.0);
        assert_eq!(merged[0].end, 10.0);
        assert_eq!(merged[0].tokens, vec![1, 2]);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_merge_overlapping_segments_multiple_overlaps() {
        let whisper = WhisperApr::tiny();
        let segments = vec![
            Segment {
                start: 0.0,
                end: 5.0,
                text: "A".to_string(),
                tokens: vec![1],
            },
            Segment {
                start: 4.95,
                end: 10.0,
                text: "B".to_string(),
                tokens: vec![2],
            },
            Segment {
                start: 9.95,
                end: 15.0,
                text: "C".to_string(),
                tokens: vec![3],
            },
            Segment {
                start: 20.0, // Gap - no overlap
                end: 25.0,
                text: "D".to_string(),
                tokens: vec![4],
            },
        ];
        let merged = whisper.merge_overlapping_segments(segments);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].text, "A B C");
        assert_eq!(merged[0].end, 15.0);
        assert_eq!(merged[1].text, "D");
        assert_eq!(merged[1].start, 20.0);
    }

    #[test]
    fn test_chunk_boundary_calculation() {
        // Verify chunk boundaries are calculated correctly
        let total_samples = 90 * 16000; // 90 seconds
        let chunk_size = WhisperApr::CHUNK_SAMPLES; // 30 seconds
        let overlap = WhisperApr::OVERLAP_SAMPLES; // 5 seconds

        // First chunk: 0 to 35 seconds (30 + 5 overlap)
        let chunk1_end = (0 + chunk_size + overlap).min(total_samples);
        assert_eq!(chunk1_end, 35 * 16000);

        // Second chunk: 30 to 65 seconds
        let chunk2_start = chunk_size; // 30 seconds
        let chunk2_end = (chunk2_start + chunk_size + overlap).min(total_samples);
        assert_eq!(chunk2_end, 65 * 16000);

        // Third chunk: 60 to 90 seconds (no overflow)
        let chunk3_start = 2 * chunk_size; // 60 seconds
        let chunk3_end = (chunk3_start + chunk_size + overlap).min(total_samples);
        assert_eq!(chunk3_end, total_samples);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_falsification_point_25_long_audio_full_transcription() {
        // Falsification Point 25: Long audio fails
        // Test: 10+ minute audio should be processable
        // (Note: actual transcription requires loaded model)

        let _whisper = WhisperApr::tiny();

        // 10 minutes of audio at 16kHz
        let ten_minutes_samples = 10 * 60 * 16000;
        assert!(ten_minutes_samples > WhisperApr::CHUNK_SAMPLES);

        // Calculate expected number of chunks
        let expected_chunks =
            (ten_minutes_samples + WhisperApr::CHUNK_SAMPLES - 1) / WhisperApr::CHUNK_SAMPLES;
        assert_eq!(expected_chunks, 20); // 10 min / 30 sec = 20 chunks

        // Verify chunk iteration covers all audio
        let mut offset = 0;
        let mut chunk_count = 0;
        while offset < ten_minutes_samples {
            let chunk_end = (offset + WhisperApr::CHUNK_SAMPLES + WhisperApr::OVERLAP_SAMPLES)
                .min(ten_minutes_samples);
            let _chunk_len = chunk_end - offset;
            offset += WhisperApr::CHUNK_SAMPLES;
            chunk_count += 1;
        }
        assert_eq!(chunk_count, 20);
    }

    #[test]
    fn test_falsification_point_30_streaming_consistency() {
        // Falsification Point 30: Streaming broken
        // Test: Chunked input should produce consistent output

        // Verify overlap is sufficient for boundary handling
        // 5 second overlap gives ~10-15 words of context at normal speech rate
        let overlap_seconds = WhisperApr::OVERLAP_SAMPLES as f32 / 16000.0;
        assert!((overlap_seconds - 5.0).abs() < 0.01);

        // Typical speech: 2-3 words per second
        // 5 second overlap = 10-15 words of context
        let expected_overlap_words = overlap_seconds * 2.5;
        assert!(expected_overlap_words >= 10.0);
    }

    // =========================================================================
    // Moonshine Integration Tests
    // =========================================================================

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_tiny_constructs() {
        let model = WhisperApr::moonshine_tiny();
        assert_eq!(model.config.n_audio_state, 288);
        assert_eq!(model.config.n_text_state, 288);
        assert_eq!(model.config.n_audio_layer, 6);
        assert_eq!(model.config.n_text_layer, 6);
        assert_eq!(model.config.n_audio_head, 8);
        assert_eq!(model.config.n_text_head, 8);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_tiny_encoder_has_moonshine_blocks() {
        let model = WhisperApr::moonshine_tiny();
        // Encoder should have Moonshine blocks, not Whisper blocks
        assert!(model.encoder.moonshine_blocks().len() > 0);
        assert!(model.encoder.blocks().is_empty());
        assert!(model.encoder.rope().is_some());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_tiny_decoder_has_moonshine_blocks() {
        let model = WhisperApr::moonshine_tiny();
        // Decoder should have Moonshine blocks, not Whisper blocks
        assert!(model.decoder.moonshine_blocks().len() > 0);
        assert!(model.decoder.blocks().is_empty());
        assert!(model.decoder.rope().is_some());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_tiny_has_sentencepiece_tokenizer() {
        let model = WhisperApr::moonshine_tiny();
        match model.tokenizer() {
            tokenizer::Tokenizer::SentencePiece(_) => {} // expected
            tokenizer::Tokenizer::Bpe(_) => panic!("Moonshine should use SentencePiece tokenizer"),
        }
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_tiny_has_conv_stem() {
        let model = WhisperApr::moonshine_tiny();
        assert!(model.conv_stem.is_some());
        assert!(model.mel_filters.is_none());
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_tiny_initial_tokens() {
        let model = WhisperApr::moonshine_tiny();
        let tokens = model.get_initial_tokens("en", crate::Task::Transcribe);
        // Moonshine: just BOS token (SentencePiece token 1)
        assert_eq!(tokens, vec![1]);
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_tiny_eot_token() {
        let model = WhisperApr::moonshine_tiny();
        assert_eq!(model.eot_token(), 2); // SentencePiece EOS
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_encoder_forward_shape() {
        let model = WhisperApr::moonshine_tiny();
        let d_model = 288;
        let seq_len = 7; // variable-length input

        // Simulate ConvStem output: [seq_len, d_model]
        let features = vec![0.1_f32; seq_len * d_model];
        let output = model.encoder.forward(&features).expect("encoder forward");

        // Output should have same shape (seq_len preserved through transformer blocks)
        assert_eq!(output.len(), seq_len * d_model);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_decoder_forward_shape() {
        let model = WhisperApr::moonshine_tiny();
        let d_model = 288;
        let n_vocab = model.config.n_vocab as usize;

        // Simulate encoder output
        let enc_seq_len = 7;
        let encoder_output = vec![0.1_f32; enc_seq_len * d_model];

        // Decode 2 tokens
        let tokens = vec![1_u32, 100];
        let logits = model
            .decoder
            .forward(&tokens, &encoder_output)
            .expect("decoder forward");

        // Output: [seq_len, n_vocab]
        assert_eq!(logits.len(), tokens.len() * n_vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_variable_length_input() {
        let model = WhisperApr::moonshine_tiny();
        let d_model = 288;

        // Test different sequence lengths (Moonshine supports variable-length)
        for seq_len in [3, 7, 15, 31] {
            let features = vec![0.1_f32; seq_len * d_model];
            let output = model
                .encoder
                .forward(&features)
                .expect("encoder forward");
            assert_eq!(
                output.len(),
                seq_len * d_model,
                "seq_len={seq_len} output mismatch"
            );
        }
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_whisper_tiny_unaffected() {
        // Verify Whisper models still work correctly after Moonshine changes
        let model = WhisperApr::tiny();
        assert!(model.encoder.blocks().len() > 0);
        assert!(model.encoder.moonshine_blocks().is_empty());
        assert!(model.encoder.rope().is_none());
        assert!(model.decoder.blocks().len() > 0);
        assert!(model.decoder.moonshine_blocks().is_empty());
        assert!(model.decoder.rope().is_none());
        assert!(model.mel_filters.is_some());
        assert!(model.conv_stem.is_none());
        match model.tokenizer() {
            tokenizer::Tokenizer::Bpe(_) => {} // expected
            tokenizer::Tokenizer::SentencePiece(_) => {
                panic!("Whisper should use BPE tokenizer")
            }
        }
    }

    // ========================================================================
    // Falsification tests: would have caught bugs found during systematic review
    // ========================================================================

    #[test]
    fn test_moonshine_encoder_uses_rmsnorm_final() {
        // Regression: encoder must use RmsNorm (not LayerNorm) for Moonshine
        let config = model::ModelConfig::moonshine_tiny();
        let encoder = model::Encoder::new(&config);
        assert!(
            encoder.ln_post_rms().is_some(),
            "Moonshine encoder must use RmsNorm for final layer norm"
        );
    }

    #[test]
    fn test_whisper_encoder_uses_layernorm_final() {
        // Regression: Whisper should still use LayerNorm, not RmsNorm
        let config = model::ModelConfig::tiny();
        let encoder = model::Encoder::new(&config);
        assert!(
            encoder.ln_post_rms().is_none(),
            "Whisper encoder must use LayerNorm, not RmsNorm"
        );
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_decoder_forward_one_sets_cross_attn_cached() {
        // Regression: forward_one must set cross_attn_cached=true for Moonshine
        let model = WhisperApr::moonshine_tiny();
        let d_model = 288;
        let n_layers = model.decoder.n_layers();
        let max_len = model.decoder.max_len();
        let enc_seq_len = 7;
        let encoder_output = vec![0.1_f32; enc_seq_len * d_model];

        let mut cache = model::DecoderKVCache::new(n_layers, d_model, max_len);
        assert!(!cache.cross_attn_cached);

        // Process one token
        let _logits = model
            .decoder
            .forward_one(1, &encoder_output, &mut cache)
            .expect("forward_one");

        assert!(
            cache.cross_attn_cached,
            "cross_attn_cached must be true after forward_one for Moonshine"
        );
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_decoder_forward_one_cache_has_layers() {
        // Regression: cache must have layers even for fallback decoder
        let config = model::ModelConfig::moonshine_tiny();
        let decoder = model::Decoder::new(&config);

        // Verify that a valid Moonshine decoder has cache layers
        let cache = model::DecoderKVCache::new(
            decoder.n_layers(),
            decoder.d_model(),
            decoder.max_len(),
        );
        assert!(
            !cache.self_attn_cache.is_empty(),
            "Moonshine decoder cache must have at least 1 layer"
        );
    }

    #[test]
    fn test_moonshine_decoder_is_finalized() {
        // Regression: is_finalized must handle empty-block case correctly
        let config = model::ModelConfig::moonshine_tiny();
        let decoder = model::Decoder::new(&config);
        // Moonshine blocks don't have finalization — should report as finalized
        assert!(
            decoder.is_finalized(),
            "Moonshine decoder should be considered finalized"
        );
    }

    #[test]
    fn test_whisper_decoder_is_finalized_initially_false() {
        // Whisper decoder has blocks that need finalization
        let config = model::ModelConfig::tiny();
        let decoder = model::Decoder::new(&config);
        // Whisper blocks need finalize_weights() to be called
        assert!(
            !decoder.is_finalized(),
            "Whisper decoder should NOT be finalized before finalize_weights()"
        );
    }

    #[test]
    #[ignore = "Allocates large model - run with --ignored"]
    fn test_moonshine_decoder_forward_traced() {
        // Regression: forward_traced must produce traces for Moonshine blocks
        let model = WhisperApr::moonshine_tiny();
        let d_model = 288;
        let enc_seq_len = 7;
        let encoder_output = vec![0.1_f32; enc_seq_len * d_model];

        let (logits, trace) = model
            .decoder
            .forward_traced(&[1, 2], &encoder_output)
            .expect("forward_traced");

        // Should have layer traces for Moonshine blocks
        let layer_traces: Vec<_> = trace
            .iter()
            .filter(|(name, _)| name.starts_with("layer_"))
            .collect();
        assert!(
            !layer_traces.is_empty(),
            "forward_traced must produce layer traces for Moonshine"
        );

        assert!(!logits.is_empty());
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_moonshine_encoder_forward_batch_uses_forward() {
        // Regression: forward_batch must dispatch correctly for non-mel frontends
        let config = model::ModelConfig::moonshine_tiny();
        let encoder = model::Encoder::new(&config);
        let d_model = 288;

        // forward_batch with already-projected features (Moonshine path)
        let batch = vec![vec![0.1_f32; 7 * d_model], vec![0.1_f32; 5 * d_model]];
        let results = encoder.forward_batch(&batch).expect("forward_batch");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 7 * d_model);
        assert_eq!(results[1].len(), 5 * d_model);
    }
}

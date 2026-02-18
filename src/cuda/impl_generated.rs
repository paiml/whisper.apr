//! CUDA GPU acceleration for Whisper inference.
//!
//! Provides GPU-resident model execution via trueno-gpu PTX kernels
//! through realizar's `CudaExecutor`.
//!
//! # Architecture
//!
//! The GPU path pre-uploads model weights to VRAM once at initialization,
//! then runs encoder/decoder forward passes entirely on GPU with minimal
//! host synchronization.
//!
//! # APR-Style Tracing (WAPR-PERF-004)
//!
//! Integrates with realizar's `InferenceTracer` for step-level performance
//! visibility per AWS Step Functions event model. Each step (ENCODE, EMBED,
//! TRANSFORMER_BLOCK, LM_HEAD, SAMPLE, DECODE) emits TaskStateEntered/Exited
//! events with TensorStats for anomaly detection (Jidoka).
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::cuda::WhisperCuda;
//! use realizar::inference_trace::{TraceConfig, InferenceTracer};
//!
//! // Create GPU-accelerated model with tracing
//! let model = WhisperModel::load("tiny.apr")?;
//! let mut cuda_model = WhisperCuda::new(model, 0)?;
//! cuda_model.enable_tracing(TraceConfig::enabled());
//!
//! // Run inference on GPU - trace events collected automatically
//! let result = cuda_model.transcribe_gpu(&audio, options)?;
//!
//! // Analyze trace for bottlenecks
//! for event in cuda_model.tracer().events() {
//!     println!("{}: {:?} took {}µs", event.step.name(), event.stats, event.duration_us);
//! }
//! ```

use crate::audio::{MelConfig, MelFilterbank, SAMPLE_RATE};
use crate::error::{WhisperError, WhisperResult};
use crate::model::{Decoder, DecoderKVCache, Encoder, ModelConfig};
use crate::tokenizer::BpeTokenizer;
use crate::{DecodingStrategy, Task, TranscribeOptions, TranscriptionResult};
use realizar::cuda::CudaExecutor;
use realizar::inference_trace::{InferenceTracer, ModelInfo, TraceConfig, TraceStep};

// GPU-Resident Tensor imports (WAPR-PERF-004)
#[cfg(feature = "cuda")]
#[allow(unused_imports)] // total_d2h_transfers, total_h2d_transfers used only in tests
use trueno_gpu::memory::resident::{
    batched_multihead_attention,
    forward_encoder_block_gpu,
    incremental_attention_gpu,
    incremental_attention_gpu_with_stream, // WAPR-PERF-014: shared stream variant
    kernel_cache_hits,
    kernel_cache_misses,
    kv_cache_scatter_gpu,
    reset_transfer_counters,
    total_d2h_transfers,
    total_h2d_transfers,
    GpuConvFrontendWeights,
    GpuDecoderBlockWeights,
    GpuDecoderConfig,
    GpuEncoderBlockWeights,
    GpuEncoderConfig,
    GpuKvCache,
    GpuResidentTensor,
    TransferStats,
};

/// GELU activation function (Gaussian Error Linear Unit)
#[inline]
fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// GPU-accelerated Whisper model.
///
/// Wraps encoder and decoder with CUDA execution capability.
/// Model weights are pre-uploaded to GPU memory for minimal latency.
///
/// # APR-Style Tracing (WAPR-PERF-004)
///
/// Integrates with realizar's `InferenceTracer` for performance visibility:
/// - Enable tracing: `cuda_model.enable_tracing(TraceConfig::enabled())`
/// - Access events: `cuda_model.tracer().events()`
/// - Print summary: `cuda_model.tracer().print_summary()`
pub struct WhisperCuda {
    /// Whisper encoder (audio → hidden states)
    encoder: Encoder,
    /// Whisper decoder (hidden states → tokens)
    decoder: Decoder,
    /// CUDA executor for GPU kernel dispatch
    executor: CudaExecutor,
    /// Model configuration
    config: ModelConfig,
    /// BPE tokenizer
    tokenizer: BpeTokenizer,
    /// Mel filterbank
    mel_filters: MelFilterbank,
    /// GPU device name (e.g., "NVIDIA GeForce RTX 4090")
    device_name: String,
    /// GPU memory info (free_bytes, total_bytes)
    memory_info: (usize, usize),
    /// Whether GPU weights have been uploaded
    weights_uploaded: bool,
    /// Whether GPU KV caches are initialized
    kv_cache_initialized: bool,
    /// Inference tracer for APR-style step-level visibility (realizar::InferenceTracer)
    tracer: InferenceTracer,
    /// GPU-resident encoder block weights (WAPR-PERF-004: Total Offload)
    #[cfg(feature = "cuda")]
    gpu_encoder_weights: Option<Vec<GpuEncoderBlockWeights>>,
    /// GPU encoder configuration
    #[cfg(feature = "cuda")]
    gpu_encoder_config: Option<GpuEncoderConfig>,
    /// WAPR-PERF-012: GPU-resident conv frontend weights
    #[cfg(feature = "cuda")]
    gpu_conv_weights: Option<GpuConvFrontendWeights>,
    /// WAPR-PERF-013: GPU-resident decoder block weights
    #[cfg(feature = "cuda")]
    gpu_decoder_weights: Option<Vec<GpuDecoderBlockWeights>>,
    /// WAPR-PERF-013: GPU decoder configuration
    #[cfg(feature = "cuda")]
    gpu_decoder_config: Option<GpuDecoderConfig>,
    /// WAPR-PERF-013: GPU-resident KV caches (self-attention per layer)
    #[cfg(feature = "cuda")]
    gpu_self_kv_cache: Option<Vec<GpuKvCache>>,
    /// WAPR-PERF-013: GPU-resident cross-attention KV cache (encoder K/V, per layer)
    #[cfg(feature = "cuda")]
    gpu_cross_kv_cache: Option<Vec<GpuKvCache>>,
    /// WAPR-PERF-013: Head-first self-attention K cache [n_heads, max_seq_len, head_dim]
    #[cfg(feature = "cuda")]
    gpu_self_k_head_first: Option<Vec<GpuResidentTensor<f32>>>,
    /// WAPR-PERF-013: Head-first self-attention V cache [n_heads, max_seq_len, head_dim]
    #[cfg(feature = "cuda")]
    gpu_self_v_head_first: Option<Vec<GpuResidentTensor<f32>>>,
    /// WAPR-PERF-013: Head-first cross-attention K cache [n_heads, enc_seq_len, head_dim]
    #[cfg(feature = "cuda")]
    gpu_cross_k_head_first: Option<Vec<GpuResidentTensor<f32>>>,
    /// WAPR-PERF-013: Head-first cross-attention V cache [n_heads, enc_seq_len, head_dim]
    #[cfg(feature = "cuda")]
    gpu_cross_v_head_first: Option<Vec<GpuResidentTensor<f32>>>,
    /// WAPR-PERF-013: Current sequence position for decoder
    #[cfg(feature = "cuda")]
    gpu_decoder_pos: usize,
    /// WAPR-PERF-019: GPU-resident encoder post-norm gamma (final layer norm)
    #[cfg(feature = "cuda")]
    gpu_enc_ln_post_gamma: Option<GpuResidentTensor<f32>>,
    /// WAPR-PERF-019: GPU-resident encoder post-norm beta (final layer norm)
    #[cfg(feature = "cuda")]
    gpu_enc_ln_post_beta: Option<GpuResidentTensor<f32>>,
}

impl WhisperCuda {
    /// Create a new CUDA-accelerated Whisper model.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Pre-loaded Whisper encoder
    /// * `decoder` - Pre-loaded Whisper decoder
    /// * `config` - Model configuration
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn new(
        encoder: Encoder,
        decoder: Decoder,
        config: ModelConfig,
        device_ordinal: i32,
    ) -> WhisperResult<Self> {
        Self::new_with_tokenizer(
            encoder,
            decoder,
            config,
            BpeTokenizer::with_base_tokens(),
            device_ordinal,
        )
    }

    /// Create a new CUDA-accelerated Whisper model with a pre-loaded tokenizer.
    ///
    /// This is the preferred constructor when converting from WhisperApr, as it
    /// preserves the full vocabulary from the APR file.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Pre-loaded Whisper encoder
    /// * `decoder` - Pre-loaded Whisper decoder
    /// * `config` - Model configuration
    /// * `tokenizer` - Pre-loaded BPE tokenizer with full vocabulary
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn new_with_tokenizer(
        encoder: Encoder,
        decoder: Decoder,
        config: ModelConfig,
        tokenizer: BpeTokenizer,
        device_ordinal: i32,
    ) -> WhisperResult<Self> {
        let mel_filters = MelFilterbank::new(
            &MelConfig { n_mels: config.n_mels as usize, ..MelConfig::whisper() },
        );
        Self::new_with_components(
            encoder,
            decoder,
            config,
            tokenizer,
            mel_filters,
            device_ordinal,
        )
    }

    /// Create a new CUDA-accelerated Whisper model with all components.
    ///
    /// This is the preferred constructor when converting from WhisperApr, as it
    /// preserves all components including the mel filterbank loaded from APR.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Pre-loaded Whisper encoder
    /// * `decoder` - Pre-loaded Whisper decoder
    /// * `config` - Model configuration
    /// * `tokenizer` - Pre-loaded BPE tokenizer with full vocabulary
    /// * `mel_filters` - Pre-loaded mel filterbank
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn new_with_components(
        encoder: Encoder,
        decoder: Decoder,
        config: ModelConfig,
        tokenizer: BpeTokenizer,
        mel_filters: MelFilterbank,
        device_ordinal: i32,
    ) -> WhisperResult<Self> {
        if !CudaExecutor::is_available() {
            return Err(WhisperError::Inference(
                "CUDA not available. Install CUDA drivers or use CPU backend.".into(),
            ));
        }

        let executor = CudaExecutor::new(device_ordinal)
            .map_err(|e| WhisperError::Inference(format!("CUDA initialization failed: {e}")))?;

        let device_name = executor
            .device_name()
            .unwrap_or_else(|_| "Unknown GPU".into());

        let memory_info = executor.memory_info().unwrap_or((0, 0));

        // Initialize tracer with model info (disabled by default for zero overhead)
        let mut tracer = InferenceTracer::disabled();
        tracer.set_model_info(ModelInfo {
            name: device_name.clone(),
            num_layers: config.n_text_layer as usize,
            hidden_dim: config.n_text_state as usize,
            vocab_size: config.n_vocab as usize,
            num_heads: config.n_text_head as usize,
            quant_type: None, // f32 for now
        });

        let mut model = Self {
            encoder,
            decoder,
            executor,
            config,
            tokenizer,
            mel_filters,
            device_name,
            memory_info,
            weights_uploaded: false,
            kv_cache_initialized: false,
            tracer,
            #[cfg(feature = "cuda")]
            gpu_encoder_weights: None,
            #[cfg(feature = "cuda")]
            gpu_encoder_config: None,
            #[cfg(feature = "cuda")]
            gpu_conv_weights: None,
            #[cfg(feature = "cuda")]
            gpu_decoder_weights: None,
            #[cfg(feature = "cuda")]
            gpu_decoder_config: None,
            #[cfg(feature = "cuda")]
            gpu_self_kv_cache: None,
            #[cfg(feature = "cuda")]
            gpu_cross_kv_cache: None,
            #[cfg(feature = "cuda")]
            gpu_self_k_head_first: None,
            #[cfg(feature = "cuda")]
            gpu_self_v_head_first: None,
            #[cfg(feature = "cuda")]
            gpu_cross_k_head_first: None,
            #[cfg(feature = "cuda")]
            gpu_cross_v_head_first: None,
            #[cfg(feature = "cuda")]
            gpu_decoder_pos: 0,
            #[cfg(feature = "cuda")]
            gpu_enc_ln_post_gamma: None,
            #[cfg(feature = "cuda")]
            gpu_enc_ln_post_beta: None,
        };

        // Initialize GPU KV caches for decoder self-attention
        model.init_gpu_kv_cache()?;

        Ok(model)
    }

    /// Enable APR-style inference tracing (realizar::InferenceTracer).
    ///
    /// When enabled, trace events are collected for each inference step:
    /// - ENCODE: Audio preprocessing and encoder forward pass
    /// - EMBED: Token embedding lookup
    /// - TRANSFORMER_BLOCK: Each decoder layer (×n_layers per token)
    /// - LM_HEAD: Output projection to vocabulary
    /// - SAMPLE: Token sampling (argmax/beam/top-k)
    /// - DECODE: Token detokenization
    ///
    /// # Performance Note
    ///
    /// Tracing adds ~1-5% overhead when enabled. Use `TraceConfig::enabled()`
    /// for full visibility or configure specific steps to trace.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use realizar::inference_trace::TraceConfig;
    /// cuda_model.enable_tracing(TraceConfig::enabled());
    /// ```
    pub fn enable_tracing(&mut self, config: TraceConfig) {
        self.tracer = InferenceTracer::new(config);
        self.tracer.set_model_info(ModelInfo {
            name: self.device_name.clone(),
            num_layers: self.config.n_text_layer as usize,
            hidden_dim: self.config.n_text_state as usize,
            vocab_size: self.config.n_vocab as usize,
            num_heads: self.config.n_text_head as usize,
            quant_type: None,
        });
    }

    /// Get reference to the inference tracer.
    ///
    /// Use this to access collected trace events for analysis.
    pub fn tracer(&self) -> &InferenceTracer {
        &self.tracer
    }

    /// Get mutable reference to the inference tracer.
    pub fn tracer_mut(&mut self) -> &mut InferenceTracer {
        &mut self.tracer
    }

    /// Reset tracer for a new inference run.
    pub fn reset_tracer(&mut self) {
        let config = if self.tracer.is_enabled() {
            TraceConfig::enabled()
        } else {
            TraceConfig::default()
        };
        self.tracer = InferenceTracer::new(config);
        self.tracer.set_model_info(ModelInfo {
            name: self.device_name.clone(),
            num_layers: self.config.n_text_layer as usize,
            hidden_dim: self.config.n_text_state as usize,
            vocab_size: self.config.n_vocab as usize,
            num_heads: self.config.n_text_head as usize,
            quant_type: None,
        });
    }

    /// Get GPU device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get GPU memory info (free_bytes, total_bytes).
    pub fn memory_info(&self) -> (usize, usize) {
        self.memory_info
    }

    /// Check if model weights are uploaded to GPU.
    pub fn weights_uploaded(&self) -> bool {
        self.weights_uploaded
    }

    /// Get model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Upload model weights to GPU memory.
    ///
    /// This is called automatically on first inference if not done explicitly.
    /// Pre-uploading weights avoids latency on first transcription.
    ///
    /// Uploads:
    /// - Output projection (token embedding): 51865 × 384 ≈ 80MB
    /// - All decoder block weights (attention + FFN): ~90MB for tiny
    ///
    /// # Returns
    ///
    /// Number of bytes uploaded to GPU.
    pub fn upload_weights(&mut self) -> WhisperResult<usize> {
        if self.weights_uploaded {
            return Ok(0);
        }

        let mut total_bytes = 0_usize;

        // Upload the most expensive weight: token embedding for output projection
        // This is [n_vocab × d_model] = [51865 × 384] ≈ 80MB for tiny model
        // WAPR-PERF-014 FIX: GEMV kernel expects [K × N] but token_emb is [N × K]
        // Must transpose from [n_vocab × d_model] to [d_model × n_vocab]
        let token_emb = self.decoder.token_embedding();
        let n_vocab = self.config.n_vocab as usize;
        let d_model = self.config.n_text_state as usize;
        let mut token_emb_transposed = vec![0.0f32; n_vocab * d_model];
        for row in 0..n_vocab {
            for col in 0..d_model {
                // Source: [row, col] = row * d_model + col
                // Dest: [col, row] = col * n_vocab + row
                token_emb_transposed[col * n_vocab + row] = token_emb[row * d_model + col];
            }
        }
        let bytes = self
            .executor
            .load_weights("whisper_output_proj", &token_emb_transposed)
            .map_err(|e| {
                WhisperError::Inference(format!("Failed to upload output projection: {e}"))
            })?;
        total_bytes += bytes;

        // Upload all decoder block weights for full GPU acceleration
        // Each block has: self_attn (Q,K,V,O), cross_attn (Q,K,V,O), ffn (fc1, fc2)
        for (block_idx, block) in self.decoder.blocks().iter().enumerate() {
            // Self-attention weights
            let name = format!("dec_b{block_idx}_self_q");
            let bytes = self
                .executor
                .load_weights(&name, &block.self_attn.w_q().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_self_k");
            let bytes = self
                .executor
                .load_weights(&name, &block.self_attn.w_k().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_self_v");
            let bytes = self
                .executor
                .load_weights(&name, &block.self_attn.w_v().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_self_o");
            let bytes = self
                .executor
                .load_weights(&name, &block.self_attn.w_o().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            // Cross-attention weights
            let name = format!("dec_b{block_idx}_cross_q");
            let bytes = self
                .executor
                .load_weights(&name, &block.cross_attn.w_q().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_cross_k");
            let bytes = self
                .executor
                .load_weights(&name, &block.cross_attn.w_k().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_cross_v");
            let bytes = self
                .executor
                .load_weights(&name, &block.cross_attn.w_v().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_cross_o");
            let bytes = self
                .executor
                .load_weights(&name, &block.cross_attn.w_o().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            // FFN weights
            let name = format!("dec_b{block_idx}_ffn_fc1");
            let bytes = self
                .executor
                .load_weights(&name, &block.ffn.fc1.weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_ffn_fc2");
            let bytes = self
                .executor
                .load_weights(&name, &block.ffn.fc2.weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;
        }

        self.weights_uploaded = true;
        Ok(total_bytes)
    }

    /// Upload encoder weights to GPU memory.
    ///
    /// # WAPR-PERF-005: GPU Encoder
    ///
    /// Uploads encoder weights for GPU-accelerated encoding:
    /// - Conv1/Conv2 frontend weights
    /// - Encoder block attention weights (Q, K, V, O per layer)
    /// - Encoder block FFN weights (fc1, fc2 per layer)
    pub fn upload_encoder_weights(&mut self) -> WhisperResult<usize> {
        let mut total_bytes = 0_usize;

        // Upload encoder block weights
        for (block_idx, block) in self.encoder.blocks().iter().enumerate() {
            // Self-attention weights
            let name = format!("enc_b{block_idx}_self_q");
            let bytes = self
                .executor
                .load_weights(&name, &block.self_attn.w_q().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("enc_b{block_idx}_self_k");
            let bytes = self
                .executor
                .load_weights(&name, &block.self_attn.w_k().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("enc_b{block_idx}_self_v");
            let bytes = self
                .executor
                .load_weights(&name, &block.self_attn.w_v().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("enc_b{block_idx}_self_o");
            let bytes = self
                .executor
                .load_weights(&name, &block.self_attn.w_o().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            // FFN weights
            let name = format!("enc_b{block_idx}_ffn_fc1");
            let bytes = self
                .executor
                .load_weights(&name, &block.ffn.fc1.weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("enc_b{block_idx}_ffn_fc2");
            let bytes = self
                .executor
                .load_weights(&name, &block.ffn.fc2.weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;
        }

        Ok(total_bytes)
    }

    /// GPU-accelerated encoder forward pass.
    ///
    /// # WAPR-PERF-005: 20x Speedup Target
    ///
    /// Current: 6.15s on CPU (98.7% of total time)
    /// Target: <300ms on GPU (matching whisper.cpp)
    ///
    /// Uses `flash_attention_multi_head` for encoder self-attention.
    pub fn encode_gpu(&mut self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        let _n_mels = self.config.n_mels as usize;
        let d_model = self.config.n_audio_state as usize;
        let n_heads = self.config.n_audio_head as usize;
        let head_dim = d_model / n_heads;
        let n_layers = self.config.n_audio_layer as usize;

        // Step 1: Convolutional frontend (CPU - small compared to attention)
        let conv_frontend = self.encoder.conv_frontend().ok_or_else(|| {
            WhisperError::Inference("no conv frontend (Whisper models require it)".into())
        })?;
        let conv_output = conv_frontend.forward(mel)?;
        let seq_len = conv_output.len() / d_model;

        // Step 2: Add positional embedding
        let mut x = conv_output;
        let pos_emb = self.encoder.positional_embedding();
        for pos in 0..seq_len {
            for d in 0..d_model {
                x[pos * d_model + d] += pos_emb[pos * d_model + d];
            }
        }

        // Step 3: Process encoder blocks with GPU attention
        for layer_idx in 0..n_layers {
            x = self.forward_encoder_block_gpu(layer_idx, &x, seq_len, n_heads, head_dim)?;
        }

        // Step 4: Final layer norm (CPU)
        self.encoder.ln_post().forward(&x)
    }

    /// Upload all encoder weights to GPU for Total Offload (WAPR-PERF-004)
    ///
    /// Pre-uploads all encoder block weights once at initialization.
    /// Subsequent encode calls use GPU-resident weights with zero transfer overhead.
    ///
    /// IMPORTANT: Weight matrices are transposed before upload because:
    /// - CPU stores weights as [out_features, in_features] for y = x @ W.T + b
    /// - GPU linear expects [in_features, out_features] for y = x @ W + b
    #[cfg(feature = "cuda")]
    pub fn upload_encoder_weights_to_gpu(&mut self) -> WhisperResult<()> {
        if self.gpu_encoder_weights.is_some() {
            return Ok(()); // Already uploaded
        }

        // Helper to transpose weight matrix from [rows, cols] to [cols, rows]
        fn transpose_weights(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            let mut transposed = vec![0.0_f32; weights.len()];
            for r in 0..rows {
                for c in 0..cols {
                    transposed[c * rows + r] = weights[r * cols + c];
                }
            }
            transposed
        }

        let ctx = self.executor.context();
        let d_model = self.config.n_audio_state as usize;
        let n_heads = self.config.n_audio_head as usize;
        let n_layers = self.config.n_audio_layer as usize;
        let d_ff = d_model * 4; // Standard 4x expansion

        let mut gpu_weights = Vec::with_capacity(n_layers);

        for layer_idx in 0..n_layers {
            let block = &self.encoder.blocks()[layer_idx];

            // Upload LayerNorm 1 weights (no transpose needed for 1D vectors)
            let ln1_gamma = GpuResidentTensor::from_host(ctx, &block.ln1.weight)
                .map_err(|e| WhisperError::Inference(format!("ln1_gamma upload: {e}")))?;
            let ln1_beta = GpuResidentTensor::from_host(ctx, &block.ln1.bias)
                .map_err(|e| WhisperError::Inference(format!("ln1_beta upload: {e}")))?;

            // Upload Q/K/V/O projection weights (TRANSPOSED from [out, in] to [in, out])
            // CPU: [d_model, d_model] as [out_features, in_features]
            // GPU: [d_model, d_model] as [in_features, out_features]
            let w_q_t = transpose_weights(&block.self_attn.w_q().weight, d_model, d_model);
            let w_q = GpuResidentTensor::from_host(ctx, &w_q_t)
                .map_err(|e| WhisperError::Inference(format!("w_q upload: {e}")))?;
            let b_q = GpuResidentTensor::from_host(ctx, &block.self_attn.w_q().bias)
                .map_err(|e| WhisperError::Inference(format!("b_q upload: {e}")))?;

            let w_k_t = transpose_weights(&block.self_attn.w_k().weight, d_model, d_model);
            let w_k = GpuResidentTensor::from_host(ctx, &w_k_t)
                .map_err(|e| WhisperError::Inference(format!("w_k upload: {e}")))?;
            let b_k = GpuResidentTensor::from_host(ctx, &block.self_attn.w_k().bias)
                .map_err(|e| WhisperError::Inference(format!("b_k upload: {e}")))?;

            let w_v_t = transpose_weights(&block.self_attn.w_v().weight, d_model, d_model);
            let w_v = GpuResidentTensor::from_host(ctx, &w_v_t)
                .map_err(|e| WhisperError::Inference(format!("w_v upload: {e}")))?;
            let b_v = GpuResidentTensor::from_host(ctx, &block.self_attn.w_v().bias)
                .map_err(|e| WhisperError::Inference(format!("b_v upload: {e}")))?;

            let w_o_t = transpose_weights(&block.self_attn.w_o().weight, d_model, d_model);
            let w_o = GpuResidentTensor::from_host(ctx, &w_o_t)
                .map_err(|e| WhisperError::Inference(format!("w_o upload: {e}")))?;
            let b_o = GpuResidentTensor::from_host(ctx, &block.self_attn.w_o().bias)
                .map_err(|e| WhisperError::Inference(format!("b_o upload: {e}")))?;

            // Upload LayerNorm 2 weights (no transpose needed for 1D vectors)
            let ln2_gamma = GpuResidentTensor::from_host(ctx, &block.ln2.weight)
                .map_err(|e| WhisperError::Inference(format!("ln2_gamma upload: {e}")))?;
            let ln2_beta = GpuResidentTensor::from_host(ctx, &block.ln2.bias)
                .map_err(|e| WhisperError::Inference(format!("ln2_beta upload: {e}")))?;

            // Upload FFN weights (TRANSPOSED)
            // FFN up: [d_ff, d_model] -> transposed to [d_model, d_ff]
            let ffn_up_t = transpose_weights(&block.ffn.fc1.weight, d_ff, d_model);
            let ffn_up_w = GpuResidentTensor::from_host(ctx, &ffn_up_t)
                .map_err(|e| WhisperError::Inference(format!("ffn_up_w upload: {e}")))?;
            let ffn_up_b = GpuResidentTensor::from_host(ctx, &block.ffn.fc1.bias)
                .map_err(|e| WhisperError::Inference(format!("ffn_up_b upload: {e}")))?;

            // FFN down: [d_model, d_ff] -> transposed to [d_ff, d_model]
            let ffn_down_t = transpose_weights(&block.ffn.fc2.weight, d_model, d_ff);
            let ffn_down_w = GpuResidentTensor::from_host(ctx, &ffn_down_t)
                .map_err(|e| WhisperError::Inference(format!("ffn_down_w upload: {e}")))?;
            let ffn_down_b = GpuResidentTensor::from_host(ctx, &block.ffn.fc2.bias)
                .map_err(|e| WhisperError::Inference(format!("ffn_down_b upload: {e}")))?;

            gpu_weights.push(GpuEncoderBlockWeights {
                ln1_gamma,
                ln1_beta,
                w_q,
                b_q,
                w_k,
                b_k,
                w_v,
                b_v,
                w_o,
                b_o,
                ln2_gamma,
                ln2_beta,
                ffn_up_w,
                ffn_up_b,
                ffn_down_w,
                ffn_down_b,
            });
        }

        self.gpu_encoder_weights = Some(gpu_weights);
        self.gpu_encoder_config = Some(GpuEncoderConfig {
            d_model: d_model as u32,
            n_heads: n_heads as u32,
            ffn_dim: d_ff as u32,
        });

        // WAPR-PERF-019: Upload encoder post-norm weights to eliminate D2H→CPU→H2D round-trip
        let ln_post = self.encoder.ln_post();
        self.gpu_enc_ln_post_gamma = Some(
            GpuResidentTensor::from_host(ctx, &ln_post.weight)
                .map_err(|e| WhisperError::Inference(format!("enc ln_post_gamma upload: {e}")))?,
        );
        self.gpu_enc_ln_post_beta = Some(
            GpuResidentTensor::from_host(ctx, &ln_post.bias)
                .map_err(|e| WhisperError::Inference(format!("enc ln_post_beta upload: {e}")))?,
        );

        Ok(())
    }

    /// WAPR-PERF-012: Upload convolutional frontend weights to GPU
    ///
    /// Uploads conv1/conv2 weights and biases for GPU-accelerated audio processing.
    /// Target: Move 588ms CPU conv to GPU (<50ms).
    #[cfg(feature = "cuda")]
    pub fn upload_conv_weights_to_gpu(&mut self) -> WhisperResult<()> {
        if self.gpu_conv_weights.is_some() {
            return Ok(()); // Already uploaded
        }

        let ctx = self.executor.context();
        let conv_frontend = self
            .encoder
            .conv_frontend()
            .ok_or_else(|| WhisperError::Inference("no conv frontend for GPU upload".into()))?;

        // Upload conv1 weights [out_channels, in_channels, kernel_size]
        let conv1_weight = GpuResidentTensor::from_host(ctx, &conv_frontend.conv1.weight)
            .map_err(|e| WhisperError::Inference(format!("conv1_weight upload: {e}")))?;
        let conv1_bias = GpuResidentTensor::from_host(ctx, &conv_frontend.conv1.bias)
            .map_err(|e| WhisperError::Inference(format!("conv1_bias upload: {e}")))?;

        // Upload conv2 weights [out_channels, in_channels, kernel_size]
        let conv2_weight = GpuResidentTensor::from_host(ctx, &conv_frontend.conv2.weight)
            .map_err(|e| WhisperError::Inference(format!("conv2_weight upload: {e}")))?;
        let conv2_bias = GpuResidentTensor::from_host(ctx, &conv_frontend.conv2.bias)
            .map_err(|e| WhisperError::Inference(format!("conv2_bias upload: {e}")))?;

        self.gpu_conv_weights = Some(GpuConvFrontendWeights {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
        });

        Ok(())
    }

    /// WAPR-PERF-013: Upload decoder block weights to GPU
    ///
    /// Uploads all decoder weights for full GPU residence:
    /// - Self-attention: LN1, Q/K/V/O projections
    /// - Cross-attention: LN2, Q/K/V/O projections
    /// - FFN: LN3, FC1, FC2
    #[cfg(feature = "cuda")]
    pub fn upload_decoder_weights_to_gpu(&mut self) -> WhisperResult<()> {
        if self.gpu_decoder_weights.is_some() {
            return Ok(()); // Already uploaded
        }

        // Helper to transpose weight matrix from [rows, cols] to [cols, rows]
        fn transpose_weights(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            let mut transposed = vec![0.0_f32; weights.len()];
            for r in 0..rows {
                for c in 0..cols {
                    transposed[c * rows + r] = weights[r * cols + c];
                }
            }
            transposed
        }

        let ctx = self.executor.context();
        let d_model = self.config.n_text_state as usize;
        let n_heads = self.config.n_text_head as usize;
        let n_layers = self.config.n_text_layer as usize;
        let d_ff = d_model * 4; // Standard 4x expansion
        let max_seq_len = self.config.n_text_ctx as usize;

        let mut gpu_weights = Vec::with_capacity(n_layers);

        for layer_idx in 0..n_layers {
            let block = &self.decoder.blocks()[layer_idx];

            // Self-Attention weights
            let ln1_gamma = GpuResidentTensor::from_host(ctx, &block.ln1.weight)
                .map_err(|e| WhisperError::Inference(format!("dec ln1_gamma L{layer_idx}: {e}")))?;
            let ln1_beta = GpuResidentTensor::from_host(ctx, &block.ln1.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ln1_beta L{layer_idx}: {e}")))?;

            // Self-attention Q/K/V/O (transposed for GPU linear: [in, out])
            let self_w_q_t = transpose_weights(&block.self_attn.w_q().weight, d_model, d_model);
            let self_w_q = GpuResidentTensor::from_host(ctx, &self_w_q_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_q L{layer_idx}: {e}")))?;
            let self_b_q = GpuResidentTensor::from_host(ctx, &block.self_attn.w_q().bias)
                .map_err(|e| WhisperError::Inference(format!("dec self_b_q L{layer_idx}: {e}")))?;

            let self_w_k_t = transpose_weights(&block.self_attn.w_k().weight, d_model, d_model);
            let self_w_k = GpuResidentTensor::from_host(ctx, &self_w_k_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_k L{layer_idx}: {e}")))?;
            let self_b_k = GpuResidentTensor::from_host(ctx, &block.self_attn.w_k().bias)
                .map_err(|e| WhisperError::Inference(format!("dec self_b_k L{layer_idx}: {e}")))?;

            let self_w_v_t = transpose_weights(&block.self_attn.w_v().weight, d_model, d_model);
            let self_w_v = GpuResidentTensor::from_host(ctx, &self_w_v_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_v L{layer_idx}: {e}")))?;
            let self_b_v = GpuResidentTensor::from_host(ctx, &block.self_attn.w_v().bias)
                .map_err(|e| WhisperError::Inference(format!("dec self_b_v L{layer_idx}: {e}")))?;

            let self_w_o_t = transpose_weights(&block.self_attn.w_o().weight, d_model, d_model);
            let self_w_o = GpuResidentTensor::from_host(ctx, &self_w_o_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_o L{layer_idx}: {e}")))?;
            let self_b_o = GpuResidentTensor::from_host(ctx, &block.self_attn.w_o().bias)
                .map_err(|e| WhisperError::Inference(format!("dec self_b_o L{layer_idx}: {e}")))?;

            // Cross-Attention weights
            let ln2_gamma = GpuResidentTensor::from_host(ctx, &block.ln2.weight)
                .map_err(|e| WhisperError::Inference(format!("dec ln2_gamma L{layer_idx}: {e}")))?;
            let ln2_beta = GpuResidentTensor::from_host(ctx, &block.ln2.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ln2_beta L{layer_idx}: {e}")))?;

            let cross_w_q_t = transpose_weights(&block.cross_attn.w_q().weight, d_model, d_model);
            let cross_w_q = GpuResidentTensor::from_host(ctx, &cross_w_q_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_q L{layer_idx}: {e}")))?;
            let cross_b_q = GpuResidentTensor::from_host(ctx, &block.cross_attn.w_q().bias)
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_q L{layer_idx}: {e}")))?;

            let cross_w_k_t = transpose_weights(&block.cross_attn.w_k().weight, d_model, d_model);
            let cross_w_k = GpuResidentTensor::from_host(ctx, &cross_w_k_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_k L{layer_idx}: {e}")))?;
            let cross_b_k = GpuResidentTensor::from_host(ctx, &block.cross_attn.w_k().bias)
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_k L{layer_idx}: {e}")))?;

            let cross_w_v_t = transpose_weights(&block.cross_attn.w_v().weight, d_model, d_model);
            let cross_w_v = GpuResidentTensor::from_host(ctx, &cross_w_v_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_v L{layer_idx}: {e}")))?;
            let cross_b_v = GpuResidentTensor::from_host(ctx, &block.cross_attn.w_v().bias)
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_v L{layer_idx}: {e}")))?;

            let cross_w_o_t = transpose_weights(&block.cross_attn.w_o().weight, d_model, d_model);
            let cross_w_o = GpuResidentTensor::from_host(ctx, &cross_w_o_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_o L{layer_idx}: {e}")))?;
            let cross_b_o = GpuResidentTensor::from_host(ctx, &block.cross_attn.w_o().bias)
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_o L{layer_idx}: {e}")))?;

            // FFN weights
            let ln3_gamma = GpuResidentTensor::from_host(ctx, &block.ln3.weight)
                .map_err(|e| WhisperError::Inference(format!("dec ln3_gamma L{layer_idx}: {e}")))?;
            let ln3_beta = GpuResidentTensor::from_host(ctx, &block.ln3.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ln3_beta L{layer_idx}: {e}")))?;

            // FFN up: [d_ff, d_model] -> transposed to [d_model, d_ff]
            let ffn_up_t = transpose_weights(&block.ffn.fc1.weight, d_ff, d_model);
            let ffn_up_w = GpuResidentTensor::from_host(ctx, &ffn_up_t)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_up_w L{layer_idx}: {e}")))?;
            let ffn_up_b = GpuResidentTensor::from_host(ctx, &block.ffn.fc1.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_up_b L{layer_idx}: {e}")))?;

            // FFN down: [d_model, d_ff] -> transposed to [d_ff, d_model]
            let ffn_down_t = transpose_weights(&block.ffn.fc2.weight, d_model, d_ff);
            let ffn_down_w = GpuResidentTensor::from_host(ctx, &ffn_down_t).map_err(|e| {
                WhisperError::Inference(format!("dec ffn_down_w L{layer_idx}: {e}"))
            })?;
            let ffn_down_b =
                GpuResidentTensor::from_host(ctx, &block.ffn.fc2.bias).map_err(|e| {
                    WhisperError::Inference(format!("dec ffn_down_b L{layer_idx}: {e}"))
                })?;

            gpu_weights.push(GpuDecoderBlockWeights {
                ln1_gamma,
                ln1_beta,
                self_w_q,
                self_b_q,
                self_w_k,
                self_b_k,
                self_w_v,
                self_b_v,
                self_w_o,
                self_b_o,
                ln2_gamma,
                ln2_beta,
                cross_w_q,
                cross_b_q,
                cross_w_k,
                cross_b_k,
                cross_w_v,
                cross_b_v,
                cross_w_o,
                cross_b_o,
                ln3_gamma,
                ln3_beta,
                ffn_up_w,
                ffn_up_b,
                ffn_down_w,
                ffn_down_b,
            });
        }

        self.gpu_decoder_weights = Some(gpu_weights);
        self.gpu_decoder_config = Some(GpuDecoderConfig {
            d_model: d_model as u32,
            n_heads: n_heads as u32,
            ffn_dim: d_ff as u32,
            max_seq_len: max_seq_len as u32,
            n_layers: n_layers as u32,
        });

        Ok(())
    }

    /// WAPR-PERF-013: Initialize GPU KV caches for decoder
    #[cfg(feature = "cuda")]
    pub fn init_gpu_decoder_kv_cache(&mut self) -> WhisperResult<()> {
        if self.gpu_self_kv_cache.is_some() {
            return Ok(()); // Already initialized
        }

        let ctx = self.executor.context();
        let d_model = self.config.n_text_state as usize;
        let n_layers = self.config.n_text_layer as usize;
        let max_seq_len = self.config.n_text_ctx as usize;

        // Self-attention KV caches (one per layer)
        let mut self_kv_caches = Vec::with_capacity(n_layers);
        for _layer in 0..n_layers {
            let cache = GpuKvCache::new(ctx, max_seq_len, d_model)
                .map_err(|e| WhisperError::Inference(format!("GPU self KV cache: {e}")))?;
            self_kv_caches.push(cache);
        }

        // Cross-attention KV caches (one per layer, for encoder K/V)
        // These are computed once from encoder output
        let mut cross_kv_caches = Vec::with_capacity(n_layers);
        for _layer in 0..n_layers {
            // Use encoder output length (1500 for Whisper tiny)
            let enc_seq_len = 1500; // Fixed for Whisper
            let cache = GpuKvCache::new(ctx, enc_seq_len, d_model)
                .map_err(|e| WhisperError::Inference(format!("GPU cross KV cache: {e}")))?;
            cross_kv_caches.push(cache);
        }

        self.gpu_self_kv_cache = Some(self_kv_caches);
        self.gpu_cross_kv_cache = Some(cross_kv_caches);

        Ok(())
    }

    /// WAPR-PERF-013: Initialize head-first KV caches for GPU decoder
    ///
    /// Creates KV caches in head-first layout [n_heads, max_seq_len, head_dim]
    /// required by `incremental_attention_gpu`.
    #[cfg(feature = "cuda")]
    pub fn init_gpu_decoder_kv_cache_head_first(&mut self) -> WhisperResult<()> {
        if self.gpu_self_k_head_first.is_some() {
            return Ok(()); // Already initialized
        }

        let ctx = self.executor.context();
        let d_model = self.config.n_text_state as usize;
        let n_heads = self.config.n_text_head as usize;
        let n_layers = self.config.n_text_layer as usize;
        let head_dim = d_model / n_heads;
        let max_seq_len = self.config.n_text_ctx as usize;
        let enc_seq_len = 1500_usize; // Fixed for Whisper

        // Head-first cache size: [n_heads, seq_len, head_dim]
        let self_cache_size = n_heads * max_seq_len * head_dim;
        let cross_cache_size = n_heads * enc_seq_len * head_dim;

        let mut self_k_caches = Vec::with_capacity(n_layers);
        let mut self_v_caches = Vec::with_capacity(n_layers);
        let mut cross_k_caches = Vec::with_capacity(n_layers);
        let mut cross_v_caches = Vec::with_capacity(n_layers);

        for _layer in 0..n_layers {
            // Self-attention caches
            let zeros_self = vec![0.0f32; self_cache_size];
            let k_self = GpuResidentTensor::from_host(ctx, &zeros_self)
                .map_err(|e| WhisperError::Inference(format!("self K cache: {e}")))?;
            let v_self = GpuResidentTensor::from_host(ctx, &zeros_self)
                .map_err(|e| WhisperError::Inference(format!("self V cache: {e}")))?;
            self_k_caches.push(k_self);
            self_v_caches.push(v_self);

            // Cross-attention caches
            let zeros_cross = vec![0.0f32; cross_cache_size];
            let k_cross = GpuResidentTensor::from_host(ctx, &zeros_cross)
                .map_err(|e| WhisperError::Inference(format!("cross K cache: {e}")))?;
            let v_cross = GpuResidentTensor::from_host(ctx, &zeros_cross)
                .map_err(|e| WhisperError::Inference(format!("cross V cache: {e}")))?;
            cross_k_caches.push(k_cross);
            cross_v_caches.push(v_cross);
        }

        self.gpu_self_k_head_first = Some(self_k_caches);
        self.gpu_self_v_head_first = Some(self_v_caches);
        self.gpu_cross_k_head_first = Some(cross_k_caches);
        self.gpu_cross_v_head_first = Some(cross_v_caches);
        self.gpu_decoder_pos = 0;

        Ok(())
    }

    /// WAPR-PERF-013: Reset decoder position for new sequence
    #[cfg(feature = "cuda")]
    pub fn reset_gpu_decoder_pos(&mut self) {
        self.gpu_decoder_pos = 0;
    }

    /// WAPR-PERF-014: Reset GPU decoder KV caches (forces re-initialization)
    ///
    /// Clears all head-first KV caches to force fresh allocation on next init.
    /// Call this before switching between GPU path and Executor path in benchmarks.
    /// Resets BOTH cache formats to ensure clean state.
    #[cfg(feature = "cuda")]
    pub fn reset_gpu_decoder_kv_cache(&mut self) {
        // Head-first format (Executor path)
        self.gpu_self_k_head_first = None;
        self.gpu_self_v_head_first = None;
        self.gpu_cross_k_head_first = None;
        self.gpu_cross_v_head_first = None;
        // Layer-major format (GPU path)
        self.gpu_self_kv_cache = None;
        self.gpu_cross_kv_cache = None;
    }

    /// WAPR-PERF-013: GPU decoder block forward pass
    ///
    /// Processes a single token through one decoder block on GPU.
    /// Uses head-first KV caches for zero-conversion attention.
    ///
    /// # Architecture
    ///
    /// ```text
    /// x → LN1 → Q/K/V (GPU) → scatter K/V → incr_attn (GPU) → O (GPU) → residual
    ///   → LN2 → Q (GPU) → cross_attn (GPU) → O (GPU) → residual
    ///   → LN3 → FC1 (GPU) → GELU → FC2 (GPU) → residual
    /// ```
    ///
    /// # Point 149 Compliance
    ///
    /// All GPU operations chain on implicit stream. No explicit sync inside.
    /// Caller must sync only when reading final output.
    ///
    /// # Parameters
    ///
    /// - `encoder_output`: Optional encoder hidden states for cross-attention.
    ///   If None, cross-attention is skipped (useful for testing self-attention).
    #[cfg(feature = "cuda")]
    pub fn forward_decoder_block_gpu(
        &mut self,
        layer_idx: usize,
        x: &[f32],
        pos: usize,
        encoder_output: Option<&[f32]>,
    ) -> WhisperResult<Vec<f32>> {
        use trueno_gpu::driver::CudaStream;

        let ctx = self.executor.context();
        let d_model = self.config.n_text_state as usize;
        let n_heads = self.config.n_text_head as usize;
        let head_dim = d_model / n_heads;
        let max_seq_len = self.config.n_text_ctx as usize;

        // Get GPU weights for this layer
        let weights = self
            .gpu_decoder_weights
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("Decoder weights not uploaded".into()))?;
        let layer_weights = &weights[layer_idx];

        // Get head-first KV caches
        let self_k_caches = self
            .gpu_self_k_head_first
            .as_mut()
            .ok_or_else(|| WhisperError::Inference("Self K cache not initialized".into()))?;
        let self_v_caches = self
            .gpu_self_v_head_first
            .as_mut()
            .ok_or_else(|| WhisperError::Inference("Self V cache not initialized".into()))?;

        let block = &self.decoder.blocks()[layer_idx];

        // === Self-Attention ===

        // LN1 (CPU - simple and correct)
        let normed = block.ln1.forward(x)?;

        // Upload normed input to GPU
        let x_gpu = GpuResidentTensor::from_host(ctx, &normed)
            .map_err(|e| WhisperError::Inference(format!("x upload: {e}")))?;

        // Q/K/V projections on GPU: [1, d_model] @ [d_model, d_model] = [1, d_model]
        let q = x_gpu
            .linear(
                ctx,
                &layer_weights.self_w_q,
                Some(&layer_weights.self_b_q),
                1,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("Q projection: {e}")))?;
        let k = x_gpu
            .linear(
                ctx,
                &layer_weights.self_w_k,
                Some(&layer_weights.self_b_k),
                1,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("K projection: {e}")))?;
        let v = x_gpu
            .linear(
                ctx,
                &layer_weights.self_w_v,
                Some(&layer_weights.self_b_v),
                1,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("V projection: {e}")))?;

        // Scatter K/V to head-first caches
        let stream =
            CudaStream::new(ctx).map_err(|e| WhisperError::Inference(format!("Stream: {e}")))?;

        kv_cache_scatter_gpu(
            ctx,
            &k,
            &mut self_k_caches[layer_idx],
            pos as u32,
            n_heads as u32,
            head_dim as u32,
            max_seq_len as u32,
            &stream,
        )
        .map_err(|e| WhisperError::Inference(format!("K scatter: {e}")))?;

        kv_cache_scatter_gpu(
            ctx,
            &v,
            &mut self_v_caches[layer_idx],
            pos as u32,
            n_heads as u32,
            head_dim as u32,
            max_seq_len as u32,
            &stream,
        )
        .map_err(|e| WhisperError::Inference(format!("V scatter: {e}")))?;

        // Incremental self-attention: Q @ cached_K^T → softmax → @ cached_V
        let seq_len = (pos + 1) as u32; // Include current position
        let attn_out = incremental_attention_gpu(
            ctx,
            &q,
            &self_k_caches[layer_idx],
            &self_v_caches[layer_idx],
            n_heads as u32,
            head_dim as u32,
            seq_len,
            max_seq_len as u32,
        )
        .map_err(|e| WhisperError::Inference(format!("Self attention: {e}")))?;

        // Output projection
        let mut attn_proj = attn_out
            .linear(
                ctx,
                &layer_weights.self_w_o,
                Some(&layer_weights.self_b_o),
                1,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("O projection: {e}")))?;

        // Download and add residual (sync point)
        let attn_proj_host = attn_proj
            .to_host()
            .map_err(|e| WhisperError::Inference(format!("Attn D2H: {e}")))?;

        // Residual connection
        let mut residual: Vec<f32> = x
            .iter()
            .zip(attn_proj_host.iter())
            .map(|(a, b)| a + b)
            .collect();

        // === Cross-Attention ===
        if let Some(enc_out) = encoder_output {
            let normed2 = block.ln2.forward(&residual)?;
            let cross_out = block.cross_attn.forward_cross_dispatch(
                &normed2, enc_out,
                None, // Cross-attention K/V caching tracked in WAPR-PERF-007
            )?;
            for (r, c) in residual.iter_mut().zip(cross_out.iter()) {
                *r += c;
            }
        }
        // Note: When encoder_output is None, cross-attention is skipped.
        // This is useful for testing self-attention in isolation.

        // === FFN (CPU for now) ===
        let normed3 = block.ln3.forward(&residual)?;
        let ffn_out = block.ffn.forward(&normed3)?;
        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }

    /// WAPR-PERF-017: GPU decoder block with external stream (CUDA Graph capturable)
    ///
    /// Same as `forward_decoder_block_gpu` but uses external stream for all operations.
    /// Does NOT synchronize internally - caller controls when to sync.
    ///
    /// This enables CUDA Graph capture: all operations recorded to a graph that can
    /// be replayed with ~3-10µs launch overhead instead of ~20-50µs per kernel.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Decoder layer index
    /// * `x_gpu` - Input tensor on GPU [1, d_model]
    /// * `pos` - Current position in sequence
    /// * `stream` - Caller-provided CUDA stream for graph capture
    /// * `enc_seq_len` - If Some, enables cross-attention using cached encoder K/V
    ///
    /// # Returns
    ///
    /// Output tensor on GPU [1, d_model] (still on GPU, no D2H)
    #[cfg(feature = "cuda")]
    pub fn forward_decoder_block_gpu_stream(
        &mut self,
        layer_idx: usize,
        x_gpu: &GpuResidentTensor<f32>,
        pos: usize,
        stream: &trueno_gpu::driver::CudaStream,
        enc_seq_len: Option<usize>,
    ) -> WhisperResult<GpuResidentTensor<f32>> {
        let ctx = self.executor.context();
        let d_model = self.config.n_text_state as usize;
        let n_heads = self.config.n_text_head as usize;
        let head_dim = d_model / n_heads;
        let max_seq_len = self.config.n_text_ctx as usize;

        // Get GPU weights for this layer
        let weights = self
            .gpu_decoder_weights
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("Decoder weights not uploaded".into()))?;
        let layer_weights = &weights[layer_idx];

        // Get head-first KV caches
        let self_k_caches = self
            .gpu_self_k_head_first
            .as_mut()
            .ok_or_else(|| WhisperError::Inference("Self K cache not initialized".into()))?;
        let self_v_caches = self
            .gpu_self_v_head_first
            .as_mut()
            .ok_or_else(|| WhisperError::Inference("Self V cache not initialized".into()))?;

        // === Self-Attention (all GPU, using external stream) ===

        // LN1 on GPU using stream
        let normed = x_gpu
            .layer_norm_with_stream(
                ctx,
                &layer_weights.ln1_gamma,
                &layer_weights.ln1_beta,
                d_model as u32,
                1, // batch_size = 1 for single token
                stream,
            )
            .map_err(|e| WhisperError::Inference(format!("LN1: {e}")))?;

        // Q/K/V projections (use matmul_with_stream internally via linear)
        // Note: linear() creates its own stream, but we can still capture the sequence
        let q = normed
            .linear(
                ctx,
                &layer_weights.self_w_q,
                Some(&layer_weights.self_b_q),
                1,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("Q projection: {e}")))?;
        let k = normed
            .linear(
                ctx,
                &layer_weights.self_w_k,
                Some(&layer_weights.self_b_k),
                1,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("K projection: {e}")))?;
        let v = normed
            .linear(
                ctx,
                &layer_weights.self_w_v,
                Some(&layer_weights.self_b_v),
                1,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("V projection: {e}")))?;

        // KV cache scatter (uses provided stream)
        kv_cache_scatter_gpu(
            ctx,
            &k,
            &mut self_k_caches[layer_idx],
            pos as u32,
            n_heads as u32,
            head_dim as u32,
            max_seq_len as u32,
            stream,
        )
        .map_err(|e| WhisperError::Inference(format!("K scatter: {e}")))?;

        kv_cache_scatter_gpu(
            ctx,
            &v,
            &mut self_v_caches[layer_idx],
            pos as u32,
            n_heads as u32,
            head_dim as u32,
            max_seq_len as u32,
            stream,
        )
        .map_err(|e| WhisperError::Inference(format!("V scatter: {e}")))?;

        // Incremental attention using stream
        let seq_len = (pos + 1) as u32;
        let attn_out = incremental_attention_gpu_with_stream(
            ctx,
            &q,
            &self_k_caches[layer_idx],
            &self_v_caches[layer_idx],
            n_heads as u32,
            head_dim as u32,
            seq_len,
            max_seq_len as u32,
            stream,
        )
        .map_err(|e| WhisperError::Inference(format!("Self attention: {e}")))?;

        // Output projection
        let attn_proj = attn_out
            .linear(
                ctx,
                &layer_weights.self_w_o,
                Some(&layer_weights.self_b_o),
                1,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("O projection: {e}")))?;

        // Residual connection (GPU)
        let residual1 = x_gpu
            .add_with_stream(ctx, &attn_proj, stream)
            .map_err(|e| WhisperError::Inference(format!("Residual 1: {e}")))?;

        // === Cross-Attention (GPU, if encoder output provided) ===
        let residual2 =
            if let Some(enc_len) = enc_seq_len {
                // Get cross-attention K/V caches
                let cross_k_caches = self.gpu_cross_k_head_first.as_ref().ok_or_else(|| {
                    WhisperError::Inference("Cross K cache not initialized".into())
                })?;
                let cross_v_caches = self.gpu_cross_v_head_first.as_ref().ok_or_else(|| {
                    WhisperError::Inference("Cross V cache not initialized".into())
                })?;

                // LN2 on residual1
                let normed2 = residual1
                    .layer_norm_with_stream(
                        ctx,
                        &layer_weights.ln2_gamma,
                        &layer_weights.ln2_beta,
                        d_model as u32,
                        1,
                        stream,
                    )
                    .map_err(|e| WhisperError::Inference(format!("LN2: {e}")))?;

                // Cross-attention Q from decoder hidden state
                let q_cross = normed2
                    .linear(
                        ctx,
                        &layer_weights.cross_w_q,
                        Some(&layer_weights.cross_b_q),
                        1,
                        d_model as u32,
                        d_model as u32,
                    )
                    .map_err(|e| WhisperError::Inference(format!("Cross Q: {e}")))?;

                // Cross-attention using cached encoder K/V
                // Note: We use incremental_attention_gpu but with full enc_len as seq_len
                let cross_attn_out = incremental_attention_gpu_with_stream(
                    ctx,
                    &q_cross,
                    &cross_k_caches[layer_idx],
                    &cross_v_caches[layer_idx],
                    n_heads as u32,
                    head_dim as u32,
                    enc_len as u32,
                    enc_len as u32, // max_seq_len = enc_len for cross-attention
                    stream,
                )
                .map_err(|e| WhisperError::Inference(format!("Cross attention: {e}")))?;

                // Output projection
                let cross_proj = cross_attn_out
                    .linear(
                        ctx,
                        &layer_weights.cross_w_o,
                        Some(&layer_weights.cross_b_o),
                        1,
                        d_model as u32,
                        d_model as u32,
                    )
                    .map_err(|e| WhisperError::Inference(format!("Cross O: {e}")))?;

                // Cross-attention residual
                residual1
                    .add_with_stream(ctx, &cross_proj, stream)
                    .map_err(|e| WhisperError::Inference(format!("Cross residual: {e}")))?
            } else {
                // No cross-attention (self-attention only path for testing)
                residual1
            };

        // === FFN (all GPU, using external stream) ===

        // LN3 on GPU
        let normed3 = residual2
            .layer_norm_with_stream(
                ctx,
                &layer_weights.ln3_gamma,
                &layer_weights.ln3_beta,
                d_model as u32,
                1,
                stream,
            )
            .map_err(|e| WhisperError::Inference(format!("LN3: {e}")))?;

        // FFN up projection + GELU
        let ffn_up = normed3
            .linear(
                ctx,
                &layer_weights.ffn_up_w,
                Some(&layer_weights.ffn_up_b),
                1,
                d_model as u32,
                (d_model * 4) as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("FFN up: {e}")))?;
        let ffn_gelu = ffn_up
            .gelu_with_stream(ctx, stream)
            .map_err(|e| WhisperError::Inference(format!("GELU: {e}")))?;

        // FFN down projection
        let ffn_down = ffn_gelu
            .linear(
                ctx,
                &layer_weights.ffn_down_w,
                Some(&layer_weights.ffn_down_b),
                1,
                (d_model * 4) as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("FFN down: {e}")))?;

        // Final residual connection
        let output = residual2
            .add_with_stream(ctx, &ffn_down, stream)
            .map_err(|e| WhisperError::Inference(format!("FFN residual: {e}")))?;

        Ok(output)
    }

    /// WAPR-PERF-013: Full GPU decoder forward pass for single token
    ///
    /// Runs a single token through all decoder layers on GPU.
    /// Uses head-first KV caches for zero-conversion attention.
    ///
    /// # Point 157 Compliance
    ///
    /// - Single H2D at start (token embedding)
    /// - Minimal sync points (once per token, not per layer)
    /// - Target: Full transcription ≤1984ms (2x whisper.cpp @ 992ms)
    ///
    /// # Parameters
    ///
    /// - `token_embedding`: Embedded token vector [d_model]
    /// - `pos`: Current position in sequence
    /// - `encoder_output`: Encoder hidden states for cross-attention
    #[cfg(feature = "cuda")]
    pub fn forward_decoder_token_gpu(
        &mut self,
        token_embedding: &[f32],
        pos: usize,
        encoder_output: &[f32],
    ) -> WhisperResult<Vec<f32>> {
        let n_layers = self.config.n_text_layer as usize;

        // Ensure weights and KV caches are initialized
        if self.gpu_decoder_weights.is_none() {
            self.upload_decoder_weights_to_gpu()?;
        }
        if self.gpu_self_k_head_first.is_none() {
            self.init_gpu_decoder_kv_cache_head_first()?;
        }

        // Process through all layers
        let mut hidden = token_embedding.to_vec();
        for layer_idx in 0..n_layers {
            hidden =
                self.forward_decoder_block_gpu(layer_idx, &hidden, pos, Some(encoder_output))?;
        }

        Ok(hidden)
    }

    /// WAPR-PERF-017: Stream-based decoder token forward pass
    ///
    /// All-GPU implementation using external stream for CUDA graph compatibility.
    /// Achieves 97x speedup when combined with graph capture.
    ///
    /// # Parameters
    ///
    /// - `token_embedding`: Embedded token vector [d_model]
    /// - `pos`: Current position in sequence
    /// - `stream`: External CUDA stream for graph capture
    /// - `enc_seq_len`: If Some, enables cross-attention using cached encoder K/V
    ///
    /// # Returns
    ///
    /// GPU-resident output tensor (no D2H - caller handles sync and download)
    #[cfg(feature = "cuda")]
    pub fn forward_decoder_token_gpu_stream(
        &mut self,
        token_embedding: &[f32],
        pos: usize,
        stream: &trueno_gpu::driver::CudaStream,
        enc_seq_len: Option<usize>,
    ) -> WhisperResult<GpuResidentTensor<f32>> {
        let n_layers = self.config.n_text_layer as usize;
        let profile_layers = std::env::var("WHISPER_PROFILE_DECODER_LAYERS").is_ok();

        // Ensure weights and KV caches are initialized (before borrowing ctx)
        if self.gpu_decoder_weights.is_none() {
            self.upload_decoder_weights_to_gpu()?;
        }
        if self.gpu_self_k_head_first.is_none() {
            self.init_gpu_decoder_kv_cache_head_first()?;
        }

        // Get context after mutable initialization
        let ctx = self.executor.context();

        // Upload embedding to GPU
        let embed_start = std::time::Instant::now();
        let mut hidden_gpu = GpuResidentTensor::from_host(ctx, token_embedding)
            .map_err(|e| WhisperError::Inference(format!("embedding upload: {e}")))?;
        if profile_layers {
            stream.synchronize().ok();
            eprintln!(
                "[PROFILE-DEC-EMBED] pos={} embed_upload: {:.2}ms",
                pos,
                embed_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Process through all layers using stream-based path
        for layer_idx in 0..n_layers {
            let layer_start = std::time::Instant::now();
            hidden_gpu = self.forward_decoder_block_gpu_stream(
                layer_idx,
                &hidden_gpu,
                pos,
                stream,
                enc_seq_len,
            )?;
            if profile_layers {
                stream.synchronize().ok();
                eprintln!(
                    "[PROFILE-DEC-LAYER] pos={} layer={} time: {:.2}ms",
                    pos,
                    layer_idx,
                    layer_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }

        Ok(hidden_gpu)
    }

    /// Full GPU encoder (WAPR-PERF-004: Total Offload)
    ///
    /// Runs the entire encoder on GPU with minimal host transfers:
    /// - 1 H2D: Input mel spectrogram upload
    /// - 0 transfers during forward pass (all weights pre-uploaded)
    /// - 1 D2H: Final output download
    ///
    /// Requires: `upload_encoder_weights_to_gpu()` called first.
    #[cfg(feature = "cuda")]
    pub fn encode_gpu_total_offload(&mut self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        // Ensure weights are uploaded
        if self.gpu_encoder_weights.is_none() {
            self.upload_encoder_weights_to_gpu()?;
        }

        // WAPR-PERF-012: Upload conv weights if not already
        if self.gpu_conv_weights.is_none() {
            self.upload_conv_weights_to_gpu()?;
        }

        let ctx = self.executor.context();
        let d_model = self.config.n_audio_state as usize;
        let n_layers = self.config.n_audio_layer as usize;
        let n_mels = self.config.n_mels as usize;

        // Reset transfer counters for monitoring
        reset_transfer_counters();

        // WAPR-PERF-011: Detailed timing breakdown
        let profile_detail = std::env::var("WHISPER_PROFILE_LAYERS").is_ok();
        let total_start = std::time::Instant::now();

        // WAPR-PERF-012: GPU Convolutional frontend
        let conv_start = std::time::Instant::now();
        let seq_len_in = mel.len() / n_mels;

        // Upload mel to GPU
        let mel_gpu = GpuResidentTensor::from_host(ctx, mel)
            .map_err(|e| WhisperError::Inference(format!("mel upload: {e}")))?;

        let conv_weights = self
            .gpu_conv_weights
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("GPU conv weights not uploaded".into()))?;

        // Conv1: 80 → 384, kernel=3, stride=1, padding=1 + GELU
        let conv1_out = mel_gpu
            .conv1d(
                ctx,
                &conv_weights.conv1_weight,
                Some(&conv_weights.conv1_bias),
                n_mels as u32,     // in_channels
                d_model as u32,    // out_channels
                3,                 // kernel_size
                1,                 // stride
                1,                 // padding
                seq_len_in as u32, // seq_len
            )
            .map_err(|e| WhisperError::Inference(format!("conv1 GPU: {e}")))?;

        // After conv1: seq_len stays same (stride=1), channels = d_model
        let seq_len_after_conv1 = seq_len_in;

        // Conv2: 384 → 384, kernel=3, stride=2, padding=1 + GELU
        let mut conv2_out = conv1_out
            .conv1d(
                ctx,
                &conv_weights.conv2_weight,
                Some(&conv_weights.conv2_bias),
                d_model as u32, // in_channels
                d_model as u32, // out_channels
                3,              // kernel_size
                2,              // stride
                1,              // padding
                seq_len_after_conv1 as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("conv2 GPU: {e}")))?;

        let conv_time = conv_start.elapsed();

        // After conv2: seq_len halved (stride=2)
        let seq_len = (seq_len_after_conv1 + 2 - 3) / 2 + 1;

        // Download conv output to add positional embedding (CPU - small overhead)
        let pos_start = std::time::Instant::now();
        let mut x = conv2_out
            .to_host()
            .map_err(|e| WhisperError::Inference(format!("conv output download: {e}")))?;

        // Add positional embedding
        let pos_emb = self.encoder.positional_embedding();
        for pos in 0..seq_len {
            for d in 0..d_model {
                x[pos * d_model + d] += pos_emb[pos * d_model + d];
            }
        }
        let pos_time = pos_start.elapsed();

        // Upload to GPU for transformer blocks
        let upload_start = std::time::Instant::now();
        let mut x_gpu = GpuResidentTensor::from_host(ctx, &x)
            .map_err(|e| WhisperError::Inference(format!("input upload: {e}")))?;
        let upload_time = upload_start.elapsed();

        if profile_detail {
            eprintln!(
                "[PROFILE-BREAKDOWN] Conv(GPU): {:.1}ms, PosEmb: {:.1}ms, Upload: {:.1}ms",
                conv_time.as_millis(),
                pos_time.as_millis(),
                upload_time.as_millis()
            );
        }

        // Step 4: Process all encoder blocks on GPU (0 transfers)
        let weights = self
            .gpu_encoder_weights
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("GPU weights not uploaded".into()))?;
        let config = self
            .gpu_encoder_config
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("GPU config not set".into()))?;

        // Process all encoder layers on GPU
        let mut layer_times: Vec<u128> = Vec::new();
        for layer_idx in 0..n_layers {
            let layer_start = std::time::Instant::now();
            x_gpu = forward_encoder_block_gpu(ctx, &x_gpu, &weights[layer_idx], config)
                .map_err(|e| WhisperError::Inference(format!("encoder block {layer_idx}: {e}")))?;
            layer_times.push(layer_start.elapsed().as_micros());
        }

        // Step 5: Download output (1 D2H transfer)
        let download_start = std::time::Instant::now();
        let output = x_gpu
            .to_host()
            .map_err(|e| WhisperError::Inference(format!("output download: {e}")))?;
        let download_time = download_start.elapsed();

        // Step 6: Final layer norm (CPU - small overhead)
        let result = self.encoder.ln_post().forward(&output)?;

        Ok(result)
    }

    /// WAPR-PERF-020: Pre-compile all GPU kernels for predictable latency
    ///
    /// Runs a complete encoder+decoder forward pass with dummy data to JIT compile
    /// all PTX kernels. This moves the ~200ms compilation overhead from first
    /// transcription to model initialization.
    ///
    /// # When to Call
    ///
    /// Call `warmup()` after `into_cuda()` to ensure all subsequent transcriptions
    /// run at full speed (~10ms instead of ~200ms for first transcription).
    ///
    /// ```rust,ignore
    /// let mut cuda_model = apr.into_cuda(0)?;
    /// cuda_model.warmup()?;  // ~200ms kernel compilation
    /// // All subsequent calls now fast:
    /// let result = cuda_model.transcribe_gpu(&audio, options)?;  // ~10ms
    /// ```
    ///
    /// # Returns
    ///
    /// Time taken for warmup in milliseconds.
    #[cfg(feature = "cuda")]
    pub fn warmup(&mut self) -> WhisperResult<u64> {
        use trueno_gpu::driver::CudaStream;

        let start = std::time::Instant::now();

        // Step 1: Upload all weights
        self.upload_encoder_weights_to_gpu()?;
        self.upload_conv_weights_to_gpu()?;
        self.upload_decoder_weights_to_gpu()?;
        self.init_gpu_decoder_kv_cache_head_first()?;

        // Step 2: Run encoder to compile kernels - match actual transcribe_gpu path
        // WAPR-PERF-020: Check which encoder path will be used in transcribe_gpu
        let use_gpu_total_offload = std::env::var("WHISPER_GPU_TOTAL_OFFLOAD").is_ok();
        let use_gpu_encoder = std::env::var("WHISPER_GPU_ENCODER").is_ok();
        let use_gpu_decoder = std::env::var("WHISPER_GPU_DECODER_OFFLOAD").is_ok();

        let n_mels = self.config.n_mels as usize;
        let n_frames = 3000; // Whisper expects exactly 3000 frames for 30s audio
        let d_model = self.config.n_text_state as usize;
        let dummy_mel: Vec<f32> = vec![0.0; n_mels * n_frames];

        let enc_output = if use_gpu_total_offload {
            // Full GPU encoder path - compile all encoder kernels
            self.encode_gpu_total_offload(&dummy_mel)?
        } else if use_gpu_encoder {
            // Partial GPU encoder path - compile attention kernels
            self.encode_gpu(&dummy_mel)?
        } else {
            // CPU encoder path - no GPU kernels to compile for encoder
            // Use a small mel to avoid wasting time on CPU encoder
            let small_mel: Vec<f32> = vec![0.0; n_mels * 100]; // 100 frames
            self.encoder.forward_mel(&small_mel)?
        };
        let enc_seq_len = enc_output.len() / d_model;

        // Only warm up decoder kernels if GPU decoder will be used
        if !use_gpu_decoder {
            return Ok(start.elapsed().as_millis() as u64);
        }

        // Step 3: Get context and stream for GPU operations
        let ctx = self.executor.context();
        let stream = CudaStream::new(ctx)
            .map_err(|e| WhisperError::Inference(format!("warmup stream: {e}")))?;

        // Step 4: Upload encoder output to GPU for cross-attention warmup
        let enc_gpu = GpuResidentTensor::from_host(ctx, &enc_output)
            .map_err(|e| WhisperError::Inference(format!("warmup enc upload: {e}")))?;

        // Step 5: Populate cross K/V to compile permute kernels
        self.populate_cross_kv_caches_gpu(&enc_gpu, &stream)?;

        // Step 4: Run decoder to compile decoder kernels
        let dummy_embedding: Vec<f32> = vec![0.0; d_model];
        let _dec_out =
            self.forward_decoder_token_gpu_stream(&dummy_embedding, 0, &stream, Some(enc_seq_len))?;
        stream
            .synchronize()
            .map_err(|e| WhisperError::Inference(format!("warmup sync: {e}")))?;

        // Step 5: Reset decoder state for clean subsequent runs
        self.reset_gpu_decoder_kv_cache();
        self.init_gpu_decoder_kv_cache_head_first()?;

        Ok(start.elapsed().as_millis() as u64)
    }

    /// WAPR-PERF-018: GPU-resident encoder output for graph-captured cross-attention
    ///
    /// Same as `encode_gpu_total_offload` but returns GpuResidentTensor instead of Vec<f32>.
    /// Enables decoder cross-attention to stay on GPU without D2H→H2D transfer.
    ///
    /// # Returns
    ///
    /// - `GpuResidentTensor<f32>` - Encoder output on GPU [seq_len, d_model]
    #[cfg(feature = "cuda")]
    pub fn encode_gpu_resident(&mut self, mel: &[f32]) -> WhisperResult<GpuResidentTensor<f32>> {
        // Ensure weights are uploaded
        if self.gpu_encoder_weights.is_none() {
            self.upload_encoder_weights_to_gpu()?;
        }
        if self.gpu_conv_weights.is_none() {
            self.upload_conv_weights_to_gpu()?;
        }

        let ctx = self.executor.context();
        let d_model = self.config.n_audio_state as usize;
        let n_layers = self.config.n_audio_layer as usize;
        let n_mels = self.config.n_mels as usize;

        // Step 1: Convolutional frontend on GPU
        let seq_len_in = mel.len() / n_mels;
        let mel_gpu = GpuResidentTensor::from_host(ctx, mel)
            .map_err(|e| WhisperError::Inference(format!("mel upload: {e}")))?;

        let conv_weights = self
            .gpu_conv_weights
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("Conv weights not uploaded".into()))?;

        // Conv1: 80 → d_model, kernel=3, stride=1, padding=1 + GELU
        let conv1_out = mel_gpu
            .conv1d(
                ctx,
                &conv_weights.conv1_weight,
                Some(&conv_weights.conv1_bias),
                n_mels as u32,
                d_model as u32,
                3,
                1,
                1,
                seq_len_in as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("conv1: {e}")))?;

        // Conv2: d_model → d_model, kernel=3, stride=2, padding=1 + GELU
        let conv2_out = conv1_out
            .conv1d(
                ctx,
                &conv_weights.conv2_weight,
                Some(&conv_weights.conv2_bias),
                d_model as u32,
                d_model as u32,
                3,
                2,
                1,
                seq_len_in as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("conv2: {e}")))?;

        // After conv2: seq_len halved (stride=2)
        let seq_len = (seq_len_in + 2 - 3) / 2 + 1;
        let pos_emb = self.encoder.positional_embedding();
        let pos_slice = &pos_emb[..seq_len * d_model];
        let pos_gpu = GpuResidentTensor::from_host(ctx, pos_slice)
            .map_err(|e| WhisperError::Inference(format!("pos_emb upload: {e}")))?;

        let mut x_gpu = conv2_out
            .add(ctx, &pos_gpu)
            .map_err(|e| WhisperError::Inference(format!("pos_emb add: {e}")))?;

        // Step 3: Process encoder blocks on GPU
        let weights = self
            .gpu_encoder_weights
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("GPU weights not uploaded".into()))?;
        let config = self
            .gpu_encoder_config
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("GPU config not set".into()))?;

        for layer_idx in 0..n_layers {
            x_gpu = forward_encoder_block_gpu(ctx, &x_gpu, &weights[layer_idx], config)
                .map_err(|e| WhisperError::Inference(format!("encoder block {layer_idx}: {e}")))?;
        }

        // Step 4: Final layer norm on GPU (WAPR-PERF-019: eliminates D2H→CPU→H2D round-trip)
        let ln_post_gamma = self
            .gpu_enc_ln_post_gamma
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("enc ln_post_gamma not uploaded".into()))?;
        let ln_post_beta = self
            .gpu_enc_ln_post_beta
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("enc ln_post_beta not uploaded".into()))?;

        let result_gpu = x_gpu
            .layer_norm(
                ctx,
                ln_post_gamma,
                ln_post_beta,
                d_model as u32,
                seq_len as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("encoder ln_post: {e}")))?;

        Ok(result_gpu)
    }

    /// WAPR-PERF-018: Populate cross-attention K/V caches from encoder output
    ///
    /// Projects encoder output through cross-attention K/V weights and stores in
    /// head-first format for GPU cross-attention. Called once per sequence.
    ///
    /// # Arguments
    ///
    /// * `encoder_output_gpu` - GPU-resident encoder output [enc_len, d_model]
    /// * `stream` - CUDA stream for GPU operations
    ///
    /// # Layout
    ///
    /// Input: [enc_len, d_model] where d_model = n_heads * head_dim
    /// Cache: [n_heads, enc_len, head_dim] (head-first for incremental_attention_gpu)
    #[cfg(feature = "cuda")]
    pub fn populate_cross_kv_caches_gpu(
        &mut self,
        encoder_output_gpu: &GpuResidentTensor<f32>,
        stream: &trueno_gpu::driver::CudaStream,
    ) -> WhisperResult<()> {
        // Ensure decoder weights are uploaded
        if self.gpu_decoder_weights.is_none() {
            self.upload_decoder_weights_to_gpu()?;
        }
        // Ensure KV caches are initialized
        if self.gpu_cross_k_head_first.is_none() {
            self.init_gpu_decoder_kv_cache_head_first()?;
        }

        let ctx = self.executor.context();
        let d_model = self.config.n_text_state as usize;
        let n_heads = self.config.n_text_head as usize;
        let n_layers = self.config.n_text_layer as usize;
        let head_dim = d_model / n_heads;

        // Encoder sequence length from tensor size
        let enc_len = encoder_output_gpu.len() / d_model;

        // Get weights and caches
        let weights = self
            .gpu_decoder_weights
            .as_ref()
            .ok_or_else(|| WhisperError::Inference("Decoder weights not uploaded".into()))?;
        let cross_k_caches = self
            .gpu_cross_k_head_first
            .as_mut()
            .ok_or_else(|| WhisperError::Inference("Cross K cache not initialized".into()))?;
        let cross_v_caches = self
            .gpu_cross_v_head_first
            .as_mut()
            .ok_or_else(|| WhisperError::Inference("Cross V cache not initialized".into()))?;

        // For each layer, project encoder output and reshape to head-first
        for layer_idx in 0..n_layers {
            let layer_weights = &weights[layer_idx];

            // Project K: [enc_len, d_model] @ W_k^T -> [enc_len, d_model]
            let k_proj = encoder_output_gpu
                .linear(
                    ctx,
                    &layer_weights.cross_w_k,
                    Some(&layer_weights.cross_b_k),
                    enc_len as u32,
                    d_model as u32,
                    d_model as u32,
                )
                .map_err(|e| WhisperError::Inference(format!("cross K proj L{layer_idx}: {e}")))?;

            // Project V: [enc_len, d_model] @ W_v^T -> [enc_len, d_model]
            let v_proj = encoder_output_gpu
                .linear(
                    ctx,
                    &layer_weights.cross_w_v,
                    Some(&layer_weights.cross_b_v),
                    enc_len as u32,
                    d_model as u32,
                    d_model as u32,
                )
                .map_err(|e| WhisperError::Inference(format!("cross V proj L{layer_idx}: {e}")))?;

            // Reshape from [enc_len, d_model] to [n_heads, enc_len, head_dim] on GPU
            // Uses InterleavedToBatchedKernel for zero-copy permute
            let k_head_first = k_proj
                .interleaved_to_head_first(
                    ctx,
                    enc_len as u32,
                    n_heads as u32,
                    head_dim as u32,
                    stream,
                )
                .map_err(|e| {
                    WhisperError::Inference(format!("cross K permute L{layer_idx}: {e}"))
                })?;

            let v_head_first = v_proj
                .interleaved_to_head_first(
                    ctx,
                    enc_len as u32,
                    n_heads as u32,
                    head_dim as u32,
                    stream,
                )
                .map_err(|e| {
                    WhisperError::Inference(format!("cross V permute L{layer_idx}: {e}"))
                })?;

            // Store in caches (direct assignment, both are now GPU-resident)
            let cache_k = &mut cross_k_caches[layer_idx];
            let cache_v = &mut cross_v_caches[layer_idx];
            *cache_k = k_head_first;
            *cache_v = v_head_first;
        }

        // Sync stream to ensure all uploads complete
        stream
            .synchronize()
            .map_err(|e| WhisperError::Inference(format!("stream sync: {e}")))?;

        Ok(())
    }

    /// Forward pass through a single encoder block with GPU attention.
    ///
    /// Architecture: Pre-norm with residual connections
    /// x + Attention(LN(x)) then x + FFN(LN(x))
    fn forward_encoder_block_gpu(
        &mut self,
        layer_idx: usize,
        x: &[f32],
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> WhisperResult<Vec<f32>> {
        // Extract block data first to avoid borrow conflicts with self.attention_via_gemm
        let (_normed, q, k, v) = {
            let block = &self.encoder.blocks()[layer_idx];
            let normed = block.ln1.forward(x)?;
            let q = block.self_attn.w_q().forward(&normed, seq_len)?;
            let k = block.self_attn.w_k().forward(&normed, seq_len)?;
            let v = block.self_attn.w_v().forward(&normed, seq_len)?;
            (normed, q, k, v)
        };

        // GPU attention dispatch (WAPR-PERF-004 vs WAPR-PERF-005)
        // WHISPER_GPU_RESIDENT=1: GPU-resident path with minimal transfers
        // Otherwise: gemm-per-head path (higher transfer overhead)
        #[cfg(feature = "cuda")]
        let attn_output = {
            let use_gpu_resident = std::env::var("WHISPER_GPU_RESIDENT").is_ok();
            if use_gpu_resident {
                // New path: GPU-resident attention with trueno-gpu (WAPR-PERF-004)
                self.attention_gpu_resident(&q, &k, &v, seq_len, n_heads, head_dim)?
            } else {
                // Old path: gemm per head (WAPR-PERF-005)
                self.attention_via_gemm(&q, &k, &v, seq_len, n_heads, head_dim)?
            }
        };
        #[cfg(not(feature = "cuda"))]
        let attn_output = self.attention_via_gemm(&q, &k, &v, seq_len, n_heads, head_dim)?;

        // Output projection and residual (need block reference again)
        let (attn_proj, _normed2, ffn_out) = {
            let block = &self.encoder.blocks()[layer_idx];
            let attn_proj = block.self_attn.w_o().forward(&attn_output, seq_len)?;

            // Residual connection
            let residual: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

            // Pre-norm for FFN
            let normed2 = block.ln2.forward(&residual)?;

            // FFN (CPU)
            let ffn_out = block.ffn.forward(&normed2)?;

            (attn_proj, normed2, ffn_out)
        };

        // Final residual (compute from x and attn_proj, then add ffn_out)
        let mut residual: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();
        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }

    /// GPU attention using basic gemm primitives (WAPR-PERF-005).
    ///
    /// This is the "dumb but working" approach that bypasses the failing
    /// flash_attention_multi_head kernel. Uses basic matrix multiplication:
    ///
    /// For each head h:
    /// 1. scores = Q_h @ K_h^T  (gemm: [seq, head_dim] @ [head_dim, seq] = [seq, seq])
    /// 2. scores = scores / sqrt(head_dim)
    /// 3. attn_weights = softmax(scores)
    /// 4. output_h = attn_weights @ V_h  (gemm: [seq, seq] @ [seq, head_dim] = [seq, head_dim])
    ///
    /// Then concatenate heads.
    fn attention_via_gemm(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> WhisperResult<Vec<f32>> {
        let d_model = n_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Output buffer: [seq_len, d_model]
        let mut output = vec![0.0f32; seq_len * d_model];

        // Process each head
        for head in 0..n_heads {
            let head_offset = head * head_dim;

            // Extract Q_h, K_h, V_h for this head (strided access)
            let mut q_head = vec![0.0f32; seq_len * head_dim];
            let mut k_head = vec![0.0f32; seq_len * head_dim];
            let mut v_head = vec![0.0f32; seq_len * head_dim];

            for pos in 0..seq_len {
                for d in 0..head_dim {
                    q_head[pos * head_dim + d] = q[pos * d_model + head_offset + d];
                    k_head[pos * head_dim + d] = k[pos * d_model + head_offset + d];
                    v_head[pos * head_dim + d] = v[pos * d_model + head_offset + d];
                }
            }

            // Step 1: scores = Q_h @ K_h^T using GPU gemm
            // Q_h: [seq_len, head_dim], K_h^T: [head_dim, seq_len] -> scores: [seq_len, seq_len]
            let mut scores = vec![0.0f32; seq_len * seq_len];

            // Transpose K for K^T
            let mut k_t = vec![0.0f32; head_dim * seq_len];
            for i in 0..seq_len {
                for j in 0..head_dim {
                    k_t[j * seq_len + i] = k_head[i * head_dim + j];
                }
            }

            // GPU gemm: scores = Q_h @ K_h^T
            self.executor
                .gemm(
                    &q_head,
                    &k_t,
                    &mut scores,
                    seq_len as u32,  // M
                    seq_len as u32,  // N
                    head_dim as u32, // K
                )
                .map_err(|e| WhisperError::Inference(format!("GPU gemm (Q@K^T) failed: {e}")))?;

            // Step 2: Scale scores
            for s in &mut scores {
                *s *= scale;
            }

            // Step 3: Softmax (CPU - simple row-wise softmax)
            for row in 0..seq_len {
                let row_start = row * seq_len;
                let row_slice = &mut scores[row_start..row_start + seq_len];

                // Find max for numerical stability
                let max_val = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                // Exp and sum
                let mut sum = 0.0f32;
                for val in row_slice.iter_mut() {
                    *val = (*val - max_val).exp();
                    sum += *val;
                }

                // Normalize
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for val in row_slice.iter_mut() {
                    *val *= inv_sum;
                }
            }

            // Step 4: output_h = attn_weights @ V_h using GPU gemm
            // scores: [seq_len, seq_len], V_h: [seq_len, head_dim] -> output_h: [seq_len, head_dim]
            let mut output_head = vec![0.0f32; seq_len * head_dim];
            self.executor
                .gemm(
                    &scores,
                    &v_head,
                    &mut output_head,
                    seq_len as u32,  // M
                    head_dim as u32, // N
                    seq_len as u32,  // K
                )
                .map_err(|e| WhisperError::Inference(format!("GPU gemm (attn@V) failed: {e}")))?;

            // Copy output_h to output buffer at correct head offset
            for pos in 0..seq_len {
                for d in 0..head_dim {
                    output[pos * d_model + head_offset + d] = output_head[pos * head_dim + d];
                }
            }
        }

        Ok(output)
    }

    /// GPU-resident attention (WAPR-PERF-004)
    ///
    /// Uses trueno-gpu's `GpuResidentTensor` and `batched_multihead_attention`
    /// to compute attention with ZERO intermediate host↔device transfers.
    ///
    /// This is the high-performance path that eliminates the ~150 transfers
    /// per encoder pass that plague `attention_via_gemm`.
    ///
    /// # Performance
    ///
    /// - Old path (attention_via_gemm): ~150 H2D/D2H transfers per forward
    /// - New path (attention_gpu_resident): 3 H2D + 0 intermediate + 1 D2H
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor [seq_len * d_model]
    /// * `k` - Key tensor [seq_len * d_model]
    /// * `v` - Value tensor [seq_len * d_model]
    /// * `seq_len` - Sequence length
    /// * `n_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    ///
    /// # Returns
    ///
    /// Attention output [seq_len * d_model]
    #[cfg(feature = "cuda")]
    pub fn attention_gpu_resident(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> WhisperResult<Vec<f32>> {
        // Get CudaContext from executor
        let ctx = self.executor.context();

        // Reset transfer counters for monitoring
        reset_transfer_counters();

        // Upload Q, K, V to GPU (3 H2D transfers)
        let q_gpu = GpuResidentTensor::from_host(ctx, q)
            .map_err(|e| WhisperError::Inference(format!("Failed to upload Q: {e}")))?;
        let k_gpu = GpuResidentTensor::from_host(ctx, k)
            .map_err(|e| WhisperError::Inference(format!("Failed to upload K: {e}")))?;
        let v_gpu = GpuResidentTensor::from_host(ctx, v)
            .map_err(|e| WhisperError::Inference(format!("Failed to upload V: {e}")))?;

        // GPU-resident attention: ZERO intermediate transfers!
        let mut output_gpu = batched_multihead_attention(
            ctx,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            n_heads as u32,
            head_dim as u32,
            seq_len as u32,
        )
        .map_err(|e| WhisperError::Inference(format!("GPU attention failed: {e}")))?;

        // Download result (1 D2H transfer)
        let output = output_gpu.to_host().map_err(|e| {
            WhisperError::Inference(format!("Failed to download attention output: {e}"))
        })?;

        // Log transfer stats for debugging
        if std::env::var("WHISPER_DEBUG_GPU_RESIDENT").is_ok() {
            let stats = TransferStats::capture();
            eprintln!(
                "[GPU-RESIDENT] attention: {} H2D, {} D2H (expected: 3 H2D, 1 D2H)",
                stats.h2d_transfers, stats.d2h_transfers
            );
        }

        Ok(output)
    }

    /// Initialize GPU KV caches for decoder self-attention.
    ///
    /// This pre-allocates GPU memory for KV caches, enabling GPU-resident
    /// incremental attention without host-device transfers per token.
    ///
    /// # Architecture (WAPR-PERF-004)
    ///
    /// Whisper decoder has:
    /// - n_layer decoder blocks (4 for tiny)
    /// - n_head attention heads (6 for tiny)
    /// - head_dim = d_model / n_head (64 for tiny)
    /// - max_seq_len tokens (448 for Whisper)
    ///
    /// KV cache layout per layer: [n_head, max_len, head_dim]
    fn init_gpu_kv_cache(&mut self) -> WhisperResult<()> {
        if self.kv_cache_initialized {
            return Ok(());
        }

        let n_layers = self.config.n_text_layer as usize;
        let n_heads = self.config.n_text_head as usize;
        let head_dim = self.config.n_text_state as usize / n_heads;
        let max_len = self.config.n_text_ctx as usize; // 448 for Whisper

        // Initialize GPU KV cache via realizar
        self.executor
            .init_kv_cache_gpu(n_layers, n_heads, n_heads, head_dim, max_len)
            .map_err(|e| WhisperError::Inference(format!("Failed to init GPU KV cache: {e}")))?;

        self.kv_cache_initialized = true;
        Ok(())
    }

    /// Clear GPU KV caches for a new transcription.
    ///
    /// This resets the cache positions without deallocating GPU memory.
    pub fn clear_kv_cache(&mut self) {
        self.executor.reset_kv_cache_gpu();
    }

    /// GPU-accelerated forward pass for a single decoder token using flash_attention_cached.
    ///
    /// This uses realizar's `flash_attention_cached` which handles GPU buffer management
    /// internally, providing GPU-accelerated attention while keeping a simple CPU-side API.
    ///
    /// # WAPR-PERF-004: Performance
    ///
    /// - Self-attention uses GPU KV cache via `flash_attention_cached`
    /// - Cross-attention and FFN run on CPU with SIMD
    /// - Output projection on CPU (workaround for gemv bug)
    ///
    /// # Arguments
    ///
    /// * `token` - Input token ID
    /// * `encoder_output` - Encoder hidden states (for cross-attention)
    /// * `position` - Current position in sequence
    ///
    /// # Returns
    ///
    /// Logits over vocabulary (n_vocab)
    pub fn forward_one_gpu_resident(
        &mut self,
        token: u32,
        encoder_output: &[f32],
        position: usize,
    ) -> WhisperResult<Vec<f32>> {
        // Ensure weights and KV caches are ready
        if !self.weights_uploaded {
            self.upload_weights()?;
        }
        if !self.kv_cache_initialized {
            self.init_gpu_kv_cache()?;
        }

        let d_model = self.config.n_text_state as usize;
        let n_heads = self.config.n_text_head as usize;
        let head_dim = d_model / n_heads;
        let n_layers = self.config.n_text_layer as usize;

        // Step 1: Token embedding + positional embedding
        if token as usize >= self.config.n_vocab as usize {
            return Err(WhisperError::Model(format!(
                "token {} out of vocabulary range {}",
                token, self.config.n_vocab
            )));
        }
        let emb_start = (token as usize) * d_model;
        let mut x: Vec<f32> =
            self.decoder.token_embedding()[emb_start..emb_start + d_model].to_vec();

        // Add positional embedding
        let pos_start = position * d_model;
        for (x_elem, pos_emb) in x
            .iter_mut()
            .zip(&self.decoder.positional_embedding()[pos_start..pos_start + d_model])
        {
            *x_elem += pos_emb;
        }

        // Step 2: Process decoder blocks with GPU-accelerated self-attention
        for layer_idx in 0..n_layers {
            x = self.forward_block_gpu_flash(layer_idx, &x, encoder_output, n_heads, head_dim)?;
        }

        // Step 3: Final layer norm
        let x_normed = self.decoder.ln_post().forward(&x)?;

        // Step 4: Output projection to vocabulary (CPU path per WAPR-PERF-006)
        let logits = self.decoder.project_to_vocab_debug(&x_normed);

        Ok(logits)
    }

    /// Forward pass through a single decoder block.
    ///
    /// Attempts GPU-accelerated self-attention via flash_attention_cached,
    /// falls back to CPU attention if GPU fails.
    fn forward_block_gpu_flash(
        &mut self,
        layer_idx: usize,
        x: &[f32],
        encoder_output: &[f32],
        n_heads: usize,
        head_dim: usize,
    ) -> WhisperResult<Vec<f32>> {
        let d_model = n_heads * head_dim;
        let block = &self.decoder.blocks()[layer_idx];

        // === Self-attention ===
        // Pre-norm
        let normed = block.ln1.forward(x)?;

        // Q/K/V projections (CPU with SIMD)
        let q = block.self_attn.w_q().forward_simd(&normed, 1)?;
        let k = block.self_attn.w_k().forward_simd(&normed, 1)?;
        let v = block.self_attn.w_v().forward_simd(&normed, 1)?;

        // Try GPU-accelerated attention, fall back to CPU if it fails
        let mut attn_out = vec![0.0f32; d_model];
        let gpu_result = self
            .executor
            .flash_attention_cached(layer_idx, &q, &k, &v, &mut attn_out);

        // If GPU fails, use CPU attention (this is for robustness during development)
        match gpu_result {
            Ok(_seq_len) => {
                if std::env::var("WHISPER_DEBUG_GPU").is_ok() && layer_idx == 0 {
                    eprintln!("[GPU] flash_attention_cached SUCCESS layer={}", layer_idx);
                }
            }
            Err(e) => {
                if std::env::var("WHISPER_DEBUG_GPU").is_ok() {
                    eprintln!(
                        "[GPU] flash_attention_cached failed layer={} q.len={} k.len={} v.len={} d_model={}: {}",
                        layer_idx, q.len(), k.len(), v.len(), d_model, e
                    );
                }
                // CPU self-attention fallback
                attn_out = self.compute_self_attention(&q, &k, &v, n_heads, head_dim)?;
            }
        }

        // Output projection + residual
        let attn_proj = block.self_attn.w_o().forward_simd(&attn_out, 1)?;
        let mut residual: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

        // === Cross-attention (CPU with SIMD) ===
        let normed = block.ln2.forward(&residual)?;

        // Cross-attention Q projection
        let q_cross = block.cross_attn.w_q().forward_simd(&normed, 1)?;

        // Cross-attention K/V from encoder
        let enc_len = encoder_output.len() / d_model;
        let k_enc = block
            .cross_attn
            .w_k()
            .forward_simd(encoder_output, enc_len)?;
        let v_enc = block
            .cross_attn
            .w_v()
            .forward_simd(encoder_output, enc_len)?;

        // Compute cross-attention (CPU)
        let cross_attn_out =
            self.compute_cross_attention(&q_cross, &k_enc, &v_enc, n_heads, head_dim)?;

        let cross_proj = block.cross_attn.w_o().forward_simd(&cross_attn_out, 1)?;
        for (r, c) in residual.iter_mut().zip(cross_proj.iter()) {
            *r += c;
        }

        // === FFN (CPU with SIMD) ===
        let normed = block.ln3.forward(&residual)?;
        let fc1_out = block.ffn.fc1.forward_simd(&normed, 1)?;

        // GELU activation
        let gelu_out: Vec<f32> = fc1_out.iter().map(|&val| gelu(val)).collect();

        let fc2_out = block.ffn.fc2.forward_simd(&gelu_out, 1)?;
        for (r, f) in residual.iter_mut().zip(fc2_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }

    /// Compute cross-attention (query attends to encoder keys/values).
    fn compute_cross_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: usize,
        head_dim: usize,
    ) -> WhisperResult<Vec<f32>> {
        let d_model = n_heads * head_dim;
        let enc_len = k.len() / d_model;

        let mut output = vec![0.0f32; d_model];

        // Multi-head attention
        for h in 0..n_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            // Compute attention scores
            let mut scores = vec![0.0f32; enc_len];
            for pos in 0..enc_len {
                let k_head = &k[pos * d_model + h * head_dim..pos * d_model + (h + 1) * head_dim];
                scores[pos] = q_head
                    .iter()
                    .zip(k_head.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / (head_dim as f32).sqrt();
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let attn_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // Weighted sum of values
            for pos in 0..enc_len {
                let v_head = &v[pos * d_model + h * head_dim..pos * d_model + (h + 1) * head_dim];
                let weight = attn_weights[pos];
                for (i, &val) in v_head.iter().enumerate() {
                    output[h * head_dim + i] += weight * val;
                }
            }
        }

        Ok(output)
    }

    /// Compute self-attention (CPU fallback for incremental decoding).
    ///
    /// This is a simplified self-attention that only attends to the current K/V.
    /// In a proper incremental decoder, this would accumulate K/V history.
    fn compute_self_attention(
        &self,
        _q: &[f32],
        _k: &[f32],
        v: &[f32],
        n_heads: usize,
        head_dim: usize,
    ) -> WhisperResult<Vec<f32>> {
        let d_model = n_heads * head_dim;
        let mut output = vec![0.0f32; d_model];

        // For single-token incremental decoding with no history,
        // attention to self is just the value (attention weight = 1.0)
        // This is a simplified fallback - proper incremental attention
        // would accumulate K/V history
        for h in 0..n_heads {
            let v_head = &v[h * head_dim..(h + 1) * head_dim];
            for (i, &val) in v_head.iter().enumerate() {
                output[h * head_dim + i] = val;
            }
        }

        Ok(output)
    }

    /// Run encoder forward pass on GPU.
    ///
    /// Uses CudaExecutor for matrix multiplications when possible.
    /// Falls back to CPU for operations not yet GPU-accelerated.
    ///
    /// # Arguments
    ///
    /// * `mel` - Mel spectrogram features [n_mels * n_frames]
    ///
    /// # Returns
    ///
    /// Encoder hidden states [seq_len * n_state]
    pub fn encode_cuda(&mut self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        // Ensure weights are uploaded
        if !self.weights_uploaded {
            self.upload_weights()?;
        }

        // WAPR-PERF-004: GPU encoder forward pass
        // The encoder processes mel spectrogram through:
        // 1. Conv1d layers (CPU - complex kernel, not worth GPU overhead)
        // 2. Positional embedding (CPU - element-wise addition)
        // 3. Transformer blocks (GPU - heavy matmul operations)
        // 4. Final layer norm (CPU - reduction operation)
        //
        // For now, use CPU encoder as GPU wiring requires architecture changes.
        // The main bottleneck is the decoder, not the encoder.
        self.encoder.forward(mel)
    }

    /// Run decoder forward pass on GPU.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Input token IDs
    /// * `encoder_output` - Encoder hidden states
    ///
    /// # Returns
    ///
    /// Logits over vocabulary [vocab_size]
    pub fn decode_cuda(
        &mut self,
        tokens: &[u32],
        encoder_output: &[f32],
    ) -> WhisperResult<Vec<f32>> {
        // Ensure weights are uploaded
        if !self.weights_uploaded {
            self.upload_weights()?;
        }

        // GPU decoder forward tracked in WAPR-PERF-009
        // CPU decoder achieves target RTF; GPU optimization deferred
        self.decoder.forward(tokens, encoder_output)
    }

    /// Run full transcription on GPU.
    ///
    /// # Arguments
    ///
    /// * `mel` - Mel spectrogram features
    ///
    /// # Returns
    ///
    /// Transcribed token IDs
    pub fn transcribe_cuda(&mut self, mel: &[f32]) -> WhisperResult<Vec<u32>> {
        let encoder_output = self.encode_cuda(mel)?;

        // Simple greedy decode
        let mut tokens = vec![50258_u32]; // SOT token
        let max_tokens = 448;

        for _ in 0..max_tokens {
            let logits = self.decode_cuda(&tokens, &encoder_output)?;

            // Argmax for greedy decode
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(50257); // EOT fallback

            if next_token == 50257 {
                // EOT
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Transcribe audio samples using GPU acceleration.
    ///
    /// This is the main entry point for GPU-accelerated transcription.
    /// It converts audio to mel spectrogram, runs the encoder/decoder on GPU,
    /// and decodes tokens to text.
    ///
    /// # Arguments
    ///
    /// * `audio` - Mono audio samples at 16kHz, normalized to [-1, 1]
    /// * `options` - Transcription options (language, task, strategy)
    ///
    /// # Returns
    ///
    /// TranscriptionResult containing the transcribed text and optional timestamps.
    #[allow(unused_variables)]
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        options: TranscribeOptions,
    ) -> WhisperResult<TranscriptionResult> {
        // Whisper constants
        const N_SAMPLES_30S: usize = 480_000; // 30 seconds at 16kHz
        const N_FRAMES: usize = 3000; // Whisper expects exactly 3000 frames
        const N_MELS: usize = 80;

        // Pad/truncate audio to 30 seconds (same as WhisperApr::compute_mel)
        let padded_audio = match audio.len().cmp(&N_SAMPLES_30S) {
            std::cmp::Ordering::Equal => audio.to_vec(),
            std::cmp::Ordering::Less => {
                let mut padded = vec![0.0_f32; N_SAMPLES_30S];
                padded[..audio.len()].copy_from_slice(audio);
                padded
            }
            std::cmp::Ordering::Greater => audio[..N_SAMPLES_30S].to_vec(),
        };

        // Compute mel spectrogram
        let mut mel = self
            .mel_filters
            .compute(&padded_audio)
            .map_err(|e| WhisperError::Audio(e.to_string()))?;
        let actual_frames = mel.len() / N_MELS;

        // Ensure exactly 3000 frames (pad or truncate)
        if actual_frames < N_FRAMES {
            let pad_value = -1.0_f32;
            let mut padded_mel = vec![pad_value; N_FRAMES * N_MELS];
            padded_mel[..mel.len()].copy_from_slice(&mel);
            mel = padded_mel;
        } else if actual_frames > N_FRAMES {
            mel.truncate(N_FRAMES * N_MELS);
        }

        // Run encoder - use forward_mel which handles the conv frontend
        let encoder_output = self.encoder.forward_mel(&mel)?;

        // Build initial tokens based on task and language using SpecialTokens
        use crate::tokenizer::special_tokens::{self, SpecialTokens};

        let specials = SpecialTokens::for_vocab_size(self.config.n_vocab as usize);
        let mut tokens = vec![specials.sot];

        // Add language token for multilingual models
        if specials.is_multilingual {
            let language = options.language.as_deref().unwrap_or("en");
            let lang_offset = special_tokens::language_offset(language).unwrap_or(0);
            tokens.push(specials.lang_base + lang_offset);
        }

        // Add task token
        match options.task {
            Task::Transcribe => tokens.push(specials.transcribe),
            Task::Translate => tokens.push(special_tokens::TRANSLATE),
        }

        // Timestamp mode: do NOT push no_timestamps token.
        // This enables the decoder to produce <|t.tt|> timestamp tokens
        // which are needed for proper SRT/VTT segment timing.

        // Decode loop using incremental decoding with KV cache
        let max_tokens = self.config.n_text_ctx as usize;
        let d_model = self.config.n_text_state as usize;
        let n_layers = self.config.n_text_layer as usize;
        let n_vocab = self.config.n_vocab as usize;
        let eot_token = specials.eot;

        // Create KV cache for incremental decoding
        let mut cache = crate::model::DecoderKVCache::new(n_layers, d_model, max_tokens);

        // Create token suppressor — do NOT suppress timestamps
        let suppressor = crate::inference::WhisperTokenSuppressor::new()
            .with_timestamp_suppression(false)
            .with_vocab_size(n_vocab);

        // Process initial tokens to populate cache
        for &token in &tokens {
            let _ = self
                .decoder
                .forward_one(token, &encoder_output, &mut cache)?;
        }

        // Generate tokens
        for _ in 0..max_tokens.saturating_sub(tokens.len()) {
            // Get logits for last token
            let last_token = *tokens.last().unwrap_or(&specials.sot);
            let mut logits = self
                .decoder
                .forward_one(last_token, &encoder_output, &mut cache)?;

            // Apply token suppression
            suppressor.apply(&mut logits);

            // Get next token based on strategy
            let next_token = match options.strategy {
                DecodingStrategy::Greedy => logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(eot_token),
                DecodingStrategy::BeamSearch { .. } | DecodingStrategy::Sampling { .. } => {
                    // For now, fall back to greedy for beam search and sampling
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(eot_token)
                }
            };

            // Check for EOT
            if next_token == eot_token {
                break;
            }

            tokens.push(next_token);
        }

        // Decode tokens to text, skipping special tokens
        let text = self.tokenizer.decode_with_options(&tokens, true)?;

        // Get language from options or default to "en"
        let language = options.language.clone().unwrap_or_else(|| "en".to_string());

        Ok(TranscriptionResult {
            text,
            language,
            segments: Vec::new(), // No segments for now
            profiling: None,
        })
    }

    /// Get a reference to the underlying CudaExecutor.
    pub fn executor(&self) -> &CudaExecutor {
        &self.executor
    }

    /// Get a mutable reference to the underlying CudaExecutor.
    pub fn executor_mut(&mut self) -> &mut CudaExecutor {
        &mut self.executor
    }

    /// GPU-accelerated single token forward pass.
    ///
    /// Uses CPU for token embedding + transformer blocks, then GPU gemv for
    /// the output projection (51865 × 384 matmul - the main bottleneck).
    ///
    /// # Performance
    ///
    /// The output projection is O(n_vocab × d_model) = O(51865 × 384) ≈ 20M FLOPs.
    /// GPU gemv provides ~10-50x speedup over CPU for this operation.
    ///
    /// # Arguments
    ///
    /// * `token` - Input token ID
    /// * `encoder_output` - Encoder hidden states
    /// * `cache` - KV cache for incremental decoding
    ///
    /// # Returns
    ///
    /// Logits over vocabulary (n_vocab)
    pub fn forward_one_gpu(
        &mut self,
        token: u32,
        encoder_output: &[f32],
        cache: &mut DecoderKVCache,
    ) -> WhisperResult<Vec<f32>> {
        // Ensure weights are uploaded to GPU
        if !self.weights_uploaded {
            self.upload_weights()?;
        }

        // Use CPU decoder to get hidden state after ln_post
        // This runs: embedding → transformer blocks → layer norm
        let hidden = self
            .decoder
            .forward_one_hidden(token, encoder_output, cache)?;

        // Debug: print CPU decoder hidden state stats
        let pos = cache.seq_len().saturating_sub(1);
        if pos < 5 && std::env::var("WHISPER_DEBUG_GPU").is_ok() {
            let hidden_mean = hidden.iter().sum::<f32>() / hidden.len() as f32;
            let hidden_std = (hidden
                .iter()
                .map(|v| (v - hidden_mean).powi(2))
                .sum::<f32>()
                / hidden.len() as f32)
                .sqrt();
            eprintln!(
                "[DEBUG-CPU-HIDDEN] pos={} after_ln: len={} mean={:.4} std={:.4}",
                pos,
                hidden.len(),
                hidden_mean,
                hidden_std
            );
        }

        // Output projection on GPU using direct gemm
        // FIX 1 (WAPR-PERF-004): Use executor.gemm() which works correctly,
        // unlike the buggy gemv_cached (WAPR-PERF-006).
        let logits = self.project_to_vocab_gpu(&hidden)?;

        Ok(logits)
    }

    /// WAPR-PERF-013: Full GPU decoder forward pass for single token
    ///
    /// Uses GPU self-attention with head-first KV caches for all decoder blocks.
    /// This is the "Total Offload" path that minimizes host-device sync.
    ///
    /// # Architecture
    ///
    /// ```text
    /// token → embed (CPU) → decoder blocks (GPU self-attn) → ln_post (CPU) → logits (GPU)
    /// ```
    ///
    /// # Point 157 Compliance
    ///
    /// - GPU self-attention with head-first KV cache (no layout conversion)
    /// - Cross-attention on CPU (encoder K/V not GPU-resident yet)
    /// - Single sync point at the end (after all blocks)
    #[cfg(feature = "cuda")]
    pub fn forward_one_gpu_total_offload(
        &mut self,
        token: u32,
        encoder_output: &[f32],
    ) -> WhisperResult<Vec<f32>> {
        use trueno_gpu::driver::CudaStream;

        let d_model = self.config.n_text_state as usize;
        let n_vocab = self.config.n_vocab as usize;

        // Ensure GPU decoder infrastructure is initialized
        if self.gpu_decoder_weights.is_none() {
            self.upload_decoder_weights_to_gpu()?;
        }
        if self.gpu_self_k_head_first.is_none() {
            self.init_gpu_decoder_kv_cache_head_first()?;
        }

        let pos = self.gpu_decoder_pos;

        // 1. Embed token + positional embedding (CPU - fast)
        if token as usize >= n_vocab {
            return Err(WhisperError::Inference(format!(
                "token {} out of vocabulary range {}",
                token, n_vocab
            )));
        }

        let emb_start = (token as usize) * d_model;
        let token_emb = self.decoder.token_embedding();
        let pos_emb = self.decoder.positional_embedding();
        let max_len = self.config.n_text_ctx as usize;

        if pos >= max_len {
            return Err(WhisperError::Inference(format!(
                "position {} exceeds max {}",
                pos, max_len
            )));
        }

        let pos_start = pos * d_model;
        let token_embedding: Vec<f32> = token_emb[emb_start..emb_start + d_model]
            .iter()
            .zip(&pos_emb[pos_start..pos_start + d_model])
            .map(|(t, p)| t + p)
            .collect();

        // 2. Compute encoder sequence length for cross-attention
        let enc_seq_len = encoder_output.len() / d_model;

        // Debug: print encoder output stats on first token
        if pos == 0 {
            eprintln!(
                "[DEBUG-GPU-DEC] enc_output: len={} seq_len={} d_model={}",
                encoder_output.len(),
                enc_seq_len,
                d_model
            );
            let enc_mean = encoder_output.iter().sum::<f32>() / encoder_output.len() as f32;
            let enc_std = (encoder_output
                .iter()
                .map(|x| (x - enc_mean).powi(2))
                .sum::<f32>()
                / encoder_output.len() as f32)
                .sqrt();
            eprintln!(
                "[DEBUG-GPU-DEC] enc_output stats: mean={:.4} std={:.4}",
                enc_mean, enc_std
            );
        }

        // 3. Populate cross K/V on first token (pos=0) - encoder output changes per transcription
        // Cross K/V caches are pre-allocated with zeros but need actual encoder projections
        if pos == 0 {
            eprintln!("[DEBUG-GPU-DEC] Populating cross K/V caches (pos=0)...");
            let ctx = self.executor.context();
            let stream = CudaStream::new(ctx)
                .map_err(|e| WhisperError::Inference(format!("stream: {e}")))?;
            let enc_gpu = GpuResidentTensor::from_host(ctx, encoder_output)
                .map_err(|e| WhisperError::Inference(format!("enc upload: {e}")))?;
            self.populate_cross_kv_caches_gpu(&enc_gpu, &stream)?;
            stream
                .synchronize()
                .map_err(|e| WhisperError::Inference(format!("cross kv sync: {e}")))?;
            eprintln!("[DEBUG-GPU-DEC] Cross K/V populated successfully");
        }

        // 4. Create single stream for all decoder operations (WAPR-PERF-023)
        let ctx = self.executor.context();
        let stream = CudaStream::new(ctx)
            .map_err(|e| WhisperError::Inference(format!("decoder stream: {e}")))?;

        // 5. Run decoder using stream-based path (single stream, all GPU)
        let mut hidden_gpu = self.forward_decoder_token_gpu_stream(
            &token_embedding,
            pos,
            &stream,
            Some(enc_seq_len),
        )?;

        // 6. Download hidden state for final layer norm
        stream
            .synchronize()
            .map_err(|e| WhisperError::Inference(format!("stream sync: {e}")))?;
        let x = hidden_gpu
            .to_host()
            .map_err(|e| WhisperError::Inference(format!("hidden download: {e}")))?;

        // Debug: print hidden state stats
        if pos < 5 && std::env::var("WHISPER_DEBUG_GPU").is_ok() {
            let hidden_mean = x.iter().sum::<f32>() / x.len() as f32;
            let hidden_std =
                (x.iter().map(|v| (v - hidden_mean).powi(2)).sum::<f32>() / x.len() as f32).sqrt();
            eprintln!(
                "[DEBUG-GPU-HIDDEN] pos={} before_ln: len={} mean={:.4} std={:.4}",
                pos,
                x.len(),
                hidden_mean,
                hidden_std
            );
        }

        // 7. Final layer norm (CPU - simple)
        let hidden = self.decoder.ln_post().forward(&x)?;

        // Debug: print hidden state after ln
        if pos < 5 && std::env::var("WHISPER_DEBUG_GPU").is_ok() {
            let hidden_mean = hidden.iter().sum::<f32>() / hidden.len() as f32;
            let hidden_std = (hidden
                .iter()
                .map(|v| (v - hidden_mean).powi(2))
                .sum::<f32>()
                / hidden.len() as f32)
                .sqrt();
            eprintln!(
                "[DEBUG-GPU-HIDDEN] pos={} after_ln: len={} mean={:.4} std={:.4}",
                pos,
                hidden.len(),
                hidden_mean,
                hidden_std
            );
        }

        // 8. Increment position for next token
        self.gpu_decoder_pos += 1;

        // 9. Output projection on GPU
        self.project_to_vocab_gpu(&hidden)
    }

    /// WAPR-PERF-014: Executor-based single token forward pass
    ///
    /// Uses `forward_decoder_block_executor()` which uses the executor's
    /// persistent stream for GEMV operations, avoiding stream creation overhead.
    #[cfg(feature = "cuda")]
    pub fn forward_one_executor(
        &mut self,
        token: u32,
        encoder_output: &[f32],
    ) -> WhisperResult<Vec<f32>> {
        let d_model = self.config.n_text_state as usize;
        let n_layers = self.config.n_text_layer as usize;
        let n_vocab = self.config.n_vocab as usize;

        // Ensure GPU decoder infrastructure is initialized
        if self.gpu_decoder_weights.is_none() {
            self.upload_decoder_weights_to_gpu()?;
        }
        if self.gpu_self_k_head_first.is_none() {
            self.init_gpu_decoder_kv_cache_head_first()?;
        }
        // Also ensure executor weights are uploaded
        if self.executor.cached_weight_count() == 0 {
            self.upload_decoder_weights_to_executor()?;
        }

        let pos = self.gpu_decoder_pos;

        // 1. Embed token + positional embedding (CPU - fast)
        if token as usize >= n_vocab {
            return Err(WhisperError::Inference(format!(
                "token {} out of vocabulary range {}",
                token, n_vocab
            )));
        }

        let emb_start = (token as usize) * d_model;
        let token_emb = self.decoder.token_embedding();
        let pos_emb = self.decoder.positional_embedding();
        let max_len = self.config.n_text_ctx as usize;

        if pos >= max_len {
            return Err(WhisperError::Inference(format!(
                "position {} exceeds max {}",
                pos, max_len
            )));
        }

        let pos_start = pos * d_model;
        let mut x: Vec<f32> = token_emb[emb_start..emb_start + d_model]
            .iter()
            .zip(&pos_emb[pos_start..pos_start + d_model])
            .map(|(t, p)| t + p)
            .collect();

        // 2. Run through all decoder blocks (executor path)
        for layer_idx in 0..n_layers {
            x = self.forward_decoder_block_executor(layer_idx, &x, pos, Some(encoder_output))?;
        }

        // 3. Final layer norm (CPU - simple)
        let hidden = self.decoder.ln_post().forward(&x)?;

        // 4. Increment position for next token
        self.gpu_decoder_pos += 1;

        // 5. Output projection on GPU
        self.project_to_vocab_gpu(&hidden)
    }

    /// GPU-accelerated output projection using direct gemm.
    ///
    /// FIX 1 (WAPR-PERF-004): Use `executor.gemm()` directly instead of
    /// the buggy `gemv_cached`. This allocates fresh GPU buffers per call
    /// but produces correct results.
    ///
    /// # Arguments
    ///
    /// * `hidden` - Hidden state after final layer norm [d_model]
    ///
    /// # Returns
    ///
    /// Logits over vocabulary [n_vocab]
    pub fn project_to_vocab_gpu(&mut self, hidden: &[f32]) -> WhisperResult<Vec<f32>> {
        let n_vocab = self.config.n_vocab as usize;
        let d_model = self.config.n_text_state as usize;

        // Validate dimensions
        if hidden.len() != d_model {
            return Err(WhisperError::Inference(format!(
                "Hidden state dimension mismatch: got {}, expected {}",
                hidden.len(),
                d_model
            )));
        }

        let mut output = vec![0.0f32; n_vocab];

        // WAPR-PERF-014: Try cached weights first, fall back to direct gemm
        // GEMV: y[n] = W[n,k] @ x[k] where W = token_embedding [n_vocab × d_model]
        let k = d_model as u32;
        let n = n_vocab as u32;

        if self.executor.has_weights("whisper_output_proj") {
            // Fast path: use cached weights (persistent GPU buffer, no allocation)
            self.executor
                .gemv_cached("whisper_output_proj", hidden, &mut output, k, n)
                .map_err(|e| WhisperError::Inference(format!("GPU gemv_cached failed: {e}")))?;
        } else {
            // Fallback: allocate per-call (GPU path before executor weights uploaded)
            let weights = self.decoder.token_embedding();
            let m = n_vocab as u32;
            self.executor
                .gemm(weights, hidden, &mut output, m, 1, k)
                .map_err(|e| WhisperError::Inference(format!("GPU gemm failed: {e}")))?;
        }

        if std::env::var("WHISPER_DEBUG_GPU").is_ok() {
            let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let argmax = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            eprintln!(
                "[GPU] project_to_vocab_gpu: max={:.4} argmax={}",
                max_val, argmax
            );
        }

        Ok(output)
    }

    /// Transcribe using GPU-accelerated decoding.
    ///
    /// Uses SIMD-accelerated CPU decoder with proper KV caching.
    /// GPU acceleration is available for weights upload but self-attention
    /// runs on CPU due to CUDA kernel compatibility issues (WAPR-PERF-006).
    ///
    /// # WAPR-PERF-004: Performance Analysis
    ///
    /// Current approach:
    /// 1. CPU encoder (SIMD-accelerated via trueno)
    /// 2. CPU decoder with KV cache (proven correct)
    /// 3. CPU output projection (trueno matmul)
    ///
    /// GPU flash_attention_cached has kernel compatibility issues that need investigation.
    /// The CPU path with trueno SIMD is the reliable baseline.
    ///
    /// # APR-Style Tracing
    ///
    /// When tracing is enabled via `enable_tracing()`, this function emits:
    /// - `TraceStep::Embed`: Mel spectrogram computation
    /// - `TraceStep::TransformerBlock`: Encoder forward pass
    /// - `TraceStep::LmHead`: Output projection per token
    /// - `TraceStep::Sample`: Token sampling per token
    /// - `TraceStep::Decode`: Final detokenization
    pub fn transcribe_gpu(
        &mut self,
        audio: &[f32],
        options: TranscribeOptions,
    ) -> WhisperResult<TranscriptionResult> {
        const N_SAMPLES_30S: usize = 480_000; // 30 seconds at 16kHz

        // Use chunked processing for audio longer than 30 seconds
        if audio.len() > N_SAMPLES_30S {
            return self.transcribe_gpu_chunked(audio, options);
        }

        self.transcribe_gpu_single_chunk(audio, &options)
    }

    /// Transcribe a single chunk of audio (<=30 seconds) on GPU.
    fn transcribe_gpu_single_chunk(
        &mut self,
        audio: &[f32],
        options: &TranscribeOptions,
    ) -> WhisperResult<TranscriptionResult> {
        const N_SAMPLES_30S: usize = 480_000;
        const N_FRAMES: usize = 3000;
        const N_MELS: usize = 80;

        let profile_all = std::env::var("WHISPER_PROFILE_DECODER").is_ok();
        let transcribe_start = std::time::Instant::now();

        // === TRACE: Mel spectrogram (mapped to EMBED step) ===
        self.tracer.start_step(TraceStep::Embed);

        let mel = self.prepare_mel(audio, N_SAMPLES_30S, N_FRAMES, N_MELS, profile_all)?;

        // Trace mel computation (token_count=N_FRAMES, hidden_dim=N_MELS)
        self.tracer.trace_embed(N_FRAMES, N_MELS, Some(&mel));

        // === TRACE: Encoder forward pass (mapped to TRANSFORMER_BLOCK) ===
        self.tracer.start_step(TraceStep::TransformerBlock);

        // Select and run encoder path
        let encoder_output = self.run_encoder(&mel)?;

        // Trace encoder output (layer_idx=0 for encoder, iteration=0 for prefill)
        let d_model = self.config.n_text_state as usize;
        let enc_seq_len = encoder_output.len() / d_model;
        self.tracer
            .trace_layer(0, 0, Some(&encoder_output), enc_seq_len, d_model);

        // Build initial tokens
        use crate::tokenizer::special_tokens::SpecialTokens;
        let specials = SpecialTokens::for_vocab_size(self.config.n_vocab as usize);
        let mut tokens = Self::build_initial_tokens(&specials, options);

        // Hybrid GPU path:
        // - CPU decoder blocks (GPU flash_attention_cached has kernel compatibility issues)
        // - GPU output projection via executor.gemm() (FIX 1 - working!)
        let max_tokens = self.config.n_text_ctx as usize;
        let n_layers = self.config.n_text_layer as usize;
        let n_vocab = self.config.n_vocab as usize;
        let eot_token = specials.eot;

        let mut cache = DecoderKVCache::new(n_layers, d_model, max_tokens);

        // Token suppressor — do NOT suppress timestamps (needed for segment extraction)
        let suppressor = crate::inference::WhisperTokenSuppressor::new()
            .with_timestamp_suppression(false)
            .with_vocab_size(n_vocab);

        // Process initial tokens (prefill)
        self.prefill_tokens(&tokens, &encoder_output, &mut cache)?;

        // Generate tokens
        let profile_decoder = std::env::var("WHISPER_PROFILE_DECODER").is_ok();
        let mut decoder_token_times: Vec<u128> = Vec::new();
        let decoder_start = std::time::Instant::now();
        for gen_idx in 0..max_tokens.saturating_sub(tokens.len()) {
            let token_start = std::time::Instant::now();
            let last_token = *tokens.last().unwrap_or(&specials.sot);

            // === TRACE: LM_HEAD (output projection) ===
            self.tracer.start_step(TraceStep::LmHead);

            let mut logits = self.forward_one_gpu(last_token, &encoder_output, &mut cache)?;

            // Trace output projection
            self.tracer.trace_lm_head(gen_idx, &logits, n_vocab);

            Self::debug_logits(profile_decoder, gen_idx, &logits);

            // === TRACE: SAMPLE (token selection) ===
            self.tracer.start_step(TraceStep::Sample);

            suppressor.apply(&mut logits);

            // All strategies currently use greedy argmax
            let next_token = crate::simd::argmax(&logits) as u32;

            // Trace sampling result (temperature=0.0 for greedy, top_k=1)
            self.tracer
                .trace_sample(gen_idx, &logits, next_token, 0.0, 1);

            // Track per-token timing
            if profile_decoder {
                decoder_token_times.push(token_start.elapsed().as_micros());
            }

            if next_token == eot_token {
                break;
            }

            tokens.push(next_token);
        }

        Self::print_decoder_profile(profile_decoder, &decoder_token_times, &decoder_start);

        // === TRACE: DECODE (detokenization) ===
        self.tracer.start_step(TraceStep::Decode);

        let text = self.tokenizer.decode_with_options(&tokens, true)?;

        // Trace decode result (iteration=0 for final decode, last token, vocab_size)
        let last_token = tokens.last().copied().unwrap_or(0);
        self.tracer.trace_decode(0, last_token, &text, n_vocab);

        let language = options.language.clone().unwrap_or_else(|| "en".to_string());

        if profile_all {
            eprintln!(
                "[PROFILE-TRANSCRIBE] Total transcribe_gpu: {:.1}ms",
                transcribe_start.elapsed().as_millis()
            );
        }

        let segments =
            Self::build_segments(&tokens, &text, audio.len(), &self.tokenizer);

        Ok(TranscriptionResult {
            text,
            language,
            segments,
            profiling: None,
        })
    }

    /// Transcribe long audio using chunked streaming on GPU.
    ///
    /// Splits audio into 30-second chunks, transcribes each independently,
    /// then merges results with timestamp adjustment.
    fn transcribe_gpu_chunked(
        &mut self,
        audio: &[f32],
        options: TranscribeOptions,
    ) -> WhisperResult<TranscriptionResult> {
        const CHUNK_SAMPLES: usize = 480_000; // 30 seconds at 16kHz

        let language = options.language.clone().unwrap_or_else(|| "en".to_string());
        let mut all_segments: Vec<crate::Segment> = Vec::new();
        let mut all_text = String::new();

        let mut offset = 0;
        while offset < audio.len() {
            let chunk_end = (offset + CHUNK_SAMPLES).min(audio.len());
            let chunk = &audio[offset..chunk_end];

            // Skip very short final chunks (less than 0.5 seconds)
            if chunk.len() < crate::audio::SAMPLE_RATE as usize / 2 {
                break;
            }

            let chunk_options = TranscribeOptions {
                language: Some(language.clone()),
                task: options.task,
                strategy: options.strategy,
                word_timestamps: options.word_timestamps,
                profile: options.profile,
                prompt: options.prompt.clone(),
                hotwords: options.hotwords.clone(),
            };

            let chunk_result = self.transcribe_gpu_single_chunk(chunk, &chunk_options)?;
            let time_offset = offset as f32 / crate::audio::SAMPLE_RATE as f32;

            // Append text
            if !chunk_result.text.trim().is_empty() {
                if !all_text.is_empty() {
                    all_text.push(' ');
                }
                all_text.push_str(&chunk_result.text);
            }

            // Adjust segment timestamps and collect
            for mut seg in chunk_result.segments {
                seg.start += time_offset;
                seg.end += time_offset;
                all_segments.push(seg);
            }

            offset += CHUNK_SAMPLES;
        }

        // Split any remaining long segments at sentence boundaries
        let final_segments =
            crate::timestamps::split_long_segments(&all_segments, 10.0);

        Ok(TranscriptionResult {
            text: all_text,
            language,
            segments: final_segments,
            profiling: None,
        })
    }

    /// Select and run the encoder path based on environment variables.
    fn run_encoder(&mut self, mel: &[f32]) -> WhisperResult<Vec<f32>> {
        #[cfg(feature = "cuda")]
        {
            let use_gpu_total_offload = std::env::var("WHISPER_GPU_TOTAL_OFFLOAD").is_ok();
            let use_gpu_encoder = std::env::var("WHISPER_GPU_ENCODER").is_ok();
            if use_gpu_total_offload {
                eprintln!("[WAPR-PERF-014] Using GPU total-offload encoder...");
                return self.encode_gpu_total_offload(mel);
            } else if use_gpu_encoder {
                eprintln!("[WAPR-PERF-005] Using GPU attention-only encoder...");
                return self.encode_gpu(mel);
            }
        }
        self.encoder.forward_mel(mel)
    }

    /// Run decoder prefill on initial tokens.
    fn prefill_tokens(
        &mut self,
        tokens: &[u32],
        encoder_output: &[f32],
        cache: &mut DecoderKVCache,
    ) -> WhisperResult<()> {
        let prefill_start = std::time::Instant::now();
        for &token in tokens {
            let _ = self
                .decoder
                .forward_one(token, encoder_output, cache)?;
        }
        if std::env::var("WHISPER_PROFILE_DECODER").is_ok() {
            let prefill_time = prefill_start.elapsed();
            eprintln!(
                "[PROFILE-PREFILL] {} tokens in {:.1}ms ({:.1}ms/token)",
                tokens.len(),
                prefill_time.as_millis(),
                prefill_time.as_millis() as f64 / tokens.len() as f64
            );
        }
        Ok(())
    }

    /// Build the initial token sequence for decoder prefill.
    fn build_initial_tokens(
        specials: &crate::tokenizer::special_tokens::SpecialTokens,
        options: &TranscribeOptions,
    ) -> Vec<u32> {
        use crate::tokenizer::special_tokens;
        let mut tokens = vec![specials.sot];
        if specials.is_multilingual {
            let language = options.language.as_deref().unwrap_or("en");
            let lang_offset = special_tokens::language_offset(language).unwrap_or(0);
            tokens.push(specials.lang_base + lang_offset);
        }
        match options.task {
            Task::Transcribe => tokens.push(specials.transcribe),
            Task::Translate => tokens.push(special_tokens::TRANSLATE),
        }
        // Timestamp mode: do NOT push no_timestamps token.
        // This enables the decoder to produce <|t.tt|> timestamp tokens
        // which are needed for proper SRT/VTT segment timing.
        tokens
    }

    /// Prepare mel spectrogram from audio (pad/truncate + compute + frame normalization).
    fn prepare_mel(
        &self,
        audio: &[f32],
        n_samples_30s: usize,
        n_frames: usize,
        n_mels: usize,
        profile: bool,
    ) -> WhisperResult<Vec<f32>> {
        let mel_start = std::time::Instant::now();
        let padded_audio = match audio.len().cmp(&n_samples_30s) {
            std::cmp::Ordering::Equal => audio.to_vec(),
            std::cmp::Ordering::Less => {
                let mut padded = vec![0.0_f32; n_samples_30s];
                padded[..audio.len()].copy_from_slice(audio);
                padded
            }
            std::cmp::Ordering::Greater => audio[..n_samples_30s].to_vec(),
        };
        let mut mel = self
            .mel_filters
            .compute(&padded_audio)
            .map_err(|e| WhisperError::Audio(e.to_string()))?;
        if profile {
            eprintln!(
                "[PROFILE-MEL] Mel spectrogram: {:.1}ms",
                mel_start.elapsed().as_millis()
            );
        }
        let actual_frames = mel.len() / n_mels;
        if actual_frames < n_frames {
            let mut padded_mel = vec![-1.0_f32; n_frames * n_mels];
            padded_mel[..mel.len()].copy_from_slice(&mel);
            mel = padded_mel;
        } else if actual_frames > n_frames {
            mel.truncate(n_frames * n_mels);
        }
        Ok(mel)
    }

    /// Print debug logits stats for early generation steps.
    fn debug_logits(enabled: bool, gen_idx: usize, logits: &[f32]) {
        if enabled && gen_idx < 3 {
            let logits_mean = logits.iter().sum::<f32>() / logits.len() as f32;
            let logits_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let argmax = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            eprintln!(
                "[DEBUG-LOGITS] gen_idx={gen_idx} len={} mean={logits_mean:.4} max={logits_max:.4} argmax={argmax}",
                logits.len(),
            );
        }
    }

    /// Print decoder profiling summary if enabled.
    fn print_decoder_profile(
        enabled: bool,
        token_times: &[u128],
        decoder_start: &std::time::Instant,
    ) {
        if enabled && !token_times.is_empty() {
            let decoder_total = decoder_start.elapsed();
            let sum: u128 = token_times.iter().sum();
            let avg = sum as f64 / token_times.len() as f64;
            let max = token_times.iter().max().copied().unwrap_or(0);
            let min = token_times.iter().min().copied().unwrap_or(0);
            eprintln!(
                "[PROFILE-DECODER] {} tokens, total {:.1}ms, avg {:.1}ms, min {:.1}ms, max {:.1}ms",
                token_times.len(),
                decoder_total.as_millis(),
                avg / 1000.0,
                min as f64 / 1000.0,
                max as f64 / 1000.0
            );
        }
    }

    /// Build segments from decoded tokens (mirrors CPU path).
    fn build_segments(
        tokens: &[u32],
        text: &str,
        audio_len: usize,
        tokenizer: &BpeTokenizer,
    ) -> Vec<crate::Segment> {
        if crate::timestamps::has_timestamps(tokens) {
            crate::timestamps::extract_segments(tokens, |ts| tokenizer.decode(ts).ok())
        } else if !text.trim().is_empty() {
            let duration = audio_len as f32 / crate::audio::SAMPLE_RATE as f32;
            let single = vec![crate::Segment {
                start: 0.0,
                end: duration,
                text: text.to_string(),
                tokens: tokens.to_vec(),
            }];
            crate::timestamps::split_long_segments(&single, 10.0)
        } else {
            Vec::new()
        }
    }

    /// Print trace summary to stderr.
    ///
    /// Shows timing breakdown by step, total duration, and any detected anomalies.
    /// Call this after `transcribe_gpu()` to see performance analysis.
    pub fn print_trace_summary(&self) {
        if !self.tracer.is_enabled() {
            eprintln!(
                "[TRACE] Tracing not enabled. Call enable_tracing(TraceConfig::enabled()) first."
            );
            return;
        }

        let events = self.tracer.events();
        if events.is_empty() {
            eprintln!("[TRACE] No events collected.");
            return;
        }

        // Compute totals by step
        let mut step_durations: std::collections::HashMap<&'static str, u64> =
            std::collections::HashMap::new();
        let mut step_counts: std::collections::HashMap<&'static str, usize> =
            std::collections::HashMap::new();

        for event in events {
            *step_durations.entry(event.step.name()).or_insert(0) += event.duration_us;
            *step_counts.entry(event.step.name()).or_insert(0) += 1;
        }

        let total_us: u64 = step_durations.values().sum();
        let total_ms = total_us as f64 / 1000.0;

        eprintln!("=== APR-Style Inference Trace Summary ===");
        eprintln!("Total: {:.2}ms ({} events)", total_ms, events.len());
        eprintln!();
        eprintln!(
            "{:20} {:>8} {:>8} {:>8}",
            "STEP", "COUNT", "TIME(ms)", "PCT"
        );
        eprintln!("{:-<20} {:->8} {:->8} {:->8}", "", "", "", "");

        // Sort by duration descending
        let mut steps: Vec<_> = step_durations.iter().collect();
        steps.sort_by(|a, b| b.1.cmp(a.1));

        for (step, us) in steps {
            let count = step_counts.get(step).unwrap_or(&0);
            let ms = *us as f64 / 1000.0;
            let pct = if total_us > 0 {
                (*us as f64 / total_us as f64) * 100.0
            } else {
                0.0
            };
            eprintln!("{:20} {:>8} {:>8.2} {:>7.1}%", step, count, ms, pct);
        }
        eprintln!();
    }

    // ========================================================================
    // WAPR-PERF-014: Stream-Optimized Decoder (CudaExecutor-based)
    // ========================================================================
    //
    // Root cause of 10x slowdown: GpuResidentTensor creates new CUDA stream
    // per operation (~40 streams per token). Fix: use CudaExecutor's persistent
    // compute_stream for all operations.

    /// WAPR-PERF-014: Upload decoder weights to CudaExecutor's weight_cache
    ///
    /// Unlike `upload_decoder_weights_to_gpu()` which stores in GpuResidentTensor,
    /// this uploads to executor's weight_cache for use with `gemm_cached_async()`.
    ///
    /// # Naming Convention
    ///
    /// Weights are cached with names: `dec.L{layer}.{component}`
    /// - `dec.L0.self_w_q` - Self-attention Q projection
    /// - `dec.L0.ffn_fc1` - FFN first layer
    #[cfg(feature = "cuda")]
    pub fn upload_decoder_weights_to_executor(&mut self) -> WhisperResult<usize> {
        let d_model = self.config.n_text_state as usize;
        let n_layers = self.config.n_text_layer as usize;
        let d_ff = d_model * 4;
        let mut total_bytes = 0;

        // Helper to transpose weight matrix from [rows, cols] to [cols, rows]
        fn transpose_weights(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            let mut transposed = vec![0.0_f32; weights.len()];
            for r in 0..rows {
                for c in 0..cols {
                    transposed[c * rows + r] = weights[r * cols + c];
                }
            }
            transposed
        }

        for layer_idx in 0..n_layers {
            let block = &self.decoder.blocks()[layer_idx];

            // Self-attention Q/K/V/O (transposed for GPU: [in, out])
            let w_q_t = transpose_weights(&block.self_attn.w_q().weight, d_model, d_model);
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.self_w_q"), &w_q_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_q L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(
                    &format!("dec.L{layer_idx}.self_b_q"),
                    &block.self_attn.w_q().bias,
                )
                .map_err(|e| WhisperError::Inference(format!("dec self_b_q L{layer_idx}: {e}")))?;

            let w_k_t = transpose_weights(&block.self_attn.w_k().weight, d_model, d_model);
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.self_w_k"), &w_k_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_k L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(
                    &format!("dec.L{layer_idx}.self_b_k"),
                    &block.self_attn.w_k().bias,
                )
                .map_err(|e| WhisperError::Inference(format!("dec self_b_k L{layer_idx}: {e}")))?;

            let w_v_t = transpose_weights(&block.self_attn.w_v().weight, d_model, d_model);
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.self_w_v"), &w_v_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_v L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(
                    &format!("dec.L{layer_idx}.self_b_v"),
                    &block.self_attn.w_v().bias,
                )
                .map_err(|e| WhisperError::Inference(format!("dec self_b_v L{layer_idx}: {e}")))?;

            let w_o_t = transpose_weights(&block.self_attn.w_o().weight, d_model, d_model);
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.self_w_o"), &w_o_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_o L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(
                    &format!("dec.L{layer_idx}.self_b_o"),
                    &block.self_attn.w_o().bias,
                )
                .map_err(|e| WhisperError::Inference(format!("dec self_b_o L{layer_idx}: {e}")))?;

            // Cross-attention Q/K/V/O
            let cross_w_q_t = transpose_weights(&block.cross_attn.w_q().weight, d_model, d_model);
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.cross_w_q"), &cross_w_q_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_q L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(
                    &format!("dec.L{layer_idx}.cross_b_q"),
                    &block.cross_attn.w_q().bias,
                )
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_q L{layer_idx}: {e}")))?;

            let cross_w_k_t = transpose_weights(&block.cross_attn.w_k().weight, d_model, d_model);
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.cross_w_k"), &cross_w_k_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_k L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(
                    &format!("dec.L{layer_idx}.cross_b_k"),
                    &block.cross_attn.w_k().bias,
                )
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_k L{layer_idx}: {e}")))?;

            let cross_w_v_t = transpose_weights(&block.cross_attn.w_v().weight, d_model, d_model);
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.cross_w_v"), &cross_w_v_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_v L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(
                    &format!("dec.L{layer_idx}.cross_b_v"),
                    &block.cross_attn.w_v().bias,
                )
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_v L{layer_idx}: {e}")))?;

            let cross_w_o_t = transpose_weights(&block.cross_attn.w_o().weight, d_model, d_model);
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.cross_w_o"), &cross_w_o_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_o L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(
                    &format!("dec.L{layer_idx}.cross_b_o"),
                    &block.cross_attn.w_o().bias,
                )
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_o L{layer_idx}: {e}")))?;

            // FFN weights (fc1: d_model -> d_ff, fc2: d_ff -> d_model)
            let fc1_t = transpose_weights(&block.ffn.fc1.weight, d_ff, d_model);
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.ffn_fc1"), &fc1_t)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_fc1 L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.ffn_b1"), &block.ffn.fc1.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_b1 L{layer_idx}: {e}")))?;

            let fc2_t = transpose_weights(&block.ffn.fc2.weight, d_model, d_ff);
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.ffn_fc2"), &fc2_t)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_fc2 L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.ffn_b2"), &block.ffn.fc2.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_b2 L{layer_idx}: {e}")))?;

            // LayerNorm weights (gamma/beta)
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.ln1_gamma"), &block.ln1.weight)
                .map_err(|e| WhisperError::Inference(format!("dec ln1_gamma L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.ln1_beta"), &block.ln1.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ln1_beta L{layer_idx}: {e}")))?;

            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.ln2_gamma"), &block.ln2.weight)
                .map_err(|e| WhisperError::Inference(format!("dec ln2_gamma L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.ln2_beta"), &block.ln2.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ln2_beta L{layer_idx}: {e}")))?;

            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.ln3_gamma"), &block.ln3.weight)
                .map_err(|e| WhisperError::Inference(format!("dec ln3_gamma L{layer_idx}: {e}")))?;
            total_bytes += self
                .executor
                .load_weights(&format!("dec.L{layer_idx}.ln3_beta"), &block.ln3.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ln3_beta L{layer_idx}: {e}")))?;
        }

        // Output projection weights (token embedding)
        // WAPR-PERF-014: Token embedding is [n_vocab, d_model] but GEMV kernel expects [k, n]
        // where k=d_model (input) and n=n_vocab (output), so transpose to [d_model, n_vocab]
        let n_vocab = self.config.n_vocab as usize;
        let token_emb = self.decoder.token_embedding();
        let token_emb_t = transpose_weights(token_emb, n_vocab, d_model);
        total_bytes += self
            .executor
            .load_weights("dec.output_proj", &token_emb_t)
            .map_err(|e| WhisperError::Inference(format!("dec output_proj: {e}")))?;

        // Final layer norm
        let ln_post = self.decoder.ln_post();
        total_bytes += self
            .executor
            .load_weights("dec.ln_post_gamma", &ln_post.weight)
            .map_err(|e| WhisperError::Inference(format!("dec ln_post_gamma: {e}")))?;
        total_bytes += self
            .executor
            .load_weights("dec.ln_post_beta", &ln_post.bias)
            .map_err(|e| WhisperError::Inference(format!("dec ln_post_beta: {e}")))?;

        if std::env::var("WHISPER_DEBUG_GPU").is_ok() {
            eprintln!(
                "[WAPR-PERF-014] Uploaded {} decoder weight tensors ({:.2} MB) to executor",
                self.executor.cached_weight_count(),
                total_bytes as f64 / 1_048_576.0
            );
        }

        Ok(total_bytes)
    }

    /// WAPR-PERF-014: Executor-based decoder block forward pass
    ///
    /// Uses CudaExecutor's persistent stream for all GEMV operations, avoiding
    /// the ~40 stream creations per token that caused 10x slowdown.
    ///
    /// # Key Optimizations
    ///
    /// 1. Uses `executor.gemv_cached()` with pre-uploaded weights (persistent stream)
    /// 2. Minimizes H2D/D2H transfers (only input/output, not per-projection)
    /// 3. Keeps LayerNorm on CPU (fast enough, avoids gamma/beta upload overhead)
    ///
    /// # Parameters
    ///
    /// - `layer_idx`: Decoder layer index
    /// - `x`: Input hidden state [d_model]
    /// - `pos`: Current position in sequence
    /// - `encoder_output`: Optional encoder hidden states for cross-attention
    #[cfg(feature = "cuda")]
    pub fn forward_decoder_block_executor(
        &mut self,
        layer_idx: usize,
        x: &[f32],
        pos: usize,
        encoder_output: Option<&[f32]>,
    ) -> WhisperResult<Vec<f32>> {
        use trueno_gpu::driver::CudaStream;

        let d_model = self.config.n_text_state as usize;
        let n_heads = self.config.n_text_head as usize;
        let head_dim = d_model / n_heads;
        let max_seq_len = self.config.n_text_ctx as usize;

        // Copy biases first (before any mutable borrows)
        let block = &self.decoder.blocks()[layer_idx];
        let b_q = block.self_attn.w_q().bias.clone();
        let b_k = block.self_attn.w_k().bias.clone();
        let b_v = block.self_attn.w_v().bias.clone();
        let b_o = block.self_attn.w_o().bias.clone();

        // LN1 (CPU - simple and correct)
        let normed = block.ln1.forward(x)?;

        // === Q/K/V projections via executor (persistent stream, no new streams!) ===
        let mut q = vec![0.0f32; d_model];
        let mut k = vec![0.0f32; d_model];
        let mut v = vec![0.0f32; d_model];

        self.executor
            .gemv_cached(
                &format!("dec.L{layer_idx}.self_w_q"),
                &normed,
                &mut q,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("Q projection: {e}")))?;

        self.executor
            .gemv_cached(
                &format!("dec.L{layer_idx}.self_w_k"),
                &normed,
                &mut k,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("K projection: {e}")))?;

        self.executor
            .gemv_cached(
                &format!("dec.L{layer_idx}.self_w_v"),
                &normed,
                &mut v,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("V projection: {e}")))?;

        // Add biases (CPU - fast)
        for i in 0..d_model {
            q[i] += b_q[i];
            k[i] += b_k[i];
            v[i] += b_v[i];
        }

        // Now get context for GPU tensor operations (after gemv_cached calls)
        let ctx = self.executor.context();

        // Get head-first KV caches
        let self_k_caches = self
            .gpu_self_k_head_first
            .as_mut()
            .ok_or_else(|| WhisperError::Inference("Self K cache not initialized".into()))?;
        let self_v_caches = self
            .gpu_self_v_head_first
            .as_mut()
            .ok_or_else(|| WhisperError::Inference("Self V cache not initialized".into()))?;

        // Upload Q/K/V for KV cache scatter + attention
        let q_gpu = GpuResidentTensor::from_host(ctx, &q)
            .map_err(|e| WhisperError::Inference(format!("Q upload: {e}")))?;
        let k_gpu = GpuResidentTensor::from_host(ctx, &k)
            .map_err(|e| WhisperError::Inference(format!("K upload: {e}")))?;
        let v_gpu = GpuResidentTensor::from_host(ctx, &v)
            .map_err(|e| WhisperError::Inference(format!("V upload: {e}")))?;

        // Create stream for scatter + attention (stream pooling tracked in WAPR-PERF-010)
        let stream =
            CudaStream::new(ctx).map_err(|e| WhisperError::Inference(format!("Stream: {e}")))?;

        kv_cache_scatter_gpu(
            ctx,
            &k_gpu,
            &mut self_k_caches[layer_idx],
            pos as u32,
            n_heads as u32,
            head_dim as u32,
            max_seq_len as u32,
            &stream,
        )
        .map_err(|e| WhisperError::Inference(format!("K scatter: {e}")))?;

        kv_cache_scatter_gpu(
            ctx,
            &v_gpu,
            &mut self_v_caches[layer_idx],
            pos as u32,
            n_heads as u32,
            head_dim as u32,
            max_seq_len as u32,
            &stream,
        )
        .map_err(|e| WhisperError::Inference(format!("V scatter: {e}")))?;

        // Incremental self-attention (WAPR-PERF-014: use shared stream)
        let seq_len = (pos + 1) as u32;
        let attn_out = incremental_attention_gpu_with_stream(
            ctx,
            &q_gpu,
            &self_k_caches[layer_idx],
            &self_v_caches[layer_idx],
            n_heads as u32,
            head_dim as u32,
            seq_len,
            max_seq_len as u32,
            &stream, // Reuse stream from KV scatter (no new stream creation!)
        )
        .map_err(|e| WhisperError::Inference(format!("Self attention: {e}")))?;

        // Sync before reading back (all kernels launched on shared stream)
        stream
            .synchronize()
            .map_err(|e| WhisperError::Inference(format!("Stream sync: {e}")))?;

        // WAPR-PERF-014: Drop ctx/stream borrows (NLL ends them) before &mut self.executor
        let mut attn_out = attn_out; // Move to local
        let attn_out_host = attn_out
            .to_host()
            .map_err(|e| WhisperError::Inference(format!("Attn D2H: {e}")))?;

        let mut attn_proj = vec![0.0f32; d_model];
        self.executor
            .gemv_cached(
                &format!("dec.L{layer_idx}.self_w_o"),
                &attn_out_host,
                &mut attn_proj,
                d_model as u32,
                d_model as u32,
            )
            .map_err(|e| WhisperError::Inference(format!("O projection: {e}")))?;

        // Add O bias (using pre-copied b_o)
        for i in 0..d_model {
            attn_proj[i] += b_o[i];
        }

        // Residual connection
        let mut residual: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

        // === Cross-Attention (re-borrow block) ===
        if let Some(enc_out) = encoder_output {
            let block = &self.decoder.blocks()[layer_idx];
            let normed2 = block.ln2.forward(&residual)?;
            let cross_out = block
                .cross_attn
                .forward_cross_dispatch(&normed2, enc_out, None)?;
            for (r, c) in residual.iter_mut().zip(cross_out.iter()) {
                *r += c;
            }
        }

        // === FFN (CPU - already optimized with SIMD) ===
        let block = &self.decoder.blocks()[layer_idx];
        let normed3 = block.ln3.forward(&residual)?;
        let ffn_out = block.ffn.forward(&normed3)?;
        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        Ok(residual)
    }

    /// WAPR-PERF-014: Full executor-based decoder forward pass
    ///
    /// Uses `forward_decoder_block_executor` for all layers.
    #[cfg(feature = "cuda")]
    pub fn forward_decoder_token_executor(
        &mut self,
        token_embedding: &[f32],
        pos: usize,
        encoder_output: &[f32],
    ) -> WhisperResult<Vec<f32>> {
        let n_layers = self.config.n_text_layer as usize;

        // Ensure executor weights are uploaded
        if self.executor.cached_weight_count() == 0 {
            self.upload_decoder_weights_to_executor()?;
        }

        // Also ensure GPU weights for KV cache scatter
        if self.gpu_decoder_weights.is_none() {
            self.upload_decoder_weights_to_gpu()?;
        }
        if self.gpu_self_k_head_first.is_none() {
            self.init_gpu_decoder_kv_cache_head_first()?;
        }

        // Process through all layers
        let mut hidden = token_embedding.to_vec();
        for layer_idx in 0..n_layers {
            hidden =
                self.forward_decoder_block_executor(layer_idx, &hidden, pos, Some(encoder_output))?;
        }

        Ok(hidden)
    }
}

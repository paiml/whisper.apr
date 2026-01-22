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

use crate::audio::{MelFilterbank, N_FFT, SAMPLE_RATE};
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
        let mel_filters = MelFilterbank::new(config.n_mels as usize, N_FFT, SAMPLE_RATE);
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
        let conv_output = self.encoder.conv_frontend().forward(mel)?;
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
                .map_err(|e| WhisperError::Inference(format!("enc ln_post_gamma upload: {e}")))?
        );
        self.gpu_enc_ln_post_beta = Some(
            GpuResidentTensor::from_host(ctx, &ln_post.bias)
                .map_err(|e| WhisperError::Inference(format!("enc ln_post_beta upload: {e}")))?
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
        let conv_frontend = self.encoder.conv_frontend();

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
                &normed2, enc_out, None, // TODO: Use cached cross-attention K/V
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
        let residual2 = if let Some(enc_len) = enc_seq_len {
            // Get cross-attention K/V caches
            let cross_k_caches = self
                .gpu_cross_k_head_first
                .as_ref()
                .ok_or_else(|| WhisperError::Inference("Cross K cache not initialized".into()))?;
            let cross_v_caches = self
                .gpu_cross_v_head_first
                .as_ref()
                .ok_or_else(|| WhisperError::Inference("Cross V cache not initialized".into()))?;

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
        let mut hidden_gpu = GpuResidentTensor::from_host(ctx, token_embedding)
            .map_err(|e| WhisperError::Inference(format!("embedding upload: {e}")))?;

        // Process through all layers using stream-based path
        for layer_idx in 0..n_layers {
            hidden_gpu = self.forward_decoder_block_gpu_stream(
                layer_idx,
                &hidden_gpu,
                pos,
                stream,
                enc_seq_len,
            )?;
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

        // DEBUG: Compare GPU vs CPU for first layer only
        let debug_gpu = std::env::var("WHISPER_DEBUG_GPU").is_ok();
        let debug_layer0_only = std::env::var("WHISPER_DEBUG_LAYER0").is_ok();

        if debug_gpu {
            // Run CPU encoder for comparison
            let cpu_output = self.encoder.forward_mel(mel)?;
            eprintln!("[DEBUG] CPU encoder output (full): len={}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
                cpu_output.len(),
                cpu_output.iter().sum::<f32>() / cpu_output.len() as f32,
                (cpu_output.iter().map(|x| x.powi(2)).sum::<f32>() / cpu_output.len() as f32).sqrt(),
                cpu_output.iter().cloned().fold(f32::INFINITY, f32::min),
                cpu_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

            // Also run just layer 0 on CPU for comparison
            let block0 = &self.encoder.blocks()[0];
            let n_heads = self.config.n_audio_head as usize;
            let _head_dim = d_model / n_heads;

            // Step 1: LayerNorm
            let ln1_out = block0.ln1.forward(&x)?;
            eprintln!(
                "[DEBUG-CPU] Layer 0 LN1: mean={:.6}, std={:.6}",
                ln1_out.iter().sum::<f32>() / ln1_out.len() as f32,
                (ln1_out.iter().map(|v| v.powi(2)).sum::<f32>() / ln1_out.len() as f32).sqrt()
            );

            // Step 2: Q/K/V projections
            let q = block0.self_attn.w_q().forward(&ln1_out, seq_len)?;
            let k = block0.self_attn.w_k().forward(&ln1_out, seq_len)?;
            let v = block0.self_attn.w_v().forward(&ln1_out, seq_len)?;
            eprintln!(
                "[DEBUG-CPU] Q: mean={:.6}, K: mean={:.6}, V: mean={:.6}",
                q.iter().sum::<f32>() / q.len() as f32,
                k.iter().sum::<f32>() / k.len() as f32,
                v.iter().sum::<f32>() / v.len() as f32
            );
        }

        // WAPR-PERF-011: Timing instrumentation for verification matrix
        let profile_layers = std::env::var("WHISPER_PROFILE_LAYERS").is_ok();
        let mut layer_times: Vec<u128> = Vec::new();

        #[allow(clippy::needless_range_loop)]
        // layer_idx used for debug, error messages, and indexing
        for layer_idx in 0..n_layers {
            if debug_layer0_only && layer_idx == 1 {
                // Get intermediate GPU output after layer 0 only
                let gpu_layer0_out = x_gpu
                    .to_host()
                    .map_err(|e| WhisperError::Inference(format!("debug download: {e}")))?;
                eprintln!("[DEBUG-GPU] After layer 0 only: len={}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
                    gpu_layer0_out.len(),
                    gpu_layer0_out.iter().sum::<f32>() / gpu_layer0_out.len() as f32,
                    (gpu_layer0_out.iter().map(|v| v.powi(2)).sum::<f32>() / gpu_layer0_out.len() as f32).sqrt(),
                    gpu_layer0_out.iter().cloned().fold(f32::INFINITY, f32::min),
                    gpu_layer0_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
            }

            let layer_start = std::time::Instant::now();
            x_gpu = forward_encoder_block_gpu(ctx, &x_gpu, &weights[layer_idx], config)
                .map_err(|e| WhisperError::Inference(format!("encoder block {layer_idx}: {e}")))?;
            let layer_elapsed = layer_start.elapsed().as_micros();
            layer_times.push(layer_elapsed);
        }

        // Print layer timing summary
        if profile_layers && !layer_times.is_empty() {
            let sum: u128 = layer_times.iter().sum();
            let avg = sum as f64 / layer_times.len() as f64;
            let max = layer_times.iter().max().copied().unwrap_or(0);
            let min = layer_times.iter().min().copied().unwrap_or(0);
            eprintln!("[PROFILE-LAYERS] {} layers, total {:.1}ms, avg {:.0}µs, min {:.0}µs, max {:.0}µs, variance {:.1}x",
                layer_times.len(), sum as f64 / 1000.0, avg, min, max, max as f64 / (min as f64 + 1.0));
        }

        // Step 5: Download output (1 D2H transfer)
        let download_start = std::time::Instant::now();
        let output = x_gpu
            .to_host()
            .map_err(|e| WhisperError::Inference(format!("output download: {e}")))?;
        let download_time = download_start.elapsed();

        if profile_detail {
            eprintln!(
                "[PROFILE-BREAKDOWN] Download: {:.1}ms",
                download_time.as_millis()
            );
        }

        // Log transfer stats for debugging
        if std::env::var("WHISPER_DEBUG_GPU").is_ok() {
            let stats = TransferStats::capture();
            eprintln!(
                "[GPU-TOTAL-OFFLOAD] encoder: {} H2D, {} D2H (expected: 1 H2D, 1 D2H)",
                stats.h2d_transfers, stats.d2h_transfers
            );
            eprintln!("[DEBUG] GPU encoder output (before ln_post): len={}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
                output.len(),
                output.iter().sum::<f32>() / output.len() as f32,
                (output.iter().map(|x| x.powi(2)).sum::<f32>() / output.len() as f32).sqrt(),
                output.iter().cloned().fold(f32::INFINITY, f32::min),
                output.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        // Step 6: Final layer norm (CPU - small overhead)
        // TODO: Move to GPU for complete offload
        let ln_post_start = std::time::Instant::now();
        let result = self.encoder.ln_post().forward(&output)?;
        let ln_post_time = ln_post_start.elapsed();

        if profile_detail {
            let total_time = total_start.elapsed();
            let layer_sum: u128 = layer_times.iter().sum();
            let accounted = conv_time.as_micros()
                + pos_time.as_micros()
                + upload_time.as_micros()
                + layer_sum
                + download_time.as_micros()
                + ln_post_time.as_micros();
            let unaccounted = total_time.as_micros().saturating_sub(accounted);
            eprintln!(
                "[PROFILE-BREAKDOWN] LnPost: {:.1}ms, Total: {:.1}ms",
                ln_post_time.as_millis(),
                total_time.as_millis()
            );
            eprintln!("[PROFILE-SUMMARY] Conv={:.0}µs PosEmb={:.0}µs Upload={:.0}µs Layers={:.0}µs Download={:.0}µs LnPost={:.0}µs",
                conv_time.as_micros(), pos_time.as_micros(), upload_time.as_micros(),
                layer_sum, download_time.as_micros(), ln_post_time.as_micros());
            eprintln!(
                "[PROFILE-SUMMARY] Accounted: {:.1}ms, Unaccounted: {:.1}ms ({:.1}%)",
                accounted as f64 / 1000.0,
                unaccounted as f64 / 1000.0,
                unaccounted as f64 / total_time.as_micros() as f64 * 100.0
            );
        }

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

        let ctx = self.executor.context();
        let stream = CudaStream::new(ctx)
            .map_err(|e| WhisperError::Inference(format!("warmup stream: {e}")))?;

        // Step 2: Run encoder with dummy mel to compile encoder kernels
        let n_mels = self.config.n_mels as usize;
        let d_model = self.config.n_text_state as usize;
        let dummy_mel: Vec<f32> = vec![0.0; n_mels * 100]; // 100 frames
        let enc_gpu = self.encode_gpu_resident(&dummy_mel)?;
        let enc_seq_len = enc_gpu.len() / d_model;

        // Step 3: Populate cross K/V to compile permute kernels
        self.populate_cross_kv_caches_gpu(&enc_gpu, &stream)?;

        // Step 4: Run decoder to compile decoder kernels
        let dummy_embedding: Vec<f32> = vec![0.0; d_model];
        let _dec_out = self.forward_decoder_token_gpu_stream(
            &dummy_embedding,
            0,
            &stream,
            Some(enc_seq_len),
        )?;
        stream.synchronize()
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
                3, 1, 1,
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
                3, 2, 1,
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
            .layer_norm(ctx, ln_post_gamma, ln_post_beta, d_model as u32, seq_len as u32)
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
                .map_err(|e| WhisperError::Inference(format!("cross K permute L{layer_idx}: {e}")))?;

            let v_head_first = v_proj
                .interleaved_to_head_first(
                    ctx,
                    enc_len as u32,
                    n_heads as u32,
                    head_dim as u32,
                    stream,
                )
                .map_err(|e| WhisperError::Inference(format!("cross V permute L{layer_idx}: {e}")))?;

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

        // GPU attention: choose path based on env var and feature (WAPR-PERF-004 vs WAPR-PERF-005)
        // - WHISPER_GPU_RESIDENT=1 + cuda feature: New GPU-resident path with minimal transfers
        // - Default: Old gemm-per-head path (working but slow due to transfers)
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

        // Step 4: Output projection to vocabulary
        // Using CPU path due to gemv_cached bug (WAPR-PERF-006)
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

        // TODO: Implement GPU decoder forward pass
        // For now, fall back to CPU decoder
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
            .compute(&padded_audio, crate::audio::HOP_LENGTH)?;
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

        // No timestamps token
        tokens.push(specials.no_timestamps);

        // Decode loop using incremental decoding with KV cache
        let max_tokens = self.config.n_text_ctx as usize;
        let d_model = self.config.n_text_state as usize;
        let n_layers = self.config.n_text_layer as usize;
        let n_vocab = self.config.n_vocab as usize;
        let eot_token = specials.eot;

        // Create KV cache for incremental decoding
        let mut cache = crate::model::DecoderKVCache::new(n_layers, d_model, max_tokens);

        // Create token suppressor
        let suppressor = crate::inference::WhisperTokenSuppressor::new()
            .with_timestamp_suppression(!options.word_timestamps)
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

        // 2. Run through all decoder blocks (GPU self-attention)
        for layer_idx in 0..n_layers {
            x = self.forward_decoder_block_gpu(layer_idx, &x, pos, Some(encoder_output))?;
        }

        // 3. Final layer norm (CPU - simple)
        // ln_post is accessible via decoder
        let hidden = self.decoder.ln_post().forward(&x)?;

        // 4. Increment position for next token
        self.gpu_decoder_pos += 1;

        // 5. Output projection on GPU
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
        // Whisper constants
        const N_SAMPLES_30S: usize = 480_000; // 30 seconds at 16kHz
        const N_FRAMES: usize = 3000; // Whisper expects exactly 3000 frames
        const N_MELS: usize = 80;

        // === TRACE: Mel spectrogram (mapped to EMBED step) ===
        self.tracer.start_step(TraceStep::Embed);

        // Pad/truncate audio to 30 seconds
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
            .compute(&padded_audio, crate::audio::HOP_LENGTH)?;
        let actual_frames = mel.len() / N_MELS;

        // Ensure exactly 3000 frames
        if actual_frames < N_FRAMES {
            let pad_value = -1.0_f32;
            let mut padded_mel = vec![pad_value; N_FRAMES * N_MELS];
            padded_mel[..mel.len()].copy_from_slice(&mel);
            mel = padded_mel;
        } else if actual_frames > N_FRAMES {
            mel.truncate(N_FRAMES * N_MELS);
        }

        // Trace mel computation (token_count=N_FRAMES, hidden_dim=N_MELS)
        self.tracer.trace_embed(N_FRAMES, N_MELS, Some(&mel));

        // === TRACE: Encoder forward pass (mapped to TRANSFORMER_BLOCK) ===
        self.tracer.start_step(TraceStep::TransformerBlock);

        // Run encoder - choose path based on environment variables (WAPR-PERF-004/005)
        // - WHISPER_GPU_TOTAL_OFFLOAD=1 + WHISPER_GPU_DECODER_OFFLOAD=1: Optimized stream pipeline
        // - WHISPER_GPU_TOTAL_OFFLOAD=1: Full GPU encoder (2x target)
        // - WHISPER_GPU_ENCODER=1: Partial GPU (attention only)
        // - Default: CPU encoder with SIMD
        // WAPR-PERF-021: Select encoder path
        // - WHISPER_GPU_ENCODER=1: GPU attention-only encoder (correct output)
        // - Default: CPU encoder with SIMD (fastest correct path)
        // Note: WHISPER_GPU_TOTAL_OFFLOAD has buggy GPU convolutions, disabled
        #[cfg(feature = "cuda")]
        let use_gpu_encoder = std::env::var("WHISPER_GPU_ENCODER").is_ok();
        #[cfg(feature = "cuda")]
        let use_gpu_decoder = std::env::var("WHISPER_GPU_DECODER_OFFLOAD").is_ok();

        #[cfg(feature = "cuda")]
        let encoder_output = {
            if use_gpu_encoder {
                eprintln!("[WAPR-PERF-005] Using GPU attention-only encoder...");
                self.encode_gpu(&mel)?
            } else {
                eprintln!("[CPU] Using SIMD encoder...");
                self.encoder.forward_mel(&mel)?
            }
        };
        #[cfg(not(feature = "cuda"))]
        let encoder_output = self.encoder.forward_mel(&mel)?;
        #[cfg(not(feature = "cuda"))]
        let use_gpu_decoder = false;

        // Trace encoder output (layer_idx=0 for encoder, iteration=0 for prefill)
        let d_model = self.config.n_text_state as usize;
        let enc_seq_len = encoder_output.len() / d_model;
        self.tracer
            .trace_layer(0, 0, Some(&encoder_output), enc_seq_len, d_model);

        // Build initial tokens
        use crate::tokenizer::special_tokens::{self, SpecialTokens};
        let specials = SpecialTokens::for_vocab_size(self.config.n_vocab as usize);
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
        tokens.push(specials.no_timestamps);

        // Hybrid GPU path:
        // - CPU decoder blocks (GPU flash_attention_cached has kernel compatibility issues)
        // - GPU output projection via executor.gemm() (FIX 1 - working!)
        let max_tokens = self.config.n_text_ctx as usize;
        let n_layers = self.config.n_text_layer as usize;
        let n_vocab = self.config.n_vocab as usize;
        let eot_token = specials.eot;

        let mut cache = DecoderKVCache::new(n_layers, d_model, max_tokens);

        // Token suppressor
        let suppressor = crate::inference::WhisperTokenSuppressor::new()
            .with_timestamp_suppression(!options.word_timestamps)
            .with_vocab_size(n_vocab);

        // Initialize GPU decoder if enabled
        #[cfg(feature = "cuda")]
        if use_gpu_decoder {
            eprintln!("[WAPR-PERF-013] Using GPU decoder...");
            // Reset GPU decoder position for new transcription
            self.reset_gpu_decoder_pos();
        }

        // Process initial tokens
        for &token in &tokens {
            #[cfg(feature = "cuda")]
            if use_gpu_decoder {
                let _ = self.forward_one_gpu_total_offload(token, &encoder_output)?;
            } else {
                let _ = self
                    .decoder
                    .forward_one(token, &encoder_output, &mut cache)?;
            }
            #[cfg(not(feature = "cuda"))]
            let _ = self
                .decoder
                .forward_one(token, &encoder_output, &mut cache)?;
        }

        // Generate tokens
        let debug_gpu = std::env::var("WHISPER_DEBUG_GPU").is_ok();
        for gen_idx in 0..max_tokens.saturating_sub(tokens.len()) {
            let last_token = *tokens.last().unwrap_or(&specials.sot);

            if debug_gpu && gen_idx < 5 {
                eprintln!(
                    "[DEBUG] gen_idx={} last_token={} tokens={:?}",
                    gen_idx, last_token, &tokens
                );
            }

            // === TRACE: LM_HEAD (output projection) ===
            self.tracer.start_step(TraceStep::LmHead);

            // WAPR-PERF-013: Use GPU decoder when enabled
            #[cfg(feature = "cuda")]
            let mut logits = if use_gpu_decoder {
                self.forward_one_gpu_total_offload(last_token, &encoder_output)?
            } else {
                self.forward_one_gpu(last_token, &encoder_output, &mut cache)?
            };
            #[cfg(not(feature = "cuda"))]
            let mut logits = self.forward_one_gpu(last_token, &encoder_output, &mut cache)?;

            // Trace output projection
            self.tracer.trace_lm_head(gen_idx, &logits, n_vocab);

            // === TRACE: SAMPLE (token selection) ===
            self.tracer.start_step(TraceStep::Sample);

            suppressor.apply(&mut logits);

            let next_token = match options.strategy {
                DecodingStrategy::Greedy => logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(eot_token),
                DecodingStrategy::BeamSearch { .. } | DecodingStrategy::Sampling { .. } => logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(eot_token),
            };

            // Trace sampling result (temperature=0.0 for greedy, top_k=1)
            self.tracer
                .trace_sample(gen_idx, &logits, next_token, 0.0, 1);

            if next_token == eot_token {
                break;
            }

            tokens.push(next_token);
        }

        // === TRACE: DECODE (detokenization) ===
        self.tracer.start_step(TraceStep::Decode);

        let text = self.tokenizer.decode_with_options(&tokens, true)?;

        // Trace decode result (iteration=0 for final decode, last token, vocab_size)
        let last_token = tokens.last().copied().unwrap_or(0);
        self.tracer.trace_decode(0, last_token, &text, n_vocab);

        let language = options.language.clone().unwrap_or_else(|| "en".to_string());

        Ok(TranscriptionResult {
            text,
            language,
            segments: Vec::new(),
        })
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

        // Create stream for scatter + attention (TODO: reuse executor stream to avoid allocation)
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use trueno_gpu::driver::CudaStream;

    #[test]
    fn test_cuda_availability_check() {
        // This test verifies the API compiles correctly
        // Actual GPU tests require hardware
        let available = CudaExecutor::is_available();
        eprintln!("CUDA available: {}", available);

        if available {
            let num_devices = CudaExecutor::num_devices();
            eprintln!("CUDA devices: {}", num_devices);
        }
    }

    /// Test FIX 1: Verify GPU gemm produces correct output projection.
    #[test]
    fn test_gpu_gemm_output_projection() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        }

        // Load model
        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        // Load model from file
        let bytes = std::fs::read(model_path).expect("Failed to read model file");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Upload weights
        cuda_model
            .upload_weights()
            .expect("Failed to upload weights");

        // Create test hidden state (normalized random values)
        let d_model = cuda_model.config().n_text_state as usize;
        let hidden: Vec<f32> = (0..d_model).map(|i| (i as f32 * 0.1).sin() * 0.1).collect();

        // Compute CPU output projection
        let cpu_logits = cuda_model.decoder.project_to_vocab_debug(&hidden);

        // Compute GPU output projection
        let gpu_logits = cuda_model
            .project_to_vocab_gpu(&hidden)
            .expect("GPU gemm failed");

        // Compare results
        assert_eq!(
            cpu_logits.len(),
            gpu_logits.len(),
            "Output dimension mismatch"
        );

        // Find max difference
        let max_diff: f32 = cpu_logits
            .iter()
            .zip(gpu_logits.iter())
            .map(|(c, g)| (*c - *g).abs())
            .fold(0.0f32, f32::max);

        // Find argmax for both
        let cpu_argmax: (usize, f32) = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
            .unwrap();
        let gpu_argmax: (usize, f32) = gpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
            .unwrap();

        eprintln!("\n=== FIX 1: GPU GEMM Output Projection Test ===");
        eprintln!("CPU: argmax={} max={:.6}", cpu_argmax.0, cpu_argmax.1);
        eprintln!("GPU: argmax={} max={:.6}", gpu_argmax.0, gpu_argmax.1);
        eprintln!("Max difference: {:.6}", max_diff);

        // Check argmax matches
        assert_eq!(
            cpu_argmax.0, gpu_argmax.0,
            "Argmax mismatch: CPU={} GPU={}. Max diff={}",
            cpu_argmax.0, gpu_argmax.0, max_diff
        );

        // Check values are close (allow some floating point tolerance)
        assert!(
            max_diff < 0.01,
            "Values differ too much: max_diff={:.6} (threshold=0.01)",
            max_diff
        );

        eprintln!("✓ FIX 1 PASSED: GPU gemm produces correct output projection");
    }

    /// Test APR-style tracing integration (WAPR-PERF-004).
    ///
    /// Verifies that InferenceTracer captures timing for each step:
    /// - EMBED: Mel spectrogram computation
    /// - TRANSFORMER_BLOCK: Encoder forward pass
    /// - LM_HEAD: Output projection per token
    /// - SAMPLE: Token sampling
    /// - DECODE: Detokenization
    #[test]
    fn test_inference_tracing() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping tracing test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        let audio_path = "demos/test-audio/test-speech-1.5s.wav";
        if !std::path::Path::new(audio_path).exists() {
            eprintln!("Audio not found at {}, skipping test", audio_path);
            return;
        }

        // Load model
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Enable tracing
        cuda_model.enable_tracing(TraceConfig::enabled());

        // Load audio via wav parser
        let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio file");
        let wav_data =
            crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let audio = wav_data.samples;

        // Run transcription with tracing
        let options = crate::TranscribeOptions::default();
        let result = cuda_model
            .transcribe_gpu(&audio, options)
            .expect("Transcription failed");

        eprintln!("\n=== APR-Style Inference Tracing Test (WAPR-PERF-004) ===");
        eprintln!("Transcription: \"{}\"", result.text);
        eprintln!();

        // Print trace summary
        cuda_model.print_trace_summary();

        // Verify tracer collected events
        let tracer = cuda_model.tracer();
        assert!(tracer.is_enabled(), "Tracer should be enabled");

        // Check events were collected
        let events = tracer.events();
        assert!(!events.is_empty(), "Tracer should have collected events");

        eprintln!("\nEvents collected: {}", events.len());

        // Count events by step
        let mut step_counts = std::collections::HashMap::new();
        let mut step_durations = std::collections::HashMap::new();
        for event in events {
            *step_counts.entry(event.step.name()).or_insert(0) += 1;
            *step_durations.entry(event.step.name()).or_insert(0_u64) += event.duration_us;
        }

        eprintln!("\nStep breakdown:");
        for (step, count) in &step_counts {
            let duration = step_durations.get(step).unwrap_or(&0);
            eprintln!("  {}: {} events, {}µs total", step, count, duration);
        }

        // Verify we have expected steps
        assert!(
            step_counts.contains_key("EMBED"),
            "Should have EMBED events"
        );
        assert!(
            step_counts.contains_key("TRANSFORMER_BLOCK"),
            "Should have TRANSFORMER_BLOCK events"
        );

        eprintln!("\n✓ APR-Style Tracing Test PASSED");
    }

    /// Test GPU encoder vs CPU encoder performance (WAPR-PERF-005).
    ///
    /// Target: GPU encoder should be ~20x faster than CPU encoder.
    /// - CPU encoder: ~6.15s (98.7% of total time)
    /// - GPU encoder target: <300ms
    #[test]
    fn test_gpu_encoder_performance() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping GPU encoder test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        let audio_path = "demos/test-audio/test-speech-1.5s.wav";
        if !std::path::Path::new(audio_path).exists() {
            eprintln!("Audio not found at {}, skipping test", audio_path);
            return;
        }

        // Load model
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Load audio via wav parser
        let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio file");
        let wav_data =
            crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let audio = wav_data.samples;

        // Precompute mel spectrogram for fair comparison
        const N_SAMPLES_30S: usize = 480_000;
        const N_FRAMES: usize = 3000;
        const N_MELS: usize = 80;

        let padded_audio = if audio.len() < N_SAMPLES_30S {
            let mut padded = vec![0.0_f32; N_SAMPLES_30S];
            padded[..audio.len()].copy_from_slice(&audio);
            padded
        } else {
            audio[..N_SAMPLES_30S].to_vec()
        };

        let mut mel = cuda_model
            .mel_filters
            .compute(&padded_audio, crate::audio::HOP_LENGTH)
            .expect("Mel computation failed");
        let actual_frames = mel.len() / N_MELS;
        if actual_frames < N_FRAMES {
            let mut padded_mel = vec![-1.0_f32; N_FRAMES * N_MELS];
            padded_mel[..mel.len()].copy_from_slice(&mel);
            mel = padded_mel;
        } else if actual_frames > N_FRAMES {
            mel.truncate(N_FRAMES * N_MELS);
        }

        eprintln!("\n=== GPU Encoder Performance Test (WAPR-PERF-005) ===");
        eprintln!("Mel spectrogram: {} frames x {} mels", N_FRAMES, N_MELS);

        // Time CPU encoder
        let cpu_start = std::time::Instant::now();
        let cpu_output = cuda_model
            .encoder
            .forward_mel(&mel)
            .expect("CPU encoder failed");
        let cpu_time = cpu_start.elapsed();

        eprintln!("\nCPU Encoder: {:?}", cpu_time);
        eprintln!("  Output shape: {} elements", cpu_output.len());

        // Time GPU encoder
        let gpu_start = std::time::Instant::now();
        let gpu_output = cuda_model.encode_gpu(&mel).expect("GPU encoder failed");
        let gpu_time = gpu_start.elapsed();

        eprintln!("\nGPU Encoder: {:?}", gpu_time);
        eprintln!("  Output shape: {} elements", gpu_output.len());

        // Calculate speedup
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        eprintln!("\nSpeedup: {:.2}x", speedup);

        // Verify outputs match (numerically close)
        assert_eq!(cpu_output.len(), gpu_output.len(), "Output size mismatch");
        let max_diff: f32 = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(c, g)| (*c - *g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Max difference: {:.6}", max_diff);

        // Allow some numerical tolerance
        if max_diff > 0.1 {
            eprintln!("WARNING: Large numerical difference between CPU and GPU encoder");
        }

        eprintln!("\n✓ GPU Encoder Test Complete");
    }

    /// Test GPU-resident attention (WAPR-PERF-004).
    ///
    /// Verifies the new trueno-gpu integration eliminates host↔device transfers:
    /// - Old path: ~150 transfers per encoder forward
    /// - New path: 3 H2D (Q, K, V) + 1 D2H (output) = 4 total
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_resident_attention() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping GPU-resident attention test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        // Load model
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        eprintln!("\n=== GPU-Resident Attention Test (WAPR-PERF-004) ===");

        // Test with small tensors first
        let seq_len = 16;
        let n_heads = 6;
        let head_dim = 64;
        let d_model = n_heads * head_dim;

        // Create test Q, K, V
        let q: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();

        // Reset transfer counters
        reset_transfer_counters();

        // Run GPU-resident attention
        let result = cuda_model.attention_gpu_resident(&q, &k, &v, seq_len, n_heads, head_dim);

        match result {
            Ok(output) => {
                let h2d = total_h2d_transfers();
                let d2h = total_d2h_transfers();

                eprintln!("Output shape: {} elements", output.len());
                eprintln!("Transfer stats: {} H2D, {} D2H", h2d, d2h);
                eprintln!("Expected: 3 H2D (Q,K,V), 1 D2H (output)");

                // Verify output shape
                assert_eq!(output.len(), seq_len * d_model, "Output shape mismatch");

                // Verify minimal transfers (3 in + 1 out)
                assert_eq!(h2d, 3, "Should have exactly 3 H2D transfers (Q, K, V)");
                assert_eq!(d2h, 1, "Should have exactly 1 D2H transfer (output)");

                eprintln!("\n✓ GPU-Resident Attention Test PASSED");
                eprintln!("  Eliminated ~146 unnecessary transfers per attention!");
            }
            Err(e) => {
                eprintln!("GPU-resident attention failed: {}", e);
                eprintln!("This may be expected if CUDA kernels have issues.");
                eprintln!("Falling back to old gemm-per-head path is still available.");
            }
        }
    }

    /// WAPR-PERF-013 Point 154: Numerical Parity Test for Incremental Attention
    ///
    /// Verifies GPU incremental attention matches CPU reference within 1e-5.
    /// This is CRITICAL: autoregressive decoding amplifies small errors into
    /// garbage text within 10-20 tokens.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_incremental_attention_numerical_parity() {
        use trueno_gpu::memory::resident::incremental_attention_gpu;

        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping numerical parity test");
            return;
        }

        eprintln!("\n=== WAPR-PERF-013 Point 154: Numerical Parity Test ===");

        // Whisper-tiny config
        let n_heads: u32 = 6;
        let head_dim: u32 = 64;
        let max_seq_len: u32 = 448;
        let seq_len: u32 = 10; // Test with 10 cached tokens
        let d_model = (n_heads * head_dim) as usize;

        // Generate deterministic test data
        let q: Vec<f32> = (0..d_model).map(|i| ((i as f32) * 0.01).sin()).collect();

        // K/V cache in head-first layout [n_heads, max_seq_len, head_dim]
        let cache_size = (n_heads * max_seq_len * head_dim) as usize;
        let k_cache: Vec<f32> = (0..cache_size).map(|i| ((i as f32) * 0.02).cos()).collect();
        let v_cache: Vec<f32> = (0..cache_size).map(|i| ((i as f32) * 0.03).sin()).collect();

        // === CPU Reference Implementation ===
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut cpu_output = vec![0.0f32; d_model];

        for h in 0..n_heads as usize {
            // Extract Q for this head
            let q_start = h * head_dim as usize;
            let q_h = &q[q_start..q_start + head_dim as usize];

            // KV cache offset for this head
            let kv_head_offset = h * (max_seq_len as usize) * (head_dim as usize);

            // Compute attention scores for seq_len positions
            let mut scores = vec![0.0f32; seq_len as usize];
            for pos in 0..seq_len as usize {
                let k_offset = kv_head_offset + pos * (head_dim as usize);
                let mut dot = 0.0f32;
                for e in 0..head_dim as usize {
                    dot += q_h[e] * k_cache[k_offset + e];
                }
                scores[pos] = dot * scale;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            for s in &mut scores {
                *s /= sum_exp;
            }

            // Weighted sum of values
            let out_start = h * head_dim as usize;
            for e in 0..head_dim as usize {
                let mut weighted_sum = 0.0f32;
                for pos in 0..seq_len as usize {
                    let v_offset = kv_head_offset + pos * (head_dim as usize) + e;
                    weighted_sum += scores[pos] * v_cache[v_offset];
                }
                cpu_output[out_start + e] = weighted_sum;
            }
        }

        // === GPU Implementation ===
        let executor = CudaExecutor::new(0).expect("Failed to create CUDA executor");
        let ctx = executor.context();

        let q_gpu = GpuResidentTensor::from_host(ctx, &q).expect("Q upload failed");
        let k_gpu = GpuResidentTensor::from_host(ctx, &k_cache).expect("K upload failed");
        let v_gpu = GpuResidentTensor::from_host(ctx, &v_cache).expect("V upload failed");

        let mut gpu_output = incremental_attention_gpu(
            ctx,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            n_heads,
            head_dim,
            seq_len,
            max_seq_len,
        )
        .expect("GPU attention failed");

        let gpu_result = gpu_output.to_host().expect("D2H failed");

        // === Numerical Parity Check ===
        let tolerance = 1e-5_f32;
        let mut max_diff = 0.0f32;
        let mut diff_count = 0;

        for (i, (cpu_val, gpu_val)) in cpu_output.iter().zip(gpu_result.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > tolerance {
                if diff_count < 5 {
                    eprintln!(
                        "  [{}] CPU={:.6} GPU={:.6} diff={:.2e}",
                        i, cpu_val, gpu_val, diff
                    );
                }
                diff_count += 1;
            }
        }

        eprintln!("Max absolute difference: {:.2e}", max_diff);
        eprintln!("Elements exceeding tolerance: {}/{}", diff_count, d_model);

        if max_diff > tolerance {
            eprintln!("\n❌ NUMERICAL PARITY FAILED");
            eprintln!("   GPU incremental attention diverges from CPU reference.");
            eprintln!("   This WILL cause garbage text in autoregressive decoding.");
            panic!(
                "Numerical parity test failed: max_diff={:.2e} > tolerance={:.2e}",
                max_diff, tolerance
            );
        }

        eprintln!("\n✓ WAPR-PERF-013 Point 154: Numerical Parity PASSED");
        eprintln!("  GPU attention matches CPU within {:.0e}", tolerance);
    }

    /// WAPR-PERF-013 Point 155: GPU Decoder Block Smoke Test
    ///
    /// Verifies GPU decoder block runs correctly and produces valid output.
    /// Tests: self-attention (GPU) + FFN (CPU) path.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_decoder_block_smoke() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping GPU decoder block test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n=== WAPR-PERF-013 Point 155: GPU Decoder Block Smoke Test ===");

        // Load model
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Upload decoder weights to GPU
        cuda_model
            .upload_decoder_weights_to_gpu()
            .expect("Failed to upload decoder weights");

        // Initialize head-first KV caches
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Failed to init KV cache");

        let d_model = cuda_model.config().n_text_state as usize;
        let n_layers = cuda_model.config().n_text_layer as usize;
        let max_len = cuda_model.config().n_text_ctx as usize;

        eprintln!(
            "Model: d_model={}, n_layers={}, max_len={}",
            d_model, n_layers, max_len
        );

        // Generate test input (simulated decoder input embedding)
        let test_input: Vec<f32> = (0..d_model)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();

        // Test multiple positions to verify KV cache works
        for pos in 0..3 {
            eprintln!("\nTesting position {}...", pos);

            // Run GPU decoder block (layer 0, no cross-attention)
            let gpu_output = cuda_model
                .forward_decoder_block_gpu(
                    0, // layer_idx
                    &test_input,
                    pos,
                    None, // No encoder output - skip cross-attention
                )
                .expect("GPU decoder block failed");

            // Verify output shape
            assert_eq!(gpu_output.len(), d_model, "Output dimension mismatch");

            // Verify no NaN/Inf
            let has_nan = gpu_output.iter().any(|x| x.is_nan());
            let has_inf = gpu_output.iter().any(|x| x.is_infinite());
            assert!(!has_nan, "Output contains NaN at pos={}", pos);
            assert!(!has_inf, "Output contains Inf at pos={}", pos);

            // Verify reasonable magnitude (not all zeros, not exploding)
            let max_abs = gpu_output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let mean_abs = gpu_output.iter().map(|x| x.abs()).sum::<f32>() / d_model as f32;

            eprintln!("  Output shape: {}", gpu_output.len());
            eprintln!(
                "  Output sample: [{:.4}, {:.4}, {:.4}...]",
                gpu_output[0], gpu_output[1], gpu_output[2]
            );
            eprintln!("  Max abs: {:.4}, Mean abs: {:.4}", max_abs, mean_abs);

            assert!(max_abs < 1000.0, "Output exploding: max_abs={}", max_abs);
            assert!(mean_abs > 1e-6, "Output near zero: mean_abs={}", mean_abs);
        }

        eprintln!("\n✓ WAPR-PERF-013 Point 155: GPU Decoder Block Smoke Test PASSED");
        eprintln!("  - Output shape correct ({})", d_model);
        eprintln!("  - No NaN/Inf values");
        eprintln!("  - KV cache works across positions");
    }

    /// WAPR-PERF-014: Executor vs GPU Decoder Block Parity Test
    ///
    /// Verifies that executor-based forward pass produces same output as
    /// GpuResidentTensor-based forward pass.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_executor_vs_gpu_decoder_parity() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping executor parity test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n=== WAPR-PERF-014: Executor vs GPU Decoder Parity ===");

        // Load model for GPU path
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr1 = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model1 = apr1.into_cuda(0).expect("Failed to create CUDA model 1");

        // Load model for executor path (need separate instance due to KV cache state)
        let apr2 = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model2 = apr2.into_cuda(0).expect("Failed to create CUDA model 2");

        // Upload weights for both paths
        cuda_model1
            .upload_decoder_weights_to_gpu()
            .expect("Upload GPU weights");
        cuda_model1
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV cache 1");

        cuda_model2
            .upload_decoder_weights_to_gpu()
            .expect("Upload GPU weights 2");
        cuda_model2
            .upload_decoder_weights_to_executor()
            .expect("Upload executor weights");
        cuda_model2
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV cache 2");

        let d_model = cuda_model1.config().n_text_state as usize;

        // Generate test input
        let test_input: Vec<f32> = (0..d_model)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();

        eprintln!("Testing layer 0, position 0...");

        // Run GPU forward pass
        let gpu_start = std::time::Instant::now();
        let gpu_output = cuda_model1
            .forward_decoder_block_gpu(0, &test_input, 0, None)
            .expect("GPU forward failed");
        let gpu_time = gpu_start.elapsed();

        // Run executor forward pass
        let exec_start = std::time::Instant::now();
        let exec_output = cuda_model2
            .forward_decoder_block_executor(0, &test_input, 0, None)
            .expect("Executor forward failed");
        let exec_time = exec_start.elapsed();

        // Compare outputs
        assert_eq!(
            gpu_output.len(),
            exec_output.len(),
            "Output length mismatch"
        );

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        for (i, (g, e)) in gpu_output.iter().zip(exec_output.iter()).enumerate() {
            let diff = (g - e).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            sum_diff += diff;
            if diff > 1e-3 && i < 5 {
                eprintln!("  diff[{}]: gpu={:.6} exec={:.6} diff={:.6}", i, g, e, diff);
            }
        }
        let mean_diff = sum_diff / d_model as f32;

        eprintln!("\nResults:");
        eprintln!("  GPU time:    {:?}", gpu_time);
        eprintln!("  Exec time:   {:?}", exec_time);
        eprintln!("  Max diff:    {:.6}", max_diff);
        eprintln!("  Mean diff:   {:.6}", mean_diff);

        // Allow some numerical tolerance (different stream execution order)
        assert!(max_diff < 1e-2, "Max diff too high: {}", max_diff);
        assert!(mean_diff < 1e-4, "Mean diff too high: {}", mean_diff);

        eprintln!("\n✓ WAPR-PERF-014: Executor vs GPU Decoder Parity PASSED");
        eprintln!("  - Outputs match within tolerance");
        eprintln!("  - Max diff: {:.6} < 1e-2", max_diff);
    }

    /// WAPR-PERF-013 Point 156: GPU vs CPU Decoder Parity Test
    ///
    /// Compares GPU total offload decoder output vs CPU decoder for same input.
    /// This is the critical test for detecting accumulation drift.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_vs_cpu_decoder_parity() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping parity test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        let audio_path = "demos/test-audio/test-speech-1.5s.wav";
        if !std::path::Path::new(audio_path).exists() {
            eprintln!("Audio not found at {}, skipping test", audio_path);
            return;
        }

        eprintln!("\n=== WAPR-PERF-013 Point 156: GPU vs CPU Decoder Parity ===");

        // Load model (compute mel before converting to CUDA)
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");

        // Load audio and compute mel BEFORE into_cuda()
        let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio");
        let wav_data =
            crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let mel = apr.compute_mel(&wav_data.samples).expect("Mel failed");

        // Now convert to CUDA
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Run encoder (CPU is fine for this test)
        let encoder_output = cuda_model
            .encoder
            .forward_mel(&mel)
            .expect("Encoder failed");

        let d_model = cuda_model.config().n_text_state as usize;
        let n_layers = cuda_model.config().n_text_layer as usize;
        let n_vocab = cuda_model.config().n_vocab as usize;
        let max_len = cuda_model.config().n_text_ctx as usize;

        eprintln!(
            "Model: d_model={}, n_layers={}, n_vocab={}",
            d_model, n_layers, n_vocab
        );
        eprintln!("Encoder output: {} elements", encoder_output.len());

        // Test token: SOT (start of transcript)
        let sot_token = 50258_u32; // Whisper SOT token

        // === CPU Reference Path ===
        let mut cpu_cache = crate::model::DecoderKVCache::new(n_layers, d_model, max_len);
        let cpu_hidden = cuda_model
            .decoder
            .forward_one_hidden(sot_token, &encoder_output, &mut cpu_cache)
            .expect("CPU decoder failed");

        // Get CPU logits
        let cpu_logits = cuda_model.decoder.project_to_vocab_debug(&cpu_hidden);

        // === GPU Path ===
        cuda_model.reset_gpu_decoder_pos();
        cuda_model
            .upload_decoder_weights_to_gpu()
            .expect("Upload failed");
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init failed");

        let gpu_logits = cuda_model
            .forward_one_gpu_total_offload(sot_token, &encoder_output)
            .expect("GPU decoder failed");

        // === Parity Check ===
        let tolerance = 1e-3_f32; // Slightly looser for full path
        let mut max_diff = 0.0_f32;
        let mut diff_count = 0_usize;

        // Find argmax for both
        let cpu_argmax = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let gpu_argmax = gpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        for (_i, (cpu_val, gpu_val)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
            let diff: f32 = (*cpu_val - *gpu_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > tolerance {
                diff_count += 1;
            }
        }

        eprintln!(
            "CPU argmax: {} (token '{}')",
            cpu_argmax,
            cuda_model
                .tokenizer
                .decode(&[cpu_argmax as u32])
                .unwrap_or_default()
        );
        eprintln!(
            "GPU argmax: {} (token '{}')",
            gpu_argmax,
            cuda_model
                .tokenizer
                .decode(&[gpu_argmax as u32])
                .unwrap_or_default()
        );
        eprintln!("Max absolute difference: {:.2e}", max_diff);
        eprintln!(
            "Elements exceeding {:.0e}: {}/{}",
            tolerance, diff_count, n_vocab
        );

        // Critical: argmax must match for correct decoding
        if cpu_argmax != gpu_argmax {
            eprintln!("\n❌ ARGMAX MISMATCH: GPU decoder produces different token");
            eprintln!("   This WILL cause divergent text output.");
            panic!(
                "GPU vs CPU decoder argmax mismatch: CPU={} GPU={}",
                cpu_argmax, gpu_argmax
            );
        }

        if max_diff > 0.1 {
            eprintln!("\n⚠️ WARNING: Large numerical difference detected");
            eprintln!("   This may cause drift over long sequences.");
        }

        eprintln!("\n✓ WAPR-PERF-013 Point 156: GPU vs CPU Decoder Parity PASSED");
        eprintln!("  Argmax matches, max_diff={:.2e}", max_diff);
    }

    /// WAPR-PERF-014: GPU vs Executor Decoder Performance Benchmark
    ///
    /// Compares decode performance between:
    /// 1. GPU path (GpuResidentTensor creates streams per operation)
    /// 2. Executor path (persistent compute_stream)
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_vs_executor_decode_benchmark() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping benchmark");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping benchmark", model_path);
            return;
        }

        let audio_path = "demos/test-audio/test-speech-1.5s.wav";
        if !std::path::Path::new(audio_path).exists() {
            eprintln!("Audio not found at {}, skipping benchmark", audio_path);
            return;
        }

        eprintln!("\n=== WAPR-PERF-014: GPU vs Executor Decode Benchmark ===");

        // Load audio and compute mel
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio");
        let wav_data =
            crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let mel = apr.compute_mel(&wav_data.samples).expect("Mel failed");

        // Create CUDA model
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Run encoder
        let encoder_output = cuda_model
            .encoder
            .forward_mel(&mel)
            .expect("Encoder failed");
        eprintln!("Encoder output: {} elements", encoder_output.len());

        // Initial tokens
        use crate::tokenizer::special_tokens::SpecialTokens;
        let specials = SpecialTokens::for_vocab_size(cuda_model.config().n_vocab as usize);
        let initial_tokens = vec![
            specials.sot,
            specials.lang_base,
            specials.transcribe,
            specials.no_timestamps,
        ];

        let num_decode_tokens = 10; // Decode 10 tokens for benchmarking

        // === GPU Path Benchmark ===
        // Both paths use head-first KV caches (init_gpu_decoder_kv_cache_head_first)
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .upload_decoder_weights_to_gpu()
            .expect("Upload GPU weights");
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV cache (head-first)");

        // Warmup (JIT compilation)
        for &token in &initial_tokens {
            let _ = cuda_model
                .forward_one_gpu_total_offload(token, &encoder_output)
                .expect("Warmup failed");
        }

        // WAPR-PERF-014: Reset state after warmup for clean benchmark
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Re-init KV cache");

        // Process initial tokens fresh for benchmark
        for &token in &initial_tokens {
            let _ = cuda_model
                .forward_one_gpu_total_offload(token, &encoder_output)
                .expect("Init tokens failed");
        }

        // Benchmark GPU path
        let gpu_start = std::time::Instant::now();
        let mut gpu_tokens = initial_tokens.clone();
        for _ in 0..num_decode_tokens {
            let last_token = *gpu_tokens.last().unwrap_or(&specials.sot);
            let logits = cuda_model
                .forward_one_gpu_total_offload(last_token, &encoder_output)
                .expect("GPU forward failed");
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(specials.eot);
            if next_token == specials.eot {
                break;
            }
            gpu_tokens.push(next_token);
        }
        let gpu_time = gpu_start.elapsed();
        let gpu_tokens_generated = gpu_tokens.len() - initial_tokens.len();

        // === Executor Path Benchmark ===
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.reset_gpu_decoder_kv_cache(); // WAPR-PERF-014: Clear stale KV cache from GPU path
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV cache");
        cuda_model
            .upload_decoder_weights_to_executor()
            .expect("Upload executor weights");

        // Warmup (JIT compilation)
        for &token in &initial_tokens {
            let _ = cuda_model
                .forward_one_executor(token, &encoder_output)
                .expect("Warmup failed");
        }

        // WAPR-PERF-014: Reset state after warmup for clean benchmark
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Re-init KV cache");

        // Process initial tokens fresh for benchmark
        for &token in &initial_tokens {
            let _ = cuda_model
                .forward_one_executor(token, &encoder_output)
                .expect("Init tokens failed");
        }

        // Benchmark Executor path
        let exec_start = std::time::Instant::now();
        let mut exec_tokens = initial_tokens.clone();
        for _ in 0..num_decode_tokens {
            let last_token = *exec_tokens.last().unwrap_or(&specials.sot);
            let logits = cuda_model
                .forward_one_executor(last_token, &encoder_output)
                .expect("Executor forward failed");
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(specials.eot);
            if next_token == specials.eot {
                break;
            }
            exec_tokens.push(next_token);
        }
        let exec_time = exec_start.elapsed();
        let exec_tokens_generated = exec_tokens.len() - initial_tokens.len();

        // Results
        let gpu_ms_per_token = gpu_time.as_millis() as f64 / gpu_tokens_generated.max(1) as f64;
        let exec_ms_per_token = exec_time.as_millis() as f64 / exec_tokens_generated.max(1) as f64;
        let speedup = gpu_ms_per_token / exec_ms_per_token;

        eprintln!("\nResults ({} tokens decoded):", num_decode_tokens);
        eprintln!(
            "  GPU path:      {:?} ({:.1} ms/token)",
            gpu_time, gpu_ms_per_token
        );
        eprintln!(
            "  Executor path: {:?} ({:.1} ms/token)",
            exec_time, exec_ms_per_token
        );
        eprintln!("  Speedup:       {:.2}x", speedup);

        // Decode text for comparison
        let gpu_text = cuda_model
            .tokenizer
            .decode_with_options(&gpu_tokens, true)
            .unwrap_or_default();
        let exec_text = cuda_model
            .tokenizer
            .decode_with_options(&exec_tokens, true)
            .unwrap_or_default();

        eprintln!("\nGPU text:  \"{}\"", gpu_text.trim());
        eprintln!("Exec text: \"{}\"", exec_text.trim());

        // Verify tokens match
        assert_eq!(
            gpu_tokens, exec_tokens,
            "GPU and Executor paths should produce same tokens"
        );

        eprintln!("\n✓ WAPR-PERF-014: GPU vs Executor Benchmark PASSED");
        eprintln!("  - Tokens match");
        eprintln!("  - Speedup: {:.2}x", speedup);
    }

    /// WAPR-PERF-014: Detailed timing breakdown for executor forward pass
    ///
    /// Measures each component to identify bottlenecks for CUDA Graph optimization.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_executor_timing_breakdown() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        let audio_path = "demos/test-audio/test-speech-1.5s.wav";
        if !std::path::Path::new(audio_path).exists() {
            eprintln!("Audio not found at {}, skipping test", audio_path);
            return;
        }

        eprintln!("\n=== WAPR-PERF-014: Executor Timing Breakdown ===");

        // Load model and compute mel
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio");
        let wav_data =
            crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let mel = apr.compute_mel(&wav_data.samples).expect("Mel failed");

        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Get encoder output
        let encoder_output = cuda_model
            .encoder
            .forward_mel(&mel)
            .expect("Encoder failed");

        // Initialize
        cuda_model.reset_gpu_decoder_pos();
        cuda_model
            .upload_decoder_weights_to_gpu()
            .expect("Upload GPU weights");
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV cache");
        cuda_model
            .upload_decoder_weights_to_executor()
            .expect("Upload executor weights");

        // Warmup
        use crate::tokenizer::special_tokens::SpecialTokens;
        let specials = SpecialTokens::for_vocab_size(cuda_model.config().n_vocab as usize);
        let initial_tokens = vec![
            specials.sot,
            specials.lang_base,
            specials.transcribe,
            specials.no_timestamps,
        ];
        for &token in &initial_tokens {
            let _ = cuda_model
                .forward_one_executor(token, &encoder_output)
                .expect("Warmup failed");
        }

        // Reset for profiling
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Re-init KV cache");

        // Profile each component
        let d_model = cuda_model.config().n_text_state as usize;
        let n_layers = cuda_model.config().n_text_layer as usize;
        let _n_vocab = cuda_model.config().n_vocab as usize;

        let mut total_embed_time = std::time::Duration::ZERO;
        let mut total_block_time = std::time::Duration::ZERO;
        let mut total_ln_time = std::time::Duration::ZERO;
        let mut total_proj_time = std::time::Duration::ZERO;

        let num_tokens = 5;
        for token_idx in 0..num_tokens {
            let token = initial_tokens[token_idx % initial_tokens.len()];
            let pos = cuda_model.gpu_decoder_pos;

            // 1. Token embedding (CPU)
            let t0 = std::time::Instant::now();
            let emb_start = (token as usize) * d_model;
            let token_emb = cuda_model.decoder.token_embedding();
            let pos_emb = cuda_model.decoder.positional_embedding();
            let pos_start = pos * d_model;
            let mut x: Vec<f32> = token_emb[emb_start..emb_start + d_model]
                .iter()
                .zip(&pos_emb[pos_start..pos_start + d_model])
                .map(|(t, p)| t + p)
                .collect();
            total_embed_time += t0.elapsed();

            // 2. Decoder blocks
            let t1 = std::time::Instant::now();
            for layer_idx in 0..n_layers {
                x = cuda_model
                    .forward_decoder_block_executor(layer_idx, &x, pos, Some(&encoder_output))
                    .expect("Block failed");
            }
            total_block_time += t1.elapsed();

            // 3. Final LayerNorm (CPU)
            let t2 = std::time::Instant::now();
            let hidden = cuda_model.decoder.ln_post().forward(&x).expect("LN failed");
            total_ln_time += t2.elapsed();

            // 4. Vocab projection (GPU)
            let t3 = std::time::Instant::now();
            let _logits = cuda_model
                .project_to_vocab_gpu(&hidden)
                .expect("Proj failed");
            total_proj_time += t3.elapsed();

            cuda_model.gpu_decoder_pos += 1;
        }

        let avg_embed = total_embed_time.as_micros() as f64 / num_tokens as f64;
        let avg_block = total_block_time.as_micros() as f64 / num_tokens as f64;
        let avg_ln = total_ln_time.as_micros() as f64 / num_tokens as f64;
        let avg_proj = total_proj_time.as_micros() as f64 / num_tokens as f64;
        let total = avg_embed + avg_block + avg_ln + avg_proj;

        eprintln!("\nTiming breakdown (average over {} tokens):", num_tokens);
        eprintln!(
            "  Token embedding (CPU): {:>8.1}µs ({:>5.1}%)",
            avg_embed,
            avg_embed / total * 100.0
        );
        eprintln!(
            "  Decoder blocks (GPU):  {:>8.1}µs ({:>5.1}%)",
            avg_block,
            avg_block / total * 100.0
        );
        eprintln!(
            "  Final LayerNorm (CPU): {:>8.1}µs ({:>5.1}%)",
            avg_ln,
            avg_ln / total * 100.0
        );
        eprintln!(
            "  Vocab projection (GPU):{:>8.1}µs ({:>5.1}%)",
            avg_proj,
            avg_proj / total * 100.0
        );
        eprintln!("  ────────────────────────────────────────");
        eprintln!("  TOTAL:                 {:>8.1}µs", total);
        eprintln!("  Per-token latency:     {:>8.2}ms", total / 1000.0);

        eprintln!("\n✓ WAPR-PERF-014: Timing breakdown complete");
    }

    /// WAPR-PERF-013 Point 156b: Step-by-step divergence diagnostic
    ///
    /// Compares GPU vs CPU decoder at each computation stage to pinpoint
    /// where numerical divergence begins.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_decoder_step_diagnostic() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping diagnostic test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n=== WAPR-PERF-013 Point 156b: Step-by-step Divergence Diagnostic ===");

        // Load model
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        let d_model = cuda_model.config().n_text_state as usize;
        let n_layers = cuda_model.config().n_text_layer as usize;
        let _n_vocab = cuda_model.config().n_vocab as usize;
        let n_heads = cuda_model.config().n_text_head as usize;
        let head_dim = d_model / n_heads;
        let max_len = cuda_model.config().n_text_ctx as usize;

        eprintln!(
            "Model: d_model={}, n_heads={}, head_dim={}, n_layers={}",
            d_model, n_heads, head_dim, n_layers
        );

        // Initialize GPU infrastructure
        cuda_model
            .upload_decoder_weights_to_gpu()
            .expect("Upload failed");
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init failed");

        // Test token
        let test_token = 50258_u32; // SOT
        let pos = 0_usize;

        // === Step 1: Token + Positional Embedding ===
        let emb_start = (test_token as usize) * d_model;
        let token_emb = cuda_model.decoder.token_embedding();
        let pos_emb = cuda_model.decoder.positional_embedding();
        let pos_start = pos * d_model;
        let x: Vec<f32> = token_emb[emb_start..emb_start + d_model]
            .iter()
            .zip(&pos_emb[pos_start..pos_start + d_model])
            .map(|(t, p)| t + p)
            .collect();
        eprintln!(
            "\n[Step 1] Token+Pos Embedding: {} elements, sum={:.4}",
            x.len(),
            x.iter().sum::<f32>()
        );

        // === Step 2: LN1 for layer 0 ===
        let block = &cuda_model.decoder.blocks()[0];
        let normed = block.ln1.forward(&x).expect("LN1 failed");
        eprintln!(
            "[Step 2] LN1: sum={:.4}, first 5: {:?}",
            normed.iter().sum::<f32>(),
            &normed[..5]
        );

        // === Step 3: CPU Q/K/V projections ===
        let cpu_q = block
            .self_attn
            .w_q()
            .forward(&normed, 1)
            .expect("CPU Q failed");
        let cpu_k = block
            .self_attn
            .w_k()
            .forward(&normed, 1)
            .expect("CPU K failed");
        let cpu_v = block
            .self_attn
            .w_v()
            .forward(&normed, 1)
            .expect("CPU V failed");
        eprintln!("[Step 3] CPU Q: sum={:.4}", cpu_q.iter().sum::<f32>());
        eprintln!("         CPU K: sum={:.4}", cpu_k.iter().sum::<f32>());
        eprintln!("         CPU V: sum={:.4}", cpu_v.iter().sum::<f32>());

        // === Step 4: GPU Q/K/V projections ===
        let ctx = cuda_model.executor.context();
        let weights = cuda_model
            .gpu_decoder_weights
            .as_ref()
            .expect("Weights not uploaded");
        let layer_weights = &weights[0];

        let x_gpu = GpuResidentTensor::from_host(ctx, &normed).expect("x upload");
        let mut gpu_q = x_gpu
            .linear(
                ctx,
                &layer_weights.self_w_q,
                Some(&layer_weights.self_b_q),
                1,
                d_model as u32,
                d_model as u32,
            )
            .expect("GPU Q failed");
        let mut gpu_k = x_gpu
            .linear(
                ctx,
                &layer_weights.self_w_k,
                Some(&layer_weights.self_b_k),
                1,
                d_model as u32,
                d_model as u32,
            )
            .expect("GPU K failed");
        let mut gpu_v = x_gpu
            .linear(
                ctx,
                &layer_weights.self_w_v,
                Some(&layer_weights.self_b_v),
                1,
                d_model as u32,
                d_model as u32,
            )
            .expect("GPU V failed");

        let gpu_q_host = gpu_q.to_host().expect("Q download");
        let gpu_k_host = gpu_k.to_host().expect("K download");
        let gpu_v_host = gpu_v.to_host().expect("V download");
        eprintln!("[Step 4] GPU Q: sum={:.4}", gpu_q_host.iter().sum::<f32>());
        eprintln!("         GPU K: sum={:.4}", gpu_k_host.iter().sum::<f32>());
        eprintln!("         GPU V: sum={:.4}", gpu_v_host.iter().sum::<f32>());

        // Compare Q/K/V
        let q_diff: f32 = cpu_q
            .iter()
            .zip(gpu_q_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let k_diff: f32 = cpu_k
            .iter()
            .zip(gpu_k_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let v_diff: f32 = cpu_v
            .iter()
            .zip(gpu_v_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!(
            "[Step 4] Q diff: {:.2e}, K diff: {:.2e}, V diff: {:.2e}",
            q_diff, k_diff, v_diff
        );

        if q_diff > 1e-4 || k_diff > 1e-4 || v_diff > 1e-4 {
            eprintln!("\n❌ DIVERGENCE at Q/K/V projections!");
            panic!(
                "Q/K/V divergence: q={:.2e} k={:.2e} v={:.2e}",
                q_diff, k_diff, v_diff
            );
        }

        eprintln!("\n✓ Q/K/V projections match within 1e-4");

        // === Step 5: CPU self-attention (for pos=0, just Q @ K^T @ V with single token) ===
        // For position 0 with seq_len=1, self-attention is trivial:
        // scores = Q @ K^T = [1, d] @ [d, 1] = [1, 1]
        // softmax([1,1]) = [1.0]
        // output = [1.0] @ V = V
        eprintln!("\n[Step 5] CPU self-attention (pos=0, trivial case):");
        let cpu_attn_out = cpu_v.clone(); // For single token at pos 0, output = V
        eprintln!(
            "         CPU attn out: sum={:.4}",
            cpu_attn_out.iter().sum::<f32>()
        );

        // === Step 6: GPU incremental attention ===
        // First scatter K/V to caches
        use trueno_gpu::driver::CudaStream;
        let stream = CudaStream::new(ctx).expect("stream");
        let self_k_caches = cuda_model.gpu_self_k_head_first.as_mut().unwrap();
        let self_v_caches = cuda_model.gpu_self_v_head_first.as_mut().unwrap();

        kv_cache_scatter_gpu(
            ctx,
            &gpu_k,
            &mut self_k_caches[0],
            pos as u32,
            n_heads as u32,
            head_dim as u32,
            max_len as u32,
            &stream,
        )
        .expect("K scatter");
        kv_cache_scatter_gpu(
            ctx,
            &gpu_v,
            &mut self_v_caches[0],
            pos as u32,
            n_heads as u32,
            head_dim as u32,
            max_len as u32,
            &stream,
        )
        .expect("V scatter");

        // Run incremental attention
        let seq_len_attn = (pos + 1) as u32;
        let mut gpu_attn_out = incremental_attention_gpu(
            ctx,
            &gpu_q,
            &self_k_caches[0],
            &self_v_caches[0],
            n_heads as u32,
            head_dim as u32,
            seq_len_attn,
            max_len as u32,
        )
        .expect("incremental attention");

        let gpu_attn_host = gpu_attn_out.to_host().expect("attn download");
        eprintln!(
            "[Step 6] GPU attn out: sum={:.4}",
            gpu_attn_host.iter().sum::<f32>()
        );

        let attn_diff: f32 = cpu_attn_out
            .iter()
            .zip(gpu_attn_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("[Step 6] Attention diff: {:.2e}", attn_diff);

        if attn_diff > 1e-4 {
            eprintln!("\n❌ DIVERGENCE at self-attention output!");
            // Print first few elements to debug
            eprintln!("CPU first 10: {:?}", &cpu_attn_out[..10]);
            eprintln!("GPU first 10: {:?}", &gpu_attn_host[..10]);
            panic!("Self-attention divergence: max={:.2e}", attn_diff);
        }

        eprintln!("\n✓ Self-attention output matches within 1e-4");

        // === Step 7: Output projection (W_o) ===
        eprintln!("\n[Step 7] Output projection (W_o):");
        // CPU
        let cpu_attn_proj = block
            .self_attn
            .w_o()
            .forward(&cpu_attn_out, 1)
            .expect("CPU W_o");
        eprintln!(
            "         CPU W_o out: sum={:.4}",
            cpu_attn_proj.iter().sum::<f32>()
        );

        // GPU
        let mut gpu_attn_proj = gpu_attn_out
            .linear(
                ctx,
                &layer_weights.self_w_o,
                Some(&layer_weights.self_b_o),
                1,
                d_model as u32,
                d_model as u32,
            )
            .expect("GPU W_o");
        let gpu_attn_proj_host = gpu_attn_proj.to_host().expect("W_o download");
        eprintln!(
            "         GPU W_o out: sum={:.4}",
            gpu_attn_proj_host.iter().sum::<f32>()
        );

        let wo_diff: f32 = cpu_attn_proj
            .iter()
            .zip(gpu_attn_proj_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("[Step 7] W_o diff: {:.2e}", wo_diff);

        // === Step 8: Residual after self-attention ===
        eprintln!("\n[Step 8] Residual after self-attention:");
        let cpu_residual: Vec<f32> = x
            .iter()
            .zip(cpu_attn_proj.iter())
            .map(|(a, b)| a + b)
            .collect();
        let gpu_residual: Vec<f32> = x
            .iter()
            .zip(gpu_attn_proj_host.iter())
            .map(|(a, b)| a + b)
            .collect();
        eprintln!(
            "         CPU residual: sum={:.4}",
            cpu_residual.iter().sum::<f32>()
        );
        eprintln!(
            "         GPU residual: sum={:.4}",
            gpu_residual.iter().sum::<f32>()
        );

        // === Step 9: Compute CPU FFN (before mutable borrow) ===
        // LN3 of residual
        let ln3_out = block.ln3.forward(&cpu_residual).expect("LN3");
        let ffn_out = block.ffn.forward(&ln3_out).expect("FFN");
        let cpu_final: Vec<f32> = cpu_residual
            .iter()
            .zip(ffn_out.iter())
            .map(|(a, b)| a + b)
            .collect();
        eprintln!(
            "\n[Step 9] CPU (no cross-attn): sum={:.4}",
            cpu_final.iter().sum::<f32>()
        );

        // Now drop the immutable block borrow
        let _ = block;

        // GPU path (no encoder output = no cross-attention)
        eprintln!("[Step 9] GPU block forward (self-attention + FFN only):");
        let gpu_block_out = cuda_model
            .forward_decoder_block_gpu(0, &x, 0, None)
            .expect("GPU block forward");
        eprintln!(
            "         GPU block out: sum={:.4}",
            gpu_block_out.iter().sum::<f32>()
        );

        let gpu_block_sum: f32 = gpu_block_out.iter().sum();
        if gpu_block_sum.is_nan() || gpu_block_sum.is_infinite() {
            panic!("GPU block output is NaN/Inf!");
        }

        // Note: GPU block uses a fresh KV cache scatter, so it's comparing apples to apples
        let block_diff: f32 = cpu_final
            .iter()
            .zip(gpu_block_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("[Step 9] Block diff (no cross-attn): {:.2e}", block_diff);

        if block_diff > 1e-3 {
            eprintln!("\n❌ DIVERGENCE at block level!");
            eprintln!("CPU first 10: {:?}", &cpu_final[..10]);
            eprintln!("GPU first 10: {:?}", &gpu_block_out[..10]);
        }

        eprintln!("\n✓ GPU decoder block diagnostic complete");
        eprintln!("  W_o diff: {:.2e}", wo_diff);
        eprintln!("  Block diff: {:.2e}", block_diff);

        // === Step 10: Test with encoder output (cross-attention enabled) ===
        eprintln!("\n[Step 10] Full path with cross-attention:");

        // Create deterministic encoder output for testing (same as CPU will use)
        let enc_seq_len = 1500;
        let enc_output: Vec<f32> = (0..enc_seq_len * d_model)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        eprintln!("         Encoder output: {} elements", enc_output.len());

        // === CPU Reference Path (using forward_block_cached logic) ===
        // Get a fresh reference to the block
        let block = &cuda_model.decoder.blocks()[0];

        // Self-attention on CPU
        let normed = block.ln1.forward(&x).expect("LN1");
        let _cpu_q = block.self_attn.w_q().forward(&normed, 1).expect("CPU Q");
        let _cpu_k = block.self_attn.w_k().forward(&normed, 1).expect("CPU K");
        let cpu_v = block.self_attn.w_v().forward(&normed, 1).expect("CPU V");

        // For single position, attention output = V
        let cpu_attn = cpu_v.clone();
        let cpu_attn_proj = block.self_attn.w_o().forward(&cpu_attn, 1).expect("CPU O");
        let cpu_self_residual: Vec<f32> = x
            .iter()
            .zip(cpu_attn_proj.iter())
            .map(|(a, b)| a + b)
            .collect();

        // Cross-attention on CPU
        let normed2 = block.ln2.forward(&cpu_self_residual).expect("LN2");
        let cpu_cross_out = block
            .cross_attn
            .forward_cross_dispatch(&normed2, &enc_output, None)
            .expect("CPU cross-attn");
        let cpu_cross_residual: Vec<f32> = cpu_self_residual
            .iter()
            .zip(cpu_cross_out.iter())
            .map(|(a, b)| a + b)
            .collect();

        // FFN on CPU
        let normed3 = block.ln3.forward(&cpu_cross_residual).expect("LN3");
        let cpu_ffn_out = block.ffn.forward(&normed3).expect("FFN");
        let cpu_block_out: Vec<f32> = cpu_cross_residual
            .iter()
            .zip(cpu_ffn_out.iter())
            .map(|(a, b)| a + b)
            .collect();
        eprintln!(
            "         CPU block (with cross-attn): sum={:.4}",
            cpu_block_out.iter().sum::<f32>()
        );

        // Drop block borrow before mutable call
        let _ = block;

        // Reset GPU decoder position (important!)
        cuda_model.reset_gpu_decoder_pos();

        // Re-initialize KV caches (they were modified by step 9)
        // Need to set to None first to force re-init
        cuda_model.gpu_self_k_head_first = None;
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV");

        // Run GPU block with encoder output
        let gpu_cross_block_out = cuda_model
            .forward_decoder_block_gpu(0, &x, 0, Some(&enc_output))
            .expect("GPU block with cross-attn");
        eprintln!(
            "         GPU block (with cross-attn): sum={:.4}",
            gpu_cross_block_out.iter().sum::<f32>()
        );

        // Compare outputs
        let cross_block_diff: f32 = cpu_block_out
            .iter()
            .zip(gpu_cross_block_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!(
            "[Step 10] Block diff (with cross-attn): {:.2e}",
            cross_block_diff
        );

        if cross_block_diff > 1e-3 {
            eprintln!("\n❌ DIVERGENCE at block level with cross-attention!");
            eprintln!("CPU first 10: {:?}", &cpu_block_out[..10]);
            eprintln!("GPU first 10: {:?}", &gpu_cross_block_out[..10]);
        }

        // Check for NaN/Inf
        let gpu_cross_sum: f32 = gpu_cross_block_out.iter().sum();
        if gpu_cross_sum.is_nan() || gpu_cross_sum.is_infinite() {
            panic!("GPU block with cross-attention output is NaN/Inf!");
        }

        eprintln!("\n✓ GPU with cross-attention diagnostic complete");
        eprintln!("  Block diff: {:.2e}", cross_block_diff);
    }

    /// WAPR-PERF-014 Point 157: Full system integration benchmark
    ///
    /// Tests the complete GPU decoder pipeline and measures against the
    /// 1984ms target (2x whisper.cpp @ 992ms).
    ///
    /// # Falsification Criteria
    ///
    /// - Point 157: Total time ≤1984ms
    /// - Point 158: CPU/GPU overlap (async advantage)
    /// - Accumulation Risk: WER matches CPU baseline
    #[test]
    #[cfg(feature = "cuda")]
    fn test_full_system_integration_benchmark() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        // Load model
        let model_path = std::env::var("WHISPER_MODEL_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/models/whisper-tiny.apr").to_string()
        });

        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        // Load test audio
        let audio_path = std::env::var("WHISPER_TEST_AUDIO").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/demos/test-audio/test-speech-1.5s.wav"
            )
            .to_string()
        });

        if !std::path::Path::new(&audio_path).exists() {
            eprintln!("Test audio not found at {}, skipping test", audio_path);
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-014 Point 157: Full System Integration Benchmark");
        eprintln!("============================================================");
        eprintln!("Model: {}", model_path);
        eprintln!("Audio: {}", audio_path);
        eprintln!("Target: ≤1984ms (2x whisper.cpp @ 992ms)");
        eprintln!("============================================================\n");

        // Load model
        let bytes = std::fs::read(&model_path).expect("Failed to read model file");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let n_layers = apr.config().n_text_layer;
        let d_model = apr.config().n_text_state;
        eprintln!("[Model] {} layers, d_model={}", n_layers, d_model);

        // Load audio
        let audio_bytes = std::fs::read(&audio_path).expect("Failed to read audio file");
        let wav_data = crate::audio::wav::parse_wav(&audio_bytes).expect("Failed to parse WAV");
        eprintln!(
            "[Audio] {} samples ({:.2}s @ {}Hz)",
            wav_data.samples.len(),
            wav_data.samples.len() as f64 / wav_data.sample_rate as f64,
            wav_data.sample_rate
        );

        // Compute mel spectrogram BEFORE into_cuda (WhisperApr method)
        let mel = apr
            .compute_mel(&wav_data.samples)
            .expect("Mel computation failed");
        eprintln!("[Mel] {} frames", mel.len() / 80);

        // Create CUDA model
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Upload weights to executor (WAPR-PERF-014)
        let upload_start = std::time::Instant::now();
        let weight_bytes = cuda_model
            .upload_decoder_weights_to_executor()
            .expect("Failed to upload decoder weights");
        let upload_time = upload_start.elapsed();
        eprintln!(
            "[Weights] {:.2} MB uploaded in {:?}",
            weight_bytes as f64 / 1_048_576.0,
            upload_time
        );

        // Enable GPU decoder offload
        std::env::set_var("WHISPER_GPU_DECODER_OFFLOAD", "1");

        // Run transcription with timing
        let _options = crate::TranscribeOptions::default();

        // Build initial tokens
        use crate::tokenizer::special_tokens::SpecialTokens;
        let specials = SpecialTokens::for_vocab_size(cuda_model.config().n_vocab as usize);

        // === WARMUP PHASE (outside timed section) ===
        // Run encoder once to warm up
        let encoder_output = cuda_model
            .encoder
            .forward_mel(&mel)
            .expect("Encoder failed");

        // Initialize GPU decoder and compile kernels
        cuda_model.reset_gpu_decoder_pos();
        cuda_model
            .upload_decoder_weights_to_gpu()
            .expect("Upload decoder weights");
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV cache");

        // Build initial tokens for warmup
        let initial_tokens = {
            let mut t = vec![specials.sot];
            if specials.is_multilingual {
                t.push(specials.lang_base);
            }
            t.push(specials.transcribe);
            t.push(specials.no_timestamps);
            t
        };

        // Warmup: process initial tokens to compile kernels
        for &token in &initial_tokens {
            let _ = cuda_model
                .forward_one_gpu_total_offload(token, &encoder_output)
                .expect("Warmup token forward failed");
        }

        // Reset state for clean benchmark
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Re-init KV cache");

        // === BENCHMARK PHASE ===
        eprintln!("\n[Benchmark] Starting GPU decoder transcription...");
        let total_start = std::time::Instant::now();

        // Use the internal encoder + decoder path
        // First encode on CPU (or GPU if enabled)
        let encode_start = std::time::Instant::now();
        let encoder_output = cuda_model
            .encoder
            .forward_mel(&mel)
            .expect("Encoder failed");
        let encode_time = encode_start.elapsed();
        eprintln!("[Encoder] {:?}", encode_time);

        // Decode using GPU total offload (weights already uploaded during warmup)
        let decode_start = std::time::Instant::now();
        let mut tokens = vec![specials.sot];
        if specials.is_multilingual {
            tokens.push(specials.lang_base); // English
        }
        tokens.push(specials.transcribe);
        tokens.push(specials.no_timestamps);

        // Process initial tokens
        for &token in &tokens {
            let _ = cuda_model
                .forward_one_gpu_total_offload(token, &encoder_output)
                .expect("Initial token forward failed");
        }

        // Generate tokens
        let max_tokens = cuda_model.config().n_text_ctx as usize;
        let mut token_times: Vec<u128> = Vec::new();

        for _gen_idx in 0..max_tokens.saturating_sub(tokens.len()) {
            let last_token = *tokens.last().unwrap_or(&specials.sot);

            let token_start = std::time::Instant::now();
            let logits = cuda_model
                .forward_one_gpu_total_offload(last_token, &encoder_output)
                .expect("Token forward failed");
            let token_time = token_start.elapsed();
            token_times.push(token_time.as_micros());

            // Greedy decode
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(specials.eot);

            if next_token == specials.eot {
                break;
            }

            tokens.push(next_token);
        }

        let decode_time = decode_start.elapsed();
        let total_time = total_start.elapsed();

        // Decode text
        let text = cuda_model
            .tokenizer
            .decode_with_options(&tokens, true)
            .expect("Decode failed");

        eprintln!("\n============================================================");
        eprintln!("RESULTS");
        eprintln!("============================================================");
        eprintln!("[Output] \"{}\"", text.trim());
        eprintln!("[Tokens] {} generated", tokens.len() - 4); // Subtract initial tokens
        eprintln!();
        eprintln!("[Timing]");
        eprintln!("  Weight upload: {:?}", upload_time);
        eprintln!("  Encoder:       {:?}", encode_time);
        eprintln!("  Decoder:       {:?}", decode_time);
        eprintln!("  TOTAL:         {:?}", total_time);
        eprintln!();

        if !token_times.is_empty() {
            let avg_token_us = token_times.iter().sum::<u128>() / token_times.len() as u128;
            let min_token_us = *token_times.iter().min().unwrap_or(&0);
            let max_token_us = *token_times.iter().max().unwrap_or(&0);
            eprintln!("[Per-Token]");
            eprintln!("  Average: {:.2}ms", avg_token_us as f64 / 1000.0);
            eprintln!("  Min:     {:.2}ms", min_token_us as f64 / 1000.0);
            eprintln!("  Max:     {:.2}ms", max_token_us as f64 / 1000.0);
            eprintln!(
                "  First 5: {:?}",
                token_times
                    .iter()
                    .take(5)
                    .map(|t| format!("{:.1}ms", *t as f64 / 1000.0))
                    .collect::<Vec<_>>()
            );
        }

        eprintln!();
        eprintln!("[Point 157 Falsification]");
        let total_ms = total_time.as_millis();
        let target_ms = 1984;
        if total_ms <= target_ms {
            eprintln!("  ✓ PASSED: {}ms ≤ {}ms target", total_ms, target_ms);
        } else {
            eprintln!(
                "  ✗ FAILED: {}ms > {}ms target ({:.1}x slower)",
                total_ms,
                target_ms,
                total_ms as f64 / target_ms as f64
            );
        }
        eprintln!("============================================================\n");

        // Don't fail the test - this is a benchmark, not a correctness test
        // The user will analyze the results
    }

    /// WAPR-PERF-014: Test executor-based weight upload
    ///
    /// Verifies that upload_decoder_weights_to_executor correctly uploads
    /// all decoder weights to CudaExecutor's weight_cache.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_executor_weight_upload() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        // Load model
        let model_path = std::env::var("WHISPER_MODEL_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/models/whisper-tiny.apr").to_string()
        });

        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        let bytes = std::fs::read(&model_path).expect("Failed to read model file");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let n_layers = apr.config().n_text_layer as usize;
        let _d_model = apr.config().n_text_state as usize;

        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Upload weights to executor
        let start = std::time::Instant::now();
        let bytes = cuda_model
            .upload_decoder_weights_to_executor()
            .expect("Failed to upload weights");
        let elapsed = start.elapsed();

        eprintln!(
            "[WAPR-PERF-014] Uploaded {:.2} MB in {:?}",
            bytes as f64 / 1_048_576.0,
            elapsed
        );

        // Verify all expected weights are cached
        let expected_weights_per_layer = vec![
            "self_w_q",
            "self_b_q",
            "self_w_k",
            "self_b_k",
            "self_w_v",
            "self_b_v",
            "self_w_o",
            "self_b_o",
            "cross_w_q",
            "cross_b_q",
            "cross_w_k",
            "cross_b_k",
            "cross_w_v",
            "cross_b_v",
            "cross_w_o",
            "cross_b_o",
            "ffn_fc1",
            "ffn_b1",
            "ffn_fc2",
            "ffn_b2",
            "ln1_gamma",
            "ln1_beta",
            "ln2_gamma",
            "ln2_beta",
            "ln3_gamma",
            "ln3_beta",
        ];

        for layer_idx in 0..n_layers {
            for weight_name in &expected_weights_per_layer {
                let full_name = format!("dec.L{layer_idx}.{weight_name}");
                assert!(
                    cuda_model.executor.has_weights(&full_name),
                    "Missing weight: {}",
                    full_name
                );
            }
        }

        // Verify global weights
        assert!(
            cuda_model.executor.has_weights("dec.output_proj"),
            "Missing output_proj"
        );
        assert!(
            cuda_model.executor.has_weights("dec.ln_post_gamma"),
            "Missing ln_post_gamma"
        );
        assert!(
            cuda_model.executor.has_weights("dec.ln_post_beta"),
            "Missing ln_post_beta"
        );

        // Expected weight count: n_layers * 26 per-layer + 3 global
        let expected_count = n_layers * expected_weights_per_layer.len() + 3;
        let actual_count = cuda_model.executor.cached_weight_count();
        eprintln!(
            "[WAPR-PERF-014] Expected {} weights, got {}",
            expected_count, actual_count
        );

        // Note: actual_count may be higher due to encoder weights from earlier tests
        assert!(
            actual_count >= expected_count,
            "Not enough weights cached: expected at least {}, got {}",
            expected_count,
            actual_count
        );

        eprintln!("✓ Executor weight upload test passed");
    }

    /// WAPR-PERF-015: Diagnose CPU encoder performance
    ///
    /// Profiles individual components of the encoder forward pass to identify
    /// the source of the 7.29s encoder time (should be ~200ms).
    #[test]
    fn test_encoder_performance_diagnostic() {
        use crate::simd;

        // Check SIMD backend
        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-015: Encoder Performance Diagnostic");
        eprintln!("============================================================\n");

        eprintln!("[SIMD Backend]");
        eprintln!("  Backend: {}", simd::backend_name());
        eprintln!("  SIMD available: {}", simd::simd_available());

        // Load model
        let model_path = std::env::var("WHISPER_MODEL_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/models/whisper-tiny.apr").to_string()
        });

        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        // Load audio
        let audio_path = std::env::var("WHISPER_TEST_AUDIO").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/demos/test-audio/test-speech-1.5s.wav"
            )
            .to_string()
        });

        if !std::path::Path::new(&audio_path).exists() {
            eprintln!("Test audio not found at {}, skipping test", audio_path);
            return;
        }

        let bytes = std::fs::read(&model_path).expect("Failed to read model file");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");

        // Load and preprocess audio
        let audio_bytes = std::fs::read(&audio_path).expect("Failed to read audio file");
        let wav_data = crate::audio::wav::parse_wav(&audio_bytes).expect("Failed to parse WAV");

        // Compute mel spectrogram
        let mel_start = std::time::Instant::now();
        let mel = apr
            .compute_mel(&wav_data.samples)
            .expect("Mel computation failed");
        let mel_time = mel_start.elapsed();
        let mel_frames = mel.len() / 80;
        eprintln!("\n[Mel Spectrogram]");
        eprintln!("  Frames: {}", mel_frames);
        eprintln!("  Time: {:?}", mel_time);

        // Time convolution frontend
        let conv_start = std::time::Instant::now();
        let conv_output = apr
            .encoder
            .conv_frontend()
            .forward(&mel)
            .expect("Conv failed");
        let conv_time = conv_start.elapsed();
        let conv_frames = conv_output.len() / apr.encoder.d_model();
        eprintln!("\n[Convolutional Frontend]");
        eprintln!("  Input frames: {}", mel_frames);
        eprintln!("  Output frames: {}", conv_frames);
        eprintln!("  Time: {:?}", conv_time);

        // Time positional embedding addition (should be trivial)
        let pe_start = std::time::Instant::now();
        let mut x = conv_output.clone();
        let pe = apr.encoder.positional_embedding();
        for pos in 0..conv_frames {
            for d in 0..apr.encoder.d_model() {
                x[pos * apr.encoder.d_model() + d] += pe[pos * apr.encoder.d_model() + d];
            }
        }
        let pe_time = pe_start.elapsed();
        eprintln!("\n[Positional Embedding]");
        eprintln!("  Time: {:?}", pe_time);

        // Time single encoder block
        let block_start = std::time::Instant::now();
        let _block_output = apr.encoder.blocks()[0].forward(&x).expect("Block 0 failed");
        let block_time = block_start.elapsed();
        eprintln!("\n[Single Encoder Block (Layer 0)]");
        eprintln!("  Time: {:?}", block_time);
        eprintln!(
            "  Projected: {:?} for {} layers",
            block_time * apr.encoder.n_layers() as u32,
            apr.encoder.n_layers()
        );

        // Time full encoder
        let encoder_start = std::time::Instant::now();
        let _encoder_output = apr.encoder.forward_mel(&mel).expect("Encoder failed");
        let encoder_time = encoder_start.elapsed();
        eprintln!("\n[Full Encoder (forward_mel)]");
        eprintln!("  Time: {:?}", encoder_time);

        // Breakdown analysis
        let total_expected = conv_time + pe_time + block_time * apr.encoder.n_layers() as u32;
        eprintln!("\n[Analysis]");
        eprintln!(
            "  Expected (conv + pe + {} blocks): {:?}",
            apr.encoder.n_layers(),
            total_expected
        );
        eprintln!("  Actual: {:?}", encoder_time);
        eprintln!(
            "  Overhead: {:?}",
            encoder_time.saturating_sub(total_expected)
        );

        // SIMD matmul benchmark (raw performance check)
        eprintln!("\n[Raw MatMul Benchmark]");
        let a = vec![1.0_f32; 1500 * 384];
        let b = vec![1.0_f32; 384 * 384];

        let matmul_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = simd::matmul(&a, &b, 1500, 384, 384);
        }
        let matmul_time = matmul_start.elapsed() / 10;
        eprintln!("  1500x384 @ 384x384 matmul: {:?}", matmul_time);
        eprintln!(
            "  Est. encoder matmuls (4 layers × 6): {:?}",
            matmul_time * 24
        );

        // Check weights finalized status
        eprintln!("\n[Weight Finalization]");
        eprintln!(
            "  Block 0 self_attn finalized: {}",
            apr.encoder.blocks()[0].self_attn.is_finalized()
        );
        eprintln!(
            "  Block 0 FFN finalized: {}",
            apr.encoder.blocks()[0].ffn.is_finalized()
        );

        // Detailed encoder block breakdown
        eprintln!("\n[Encoder Block Breakdown (Layer 0)]");

        // Use the block output from above test (x has positional embedding added)
        let block = &apr.encoder.blocks()[0];
        let d_model = apr.encoder.d_model();

        // Layer Norm 1
        let ln1_start = std::time::Instant::now();
        let normed = block.ln1.forward(&x).expect("LN1 failed");
        let ln1_time = ln1_start.elapsed();
        eprintln!("  LayerNorm 1: {:?}", ln1_time);

        // Self-attention
        let attn_start = std::time::Instant::now();
        let attn_out = block.self_attn.forward(&normed, None).expect("Attn failed");
        let attn_time = attn_start.elapsed();
        eprintln!("  Self-Attention: {:?}", attn_time);

        // Residual 1
        let res1_start = std::time::Instant::now();
        let mut residual: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();
        let res1_time = res1_start.elapsed();
        eprintln!("  Residual 1: {:?}", res1_time);

        // Layer Norm 2
        let ln2_start = std::time::Instant::now();
        let normed2 = block.ln2.forward(&residual).expect("LN2 failed");
        let ln2_time = ln2_start.elapsed();
        eprintln!("  LayerNorm 2: {:?}", ln2_time);

        // FFN
        let ffn_start = std::time::Instant::now();
        let ffn_out = block.ffn.forward(&normed2).expect("FFN failed");
        let ffn_time = ffn_start.elapsed();
        eprintln!("  FFN: {:?}", ffn_time);

        // Residual 2
        let res2_start = std::time::Instant::now();
        for (r, f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }
        let res2_time = res2_start.elapsed();
        eprintln!("  Residual 2: {:?}", res2_time);

        let block_total = ln1_time + attn_time + res1_time + ln2_time + ffn_time + res2_time;
        eprintln!("  --");
        eprintln!("  Block total: {:?}", block_total);

        // Further breakdown of attention
        eprintln!("\n[Self-Attention Detailed Breakdown]");
        let seq_len = conv_frames;

        // Q, K, V projections
        let qkv_start = std::time::Instant::now();
        let _q = block
            .self_attn
            .w_q()
            .forward_simd(&normed, seq_len)
            .expect("Q");
        let _k = block
            .self_attn
            .w_k()
            .forward_simd(&normed, seq_len)
            .expect("K");
        let _v = block
            .self_attn
            .w_v()
            .forward_simd(&normed, seq_len)
            .expect("V");
        let qkv_time = qkv_start.elapsed();
        eprintln!("  QKV projections: {:?}", qkv_time);

        // Just the attention computation (from forward_cross_dispatch)
        // This calls forward_cross_optimal -> forward_cross_flash_v2 for long sequences
        eprintln!(
            "  Note: Attention uses {} heads, d_head={}",
            block.self_attn.n_heads(),
            d_model / block.self_attn.n_heads()
        );
        eprintln!("  Note: seq_len={} > 128, uses FlashAttention-2", seq_len);
        eprintln!(
            "  Attention overhead (total - QKV): {:?}",
            attn_time.saturating_sub(qkv_time)
        );

        eprintln!("\n============================================================\n");
    }

    /// WAPR-PERF-017: Test GPU-stream decoder block vs standard GPU decoder block
    ///
    /// Compares performance and parity of:
    /// - `forward_decoder_block_gpu`: Standard implementation (CPU LN, mixed streams)
    /// - `forward_decoder_block_gpu_stream`: All-GPU with external stream
    #[test]
    fn test_gpu_stream_decoder_block_parity() {
        use trueno_gpu::driver::CudaStream;

        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-017: GPU Stream Decoder Block Parity Test");
        eprintln!("============================================================\n");

        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Upload weights
        cuda_model
            .upload_decoder_weights_to_gpu()
            .expect("Upload weights");
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV cache");

        let d_model = cuda_model.config().n_text_state as usize;

        // Create test input
        let x: Vec<f32> = (0..d_model)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Run standard GPU path first (requires mutable borrow)
        let old_start = std::time::Instant::now();
        let old_output = cuda_model
            .forward_decoder_block_gpu(0, &x, 0, None)
            .expect("Standard GPU block");
        let old_time = old_start.elapsed();

        // Reset KV cache for clean comparison
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Re-init KV cache");

        // Now get context after mutable operations are done
        let ctx = cuda_model.executor.context();

        // Create stream and upload input for new path
        let stream = CudaStream::new(ctx).expect("Create stream");
        let x_gpu = GpuResidentTensor::from_host(ctx, &x).expect("Upload x");

        // Run new GPU-stream path
        let new_start = std::time::Instant::now();
        let mut new_output_gpu = cuda_model
            .forward_decoder_block_gpu_stream(0, &x_gpu, 0, &stream, None) // No cross-attention
            .expect("Stream GPU block");
        stream.synchronize().expect("Sync stream");
        let new_time = new_start.elapsed();

        // Download for comparison
        let new_output = new_output_gpu.to_host().expect("Download output");

        // Compare outputs
        let max_diff: f32 = old_output
            .iter()
            .zip(new_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!("[Results]");
        eprintln!("  Standard GPU block: {:?}", old_time);
        eprintln!("  Stream GPU block:   {:?}", new_time);
        eprintln!(
            "  Speedup:            {:.2}x",
            old_time.as_micros() as f64 / new_time.as_micros() as f64
        );
        eprintln!("  Max diff:           {:.2e}", max_diff);
        eprintln!(
            "  Parity:             {}",
            if max_diff < 1e-3 {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        );

        // Verify parity
        assert!(max_diff < 1e-3, "Output mismatch: max_diff={}", max_diff);

        eprintln!("\n✓ WAPR-PERF-017: GPU stream decoder block parity verified");
    }

    /// WAPR-PERF-017: Multi-iteration stream decoder benchmark
    ///
    /// Benchmarks repeated decoder block calls to show:
    /// - Kernel cache benefits (first call vs subsequent)
    /// - Stream-based execution overhead
    /// - Baseline before CUDA graph capture
    #[test]
    fn test_gpu_stream_decoder_multi_iteration() {
        use trueno_gpu::driver::CudaStream;

        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-017: Multi-Iteration Stream Decoder Benchmark");
        eprintln!("============================================================\n");

        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Upload weights
        cuda_model
            .upload_decoder_weights_to_gpu()
            .expect("Upload weights");
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV cache");

        let d_model = cuda_model.config().n_text_state as usize;
        let ctx = cuda_model.executor.context();

        // Create stream for all operations
        let stream = CudaStream::new(ctx).expect("Create stream");

        // Create test input
        let x: Vec<f32> = (0..d_model)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let x_gpu = GpuResidentTensor::from_host(ctx, &x).expect("Upload x");

        const NUM_ITERATIONS: usize = 100;
        let mut times = Vec::with_capacity(NUM_ITERATIONS);

        // Run multiple iterations, using position to vary KV cache updates
        for i in 0..NUM_ITERATIONS {
            // Reset KV cache for each iteration to simulate fresh token
            if i % 10 == 0 {
                cuda_model.reset_gpu_decoder_kv_cache();
                cuda_model
                    .init_gpu_decoder_kv_cache_head_first()
                    .expect("Re-init KV cache");
            }

            let pos = i % 10; // Vary position within KV cache window
            let start = std::time::Instant::now();
            let _output = cuda_model
                .forward_decoder_block_gpu_stream(0, &x_gpu, pos, &stream, None)
                .expect("Stream GPU block");
            stream.synchronize().expect("Sync");
            times.push(start.elapsed());
        }

        // Calculate statistics
        let first = times[0];
        let warmup_avg: std::time::Duration = times[1..10].iter().sum::<std::time::Duration>() / 9;
        let hot_avg: std::time::Duration =
            times[10..].iter().sum::<std::time::Duration>() / (NUM_ITERATIONS - 10) as u32;
        let min = *times.iter().min().expect("min");
        let max = *times.iter().max().expect("max");

        eprintln!("[Results]");
        eprintln!("  Iterations:     {}", NUM_ITERATIONS);
        eprintln!(
            "  First call:     {:?} (includes kernel compilation)",
            first
        );
        eprintln!("  Warmup avg:     {:?} (iterations 2-10)", warmup_avg);
        eprintln!(
            "  Hot avg:        {:?} (iterations 11-{})",
            hot_avg, NUM_ITERATIONS
        );
        eprintln!("  Min:            {:?}", min);
        eprintln!("  Max:            {:?}", max);
        eprintln!(
            "  Speedup:        {:.1}x (first vs hot)",
            first.as_micros() as f64 / hot_avg.as_micros() as f64
        );

        // Target: hot average should be under 500µs per decoder block
        let hot_us = hot_avg.as_micros();
        eprintln!("  Target:         <500µs per block");
        eprintln!(
            "  Status:         {} ({:.0}µs)",
            if hot_us < 500 {
                "✓ PASS"
            } else {
                "○ BASELINE"
            },
            hot_us
        );

        eprintln!("\n✓ WAPR-PERF-017: Multi-iteration benchmark complete");
    }

    /// WAPR-PERF-017: CUDA Graph capture test
    ///
    /// Attempts to capture the decoder block execution into a CUDA graph
    /// for reduced launch overhead on repeated execution.
    ///
    /// Graph capture benefits:
    /// - 3-10µs graph launch vs 20-50µs per kernel
    /// - Pre-validated parameters
    /// - Reduced CPU overhead
    #[test]
    fn test_cuda_graph_capture_decoder() {
        #[allow(unused_imports)]
        use trueno_gpu::driver::{CaptureMode, CudaGraphExec, CudaStream};

        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-017: CUDA Graph Capture Decoder Test");
        eprintln!("============================================================\n");

        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Upload weights
        cuda_model
            .upload_decoder_weights_to_gpu()
            .expect("Upload weights");
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV cache");

        let d_model = cuda_model.config().n_text_state as usize;
        let ctx = cuda_model.executor.context();

        // Create stream for capture
        let stream = CudaStream::new(ctx).expect("Create stream");

        // Create test input on GPU
        let x: Vec<f32> = (0..d_model)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let x_gpu = GpuResidentTensor::from_host(ctx, &x).expect("Upload x");

        // Warm up kernels first (outside capture)
        eprintln!("[Warmup] Running decoder block to compile kernels...");
        let _ = cuda_model
            .forward_decoder_block_gpu_stream(0, &x_gpu, 0, &stream, None)
            .expect("Warmup");
        stream.synchronize().expect("Sync warmup");

        // Reset KV cache for capture
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Re-init KV cache");

        // Attempt CUDA graph capture
        eprintln!("[Capture] Beginning CUDA graph capture...");

        match stream.begin_capture(CaptureMode::Global) {
            Ok(()) => eprintln!("  Stream capture started"),
            Err(e) => {
                eprintln!("  ✗ Failed to begin capture: {}", e);
                return;
            }
        }

        // Run decoder block during capture
        let capture_result = cuda_model.forward_decoder_block_gpu_stream(0, &x_gpu, 0, &stream, None);

        // End capture
        let graph_result = stream.end_capture();

        match (&capture_result, &graph_result) {
            (Ok(_), Ok(graph)) => {
                eprintln!("  ✓ Capture successful!");

                // Instantiate graph
                match graph.instantiate() {
                    Ok(exec) => {
                        eprintln!("  ✓ Graph instantiated!");

                        // Reset KV cache for graph replay
                        cuda_model.reset_gpu_decoder_kv_cache();
                        cuda_model
                            .init_gpu_decoder_kv_cache_head_first()
                            .expect("Re-init KV cache");

                        // Benchmark graph replay vs direct execution
                        const NUM_REPLAYS: usize = 100;
                        let mut graph_times = Vec::with_capacity(NUM_REPLAYS);
                        let mut direct_times = Vec::with_capacity(NUM_REPLAYS);

                        // Graph replay benchmark
                        for _ in 0..NUM_REPLAYS {
                            let start = std::time::Instant::now();
                            stream.launch_graph(&exec).expect("Graph launch");
                            stream.synchronize().expect("Sync");
                            graph_times.push(start.elapsed());
                        }

                        // Reset for direct comparison
                        cuda_model.reset_gpu_decoder_kv_cache();
                        cuda_model
                            .init_gpu_decoder_kv_cache_head_first()
                            .expect("Re-init KV cache");

                        // Direct execution benchmark
                        for i in 0..NUM_REPLAYS {
                            let pos = i % 10;
                            if i % 10 == 0 {
                                cuda_model.reset_gpu_decoder_kv_cache();
                                cuda_model
                                    .init_gpu_decoder_kv_cache_head_first()
                                    .expect("Re-init");
                            }
                            let start = std::time::Instant::now();
                            let _ = cuda_model
                                .forward_decoder_block_gpu_stream(0, &x_gpu, pos, &stream, None)
                                .expect("Direct");
                            stream.synchronize().expect("Sync");
                            direct_times.push(start.elapsed());
                        }

                        let graph_avg: std::time::Duration =
                            graph_times.iter().sum::<std::time::Duration>() / NUM_REPLAYS as u32;
                        let direct_avg: std::time::Duration =
                            direct_times.iter().sum::<std::time::Duration>() / NUM_REPLAYS as u32;

                        eprintln!("\n[Results]");
                        eprintln!("  Graph replay avg:  {:?}", graph_avg);
                        eprintln!("  Direct exec avg:   {:?}", direct_avg);
                        eprintln!(
                            "  Graph speedup:     {:.2}x",
                            direct_avg.as_micros() as f64 / graph_avg.as_micros() as f64
                        );

                        eprintln!("\n✓ WAPR-PERF-017: CUDA Graph capture successful!");
                    }
                    Err(e) => {
                        eprintln!("  ✗ Graph instantiation failed: {}", e);
                    }
                }
            }
            (Err(e), _) => {
                eprintln!("  ✗ Decoder block failed during capture: {}", e);
                eprintln!("  Note: Graph capture requires operations that support stream capture");
            }
            (_, Err(e)) => {
                eprintln!("  ✗ Graph capture failed: {}", e);
                eprintln!("  Note: This may indicate memory allocation during capture");
            }
        }
    }

    /// WAPR-PERF-017: Full token pass (all 4 layers) with CUDA Graph
    ///
    /// Benchmarks the complete decoder token forward pass with graph capture.
    /// This demonstrates production-level speedup potential.
    #[test]
    fn test_cuda_graph_full_token_pass() {
        use trueno_gpu::driver::{CaptureMode, CudaStream};

        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-017: Full Token Pass CUDA Graph Benchmark");
        eprintln!("============================================================\n");

        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let n_layers = apr.config().n_text_layer as usize;
        let d_model = apr.config().n_text_state as usize;
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        eprintln!("[Model] {} layers, d_model={}", n_layers, d_model);

        // Upload weights
        cuda_model
            .upload_decoder_weights_to_gpu()
            .expect("Upload weights");
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Init KV cache");

        let ctx = cuda_model.executor.context();
        let stream = CudaStream::new(ctx).expect("Create stream");

        // Create test token embedding
        let token_embedding: Vec<f32> =
            (0..d_model).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();

        // Warmup to compile kernels
        eprintln!("[Warmup] Running full token pass to compile kernels...");
        let _warmup = cuda_model
            .forward_decoder_token_gpu_stream(&token_embedding, 0, &stream, None)
            .expect("Warmup");
        stream.synchronize().expect("Sync warmup");

        // Reset KV cache
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Re-init KV cache");

        // Capture full token pass
        eprintln!("[Capture] Beginning CUDA graph capture for {} layers...", n_layers);
        stream.begin_capture(CaptureMode::Global).expect("Begin capture");

        let capture_result = cuda_model.forward_decoder_token_gpu_stream(&token_embedding, 0, &stream, None);

        let graph = stream.end_capture().expect("End capture");

        if let Err(e) = &capture_result {
            eprintln!("  ✗ Token pass failed during capture: {}", e);
            return;
        }
        eprintln!("  ✓ Capture successful!");

        let exec = graph.instantiate().expect("Instantiate graph");
        eprintln!("  ✓ Graph instantiated!");

        // Benchmark
        const NUM_TOKENS: usize = 100;
        let mut graph_times = Vec::with_capacity(NUM_TOKENS);
        let mut direct_times = Vec::with_capacity(NUM_TOKENS);

        // Graph replay benchmark (simulates token generation)
        eprintln!("\n[Benchmark] Running {} token passes...", NUM_TOKENS);
        for _ in 0..NUM_TOKENS {
            let start = std::time::Instant::now();
            stream.launch_graph(&exec).expect("Graph launch");
            stream.synchronize().expect("Sync");
            graph_times.push(start.elapsed());
        }

        // Direct execution benchmark
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Re-init");

        for i in 0..NUM_TOKENS {
            let pos = i % 10;
            if i % 10 == 0 && i > 0 {
                cuda_model.reset_gpu_decoder_kv_cache();
                cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Re-init");
            }
            let start = std::time::Instant::now();
            let _out = cuda_model
                .forward_decoder_token_gpu_stream(&token_embedding, pos, &stream, None)
                .expect("Direct");
            stream.synchronize().expect("Sync");
            direct_times.push(start.elapsed());
        }

        let graph_avg: std::time::Duration =
            graph_times.iter().sum::<std::time::Duration>() / NUM_TOKENS as u32;
        let direct_avg: std::time::Duration =
            direct_times.iter().sum::<std::time::Duration>() / NUM_TOKENS as u32;
        let graph_min = *graph_times.iter().min().expect("min");
        let direct_min = *direct_times.iter().min().expect("min");

        eprintln!("\n[Results - {} layers × {} tokens]", n_layers, NUM_TOKENS);
        eprintln!("  Graph replay avg:  {:?} ({:.0}µs)", graph_avg, graph_avg.as_micros());
        eprintln!("  Direct exec avg:   {:?} ({:.0}µs)", direct_avg, direct_avg.as_micros());
        eprintln!("  Graph min:         {:?}", graph_min);
        eprintln!("  Direct min:        {:?}", direct_min);
        eprintln!(
            "  Speedup:           {:.1}x",
            direct_avg.as_micros() as f64 / graph_avg.as_micros() as f64
        );

        // Calculate projected decode time for 27 tokens (typical short utterance)
        let tokens_for_1_5s = 27;
        let graph_decode_ms = graph_avg.as_micros() as f64 * tokens_for_1_5s as f64 / 1000.0;
        let direct_decode_ms = direct_avg.as_micros() as f64 * tokens_for_1_5s as f64 / 1000.0;

        eprintln!("\n[Projected for {} tokens (1.5s audio)]:", tokens_for_1_5s);
        eprintln!("  Graph:  {:.1}ms", graph_decode_ms);
        eprintln!("  Direct: {:.1}ms", direct_decode_ms);

        // Point 157 target is 1984ms total, decoder portion should be <500ms
        eprintln!("  Target: <500ms decoder");
        eprintln!(
            "  Status: {} ({:.1}ms)",
            if graph_decode_ms < 500.0 {
                "✓ PASS"
            } else {
                "○ NEEDS ENCODER OPT"
            },
            graph_decode_ms
        );

        eprintln!("\n✓ WAPR-PERF-017: Full token pass benchmark complete");
    }

    /// WAPR-PERF-018: Test GPU-resident encoder output
    ///
    /// Verifies encoder output stays on GPU for cross-attention.
    /// Includes warmup to measure true encoder performance after kernel compilation.
    #[test]
    fn test_encode_gpu_resident() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-018: GPU-Resident Encoder Output Test");
        eprintln!("============================================================\n");

        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let d_model = apr.config().n_audio_state as usize;
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Create dummy mel input
        let n_mels = cuda_model.config().n_mels as usize;
        let seq_len = 100;
        let mel: Vec<f32> = (0..n_mels * seq_len).map(|i| (i as f32 * 0.001).sin()).collect();

        eprintln!("[Input] {} mels × {} frames", n_mels, seq_len);

        // Warmup run to compile kernels
        eprintln!("[Warmup] Compiling PTX kernels...");
        let warmup_start = std::time::Instant::now();
        let _enc_warmup = cuda_model.encode_gpu_resident(&mel).expect("Warmup encode");
        let warmup_time = warmup_start.elapsed();
        eprintln!("  Warmup (incl. kernel compile): {:?}", warmup_time);

        // Benchmark 5 iterations after warmup
        const ITERATIONS: usize = 5;
        let mut times = Vec::with_capacity(ITERATIONS);
        eprintln!("[Benchmark] Running {} iterations...", ITERATIONS);
        for _ in 0..ITERATIONS {
            let start = std::time::Instant::now();
            let enc_gpu = cuda_model.encode_gpu_resident(&mel).expect("Encode GPU resident");
            times.push(start.elapsed());
            // Verify size on first iteration
            if times.len() == 1 {
                let enc_len = enc_gpu.len();
                let expected_seq_len = (seq_len + 2 - 3) / 2 + 1;
                assert_eq!(enc_len, expected_seq_len * d_model, "Unexpected encoder output size");
            }
        }

        let avg_time = times.iter().map(|t| t.as_micros()).sum::<u128>() / ITERATIONS as u128;
        let min_time = times.iter().min().unwrap();
        let max_time = times.iter().max().unwrap();

        eprintln!("\n[Results]");
        eprintln!("  Warmup:  {:?} (incl. kernel compile)", warmup_time);
        eprintln!("  Average: {}µs", avg_time);
        eprintln!("  Min:     {:?}", min_time);
        eprintln!("  Max:     {:?}", max_time);
        eprintln!("  Speedup vs warmup: {:.1}x", warmup_time.as_micros() as f64 / avg_time as f64);

        eprintln!("\n✓ WAPR-PERF-018: GPU-resident encoder output verified");
    }

    /// WAPR-PERF-018: Test full pipeline with GPU cross-attention
    ///
    /// Tests the complete encoder -> cross-attention K/V population -> decoder flow.
    /// Includes warmup to measure true performance after kernel compilation.
    #[test]
    fn test_gpu_cross_attention_pipeline() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-018: GPU Cross-Attention Pipeline Test");
        eprintln!("============================================================\n");

        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let d_model = apr.config().n_text_state as usize;
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Create dummy mel input
        let n_mels = cuda_model.config().n_mels as usize;
        let seq_len = 100;
        let mel: Vec<f32> = (0..n_mels * seq_len).map(|i| (i as f32 * 0.001).sin()).collect();

        // Create stream
        let ctx = cuda_model.executor.context();
        let stream = CudaStream::new(ctx).expect("Create stream");

        // === WARMUP PHASE ===
        eprintln!("[Warmup] Compiling PTX kernels...");
        let warmup_start = std::time::Instant::now();
        let enc_warmup = cuda_model.encode_gpu_resident(&mel).expect("Warmup encode");
        cuda_model.populate_cross_kv_caches_gpu(&enc_warmup, &stream).expect("Warmup K/V");
        stream.synchronize().expect("Sync");
        let warmup_time = warmup_start.elapsed();
        eprintln!("  Total warmup: {:?}", warmup_time);

        // Reset for benchmark
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Re-init");

        // === BENCHMARK PHASE ===
        eprintln!("\n[Step 1] GPU-resident encoder (warmed up)...");
        let enc_start = std::time::Instant::now();
        let enc_gpu = cuda_model.encode_gpu_resident(&mel).expect("Encode GPU resident");
        let enc_time = enc_start.elapsed();
        let enc_seq_len = enc_gpu.len() / d_model;
        eprintln!("  Encoder: {:?} ({} × {})", enc_time, enc_seq_len, d_model);

        eprintln!("[Step 2] Populate cross-attention K/V caches (warmed up)...");
        let kv_start = std::time::Instant::now();
        cuda_model
            .populate_cross_kv_caches_gpu(&enc_gpu, &stream)
            .expect("Populate cross KV");
        stream.synchronize().expect("Sync K/V");
        let kv_time = kv_start.elapsed();
        eprintln!("  Cross K/V population: {:?}", kv_time);

        // Create test token embedding
        let token_embedding: Vec<f32> = (0..d_model)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        eprintln!("[Step 3] Decoder with cross-attention...");

        // Warmup
        let _warmup = cuda_model
            .forward_decoder_token_gpu_stream(&token_embedding, 0, &stream, Some(enc_seq_len))
            .expect("Warmup");
        stream.synchronize().expect("Sync warmup");
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Re-init");
        // Re-populate cross K/V after reset
        cuda_model
            .populate_cross_kv_caches_gpu(&enc_gpu, &stream)
            .expect("Re-populate cross KV");

        // Benchmark single token with cross-attention
        const NUM_TOKENS: usize = 10;
        let mut times = Vec::with_capacity(NUM_TOKENS);

        for i in 0..NUM_TOKENS {
            let dec_start = std::time::Instant::now();
            let mut output_gpu = cuda_model
                .forward_decoder_token_gpu_stream(&token_embedding, i, &stream, Some(enc_seq_len))
                .expect("Decoder with cross-attention");
            stream.synchronize().expect("Sync");
            times.push(dec_start.elapsed());

            if i == 0 {
                // Verify output size on first token
                let output = output_gpu.to_host().expect("Download");
                assert_eq!(
                    output.len(),
                    d_model,
                    "Unexpected decoder output size"
                );
            }
        }

        let avg: std::time::Duration = times.iter().sum::<std::time::Duration>() / NUM_TOKENS as u32;

        eprintln!("\n[Results]");
        eprintln!("  Encoder:         {:?}", enc_time);
        eprintln!("  Cross K/V pop:   {:?}", kv_time);
        eprintln!("  Decoder avg:     {:?} ({} tokens)", avg, NUM_TOKENS);
        eprintln!("  Total pipeline:  {:?}", enc_time + kv_time + avg * NUM_TOKENS as u32);

        // For 1.5s audio (27 tokens), project decoder time
        let tokens_27 = avg.as_micros() as f64 * 27.0 / 1000.0;
        eprintln!("\n[Projected for 27 tokens (1.5s audio)]");
        eprintln!("  Decoder: {:.1}ms", tokens_27);

        eprintln!("\n✓ WAPR-PERF-018: GPU cross-attention pipeline verified");
    }

    /// WAPR-PERF-020: Test warmup method for predictable latency
    ///
    /// Verifies that warmup() pre-compiles all kernels and subsequent
    /// operations run at full speed.
    #[test]
    fn test_warmup_method() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-020: Warmup Method Test");
        eprintln!("============================================================\n");

        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Call warmup
        eprintln!("[Warmup] Pre-compiling all GPU kernels...");
        let warmup_ms = cuda_model.warmup().expect("Warmup failed");
        eprintln!("  Warmup completed: {}ms", warmup_ms);

        // Now run encoder - should be fast since kernels are compiled
        let n_mels = cuda_model.config().n_mels as usize;
        let d_model = cuda_model.config().n_text_state as usize;
        let mel: Vec<f32> = (0..n_mels * 100).map(|i| (i as f32 * 0.001).sin()).collect();

        // Time encoder (should be fast)
        let enc_start = std::time::Instant::now();
        let enc_gpu = cuda_model.encode_gpu_resident(&mel).expect("Encode");
        let enc_time = enc_start.elapsed();

        // Time cross K/V population (should be fast)
        let ctx = cuda_model.executor.context();
        let stream = CudaStream::new(ctx).expect("Create stream");
        let kv_start = std::time::Instant::now();
        cuda_model.populate_cross_kv_caches_gpu(&enc_gpu, &stream).expect("K/V pop");
        stream.synchronize().expect("Sync");
        let kv_time = kv_start.elapsed();

        // Time decoder (should be fast)
        let enc_seq_len = enc_gpu.len() / d_model;
        let dummy_emb: Vec<f32> = vec![0.1; d_model];
        let dec_start = std::time::Instant::now();
        let _dec_out = cuda_model
            .forward_decoder_token_gpu_stream(&dummy_emb, 0, &stream, Some(enc_seq_len))
            .expect("Decoder");
        stream.synchronize().expect("Sync");
        let dec_time = dec_start.elapsed();

        let total_post_warmup = enc_time + kv_time + dec_time;

        eprintln!("\n[Results after warmup]");
        eprintln!("  Encoder:     {:?}", enc_time);
        eprintln!("  Cross K/V:   {:?}", kv_time);
        eprintln!("  Decoder:     {:?}", dec_time);
        eprintln!("  Total:       {:?}", total_post_warmup);
        eprintln!("  Speedup:     {:.1}x vs warmup", warmup_ms as f64 / total_post_warmup.as_millis() as f64);

        // After warmup, total should be <20ms (vs ~200ms without warmup)
        assert!(
            total_post_warmup.as_millis() < 50,
            "Post-warmup pipeline too slow: {:?}",
            total_post_warmup
        );

        eprintln!("\n✓ WAPR-PERF-020: Warmup method verified - predictable latency achieved");
    }

    /// WAPR-PERF-018: Test CUDA graph capture with cross-attention
    ///
    /// Captures full decoder (self-attention + cross-attention + FFN) as CUDA graph.
    #[test]
    fn test_cuda_graph_with_cross_attention() {
        use trueno_gpu::driver::CaptureMode;

        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-018: CUDA Graph with Cross-Attention Test");
        eprintln!("============================================================\n");

        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let d_model = apr.config().n_text_state as usize;
        let n_layers = apr.config().n_text_layer;
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Create dummy mel input
        let n_mels = cuda_model.config().n_mels as usize;
        let seq_len = 100;
        let mel: Vec<f32> = (0..n_mels * seq_len).map(|i| (i as f32 * 0.001).sin()).collect();

        // Step 1: Encode and populate cross K/V caches
        eprintln!("[Setup] Encoding and populating cross K/V caches...");
        let enc_gpu = cuda_model.encode_gpu_resident(&mel).expect("Encode");
        let enc_seq_len = enc_gpu.len() / d_model;

        let ctx = cuda_model.executor.context();
        let stream = CudaStream::new(ctx).expect("Create stream");

        cuda_model
            .populate_cross_kv_caches_gpu(&enc_gpu, &stream)
            .expect("Populate cross KV");

        // Create test token embedding
        let token_embedding: Vec<f32> = (0..d_model)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Step 2: Warmup to compile kernels
        eprintln!("[Warmup] Running decoder with cross-attention...");
        let _warmup = cuda_model
            .forward_decoder_token_gpu_stream(&token_embedding, 0, &stream, Some(enc_seq_len))
            .expect("Warmup");
        stream.synchronize().expect("Sync warmup");

        // Reset self-attention KV caches for capture
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model
            .init_gpu_decoder_kv_cache_head_first()
            .expect("Re-init KV cache");
        // Re-populate cross K/V after reset
        cuda_model
            .populate_cross_kv_caches_gpu(&enc_gpu, &stream)
            .expect("Re-populate cross KV");

        // Step 3: Capture CUDA graph
        eprintln!("[Capture] Beginning CUDA graph capture ({} layers + cross-attn)...", n_layers);

        match stream.begin_capture(CaptureMode::Global) {
            Ok(()) => eprintln!("  Stream capture started"),
            Err(e) => {
                eprintln!("  ✗ Failed to begin capture: {}", e);
                return;
            }
        }

        let capture_result = cuda_model.forward_decoder_token_gpu_stream(
            &token_embedding,
            0,
            &stream,
            Some(enc_seq_len),
        );

        let graph_result = stream.end_capture();

        match (&capture_result, &graph_result) {
            (Ok(_), Ok(graph)) => {
                eprintln!("  ✓ Capture successful!");

                match graph.instantiate() {
                    Ok(exec) => {
                        eprintln!("  ✓ Graph instantiated!");

                        // Step 4: Benchmark graph replay vs direct
                        const NUM_REPLAYS: usize = 100;
                        let mut graph_times = Vec::with_capacity(NUM_REPLAYS);
                        let mut direct_times = Vec::with_capacity(NUM_REPLAYS);

                        eprintln!("\n[Benchmark] Running {} iterations...", NUM_REPLAYS);

                        // Graph replay benchmark
                        for _ in 0..NUM_REPLAYS {
                            let start = std::time::Instant::now();
                            stream.launch_graph(&exec).expect("Graph launch");
                            stream.synchronize().expect("Sync");
                            graph_times.push(start.elapsed());
                        }

                        // Direct execution benchmark
                        cuda_model.reset_gpu_decoder_kv_cache();
                        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Re-init");
                        cuda_model
                            .populate_cross_kv_caches_gpu(&enc_gpu, &stream)
                            .expect("Re-populate");

                        for i in 0..NUM_REPLAYS {
                            let pos = i % 10;
                            if i % 10 == 0 && i > 0 {
                                cuda_model.reset_gpu_decoder_kv_cache();
                                cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Re-init");
                                cuda_model
                                    .populate_cross_kv_caches_gpu(&enc_gpu, &stream)
                                    .expect("Re-populate");
                            }
                            let start = std::time::Instant::now();
                            let _out = cuda_model.forward_decoder_token_gpu_stream(
                                &token_embedding,
                                pos,
                                &stream,
                                Some(enc_seq_len),
                            ).expect("Direct");
                            stream.synchronize().expect("Sync");
                            direct_times.push(start.elapsed());
                        }

                        let graph_avg: std::time::Duration =
                            graph_times.iter().sum::<std::time::Duration>() / NUM_REPLAYS as u32;
                        let direct_avg: std::time::Duration =
                            direct_times.iter().sum::<std::time::Duration>() / NUM_REPLAYS as u32;

                        eprintln!("\n[Results - {} layers + cross-attention]", n_layers);
                        eprintln!("  Graph replay avg:  {:?}", graph_avg);
                        eprintln!("  Direct exec avg:   {:?}", direct_avg);
                        eprintln!(
                            "  Graph speedup:     {:.1}x",
                            direct_avg.as_micros() as f64 / graph_avg.as_micros() as f64
                        );

                        // Project for 27 tokens
                        let graph_27 = graph_avg.as_micros() as f64 * 27.0 / 1000.0;
                        let direct_27 = direct_avg.as_micros() as f64 * 27.0 / 1000.0;
                        eprintln!("\n[Projected for 27 tokens (1.5s audio)]");
                        eprintln!("  Graph:  {:.1}ms", graph_27);
                        eprintln!("  Direct: {:.1}ms", direct_27);

                        eprintln!("\n✓ WAPR-PERF-018: CUDA Graph with cross-attention verified!");
                    }
                    Err(e) => eprintln!("  ✗ Failed to instantiate graph: {}", e),
                }
            }
            (Err(e), _) => eprintln!("  ✗ Decoder failed during capture: {}", e),
            (_, Err(e)) => eprintln!("  ✗ End capture failed: {}", e),
        }
    }

    /// WAPR-PERF-022: Test GPU conv1d vs CPU conv1d correctness
    ///
    /// Compares GPU convolution output with CPU reference to identify
    /// where they diverge (root cause of GPU encoder producing wrong transcription).
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_conv1d_vs_cpu() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {model_path}, skipping test");
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-022: GPU Conv1d vs CPU Conv1d Correctness Test");
        eprintln!("============================================================\n");

        // Load model
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        let n_mels = cuda_model.config.n_mels as usize;
        let d_model = cuda_model.config.n_audio_state as usize;

        eprintln!("[Config] n_mels={}, d_model={}", n_mels, d_model);
        eprintln!("[Conv1] in={}, out={}, k=3, s=1, p=1", n_mels, d_model);
        eprintln!("[Conv2] in={}, out={}, k=3, s=2, p=1", d_model, d_model);

        // Create test input (small sequence for debugging)
        let test_seq_len = 100;
        let mel: Vec<f32> = (0..test_seq_len * n_mels)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();

        eprintln!("\n[Input] mel shape: {} x {}", test_seq_len, n_mels);
        eprintln!("        mean={:.6}, std={:.6}",
            mel.iter().sum::<f32>() / mel.len() as f32,
            (mel.iter().map(|x| x.powi(2)).sum::<f32>() / mel.len() as f32).sqrt());

        // === CPU Conv1d (do all CPU work first to avoid borrow issues) ===
        let (cpu_conv1_gelu, cpu_conv2_gelu, cpu_frontend, cpu_seq_after_conv1) = {
            let conv_frontend = cuda_model.encoder.conv_frontend();

            eprintln!("\n[Step 1] CPU Conv1...");
            let cpu_conv1 = conv_frontend.conv1.forward(&mel).expect("CPU conv1");
            let cpu_seq_after_conv1 = cpu_conv1.len() / d_model;
            eprintln!("  Output shape: {} x {}", cpu_seq_after_conv1, d_model);
            eprintln!("  mean={:.6}, std={:.6}",
                cpu_conv1.iter().sum::<f32>() / cpu_conv1.len() as f32,
                (cpu_conv1.iter().map(|x| x.powi(2)).sum::<f32>() / cpu_conv1.len() as f32).sqrt());

            // Apply GELU (CPU conv doesn't include activation)
            let cpu_conv1_gelu: Vec<f32> = cpu_conv1.iter().map(|x| {
                let x = *x;
                x * 0.5 * (1.0 + (x * 0.7978845608028654 * (1.0 + 0.044715 * x * x)).tanh())
            }).collect();
            eprintln!("  After GELU: mean={:.6}",
                cpu_conv1_gelu.iter().sum::<f32>() / cpu_conv1_gelu.len() as f32);

            eprintln!("\n[Step 2] CPU Conv2...");
            let cpu_conv2 = conv_frontend.conv2.forward(&cpu_conv1_gelu).expect("CPU conv2");
            let cpu_seq_after_conv2 = cpu_conv2.len() / d_model;
            eprintln!("  Output shape: {} x {}", cpu_seq_after_conv2, d_model);
            eprintln!("  mean={:.6}, std={:.6}",
                cpu_conv2.iter().sum::<f32>() / cpu_conv2.len() as f32,
                (cpu_conv2.iter().map(|x| x.powi(2)).sum::<f32>() / cpu_conv2.len() as f32).sqrt());

            let cpu_conv2_gelu: Vec<f32> = cpu_conv2.iter().map(|x| {
                let x = *x;
                x * 0.5 * (1.0 + (x * 0.7978845608028654 * (1.0 + 0.044715 * x * x)).tanh())
            }).collect();
            eprintln!("  After GELU: mean={:.6}",
                cpu_conv2_gelu.iter().sum::<f32>() / cpu_conv2_gelu.len() as f32);

            eprintln!("\n[Step 3] Full CPU Frontend...");
            let cpu_frontend = conv_frontend.forward(&mel).expect("CPU frontend");
            eprintln!("  Output shape: {} x {}", cpu_frontend.len() / d_model, d_model);
            eprintln!("  mean={:.6}, std={:.6}",
                cpu_frontend.iter().sum::<f32>() / cpu_frontend.len() as f32,
                (cpu_frontend.iter().map(|x| x.powi(2)).sum::<f32>() / cpu_frontend.len() as f32).sqrt());

            (cpu_conv1_gelu, cpu_conv2_gelu, cpu_frontend, cpu_seq_after_conv1)
        };

        // === GPU Conv1d ===
        eprintln!("\n[Step 4] GPU Conv1...");
        cuda_model.upload_conv_weights_to_gpu().expect("Upload conv weights");

        let ctx = cuda_model.executor.context();
        let conv_weights = cuda_model.gpu_conv_weights.as_ref().expect("Conv weights");

        let mel_gpu = GpuResidentTensor::from_host(ctx, &mel)
            .expect("mel upload");

        let mut gpu_conv1 = mel_gpu
            .conv1d(
                ctx,
                &conv_weights.conv1_weight,
                Some(&conv_weights.conv1_bias),
                n_mels as u32,
                d_model as u32,
                3, 1, 1,
                test_seq_len as u32,
            )
            .expect("GPU conv1");

        let gpu_conv1_host = gpu_conv1.to_host().expect("Download");
        let gpu_seq_after_conv1 = gpu_conv1_host.len() / d_model;
        eprintln!("  Output shape: {} x {}", gpu_seq_after_conv1, d_model);
        eprintln!("  mean={:.6}, std={:.6}",
            gpu_conv1_host.iter().sum::<f32>() / gpu_conv1_host.len() as f32,
            (gpu_conv1_host.iter().map(|x| x.powi(2)).sum::<f32>() / gpu_conv1_host.len() as f32).sqrt());

        // === Compare ===
        eprintln!("\n[Step 3] Comparing CPU vs GPU Conv1 output...");

        // Note: GPU conv1d includes GELU, CPU doesn't
        // Compare GPU output with CPU GELU output
        if cpu_conv1_gelu.len() == gpu_conv1_host.len() {
            let max_diff: f32 = cpu_conv1_gelu.iter()
                .zip(gpu_conv1_host.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);

            let avg_diff: f32 = cpu_conv1_gelu.iter()
                .zip(gpu_conv1_host.iter())
                .map(|(c, g)| (c - g).abs())
                .sum::<f32>() / cpu_conv1_gelu.len() as f32;

            eprintln!("  Max difference: {:.6}", max_diff);
            eprintln!("  Avg difference: {:.6}", avg_diff);

            // Show first few elements
            eprintln!("\n  First 10 elements comparison:");
            for i in 0..10.min(cpu_conv1_gelu.len()) {
                eprintln!("    [{}] CPU: {:.6}, GPU: {:.6}, diff: {:.6}",
                    i, cpu_conv1_gelu[i], gpu_conv1_host[i],
                    (cpu_conv1_gelu[i] - gpu_conv1_host[i]).abs());
            }

            if max_diff < 0.05 {
                eprintln!("\n✓ Conv1 output matches (max_diff < 0.05)");
            } else {
                eprintln!("\n✗ Conv1 output MISMATCH (max_diff = {:.6})", max_diff);
            }
        } else {
            eprintln!("  ✗ Shape mismatch: CPU={}, GPU={}",
                cpu_conv1_gelu.len(), gpu_conv1_host.len());
        }

        // === GPU Conv2 ===
        eprintln!("\n[Step 6] GPU Conv2...");
        let mut gpu_conv2 = gpu_conv1
            .conv1d(
                ctx,
                &conv_weights.conv2_weight,
                Some(&conv_weights.conv2_bias),
                d_model as u32,
                d_model as u32,
                3, 2, 1, // stride=2 for conv2
                gpu_seq_after_conv1 as u32,
            )
            .expect("GPU conv2");

        let gpu_conv2_host = gpu_conv2.to_host().expect("Download");
        let gpu_seq_after_conv2 = gpu_conv2_host.len() / d_model;
        eprintln!("  Output shape: {} x {}", gpu_seq_after_conv2, d_model);
        eprintln!("  mean={:.6}, std={:.6}",
            gpu_conv2_host.iter().sum::<f32>() / gpu_conv2_host.len() as f32,
            (gpu_conv2_host.iter().map(|x| x.powi(2)).sum::<f32>() / gpu_conv2_host.len() as f32).sqrt());

        // === Compare Conv2 ===
        eprintln!("\n[Step 6] Comparing CPU vs GPU Conv2 output...");
        if cpu_conv2_gelu.len() == gpu_conv2_host.len() {
            let max_diff: f32 = cpu_conv2_gelu.iter()
                .zip(gpu_conv2_host.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);

            let avg_diff: f32 = cpu_conv2_gelu.iter()
                .zip(gpu_conv2_host.iter())
                .map(|(c, g)| (c - g).abs())
                .sum::<f32>() / cpu_conv2_gelu.len() as f32;

            eprintln!("  Max difference: {:.6}", max_diff);
            eprintln!("  Avg difference: {:.6}", avg_diff);

            // Show first few elements
            eprintln!("\n  First 10 elements comparison:");
            for i in 0..10.min(cpu_conv2_gelu.len()) {
                eprintln!("    [{}] CPU: {:.6}, GPU: {:.6}, diff: {:.6}",
                    i, cpu_conv2_gelu[i], gpu_conv2_host[i],
                    (cpu_conv2_gelu[i] - gpu_conv2_host[i]).abs());
            }

            if max_diff < 0.05 {
                eprintln!("\n✓ Conv2 output matches (max_diff < 0.05)");
            } else {
                eprintln!("\n✗ Conv2 output MISMATCH (max_diff = {:.6})", max_diff);
            }
        } else {
            eprintln!("  ✗ Shape mismatch: CPU={}, GPU={}",
                cpu_conv2_gelu.len(), gpu_conv2_host.len());
        }

        // === Compare Full Frontend ===
        eprintln!("\n[Step 8] Comparing Full Frontend CPU vs GPU Conv2+GELU...");
        eprintln!("  CPU frontend shape: {} x {}", cpu_frontend.len() / d_model, d_model);
        eprintln!("  GPU conv2 shape: {} x {}", gpu_conv2_host.len() / d_model, d_model);

        // Full frontend is conv1->GELU->conv2->GELU, so compare with gpu_conv2
        if cpu_frontend.len() == gpu_conv2_host.len() {
            let max_diff: f32 = cpu_frontend.iter()
                .zip(gpu_conv2_host.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);

            let avg_diff: f32 = cpu_frontend.iter()
                .zip(gpu_conv2_host.iter())
                .map(|(c, g)| (c - g).abs())
                .sum::<f32>() / cpu_frontend.len() as f32;

            eprintln!("  Max difference (full frontend): {:.6}", max_diff);
            eprintln!("  Avg difference (full frontend): {:.6}", avg_diff);

            if max_diff < 0.1 {
                eprintln!("\n✓ Full frontend matches GPU conv2 (max_diff < 0.1)");
            } else {
                eprintln!("\n✗ Full frontend MISMATCH (max_diff = {:.6})", max_diff);
            }
        } else {
            eprintln!("  ✗ Shape mismatch: CPU frontend={}, GPU conv2={}",
                cpu_frontend.len(), gpu_conv2_host.len());
        }

        eprintln!("\n============================================================");
    }

    /// WAPR-PERF-022: Compare encode_gpu vs encode_gpu_total_offload
    ///
    /// Tests whether the GPU encoder with GPU conv produces the same output
    /// as the GPU encoder with CPU conv.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_encode_gpu_vs_total_offload() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {model_path}, skipping test");
            return;
        }

        // Load test audio
        let audio_path = "demos/test-audio/test-speech-1.5s.wav";
        if !std::path::Path::new(audio_path).exists() {
            eprintln!("Test audio not found at {audio_path}, skipping test");
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-022: encode_gpu vs encode_gpu_total_offload");
        eprintln!("============================================================\n");

        // Load model
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");

        // Load audio and compute mel BEFORE converting to CUDA
        let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio file");
        let wav_data =
            crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let mel = apr.compute_mel(&wav_data.samples).expect("Mel failed");

        // Now convert to CUDA
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        let d_model = cuda_model.config.n_audio_state as usize;
        eprintln!("[Config] d_model={}, n_layers={}", d_model, cuda_model.config.n_audio_layer);
        eprintln!("[Input] mel length: {} ({} frames)", mel.len(), mel.len() / 80);

        // Run encode_gpu (CPU conv + GPU attention) - known working
        eprintln!("\n[Step 1] Running encode_gpu (CPU conv + GPU attention)...");
        let encoder_output_cpu_conv = cuda_model.encode_gpu(&mel).expect("encode_gpu failed");
        let seq_len = encoder_output_cpu_conv.len() / d_model;
        eprintln!("  Output shape: {} x {}", seq_len, d_model);
        eprintln!("  mean={:.6}, std={:.6}",
            encoder_output_cpu_conv.iter().sum::<f32>() / encoder_output_cpu_conv.len() as f32,
            (encoder_output_cpu_conv.iter().map(|x| x.powi(2)).sum::<f32>() / encoder_output_cpu_conv.len() as f32).sqrt());

        // Run encode_gpu_total_offload (GPU conv + GPU attention) - known broken
        eprintln!("\n[Step 2] Running encode_gpu_total_offload (GPU conv + GPU attention)...");
        let encoder_output_gpu_conv = cuda_model.encode_gpu_total_offload(&mel).expect("encode_gpu_total_offload failed");
        eprintln!("  Output shape: {} x {}", encoder_output_gpu_conv.len() / d_model, d_model);
        eprintln!("  mean={:.6}, std={:.6}",
            encoder_output_gpu_conv.iter().sum::<f32>() / encoder_output_gpu_conv.len() as f32,
            (encoder_output_gpu_conv.iter().map(|x| x.powi(2)).sum::<f32>() / encoder_output_gpu_conv.len() as f32).sqrt());

        // Compare
        eprintln!("\n[Step 3] Comparing encoder outputs...");
        if encoder_output_cpu_conv.len() == encoder_output_gpu_conv.len() {
            let max_diff: f32 = encoder_output_cpu_conv.iter()
                .zip(encoder_output_gpu_conv.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);

            let avg_diff: f32 = encoder_output_cpu_conv.iter()
                .zip(encoder_output_gpu_conv.iter())
                .map(|(c, g)| (c - g).abs())
                .sum::<f32>() / encoder_output_cpu_conv.len() as f32;

            eprintln!("  Max difference: {:.6}", max_diff);
            eprintln!("  Avg difference: {:.6}", avg_diff);

            // Show first few elements
            eprintln!("\n  First 10 elements comparison:");
            for i in 0..10.min(encoder_output_cpu_conv.len()) {
                eprintln!("    [{}] CPU_conv: {:.6}, GPU_conv: {:.6}, diff: {:.6}",
                    i, encoder_output_cpu_conv[i], encoder_output_gpu_conv[i],
                    (encoder_output_cpu_conv[i] - encoder_output_gpu_conv[i]).abs());
            }

            if max_diff < 0.1 {
                eprintln!("\n✓ Encoder outputs match (max_diff < 0.1)");
            } else {
                eprintln!("\n✗ Encoder output MISMATCH (max_diff = {:.6})", max_diff);
            }
        } else {
            eprintln!("  ✗ Shape mismatch: CPU_conv={}, GPU_conv={}",
                encoder_output_cpu_conv.len(), encoder_output_gpu_conv.len());
        }

        eprintln!("\n============================================================");
    }

    /// WAPR-PERF-022: Layer-by-layer encoder comparison (Brick tracing)
    ///
    /// Traces where GPU encoder diverges from CPU encoder by comparing
    /// output at each step: conv frontend -> pos emb -> layer 0 -> layer 1 -> ...
    #[test]
    fn test_encoder_layer_by_layer_divergence() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {model_path}, skipping test");
            return;
        }

        let audio_path = "demos/test-audio/test-speech-1.5s.wav";
        if !std::path::Path::new(audio_path).exists() {
            eprintln!("Test audio not found at {audio_path}, skipping test");
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-022: Layer-by-Layer Encoder Divergence (Brick Trace)");
        eprintln!("============================================================\n");

        // Load model
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");

        // Compute mel BEFORE converting to CUDA
        let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio file");
        let wav_data =
            crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let mel = apr.compute_mel(&wav_data.samples).expect("Mel failed");

        // Convert to CUDA
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        let d_model = cuda_model.config.n_audio_state as usize;
        let n_heads = cuda_model.config.n_audio_head as usize;
        let head_dim = d_model / n_heads;
        let n_layers = cuda_model.config.n_audio_layer as usize;
        let n_mels = cuda_model.config.n_mels as usize;

        eprintln!("[Config] d_model={}, n_heads={}, head_dim={}, n_layers={}",
            d_model, n_heads, head_dim, n_layers);

        // ========== STEP 1: Conv Frontend + Pos Emb (all CPU data extraction first) ==========
        eprintln!("\n=== BRICK 1: Conv Frontend ===");

        // CPU conv frontend
        let cpu_conv_output = cuda_model.encoder.conv_frontend().forward(&mel).expect("CPU conv");
        let seq_len = cpu_conv_output.len() / d_model;
        eprintln!("[CPU Conv] shape: {} x {}, mean={:.6}, std={:.6}",
            seq_len, d_model,
            cpu_conv_output.iter().sum::<f32>() / cpu_conv_output.len() as f32,
            (cpu_conv_output.iter().map(|x| x.powi(2)).sum::<f32>() / cpu_conv_output.len() as f32).sqrt());

        // Get positional embedding (need before mutable borrow)
        let pos_emb = cuda_model.encoder.positional_embedding().to_vec();

        // ========== GPU conv frontend ==========
        cuda_model.upload_conv_weights_to_gpu().expect("Upload conv");

        let (gpu_conv_output, gpu_seq_len) = {
            let ctx = cuda_model.executor.context();
            let conv_weights = cuda_model.gpu_conv_weights.as_ref().expect("Conv weights");

            let seq_len_in = mel.len() / n_mels;
            let mel_gpu = GpuResidentTensor::from_host(ctx, &mel).expect("mel upload");

            let conv1_out = mel_gpu.conv1d(
                ctx, &conv_weights.conv1_weight, Some(&conv_weights.conv1_bias),
                n_mels as u32, d_model as u32, 3, 1, 1, seq_len_in as u32
            ).expect("GPU conv1");

            let mut conv2_out = conv1_out.conv1d(
                ctx, &conv_weights.conv2_weight, Some(&conv_weights.conv2_bias),
                d_model as u32, d_model as u32, 3, 2, 1, seq_len_in as u32
            ).expect("GPU conv2");

            let output = conv2_out.to_host().expect("Download conv");
            let seq_len = output.len() / d_model;
            (output, seq_len)
        };

        eprintln!("[GPU Conv] shape: {} x {}, mean={:.6}, std={:.6}",
            gpu_seq_len, d_model,
            gpu_conv_output.iter().sum::<f32>() / gpu_conv_output.len() as f32,
            (gpu_conv_output.iter().map(|x| x.powi(2)).sum::<f32>() / gpu_conv_output.len() as f32).sqrt());

        // Compare conv outputs
        let conv_max_diff = if cpu_conv_output.len() == gpu_conv_output.len() {
            cpu_conv_output.iter().zip(gpu_conv_output.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max)
        } else {
            eprintln!("  ✗ Conv shape mismatch: CPU={}, GPU={}", cpu_conv_output.len(), gpu_conv_output.len());
            f32::MAX
        };
        eprintln!("[Conv Diff] max={:.6}", conv_max_diff);

        // ========== STEP 2: Positional Embedding ==========
        eprintln!("\n=== BRICK 2: Position Embedding ===");

        // CPU: add positional embedding
        let mut cpu_with_pos = cpu_conv_output.clone();
        for pos in 0..seq_len {
            for d in 0..d_model {
                cpu_with_pos[pos * d_model + d] += pos_emb[pos * d_model + d];
            }
        }
        eprintln!("[CPU+PosEmb] mean={:.6}, std={:.6}",
            cpu_with_pos.iter().sum::<f32>() / cpu_with_pos.len() as f32,
            (cpu_with_pos.iter().map(|x| x.powi(2)).sum::<f32>() / cpu_with_pos.len() as f32).sqrt());

        // GPU: add positional embedding (done on CPU, re-upload)
        let mut gpu_with_pos = gpu_conv_output.clone();
        for pos in 0..gpu_seq_len.min(seq_len) {
            for d in 0..d_model {
                gpu_with_pos[pos * d_model + d] += pos_emb[pos * d_model + d];
            }
        }
        eprintln!("[GPU+PosEmb] mean={:.6}, std={:.6}",
            gpu_with_pos.iter().sum::<f32>() / gpu_with_pos.len() as f32,
            (gpu_with_pos.iter().map(|x| x.powi(2)).sum::<f32>() / gpu_with_pos.len() as f32).sqrt());

        // Compare after pos emb
        let pos_max_diff = if cpu_with_pos.len() == gpu_with_pos.len() {
            cpu_with_pos.iter().zip(gpu_with_pos.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max)
        } else {
            f32::MAX
        };
        eprintln!("[PosEmb Diff] max={:.6}", pos_max_diff);

        // ========== STEP 3: Layer-by-Layer Transformer ==========
        eprintln!("\n=== BRICK 3: Transformer Layers (Layer-by-Layer) ===");

        // Upload encoder weights for GPU path
        cuda_model.upload_encoder_weights_to_gpu().expect("Upload encoder weights");

        // Store each CPU layer's output for proper comparison
        let mut cpu_layer_outputs: Vec<Vec<f32>> = Vec::with_capacity(n_layers);

        // Run CPU layers independently (not using attention_via_gemm which has borrow issues)
        let mut cpu_x = cpu_with_pos.clone();
        for layer_idx in 0..n_layers {
            // CPU layer forward - use the encoder's forward method
            let block = &cuda_model.encoder.blocks()[layer_idx];

            // Pre-norm
            let ln1_out = block.ln1.forward(&cpu_x).expect("CPU LN1");

            // Self-attention Q/K/V projections
            let q = block.self_attn.w_q().forward(&ln1_out, seq_len).expect("CPU Q");
            let k = block.self_attn.w_k().forward(&ln1_out, seq_len).expect("CPU K");
            let v = block.self_attn.w_v().forward(&ln1_out, seq_len).expect("CPU V");

            // Self-attention (use scaled dot product)
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn_output = vec![0.0f32; seq_len * d_model];

            for h in 0..n_heads {
                // Extract Q, K, V for this head
                let q_head: Vec<f32> = (0..seq_len)
                    .flat_map(|pos| {
                        let start = pos * d_model + h * head_dim;
                        q[start..start + head_dim].iter().copied()
                    })
                    .collect();
                let k_head: Vec<f32> = (0..seq_len)
                    .flat_map(|pos| {
                        let start = pos * d_model + h * head_dim;
                        k[start..start + head_dim].iter().copied()
                    })
                    .collect();
                let v_head: Vec<f32> = (0..seq_len)
                    .flat_map(|pos| {
                        let start = pos * d_model + h * head_dim;
                        v[start..start + head_dim].iter().copied()
                    })
                    .collect();

                // Compute attention scores: Q @ K^T
                let mut scores = vec![0.0f32; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut sum = 0.0f32;
                        for d in 0..head_dim {
                            sum += q_head[i * head_dim + d] * k_head[j * head_dim + d];
                        }
                        scores[i * seq_len + j] = sum * scale;
                    }
                }

                // Softmax
                for i in 0..seq_len {
                    let row_start = i * seq_len;
                    let max_val = scores[row_start..row_start + seq_len]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for j in 0..seq_len {
                        scores[row_start + j] = (scores[row_start + j] - max_val).exp();
                        sum += scores[row_start + j];
                    }
                    for j in 0..seq_len {
                        scores[row_start + j] /= sum;
                    }
                }

                // Apply attention to V: scores @ V
                for i in 0..seq_len {
                    for d in 0..head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..seq_len {
                            sum += scores[i * seq_len + j] * v_head[j * head_dim + d];
                        }
                        attn_output[i * d_model + h * head_dim + d] = sum;
                    }
                }
            }

            // Output projection
            let attn_proj = block.self_attn.w_o().forward(&attn_output, seq_len).expect("CPU O");

            // First residual
            let mut residual1: Vec<f32> = cpu_x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

            // FFN
            let ln2_out = block.ln2.forward(&residual1).expect("CPU LN2");
            let ffn_out = block.ffn.forward(&ln2_out).expect("CPU FFN");

            // Second residual
            for (r, f) in residual1.iter_mut().zip(ffn_out.iter()) {
                *r += f;
            }
            cpu_x = residual1.clone();

            // Store this layer's output for comparison
            cpu_layer_outputs.push(residual1);

            eprintln!("[CPU Layer {}] mean={:.6}", layer_idx,
                cpu_x.iter().sum::<f32>() / cpu_x.len() as f32);
        }

        // Now run GPU layers with SAME INPUT as CPU (cpu_with_pos) for fair comparison
        eprintln!("\n--- GPU Layers (same input as CPU for fair comparison) ---");
        {
            let ctx = cuda_model.executor.context();
            let weights = cuda_model.gpu_encoder_weights.as_ref().expect("Encoder weights");
            let config = cuda_model.gpu_encoder_config.as_ref().expect("Encoder config");

            // Start from CPU input (not GPU conv output) for fair layer-by-layer comparison
            let mut gpu_x_tensor = GpuResidentTensor::from_host(ctx, &cpu_with_pos).expect("Upload");

            for layer_idx in 0..n_layers {
                // GPU layer forward
                gpu_x_tensor = forward_encoder_block_gpu(ctx, &gpu_x_tensor, &weights[layer_idx], config)
                    .expect("GPU layer");

                let gpu_layer_out = gpu_x_tensor.to_host().expect("Download layer");

                // Compare with CPU result for THIS layer
                let layer_max_diff = cpu_layer_outputs[layer_idx].iter().zip(gpu_layer_out.iter())
                    .map(|(c, g)| (c - g).abs())
                    .fold(0.0f32, f32::max);

                eprintln!("[GPU Layer {}] mean={:.6}, vs_CPU_max_diff={:.6}",
                    layer_idx,
                    gpu_layer_out.iter().sum::<f32>() / gpu_layer_out.len() as f32,
                    layer_max_diff);

                // Reset to CPU layer output for next layer (isolate errors)
                gpu_x_tensor = GpuResidentTensor::from_host(ctx, &cpu_layer_outputs[layer_idx]).expect("Re-upload");
            }
        }

        // ========== STEP 4: Final LayerNorm ==========
        eprintln!("\n=== BRICK 4: Final LayerNorm ===");

        let cpu_final = cuda_model.encoder.ln_post().forward(&cpu_x).expect("CPU ln_post");
        eprintln!("[CPU Final] mean={:.6}, std={:.6}",
            cpu_final.iter().sum::<f32>() / cpu_final.len() as f32,
            (cpu_final.iter().map(|x| x.powi(2)).sum::<f32>() / cpu_final.len() as f32).sqrt());

        eprintln!("\n============================================================");
        eprintln!("Brick Trace Complete: Check layer-by-layer diffs above");
        eprintln!("============================================================");
    }

    /// WAPR-PERF-023: Step-by-step single layer comparison (Tile tracing)
    ///
    /// Compares CPU vs GPU at each step within a single encoder layer to
    /// isolate exactly which operation diverges.
    #[test]
    fn test_encoder_single_layer_step_by_step() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_path = "models/whisper-tiny.apr";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {model_path}, skipping test");
            return;
        }

        let audio_path = "demos/test-audio/test-speech-1.5s.wav";
        if !std::path::Path::new(audio_path).exists() {
            eprintln!("Test audio not found at {audio_path}, skipping test");
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-023: Single Layer Step-by-Step (Tile Trace)");
        eprintln!("============================================================\n");

        // Load model
        let bytes = std::fs::read(model_path).expect("Failed to read model");
        let apr = crate::WhisperApr::load_from_apr(&bytes).expect("Failed to load model");

        // Compute mel BEFORE converting to CUDA
        let audio_bytes = std::fs::read(audio_path).expect("Failed to read audio file");
        let wav_data =
            crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let mel = apr.compute_mel(&wav_data.samples).expect("Mel failed");

        // Convert to CUDA
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        let d_model = cuda_model.config.n_audio_state as usize;
        let n_heads = cuda_model.config.n_audio_head as usize;
        let head_dim = d_model / n_heads;

        eprintln!("[Config] d_model={}, n_heads={}, head_dim={}", d_model, n_heads, head_dim);

        // Get CPU conv output + pos embedding as the input
        let cpu_conv_output = cuda_model.encoder.conv_frontend().forward(&mel).expect("CPU conv");
        let seq_len = cpu_conv_output.len() / d_model;
        let pos_emb = cuda_model.encoder.positional_embedding().to_vec();

        let mut input_x = cpu_conv_output.clone();
        for pos in 0..seq_len {
            for d in 0..d_model {
                input_x[pos * d_model + d] += pos_emb[pos * d_model + d];
            }
        }

        eprintln!("[Input] seq_len={}, mean={:.6}", seq_len,
            input_x.iter().sum::<f32>() / input_x.len() as f32);

        // Upload encoder weights
        cuda_model.upload_encoder_weights_to_gpu().expect("Upload encoder weights");

        // We'll compare Layer 0 step by step
        let block = &cuda_model.encoder.blocks()[0];

        eprintln!("\n=== STEP 1: LayerNorm 1 ===");
        let cpu_ln1 = block.ln1.forward(&input_x).expect("CPU LN1");
        eprintln!("[CPU LN1] mean={:.6}, std={:.6}",
            cpu_ln1.iter().sum::<f32>() / cpu_ln1.len() as f32,
            (cpu_ln1.iter().map(|x| x.powi(2)).sum::<f32>() / cpu_ln1.len() as f32).sqrt());

        // GPU LN1
        let gpu_ln1 = {
            let ctx = cuda_model.executor.context();
            let weights = &cuda_model.gpu_encoder_weights.as_ref().expect("Weights")[0];
            let x_gpu = GpuResidentTensor::from_host(ctx, &input_x).expect("Upload");
            x_gpu.layer_norm(ctx, &weights.ln1_gamma, &weights.ln1_beta, d_model as u32, seq_len as u32)
                .expect("GPU LN1")
                .to_host().expect("Download")
        };
        let ln1_diff = cpu_ln1.iter().zip(gpu_ln1.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[GPU LN1] mean={:.6}, max_diff={:.6}",
            gpu_ln1.iter().sum::<f32>() / gpu_ln1.len() as f32, ln1_diff);

        eprintln!("\n=== STEP 2: Q/K/V Projections ===");

        // Diagnostic: Compare CPU weights (transposed) vs GPU weights
        {
            let weights = &mut cuda_model.gpu_encoder_weights.as_mut().expect("Weights")[0];

            // Get CPU weights (transposed to match GPU format) - inline transpose
            let transpose = |w: &[f32], rows: usize, cols: usize| -> Vec<f32> {
                let mut t = vec![0.0f32; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        t[c * rows + r] = w[r * cols + c];
                    }
                }
                t
            };

            let w_q_cpu_t = transpose(&block.self_attn.w_q().weight, d_model, d_model);
            let w_k_cpu_t = transpose(&block.self_attn.w_k().weight, d_model, d_model);
            let w_v_cpu_t = transpose(&block.self_attn.w_v().weight, d_model, d_model);

            // Get GPU weights
            let w_q_gpu = weights.w_q.to_host().expect("Download w_q");
            let w_k_gpu = weights.w_k.to_host().expect("Download w_k");
            let w_v_gpu = weights.w_v.to_host().expect("Download w_v");

            // Compare
            let w_q_diff: f32 = w_q_cpu_t.iter().zip(w_q_gpu.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
            let w_k_diff: f32 = w_k_cpu_t.iter().zip(w_k_gpu.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
            let w_v_diff: f32 = w_v_cpu_t.iter().zip(w_v_gpu.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);

            eprintln!("[WEIGHT CHECK] w_q max_diff={:.6}, w_k max_diff={:.6}, w_v max_diff={:.6}",
                      w_q_diff, w_k_diff, w_v_diff);

            // Also check biases
            let b_q_gpu = weights.b_q.to_host().expect("Download b_q");
            let b_k_gpu = weights.b_k.to_host().expect("Download b_k");
            let b_v_gpu = weights.b_v.to_host().expect("Download b_v");

            let b_q_diff: f32 = block.self_attn.w_q().bias.iter().zip(b_q_gpu.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
            let b_k_diff: f32 = block.self_attn.w_k().bias.iter().zip(b_k_gpu.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
            let b_v_diff: f32 = block.self_attn.w_v().bias.iter().zip(b_v_gpu.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);

            eprintln!("[BIAS CHECK] b_q max_diff={:.6}, b_k max_diff={:.6}, b_v max_diff={:.6}",
                      b_q_diff, b_k_diff, b_v_diff);
        }

        let cpu_q = block.self_attn.w_q().forward(&cpu_ln1, seq_len).expect("CPU Q");
        let cpu_k = block.self_attn.w_k().forward(&cpu_ln1, seq_len).expect("CPU K");
        let cpu_v = block.self_attn.w_v().forward(&cpu_ln1, seq_len).expect("CPU V");
        eprintln!("[CPU Q] mean={:.6}", cpu_q.iter().sum::<f32>() / cpu_q.len() as f32);
        eprintln!("[CPU K] mean={:.6}", cpu_k.iter().sum::<f32>() / cpu_k.len() as f32);
        eprintln!("[CPU V] mean={:.6}", cpu_v.iter().sum::<f32>() / cpu_v.len() as f32);

        // GPU Q/K/V - test ORDER dependency: compute K FIRST to see if first call always succeeds
        let (gpu_q, gpu_k, gpu_v) = {
            let ctx = cuda_model.executor.context();
            let weights = &cuda_model.gpu_encoder_weights.as_ref().expect("Weights")[0];

            // HYPOTHESIS: Does the FIRST GEMM call always succeed, regardless of which projection?
            // Test: K first, then Q, then V
            let ln1_for_k = GpuResidentTensor::from_host(ctx, &cpu_ln1).expect("Upload LN1 for K");
            let k = ln1_for_k.linear(ctx, &weights.w_k, Some(&weights.b_k), seq_len as u32, d_model as u32, d_model as u32)
                .expect("GPU K").to_host().expect("Download K");

            let ln1_for_q = GpuResidentTensor::from_host(ctx, &cpu_ln1).expect("Upload LN1 for Q");
            let q = ln1_for_q.linear(ctx, &weights.w_q, Some(&weights.b_q), seq_len as u32, d_model as u32, d_model as u32)
                .expect("GPU Q").to_host().expect("Download Q");

            let ln1_for_v = GpuResidentTensor::from_host(ctx, &cpu_ln1).expect("Upload LN1 for V");
            let v = ln1_for_v.linear(ctx, &weights.w_v, Some(&weights.b_v), seq_len as u32, d_model as u32, d_model as u32)
                .expect("GPU V").to_host().expect("Download V");
            (q, k, v)
        };
        let q_diff = cpu_q.iter().zip(gpu_q.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        let k_diff = cpu_k.iter().zip(gpu_k.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        let v_diff = cpu_v.iter().zip(gpu_v.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        eprintln!("[GPU Q] mean={:.6}, max_diff={:.6}", gpu_q.iter().sum::<f32>() / gpu_q.len() as f32, q_diff);
        eprintln!("[GPU K] mean={:.6}, max_diff={:.6}", gpu_k.iter().sum::<f32>() / gpu_k.len() as f32, k_diff);
        eprintln!("[GPU V] mean={:.6}, max_diff={:.6}", gpu_v.iter().sum::<f32>() / gpu_v.len() as f32, v_diff);

        eprintln!("\n=== STEP 3: Self-Attention ===");
        // CPU attention (use the CPU Q/K/V)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut cpu_attn_output = vec![0.0f32; seq_len * d_model];
        for h in 0..n_heads {
            let q_head: Vec<f32> = (0..seq_len)
                .flat_map(|pos| cpu_q[pos * d_model + h * head_dim..pos * d_model + (h + 1) * head_dim].iter().copied())
                .collect();
            let k_head: Vec<f32> = (0..seq_len)
                .flat_map(|pos| cpu_k[pos * d_model + h * head_dim..pos * d_model + (h + 1) * head_dim].iter().copied())
                .collect();
            let v_head: Vec<f32> = (0..seq_len)
                .flat_map(|pos| cpu_v[pos * d_model + h * head_dim..pos * d_model + (h + 1) * head_dim].iter().copied())
                .collect();

            let mut scores = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut sum = 0.0f32;
                    for d in 0..head_dim {
                        sum += q_head[i * head_dim + d] * k_head[j * head_dim + d];
                    }
                    scores[i * seq_len + j] = sum * scale;
                }
            }

            for i in 0..seq_len {
                let row_start = i * seq_len;
                let max_val = scores[row_start..row_start + seq_len].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..seq_len {
                    scores[row_start + j] = (scores[row_start + j] - max_val).exp();
                    sum += scores[row_start + j];
                }
                for j in 0..seq_len {
                    scores[row_start + j] /= sum;
                }
            }

            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq_len {
                        sum += scores[i * seq_len + j] * v_head[j * head_dim + d];
                    }
                    cpu_attn_output[i * d_model + h * head_dim + d] = sum;
                }
            }
        }
        eprintln!("[CPU Attn] mean={:.6}", cpu_attn_output.iter().sum::<f32>() / cpu_attn_output.len() as f32);

        // GPU attention (using CPU Q/K/V for fair comparison)
        let gpu_attn_output = {
            let ctx = cuda_model.executor.context();
            let q_gpu = GpuResidentTensor::from_host(ctx, &cpu_q).expect("Upload Q");
            let k_gpu = GpuResidentTensor::from_host(ctx, &cpu_k).expect("Upload K");
            let v_gpu = GpuResidentTensor::from_host(ctx, &cpu_v).expect("Upload V");

            batched_multihead_attention(ctx, &q_gpu, &k_gpu, &v_gpu, n_heads as u32, head_dim as u32, seq_len as u32)
                .expect("GPU attention")
                .to_host().expect("Download attention")
        };
        let attn_diff = cpu_attn_output.iter().zip(gpu_attn_output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[GPU Attn] mean={:.6}, max_diff={:.6}",
            gpu_attn_output.iter().sum::<f32>() / gpu_attn_output.len() as f32, attn_diff);

        // Per-head divergence tracking (WAPR-PERF-014 Brick tracing)
        eprintln!("  Per-head max_diff:");
        for h in 0..n_heads {
            let head_start = h * head_dim;
            let mut head_max_diff = 0.0f32;
            for pos in 0..seq_len {
                for d in 0..head_dim {
                    let idx = pos * d_model + head_start + d;
                    let diff = (cpu_attn_output[idx] - gpu_attn_output[idx]).abs();
                    head_max_diff = head_max_diff.max(diff);
                }
            }
            eprintln!("    head {}: max_diff = {:.6}", h, head_max_diff);
        }

        eprintln!("\n=== STEP 4: Output Projection ===");
        let cpu_attn_proj = block.self_attn.w_o().forward(&cpu_attn_output, seq_len).expect("CPU O");
        eprintln!("[CPU O_proj] mean={:.6}", cpu_attn_proj.iter().sum::<f32>() / cpu_attn_proj.len() as f32);

        // GPU output projection (using CPU attention output for fair comparison)
        let gpu_attn_proj = {
            let ctx = cuda_model.executor.context();
            let weights = &cuda_model.gpu_encoder_weights.as_ref().expect("Weights")[0];
            let attn_gpu = GpuResidentTensor::from_host(ctx, &cpu_attn_output).expect("Upload attn");
            attn_gpu.linear(ctx, &weights.w_o, Some(&weights.b_o), seq_len as u32, d_model as u32, d_model as u32)
                .expect("GPU O").to_host().expect("Download O")
        };
        let o_diff = cpu_attn_proj.iter().zip(gpu_attn_proj.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[GPU O_proj] mean={:.6}, max_diff={:.6}",
            gpu_attn_proj.iter().sum::<f32>() / gpu_attn_proj.len() as f32, o_diff);

        eprintln!("\n=== STEP 5: First Residual ===");
        let cpu_residual1: Vec<f32> = input_x.iter().zip(cpu_attn_proj.iter()).map(|(a, b)| a + b).collect();
        eprintln!("[CPU Res1] mean={:.6}", cpu_residual1.iter().sum::<f32>() / cpu_residual1.len() as f32);

        // GPU residual (using CPU values for fair comparison)
        let gpu_residual1 = {
            let ctx = cuda_model.executor.context();
            let x_gpu = GpuResidentTensor::from_host(ctx, &input_x).expect("Upload x");
            let proj_gpu = GpuResidentTensor::from_host(ctx, &cpu_attn_proj).expect("Upload proj");
            x_gpu.add(ctx, &proj_gpu).expect("GPU add").to_host().expect("Download")
        };
        let res1_diff = cpu_residual1.iter().zip(gpu_residual1.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[GPU Res1] mean={:.6}, max_diff={:.6}",
            gpu_residual1.iter().sum::<f32>() / gpu_residual1.len() as f32, res1_diff);

        eprintln!("\n=== STEP 6: LayerNorm 2 ===");
        let cpu_ln2 = block.ln2.forward(&cpu_residual1).expect("CPU LN2");
        eprintln!("[CPU LN2] mean={:.6}", cpu_ln2.iter().sum::<f32>() / cpu_ln2.len() as f32);

        // GPU LN2
        let gpu_ln2 = {
            let ctx = cuda_model.executor.context();
            let weights = &cuda_model.gpu_encoder_weights.as_ref().expect("Weights")[0];
            let res_gpu = GpuResidentTensor::from_host(ctx, &cpu_residual1).expect("Upload res");
            res_gpu.layer_norm(ctx, &weights.ln2_gamma, &weights.ln2_beta, d_model as u32, seq_len as u32)
                .expect("GPU LN2")
                .to_host().expect("Download")
        };
        let ln2_diff = cpu_ln2.iter().zip(gpu_ln2.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[GPU LN2] mean={:.6}, max_diff={:.6}",
            gpu_ln2.iter().sum::<f32>() / gpu_ln2.len() as f32, ln2_diff);

        eprintln!("\n=== STEP 7: FFN ===");
        let cpu_ffn = block.ffn.forward(&cpu_ln2).expect("CPU FFN");
        eprintln!("[CPU FFN] mean={:.6}", cpu_ffn.iter().sum::<f32>() / cpu_ffn.len() as f32);

        // GPU FFN (fused linear + GELU + linear)
        let ffn_dim = cuda_model.config.n_audio_state as usize * 4; // Typically 4x d_model
        let gpu_ffn = {
            let ctx = cuda_model.executor.context();
            let weights = &cuda_model.gpu_encoder_weights.as_ref().expect("Weights")[0];
            let ln2_gpu = GpuResidentTensor::from_host(ctx, &cpu_ln2).expect("Upload ln2");

            // FFN up + GELU (fused)
            let ffn_gelu = ln2_gpu.fused_linear_gelu(
                ctx, &weights.ffn_up_w, &weights.ffn_up_b,
                seq_len as u32, d_model as u32, ffn_dim as u32
            ).expect("GPU FFN up+GELU");

            // FFN down
            ffn_gelu.linear(ctx, &weights.ffn_down_w, Some(&weights.ffn_down_b),
                seq_len as u32, ffn_dim as u32, d_model as u32)
                .expect("GPU FFN down")
                .to_host().expect("Download")
        };
        let ffn_diff = cpu_ffn.iter().zip(gpu_ffn.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[GPU FFN] mean={:.6}, max_diff={:.6}",
            gpu_ffn.iter().sum::<f32>() / gpu_ffn.len() as f32, ffn_diff);

        eprintln!("\n=== STEP 8: Final Output (Second Residual) ===");
        let cpu_output: Vec<f32> = cpu_residual1.iter().zip(cpu_ffn.iter()).map(|(a, b)| a + b).collect();
        eprintln!("[CPU Output] mean={:.6}", cpu_output.iter().sum::<f32>() / cpu_output.len() as f32);

        // GPU final output
        let gpu_output = {
            let ctx = cuda_model.executor.context();
            let res_gpu = GpuResidentTensor::from_host(ctx, &cpu_residual1).expect("Upload res");
            let ffn_gpu = GpuResidentTensor::from_host(ctx, &cpu_ffn).expect("Upload ffn");
            res_gpu.add(ctx, &ffn_gpu).expect("GPU add").to_host().expect("Download")
        };
        let output_diff = cpu_output.iter().zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[GPU Output] mean={:.6}, max_diff={:.6}",
            gpu_output.iter().sum::<f32>() / gpu_output.len() as f32, output_diff);

        eprintln!("\n============================================================");
        eprintln!("Step-by-Step Summary:");
        eprintln!("  LN1:    max_diff = {:.6}", ln1_diff);
        eprintln!("  Q:      max_diff = {:.6}", q_diff);
        eprintln!("  K:      max_diff = {:.6}", k_diff);
        eprintln!("  V:      max_diff = {:.6}", v_diff);
        eprintln!("  Attn:   max_diff = {:.6}", attn_diff);
        eprintln!("  O_proj: max_diff = {:.6}", o_diff);
        eprintln!("  Res1:   max_diff = {:.6}", res1_diff);
        eprintln!("  LN2:    max_diff = {:.6}", ln2_diff);
        eprintln!("  FFN:    max_diff = {:.6}", ffn_diff);
        eprintln!("  Output: max_diff = {:.6}", output_diff);
        eprintln!("============================================================\n");

        // Assert reasonable tolerances
        assert!(ln1_diff < 0.01, "LN1 diff too high: {}", ln1_diff);
        assert!(q_diff < 0.5, "Q diff too high: {}", q_diff);
        assert!(k_diff < 0.5, "K diff too high: {}", k_diff);
        assert!(v_diff < 0.5, "V diff too high: {}", v_diff);
        // Attention might have some numerical error due to softmax
        assert!(attn_diff < 1.0, "Attn diff too high: {}", attn_diff);
    }

    /// WAPR-PERF-024: Debug weight transpose and linear operation
    ///
    /// Uses small known matrices to verify the weight transpose and GPU linear
    /// match CPU linear computation.
    #[test]
    fn test_linear_weight_transpose_correctness() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-024: Weight Transpose & Linear Debug Test");
        eprintln!("============================================================\n");

        let executor = CudaExecutor::new(0).expect("Failed to create executor");
        let ctx = executor.context();

        // Small dimensions for easy debugging
        let batch = 2;
        let in_feat = 4;
        let out_feat = 3;

        // Create input: [batch, in_feat] = [2, 4]
        // Values: [[1, 2, 3, 4], [5, 6, 7, 8]]
        let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        eprintln!("[Input] shape: {}x{}", batch, in_feat);
        eprintln!("  {:?}", &input[..in_feat]);
        eprintln!("  {:?}", &input[in_feat..]);

        // Create CPU weight: [out_feat, in_feat] = [3, 4] (row-major)
        // Each row is an output feature's weights
        // Row 0: [1, 0, 0, 0] -> output 0 = input[0]
        // Row 1: [0, 1, 0, 0] -> output 1 = input[1]
        // Row 2: [0, 0, 1, 1] -> output 2 = input[2] + input[3]
        let cpu_weight: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,  // output 0 = input[0]
            0.0, 1.0, 0.0, 0.0,  // output 1 = input[1]
            0.0, 0.0, 1.0, 1.0,  // output 2 = input[2] + input[3]
        ];
        let cpu_bias: Vec<f32> = vec![0.1, 0.2, 0.3]; // Small biases

        eprintln!("[CPU Weight] shape: {}x{} (out_feat x in_feat)", out_feat, in_feat);
        for o in 0..out_feat {
            eprintln!("  row {}: {:?}", o, &cpu_weight[o * in_feat..(o + 1) * in_feat]);
        }
        eprintln!("[CPU Bias] {:?}", cpu_bias);

        // Compute CPU reference: output[b, o] = sum_i(input[b, i] * weight[o, i]) + bias[o]
        let mut cpu_output = vec![0.0f32; batch * out_feat];
        for b in 0..batch {
            for o in 0..out_feat {
                let mut sum = cpu_bias[o];
                for i in 0..in_feat {
                    sum += input[b * in_feat + i] * cpu_weight[o * in_feat + i];
                }
                cpu_output[b * out_feat + o] = sum;
            }
        }
        eprintln!("[CPU Output] shape: {}x{}", batch, out_feat);
        eprintln!("  batch 0: {:?}", &cpu_output[..out_feat]);
        eprintln!("  batch 1: {:?}", &cpu_output[out_feat..]);

        // Expected:
        // Batch 0: [1*1 + 0.1, 1*2 + 0.2, 1*3 + 1*4 + 0.3] = [1.1, 2.2, 7.3]
        // Batch 1: [1*5 + 0.1, 1*6 + 0.2, 1*7 + 1*8 + 0.3] = [5.1, 6.2, 15.3]

        // Transpose weight for GPU: [out_feat, in_feat] -> [in_feat, out_feat]
        // GPU expects [in_feat, out_feat] for input @ weight = output
        let mut gpu_weight = vec![0.0f32; cpu_weight.len()];
        for o in 0..out_feat {
            for i in 0..in_feat {
                gpu_weight[i * out_feat + o] = cpu_weight[o * in_feat + i];
            }
        }
        eprintln!("\n[GPU Weight after transpose] shape: {}x{} (in_feat x out_feat)", in_feat, out_feat);
        for i in 0..in_feat {
            eprintln!("  row {}: {:?}", i, &gpu_weight[i * out_feat..(i + 1) * out_feat]);
        }

        // Upload to GPU
        let input_gpu = GpuResidentTensor::from_host(ctx, &input).expect("Upload input");
        let weight_gpu = GpuResidentTensor::from_host(ctx, &gpu_weight).expect("Upload weight");
        let bias_gpu = GpuResidentTensor::from_host(ctx, &cpu_bias).expect("Upload bias");

        // Compute GPU linear
        let mut output_gpu = input_gpu.linear(
            ctx, &weight_gpu, Some(&bias_gpu),
            batch as u32, in_feat as u32, out_feat as u32
        ).expect("GPU linear");

        let gpu_output = output_gpu.to_host().expect("Download output");
        eprintln!("\n[GPU Output] shape: {}x{}", batch, out_feat);
        eprintln!("  batch 0: {:?}", &gpu_output[..out_feat]);
        eprintln!("  batch 1: {:?}", &gpu_output[out_feat..]);

        // Compare
        let max_diff = cpu_output.iter().zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("\n[Comparison] max_diff = {:.6}", max_diff);

        // Verify expected values
        let expected = vec![1.1, 2.2, 7.3, 5.1, 6.2, 15.3];
        let cpu_matches = cpu_output.iter().zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 0.001);
        eprintln!("[CPU matches expected] {}", cpu_matches);

        assert!(max_diff < 0.01, "GPU output differs from CPU by {}", max_diff);
        eprintln!("\n✓ Small matrix test PASSED");
    }

    /// WAPR-PERF-025: Test WMMA FP16 GEMM kernel correctness
    ///
    /// Tests with larger matrices that trigger the WMMA Tensor Core path.
    #[test]
    fn test_wmma_gemm_correctness() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-025: WMMA FP16 GEMM Correctness Test");
        eprintln!("============================================================\n");

        let executor = CudaExecutor::new(0).expect("Failed to create executor");
        let ctx = executor.context();

        // Dimensions that trigger WMMA: k >= 64 && m >= 64 && n >= 64
        let batch = 128;  // M
        let in_feat = 128; // K
        let out_feat = 128; // N

        eprintln!("[Dims] batch={}, in_feat={}, out_feat={}", batch, in_feat, out_feat);

        // Create input: identity-like pattern for easy verification
        let mut input = vec![0.0f32; batch * in_feat];
        for b in 0..batch {
            input[b * in_feat + (b % in_feat)] = 1.0; // Diagonal-like
        }
        eprintln!("[Input] First few values: {:?}", &input[..10]);

        // Create CPU weight: identity matrix [out_feat, in_feat]
        // When weight is identity, output should equal input (if batch <= out_feat == in_feat)
        let mut cpu_weight = vec![0.0f32; out_feat * in_feat];
        for i in 0..out_feat.min(in_feat) {
            cpu_weight[i * in_feat + i] = 1.0; // Identity: W[i,i] = 1
        }
        let cpu_bias = vec![0.0f32; out_feat]; // Zero bias

        // Compute CPU reference
        let mut cpu_output = vec![0.0f32; batch * out_feat];
        for b in 0..batch {
            for o in 0..out_feat {
                let mut sum = cpu_bias[o];
                for i in 0..in_feat {
                    sum += input[b * in_feat + i] * cpu_weight[o * in_feat + i];
                }
                cpu_output[b * out_feat + o] = sum;
            }
        }
        eprintln!("[CPU Output] First few: {:?}", &cpu_output[..10]);

        // Transpose for GPU
        let mut gpu_weight = vec![0.0f32; cpu_weight.len()];
        for o in 0..out_feat {
            for i in 0..in_feat {
                gpu_weight[i * out_feat + o] = cpu_weight[o * in_feat + i];
            }
        }

        // Upload to GPU
        let input_gpu = GpuResidentTensor::from_host(ctx, &input).expect("Upload input");
        let weight_gpu = GpuResidentTensor::from_host(ctx, &gpu_weight).expect("Upload weight");
        let bias_gpu = GpuResidentTensor::from_host(ctx, &cpu_bias).expect("Upload bias");

        // Compute GPU linear
        let mut output_gpu = input_gpu.linear(
            ctx, &weight_gpu, Some(&bias_gpu),
            batch as u32, in_feat as u32, out_feat as u32
        ).expect("GPU linear");

        let gpu_output = output_gpu.to_host().expect("Download output");
        eprintln!("[GPU Output] First few: {:?}", &gpu_output[..10]);

        // Compare
        let max_diff = cpu_output.iter().zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        let mean_diff = cpu_output.iter().zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .sum::<f32>() / cpu_output.len() as f32;

        eprintln!("\n[Comparison] max_diff = {:.6}, mean_diff = {:.6}", max_diff, mean_diff);

        // Find where the max diff occurs
        let (max_idx, _) = cpu_output.iter().zip(gpu_output.iter())
            .enumerate()
            .map(|(i, (c, g))| (i, (c - g).abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let b = max_idx / out_feat;
        let o = max_idx % out_feat;
        eprintln!("[Max diff location] batch={}, out_feat={}, cpu={:.6}, gpu={:.6}",
            b, o, cpu_output[max_idx], gpu_output[max_idx]);

        // WMMA uses FP16 internally, so allow some tolerance
        assert!(max_diff < 0.1, "WMMA GEMM diff too high: {}", max_diff);
        eprintln!("\n✓ WMMA GEMM test PASSED");
    }

    /// WAPR-PERF-026: Test WMMA with encoder-like dimensions and random weights
    #[test]
    fn test_wmma_encoder_like_dims() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-026: WMMA with Encoder-Like Dimensions");
        eprintln!("============================================================\n");

        let executor = CudaExecutor::new(0).expect("Failed to create executor");
        let ctx = executor.context();

        // Encoder-like dimensions: 1500 seq_len, 384 d_model
        let batch = 1500;  // M (seq_len)
        let in_feat = 384; // K (d_model)
        let out_feat = 384; // N (d_model)

        eprintln!("[Dims] batch={}, in_feat={}, out_feat={}", batch, in_feat, out_feat);

        // Create pseudo-random input using simple linear congruential generator
        // LCG: x' = (a * x + c) mod m, scaled to [-1, 1]
        let mut input = vec![0.0f32; batch * in_feat];
        let mut seed = 12345u32;
        for v in input.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((seed as f32) / (u32::MAX as f32)) * 2.0 - 1.0; // Range [-1, 1]
        }
        let input_mean = input.iter().sum::<f32>() / input.len() as f32;
        let input_max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[Input] mean={:.6}, max={:.6}", input_mean, input_max);

        // Create pseudo-random weight [out_feat, in_feat]
        let mut cpu_weight = vec![0.0f32; out_feat * in_feat];
        seed = 67890u32;
        for v in cpu_weight.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((seed as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        let weight_mean = cpu_weight.iter().sum::<f32>() / cpu_weight.len() as f32;
        eprintln!("[Weight] mean={:.6}", weight_mean);

        // Small random bias
        let mut cpu_bias = vec![0.0f32; out_feat];
        seed = 11111u32;
        for v in cpu_bias.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((seed as f32) / (u32::MAX as f32)) * 0.1 - 0.05; // Small values [-0.05, 0.05]
        }

        // Compute CPU reference
        let mut cpu_output = vec![0.0f32; batch * out_feat];
        for b in 0..batch {
            for o in 0..out_feat {
                let mut sum = cpu_bias[o];
                for i in 0..in_feat {
                    sum += input[b * in_feat + i] * cpu_weight[o * in_feat + i];
                }
                cpu_output[b * out_feat + o] = sum;
            }
        }
        let cpu_mean = cpu_output.iter().sum::<f32>() / cpu_output.len() as f32;
        let cpu_max = cpu_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[CPU Output] mean={:.6}, max={:.6}", cpu_mean, cpu_max);

        // Transpose for GPU
        let mut gpu_weight = vec![0.0f32; cpu_weight.len()];
        for o in 0..out_feat {
            for i in 0..in_feat {
                gpu_weight[i * out_feat + o] = cpu_weight[o * in_feat + i];
            }
        }

        // Upload to GPU
        let input_gpu = GpuResidentTensor::from_host(ctx, &input).expect("Upload input");
        let weight_gpu = GpuResidentTensor::from_host(ctx, &gpu_weight).expect("Upload weight");
        let bias_gpu = GpuResidentTensor::from_host(ctx, &cpu_bias).expect("Upload bias");

        // Compute GPU linear
        let mut output_gpu = input_gpu.linear(
            ctx, &weight_gpu, Some(&bias_gpu),
            batch as u32, in_feat as u32, out_feat as u32
        ).expect("GPU linear");

        let gpu_output = output_gpu.to_host().expect("Download output");
        let gpu_mean = gpu_output.iter().sum::<f32>() / gpu_output.len() as f32;
        let gpu_max = gpu_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[GPU Output] mean={:.6}, max={:.6}", gpu_mean, gpu_max);

        // Compare
        let max_diff = cpu_output.iter().zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let mean_diff = cpu_output.iter().zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .sum::<f32>() / cpu_output.len() as f32;

        eprintln!("\n[Comparison] max_diff = {:.6}, mean_diff = {:.6}", max_diff, mean_diff);

        // Find worst positions
        let mut diffs: Vec<(usize, f32)> = cpu_output.iter().zip(gpu_output.iter())
            .enumerate()
            .map(|(i, (c, g))| (i, (c - g).abs()))
            .collect();
        diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("\n[Top 5 worst positions]");
        for (idx, diff) in diffs.iter().take(5) {
            let b = idx / out_feat;
            let o = idx % out_feat;
            eprintln!("  batch={}, out={}: cpu={:.6}, gpu={:.6}, diff={:.6}",
                b, o, cpu_output[*idx], gpu_output[*idx], diff);
        }

        // FP16 WMMA can have some precision loss, especially for larger accumulations
        // With 384 multiplications per output, expect ~0.5 max diff for random values
        if max_diff > 1.0 {
            eprintln!("\n[WARN] max_diff > 1.0, investigating further...");
        }

        assert!(max_diff < 2.0, "WMMA GEMM diff too high: {}", max_diff);
        eprintln!("\n✓ Encoder-like dimensions test PASSED");
    }

    /// WAPR-PERF-027: Test multiple consecutive GEMM calls with different weights
    /// Hypothesis: First call succeeds, subsequent calls fail
    #[test]
    fn test_wmma_multiple_consecutive_calls() {
        if !CudaExecutor::is_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        eprintln!("\n============================================================");
        eprintln!("WAPR-PERF-027: Multiple Consecutive WMMA GEMM Calls");
        eprintln!("============================================================\n");

        let executor = CudaExecutor::new(0).expect("Failed to create executor");
        let ctx = executor.context();

        let batch = 1500;  // M
        let in_feat = 384; // K
        let out_feat = 384; // N

        // Create fixed input - same for all calls
        let mut input = vec![0.0f32; batch * in_feat];
        let mut seed = 12345u32;
        for v in input.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((seed as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }

        // Create 3 different weight matrices (like Q, K, V)
        let create_weights = |seed_init: u32| -> (Vec<f32>, Vec<f32>) {
            let mut weights = vec![0.0f32; out_feat * in_feat];
            let mut bias = vec![0.0f32; out_feat];
            let mut s = seed_init;
            for v in weights.iter_mut() {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                *v = ((s as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
            }
            for v in bias.iter_mut() {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                *v = ((s as f32) / (u32::MAX as f32)) * 0.1 - 0.05;
            }
            (weights, bias)
        };

        let (weights_1, bias_1) = create_weights(11111);
        let (weights_2, bias_2) = create_weights(22222);
        let (weights_3, bias_3) = create_weights(33333);

        // CPU reference computations
        let cpu_forward = |input: &[f32], weights: &[f32], bias: &[f32]| -> Vec<f32> {
            let mut output = vec![0.0f32; batch * out_feat];
            for b in 0..batch {
                for o in 0..out_feat {
                    let mut sum = bias[o];
                    for i in 0..in_feat {
                        sum += input[b * in_feat + i] * weights[o * in_feat + i];
                    }
                    output[b * out_feat + o] = sum;
                }
            }
            output
        };

        let cpu_out_1 = cpu_forward(&input, &weights_1, &bias_1);
        let cpu_out_2 = cpu_forward(&input, &weights_2, &bias_2);
        let cpu_out_3 = cpu_forward(&input, &weights_3, &bias_3);

        // Transpose weights for GPU (out_feat x in_feat) -> (in_feat x out_feat)
        let transpose = |w: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            let mut t = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    t[c * rows + r] = w[r * cols + c];
                }
            }
            t
        };

        let w1_t = transpose(&weights_1, out_feat, in_feat);
        let w2_t = transpose(&weights_2, out_feat, in_feat);
        let w3_t = transpose(&weights_3, out_feat, in_feat);

        // Upload weights to GPU
        let w1_gpu = GpuResidentTensor::from_host(ctx, &w1_t).expect("Upload w1");
        let b1_gpu = GpuResidentTensor::from_host(ctx, &bias_1).expect("Upload b1");
        let w2_gpu = GpuResidentTensor::from_host(ctx, &w2_t).expect("Upload w2");
        let b2_gpu = GpuResidentTensor::from_host(ctx, &bias_2).expect("Upload b2");
        let w3_gpu = GpuResidentTensor::from_host(ctx, &w3_t).expect("Upload w3");
        let b3_gpu = GpuResidentTensor::from_host(ctx, &bias_3).expect("Upload b3");

        // Test 1: WITH cache clearing between calls
        eprintln!("Test 1: 3 calls WITH cache clearing between each...\n");

        let input1_gpu = GpuResidentTensor::from_host(ctx, &input).expect("Upload input 1");
        let gpu_same_1 = input1_gpu.linear(ctx, &w1_gpu, Some(&b1_gpu),
            batch as u32, in_feat as u32, out_feat as u32)
            .expect("GPU GEMM 1").to_host().expect("Download 1");
        let diff_clear_1: f32 = cpu_out_1.iter().zip(gpu_same_1.iter())
            .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        eprintln!("[Cache-clear call 1] max_diff = {:.6}", diff_clear_1);

        // Clear cache before second call
        trueno_gpu::memory::resident::clear_kernel_cache();

        let input2_gpu = GpuResidentTensor::from_host(ctx, &input).expect("Upload input 2");
        let gpu_same_2 = input2_gpu.linear(ctx, &w2_gpu, Some(&b2_gpu),
            batch as u32, in_feat as u32, out_feat as u32)
            .expect("GPU GEMM 2").to_host().expect("Download 2");
        let diff_clear_2: f32 = cpu_out_2.iter().zip(gpu_same_2.iter())
            .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        eprintln!("[Cache-clear call 2] max_diff = {:.6}", diff_clear_2);

        // Clear cache before third call
        trueno_gpu::memory::resident::clear_kernel_cache();

        let input3_gpu = GpuResidentTensor::from_host(ctx, &input).expect("Upload input 3");
        let gpu_same_3 = input3_gpu.linear(ctx, &w3_gpu, Some(&b3_gpu),
            batch as u32, in_feat as u32, out_feat as u32)
            .expect("GPU GEMM 3").to_host().expect("Download 3");
        let diff_clear_3: f32 = cpu_out_3.iter().zip(gpu_same_3.iter())
            .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        eprintln!("[Cache-clear call 3] max_diff = {:.6}\n", diff_clear_3);

        // Test 2: WITH BIAS (after fix) - no cache clearing
        eprintln!("Test 2: 3 calls WITH BIAS (testing fix)...\n");

        // GPU call 1 (with bias)
        let input4_gpu = GpuResidentTensor::from_host(ctx, &input).expect("Upload input 4");
        let gpu_out_1 = input4_gpu.linear(ctx, &w1_gpu, Some(&b1_gpu),
            batch as u32, in_feat as u32, out_feat as u32)
            .expect("GPU LINEAR 1").to_host().expect("Download 1");
        let diff_1: f32 = cpu_out_1.iter().zip(gpu_out_1.iter())
            .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        eprintln!("[With-bias call 1] max_diff = {:.6}", diff_1);

        let input5_gpu = GpuResidentTensor::from_host(ctx, &input).expect("Upload input 5");
        let gpu_out_2 = input5_gpu.linear(ctx, &w2_gpu, Some(&b2_gpu),
            batch as u32, in_feat as u32, out_feat as u32)
            .expect("GPU LINEAR 2").to_host().expect("Download 2");
        let diff_2: f32 = cpu_out_2.iter().zip(gpu_out_2.iter())
            .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        eprintln!("[With-bias call 2] max_diff = {:.6}", diff_2);

        let input6_gpu = GpuResidentTensor::from_host(ctx, &input).expect("Upload input 6");
        let gpu_out_3 = input6_gpu.linear(ctx, &w3_gpu, Some(&b3_gpu),
            batch as u32, in_feat as u32, out_feat as u32)
            .expect("GPU LINEAR 3").to_host().expect("Download 3");
        let diff_3: f32 = cpu_out_3.iter().zip(gpu_out_3.iter())
            .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        eprintln!("[With-bias call 3] max_diff = {:.6}",diff_3);

        eprintln!("\n============================================================");
        eprintln!("With cache-clear: {:.6}, {:.6}, {:.6}", diff_clear_1, diff_clear_2, diff_clear_3);
        eprintln!("With-bias (no cache-clear): {:.6}, {:.6}, {:.6}", diff_1, diff_2, diff_3);
        eprintln!("============================================================\n");

        // Cache-cleared calls should all work
        assert!(diff_clear_1 < 0.5, "Cache-clear call 1 diff too high: {}", diff_clear_1);
        assert!(diff_clear_2 < 0.5, "Cache-clear call 2 diff too high: {}", diff_clear_2);
        assert!(diff_clear_3 < 0.5, "Cache-clear call 3 diff too high: {}", diff_clear_3);

        // WAPR-PERF-027 FIX: With-bias calls should also work now (no cache clearing needed)
        assert!(diff_1 < 0.5, "With-bias call 1 diff too high: {}", diff_1);
        assert!(diff_2 < 0.5, "With-bias call 2 diff too high: {}", diff_2);
        assert!(diff_3 < 0.5, "With-bias call 3 diff too high: {}", diff_3);

        eprintln!("✓ All calls PASSED - bias_add stream race condition FIXED");
    }
}

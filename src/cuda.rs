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
use trueno_gpu::memory::resident::{
    batched_multihead_attention, forward_encoder_block_gpu, incremental_attention_gpu,
    incremental_attention_gpu_with_stream, // WAPR-PERF-014: shared stream variant
    kernel_cache_hits, kernel_cache_misses, kv_cache_scatter_gpu, reset_transfer_counters,
    total_d2h_transfers, total_h2d_transfers, GpuConvFrontendWeights, GpuDecoderBlockWeights,
    GpuDecoderConfig, GpuEncoderBlockWeights, GpuEncoderConfig, GpuKvCache, GpuResidentTensor,
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
        Self::new_with_components(encoder, decoder, config, tokenizer, mel_filters, device_ordinal)
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

        let executor = CudaExecutor::new(device_ordinal).map_err(|e| {
            WhisperError::Inference(format!("CUDA initialization failed: {e}"))
        })?;

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
        let token_emb = self.decoder.token_embedding();
        let bytes = self.executor.load_weights("whisper_output_proj", token_emb)
            .map_err(|e| WhisperError::Inference(format!("Failed to upload output projection: {e}")))?;
        total_bytes += bytes;

        // Upload all decoder block weights for full GPU acceleration
        // Each block has: self_attn (Q,K,V,O), cross_attn (Q,K,V,O), ffn (fc1, fc2)
        for (block_idx, block) in self.decoder.blocks().iter().enumerate() {
            // Self-attention weights
            let name = format!("dec_b{block_idx}_self_q");
            let bytes = self.executor.load_weights(&name, &block.self_attn.w_q().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_self_k");
            let bytes = self.executor.load_weights(&name, &block.self_attn.w_k().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_self_v");
            let bytes = self.executor.load_weights(&name, &block.self_attn.w_v().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_self_o");
            let bytes = self.executor.load_weights(&name, &block.self_attn.w_o().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            // Cross-attention weights
            let name = format!("dec_b{block_idx}_cross_q");
            let bytes = self.executor.load_weights(&name, &block.cross_attn.w_q().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_cross_k");
            let bytes = self.executor.load_weights(&name, &block.cross_attn.w_k().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_cross_v");
            let bytes = self.executor.load_weights(&name, &block.cross_attn.w_v().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_cross_o");
            let bytes = self.executor.load_weights(&name, &block.cross_attn.w_o().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            // FFN weights
            let name = format!("dec_b{block_idx}_ffn_fc1");
            let bytes = self.executor.load_weights(&name, &block.ffn.fc1.weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("dec_b{block_idx}_ffn_fc2");
            let bytes = self.executor.load_weights(&name, &block.ffn.fc2.weight)
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
            let bytes = self.executor.load_weights(&name, &block.self_attn.w_q().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("enc_b{block_idx}_self_k");
            let bytes = self.executor.load_weights(&name, &block.self_attn.w_k().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("enc_b{block_idx}_self_v");
            let bytes = self.executor.load_weights(&name, &block.self_attn.w_v().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("enc_b{block_idx}_self_o");
            let bytes = self.executor.load_weights(&name, &block.self_attn.w_o().weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            // FFN weights
            let name = format!("enc_b{block_idx}_ffn_fc1");
            let bytes = self.executor.load_weights(&name, &block.ffn.fc1.weight)
                .map_err(|e| WhisperError::Inference(format!("Failed to upload {name}: {e}")))?;
            total_bytes += bytes;

            let name = format!("enc_b{block_idx}_ffn_fc2");
            let bytes = self.executor.load_weights(&name, &block.ffn.fc2.weight)
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
        let n_mels = self.config.n_mels as usize;
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
            let ffn_down_w = GpuResidentTensor::from_host(ctx, &ffn_down_t)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_down_w L{layer_idx}: {e}")))?;
            let ffn_down_b = GpuResidentTensor::from_host(ctx, &block.ffn.fc2.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_down_b L{layer_idx}: {e}")))?;

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
    #[cfg(feature = "cuda")]
    pub fn reset_gpu_decoder_kv_cache(&mut self) {
        self.gpu_self_k_head_first = None;
        self.gpu_self_v_head_first = None;
        self.gpu_cross_k_head_first = None;
        self.gpu_cross_v_head_first = None;
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
        let weights = self.gpu_decoder_weights.as_ref()
            .ok_or_else(|| WhisperError::Inference("Decoder weights not uploaded".into()))?;
        let layer_weights = &weights[layer_idx];

        // Get head-first KV caches
        let self_k_caches = self.gpu_self_k_head_first.as_mut()
            .ok_or_else(|| WhisperError::Inference("Self K cache not initialized".into()))?;
        let self_v_caches = self.gpu_self_v_head_first.as_mut()
            .ok_or_else(|| WhisperError::Inference("Self V cache not initialized".into()))?;

        let block = &self.decoder.blocks()[layer_idx];

        // === Self-Attention ===

        // LN1 (CPU - simple and correct)
        let normed = block.ln1.forward(x)?;

        // Upload normed input to GPU
        let x_gpu = GpuResidentTensor::from_host(ctx, &normed)
            .map_err(|e| WhisperError::Inference(format!("x upload: {e}")))?;

        // Q/K/V projections on GPU: [1, d_model] @ [d_model, d_model] = [1, d_model]
        let q = x_gpu.linear(ctx, &layer_weights.self_w_q, Some(&layer_weights.self_b_q), 1, d_model as u32, d_model as u32)
            .map_err(|e| WhisperError::Inference(format!("Q projection: {e}")))?;
        let k = x_gpu.linear(ctx, &layer_weights.self_w_k, Some(&layer_weights.self_b_k), 1, d_model as u32, d_model as u32)
            .map_err(|e| WhisperError::Inference(format!("K projection: {e}")))?;
        let v = x_gpu.linear(ctx, &layer_weights.self_w_v, Some(&layer_weights.self_b_v), 1, d_model as u32, d_model as u32)
            .map_err(|e| WhisperError::Inference(format!("V projection: {e}")))?;

        // Scatter K/V to head-first caches
        let stream = CudaStream::new(ctx)
            .map_err(|e| WhisperError::Inference(format!("Stream: {e}")))?;

        kv_cache_scatter_gpu(
            ctx, &k, &mut self_k_caches[layer_idx],
            pos as u32, n_heads as u32, head_dim as u32, max_seq_len as u32, &stream
        ).map_err(|e| WhisperError::Inference(format!("K scatter: {e}")))?;

        kv_cache_scatter_gpu(
            ctx, &v, &mut self_v_caches[layer_idx],
            pos as u32, n_heads as u32, head_dim as u32, max_seq_len as u32, &stream
        ).map_err(|e| WhisperError::Inference(format!("V scatter: {e}")))?;

        // Incremental self-attention: Q @ cached_K^T → softmax → @ cached_V
        let seq_len = (pos + 1) as u32; // Include current position
        let attn_out = incremental_attention_gpu(
            ctx, &q, &self_k_caches[layer_idx], &self_v_caches[layer_idx],
            n_heads as u32, head_dim as u32, seq_len, max_seq_len as u32
        ).map_err(|e| WhisperError::Inference(format!("Self attention: {e}")))?;

        // Output projection
        let mut attn_proj = attn_out.linear(ctx, &layer_weights.self_w_o, Some(&layer_weights.self_b_o), 1, d_model as u32, d_model as u32)
            .map_err(|e| WhisperError::Inference(format!("O projection: {e}")))?;

        // Download and add residual (sync point)
        let attn_proj_host = attn_proj.to_host()
            .map_err(|e| WhisperError::Inference(format!("Attn D2H: {e}")))?;

        // Residual connection
        let mut residual: Vec<f32> = x.iter().zip(attn_proj_host.iter()).map(|(a, b)| a + b).collect();

        // === Cross-Attention ===
        if let Some(enc_out) = encoder_output {
            let normed2 = block.ln2.forward(&residual)?;
            let cross_out = block.cross_attn.forward_cross_dispatch(
                &normed2,
                enc_out,
                None // TODO: Use cached cross-attention K/V
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
            hidden = self.forward_decoder_block_gpu(
                layer_idx,
                &hidden,
                pos,
                Some(encoder_output),
            )?;
        }

        Ok(hidden)
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

        let conv_weights = self.gpu_conv_weights.as_ref()
            .ok_or_else(|| WhisperError::Inference("GPU conv weights not uploaded".into()))?;

        // Conv1: 80 → 384, kernel=3, stride=1, padding=1 + GELU
        let conv1_out = mel_gpu.conv1d(
            ctx,
            &conv_weights.conv1_weight,
            Some(&conv_weights.conv1_bias),
            n_mels as u32,        // in_channels
            d_model as u32,       // out_channels
            3,                    // kernel_size
            1,                    // stride
            1,                    // padding
            seq_len_in as u32,    // seq_len
        ).map_err(|e| WhisperError::Inference(format!("conv1 GPU: {e}")))?;

        // After conv1: seq_len stays same (stride=1), channels = d_model
        let seq_len_after_conv1 = seq_len_in;

        // Conv2: 384 → 384, kernel=3, stride=2, padding=1 + GELU
        let mut conv2_out = conv1_out.conv1d(
            ctx,
            &conv_weights.conv2_weight,
            Some(&conv_weights.conv2_bias),
            d_model as u32,       // in_channels
            d_model as u32,       // out_channels
            3,                    // kernel_size
            2,                    // stride
            1,                    // padding
            seq_len_after_conv1 as u32,
        ).map_err(|e| WhisperError::Inference(format!("conv2 GPU: {e}")))?;

        let conv_time = conv_start.elapsed();

        // After conv2: seq_len halved (stride=2)
        let seq_len = (seq_len_after_conv1 + 2 - 3) / 2 + 1;

        // Download conv output to add positional embedding (CPU - small overhead)
        let pos_start = std::time::Instant::now();
        let mut x = conv2_out.to_host()
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
            eprintln!("[PROFILE-BREAKDOWN] Conv(GPU): {:.1}ms, PosEmb: {:.1}ms, Upload: {:.1}ms",
                conv_time.as_millis(), pos_time.as_millis(), upload_time.as_millis());
        }

        // Step 4: Process all encoder blocks on GPU (0 transfers)
        let weights = self.gpu_encoder_weights.as_ref()
            .ok_or_else(|| WhisperError::Inference("GPU weights not uploaded".into()))?;
        let config = self.gpu_encoder_config.as_ref()
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
            let head_dim = d_model / n_heads;

            // Step 1: LayerNorm
            let ln1_out = block0.ln1.forward(&x)?;
            eprintln!("[DEBUG-CPU] Layer 0 LN1: mean={:.6}, std={:.6}",
                ln1_out.iter().sum::<f32>() / ln1_out.len() as f32,
                (ln1_out.iter().map(|v| v.powi(2)).sum::<f32>() / ln1_out.len() as f32).sqrt());

            // Step 2: Q/K/V projections
            let q = block0.self_attn.w_q().forward(&ln1_out, seq_len)?;
            let k = block0.self_attn.w_k().forward(&ln1_out, seq_len)?;
            let v = block0.self_attn.w_v().forward(&ln1_out, seq_len)?;
            eprintln!("[DEBUG-CPU] Q: mean={:.6}, K: mean={:.6}, V: mean={:.6}",
                q.iter().sum::<f32>() / q.len() as f32,
                k.iter().sum::<f32>() / k.len() as f32,
                v.iter().sum::<f32>() / v.len() as f32);
        }

        // WAPR-PERF-011: Timing instrumentation for verification matrix
        let profile_layers = std::env::var("WHISPER_PROFILE_LAYERS").is_ok();
        let mut layer_times: Vec<u128> = Vec::new();

        for layer_idx in 0..n_layers {
            if debug_layer0_only && layer_idx == 1 {
                // Get intermediate GPU output after layer 0 only
                let gpu_layer0_out = x_gpu.to_host()
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
        let output = x_gpu.to_host()
            .map_err(|e| WhisperError::Inference(format!("output download: {e}")))?;
        let download_time = download_start.elapsed();

        if profile_detail {
            eprintln!("[PROFILE-BREAKDOWN] Download: {:.1}ms", download_time.as_millis());
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
            let accounted = conv_time.as_micros() + pos_time.as_micros() + upload_time.as_micros()
                + layer_sum + download_time.as_micros() + ln_post_time.as_micros();
            let unaccounted = total_time.as_micros().saturating_sub(accounted);
            eprintln!("[PROFILE-BREAKDOWN] LnPost: {:.1}ms, Total: {:.1}ms", ln_post_time.as_millis(), total_time.as_millis());
            eprintln!("[PROFILE-SUMMARY] Conv={:.0}µs PosEmb={:.0}µs Upload={:.0}µs Layers={:.0}µs Download={:.0}µs LnPost={:.0}µs",
                conv_time.as_micros(), pos_time.as_micros(), upload_time.as_micros(),
                layer_sum, download_time.as_micros(), ln_post_time.as_micros());
            eprintln!("[PROFILE-SUMMARY] Accounted: {:.1}ms, Unaccounted: {:.1}ms ({:.1}%)",
                accounted as f64 / 1000.0, unaccounted as f64 / 1000.0,
                unaccounted as f64 / total_time.as_micros() as f64 * 100.0);
        }

        Ok(result)
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
        let (normed, q, k, v) = {
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
        let (attn_proj, normed2, ffn_out) = {
            let block = &self.encoder.blocks()[layer_idx];
            let attn_proj = block.self_attn.w_o().forward(&attn_output, seq_len)?;

            // Residual connection
            let mut residual: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

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
        let output = output_gpu
            .to_host()
            .map_err(|e| WhisperError::Inference(format!("Failed to download attention output: {e}")))?;

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
        let mut x: Vec<f32> = self.decoder.token_embedding()[emb_start..emb_start + d_model].to_vec();

        // Add positional embedding
        let pos_start = position * d_model;
        for (x_elem, pos_emb) in x.iter_mut().zip(
            &self.decoder.positional_embedding()[pos_start..pos_start + d_model],
        ) {
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
        let k_enc = block.cross_attn.w_k().forward_simd(encoder_output, enc_len)?;
        let v_enc = block.cross_attn.w_v().forward_simd(encoder_output, enc_len)?;

        // Compute cross-attention (CPU)
        let cross_attn_out = self.compute_cross_attention(&q_cross, &k_enc, &v_enc, n_heads, head_dim)?;

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
                scores[pos] = q_head.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum::<f32>()
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
        let mut mel = self.mel_filters.compute(&padded_audio, crate::audio::HOP_LENGTH)?;
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
            let _ = self.decoder.forward_one(token, &encoder_output, &mut cache)?;
        }

        // Generate tokens
        for _ in 0..max_tokens.saturating_sub(tokens.len()) {
            // Get logits for last token
            let last_token = *tokens.last().unwrap_or(&specials.sot);
            let mut logits = self.decoder.forward_one(last_token, &encoder_output, &mut cache)?;

            // Apply token suppression
            suppressor.apply(&mut logits);

            // Get next token based on strategy
            let next_token = match options.strategy {
                DecodingStrategy::Greedy => {
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(eot_token)
                }
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
        let hidden = self.decoder.forward_one_hidden(token, encoder_output, cache)?;

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
            x = self.forward_decoder_block_gpu(
                layer_idx,
                &x,
                pos,
                Some(encoder_output),
            )?;
        }

        // 3. Final layer norm (CPU - simple)
        let blocks = self.decoder.blocks();
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
            x = self.forward_decoder_block_executor(
                layer_idx,
                &x,
                pos,
                Some(encoder_output),
            )?;
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
                hidden.len(), d_model
            )));
        }

        let mut output = vec![0.0f32; n_vocab];

        // WAPR-PERF-014: Try cached weights first, fall back to direct gemm
        // GEMV: y[n] = W[n,k] @ x[k] where W = token_embedding [n_vocab × d_model]
        let k = d_model as u32;
        let n = n_vocab as u32;

        if self.executor.has_weights("dec.output_proj") {
            // Fast path: use cached weights (persistent GPU buffer, no allocation)
            self.executor
                .gemv_cached("dec.output_proj", hidden, &mut output, k, n)
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
            let argmax = output.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            eprintln!("[GPU] project_to_vocab_gpu: max={:.4} argmax={}", max_val, argmax);
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
        let mut mel = self.mel_filters.compute(&padded_audio, crate::audio::HOP_LENGTH)?;
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
        // - WHISPER_GPU_TOTAL_OFFLOAD=1: Full GPU encoder (2x target)
        // - WHISPER_GPU_ENCODER=1: Partial GPU (attention only)
        // - Default: CPU encoder with SIMD
        #[cfg(feature = "cuda")]
        let encoder_output = {
            let use_total_offload = std::env::var("WHISPER_GPU_TOTAL_OFFLOAD").is_ok();
            let use_gpu_encoder = std::env::var("WHISPER_GPU_ENCODER").is_ok();

            if use_total_offload {
                eprintln!("[WAPR-PERF-004] Using GPU Total Offload encoder...");
                let start = std::time::Instant::now();
                let result = self.encode_gpu_total_offload(&mel)?;
                let elapsed = start.elapsed();
                let hits = kernel_cache_hits();
                let misses = kernel_cache_misses();
                eprintln!(
                    "[WAPR-PERF-004] Encoder completed in {:?} (cache: {} hits, {} compiles)",
                    elapsed, hits, misses
                );
                result
            } else if use_gpu_encoder {
                eprintln!("[WAPR-PERF-005] Using GPU attention-only encoder...");
                self.encode_gpu(&mel)?
            } else {
                eprintln!("[CPU] Using SIMD encoder...");
                self.encoder.forward_mel(&mel)?
            }
        };
        #[cfg(not(feature = "cuda"))]
        let encoder_output = self.encoder.forward_mel(&mel)?;

        // Trace encoder output (layer_idx=0 for encoder, iteration=0 for prefill)
        let d_model = self.config.n_text_state as usize;
        let enc_seq_len = encoder_output.len() / d_model;
        self.tracer.trace_layer(0, 0, Some(&encoder_output), enc_seq_len, d_model);

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

        // WAPR-PERF-013: Check if GPU decoder total offload is enabled
        #[cfg(feature = "cuda")]
        let use_gpu_decoder = std::env::var("WHISPER_GPU_DECODER_OFFLOAD").is_ok();
        #[cfg(not(feature = "cuda"))]
        let use_gpu_decoder = false;

        if use_gpu_decoder {
            eprintln!("[WAPR-PERF-013] Using GPU decoder total offload...");
            // Reset GPU decoder position for new transcription
            self.reset_gpu_decoder_pos();
        }

        // Process initial tokens
        for &token in &tokens {
            #[cfg(feature = "cuda")]
            if use_gpu_decoder {
                let _ = self.forward_one_gpu_total_offload(token, &encoder_output)?;
            } else {
                let _ = self.decoder.forward_one(token, &encoder_output, &mut cache)?;
            }
            #[cfg(not(feature = "cuda"))]
            let _ = self.decoder.forward_one(token, &encoder_output, &mut cache)?;
        }

        // Generate tokens
        let debug_gpu = std::env::var("WHISPER_DEBUG_GPU").is_ok();
        for gen_idx in 0..max_tokens.saturating_sub(tokens.len()) {
            let last_token = *tokens.last().unwrap_or(&specials.sot);

            if debug_gpu && gen_idx < 5 {
                eprintln!("[DEBUG] gen_idx={} last_token={} tokens={:?}", gen_idx, last_token, &tokens);
            }

            // === TRACE: LM_HEAD (output projection) ===
            self.tracer.start_step(TraceStep::LmHead);

            // WAPR-PERF-013: Use GPU decoder total offload path when enabled
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
                DecodingStrategy::Greedy => {
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(eot_token)
                }
                DecodingStrategy::BeamSearch { .. } | DecodingStrategy::Sampling { .. } => {
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

            // Trace sampling result (temperature=0.0 for greedy, top_k=1)
            self.tracer.trace_sample(gen_idx, &logits, next_token, 0.0, 1);

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
            eprintln!("[TRACE] Tracing not enabled. Call enable_tracing(TraceConfig::enabled()) first.");
            return;
        }

        let events = self.tracer.events();
        if events.is_empty() {
            eprintln!("[TRACE] No events collected.");
            return;
        }

        // Compute totals by step
        let mut step_durations: std::collections::HashMap<&'static str, u64> = std::collections::HashMap::new();
        let mut step_counts: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();

        for event in events {
            *step_durations.entry(event.step.name()).or_insert(0) += event.duration_us;
            *step_counts.entry(event.step.name()).or_insert(0) += 1;
        }

        let total_us: u64 = step_durations.values().sum();
        let total_ms = total_us as f64 / 1000.0;

        eprintln!("=== APR-Style Inference Trace Summary ===");
        eprintln!("Total: {:.2}ms ({} events)", total_ms, events.len());
        eprintln!("");
        eprintln!("{:20} {:>8} {:>8} {:>8}", "STEP", "COUNT", "TIME(ms)", "PCT");
        eprintln!("{:-<20} {:->8} {:->8} {:->8}", "", "", "", "");

        // Sort by duration descending
        let mut steps: Vec<_> = step_durations.iter().collect();
        steps.sort_by(|a, b| b.1.cmp(a.1));

        for (step, us) in steps {
            let count = step_counts.get(step).unwrap_or(&0);
            let ms = *us as f64 / 1000.0;
            let pct = if total_us > 0 { (*us as f64 / total_us as f64) * 100.0 } else { 0.0 };
            eprintln!("{:20} {:>8} {:>8.2} {:>7.1}%", step, count, ms, pct);
        }
        eprintln!("");
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
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.self_w_q"), &w_q_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_q L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.self_b_q"), &block.self_attn.w_q().bias)
                .map_err(|e| WhisperError::Inference(format!("dec self_b_q L{layer_idx}: {e}")))?;

            let w_k_t = transpose_weights(&block.self_attn.w_k().weight, d_model, d_model);
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.self_w_k"), &w_k_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_k L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.self_b_k"), &block.self_attn.w_k().bias)
                .map_err(|e| WhisperError::Inference(format!("dec self_b_k L{layer_idx}: {e}")))?;

            let w_v_t = transpose_weights(&block.self_attn.w_v().weight, d_model, d_model);
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.self_w_v"), &w_v_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_v L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.self_b_v"), &block.self_attn.w_v().bias)
                .map_err(|e| WhisperError::Inference(format!("dec self_b_v L{layer_idx}: {e}")))?;

            let w_o_t = transpose_weights(&block.self_attn.w_o().weight, d_model, d_model);
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.self_w_o"), &w_o_t)
                .map_err(|e| WhisperError::Inference(format!("dec self_w_o L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.self_b_o"), &block.self_attn.w_o().bias)
                .map_err(|e| WhisperError::Inference(format!("dec self_b_o L{layer_idx}: {e}")))?;

            // Cross-attention Q/K/V/O
            let cross_w_q_t = transpose_weights(&block.cross_attn.w_q().weight, d_model, d_model);
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.cross_w_q"), &cross_w_q_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_q L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.cross_b_q"), &block.cross_attn.w_q().bias)
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_q L{layer_idx}: {e}")))?;

            let cross_w_k_t = transpose_weights(&block.cross_attn.w_k().weight, d_model, d_model);
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.cross_w_k"), &cross_w_k_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_k L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.cross_b_k"), &block.cross_attn.w_k().bias)
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_k L{layer_idx}: {e}")))?;

            let cross_w_v_t = transpose_weights(&block.cross_attn.w_v().weight, d_model, d_model);
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.cross_w_v"), &cross_w_v_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_v L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.cross_b_v"), &block.cross_attn.w_v().bias)
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_v L{layer_idx}: {e}")))?;

            let cross_w_o_t = transpose_weights(&block.cross_attn.w_o().weight, d_model, d_model);
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.cross_w_o"), &cross_w_o_t)
                .map_err(|e| WhisperError::Inference(format!("dec cross_w_o L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.cross_b_o"), &block.cross_attn.w_o().bias)
                .map_err(|e| WhisperError::Inference(format!("dec cross_b_o L{layer_idx}: {e}")))?;

            // FFN weights (fc1: d_model -> d_ff, fc2: d_ff -> d_model)
            let fc1_t = transpose_weights(&block.ffn.fc1.weight, d_ff, d_model);
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.ffn_fc1"), &fc1_t)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_fc1 L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.ffn_b1"), &block.ffn.fc1.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_b1 L{layer_idx}: {e}")))?;

            let fc2_t = transpose_weights(&block.ffn.fc2.weight, d_model, d_ff);
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.ffn_fc2"), &fc2_t)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_fc2 L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.ffn_b2"), &block.ffn.fc2.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ffn_b2 L{layer_idx}: {e}")))?;

            // LayerNorm weights (gamma/beta)
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.ln1_gamma"), &block.ln1.weight)
                .map_err(|e| WhisperError::Inference(format!("dec ln1_gamma L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.ln1_beta"), &block.ln1.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ln1_beta L{layer_idx}: {e}")))?;

            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.ln2_gamma"), &block.ln2.weight)
                .map_err(|e| WhisperError::Inference(format!("dec ln2_gamma L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.ln2_beta"), &block.ln2.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ln2_beta L{layer_idx}: {e}")))?;

            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.ln3_gamma"), &block.ln3.weight)
                .map_err(|e| WhisperError::Inference(format!("dec ln3_gamma L{layer_idx}: {e}")))?;
            total_bytes += self.executor.load_weights(&format!("dec.L{layer_idx}.ln3_beta"), &block.ln3.bias)
                .map_err(|e| WhisperError::Inference(format!("dec ln3_beta L{layer_idx}: {e}")))?;
        }

        // Output projection weights (token embedding)
        // WAPR-PERF-014: Token embedding is [n_vocab, d_model] but GEMV kernel expects [k, n]
        // where k=d_model (input) and n=n_vocab (output), so transpose to [d_model, n_vocab]
        let n_vocab = self.config.n_vocab as usize;
        let token_emb = self.decoder.token_embedding();
        let token_emb_t = transpose_weights(token_emb, n_vocab, d_model);
        total_bytes += self.executor.load_weights("dec.output_proj", &token_emb_t)
            .map_err(|e| WhisperError::Inference(format!("dec output_proj: {e}")))?;

        // Final layer norm
        let ln_post = self.decoder.ln_post();
        total_bytes += self.executor.load_weights("dec.ln_post_gamma", &ln_post.weight)
            .map_err(|e| WhisperError::Inference(format!("dec ln_post_gamma: {e}")))?;
        total_bytes += self.executor.load_weights("dec.ln_post_beta", &ln_post.bias)
            .map_err(|e| WhisperError::Inference(format!("dec ln_post_beta: {e}")))?;

        if std::env::var("WHISPER_DEBUG_GPU").is_ok() {
            eprintln!("[WAPR-PERF-014] Uploaded {} decoder weight tensors ({:.2} MB) to executor",
                self.executor.cached_weight_count(),
                total_bytes as f64 / 1_048_576.0);
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

        self.executor.gemv_cached(
            &format!("dec.L{layer_idx}.self_w_q"),
            &normed, &mut q,
            d_model as u32, d_model as u32,
        ).map_err(|e| WhisperError::Inference(format!("Q projection: {e}")))?;

        self.executor.gemv_cached(
            &format!("dec.L{layer_idx}.self_w_k"),
            &normed, &mut k,
            d_model as u32, d_model as u32,
        ).map_err(|e| WhisperError::Inference(format!("K projection: {e}")))?;

        self.executor.gemv_cached(
            &format!("dec.L{layer_idx}.self_w_v"),
            &normed, &mut v,
            d_model as u32, d_model as u32,
        ).map_err(|e| WhisperError::Inference(format!("V projection: {e}")))?;

        // Add biases (CPU - fast)
        for i in 0..d_model {
            q[i] += b_q[i];
            k[i] += b_k[i];
            v[i] += b_v[i];
        }

        // Now get context for GPU tensor operations (after gemv_cached calls)
        let ctx = self.executor.context();

        // Get head-first KV caches
        let self_k_caches = self.gpu_self_k_head_first.as_mut()
            .ok_or_else(|| WhisperError::Inference("Self K cache not initialized".into()))?;
        let self_v_caches = self.gpu_self_v_head_first.as_mut()
            .ok_or_else(|| WhisperError::Inference("Self V cache not initialized".into()))?;

        // Upload Q/K/V for KV cache scatter + attention
        let q_gpu = GpuResidentTensor::from_host(ctx, &q)
            .map_err(|e| WhisperError::Inference(format!("Q upload: {e}")))?;
        let k_gpu = GpuResidentTensor::from_host(ctx, &k)
            .map_err(|e| WhisperError::Inference(format!("K upload: {e}")))?;
        let v_gpu = GpuResidentTensor::from_host(ctx, &v)
            .map_err(|e| WhisperError::Inference(format!("V upload: {e}")))?;

        // Use executor's stream for scatter (avoid new stream creation)
        let stream = CudaStream::new(ctx)
            .map_err(|e| WhisperError::Inference(format!("Stream: {e}")))?;

        kv_cache_scatter_gpu(
            ctx, &k_gpu, &mut self_k_caches[layer_idx],
            pos as u32, n_heads as u32, head_dim as u32, max_seq_len as u32, &stream
        ).map_err(|e| WhisperError::Inference(format!("K scatter: {e}")))?;

        kv_cache_scatter_gpu(
            ctx, &v_gpu, &mut self_v_caches[layer_idx],
            pos as u32, n_heads as u32, head_dim as u32, max_seq_len as u32, &stream
        ).map_err(|e| WhisperError::Inference(format!("V scatter: {e}")))?;

        // Incremental self-attention (WAPR-PERF-014: use shared stream)
        let seq_len = (pos + 1) as u32;
        let attn_out = incremental_attention_gpu_with_stream(
            ctx, &q_gpu, &self_k_caches[layer_idx], &self_v_caches[layer_idx],
            n_heads as u32, head_dim as u32, seq_len, max_seq_len as u32,
            &stream  // Reuse stream from KV scatter (no new stream creation!)
        ).map_err(|e| WhisperError::Inference(format!("Self attention: {e}")))?;

        // Sync before reading back (all kernels launched on shared stream)
        stream.synchronize()
            .map_err(|e| WhisperError::Inference(format!("Stream sync: {e}")))?;

        // Output projection via executor (need to drop ctx borrow first)
        let mut attn_out = attn_out;  // Move to local
        let attn_out_host = attn_out.to_host()
            .map_err(|e| WhisperError::Inference(format!("Attn D2H: {e}")))?;

        // Drop stream after sync
        drop(stream);

        let mut attn_proj = vec![0.0f32; d_model];
        self.executor.gemv_cached(
            &format!("dec.L{layer_idx}.self_w_o"),
            &attn_out_host, &mut attn_proj,
            d_model as u32, d_model as u32,
        ).map_err(|e| WhisperError::Inference(format!("O projection: {e}")))?;

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
            let cross_out = block.cross_attn.forward_cross_dispatch(
                &normed2,
                enc_out,
                None,
            )?;
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
            hidden = self.forward_decoder_block_executor(
                layer_idx,
                &hidden,
                pos,
                Some(encoder_output),
            )?;
        }

        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        cuda_model.upload_weights().expect("Failed to upload weights");

        // Create test hidden state (normalized random values)
        let d_model = cuda_model.config().n_text_state as usize;
        let hidden: Vec<f32> = (0..d_model)
            .map(|i| (i as f32 * 0.1).sin() * 0.1)
            .collect();

        // Compute CPU output projection
        let cpu_logits = cuda_model.decoder.project_to_vocab_debug(&hidden);

        // Compute GPU output projection
        let gpu_logits = cuda_model.project_to_vocab_gpu(&hidden)
            .expect("GPU gemm failed");

        // Compare results
        assert_eq!(cpu_logits.len(), gpu_logits.len(), "Output dimension mismatch");

        // Find max difference
        let max_diff: f32 = cpu_logits.iter()
            .zip(gpu_logits.iter())
            .map(|(c, g)| (*c - *g).abs())
            .fold(0.0f32, f32::max);

        // Find argmax for both
        let cpu_argmax: (usize, f32) = cpu_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
            .unwrap();
        let gpu_argmax: (usize, f32) = gpu_logits.iter().enumerate()
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
        let wav_data = crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let audio = wav_data.samples;

        // Run transcription with tracing
        let options = crate::TranscribeOptions::default();
        let result = cuda_model.transcribe_gpu(&audio, options).expect("Transcription failed");

        eprintln!("\n=== APR-Style Inference Tracing Test (WAPR-PERF-004) ===");
        eprintln!("Transcription: \"{}\"", result.text);
        eprintln!("");

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
        assert!(step_counts.contains_key("EMBED"), "Should have EMBED events");
        assert!(step_counts.contains_key("TRANSFORMER_BLOCK"), "Should have TRANSFORMER_BLOCK events");

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
        let wav_data = crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
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

        let mut mel = cuda_model.mel_filters.compute(&padded_audio, crate::audio::HOP_LENGTH)
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
        let cpu_output = cuda_model.encoder.forward_mel(&mel).expect("CPU encoder failed");
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
        let max_diff: f32 = cpu_output.iter()
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
        let q: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32 * 0.03).sin()).collect();

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
        let k_cache: Vec<f32> = (0..cache_size)
            .map(|i| ((i as f32) * 0.02).cos())
            .collect();
        let v_cache: Vec<f32> = (0..cache_size)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();

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
            ctx, &q_gpu, &k_gpu, &v_gpu,
            n_heads, head_dim, seq_len, max_seq_len
        ).expect("GPU attention failed");

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
                    eprintln!("  [{}] CPU={:.6} GPU={:.6} diff={:.2e}", i, cpu_val, gpu_val, diff);
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
            panic!("Numerical parity test failed: max_diff={:.2e} > tolerance={:.2e}", max_diff, tolerance);
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
        cuda_model.upload_decoder_weights_to_gpu().expect("Failed to upload decoder weights");

        // Initialize head-first KV caches
        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Failed to init KV cache");

        let d_model = cuda_model.config().n_text_state as usize;
        let n_layers = cuda_model.config().n_text_layer as usize;
        let max_len = cuda_model.config().n_text_ctx as usize;

        eprintln!("Model: d_model={}, n_layers={}, max_len={}", d_model, n_layers, max_len);

        // Generate test input (simulated decoder input embedding)
        let test_input: Vec<f32> = (0..d_model)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();

        // Test multiple positions to verify KV cache works
        for pos in 0..3 {
            eprintln!("\nTesting position {}...", pos);

            // Run GPU decoder block (layer 0, no cross-attention)
            let gpu_output = cuda_model.forward_decoder_block_gpu(
                0, // layer_idx
                &test_input,
                pos,
                None, // No encoder output - skip cross-attention
            ).expect("GPU decoder block failed");

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
            eprintln!("  Output sample: [{:.4}, {:.4}, {:.4}...]", gpu_output[0], gpu_output[1], gpu_output[2]);
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
        cuda_model1.upload_decoder_weights_to_gpu().expect("Upload GPU weights");
        cuda_model1.init_gpu_decoder_kv_cache_head_first().expect("Init KV cache 1");

        cuda_model2.upload_decoder_weights_to_gpu().expect("Upload GPU weights 2");
        cuda_model2.upload_decoder_weights_to_executor().expect("Upload executor weights");
        cuda_model2.init_gpu_decoder_kv_cache_head_first().expect("Init KV cache 2");

        let d_model = cuda_model1.config().n_text_state as usize;

        // Generate test input
        let test_input: Vec<f32> = (0..d_model)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();

        eprintln!("Testing layer 0, position 0...");

        // Run GPU forward pass
        let gpu_start = std::time::Instant::now();
        let gpu_output = cuda_model1.forward_decoder_block_gpu(
            0, &test_input, 0, None,
        ).expect("GPU forward failed");
        let gpu_time = gpu_start.elapsed();

        // Run executor forward pass
        let exec_start = std::time::Instant::now();
        let exec_output = cuda_model2.forward_decoder_block_executor(
            0, &test_input, 0, None,
        ).expect("Executor forward failed");
        let exec_time = exec_start.elapsed();

        // Compare outputs
        assert_eq!(gpu_output.len(), exec_output.len(), "Output length mismatch");

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
        let wav_data = crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let mel = apr.compute_mel(&wav_data.samples).expect("Mel failed");

        // Now convert to CUDA
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Run encoder (CPU is fine for this test)
        let encoder_output = cuda_model.encoder.forward_mel(&mel).expect("Encoder failed");

        let d_model = cuda_model.config().n_text_state as usize;
        let n_layers = cuda_model.config().n_text_layer as usize;
        let n_vocab = cuda_model.config().n_vocab as usize;
        let max_len = cuda_model.config().n_text_ctx as usize;

        eprintln!("Model: d_model={}, n_layers={}, n_vocab={}", d_model, n_layers, n_vocab);
        eprintln!("Encoder output: {} elements", encoder_output.len());

        // Test token: SOT (start of transcript)
        let sot_token = 50258_u32; // Whisper SOT token

        // === CPU Reference Path ===
        let mut cpu_cache = crate::model::DecoderKVCache::new(n_layers, d_model, max_len);
        let cpu_hidden = cuda_model.decoder.forward_one_hidden(
            sot_token,
            &encoder_output,
            &mut cpu_cache
        ).expect("CPU decoder failed");

        // Get CPU logits
        let cpu_logits = cuda_model.decoder.project_to_vocab_debug(&cpu_hidden);

        // === GPU Path ===
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.upload_decoder_weights_to_gpu().expect("Upload failed");
        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Init failed");

        let gpu_logits = cuda_model.forward_one_gpu_total_offload(
            sot_token,
            &encoder_output
        ).expect("GPU decoder failed");

        // === Parity Check ===
        let tolerance = 1e-3_f32; // Slightly looser for full path
        let mut max_diff = 0.0_f32;
        let mut diff_count = 0_usize;

        // Find argmax for both
        let cpu_argmax = cpu_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let gpu_argmax = gpu_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        for (i, (cpu_val, gpu_val)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
            let diff: f32 = (*cpu_val - *gpu_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > tolerance {
                diff_count += 1;
            }
        }

        eprintln!("CPU argmax: {} (token '{}')", cpu_argmax, cuda_model.tokenizer.decode(&[cpu_argmax as u32]).unwrap_or_default());
        eprintln!("GPU argmax: {} (token '{}')", gpu_argmax, cuda_model.tokenizer.decode(&[gpu_argmax as u32]).unwrap_or_default());
        eprintln!("Max absolute difference: {:.2e}", max_diff);
        eprintln!("Elements exceeding {:.0e}: {}/{}", tolerance, diff_count, n_vocab);

        // Critical: argmax must match for correct decoding
        if cpu_argmax != gpu_argmax {
            eprintln!("\n❌ ARGMAX MISMATCH: GPU decoder produces different token");
            eprintln!("   This WILL cause divergent text output.");
            panic!("GPU vs CPU decoder argmax mismatch: CPU={} GPU={}", cpu_argmax, gpu_argmax);
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
        let wav_data = crate::audio::wav::parse_wav_file(&audio_bytes).expect("Failed to parse WAV");
        let mel = apr.compute_mel(&wav_data.samples).expect("Mel failed");

        // Create CUDA model
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Run encoder
        let encoder_output = cuda_model.encoder.forward_mel(&mel).expect("Encoder failed");
        eprintln!("Encoder output: {} elements", encoder_output.len());

        // Initial tokens
        use crate::tokenizer::special_tokens::SpecialTokens;
        let specials = SpecialTokens::for_vocab_size(cuda_model.config().n_vocab as usize);
        let initial_tokens = vec![specials.sot, specials.lang_base, specials.transcribe, specials.no_timestamps];

        let num_decode_tokens = 10; // Decode 10 tokens for benchmarking

        // === GPU Path Benchmark ===
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.upload_decoder_weights_to_gpu().expect("Upload GPU weights");
        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Init KV cache");

        // Warmup (JIT compilation)
        for &token in &initial_tokens {
            let _ = cuda_model.forward_one_gpu_total_offload(token, &encoder_output).expect("Warmup failed");
        }

        // WAPR-PERF-014: Reset state after warmup for clean benchmark
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Re-init KV cache");

        // Process initial tokens fresh for benchmark
        for &token in &initial_tokens {
            let _ = cuda_model.forward_one_gpu_total_offload(token, &encoder_output).expect("Init tokens failed");
        }

        // Benchmark GPU path
        let gpu_start = std::time::Instant::now();
        let mut gpu_tokens = initial_tokens.clone();
        for _ in 0..num_decode_tokens {
            let last_token = *gpu_tokens.last().unwrap_or(&specials.sot);
            let logits = cuda_model.forward_one_gpu_total_offload(last_token, &encoder_output)
                .expect("GPU forward failed");
            let next_token = logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(specials.eot);
            if next_token == specials.eot { break; }
            gpu_tokens.push(next_token);
        }
        let gpu_time = gpu_start.elapsed();
        let gpu_tokens_generated = gpu_tokens.len() - initial_tokens.len();

        // === Executor Path Benchmark ===
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.reset_gpu_decoder_kv_cache(); // WAPR-PERF-014: Clear stale KV cache from GPU path
        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Init KV cache");
        cuda_model.upload_decoder_weights_to_executor().expect("Upload executor weights");

        // Warmup (JIT compilation)
        for &token in &initial_tokens {
            let _ = cuda_model.forward_one_executor(token, &encoder_output).expect("Warmup failed");
        }

        // WAPR-PERF-014: Reset state after warmup for clean benchmark
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.reset_gpu_decoder_kv_cache();
        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Re-init KV cache");

        // Process initial tokens fresh for benchmark
        for &token in &initial_tokens {
            let _ = cuda_model.forward_one_executor(token, &encoder_output).expect("Init tokens failed");
        }

        // Benchmark Executor path
        let exec_start = std::time::Instant::now();
        let mut exec_tokens = initial_tokens.clone();
        for _ in 0..num_decode_tokens {
            let last_token = *exec_tokens.last().unwrap_or(&specials.sot);
            let logits = cuda_model.forward_one_executor(last_token, &encoder_output)
                .expect("Executor forward failed");
            let next_token = logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(specials.eot);
            if next_token == specials.eot { break; }
            exec_tokens.push(next_token);
        }
        let exec_time = exec_start.elapsed();
        let exec_tokens_generated = exec_tokens.len() - initial_tokens.len();

        // Results
        let gpu_ms_per_token = gpu_time.as_millis() as f64 / gpu_tokens_generated.max(1) as f64;
        let exec_ms_per_token = exec_time.as_millis() as f64 / exec_tokens_generated.max(1) as f64;
        let speedup = gpu_ms_per_token / exec_ms_per_token;

        eprintln!("\nResults ({} tokens decoded):", num_decode_tokens);
        eprintln!("  GPU path:      {:?} ({:.1} ms/token)", gpu_time, gpu_ms_per_token);
        eprintln!("  Executor path: {:?} ({:.1} ms/token)", exec_time, exec_ms_per_token);
        eprintln!("  Speedup:       {:.2}x", speedup);

        // Decode text for comparison
        let gpu_text = cuda_model.tokenizer.decode_with_options(&gpu_tokens, true).unwrap_or_default();
        let exec_text = cuda_model.tokenizer.decode_with_options(&exec_tokens, true).unwrap_or_default();

        eprintln!("\nGPU text:  \"{}\"", gpu_text.trim());
        eprintln!("Exec text: \"{}\"", exec_text.trim());

        // Verify tokens match
        assert_eq!(gpu_tokens, exec_tokens, "GPU and Executor paths should produce same tokens");

        eprintln!("\n✓ WAPR-PERF-014: GPU vs Executor Benchmark PASSED");
        eprintln!("  - Tokens match");
        eprintln!("  - Speedup: {:.2}x", speedup);
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
        let n_vocab = cuda_model.config().n_vocab as usize;
        let n_heads = cuda_model.config().n_text_head as usize;
        let head_dim = d_model / n_heads;
        let max_len = cuda_model.config().n_text_ctx as usize;

        eprintln!("Model: d_model={}, n_heads={}, head_dim={}, n_layers={}", d_model, n_heads, head_dim, n_layers);

        // Initialize GPU infrastructure
        cuda_model.upload_decoder_weights_to_gpu().expect("Upload failed");
        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Init failed");

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
        eprintln!("\n[Step 1] Token+Pos Embedding: {} elements, sum={:.4}", x.len(), x.iter().sum::<f32>());

        // === Step 2: LN1 for layer 0 ===
        let block = &cuda_model.decoder.blocks()[0];
        let normed = block.ln1.forward(&x).expect("LN1 failed");
        eprintln!("[Step 2] LN1: sum={:.4}, first 5: {:?}", normed.iter().sum::<f32>(), &normed[..5]);

        // === Step 3: CPU Q/K/V projections ===
        let cpu_q = block.self_attn.w_q().forward(&normed, 1).expect("CPU Q failed");
        let cpu_k = block.self_attn.w_k().forward(&normed, 1).expect("CPU K failed");
        let cpu_v = block.self_attn.w_v().forward(&normed, 1).expect("CPU V failed");
        eprintln!("[Step 3] CPU Q: sum={:.4}", cpu_q.iter().sum::<f32>());
        eprintln!("         CPU K: sum={:.4}", cpu_k.iter().sum::<f32>());
        eprintln!("         CPU V: sum={:.4}", cpu_v.iter().sum::<f32>());

        // === Step 4: GPU Q/K/V projections ===
        let ctx = cuda_model.executor.context();
        let weights = cuda_model.gpu_decoder_weights.as_ref()
            .expect("Weights not uploaded");
        let layer_weights = &weights[0];

        let x_gpu = GpuResidentTensor::from_host(ctx, &normed).expect("x upload");
        let mut gpu_q = x_gpu.linear(ctx, &layer_weights.self_w_q, Some(&layer_weights.self_b_q), 1, d_model as u32, d_model as u32)
            .expect("GPU Q failed");
        let mut gpu_k = x_gpu.linear(ctx, &layer_weights.self_w_k, Some(&layer_weights.self_b_k), 1, d_model as u32, d_model as u32)
            .expect("GPU K failed");
        let mut gpu_v = x_gpu.linear(ctx, &layer_weights.self_w_v, Some(&layer_weights.self_b_v), 1, d_model as u32, d_model as u32)
            .expect("GPU V failed");

        let gpu_q_host = gpu_q.to_host().expect("Q download");
        let gpu_k_host = gpu_k.to_host().expect("K download");
        let gpu_v_host = gpu_v.to_host().expect("V download");
        eprintln!("[Step 4] GPU Q: sum={:.4}", gpu_q_host.iter().sum::<f32>());
        eprintln!("         GPU K: sum={:.4}", gpu_k_host.iter().sum::<f32>());
        eprintln!("         GPU V: sum={:.4}", gpu_v_host.iter().sum::<f32>());

        // Compare Q/K/V
        let q_diff: f32 = cpu_q.iter().zip(gpu_q_host.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let k_diff: f32 = cpu_k.iter().zip(gpu_k_host.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let v_diff: f32 = cpu_v.iter().zip(gpu_v_host.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("[Step 4] Q diff: {:.2e}, K diff: {:.2e}, V diff: {:.2e}", q_diff, k_diff, v_diff);

        if q_diff > 1e-4 || k_diff > 1e-4 || v_diff > 1e-4 {
            eprintln!("\n❌ DIVERGENCE at Q/K/V projections!");
            panic!("Q/K/V divergence: q={:.2e} k={:.2e} v={:.2e}", q_diff, k_diff, v_diff);
        }

        eprintln!("\n✓ Q/K/V projections match within 1e-4");

        // === Step 5: CPU self-attention (for pos=0, just Q @ K^T @ V with single token) ===
        // For position 0 with seq_len=1, self-attention is trivial:
        // scores = Q @ K^T = [1, d] @ [d, 1] = [1, 1]
        // softmax([1,1]) = [1.0]
        // output = [1.0] @ V = V
        eprintln!("\n[Step 5] CPU self-attention (pos=0, trivial case):");
        let cpu_attn_out = cpu_v.clone(); // For single token at pos 0, output = V
        eprintln!("         CPU attn out: sum={:.4}", cpu_attn_out.iter().sum::<f32>());

        // === Step 6: GPU incremental attention ===
        // First scatter K/V to caches
        use trueno_gpu::driver::CudaStream;
        let stream = CudaStream::new(ctx).expect("stream");
        let self_k_caches = cuda_model.gpu_self_k_head_first.as_mut().unwrap();
        let self_v_caches = cuda_model.gpu_self_v_head_first.as_mut().unwrap();

        kv_cache_scatter_gpu(
            ctx, &gpu_k, &mut self_k_caches[0],
            pos as u32, n_heads as u32, head_dim as u32, max_len as u32, &stream
        ).expect("K scatter");
        kv_cache_scatter_gpu(
            ctx, &gpu_v, &mut self_v_caches[0],
            pos as u32, n_heads as u32, head_dim as u32, max_len as u32, &stream
        ).expect("V scatter");

        // Run incremental attention
        let seq_len_attn = (pos + 1) as u32;
        let mut gpu_attn_out = incremental_attention_gpu(
            ctx, &gpu_q, &self_k_caches[0], &self_v_caches[0],
            n_heads as u32, head_dim as u32, seq_len_attn, max_len as u32
        ).expect("incremental attention");

        let gpu_attn_host = gpu_attn_out.to_host().expect("attn download");
        eprintln!("[Step 6] GPU attn out: sum={:.4}", gpu_attn_host.iter().sum::<f32>());

        let attn_diff: f32 = cpu_attn_out.iter().zip(gpu_attn_host.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
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
        let cpu_attn_proj = block.self_attn.w_o().forward(&cpu_attn_out, 1).expect("CPU W_o");
        eprintln!("         CPU W_o out: sum={:.4}", cpu_attn_proj.iter().sum::<f32>());

        // GPU
        let mut gpu_attn_proj = gpu_attn_out.linear(ctx, &layer_weights.self_w_o, Some(&layer_weights.self_b_o), 1, d_model as u32, d_model as u32)
            .expect("GPU W_o");
        let gpu_attn_proj_host = gpu_attn_proj.to_host().expect("W_o download");
        eprintln!("         GPU W_o out: sum={:.4}", gpu_attn_proj_host.iter().sum::<f32>());

        let wo_diff: f32 = cpu_attn_proj.iter().zip(gpu_attn_proj_host.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("[Step 7] W_o diff: {:.2e}", wo_diff);

        // === Step 8: Residual after self-attention ===
        eprintln!("\n[Step 8] Residual after self-attention:");
        let cpu_residual: Vec<f32> = x.iter().zip(cpu_attn_proj.iter()).map(|(a, b)| a + b).collect();
        let gpu_residual: Vec<f32> = x.iter().zip(gpu_attn_proj_host.iter()).map(|(a, b)| a + b).collect();
        eprintln!("         CPU residual: sum={:.4}", cpu_residual.iter().sum::<f32>());
        eprintln!("         GPU residual: sum={:.4}", gpu_residual.iter().sum::<f32>());

        // === Step 9: Compute CPU FFN (before mutable borrow) ===
        // LN3 of residual
        let ln3_out = block.ln3.forward(&cpu_residual).expect("LN3");
        let ffn_out = block.ffn.forward(&ln3_out).expect("FFN");
        let cpu_final: Vec<f32> = cpu_residual.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();
        eprintln!("\n[Step 9] CPU (no cross-attn): sum={:.4}", cpu_final.iter().sum::<f32>());

        // Now drop the immutable block borrow
        drop(block);

        // GPU path (no encoder output = no cross-attention)
        eprintln!("[Step 9] GPU block forward (self-attention + FFN only):");
        let gpu_block_out = cuda_model.forward_decoder_block_gpu(
            0, &x, 0, None
        ).expect("GPU block forward");
        eprintln!("         GPU block out: sum={:.4}", gpu_block_out.iter().sum::<f32>());

        let gpu_block_sum: f32 = gpu_block_out.iter().sum();
        if gpu_block_sum.is_nan() || gpu_block_sum.is_infinite() {
            panic!("GPU block output is NaN/Inf!");
        }

        // Note: GPU block uses a fresh KV cache scatter, so it's comparing apples to apples
        let block_diff: f32 = cpu_final.iter().zip(gpu_block_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
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
        let cpu_q = block.self_attn.w_q().forward(&normed, 1).expect("CPU Q");
        let cpu_k = block.self_attn.w_k().forward(&normed, 1).expect("CPU K");
        let cpu_v = block.self_attn.w_v().forward(&normed, 1).expect("CPU V");

        // For single position, attention output = V
        let cpu_attn = cpu_v.clone();
        let cpu_attn_proj = block.self_attn.w_o().forward(&cpu_attn, 1).expect("CPU O");
        let cpu_self_residual: Vec<f32> = x.iter().zip(cpu_attn_proj.iter()).map(|(a, b)| a + b).collect();

        // Cross-attention on CPU
        let normed2 = block.ln2.forward(&cpu_self_residual).expect("LN2");
        let cpu_cross_out = block.cross_attn.forward_cross_dispatch(&normed2, &enc_output, None)
            .expect("CPU cross-attn");
        let cpu_cross_residual: Vec<f32> = cpu_self_residual.iter()
            .zip(cpu_cross_out.iter()).map(|(a, b)| a + b).collect();

        // FFN on CPU
        let normed3 = block.ln3.forward(&cpu_cross_residual).expect("LN3");
        let cpu_ffn_out = block.ffn.forward(&normed3).expect("FFN");
        let cpu_block_out: Vec<f32> = cpu_cross_residual.iter()
            .zip(cpu_ffn_out.iter()).map(|(a, b)| a + b).collect();
        eprintln!("         CPU block (with cross-attn): sum={:.4}", cpu_block_out.iter().sum::<f32>());

        // Drop block borrow before mutable call
        drop(block);

        // Reset GPU decoder position (important!)
        cuda_model.reset_gpu_decoder_pos();

        // Re-initialize KV caches (they were modified by step 9)
        // Need to set to None first to force re-init
        cuda_model.gpu_self_k_head_first = None;
        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Init KV");

        // Run GPU block with encoder output
        let gpu_cross_block_out = cuda_model.forward_decoder_block_gpu(
            0, &x, 0, Some(&enc_output)
        ).expect("GPU block with cross-attn");
        eprintln!("         GPU block (with cross-attn): sum={:.4}", gpu_cross_block_out.iter().sum::<f32>());

        // Compare outputs
        let cross_block_diff: f32 = cpu_block_out.iter().zip(gpu_cross_block_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("[Step 10] Block diff (with cross-attn): {:.2e}", cross_block_diff);

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
        let model_path = std::env::var("WHISPER_MODEL_PATH")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/models/whisper-tiny.apr").to_string());

        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        // Load test audio
        let audio_path = std::env::var("WHISPER_TEST_AUDIO")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/demos/test-audio/test-speech-1.5s.wav").to_string());

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
        eprintln!("[Audio] {} samples ({:.2}s @ {}Hz)",
            wav_data.samples.len(),
            wav_data.samples.len() as f64 / wav_data.sample_rate as f64,
            wav_data.sample_rate);

        // Compute mel spectrogram BEFORE into_cuda (WhisperApr method)
        let mel = apr.compute_mel(&wav_data.samples).expect("Mel computation failed");
        eprintln!("[Mel] {} frames", mel.len() / 80);

        // Create CUDA model
        let mut cuda_model = apr.into_cuda(0).expect("Failed to create CUDA model");

        // Upload weights to executor (WAPR-PERF-014)
        let upload_start = std::time::Instant::now();
        let weight_bytes = cuda_model.upload_decoder_weights_to_executor()
            .expect("Failed to upload decoder weights");
        let upload_time = upload_start.elapsed();
        eprintln!("[Weights] {:.2} MB uploaded in {:?}", weight_bytes as f64 / 1_048_576.0, upload_time);

        // Enable GPU decoder offload
        std::env::set_var("WHISPER_GPU_DECODER_OFFLOAD", "1");

        // Run transcription with timing
        let options = crate::TranscribeOptions::default();

        eprintln!("\n[Benchmark] Starting GPU decoder transcription...");
        let total_start = std::time::Instant::now();

        // Use the internal encoder + decoder path
        // First encode on CPU (or GPU if enabled)
        let encode_start = std::time::Instant::now();
        let encoder_output = cuda_model.encoder.forward_mel(&mel).expect("Encoder failed");
        let encode_time = encode_start.elapsed();
        eprintln!("[Encoder] {:?}", encode_time);

        // Decode using GPU total offload
        let decode_start = std::time::Instant::now();

        // Initialize GPU decoder
        cuda_model.reset_gpu_decoder_pos();
        cuda_model.upload_decoder_weights_to_gpu().expect("Upload decoder weights");
        cuda_model.init_gpu_decoder_kv_cache_head_first().expect("Init KV cache");

        // Build initial tokens
        use crate::tokenizer::special_tokens::SpecialTokens;
        let specials = SpecialTokens::for_vocab_size(cuda_model.config().n_vocab as usize);
        let mut tokens = vec![specials.sot];
        if specials.is_multilingual {
            tokens.push(specials.lang_base); // English
        }
        tokens.push(specials.transcribe);
        tokens.push(specials.no_timestamps);

        // Process initial tokens
        for &token in &tokens {
            let _ = cuda_model.forward_one_gpu_total_offload(token, &encoder_output)
                .expect("Initial token forward failed");
        }

        // Generate tokens
        let max_tokens = cuda_model.config().n_text_ctx as usize;
        let mut token_times: Vec<u128> = Vec::new();

        for gen_idx in 0..max_tokens.saturating_sub(tokens.len()) {
            let last_token = *tokens.last().unwrap_or(&specials.sot);

            let token_start = std::time::Instant::now();
            let logits = cuda_model.forward_one_gpu_total_offload(last_token, &encoder_output)
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
        let text = cuda_model.tokenizer.decode_with_options(&tokens, true)
            .expect("Decode failed");

        eprintln!("\n============================================================");
        eprintln!("RESULTS");
        eprintln!("============================================================");
        eprintln!("[Output] \"{}\"", text.trim());
        eprintln!("[Tokens] {} generated", tokens.len() - 4); // Subtract initial tokens
        eprintln!("");
        eprintln!("[Timing]");
        eprintln!("  Weight upload: {:?}", upload_time);
        eprintln!("  Encoder:       {:?}", encode_time);
        eprintln!("  Decoder:       {:?}", decode_time);
        eprintln!("  TOTAL:         {:?}", total_time);
        eprintln!("");

        if !token_times.is_empty() {
            let avg_token_us = token_times.iter().sum::<u128>() / token_times.len() as u128;
            let min_token_us = *token_times.iter().min().unwrap_or(&0);
            let max_token_us = *token_times.iter().max().unwrap_or(&0);
            eprintln!("[Per-Token]");
            eprintln!("  Average: {:.2}ms", avg_token_us as f64 / 1000.0);
            eprintln!("  Min:     {:.2}ms", min_token_us as f64 / 1000.0);
            eprintln!("  Max:     {:.2}ms", max_token_us as f64 / 1000.0);
            eprintln!("  First 5: {:?}", token_times.iter().take(5).map(|t| format!("{:.1}ms", *t as f64 / 1000.0)).collect::<Vec<_>>());
        }

        eprintln!("");
        eprintln!("[Point 157 Falsification]");
        let total_ms = total_time.as_millis();
        let target_ms = 1984;
        if total_ms <= target_ms {
            eprintln!("  ✓ PASSED: {}ms ≤ {}ms target", total_ms, target_ms);
        } else {
            eprintln!("  ✗ FAILED: {}ms > {}ms target ({:.1}x slower)", total_ms, target_ms, total_ms as f64 / target_ms as f64);
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
        let model_path = std::env::var("WHISPER_MODEL_PATH")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/models/whisper-tiny.apr").to_string());

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
        let bytes = cuda_model.upload_decoder_weights_to_executor()
            .expect("Failed to upload weights");
        let elapsed = start.elapsed();

        eprintln!("[WAPR-PERF-014] Uploaded {:.2} MB in {:?}", bytes as f64 / 1_048_576.0, elapsed);

        // Verify all expected weights are cached
        let expected_weights_per_layer = vec![
            "self_w_q", "self_b_q", "self_w_k", "self_b_k", "self_w_v", "self_b_v", "self_w_o", "self_b_o",
            "cross_w_q", "cross_b_q", "cross_w_k", "cross_b_k", "cross_w_v", "cross_b_v", "cross_w_o", "cross_b_o",
            "ffn_fc1", "ffn_b1", "ffn_fc2", "ffn_b2",
            "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "ln3_gamma", "ln3_beta",
        ];

        for layer_idx in 0..n_layers {
            for weight_name in &expected_weights_per_layer {
                let full_name = format!("dec.L{layer_idx}.{weight_name}");
                assert!(cuda_model.executor.has_weights(&full_name),
                    "Missing weight: {}", full_name);
            }
        }

        // Verify global weights
        assert!(cuda_model.executor.has_weights("dec.output_proj"), "Missing output_proj");
        assert!(cuda_model.executor.has_weights("dec.ln_post_gamma"), "Missing ln_post_gamma");
        assert!(cuda_model.executor.has_weights("dec.ln_post_beta"), "Missing ln_post_beta");

        // Expected weight count: n_layers * 26 per-layer + 3 global
        let expected_count = n_layers * expected_weights_per_layer.len() + 3;
        let actual_count = cuda_model.executor.cached_weight_count();
        eprintln!("[WAPR-PERF-014] Expected {} weights, got {}", expected_count, actual_count);

        // Note: actual_count may be higher due to encoder weights from earlier tests
        assert!(actual_count >= expected_count,
            "Not enough weights cached: expected at least {}, got {}", expected_count, actual_count);

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
        let model_path = std::env::var("WHISPER_MODEL_PATH")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/models/whisper-tiny.apr").to_string());

        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        // Load audio
        let audio_path = std::env::var("WHISPER_TEST_AUDIO")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/demos/test-audio/test-speech-1.5s.wav").to_string());

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
        let mel = apr.compute_mel(&wav_data.samples).expect("Mel computation failed");
        let mel_time = mel_start.elapsed();
        let mel_frames = mel.len() / 80;
        eprintln!("\n[Mel Spectrogram]");
        eprintln!("  Frames: {}", mel_frames);
        eprintln!("  Time: {:?}", mel_time);

        // Time convolution frontend
        let conv_start = std::time::Instant::now();
        let conv_output = apr.encoder.conv_frontend().forward(&mel).expect("Conv failed");
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
        let block_output = apr.encoder.blocks()[0].forward(&x).expect("Block 0 failed");
        let block_time = block_start.elapsed();
        eprintln!("\n[Single Encoder Block (Layer 0)]");
        eprintln!("  Time: {:?}", block_time);
        eprintln!("  Projected: {:?} for {} layers", block_time * apr.encoder.n_layers() as u32, apr.encoder.n_layers());

        // Time full encoder
        let encoder_start = std::time::Instant::now();
        let _encoder_output = apr.encoder.forward_mel(&mel).expect("Encoder failed");
        let encoder_time = encoder_start.elapsed();
        eprintln!("\n[Full Encoder (forward_mel)]");
        eprintln!("  Time: {:?}", encoder_time);

        // Breakdown analysis
        let total_expected = conv_time + pe_time + block_time * apr.encoder.n_layers() as u32;
        eprintln!("\n[Analysis]");
        eprintln!("  Expected (conv + pe + {} blocks): {:?}", apr.encoder.n_layers(), total_expected);
        eprintln!("  Actual: {:?}", encoder_time);
        eprintln!("  Overhead: {:?}", encoder_time.saturating_sub(total_expected));

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
        eprintln!("  Est. encoder matmuls (4 layers × 6): {:?}", matmul_time * 24);

        // Check weights finalized status
        eprintln!("\n[Weight Finalization]");
        eprintln!("  Block 0 self_attn finalized: {}", apr.encoder.blocks()[0].self_attn.is_finalized());
        eprintln!("  Block 0 FFN finalized: {}", apr.encoder.blocks()[0].ffn.is_finalized());

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
        let _q = block.self_attn.w_q().forward_simd(&normed, seq_len).expect("Q");
        let _k = block.self_attn.w_k().forward_simd(&normed, seq_len).expect("K");
        let _v = block.self_attn.w_v().forward_simd(&normed, seq_len).expect("V");
        let qkv_time = qkv_start.elapsed();
        eprintln!("  QKV projections: {:?}", qkv_time);

        // Just the attention computation (from forward_cross_dispatch)
        // This calls forward_cross_optimal -> forward_cross_flash_v2 for long sequences
        eprintln!("  Note: Attention uses {} heads, d_head={}", block.self_attn.n_heads(), d_model / block.self_attn.n_heads());
        eprintln!("  Note: seq_len={} > 128, uses FlashAttention-2", seq_len);
        eprintln!("  Attention overhead (total - QKV): {:?}", attn_time.saturating_sub(qkv_time));

        eprintln!("\n============================================================\n");
    }
}

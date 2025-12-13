//! Model loading and inference
//!
//! Handles loading .apr model files and running transformer inference.

mod attention;
mod decoder;
mod encoder;
pub mod quantized;

pub use attention::{LinearWeights, MultiHeadAttention};
pub use decoder::{
    BatchDecoderCache, BatchDecoderOutput, Decoder, DecoderBlock, DecoderKVCache, LayerKVCache,
    StreamingCacheStats, StreamingKVCache,
};
pub use encoder::{Conv1d, ConvFrontend, Encoder, EncoderBlock, FeedForward, LayerNorm};
pub use quantized::{QuantizedLinear, QuantizedTensor};

use crate::error::WhisperResult;
use crate::ModelType;

/// Whisper model configuration from .apr header
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model type (tiny, base, small, etc.)
    pub model_type: ModelType,
    /// Vocabulary size
    pub n_vocab: u32,
    /// Audio context length
    pub n_audio_ctx: u32,
    /// Audio hidden state dimension
    pub n_audio_state: u32,
    /// Number of audio attention heads
    pub n_audio_head: u32,
    /// Number of audio encoder layers
    pub n_audio_layer: u32,
    /// Text context length
    pub n_text_ctx: u32,
    /// Text hidden state dimension
    pub n_text_state: u32,
    /// Number of text attention heads
    pub n_text_head: u32,
    /// Number of text decoder layers
    pub n_text_layer: u32,
    /// Number of mel filterbank channels
    pub n_mels: u32,
}

impl ModelConfig {
    /// Create configuration for tiny model
    #[must_use]
    pub const fn tiny() -> Self {
        Self {
            model_type: ModelType::Tiny,
            n_vocab: 51865,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            n_mels: 80,
        }
    }

    /// Create configuration for base model
    #[must_use]
    pub const fn base() -> Self {
        Self {
            model_type: ModelType::Base,
            n_vocab: 51865,
            n_audio_ctx: 1500,
            n_audio_state: 512,
            n_audio_head: 8,
            n_audio_layer: 6,
            n_text_ctx: 448,
            n_text_state: 512,
            n_text_head: 8,
            n_text_layer: 6,
            n_mels: 80,
        }
    }

    /// Create configuration for small model
    #[must_use]
    pub const fn small() -> Self {
        Self {
            model_type: ModelType::Small,
            n_vocab: 51865,
            n_audio_ctx: 1500,
            n_audio_state: 768,
            n_audio_head: 12,
            n_audio_layer: 12,
            n_text_ctx: 448,
            n_text_state: 768,
            n_text_head: 12,
            n_text_layer: 12,
            n_mels: 80,
        }
    }

    /// Create configuration for medium model
    #[must_use]
    pub const fn medium() -> Self {
        Self {
            model_type: ModelType::Medium,
            n_vocab: 51865,
            n_audio_ctx: 1500,
            n_audio_state: 1024,
            n_audio_head: 16,
            n_audio_layer: 24,
            n_text_ctx: 448,
            n_text_state: 1024,
            n_text_head: 16,
            n_text_layer: 24,
            n_mels: 80,
        }
    }

    /// Create configuration for large model
    #[must_use]
    pub const fn large() -> Self {
        Self {
            model_type: ModelType::Large,
            n_vocab: 51865,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_text_ctx: 448,
            n_text_state: 1280,
            n_text_head: 20,
            n_text_layer: 32,
            n_mels: 80,
        }
    }

    // =========================================================================
    // Memory Estimation (Spec 9.4)
    // =========================================================================

    /// Estimate model parameters count
    ///
    /// Accounts for:
    /// - Encoder: conv layers + transformer blocks
    /// - Decoder: embedding + transformer blocks + output projection
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let d_model = self.n_audio_state as usize;
        let d_text = self.n_text_state as usize;
        let n_vocab = self.n_vocab as usize;
        let n_mels = self.n_mels as usize;

        // Encoder parameters
        let encoder_conv1 = n_mels * d_model * 3 + d_model; // 3x1 conv
        let encoder_conv2 = d_model * d_model * 3 + d_model; // 3x3 conv
        let encoder_embed = self.n_audio_ctx as usize * d_model; // positional embedding
        let encoder_block = self.attention_block_params(d_model);
        let encoder_total = encoder_conv1
            + encoder_conv2
            + encoder_embed
            + encoder_block * self.n_audio_layer as usize;

        // Decoder parameters
        let decoder_embed = n_vocab * d_text; // token embedding
        let decoder_pos = self.n_text_ctx as usize * d_text; // positional embedding
        let decoder_block =
            self.attention_block_params(d_text) + self.cross_attention_params(d_text, d_model);
        let decoder_ln = d_text * 2; // final layer norm
        let decoder_proj = d_text * n_vocab; // output projection
        let decoder_total = decoder_embed
            + decoder_pos
            + decoder_block * self.n_text_layer as usize
            + decoder_ln
            + decoder_proj;

        encoder_total + decoder_total
    }

    /// Parameters in a self-attention block
    fn attention_block_params(&self, d_model: usize) -> usize {
        // Self-attention: Q, K, V, O projections
        let attn = d_model * d_model * 4 + d_model * 4;
        // FFN: up projection, down projection
        let ffn = d_model * d_model * 4 * 2 + d_model * 4 + d_model;
        // Layer norms: 2 per block
        let ln = d_model * 4;
        attn + ffn + ln
    }

    /// Parameters for cross-attention
    fn cross_attention_params(&self, d_text: usize, d_audio: usize) -> usize {
        // Cross-attention Q from text, K/V from audio
        let attn = d_text * d_text + d_audio * d_text * 2 + d_text * d_text + d_text * 4;
        // Layer norm
        let ln = d_text * 2;
        attn + ln
    }

    /// Estimate model weights memory in bytes (F32)
    #[must_use]
    pub fn weights_memory_bytes(&self) -> usize {
        self.parameter_count() * 4 // f32 = 4 bytes
    }

    /// Estimate model weights memory in MB
    #[must_use]
    pub fn weights_memory_mb(&self) -> f32 {
        self.weights_memory_bytes() as f32 / (1024.0 * 1024.0)
    }

    /// Estimate KV cache memory for a given sequence length
    ///
    /// KV cache stores key/value tensors for all decoder layers
    #[must_use]
    pub fn kv_cache_memory_bytes(&self, seq_len: usize) -> usize {
        let d_text = self.n_text_state as usize;
        let n_layers = self.n_text_layer as usize;
        let n_heads = self.n_text_head as usize;
        let head_dim = d_text / n_heads;

        // Each layer stores K and V for self-attention and cross-attention
        // Shape: [n_heads, seq_len, head_dim] for K and V each
        let kv_per_layer = 2 * n_heads * seq_len * head_dim * 4; // 2 for K,V, 4 for f32
        let cross_kv_per_layer = 2 * n_heads * self.n_audio_ctx as usize * head_dim * 4;

        (kv_per_layer + cross_kv_per_layer) * n_layers
    }

    /// Estimate activation memory during forward pass
    ///
    /// This is the peak memory for intermediate tensors
    #[must_use]
    pub fn activation_memory_bytes(&self) -> usize {
        let d_audio = self.n_audio_state as usize;
        let d_text = self.n_text_state as usize;
        let audio_ctx = self.n_audio_ctx as usize;
        let text_ctx = self.n_text_ctx as usize;

        // Encoder activations (largest tensor is attention scores)
        let encoder_attn = self.n_audio_head as usize * audio_ctx * audio_ctx * 4;
        let encoder_ffn = audio_ctx * d_audio * 4 * 4; // 4x expansion

        // Decoder activations
        let decoder_attn = self.n_text_head as usize * text_ctx * text_ctx * 4;
        let decoder_cross = self.n_text_head as usize * text_ctx * audio_ctx * 4;
        let decoder_ffn = text_ctx * d_text * 4 * 4;

        // Take max of encoder/decoder peaks + some buffer
        let encoder_peak = encoder_attn.max(encoder_ffn);
        let decoder_peak = decoder_attn.max(decoder_cross).max(decoder_ffn);

        (encoder_peak + decoder_peak) * 2 // 2x for gradient-like buffers
    }

    /// Estimate total peak memory usage in bytes
    ///
    /// Includes: weights + KV cache + activations + working buffers
    #[must_use]
    pub fn peak_memory_bytes(&self) -> usize {
        let weights = self.weights_memory_bytes();
        let kv_cache = self.kv_cache_memory_bytes(self.n_text_ctx as usize);
        let activations = self.activation_memory_bytes();
        let working_buffers = 10 * 1024 * 1024; // 10MB for misc buffers

        weights + kv_cache + activations + working_buffers
    }

    /// Estimate total peak memory in MB
    #[must_use]
    pub fn peak_memory_mb(&self) -> f32 {
        self.peak_memory_bytes() as f32 / (1024.0 * 1024.0)
    }

    /// Get recommended minimum WASM memory pages
    ///
    /// WASM pages are 64KB each
    #[must_use]
    pub fn recommended_wasm_pages(&self) -> u32 {
        let bytes = self.peak_memory_bytes();
        let pages = (bytes + 65535) / 65536; // Round up
        let pages = pages.max(256); // Minimum 16MB
        pages as u32
    }

    /// Check if model can run with given memory limit
    #[must_use]
    pub fn can_run_with_memory(&self, available_mb: u32) -> bool {
        self.peak_memory_mb() <= available_mb as f32
    }

    /// Get human-readable memory requirements
    #[must_use]
    pub fn memory_summary(&self) -> String {
        format!(
            "Model: {:?}\n  Parameters: {:.1}M\n  Weights: {:.1} MB\n  Peak Memory: {:.1} MB\n  WASM Pages: {}",
            self.model_type,
            self.parameter_count() as f32 / 1_000_000.0,
            self.weights_memory_mb(),
            self.peak_memory_mb(),
            self.recommended_wasm_pages()
        )
    }
}

/// Loaded Whisper model
pub struct WhisperModel {
    config: ModelConfig,
    encoder: Encoder,
    decoder: Decoder,
}

impl WhisperModel {
    /// Load model from .apr file bytes
    ///
    /// # Arguments
    /// * `data` - Raw .apr file bytes
    ///
    /// # Errors
    /// Returns error if model data is invalid
    pub fn load(data: &[u8]) -> WhisperResult<Self> {
        let config = ModelConfig::tiny();
        let encoder = Encoder::new(&config);
        let decoder = Decoder::new(&config);

        let _ = data; // Will be used when implementing actual loading

        Ok(Self {
            config,
            encoder,
            decoder,
        })
    }

    /// Get model configuration
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get encoder reference
    #[must_use]
    pub const fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// Get decoder reference
    #[must_use]
    pub const fn decoder(&self) -> &Decoder {
        &self.decoder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_config() {
        let config = ModelConfig::tiny();
        assert_eq!(config.n_audio_state, 384);
        assert_eq!(config.n_audio_layer, 4);
    }

    #[test]
    fn test_base_config() {
        let config = ModelConfig::base();
        assert_eq!(config.n_audio_state, 512);
        assert_eq!(config.n_audio_layer, 6);
    }

    #[test]
    fn test_model_load() {
        let result = WhisperModel::load(&[]);
        assert!(result.is_ok());
    }

    // =========================================================================
    // Memory Estimation Tests
    // =========================================================================

    #[test]
    fn test_parameter_count_tiny() {
        let config = ModelConfig::tiny();
        let params = config.parameter_count();
        // Estimation includes all weights in encoder/decoder blocks
        assert!(
            params > 50_000_000,
            "Tiny should have >50M params, got {params}"
        );
        assert!(
            params < 70_000_000,
            "Tiny should have <70M params, got {params}"
        );
    }

    #[test]
    fn test_parameter_count_base() {
        let config = ModelConfig::base();
        let params = config.parameter_count();
        // Base model has more parameters than tiny
        assert!(
            params > 90_000_000,
            "Base should have >90M params, got {params}"
        );
        assert!(
            params < 130_000_000,
            "Base should have <130M params, got {params}"
        );
    }

    #[test]
    fn test_weights_memory_tiny() {
        let config = ModelConfig::tiny();
        let mb = config.weights_memory_mb();
        // Weights = params * 4 bytes
        assert!(mb > 200.0, "Tiny weights should be >200MB, got {mb}");
        assert!(mb < 300.0, "Tiny weights should be <300MB, got {mb}");
    }

    #[test]
    fn test_weights_memory_base() {
        let config = ModelConfig::base();
        let mb = config.weights_memory_mb();
        // Base has more weights than tiny
        assert!(mb > 350.0, "Base weights should be >350MB, got {mb}");
        assert!(mb < 550.0, "Base weights should be <550MB, got {mb}");
    }

    #[test]
    fn test_kv_cache_memory() {
        let config = ModelConfig::tiny();
        let kv_bytes = config.kv_cache_memory_bytes(448); // Full context
                                                          // KV cache should be significant but much smaller than weights
        assert!(kv_bytes > 1_000_000, "KV cache should be >1MB");
        assert!(kv_bytes < 100_000_000, "KV cache should be <100MB");
    }

    #[test]
    fn test_activation_memory() {
        let config = ModelConfig::tiny();
        let act_bytes = config.activation_memory_bytes();
        // Activations should be substantial
        assert!(act_bytes > 10_000_000, "Activations should be >10MB");
        assert!(act_bytes < 500_000_000, "Activations should be <500MB");
    }

    #[test]
    fn test_peak_memory() {
        let config = ModelConfig::tiny();
        let peak_mb = config.peak_memory_mb();
        // Peak should be more than weights alone
        let weights_mb = config.weights_memory_mb();
        assert!(peak_mb > weights_mb, "Peak should exceed weights");
        // But reasonable for a tiny model
        assert!(peak_mb < 500.0, "Tiny peak should be <500MB, got {peak_mb}");
    }

    #[test]
    fn test_wasm_pages() {
        let config = ModelConfig::tiny();
        let pages = config.recommended_wasm_pages();
        // Minimum 256 pages (16MB)
        assert!(pages >= 256, "Should have at least 256 pages");
        // Reasonable upper bound for tiny model
        assert!(
            pages < 10000,
            "Tiny shouldn't need >10000 pages, got {pages}"
        );
    }

    #[test]
    fn test_can_run_with_memory() {
        let tiny = ModelConfig::tiny();
        let base = ModelConfig::base();

        // 2GB should be enough for both
        assert!(tiny.can_run_with_memory(2048));
        assert!(base.can_run_with_memory(2048));

        // 50MB should not be enough
        assert!(!tiny.can_run_with_memory(50));
        assert!(!base.can_run_with_memory(50));
    }

    #[test]
    fn test_memory_summary() {
        let config = ModelConfig::tiny();
        let summary = config.memory_summary();
        assert!(summary.contains("Tiny"));
        assert!(summary.contains("Parameters"));
        assert!(summary.contains("MB"));
        assert!(summary.contains("WASM Pages"));
    }

    #[test]
    fn test_base_requires_more_memory_than_tiny() {
        let tiny = ModelConfig::tiny();
        let base = ModelConfig::base();

        assert!(base.parameter_count() > tiny.parameter_count());
        assert!(base.weights_memory_mb() > tiny.weights_memory_mb());
        assert!(base.peak_memory_mb() > tiny.peak_memory_mb());
    }

    // =========================================================================
    // v1.1 Extended Model Support - RED Tests (WAPR-070, WAPR-071)
    // =========================================================================

    #[test]
    fn test_small_config() {
        let config = ModelConfig::small();
        // Small: 768-dim, 12 layers, 12 heads (per OpenAI Whisper specs)
        assert_eq!(config.n_audio_state, 768);
        assert_eq!(config.n_audio_layer, 12);
        assert_eq!(config.n_audio_head, 12);
        assert_eq!(config.n_text_state, 768);
        assert_eq!(config.n_text_layer, 12);
        assert_eq!(config.n_text_head, 12);
    }

    #[test]
    fn test_medium_config() {
        let config = ModelConfig::medium();
        // Medium: 1024-dim, 24 layers, 16 heads (per OpenAI Whisper specs)
        assert_eq!(config.n_audio_state, 1024);
        assert_eq!(config.n_audio_layer, 24);
        assert_eq!(config.n_audio_head, 16);
        assert_eq!(config.n_text_state, 1024);
        assert_eq!(config.n_text_layer, 24);
        assert_eq!(config.n_text_head, 16);
    }

    #[test]
    fn test_large_config() {
        let config = ModelConfig::large();
        // Large: 1280-dim, 32 layers, 20 heads (per OpenAI Whisper specs)
        assert_eq!(config.n_audio_state, 1280);
        assert_eq!(config.n_audio_layer, 32);
        assert_eq!(config.n_audio_head, 20);
        assert_eq!(config.n_text_state, 1280);
        assert_eq!(config.n_text_layer, 32);
        assert_eq!(config.n_text_head, 20);
    }

    #[test]
    fn test_medium_requires_more_memory_than_small() {
        let small = ModelConfig::small();
        let medium = ModelConfig::medium();

        assert!(medium.parameter_count() > small.parameter_count());
        assert!(medium.weights_memory_mb() > small.weights_memory_mb());
        assert!(medium.peak_memory_mb() > small.peak_memory_mb());
    }

    #[test]
    fn test_large_requires_more_memory_than_medium() {
        let medium = ModelConfig::medium();
        let large = ModelConfig::large();

        assert!(large.parameter_count() > medium.parameter_count());
        assert!(large.weights_memory_mb() > medium.weights_memory_mb());
        assert!(large.peak_memory_mb() > medium.peak_memory_mb());
    }

    #[test]
    fn test_parameter_count_small() {
        let config = ModelConfig::small();
        let params = config.parameter_count();
        // Small model should have ~244M parameters
        assert!(
            params > 200_000_000,
            "Small should have >200M params, got {params}"
        );
        assert!(
            params < 350_000_000,
            "Small should have <350M params, got {params}"
        );
    }

    #[test]
    fn test_parameter_count_medium() {
        let config = ModelConfig::medium();
        let params = config.parameter_count();
        // Medium model should have ~769M parameters
        assert!(
            params > 600_000_000,
            "Medium should have >600M params, got {params}"
        );
        assert!(
            params < 900_000_000,
            "Medium should have <900M params, got {params}"
        );
    }

    #[test]
    fn test_parameter_count_large() {
        let config = ModelConfig::large();
        let params = config.parameter_count();
        // Large model should have ~1.5B parameters
        assert!(
            params > 1_200_000_000,
            "Large should have >1.2B params, got {params}"
        );
        assert!(
            params < 2_000_000_000,
            "Large should have <2B params, got {params}"
        );
    }

    #[test]
    fn test_medium_memory_requirements() {
        let config = ModelConfig::medium();
        let mb = config.weights_memory_mb();
        // Medium weights should be ~3GB
        assert!(mb > 2000.0, "Medium weights should be >2GB, got {mb}");
        assert!(mb < 4000.0, "Medium weights should be <4GB, got {mb}");
    }

    #[test]
    fn test_large_memory_requirements() {
        let config = ModelConfig::large();
        let mb = config.weights_memory_mb();
        // Large weights should be ~5-6GB
        assert!(mb > 4000.0, "Large weights should be >4GB, got {mb}");
        assert!(mb < 8000.0, "Large weights should be <8GB, got {mb}");
    }

    #[test]
    fn test_all_model_sizes_hierarchy() {
        let tiny = ModelConfig::tiny();
        let base = ModelConfig::base();
        let small = ModelConfig::small();
        let medium = ModelConfig::medium();
        let large = ModelConfig::large();

        // Verify size hierarchy
        assert!(tiny.parameter_count() < base.parameter_count());
        assert!(base.parameter_count() < small.parameter_count());
        assert!(small.parameter_count() < medium.parameter_count());
        assert!(medium.parameter_count() < large.parameter_count());
    }
}

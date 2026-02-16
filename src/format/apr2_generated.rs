#![allow(clippy::all, clippy::pedantic, clippy::restriction, clippy::nursery)]
//! APR Format - LLM Architecture Support
//!
//! This module provides LLM-specific model configuration for architectures
//! like LFM2-2.6B-Transcript, extending the canonical APR format with:
//!
//! - Grouped Query Attention (GQA)
//! - SwiGLU FFN activation
//! - Hybrid Conv/Attention layers
//! - RoPE positional encoding
//! - int4 AWQ/GPTQ quantization
//!
//! Uses the canonical APR format from aprender::format::v2 ("APR\0" magic).
//!
//! # Spec Reference
//!
//! See `docs/specifications/1.0-whisper-apr.md` Section 18.8 for full specification.

use crate::error::{WhisperError, WhisperResult};

// Use canonical APR v2 magic from aprender
pub use aprender::format::v2::MAGIC_V2 as MAGIC_APR2;

/// APR format version (matches aprender::format::v2)
pub const APR2_VERSION: u16 = 2;

// =============================================================================
// Model Family
// =============================================================================

/// Model architecture family
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelFamily {
    /// OpenAI Whisper (ASR)
    Whisper = 0,
    /// LiquidAI LFM2 (LLM for transcript summarization)
    Lfm2 = 1,
    /// Meta Llama-style architecture
    Llama = 2,
    /// Useful Sensors Moonshine (ASR, variable-length input)
    Moonshine = 3,
    /// Generic transformer
    Generic = 255,
}

impl TryFrom<u8> for ModelFamily {
    type Error = WhisperError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Whisper),
            1 => Ok(Self::Lfm2),
            2 => Ok(Self::Llama),
            3 => Ok(Self::Moonshine),
            255 => Ok(Self::Generic),
            _ => Err(WhisperError::Format(format!(
                "unknown model family: {value}"
            ))),
        }
    }
}

impl core::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Whisper => write!(f, "whisper"),
            Self::Lfm2 => write!(f, "lfm2"),
            Self::Llama => write!(f, "llama"),
            Self::Moonshine => write!(f, "moonshine"),
            Self::Generic => write!(f, "generic"),
        }
    }
}

// =============================================================================
// Quantization Config
// =============================================================================

/// Quantization method for APR2 format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Apr2Quantization {
    /// 32-bit floating point (no quantization)
    F32 = 0,
    /// 16-bit floating point
    F16 = 1,
    /// BFloat16
    Bf16 = 2,
    /// 8-bit integer (absmax per-tensor)
    Int8 = 3,
    /// 4-bit integer (absmax per-tensor)
    Int4 = 4,
    /// 4-bit AWQ (Activation-aware Weight Quantization)
    Int4Awq = 5,
    /// 4-bit GPTQ (GPT Quantization)
    Int4Gptq = 6,
}

impl TryFrom<u8> for Apr2Quantization {
    type Error = WhisperError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Bf16),
            3 => Ok(Self::Int8),
            4 => Ok(Self::Int4),
            5 => Ok(Self::Int4Awq),
            6 => Ok(Self::Int4Gptq),
            _ => Err(WhisperError::Format(format!(
                "unknown quantization: {value}"
            ))),
        }
    }
}

impl Apr2Quantization {
    /// Bytes per element (approximate for sub-byte quantization)
    #[must_use]
    pub const fn bytes_per_element(&self) -> f32 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::Bf16 => 2.0,
            Self::Int8 => 1.0,
            Self::Int4 | Self::Int4Awq | Self::Int4Gptq => 0.5,
        }
    }

    /// Whether this quantization uses group-wise scaling
    #[must_use]
    pub const fn is_grouped(&self) -> bool {
        matches!(self, Self::Int4Awq | Self::Int4Gptq)
    }
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Primary quantization method
    pub method: Apr2Quantization,
    /// Group size for grouped quantization (0 = per-tensor)
    pub group_size: u32,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            method: Apr2Quantization::F32,
            group_size: 0,
            symmetric: true,
        }
    }
}

impl QuantConfig {
    /// Create int8 config
    #[must_use]
    pub fn int8(group_size: u32) -> Self {
        Self {
            method: Apr2Quantization::Int8,
            group_size,
            symmetric: true,
        }
    }

    /// Create fp16 config
    #[must_use]
    pub fn fp16() -> Self {
        Self {
            method: Apr2Quantization::F16,
            group_size: 0,
            symmetric: true,
        }
    }

    /// Create bf16 config
    #[must_use]
    pub fn bf16() -> Self {
        Self {
            method: Apr2Quantization::Bf16,
            group_size: 0,
            symmetric: true,
        }
    }

    /// Create int4 AWQ config
    #[must_use]
    pub fn int4_awq(group_size: u32) -> Self {
        Self {
            method: Apr2Quantization::Int4Awq,
            group_size,
            symmetric: false,
        }
    }

    /// Create int4 GPTQ config
    #[must_use]
    pub fn int4_gptq(group_size: u32) -> Self {
        Self {
            method: Apr2Quantization::Int4Gptq,
            group_size,
            symmetric: false,
        }
    }

    /// Serialize to bytes (8 bytes)
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        bytes[0] = self.method as u8;
        bytes[1..5].copy_from_slice(&self.group_size.to_le_bytes());
        bytes[5] = u8::from(self.symmetric);
        // bytes[6..8] reserved
        bytes
    }

    /// Parse from bytes
    ///
    /// # Errors
    /// Returns error if data is invalid
    pub fn from_bytes(data: &[u8]) -> WhisperResult<Self> {
        if data.len() < 8 {
            return Err(WhisperError::Format("quant config too short".into()));
        }
        Ok(Self {
            method: Apr2Quantization::try_from(data[0])?,
            group_size: u32::from_le_bytes([data[1], data[2], data[3], data[4]]),
            symmetric: data[5] != 0,
        })
    }
}

// =============================================================================
// Layer Types
// =============================================================================

/// Layer type in hybrid architectures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    /// 1D Convolution layer
    Convolution {
        /// Kernel size
        kernel_size: u32,
        /// Cache length for streaming
        cache_len: u32,
    },
    /// Full attention layer
    Attention {
        /// Whether to use Grouped Query Attention
        use_gqa: bool,
    },
    /// Feed-forward network
    Ffn {
        /// Activation function
        activation: FfnActivation,
    },
}

impl LayerType {
    /// Serialize to bytes (8 bytes)
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        match self {
            Self::Convolution {
                kernel_size,
                cache_len,
            } => {
                bytes[0] = 0; // Convolution type
                bytes[1..5].copy_from_slice(&kernel_size.to_le_bytes());
                bytes[5] = *cache_len as u8;
            }
            Self::Attention { use_gqa } => {
                bytes[0] = 1; // Attention type
                bytes[1] = u8::from(*use_gqa);
            }
            Self::Ffn { activation } => {
                bytes[0] = 2; // FFN type
                bytes[1] = *activation as u8;
            }
        }
        bytes
    }

    /// Parse from bytes
    ///
    /// # Errors
    /// Returns error if data is invalid
    pub fn from_bytes(data: &[u8]) -> WhisperResult<Self> {
        if data.len() < 8 {
            return Err(WhisperError::Format("layer type too short".into()));
        }
        match data[0] {
            0 => Ok(Self::Convolution {
                kernel_size: u32::from_le_bytes([data[1], data[2], data[3], data[4]]),
                cache_len: u32::from(data[5]),
            }),
            1 => Ok(Self::Attention {
                use_gqa: data[1] != 0,
            }),
            2 => Ok(Self::Ffn {
                activation: FfnActivation::try_from(data[1])?,
            }),
            t => Err(WhisperError::Format(format!("unknown layer type: {t}"))),
        }
    }
}

/// FFN activation function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FfnActivation {
    /// GELU activation (Whisper, BERT)
    Gelu = 0,
    /// SiLU/Swish activation (Llama)
    Silu = 1,
    /// SwiGLU activation (LFM2)
    Swiglu = 2,
    /// ReLU activation
    Relu = 3,
}

impl TryFrom<u8> for FfnActivation {
    type Error = WhisperError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Gelu),
            1 => Ok(Self::Silu),
            2 => Ok(Self::Swiglu),
            3 => Ok(Self::Relu),
            _ => Err(WhisperError::Format(format!("unknown activation: {value}"))),
        }
    }
}

// =============================================================================
// LFM2 Architecture Config
// =============================================================================

/// LFM2 architecture configuration
///
/// Based on config.json from `LiquidAI/LFM2-2.6B-Transcript`:
/// - hidden_size: 2048
/// - num_hidden_layers: 30
/// - num_attention_heads: 32
/// - num_key_value_heads: 8 (GQA)
/// - intermediate_size: 10752
/// - vocab_size: 65536
/// - rope_theta: 1000000.0
#[derive(Debug, Clone)]
pub struct Lfm2Config {
    /// Hidden state dimension
    pub hidden_size: u32,
    /// Number of transformer layers
    pub num_layers: u32,
    /// Number of query attention heads
    pub num_q_heads: u32,
    /// Number of key/value heads (for GQA)
    pub num_kv_heads: u32,
    /// FFN intermediate size
    pub intermediate_size: u32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// RoPE theta for positional encoding
    pub rope_theta: f32,
    /// Convolution dimension
    pub conv_dimension: u32,
    /// Maximum sequence length
    pub max_seq_len: u32,
    /// Layer types (conv/attention pattern)
    pub layer_types: Vec<LayerType>,
}

impl Default for Lfm2Config {
    /// Default config for LFM2-2.6B-Transcript
    fn default() -> Self {
        Self::lfm2_2_6b()
    }
}

impl Lfm2Config {
    /// LFM2-2.6B-Transcript configuration
    #[must_use]
    pub fn lfm2_2_6b() -> Self {
        // Generate layer types: pattern of Conv, Conv, Attention repeated
        let mut layer_types = Vec::with_capacity(30);
        for i in 0..30 {
            if i % 3 == 2 {
                // Every 3rd layer is attention
                layer_types.push(LayerType::Attention { use_gqa: true });
            } else {
                layer_types.push(LayerType::Convolution {
                    kernel_size: 4,
                    cache_len: 3,
                });
            }
        }

        Self {
            hidden_size: 2048,
            num_layers: 30,
            num_q_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 10752,
            vocab_size: 65536,
            rope_theta: 1_000_000.0,
            conv_dimension: 2048,
            max_seq_len: 128000,
            layer_types,
        }
    }

    /// LLaMA 7B configuration
    ///
    /// Based on Meta's LLaMA architecture with standard attention.
    #[must_use]
    pub fn llama_7b() -> Self {
        // LLaMA uses all attention layers (no conv)
        let layer_types = vec![LayerType::Attention { use_gqa: false }; 32];

        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_q_heads: 32,
            num_kv_heads: 32, // Standard MHA (no GQA in LLaMA-1)
            intermediate_size: 11008,
            vocab_size: 32000,
            rope_theta: 10_000.0, // Standard RoPE theta
            conv_dimension: 0,    // No conv layers
            max_seq_len: 4096,
            layer_types,
        }
    }

    /// LLaMA 2 7B configuration with GQA
    #[must_use]
    pub fn llama2_7b() -> Self {
        let layer_types = vec![LayerType::Attention { use_gqa: true }; 32];

        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_q_heads: 32,
            num_kv_heads: 8, // GQA with 4:1 ratio
            intermediate_size: 11008,
            vocab_size: 32000,
            rope_theta: 10_000.0,
            conv_dimension: 0,
            max_seq_len: 4096,
            layer_types,
        }
    }

    /// Whisper tiny configuration (for summarization adapter)
    ///
    /// Note: This is adapted for text-only summarization, not audio encoding.
    /// The original Whisper encoder uses different attention patterns.
    #[must_use]
    pub fn whisper_tiny() -> Self {
        let layer_types = vec![LayerType::Attention { use_gqa: false }; 4];

        Self {
            hidden_size: 384,
            num_layers: 4,
            num_q_heads: 6,
            num_kv_heads: 6,         // Standard MHA
            intermediate_size: 1536, // 4x hidden
            vocab_size: 51865,       // Whisper vocab
            rope_theta: 10_000.0,
            conv_dimension: 0,
            max_seq_len: 1500, // Audio frames
            layer_types,
        }
    }

    /// Whisper base configuration
    #[must_use]
    pub fn whisper_base() -> Self {
        let layer_types = vec![LayerType::Attention { use_gqa: false }; 6];

        Self {
            hidden_size: 512,
            num_layers: 6,
            num_q_heads: 8,
            num_kv_heads: 8,
            intermediate_size: 2048,
            vocab_size: 51865,
            rope_theta: 10_000.0,
            conv_dimension: 0,
            max_seq_len: 1500,
            layer_types,
        }
    }

    /// Moonshine tiny configuration (Useful Sensors)
    ///
    /// Variable-length ASR with MHA, GELU/SiLU FFN, and RoPE.
    /// 27.1M params, 288-dim, 6 encoder + 6 decoder layers.
    /// Matches `usefulsensors/moonshine-tiny` on HuggingFace.
    #[must_use]
    pub fn moonshine_tiny() -> Self {
        let layer_types = vec![LayerType::Attention { use_gqa: false }; 6];

        Self {
            hidden_size: 288,
            num_layers: 6,
            num_q_heads: 8,
            num_kv_heads: 8, // MHA (kv_heads = q_heads)
            intermediate_size: 1152, // 4x expansion
            vocab_size: 32768,       // SentencePiece
            rope_theta: 10_000.0,
            conv_dimension: 0,
            max_seq_len: 2048,
            layer_types,
        }
    }

    /// Moonshine base configuration (Useful Sensors)
    ///
    /// Variable-length ASR with MHA, GELU/SiLU FFN, and RoPE.
    /// 61.5M params, 416-dim, 8 encoder + 8 decoder layers.
    /// Matches `usefulsensors/moonshine-base` on HuggingFace.
    #[must_use]
    pub fn moonshine_base() -> Self {
        let layer_types = vec![LayerType::Attention { use_gqa: false }; 8];

        Self {
            hidden_size: 416,
            num_layers: 8,
            num_q_heads: 8,
            num_kv_heads: 8, // MHA (kv_heads = q_heads)
            intermediate_size: 1664, // 4x expansion
            vocab_size: 32768,
            rope_theta: 10_000.0,
            conv_dimension: 0,
            max_seq_len: 2048,
            layer_types,
        }
    }

    /// Whisper small configuration
    #[must_use]
    pub fn whisper_small() -> Self {
        let layer_types = vec![LayerType::Attention { use_gqa: false }; 12];

        Self {
            hidden_size: 768,
            num_layers: 12,
            num_q_heads: 12,
            num_kv_heads: 12,
            intermediate_size: 3072,
            vocab_size: 51865,
            rope_theta: 10_000.0,
            conv_dimension: 0,
            max_seq_len: 1500,
            layer_types,
        }
    }

    /// Calculate GQA ratio (query heads per KV head)
    #[must_use]
    pub const fn gqa_ratio(&self) -> u32 {
        if self.num_kv_heads > 0 {
            self.num_q_heads / self.num_kv_heads
        } else {
            1
        }
    }

    /// Estimate model size in bytes for given quantization
    ///
    /// Uses the model's parameter count (2.6B for LFM2-2.6B) to estimate storage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_size_bytes(&self, quant: Apr2Quantization) -> u64 {
        let bytes_per_param = quant.bytes_per_element();

        // LFM2-2.6B has 2.6 billion parameters
        // We calculate this from architecture to be precise:
        //
        // Embedding: vocab_size * hidden_size = 65536 * 2048 = 134M
        // For each of 30 layers:
        //   - Attention/Conv weights
        //   - FFN (SwiGLU): 3 * hidden * intermediate = 3 * 2048 * 10752 = 66M
        //   - LayerNorm: 2 * hidden = 4K (negligible)
        //
        // Total is approximately 2.6B parameters

        // Embedding parameters
        let embedding = u64::from(self.vocab_size) * u64::from(self.hidden_size);

        // Per layer FFN (SwiGLU has 3 matrices: gate, up, down)
        let ffn_per_layer = 3 * u64::from(self.hidden_size) * u64::from(self.intermediate_size);

        // Per attention layer: Q, K, V, O projections
        let h = u64::from(self.hidden_size);
        let kv_dim = u64::from(self.num_kv_heads) * (h / u64::from(self.num_q_heads));
        let attn_per_layer = h * h    // Q projection
            + h * kv_dim              // K projection (GQA)
            + h * kv_dim              // V projection (GQA)
            + h * h; // O projection

        // Per conv layer (simplified)
        let conv_per_layer = u64::from(self.conv_dimension) * u64::from(self.conv_dimension);

        // Count layer types
        let num_attn_layers = self
            .layer_types
            .iter()
            .filter(|l| matches!(l, LayerType::Attention { .. }))
            .count() as u64;
        let num_conv_layers = self
            .layer_types
            .iter()
            .filter(|l| matches!(l, LayerType::Convolution { .. }))
            .count() as u64;

        // Total layers contribute FFN + attention/conv
        let total_params = embedding  // input embedding
            + u64::from(self.num_layers) * ffn_per_layer  // FFN for all layers
            + num_attn_layers * attn_per_layer            // attention layers
            + num_conv_layers * conv_per_layer            // conv layers
            + embedding; // output projection (typically tied, but count for safety)

        #[allow(clippy::cast_sign_loss)]
        let size = (total_params as f64 * f64::from(bytes_per_param)) as u64;
        size
    }

    /// Estimate KV cache size per token in bytes
    ///
    /// KV cache stores key and value states for each attention layer.
    /// With GQA, we only store num_kv_heads (not num_q_heads) K/V pairs.
    #[must_use]
    pub fn kv_cache_per_token_bytes(&self) -> u64 {
        // K and V for each attention layer
        // Shape: [num_kv_heads, head_dim] for both K and V
        let head_dim = u64::from(self.hidden_size / self.num_q_heads);
        // 2 for K+V, 2 bytes per fp16 element
        let kv_per_layer = 2 * u64::from(self.num_kv_heads) * head_dim * 2;

        // Count attention layers (not conv layers)
        let num_attn_layers = self
            .layer_types
            .iter()
            .filter(|l| matches!(l, LayerType::Attention { .. }))
            .count() as u64;

        kv_per_layer * num_attn_layers
    }

    /// Header size in bytes
    pub const HEADER_SIZE: usize = 48;

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; Self::HEADER_SIZE];

        bytes[0..4].copy_from_slice(&self.hidden_size.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.num_layers.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.num_q_heads.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.num_kv_heads.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.intermediate_size.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.vocab_size.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.rope_theta.to_le_bytes());
        bytes[28..32].copy_from_slice(&self.conv_dimension.to_le_bytes());
        bytes[32..36].copy_from_slice(&self.max_seq_len.to_le_bytes());
        bytes[36..40].copy_from_slice(&(self.layer_types.len() as u32).to_le_bytes());
        // bytes[40..48] reserved

        // Append layer types
        for layer in &self.layer_types {
            bytes.extend_from_slice(&layer.to_bytes());
        }

        bytes
    }

    /// Parse from bytes
    ///
    /// # Errors
    /// Returns error if data is invalid
    pub fn from_bytes(data: &[u8]) -> WhisperResult<Self> {
        if data.len() < Self::HEADER_SIZE {
            return Err(WhisperError::Format("lfm2 config too short".into()));
        }

        let hidden_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let num_layers = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let num_q_heads = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let num_kv_heads = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let intermediate_size = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let vocab_size = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        let rope_theta = f32::from_le_bytes([data[24], data[25], data[26], data[27]]);
        let conv_dimension = u32::from_le_bytes([data[28], data[29], data[30], data[31]]);
        let max_seq_len = u32::from_le_bytes([data[32], data[33], data[34], data[35]]);
        let num_layer_types = u32::from_le_bytes([data[36], data[37], data[38], data[39]]) as usize;

        // Parse layer types
        let layer_data_start = Self::HEADER_SIZE;
        let layer_data_end = layer_data_start + num_layer_types * 8;
        if data.len() < layer_data_end {
            return Err(WhisperError::Format("layer types data too short".into()));
        }

        let mut layer_types = Vec::with_capacity(num_layer_types);
        for i in 0..num_layer_types {
            let offset = layer_data_start + i * 8;
            layer_types.push(LayerType::from_bytes(&data[offset..offset + 8])?);
        }

        Ok(Self {
            hidden_size,
            num_layers,
            num_q_heads,
            num_kv_heads,
            intermediate_size,
            vocab_size,
            rope_theta,
            conv_dimension,
            max_seq_len,
            layer_types,
        })
    }
}

// =============================================================================
// APR2 Header
// =============================================================================

/// APR2 file header
#[derive(Debug, Clone)]
pub struct Apr2Header {
    /// Format version
    pub version: u16,
    /// Model architecture family
    pub family: ModelFamily,
    /// Quantization configuration
    pub quant: QuantConfig,
    /// Number of tensors
    pub n_tensors: u32,
    /// Architecture-specific config (serialized)
    pub arch_config: Vec<u8>,
}

impl Apr2Header {
    /// Base header size (before arch config)
    pub const BASE_SIZE: usize = 16;

    /// Create header for LFM2 model
    #[must_use]
    pub fn lfm2(config: Lfm2Config, quant: QuantConfig) -> Self {
        Self {
            version: APR2_VERSION,
            family: ModelFamily::Lfm2,
            quant,
            n_tensors: 0,
            arch_config: config.to_bytes(),
        }
    }

    /// Parse LFM2 config from arch_config bytes
    ///
    /// # Errors
    /// Returns error if family is not LFM2 or config is invalid
    pub fn lfm2_config(&self) -> WhisperResult<Lfm2Config> {
        if self.family != ModelFamily::Lfm2 {
            return Err(WhisperError::Format(format!(
                "expected LFM2 family, got {:?}",
                self.family
            )));
        }
        Lfm2Config::from_bytes(&self.arch_config)
    }

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let arch_len = self.arch_config.len() as u32;
        let total_size = Self::BASE_SIZE + self.arch_config.len();
        let mut bytes = Vec::with_capacity(total_size);

        // Version (2 bytes)
        bytes.extend_from_slice(&self.version.to_le_bytes());
        // Family (1 byte)
        bytes.push(self.family as u8);
        // Reserved (1 byte)
        bytes.push(0);
        // n_tensors (4 bytes)
        bytes.extend_from_slice(&self.n_tensors.to_le_bytes());
        // Quant config (8 bytes)
        bytes.extend_from_slice(&self.quant.to_bytes());
        // Arch config length (4 bytes) - at offset 14, need 2 more bytes for alignment
        // Actually let's reorganize: BASE_SIZE should include arch_len field
        // Recompute: version(2) + family(1) + reserved(1) + n_tensors(4) + quant(8) = 16
        // We need arch_len somewhere. Let's put it in the reserved area or extend.

        // Actually, let me fix the layout:
        // 0..2: version
        // 2: family
        // 3: reserved
        // 4..8: n_tensors
        // 8..12: arch_config_len
        // 12..20: quant_config
        // 20..: arch_config

        // Let me rewrite this properly
        let mut bytes = Vec::with_capacity(20 + self.arch_config.len());
        bytes.extend_from_slice(&self.version.to_le_bytes()); // 0..2
        bytes.push(self.family as u8); // 2
        bytes.push(0); // 3 reserved
        bytes.extend_from_slice(&self.n_tensors.to_le_bytes()); // 4..8
        bytes.extend_from_slice(&arch_len.to_le_bytes()); // 8..12
        bytes.extend_from_slice(&self.quant.to_bytes()); // 12..20
        bytes.extend_from_slice(&self.arch_config); // 20..

        bytes
    }

    /// Parse from bytes
    ///
    /// # Errors
    /// Returns error if data is invalid
    pub fn from_bytes(data: &[u8]) -> WhisperResult<Self> {
        if data.len() < 20 {
            return Err(WhisperError::Format("apr2 header too short".into()));
        }

        let version = u16::from_le_bytes([data[0], data[1]]);
        if version > APR2_VERSION {
            return Err(WhisperError::Format(format!(
                "unsupported apr2 version: {version}"
            )));
        }

        let family = ModelFamily::try_from(data[2])?;
        let n_tensors = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let arch_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let quant = QuantConfig::from_bytes(&data[12..20])?;

        if data.len() < 20 + arch_len {
            return Err(WhisperError::Format("arch config truncated".into()));
        }

        let arch_config = data[20..20 + arch_len].to_vec();

        Ok(Self {
            version,
            family,
            quant,
            n_tensors,
            arch_config,
        })
    }
}

// =============================================================================
// WASM Configuration (from spec Section 18.7)
// =============================================================================

/// WASM-optimized configuration for LFM2
///
/// From spec Section 18.7: Memory budget ~2.5GB for int4 + 4K context.
#[derive(Debug, Clone)]
pub struct Lfm2WasmConfig {
    /// Quantization method (int4 AWQ recommended)
    pub quantization: Apr2Quantization,
    /// Maximum context length (4096 recommended for WASM)
    pub max_context: u32,
    /// Sliding window size for bounded KV cache
    pub sliding_window: Option<u32>,
    /// Whether to use WebGPU acceleration
    pub use_webgpu: bool,
    /// Whether to stream tokens during generation
    pub streaming: bool,
}

impl Default for Lfm2WasmConfig {
    fn default() -> Self {
        Self {
            quantization: Apr2Quantization::Int4Awq,
            max_context: 4096,
            sliding_window: Some(2048),
            use_webgpu: true,
            streaming: true,
        }
    }
}

impl Lfm2WasmConfig {
    /// Estimate total memory usage in bytes
    #[must_use]
    pub fn estimate_memory_bytes(&self, config: &Lfm2Config) -> u64 {
        // Model weights
        let model_bytes = config.estimate_size_bytes(self.quantization);

        // KV cache
        let cache_len = self.sliding_window.unwrap_or(self.max_context);
        let kv_bytes = config.kv_cache_per_token_bytes() * u64::from(cache_len);

        // Runtime overhead (~200MB)
        let overhead: u64 = 200 * 1024 * 1024;

        model_bytes + kv_bytes + overhead
    }

    /// Check if configuration fits in WASM memory limit
    #[must_use]
    pub fn fits_in_wasm(&self, config: &Lfm2Config) -> bool {
        // Browser practical limit is ~2GB
        const WASM_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
        self.estimate_memory_bytes(config) <= WASM_LIMIT
    }
}

// =============================================================================
// APR2 Tensor Descriptor
// =============================================================================

/// Tensor descriptor for APR2 format
///
/// Each tensor in an APR2 file has a descriptor containing metadata
/// about its name, shape, and location in the file.
#[derive(Debug, Clone)]
pub struct Apr2TensorDescriptor {
    /// Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight")
    pub name: String,
    /// Tensor shape (up to 4 dimensions)
    pub shape: [u32; 4],
    /// Number of dimensions
    pub n_dims: u8,
    /// Data type / quantization method
    pub dtype: Apr2Quantization,
    /// Offset from start of tensor data section
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
    /// Number of elements
    pub n_elements: u64,
}

impl Apr2TensorDescriptor {
    /// Size of each tensor descriptor entry in bytes
    pub const ENTRY_SIZE: usize = 128;

    /// Create a new tensor descriptor
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        shape: &[usize],
        dtype: Apr2Quantization,
        offset: u64,
        size: u64,
    ) -> Self {
        let mut shape_arr = [0u32; 4];
        let n_dims = shape.len().min(4);
        for (i, &dim) in shape.iter().take(4).enumerate() {
            shape_arr[i] = dim as u32;
        }

        let n_elements = shape.iter().product::<usize>() as u64;

        Self {
            name: name.into(),
            shape: shape_arr,
            n_dims: n_dims as u8,
            dtype,
            offset,
            size,
            n_elements,
        }
    }

    /// Get shape as slice
    #[must_use]
    pub fn shape(&self) -> &[u32] {
        &self.shape[..self.n_dims as usize]
    }

    /// Serialize to bytes (128 bytes)
    ///
    /// Layout:
    /// - 0..64: name (null-terminated UTF-8)
    /// - 64..68: shape[0] (u32 LE)
    /// - 68..72: shape[1] (u32 LE)
    /// - 72..76: shape[2] (u32 LE)
    /// - 76..80: shape[3] (u32 LE)
    /// - 80: n_dims (u8)
    /// - 81: dtype (u8)
    /// - 82..84: reserved
    /// - 84..92: offset (u64 LE)
    /// - 92..100: size (u64 LE)
    /// - 100..108: n_elements (u64 LE)
    /// - 108..128: reserved
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::ENTRY_SIZE] {
        let mut bytes = [0u8; Self::ENTRY_SIZE];

        // Write name (null-terminated, max 63 chars)
        let name_bytes = self.name.as_bytes();
        let name_len = name_bytes.len().min(63);
        bytes[..name_len].copy_from_slice(&name_bytes[..name_len]);
        // bytes[name_len] is already 0 (null terminator)

        // Write shape
        for (i, &dim) in self.shape.iter().enumerate() {
            let offset = 64 + i * 4;
            bytes[offset..offset + 4].copy_from_slice(&dim.to_le_bytes());
        }

        // Write n_dims and dtype
        bytes[80] = self.n_dims;
        bytes[81] = self.dtype as u8;

        // Write offset, size, n_elements
        bytes[84..92].copy_from_slice(&self.offset.to_le_bytes());
        bytes[92..100].copy_from_slice(&self.size.to_le_bytes());
        bytes[100..108].copy_from_slice(&self.n_elements.to_le_bytes());

        bytes
    }

    /// Parse from bytes
    ///
    /// # Errors
    /// Returns error if data is invalid
    pub fn from_bytes(data: &[u8]) -> WhisperResult<Self> {
        if data.len() < Self::ENTRY_SIZE {
            return Err(WhisperError::Format(
                "apr2 tensor descriptor too short".into(),
            ));
        }

        // Parse name (null-terminated, max 64 bytes)
        let name_bytes = &data[0..64];
        let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(64);
        let name = String::from_utf8_lossy(&name_bytes[..name_end]).into_owned();

        // Parse shape
        let mut shape = [0u32; 4];
        for (i, dim) in shape.iter_mut().enumerate() {
            let offset = 64 + i * 4;
            *dim = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
        }

        let n_dims = data[80];
        let dtype = Apr2Quantization::try_from(data[81])?;

        let offset = u64::from_le_bytes([
            data[84], data[85], data[86], data[87], data[88], data[89], data[90], data[91],
        ]);
        let size = u64::from_le_bytes([
            data[92], data[93], data[94], data[95], data[96], data[97], data[98], data[99],
        ]);
        let n_elements = u64::from_le_bytes([
            data[100], data[101], data[102], data[103], data[104], data[105], data[106], data[107],
        ]);

        Ok(Self {
            name,
            shape,
            n_dims,
            dtype,
            offset,
            size,
            n_elements,
        })
    }
}

// =============================================================================
// APR2 Reader
// =============================================================================

/// APR2 file reader
///
/// Reads .apr2 files containing LLM weights in the APR2 format.
#[derive(Debug)]
pub struct Apr2Reader {
    /// Parsed header
    pub header: Apr2Header,
    /// Tensor descriptors
    pub tensors: Vec<Apr2TensorDescriptor>,
    /// Offset to tensor data section
    tensor_data_offset: usize,
    /// Raw file data
    data: Vec<u8>,
}

impl Apr2Reader {
    /// Create reader from file bytes
    ///
    /// # Errors
    /// Returns error if file is invalid
    pub fn new(data: Vec<u8>) -> WhisperResult<Self> {
        // Validate magic
        if data.len() < 4 {
            return Err(WhisperError::Format("file too short".into()));
        }
        if data[..4] != MAGIC_APR2 {
            return Err(WhisperError::Format("invalid APR magic".into()));
        }

        // Parse header
        let header = Apr2Header::from_bytes(&data[4..])?;
        let header_end = 4 + 20 + header.arch_config.len();

        // Parse tensor index
        let n_tensors = header.n_tensors as usize;
        let index_size = n_tensors * Apr2TensorDescriptor::ENTRY_SIZE;
        let tensor_data_offset = header_end + index_size;

        if data.len() < tensor_data_offset {
            return Err(WhisperError::Format(
                "file too short for tensor index".into(),
            ));
        }

        let mut tensors = Vec::with_capacity(n_tensors);
        for i in 0..n_tensors {
            let start = header_end + i * Apr2TensorDescriptor::ENTRY_SIZE;
            let end = start + Apr2TensorDescriptor::ENTRY_SIZE;
            tensors.push(Apr2TensorDescriptor::from_bytes(&data[start..end])?);
        }

        Ok(Self {
            header,
            tensors,
            tensor_data_offset,
            data,
        })
    }

    /// Get LFM2 config from header
    ///
    /// # Errors
    /// Returns error if model is not LFM2
    pub fn lfm2_config(&self) -> WhisperResult<Lfm2Config> {
        self.header.lfm2_config()
    }

    /// Get number of tensors
    #[must_use]
    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Find tensor by name
    #[must_use]
    pub fn find_tensor(&self, name: &str) -> Option<&Apr2TensorDescriptor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get raw tensor data
    ///
    /// # Errors
    /// Returns error if tensor not found or data out of bounds
    pub fn tensor_data(&self, name: &str) -> WhisperResult<&[u8]> {
        let tensor = self
            .find_tensor(name)
            .ok_or_else(|| WhisperError::Format(format!("tensor not found: {name}")))?;

        let start = self.tensor_data_offset + tensor.offset as usize;
        let end = start + tensor.size as usize;

        if end > self.data.len() {
            return Err(WhisperError::Format("tensor data out of bounds".into()));
        }

        Ok(&self.data[start..end])
    }

    /// Load tensor as f32 values
    ///
    /// Handles dequantization automatically based on tensor dtype.
    ///
    /// # Errors
    /// Returns error if tensor not found or read fails
    pub fn load_tensor_f32(&self, name: &str) -> WhisperResult<Vec<f32>> {
        let tensor = self
            .find_tensor(name)
            .ok_or_else(|| WhisperError::Format(format!("tensor not found: {name}")))?;

        let raw_data = self.tensor_data(name)?;

        match tensor.dtype {
            Apr2Quantization::F32 => {
                // Direct f32 read
                let result: Vec<f32> = raw_data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Ok(result)
            }
            Apr2Quantization::F16 => {
                // f16 to f32 conversion
                let result: Vec<f32> = raw_data
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half_to_f32(bits)
                    })
                    .collect();
                Ok(result)
            }
            Apr2Quantization::Int8 => {
                // int8 dequantization (scale is stored separately - simplified here)
                let result: Vec<f32> = raw_data.iter().map(|&b| (b as i8) as f32 / 127.0).collect();
                Ok(result)
            }
            Apr2Quantization::Int4 | Apr2Quantization::Int4Awq | Apr2Quantization::Int4Gptq => {
                // int4 dequantization (packed 2 per byte)
                let mut result = Vec::with_capacity(tensor.n_elements as usize);
                for &byte in raw_data {
                    let low = (byte & 0x0F) as i8 - 8; // 4-bit signed
                    let high = ((byte >> 4) & 0x0F) as i8 - 8;
                    result.push(low as f32 / 7.0);
                    result.push(high as f32 / 7.0);
                }
                result.truncate(tensor.n_elements as usize);
                Ok(result)
            }
            Apr2Quantization::Bf16 => {
                // bf16 to f32 conversion
                let result: Vec<f32> = raw_data
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        bf16_to_f32(bits)
                    })
                    .collect();
                Ok(result)
            }
        }
    }

    /// Get file size
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.data.len()
    }
}

/// Convert half-precision (f16) bits to f32
#[inline]
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let new_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (new_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // Normal
        let new_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (new_exp << 23) | (mant << 13))
    }
}

/// Convert bfloat16 bits to f32
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    // bfloat16 is just the upper 16 bits of f32
    f32::from_bits((bits as u32) << 16)
}

// =============================================================================
// APR2 Writer
// =============================================================================

/// APR2 file writer
///
/// Creates .apr2 files containing LLM weights.
#[derive(Debug)]
pub struct Apr2Writer {
    /// File header
    header: Apr2Header,
    /// Tensors to write
    tensors: Vec<Apr2TensorData>,
}

/// Tensor data for writing
#[derive(Debug, Clone)]
pub struct Apr2TensorData {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: Apr2Quantization,
    /// Raw data bytes
    pub data: Vec<u8>,
}

impl Apr2TensorData {
    /// Create tensor from f32 data
    #[must_use]
    pub fn from_f32(name: impl Into<String>, shape: Vec<usize>, data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        Self {
            name: name.into(),
            shape,
            dtype: Apr2Quantization::F32,
            data: bytes,
        }
    }

    /// Create tensor from int8 quantized data
    #[must_use]
    pub fn from_int8(name: impl Into<String>, shape: Vec<usize>, data: &[i8]) -> Self {
        let bytes: Vec<u8> = data.iter().map(|&v| v as u8).collect();

        Self {
            name: name.into(),
            shape,
            dtype: Apr2Quantization::Int8,
            data: bytes,
        }
    }

    /// Quantize f32 data to int8
    #[must_use]
    pub fn quantize_int8(name: impl Into<String>, shape: Vec<usize>, data: &[f32]) -> Self {
        // Find absmax for scale
        let absmax = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };

        let quantized: Vec<u8> = data
            .iter()
            .map(|&v| {
                let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
                q as u8
            })
            .collect();

        Self {
            name: name.into(),
            shape,
            dtype: Apr2Quantization::Int8,
            data: quantized,
        }
    }

    /// Number of elements
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size in bytes
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }
}

impl Apr2Writer {
    /// Create new writer with LFM2 config
    #[must_use]
    pub fn lfm2(config: Lfm2Config, quant: QuantConfig) -> Self {
        Self {
            header: Apr2Header::lfm2(config, quant),
            tensors: Vec::new(),
        }
    }

    /// Add a tensor
    pub fn add_tensor(&mut self, tensor: Apr2TensorData) {
        self.tensors.push(tensor);
    }

    /// Add f32 tensor
    pub fn add_f32(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        self.add_tensor(Apr2TensorData::from_f32(name, shape, data));
    }

    /// Add int8 tensor (quantized from f32)
    pub fn add_int8_quantized(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        self.add_tensor(Apr2TensorData::quantize_int8(name, shape, data));
    }

    /// Number of tensors
    #[must_use]
    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Write to bytes
    ///
    /// # Errors
    /// Returns error if serialization fails
    pub fn to_bytes(&self) -> WhisperResult<Vec<u8>> {
        // Calculate sizes
        let header_bytes = self.header.to_bytes();
        let index_size = self.tensors.len() * Apr2TensorDescriptor::ENTRY_SIZE;
        let data_size: usize = self.tensors.iter().map(Apr2TensorData::byte_size).sum();
        let total_size = 4 + header_bytes.len() + index_size + data_size + 4; // magic + header + index + data + crc

        let mut bytes = Vec::with_capacity(total_size);

        // 1. Magic
        bytes.extend_from_slice(&MAGIC_APR2);

        // 2. Header (with updated n_tensors)
        let mut header = self.header.clone();
        header.n_tensors = self.tensors.len() as u32;
        bytes.extend_from_slice(&header.to_bytes());

        // 3. Tensor index
        let mut offset: u64 = 0;
        for tensor in &self.tensors {
            let desc = Apr2TensorDescriptor::new(
                &tensor.name,
                &tensor.shape,
                tensor.dtype,
                offset,
                tensor.byte_size() as u64,
            );
            bytes.extend_from_slice(&desc.to_bytes());
            offset += tensor.byte_size() as u64;
        }

        // 4. Tensor data
        for tensor in &self.tensors {
            bytes.extend_from_slice(&tensor.data);
        }

        // 5. CRC32
        let crc = crate::format::crc32(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());

        Ok(bytes)
    }

    /// Write to file
    ///
    /// # Errors
    /// Returns error if file write fails
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_to_file(&self, path: impl AsRef<std::path::Path>) -> WhisperResult<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(|e| WhisperError::Format(e.to_string()))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_family_roundtrip() {
        for family in [
            ModelFamily::Whisper,
            ModelFamily::Lfm2,
            ModelFamily::Llama,
            ModelFamily::Moonshine,
            ModelFamily::Generic,
        ] {
            let byte = family as u8;
            let parsed = ModelFamily::try_from(byte).expect("should parse");
            assert_eq!(parsed, family);
        }
    }

    #[test]
    fn test_quantization_bytes_per_element() {
        assert!((Apr2Quantization::F32.bytes_per_element() - 4.0).abs() < f32::EPSILON);
        assert!((Apr2Quantization::F16.bytes_per_element() - 2.0).abs() < f32::EPSILON);
        assert!((Apr2Quantization::Int8.bytes_per_element() - 1.0).abs() < f32::EPSILON);
        assert!((Apr2Quantization::Int4.bytes_per_element() - 0.5).abs() < f32::EPSILON);
        assert!((Apr2Quantization::Int4Awq.bytes_per_element() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quant_config_roundtrip() {
        let config = QuantConfig::int4_awq(128);
        let bytes = config.to_bytes();
        let parsed = QuantConfig::from_bytes(&bytes).expect("should parse");

        assert_eq!(parsed.method, config.method);
        assert_eq!(parsed.group_size, config.group_size);
        assert_eq!(parsed.symmetric, config.symmetric);
    }

    #[test]
    fn test_layer_type_roundtrip() {
        let layers = [
            LayerType::Convolution {
                kernel_size: 4,
                cache_len: 3,
            },
            LayerType::Attention { use_gqa: true },
            LayerType::Ffn {
                activation: FfnActivation::Swiglu,
            },
        ];

        for layer in layers {
            let bytes = layer.to_bytes();
            let parsed = LayerType::from_bytes(&bytes).expect("should parse");
            assert_eq!(parsed, layer);
        }
    }

    #[test]
    fn test_lfm2_config_default() {
        let config = Lfm2Config::default();

        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 30);
        assert_eq!(config.num_q_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.intermediate_size, 10752);
        assert_eq!(config.vocab_size, 65536);
        assert!((config.rope_theta - 1_000_000.0).abs() < 1.0);
        assert_eq!(config.gqa_ratio(), 4);
    }

    #[test]
    fn test_lfm2_config_roundtrip() {
        let config = Lfm2Config::lfm2_2_6b();
        let bytes = config.to_bytes();
        let parsed = Lfm2Config::from_bytes(&bytes).expect("should parse");

        assert_eq!(parsed.hidden_size, config.hidden_size);
        assert_eq!(parsed.num_layers, config.num_layers);
        assert_eq!(parsed.num_q_heads, config.num_q_heads);
        assert_eq!(parsed.num_kv_heads, config.num_kv_heads);
        assert_eq!(parsed.layer_types.len(), config.layer_types.len());
    }

    #[test]
    fn test_lfm2_size_estimation() {
        let config = Lfm2Config::lfm2_2_6b();

        // fp16 should be multi-GB (exact size depends on architecture calculation)
        let fp16_size = config.estimate_size_bytes(Apr2Quantization::F16);
        let fp16_gb = fp16_size as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            fp16_gb > 2.0,
            "fp16 size should be >2GB for 2.6B model, got {fp16_gb:.2}GB"
        );

        // int4 should be ~4x smaller than fp16
        let int4_size = config.estimate_size_bytes(Apr2Quantization::Int4);
        let ratio = fp16_size as f64 / int4_size as f64;
        assert!(
            (ratio - 4.0).abs() < 0.5,
            "int4 should be ~4x smaller than fp16, ratio={ratio:.2}"
        );

        // int8 should be ~2x smaller than fp16
        let int8_size = config.estimate_size_bytes(Apr2Quantization::Int8);
        let ratio = fp16_size as f64 / int8_size as f64;
        assert!(
            (ratio - 2.0).abs() < 0.5,
            "int8 should be ~2x smaller than fp16, ratio={ratio:.2}"
        );
    }

    #[test]
    fn test_lfm2_kv_cache_size() {
        let config = Lfm2Config::lfm2_2_6b();
        let kv_per_token = config.kv_cache_per_token_bytes();

        // With GQA (8 KV heads, 64 head_dim, 10 attn layers):
        // 2 * 8 * 64 * 2 bytes * 10 = 20,480 bytes = 20KB per token
        let kv_kb = kv_per_token as f64 / 1024.0;
        assert!(
            kv_kb > 10.0 && kv_kb < 100.0,
            "KV cache should be 10-100KB/token with GQA, got {kv_kb:.1}KB"
        );

        // 4K context with GQA should be manageable
        let kv_4k = kv_per_token * 4096;
        let kv_4k_mb = kv_4k as f64 / (1024.0 * 1024.0);
        assert!(
            kv_4k_mb > 40.0 && kv_4k_mb < 400.0,
            "4K KV cache should be 40-400MB, got {kv_4k_mb:.1}MB"
        );
    }

    #[test]
    fn test_apr2_header_roundtrip() {
        let config = Lfm2Config::lfm2_2_6b();
        let quant = QuantConfig::int4_awq(128);
        let mut header = Apr2Header::lfm2(config, quant);
        header.n_tensors = 100;

        let bytes = header.to_bytes();
        let parsed = Apr2Header::from_bytes(&bytes).expect("should parse");

        assert_eq!(parsed.version, header.version);
        assert_eq!(parsed.family, header.family);
        assert_eq!(parsed.n_tensors, header.n_tensors);
        assert_eq!(parsed.quant.method, header.quant.method);
    }

    #[test]
    fn test_lfm2_wasm_config_memory() {
        let config = Lfm2Config::lfm2_2_6b();
        let wasm_config = Lfm2WasmConfig::default();

        let memory = wasm_config.estimate_memory_bytes(&config);
        let memory_gb = memory as f64 / (1024.0 * 1024.0 * 1024.0);

        // With int4 + GQA + 2K sliding window, should be ~1-2GB
        assert!(
            memory_gb > 0.5 && memory_gb < 3.0,
            "WASM memory estimate should be reasonable, got {memory_gb:.2}GB"
        );

        // Model bytes should be dominant
        let model_bytes = config.estimate_size_bytes(wasm_config.quantization);
        assert!(
            model_bytes > memory / 2,
            "Model weights should be dominant factor"
        );
    }

    #[test]
    fn test_lfm2_wasm_config_fits() {
        let config = Lfm2Config::lfm2_2_6b();

        // Default config (int4 AWQ + sliding window) should fit
        let default_wasm = Lfm2WasmConfig::default();
        assert!(
            default_wasm.fits_in_wasm(&config),
            "Default WASM config should fit"
        );

        // fp16 without any optimization should NOT fit
        let fp16_config = Lfm2WasmConfig {
            quantization: Apr2Quantization::F16,
            max_context: 8000,
            sliding_window: None,
            ..Default::default()
        };
        assert!(
            !fp16_config.fits_in_wasm(&config),
            "fp16 with 8K context should NOT fit in WASM"
        );
    }

    #[test]
    fn test_ffn_activation_roundtrip() {
        for act in [
            FfnActivation::Gelu,
            FfnActivation::Silu,
            FfnActivation::Swiglu,
            FfnActivation::Relu,
        ] {
            let byte = act as u8;
            let parsed = FfnActivation::try_from(byte).expect("should parse");
            assert_eq!(parsed, act);
        }
    }

    // =========================================================================
    // APR2 Tensor Descriptor Tests
    // =========================================================================

    #[test]
    fn test_apr2_tensor_descriptor_new() {
        let desc = Apr2TensorDescriptor::new(
            "model.embed_tokens.weight",
            &[65536, 2048],
            Apr2Quantization::F32,
            0,
            65536 * 2048 * 4,
        );

        assert_eq!(desc.name, "model.embed_tokens.weight");
        assert_eq!(desc.shape(), &[65536, 2048]);
        assert_eq!(desc.n_dims, 2);
        assert_eq!(desc.dtype, Apr2Quantization::F32);
        assert_eq!(desc.n_elements, 65536 * 2048);
    }

    #[test]
    fn test_apr2_tensor_descriptor_roundtrip() {
        let desc = Apr2TensorDescriptor::new(
            "layer.0.self_attn.q_proj.weight",
            &[2048, 2048],
            Apr2Quantization::Int8,
            1000,
            2048 * 2048,
        );

        let bytes = desc.to_bytes();
        assert_eq!(bytes.len(), Apr2TensorDescriptor::ENTRY_SIZE);

        let parsed = Apr2TensorDescriptor::from_bytes(&bytes).expect("should parse");

        assert_eq!(parsed.name, desc.name);
        assert_eq!(parsed.shape(), desc.shape());
        assert_eq!(parsed.n_dims, desc.n_dims);
        assert_eq!(parsed.dtype, desc.dtype);
        assert_eq!(parsed.offset, desc.offset);
        assert_eq!(parsed.size, desc.size);
        assert_eq!(parsed.n_elements, desc.n_elements);
    }

    #[test]
    fn test_apr2_tensor_descriptor_4d() {
        let desc = Apr2TensorDescriptor::new(
            "conv.weight",
            &[64, 3, 7, 7],
            Apr2Quantization::F16,
            0,
            64 * 3 * 7 * 7 * 2,
        );

        assert_eq!(desc.n_dims, 4);
        assert_eq!(desc.shape(), &[64, 3, 7, 7]);
        assert_eq!(desc.n_elements, 64 * 3 * 7 * 7);
    }

    // =========================================================================
    // APR2 Writer Tests
    // =========================================================================

    #[test]
    fn test_apr2_writer_new() {
        let config = Lfm2Config::lfm2_2_6b();
        let quant = QuantConfig::int4_awq(128);
        let writer = Apr2Writer::lfm2(config, quant);

        assert_eq!(writer.n_tensors(), 0);
    }

    #[test]
    fn test_apr2_writer_add_tensor() {
        let config = Lfm2Config::lfm2_2_6b();
        let quant = QuantConfig::default();
        let mut writer = Apr2Writer::lfm2(config, quant);

        writer.add_f32("test.weight", vec![4, 4], &[0.0f32; 16]);
        assert_eq!(writer.n_tensors(), 1);

        writer.add_int8_quantized("test.bias", vec![4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(writer.n_tensors(), 2);
    }

    #[test]
    fn test_apr2_writer_to_bytes() {
        let config = Lfm2Config::lfm2_2_6b();
        let quant = QuantConfig::default();
        let mut writer = Apr2Writer::lfm2(config, quant);

        writer.add_f32("embed", vec![4], &[1.0, 2.0, 3.0, 4.0]);

        let bytes = writer.to_bytes().expect("should serialize");

        // Check magic
        assert_eq!(&bytes[0..4], &MAGIC_APR2);
    }

    // =========================================================================
    // APR2 Reader Tests
    // =========================================================================

    #[test]
    fn test_apr2_reader_roundtrip() {
        let config = Lfm2Config::lfm2_2_6b();
        let quant = QuantConfig::default();
        let mut writer = Apr2Writer::lfm2(config, quant);

        // Add some test tensors
        let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
        writer.add_f32("layer.0.weight", vec![2, 2], &test_data);
        writer.add_f32("layer.1.weight", vec![4], &[5.0, 6.0, 7.0, 8.0]);

        let bytes = writer.to_bytes().expect("should serialize");
        let reader = Apr2Reader::new(bytes).expect("should parse");

        // Check header
        assert_eq!(reader.header.family, ModelFamily::Lfm2);
        assert_eq!(reader.n_tensors(), 2);

        // Check tensors
        let tensor0 = reader.find_tensor("layer.0.weight").expect("should find");
        assert_eq!(tensor0.shape(), &[2, 2]);
        assert_eq!(tensor0.dtype, Apr2Quantization::F32);

        let tensor1 = reader.find_tensor("layer.1.weight").expect("should find");
        assert_eq!(tensor1.shape(), &[4]);

        // Check data
        let data0 = reader
            .load_tensor_f32("layer.0.weight")
            .expect("should load");
        assert_eq!(data0, test_data);
    }

    #[test]
    fn test_apr2_reader_invalid_magic() {
        let data = vec![b'X', b'Y', b'Z', b'W', 0, 0, 0, 0];
        let result = Apr2Reader::new(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr2_reader_too_short() {
        let data = vec![b'A', b'P'];
        let result = Apr2Reader::new(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr2_reader_lfm2_config() {
        let config = Lfm2Config::lfm2_2_6b();
        let quant = QuantConfig::default();
        let writer = Apr2Writer::lfm2(config.clone(), quant);

        let bytes = writer.to_bytes().expect("should serialize");
        let reader = Apr2Reader::new(bytes).expect("should parse");

        let parsed_config = reader.lfm2_config().expect("should get config");
        assert_eq!(parsed_config.hidden_size, config.hidden_size);
        assert_eq!(parsed_config.num_layers, config.num_layers);
        assert_eq!(parsed_config.num_q_heads, config.num_q_heads);
        assert_eq!(parsed_config.num_kv_heads, config.num_kv_heads);
    }

    // =========================================================================
    // APR2 Tensor Data Tests
    // =========================================================================

    #[test]
    fn test_apr2_tensor_data_from_f32() {
        let data = Apr2TensorData::from_f32("test", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);

        assert_eq!(data.name, "test");
        assert_eq!(data.shape, vec![2, 2]);
        assert_eq!(data.dtype, Apr2Quantization::F32);
        assert_eq!(data.n_elements(), 4);
        assert_eq!(data.byte_size(), 16); // 4 floats * 4 bytes
    }

    #[test]
    fn test_apr2_tensor_data_from_int8() {
        let data = Apr2TensorData::from_int8("test", vec![4], &[1, -1, 2, -2]);

        assert_eq!(data.dtype, Apr2Quantization::Int8);
        assert_eq!(data.byte_size(), 4);
    }

    #[test]
    fn test_apr2_tensor_data_quantize_int8() {
        let f32_data = vec![1.0, -1.0, 0.5, -0.5];
        let quantized = Apr2TensorData::quantize_int8("test", vec![4], &f32_data);

        assert_eq!(quantized.dtype, Apr2Quantization::Int8);
        assert_eq!(quantized.byte_size(), 4);

        // Values should be quantized to range [-127, 127]
        assert_eq!(quantized.data[0], 127u8); // 1.0 -> 127
        assert_eq!(quantized.data[1], (-127i8) as u8); // -1.0 -> -127
    }

    // =========================================================================
    // Half/BFloat16 Conversion Tests
    // =========================================================================

    #[test]
    fn test_half_to_f32_zero() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x8000), -0.0);
    }

    #[test]
    fn test_half_to_f32_one() {
        // f16 representation of 1.0: sign=0, exp=15(0x0F), mant=0
        // bits = 0 | 01111 | 0000000000 = 0x3C00
        let one = half_to_f32(0x3C00);
        assert!((one - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_half_to_f32_negative() {
        // f16 representation of -1.0: sign=1, exp=15, mant=0
        // bits = 1 | 01111 | 0000000000 = 0xBC00
        let neg_one = half_to_f32(0xBC00);
        assert!((neg_one + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32() {
        // bf16 1.0 = 0x3F80 (upper 16 bits of f32 1.0)
        let one = bf16_to_f32(0x3F80);
        assert!((one - 1.0).abs() < 1e-6);

        // bf16 -1.0 = 0xBF80
        let neg_one = bf16_to_f32(0xBF80);
        assert!((neg_one + 1.0).abs() < 1e-6);

        // bf16 0.0 = 0x0000
        let zero = bf16_to_f32(0x0000);
        assert_eq!(zero, 0.0);
    }

    // =========================================================================
    // Model Configuration Tests (WAPR-LFM2-009)
    // =========================================================================

    #[test]
    fn test_lfm2_config_llama_7b() {
        let config = Lfm2Config::llama_7b();

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_q_heads, 32);
        assert_eq!(config.num_kv_heads, 32, "LLaMA-1 uses standard MHA");
        assert_eq!(config.intermediate_size, 11008);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.rope_theta, 10_000.0);
        assert_eq!(config.conv_dimension, 0, "LLaMA has no conv layers");
        assert_eq!(config.layer_types.len(), 32);
        assert_eq!(config.gqa_ratio(), 1, "No GQA in LLaMA-1");

        // All layers should be attention without GQA
        for layer_type in &config.layer_types {
            assert!(matches!(
                layer_type,
                LayerType::Attention { use_gqa: false }
            ));
        }
    }

    #[test]
    fn test_lfm2_config_llama2_7b() {
        let config = Lfm2Config::llama2_7b();

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_q_heads, 32);
        assert_eq!(config.num_kv_heads, 8, "LLaMA-2 uses GQA with 4:1 ratio");
        assert_eq!(config.gqa_ratio(), 4);
        assert_eq!(config.layer_types.len(), 32);

        // All layers should be attention with GQA
        for layer_type in &config.layer_types {
            assert!(matches!(layer_type, LayerType::Attention { use_gqa: true }));
        }
    }

    #[test]
    fn test_lfm2_config_whisper_tiny() {
        let config = Lfm2Config::whisper_tiny();

        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.num_q_heads, 6);
        assert_eq!(config.num_kv_heads, 6);
        assert_eq!(config.intermediate_size, 1536, "4x expansion");
        assert_eq!(config.vocab_size, 51865, "Whisper vocab size");
        assert_eq!(config.max_seq_len, 1500);
        assert_eq!(config.gqa_ratio(), 1, "Standard MHA");
    }

    #[test]
    fn test_lfm2_config_whisper_base() {
        let config = Lfm2Config::whisper_base();

        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.num_q_heads, 8);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.intermediate_size, 2048);
        assert_eq!(config.vocab_size, 51865);
    }

    #[test]
    fn test_lfm2_config_whisper_small() {
        let config = Lfm2Config::whisper_small();

        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_q_heads, 12);
        assert_eq!(config.num_kv_heads, 12);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.vocab_size, 51865);
    }

    #[test]
    fn test_lfm2_config_lfm2_2_6b() {
        // Verify the original LFM2 config is correct
        let config = Lfm2Config::lfm2_2_6b();

        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 30);
        assert_eq!(config.num_q_heads, 32);
        assert_eq!(config.num_kv_heads, 8, "LFM2 uses GQA");
        assert_eq!(config.gqa_ratio(), 4);
        assert_eq!(config.intermediate_size, 10752);
        assert_eq!(config.vocab_size, 65536);
        assert_eq!(config.rope_theta, 1_000_000.0, "Long-context RoPE theta");
        assert!(config.conv_dimension > 0, "LFM2 has conv layers");
        assert_eq!(config.max_seq_len, 128000);

        // LFM2 uses hybrid conv/attention pattern
        assert!(config
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::Convolution { .. })));
        assert!(config
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::Attention { .. })));
    }

    #[test]
    fn test_lfm2_config_moonshine_tiny() {
        let config = Lfm2Config::moonshine_tiny();

        assert_eq!(config.hidden_size, 288);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.num_q_heads, 8);
        assert_eq!(config.num_kv_heads, 8, "Moonshine uses MHA (kv_heads == q_heads)");
        assert_eq!(config.intermediate_size, 1152);
        assert_eq!(config.vocab_size, 32768, "SentencePiece vocab");
        assert_eq!(config.gqa_ratio(), 1);
        assert_eq!(config.layer_types.len(), 6);

        for layer_type in &config.layer_types {
            assert!(matches!(layer_type, LayerType::Attention { use_gqa: false }));
        }
    }

    #[test]
    fn test_lfm2_config_moonshine_base() {
        let config = Lfm2Config::moonshine_base();

        assert_eq!(config.hidden_size, 416);
        assert_eq!(config.num_layers, 8);
        assert_eq!(config.num_q_heads, 8);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.intermediate_size, 1664);
        assert_eq!(config.vocab_size, 32768);
        assert_eq!(config.gqa_ratio(), 1);
        assert_eq!(config.layer_types.len(), 8);
    }

    #[test]
    fn test_model_config_head_dim_divisible() {
        // All configs should have head_dim = hidden_size / num_q_heads be a positive integer
        let configs = [
            Lfm2Config::lfm2_2_6b(),
            Lfm2Config::llama_7b(),
            Lfm2Config::llama2_7b(),
            Lfm2Config::whisper_tiny(),
            Lfm2Config::whisper_base(),
            Lfm2Config::whisper_small(),
            Lfm2Config::moonshine_tiny(),
            Lfm2Config::moonshine_base(),
        ];

        for config in configs {
            let head_dim = config.hidden_size / config.num_q_heads;
            assert!(head_dim > 0, "head_dim should be positive");
            assert_eq!(
                config.hidden_size % config.num_q_heads,
                0,
                "hidden_size should be divisible by num_q_heads"
            );
        }
    }

    #[test]
    fn test_model_config_gqa_ratio_valid() {
        // GQA ratio should always be >= 1 and num_q_heads divisible by num_kv_heads
        let configs = [
            Lfm2Config::lfm2_2_6b(),
            Lfm2Config::llama_7b(),
            Lfm2Config::llama2_7b(),
            Lfm2Config::whisper_tiny(),
            Lfm2Config::whisper_base(),
            Lfm2Config::whisper_small(),
            Lfm2Config::moonshine_tiny(),
            Lfm2Config::moonshine_base(),
        ];

        for config in configs {
            assert!(config.gqa_ratio() >= 1, "GQA ratio should be >= 1");
            assert_eq!(
                config.num_q_heads % config.num_kv_heads,
                0,
                "num_q_heads should be divisible by num_kv_heads"
            );
        }
    }
}

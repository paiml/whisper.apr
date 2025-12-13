//! .apr model format
//!
//! Handles reading and writing the optimized .apr model format.
//!
//! # Format Overview
//!
//! The .apr format is designed for efficient streaming from network or disk:
//!
//! ```text
//! ┌─────────────────┐
//! │ Magic (4 bytes) │  "APR1"
//! ├─────────────────┤
//! │ Header          │  Model configuration
//! ├─────────────────┤
//! │ Tensor Index    │  Offset table for weights
//! ├─────────────────┤
//! │ Tensor Data     │  Weight data (optionally compressed)
//! ├─────────────────┤
//! │ CRC32 (4 bytes) │  File integrity checksum
//! └─────────────────┘
//! ```

pub mod checksum;
mod compress;

pub use checksum::{crc32, Crc32};
pub use compress::Decompressor;

use crate::error::{WhisperError, WhisperResult};
use crate::model::ModelConfig;
use crate::ModelType;

/// Magic number for .apr files: "APR1"
pub const MAGIC: [u8; 4] = [b'A', b'P', b'R', b'1'];

/// Current format version
pub const FORMAT_VERSION: u16 = 1;

/// Header size in bytes (after magic)
pub const HEADER_SIZE: usize = 48;

/// Quantization type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Quantization {
    /// 32-bit floating point
    F32 = 0,
    /// 16-bit floating point
    F16 = 1,
    /// 8-bit integer
    Int8 = 2,
    /// 4-bit integer
    Int4 = 3,
}

impl TryFrom<u8> for Quantization {
    type Error = WhisperError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Int8),
            3 => Ok(Self::Int4),
            _ => Err(WhisperError::Format(format!(
                "invalid quantization type: {value}"
            ))),
        }
    }
}

impl Quantization {
    /// Get bytes per element for this quantization type
    #[must_use]
    pub const fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            // Int4 is actually 0.5 bytes, but we handle packing separately
            Self::Int8 | Self::Int4 => 1,
        }
    }
}

/// .apr file header
#[derive(Debug, Clone)]
pub struct AprHeader {
    /// Format version
    pub version: u16,
    /// Model type
    pub model_type: u8,
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
    /// Quantization type
    pub quantization: Quantization,
    /// Whether weights are compressed
    pub compressed: bool,
}

impl AprHeader {
    /// Parse header from raw bytes
    ///
    /// # Arguments
    /// * `data` - Raw header bytes (must be at least HEADER_SIZE bytes)
    ///
    /// # Errors
    /// Returns error if header is invalid
    pub fn parse(data: &[u8]) -> WhisperResult<Self> {
        if data.len() < HEADER_SIZE {
            return Err(WhisperError::Format("header too short".into()));
        }

        let version = u16::from_le_bytes([data[0], data[1]]);
        if version > FORMAT_VERSION {
            return Err(WhisperError::Format(format!(
                "unsupported format version: {version}"
            )));
        }

        let model_type = data[2];
        let quantization = Quantization::try_from(data[3])?;
        let compressed = data[4] != 0;

        // Read u32 values from offset 8 (after 8 bytes of header metadata)
        let n_vocab = read_u32_le(&data[8..12]);
        let n_audio_ctx = read_u32_le(&data[12..16]);
        let n_audio_state = read_u32_le(&data[16..20]);
        let n_audio_head = read_u32_le(&data[20..24]);
        let n_audio_layer = read_u32_le(&data[24..28]);
        let n_text_ctx = read_u32_le(&data[28..32]);
        let n_text_state = read_u32_le(&data[32..36]);
        let n_text_head = read_u32_le(&data[36..40]);
        let n_text_layer = read_u32_le(&data[40..44]);
        let n_mels = read_u32_le(&data[44..48]);

        Ok(Self {
            version,
            model_type,
            n_vocab,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
            n_text_ctx,
            n_text_state,
            n_text_head,
            n_text_layer,
            n_mels,
            quantization,
            compressed,
        })
    }

    /// Serialize header to bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; HEADER_SIZE];

        bytes[0..2].copy_from_slice(&self.version.to_le_bytes());
        bytes[2] = self.model_type;
        bytes[3] = self.quantization as u8;
        bytes[4] = u8::from(self.compressed);
        // bytes[5..8] reserved

        bytes[8..12].copy_from_slice(&self.n_vocab.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.n_audio_ctx.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.n_audio_state.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.n_audio_head.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.n_audio_layer.to_le_bytes());
        bytes[28..32].copy_from_slice(&self.n_text_ctx.to_le_bytes());
        bytes[32..36].copy_from_slice(&self.n_text_state.to_le_bytes());
        bytes[36..40].copy_from_slice(&self.n_text_head.to_le_bytes());
        bytes[40..44].copy_from_slice(&self.n_text_layer.to_le_bytes());
        bytes[44..48].copy_from_slice(&self.n_mels.to_le_bytes());

        bytes
    }

    /// Convert to ModelConfig
    #[must_use]
    pub fn to_model_config(&self) -> ModelConfig {
        let model_type = match self.model_type {
            1 => ModelType::TinyEn,
            2 => ModelType::Base,
            3 => ModelType::BaseEn,
            4 => ModelType::Small,
            5 => ModelType::SmallEn,
            6 => ModelType::Medium,
            7 => ModelType::MediumEn,
            8 => ModelType::Large,
            9 => ModelType::LargeV1,
            10 => ModelType::LargeV2,
            11 => ModelType::LargeV3,
            // 0 and unknown values default to Tiny
            _ => ModelType::Tiny,
        };

        ModelConfig {
            model_type,
            n_vocab: self.n_vocab,
            n_audio_ctx: self.n_audio_ctx,
            n_audio_state: self.n_audio_state,
            n_audio_head: self.n_audio_head,
            n_audio_layer: self.n_audio_layer,
            n_text_ctx: self.n_text_ctx,
            n_text_state: self.n_text_state,
            n_text_head: self.n_text_head,
            n_text_layer: self.n_text_layer,
            n_mels: self.n_mels,
        }
    }

    /// Create header for tiny model
    #[must_use]
    pub fn tiny() -> Self {
        let config = ModelConfig::tiny();
        Self::from_config(&config, Quantization::F32, false)
    }

    /// Create header for base model
    #[must_use]
    pub fn base() -> Self {
        let config = ModelConfig::base();
        Self::from_config(&config, Quantization::F32, false)
    }

    /// Create header from model config
    #[must_use]
    pub fn from_config(config: &ModelConfig, quantization: Quantization, compressed: bool) -> Self {
        let model_type = match config.model_type {
            ModelType::Tiny => 0,
            ModelType::TinyEn => 1,
            ModelType::Base => 2,
            ModelType::BaseEn => 3,
            ModelType::Small => 4,
            ModelType::SmallEn => 5,
            ModelType::Medium => 6,
            ModelType::MediumEn => 7,
            ModelType::Large => 8,
            ModelType::LargeV1 => 9,
            ModelType::LargeV2 => 10,
            ModelType::LargeV3 => 11,
        };

        Self {
            version: FORMAT_VERSION,
            model_type,
            n_vocab: config.n_vocab,
            n_audio_ctx: config.n_audio_ctx,
            n_audio_state: config.n_audio_state,
            n_audio_head: config.n_audio_head,
            n_audio_layer: config.n_audio_layer,
            n_text_ctx: config.n_text_ctx,
            n_text_state: config.n_text_state,
            n_text_head: config.n_text_head,
            n_text_layer: config.n_text_layer,
            n_mels: config.n_mels,
            quantization,
            compressed,
        }
    }
}

/// Tensor descriptor in the index
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    /// Tensor name
    pub name: String,
    /// Offset in file from start of tensor data section
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
    /// Number of elements
    pub n_elements: u64,
    /// Shape (up to 4 dimensions)
    pub shape: [u32; 4],
    /// Number of dimensions
    pub n_dims: u8,
}

impl TensorDescriptor {
    /// Create new tensor descriptor
    #[must_use]
    pub fn new(name: impl Into<String>, shape: &[usize], offset: u64, size: u64) -> Self {
        let mut shape_arr = [0u32; 4];
        let n_dims = shape.len().min(4);
        for (i, &dim) in shape.iter().take(4).enumerate() {
            shape_arr[i] = dim as u32;
        }

        let n_elements = shape.iter().product::<usize>() as u64;

        Self {
            name: name.into(),
            offset,
            size,
            n_elements,
            shape: shape_arr,
            n_dims: n_dims as u8,
        }
    }

    /// Get shape as slice
    #[must_use]
    pub fn shape(&self) -> &[u32] {
        &self.shape[..self.n_dims as usize]
    }

    /// Parse tensor descriptor from raw bytes
    ///
    /// Format (64 bytes total):
    /// - 0..32: name (null-terminated UTF-8)
    /// - 32..40: offset (u64 LE)
    /// - 40..48: size (u64 LE)
    /// - 48..56: n_elements (u64 LE)
    /// - 56..60: shape[0..4] (u32 LE each, only first n_dims valid)
    /// - 60..61: n_dims (u8)
    /// - 61..64: reserved
    ///
    /// # Errors
    /// Returns error if parsing fails
    pub fn parse(data: &[u8]) -> WhisperResult<Self> {
        if data.len() < 64 {
            return Err(WhisperError::Format("tensor descriptor too short".into()));
        }

        // Parse name (null-terminated)
        let name_bytes = &data[0..32];
        let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(32);
        let name = String::from_utf8_lossy(&name_bytes[..name_end]).into_owned();

        // Parse offset, size, n_elements
        let offset = u64::from_le_bytes([
            data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
        ]);
        let size = u64::from_le_bytes([
            data[40], data[41], data[42], data[43], data[44], data[45], data[46], data[47],
        ]);
        let n_elements = u64::from_le_bytes([
            data[48], data[49], data[50], data[51], data[52], data[53], data[54], data[55],
        ]);

        // Parse shape (up to 4 dimensions, packed as u16 each)
        let mut shape = [0u32; 4];
        for i in 0..4 {
            let idx = 56 + i;
            shape[i] = u32::from(data[idx]);
            // For larger dimensions, use extended format
            if i < 2 && data.len() > 62 {
                shape[i] = u32::from_le_bytes([data[56 + i * 2], data[57 + i * 2], 0, 0]);
            }
        }

        let n_dims = data[60];

        Ok(Self {
            name,
            offset,
            size,
            n_elements,
            shape,
            n_dims,
        })
    }

    /// Serialize tensor descriptor to bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; 64];

        // Write name (null-terminated)
        let name_bytes = self.name.as_bytes();
        let name_len = name_bytes.len().min(31);
        bytes[..name_len].copy_from_slice(&name_bytes[..name_len]);
        // bytes[name_len] is already 0 (null terminator)

        // Write offset, size, n_elements
        bytes[32..40].copy_from_slice(&self.offset.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.size.to_le_bytes());
        bytes[48..56].copy_from_slice(&self.n_elements.to_le_bytes());

        // Write shape (simplified: just first byte of each dimension)
        for i in 0..4 {
            bytes[56 + i] = (self.shape[i] & 0xFF) as u8;
        }

        bytes[60] = self.n_dims;

        bytes
    }
}

/// Size of each tensor index entry in bytes
pub const TENSOR_INDEX_ENTRY_SIZE: usize = 64;

/// .apr file reader
#[derive(Debug)]
pub struct AprReader {
    /// Parsed header
    pub header: AprHeader,
    /// Tensor index
    pub tensors: Vec<TensorDescriptor>,
    /// Offset to tensor data section
    tensor_data_offset: usize,
    /// Raw file data reference
    data: Vec<u8>,
}

impl AprReader {
    /// Create reader from file bytes
    ///
    /// # Errors
    /// Returns error if file is invalid
    pub fn new(data: Vec<u8>) -> WhisperResult<Self> {
        validate_magic(&data)?;

        let header = AprHeader::parse(&data[4..])?;

        // For now, assume tensor index starts right after header
        // and data immediately after. This is simplified.
        let tensor_data_offset = 4 + HEADER_SIZE;

        Ok(Self {
            header,
            tensors: Vec::new(),
            tensor_data_offset,
            data,
        })
    }

    /// Create reader from file bytes with tensor index
    ///
    /// # Arguments
    /// * `data` - Raw .apr file bytes
    /// * `n_tensors` - Number of tensors in the index
    ///
    /// # Errors
    /// Returns error if file is invalid
    pub fn with_tensors(data: Vec<u8>, n_tensors: usize) -> WhisperResult<Self> {
        validate_magic(&data)?;

        let header = AprHeader::parse(&data[4..])?;

        // Parse tensor index
        let index_start = 4 + HEADER_SIZE;
        let index_size = n_tensors * TENSOR_INDEX_ENTRY_SIZE;
        let tensor_data_offset = index_start + index_size;

        if data.len() < tensor_data_offset {
            return Err(WhisperError::Format(
                "file too short for tensor index".into(),
            ));
        }

        let mut tensors = Vec::with_capacity(n_tensors);
        for i in 0..n_tensors {
            let entry_start = index_start + i * TENSOR_INDEX_ENTRY_SIZE;
            let entry = &data[entry_start..entry_start + TENSOR_INDEX_ENTRY_SIZE];
            tensors.push(TensorDescriptor::parse(entry)?);
        }

        Ok(Self {
            header,
            tensors,
            tensor_data_offset,
            data,
        })
    }

    /// Read f32 tensor data
    ///
    /// # Errors
    /// Returns error if read fails
    pub fn read_f32_tensor(&self, offset: usize, count: usize) -> WhisperResult<Vec<f32>> {
        let start = self.tensor_data_offset + offset;
        let byte_count = count * 4;
        let end = start + byte_count;

        if end > self.data.len() {
            return Err(WhisperError::Format("tensor data out of bounds".into()));
        }

        let slice = &self.data[start..end];
        let mut result = Vec::with_capacity(count);

        for chunk in slice.chunks_exact(4) {
            result.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        Ok(result)
    }

    /// Get remaining data after header
    #[must_use]
    pub fn tensor_data(&self) -> &[u8] {
        &self.data[self.tensor_data_offset..]
    }

    /// Get total file size
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.data.len()
    }

    /// Find tensor by name
    #[must_use]
    pub fn find_tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Load tensor by name as f32 values
    ///
    /// # Errors
    /// Returns error if tensor not found or read fails
    pub fn load_tensor(&self, name: &str) -> WhisperResult<Vec<f32>> {
        let desc = self
            .find_tensor(name)
            .ok_or_else(|| WhisperError::Format(format!("tensor not found: {name}")))?;

        self.read_f32_tensor(desc.offset as usize, desc.n_elements as usize)
    }

    /// Get number of tensors
    #[must_use]
    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Get tensor data offset
    #[must_use]
    pub fn tensor_data_offset(&self) -> usize {
        self.tensor_data_offset
    }
}

/// Read u32 from little-endian bytes
fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

// =============================================================================
// Model Size Detection and Auto-Configuration (WAPR-074)
// =============================================================================

/// Detected model size based on header parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedModelSize {
    /// Tiny model (39M params): 384-dim, 4 layers
    Tiny,
    /// Base model (74M params): 512-dim, 6 layers
    Base,
    /// Small model (244M params): 768-dim, 12 layers
    Small,
    /// Medium model (769M params): 1024-dim, 24 layers
    Medium,
    /// Large model (1550M params): 1280-dim, 32 layers
    Large,
    /// Unknown model size (non-standard parameters)
    Unknown,
}

impl core::fmt::Display for DetectedModelSize {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Tiny => write!(f, "tiny"),
            Self::Base => write!(f, "base"),
            Self::Small => write!(f, "small"),
            Self::Medium => write!(f, "medium"),
            Self::Large => write!(f, "large"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

impl DetectedModelSize {
    /// Convert to `ModelType` if known
    #[must_use]
    pub const fn to_model_type(&self) -> Option<ModelType> {
        match self {
            Self::Tiny => Some(ModelType::Tiny),
            Self::Base => Some(ModelType::Base),
            Self::Small => Some(ModelType::Small),
            Self::Medium => Some(ModelType::Medium),
            Self::Large => Some(ModelType::Large),
            Self::Unknown => None,
        }
    }
}

/// Detect model size from header parameters
///
/// Uses `n_audio_state` and `n_audio_layer` to identify the model size.
#[must_use]
pub fn detect_model_size(header: &AprHeader) -> DetectedModelSize {
    // Detection based on audio hidden state dimension (most distinctive)
    match (header.n_audio_state, header.n_audio_layer) {
        (384, 4) => DetectedModelSize::Tiny,
        (512, 6) => DetectedModelSize::Base,
        (768, 12) => DetectedModelSize::Small,
        (1024, 24) => DetectedModelSize::Medium,
        (1280, 32) => DetectedModelSize::Large,
        _ => DetectedModelSize::Unknown,
    }
}

/// Auto-configure from header
///
/// Creates a `ModelConfig` using the exact parameters from the header.
/// This preserves any custom configurations while detecting the model type.
#[must_use]
pub fn auto_config_from_header(header: &AprHeader) -> ModelConfig {
    header.to_model_config()
}

/// Estimate model memory usage in MB
///
/// Calculates approximate memory needed based on model parameters
/// and quantization type.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn estimate_model_memory_mb(header: &AprHeader) -> u32 {
    // Estimate parameter count based on architecture
    // Key components:
    // - Embedding: n_vocab * n_text_state
    // - Encoder: n_audio_layer * (4 * n_audio_state^2 + ...)
    // - Decoder: n_text_layer * (4 * n_text_state^2 + ...)

    let vocab = u64::from(header.n_vocab);
    let audio_state = u64::from(header.n_audio_state);
    let audio_layers = u64::from(header.n_audio_layer);
    let text_state = u64::from(header.n_text_state);
    let text_layers = u64::from(header.n_text_layer);

    // Embedding layer
    let embedding_params = vocab * text_state;

    // Encoder attention + FFN per layer
    // Attention: 4 * d^2 (Q, K, V, O projections)
    // FFN: 2 * d * 4d (two linear layers with 4x expansion)
    let encoder_per_layer = 4 * audio_state * audio_state + 2 * audio_state * 4 * audio_state;
    let encoder_params = audio_layers * encoder_per_layer;

    // Decoder attention + cross-attention + FFN per layer
    let decoder_per_layer = 4 * text_state * text_state  // Self-attention
        + 4 * text_state * audio_state  // Cross-attention
        + 2 * text_state * 4 * text_state; // FFN
    let decoder_params = text_layers * decoder_per_layer;

    // Output projection
    let output_params = text_state * vocab;

    // Total parameters
    let total_params = embedding_params + encoder_params + decoder_params + output_params;

    // Bytes per parameter based on quantization
    let bytes_per_param = match header.quantization {
        Quantization::F32 => 4,
        Quantization::F16 => 2,
        Quantization::Int8 | Quantization::Int4 => 1, // Int4 is actually 0.5, but we round up
    };

    // Int4 is actually 0.5 bytes, so divide by 2 for that case
    let total_bytes = if header.quantization == Quantization::Int4 {
        total_params * bytes_per_param / 2
    } else {
        total_params * bytes_per_param
    };

    // Convert to MB
    (total_bytes / (1024 * 1024)) as u32
}

/// Validate .apr file magic number
///
/// # Errors
/// Returns error if magic number is invalid
pub fn validate_magic(data: &[u8]) -> WhisperResult<()> {
    if data.len() < 4 {
        return Err(WhisperError::Format("file too short".into()));
    }

    if data[..4] != MAGIC {
        return Err(WhisperError::Format("invalid magic number".into()));
    }

    Ok(())
}

/// Create a minimal valid .apr file for testing
///
/// Returns bytes for a valid .apr file with the tiny model configuration.
#[must_use]
pub fn create_test_apr() -> Vec<u8> {
    let mut data = Vec::new();

    // Magic
    data.extend_from_slice(&MAGIC);

    // Header
    let header = AprHeader::tiny();
    data.extend_from_slice(&header.to_bytes());

    // Empty tensor data section
    data.extend_from_slice(&[0u8; 16]);

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Magic Tests
    // =========================================================================

    #[test]
    fn test_magic() {
        assert_eq!(MAGIC, [b'A', b'P', b'R', b'1']);
    }

    #[test]
    fn test_validate_magic_valid() {
        let data = [b'A', b'P', b'R', b'1', 0, 0, 0, 0];
        assert!(validate_magic(&data).is_ok());
    }

    #[test]
    fn test_validate_magic_invalid() {
        let data = [b'X', b'Y', b'Z', b'W'];
        assert!(validate_magic(&data).is_err());
    }

    #[test]
    fn test_validate_magic_short() {
        let data = [b'A', b'P'];
        assert!(validate_magic(&data).is_err());
    }

    // =========================================================================
    // Quantization Tests
    // =========================================================================

    #[test]
    fn test_quantization_try_from() {
        assert_eq!(Quantization::try_from(0).ok(), Some(Quantization::F32));
        assert_eq!(Quantization::try_from(1).ok(), Some(Quantization::F16));
        assert_eq!(Quantization::try_from(2).ok(), Some(Quantization::Int8));
        assert_eq!(Quantization::try_from(3).ok(), Some(Quantization::Int4));
        assert!(Quantization::try_from(4).is_err());
    }

    #[test]
    fn test_quantization_bytes_per_element() {
        assert_eq!(Quantization::F32.bytes_per_element(), 4);
        assert_eq!(Quantization::F16.bytes_per_element(), 2);
        assert_eq!(Quantization::Int8.bytes_per_element(), 1);
        assert_eq!(Quantization::Int4.bytes_per_element(), 1);
    }

    // =========================================================================
    // Header Tests
    // =========================================================================

    #[test]
    fn test_header_tiny() {
        let header = AprHeader::tiny();
        assert_eq!(header.version, FORMAT_VERSION);
        assert_eq!(header.n_vocab, 51865);
        assert_eq!(header.n_audio_state, 384);
        assert_eq!(header.n_audio_layer, 4);
    }

    #[test]
    fn test_header_base() {
        let header = AprHeader::base();
        assert_eq!(header.n_audio_state, 512);
        assert_eq!(header.n_audio_layer, 6);
    }

    #[test]
    fn test_header_roundtrip() {
        let original = AprHeader::tiny();
        let bytes = original.to_bytes();
        let parsed = AprHeader::parse(&bytes).expect("parse should succeed");

        assert_eq!(parsed.version, original.version);
        assert_eq!(parsed.n_vocab, original.n_vocab);
        assert_eq!(parsed.n_audio_state, original.n_audio_state);
        assert_eq!(parsed.n_text_state, original.n_text_state);
        assert_eq!(parsed.quantization, original.quantization);
        assert_eq!(parsed.compressed, original.compressed);
    }

    #[test]
    fn test_header_parse_short() {
        let data = [0u8; 10]; // Too short
        assert!(AprHeader::parse(&data).is_err());
    }

    #[test]
    fn test_header_to_model_config() {
        let header = AprHeader::tiny();
        let config = header.to_model_config();

        assert_eq!(config.n_vocab, header.n_vocab);
        assert_eq!(config.n_audio_state, header.n_audio_state);
        assert_eq!(config.n_text_state, header.n_text_state);
    }

    // =========================================================================
    // Tensor Descriptor Tests
    // =========================================================================

    #[test]
    fn test_tensor_descriptor_new() {
        let desc = TensorDescriptor::new("encoder.conv1.weight", &[384, 80, 3], 0, 92160);

        assert_eq!(desc.name, "encoder.conv1.weight");
        assert_eq!(desc.n_elements, 384 * 80 * 3);
        assert_eq!(desc.shape(), &[384, 80, 3]);
    }

    #[test]
    fn test_tensor_descriptor_2d() {
        let desc = TensorDescriptor::new("embedding", &[51865, 384], 0, 1000);

        assert_eq!(desc.n_dims, 2);
        assert_eq!(desc.shape(), &[51865, 384]);
    }

    // =========================================================================
    // Reader Tests
    // =========================================================================

    #[test]
    fn test_apr_reader_new() {
        let data = create_test_apr();
        let reader = AprReader::new(data).expect("reader should succeed");

        assert_eq!(reader.header.version, FORMAT_VERSION);
    }

    #[test]
    fn test_apr_reader_invalid_magic() {
        let mut data = create_test_apr();
        data[0] = b'X'; // Corrupt magic

        assert!(AprReader::new(data).is_err());
    }

    #[test]
    fn test_apr_reader_file_size() {
        let data = create_test_apr();
        let expected_size = data.len();
        let reader = AprReader::new(data).expect("reader should succeed");

        assert_eq!(reader.file_size(), expected_size);
    }

    #[test]
    fn test_read_f32_tensor() {
        // Create test file with known f32 values
        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC);
        data.extend_from_slice(&AprHeader::tiny().to_bytes());

        // Add some f32 test data
        let test_values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        for v in &test_values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let reader = AprReader::new(data).expect("reader should succeed");
        let values = reader.read_f32_tensor(0, 4).expect("read should succeed");

        assert_eq!(values, test_values);
    }

    #[test]
    fn test_read_f32_tensor_out_of_bounds() {
        let data = create_test_apr();
        let reader = AprReader::new(data).expect("reader should succeed");

        // Try to read more data than available
        assert!(reader.read_f32_tensor(0, 1000).is_err());
    }

    // =========================================================================
    // Create Test Apr Tests
    // =========================================================================

    #[test]
    fn test_create_test_apr() {
        let data = create_test_apr();

        // Should be valid
        assert!(validate_magic(&data).is_ok());

        // Should have header
        assert!(data.len() >= 4 + HEADER_SIZE);
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_read_u32_le() {
        let bytes = [0x01, 0x02, 0x03, 0x04];
        assert_eq!(read_u32_le(&bytes), 0x04030201);
    }

    #[test]
    fn test_read_u32_le_zero() {
        let bytes = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(read_u32_le(&bytes), 0);
    }

    #[test]
    fn test_read_u32_le_max() {
        let bytes = [0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(read_u32_le(&bytes), u32::MAX);
    }

    // =========================================================================
    // Tensor Descriptor Parse/Serialize Tests
    // =========================================================================

    #[test]
    fn test_tensor_descriptor_parse() {
        let mut bytes = vec![0u8; 64];

        // Name: "encoder.pe"
        let name = b"encoder.pe";
        bytes[..name.len()].copy_from_slice(name);

        // Offset: 1000
        bytes[32..40].copy_from_slice(&1000u64.to_le_bytes());
        // Size: 2000
        bytes[40..48].copy_from_slice(&2000u64.to_le_bytes());
        // n_elements: 500
        bytes[48..56].copy_from_slice(&500u64.to_le_bytes());
        // n_dims: 2
        bytes[60] = 2;

        let desc = TensorDescriptor::parse(&bytes).expect("parse should succeed");

        assert_eq!(desc.name, "encoder.pe");
        assert_eq!(desc.offset, 1000);
        assert_eq!(desc.size, 2000);
        assert_eq!(desc.n_elements, 500);
        assert_eq!(desc.n_dims, 2);
    }

    #[test]
    fn test_tensor_descriptor_to_bytes() {
        let desc = TensorDescriptor::new("test.weight", &[384, 512], 0, 768000);
        let bytes = desc.to_bytes();

        assert_eq!(bytes.len(), 64);
        assert!(bytes[..4].starts_with(b"test"));
    }

    #[test]
    fn test_tensor_descriptor_parse_short() {
        let bytes = vec![0u8; 32]; // Too short
        let result = TensorDescriptor::parse(&bytes);
        assert!(result.is_err());
    }

    // =========================================================================
    // AprReader Additional Tests
    // =========================================================================

    #[test]
    fn test_apr_reader_find_tensor_empty() {
        let data = create_test_apr();
        let reader = AprReader::new(data).expect("reader should succeed");

        // No tensors loaded
        assert!(reader.find_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_apr_reader_n_tensors() {
        let data = create_test_apr();
        let reader = AprReader::new(data).expect("reader should succeed");

        assert_eq!(reader.n_tensors(), 0);
    }

    #[test]
    fn test_apr_reader_tensor_data_offset() {
        let data = create_test_apr();
        let reader = AprReader::new(data).expect("reader should succeed");

        assert_eq!(reader.tensor_data_offset(), 4 + HEADER_SIZE);
    }

    #[test]
    fn test_apr_reader_load_tensor_not_found() {
        let data = create_test_apr();
        let reader = AprReader::new(data).expect("reader should succeed");

        let result = reader.load_tensor("nonexistent");
        assert!(result.is_err());
    }

    // =========================================================================
    // Model Size Detection Tests (WAPR-074)
    // =========================================================================

    #[test]
    fn test_detect_model_size_tiny() {
        let header = AprHeader::from_config(&ModelConfig::tiny(), Quantization::F32, false);
        let size = detect_model_size(&header);
        assert_eq!(size, DetectedModelSize::Tiny);
    }

    #[test]
    fn test_detect_model_size_base() {
        let header = AprHeader::from_config(&ModelConfig::base(), Quantization::F32, false);
        let size = detect_model_size(&header);
        assert_eq!(size, DetectedModelSize::Base);
    }

    #[test]
    fn test_detect_model_size_small() {
        let header = AprHeader::from_config(&ModelConfig::small(), Quantization::F32, false);
        let size = detect_model_size(&header);
        assert_eq!(size, DetectedModelSize::Small);
    }

    #[test]
    fn test_detect_model_size_medium() {
        let header = AprHeader::from_config(&ModelConfig::medium(), Quantization::F32, false);
        let size = detect_model_size(&header);
        assert_eq!(size, DetectedModelSize::Medium);
    }

    #[test]
    fn test_detect_model_size_large() {
        let header = AprHeader::from_config(&ModelConfig::large(), Quantization::F32, false);
        let size = detect_model_size(&header);
        assert_eq!(size, DetectedModelSize::Large);
    }

    #[test]
    fn test_detect_model_size_unknown_defaults_to_tiny() {
        // Create header with non-standard dimensions
        let mut header = AprHeader::tiny();
        header.n_audio_state = 999; // Non-standard
        let size = detect_model_size(&header);
        assert_eq!(size, DetectedModelSize::Unknown);
    }

    #[test]
    fn test_auto_config_from_header_tiny() {
        let header = AprHeader::from_config(&ModelConfig::tiny(), Quantization::F32, false);
        let config = auto_config_from_header(&header);

        assert_eq!(config.n_audio_state, 384);
        assert_eq!(config.n_audio_layer, 4);
    }

    #[test]
    fn test_auto_config_from_header_medium() {
        let header = AprHeader::from_config(&ModelConfig::medium(), Quantization::F32, false);
        let config = auto_config_from_header(&header);

        assert_eq!(config.n_audio_state, 1024);
        assert_eq!(config.n_audio_layer, 24);
    }

    #[test]
    fn test_auto_config_from_header_large() {
        let header = AprHeader::from_config(&ModelConfig::large(), Quantization::F32, false);
        let config = auto_config_from_header(&header);

        assert_eq!(config.n_audio_state, 1280);
        assert_eq!(config.n_audio_layer, 32);
    }

    #[test]
    fn test_auto_config_preserves_exact_parameters() {
        // Create header with slightly different vocab size
        let mut header = AprHeader::from_config(&ModelConfig::tiny(), Quantization::F32, false);
        header.n_vocab = 52000; // Custom vocab

        let config = auto_config_from_header(&header);
        // Should use the header's actual values
        assert_eq!(config.n_vocab, 52000);
    }

    #[test]
    fn test_detected_model_size_display() {
        assert_eq!(format!("{}", DetectedModelSize::Tiny), "tiny");
        assert_eq!(format!("{}", DetectedModelSize::Base), "base");
        assert_eq!(format!("{}", DetectedModelSize::Small), "small");
        assert_eq!(format!("{}", DetectedModelSize::Medium), "medium");
        assert_eq!(format!("{}", DetectedModelSize::Large), "large");
        assert_eq!(format!("{}", DetectedModelSize::Unknown), "unknown");
    }

    #[test]
    fn test_detected_model_size_to_model_type() {
        assert_eq!(DetectedModelSize::Tiny.to_model_type(), Some(ModelType::Tiny));
        assert_eq!(DetectedModelSize::Base.to_model_type(), Some(ModelType::Base));
        assert_eq!(DetectedModelSize::Small.to_model_type(), Some(ModelType::Small));
        assert_eq!(DetectedModelSize::Medium.to_model_type(), Some(ModelType::Medium));
        assert_eq!(DetectedModelSize::Large.to_model_type(), Some(ModelType::Large));
        assert_eq!(DetectedModelSize::Unknown.to_model_type(), None);
    }

    #[test]
    fn test_model_size_estimation() {
        let tiny = estimate_model_memory_mb(&AprHeader::tiny());
        let base = estimate_model_memory_mb(&AprHeader::base());
        let small = estimate_model_memory_mb(&AprHeader::from_config(&ModelConfig::small(), Quantization::F32, false));
        let medium = estimate_model_memory_mb(&AprHeader::from_config(&ModelConfig::medium(), Quantization::F32, false));
        let large = estimate_model_memory_mb(&AprHeader::from_config(&ModelConfig::large(), Quantization::F32, false));

        // Sizes should be ordered: tiny < base < small < medium < large
        assert!(tiny < base, "tiny={tiny} should be < base={base}");
        assert!(base < small, "base={base} should be < small={small}");
        assert!(small < medium, "small={small} should be < medium={medium}");
        assert!(medium < large, "medium={medium} should be < large={large}");
    }

    #[test]
    fn test_model_size_estimation_quantized_smaller() {
        let fp32_header = AprHeader::from_config(&ModelConfig::medium(), Quantization::F32, false);
        let int4_header = AprHeader::from_config(&ModelConfig::medium(), Quantization::Int4, false);

        let fp32_size = estimate_model_memory_mb(&fp32_header);
        let int4_size = estimate_model_memory_mb(&int4_header);

        // Int4 should use much less memory than fp32
        assert!(int4_size < fp32_size, "int4={int4_size} should be < fp32={fp32_size}");
        let ratio = int4_size as f32 / fp32_size as f32;
        assert!(ratio < 0.5, "Int4 ratio should be < 0.5, got {ratio}");
    }
}

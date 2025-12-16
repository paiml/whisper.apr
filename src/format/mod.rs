//! .apr model format
//!
//! Handles reading and writing the optimized .apr model format.
//!
//! # Format Overview
//!
//! The .apr format is designed for efficient streaming from network or disk.
//! The canonical format implementation is in `aprender::serialization::apr`.
//!
//! ## Aprender Format (recommended for new models)
//!
//! ```text
//! ┌─────────────────┐
//! │ Magic (4 bytes) │  "APR1"
//! ├─────────────────┤
//! │ Metadata Length │  JSON metadata (vocab, filterbank, config)
//! ├─────────────────┤
//! │ Tensor Count    │
//! ├─────────────────┤
//! │ Tensor Index    │  JSON array of descriptors
//! ├─────────────────┤
//! │ Tensor Data     │  Raw weight data
//! ├─────────────────┤
//! │ CRC32 (4 bytes) │  File integrity checksum
//! └─────────────────┘
//! ```
//!
//! ## Example: Creating model with filterbank
//!
//! ```rust,ignore
//! use whisper_apr::format::aprender::{AprWriter, AprReader};
//! use serde_json::json;
//!
//! let mut writer = AprWriter::new();
//! writer.set_metadata("mel_filterbank", json!([0.0, 0.1, ...]));
//! writer.set_metadata("mel_filterbank_shape", json!([80, 201]));
//! writer.add_tensor_f32("encoder.conv1.weight", vec![384, 80, 3], &weights);
//! writer.write("model.apr")?;
//! ```

pub mod checksum;
mod compress;
pub mod validation;

/// Re-export aprender's canonical APR format
pub mod aprender {
    pub use aprender::serialization::apr::{
        AprMetadata, AprReader as AprReaderV2, AprTensorDescriptor, AprWriter as AprWriterV2,
    };
}

pub use checksum::{crc32, Crc32};
pub use compress::Decompressor;
pub use validation::{
    AprValidator, TensorStats, ValidationCheck, ValidationReport, quick_validate,
    validate_apr_bytes,
};

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
    /// Number of tensors in the file (stored in reserved bytes 5..8)
    pub n_tensors: u16,
    /// Whether vocabulary is embedded (byte 7, bit 0)
    pub has_vocab: bool,
    /// Whether mel filterbank is embedded (byte 7, bit 1)
    pub has_filterbank: bool,
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
        // Read n_tensors from reserved bytes 5..7 (u16)
        let n_tensors = u16::from_le_bytes([data[5], data[6]]);
        // Read flags from byte 7 (bit-packed for backward compatibility)
        // bit 0 = has_vocab, bit 1 = has_filterbank
        let flags = data[7];
        let has_vocab = (flags & 0x01) != 0;
        let has_filterbank = (flags & 0x02) != 0;

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
            n_tensors,
            has_vocab,
            has_filterbank,
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
        // bytes[5..7] = n_tensors (u16)
        bytes[5..7].copy_from_slice(&self.n_tensors.to_le_bytes());
        // byte 7 = flags (bit 0 = has_vocab, bit 1 = has_filterbank)
        let flags = u8::from(self.has_vocab) | (u8::from(self.has_filterbank) << 1);
        bytes[7] = flags;

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
            n_tensors: 0,         // Set by writer when serializing
            has_vocab: false,     // Set by writer when vocabulary is added
            has_filterbank: false, // Set by writer when filterbank is added
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
    /// Format (96 bytes total):
    /// - 0..48: name (null-terminated UTF-8)
    /// - 48..56: offset (u64 LE)
    /// - 56..64: size (u64 LE)
    /// - 64..72: n_elements (u64 LE)
    /// - 72..88: shape[0..4] (u32 LE each, 4 bytes per dimension)
    /// - 88..89: n_dims (u8)
    /// - 89..96: reserved
    ///
    /// # Errors
    /// Returns error if parsing fails
    pub fn parse(data: &[u8]) -> WhisperResult<Self> {
        if data.len() < 96 {
            return Err(WhisperError::Format("tensor descriptor too short".into()));
        }

        // Parse name (null-terminated, max 48 bytes)
        let name_bytes = &data[0..48];
        let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(48);
        let name = String::from_utf8_lossy(&name_bytes[..name_end]).into_owned();

        // Parse offset, size, n_elements
        let offset = u64::from_le_bytes([
            data[48], data[49], data[50], data[51], data[52], data[53], data[54], data[55],
        ]);
        let size = u64::from_le_bytes([
            data[56], data[57], data[58], data[59], data[60], data[61], data[62], data[63],
        ]);
        let n_elements = u64::from_le_bytes([
            data[64], data[65], data[66], data[67], data[68], data[69], data[70], data[71],
        ]);

        // Parse shape (up to 4 dimensions, 4 bytes each)
        let mut shape = [0u32; 4];
        for (i, dim) in shape.iter_mut().enumerate() {
            let base = 72 + i * 4;
            *dim = u32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]]);
        }

        let n_dims = data[88];

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
        let mut bytes = vec![0u8; 96];

        // Write name (null-terminated, max 47 chars)
        let name_bytes = self.name.as_bytes();
        let name_len = name_bytes.len().min(47);
        bytes[..name_len].copy_from_slice(&name_bytes[..name_len]);
        // bytes[name_len] is already 0 (null terminator)

        // Write offset, size, n_elements
        bytes[48..56].copy_from_slice(&self.offset.to_le_bytes());
        bytes[56..64].copy_from_slice(&self.size.to_le_bytes());
        bytes[64..72].copy_from_slice(&self.n_elements.to_le_bytes());

        // Write shape (4 bytes per dimension, up to 4 dimensions)
        for (i, &dim) in self.shape.iter().enumerate() {
            let base = 72 + i * 4;
            bytes[base..base + 4].copy_from_slice(&dim.to_le_bytes());
        }

        bytes[88] = self.n_dims;

        bytes
    }
}

/// Size of each tensor index entry in bytes (96 bytes for proper u32 shapes)
pub const TENSOR_INDEX_ENTRY_SIZE: usize = 96;

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
    /// Automatically parses the tensor index using n_tensors from the header.
    ///
    /// # Errors
    /// Returns error if file is invalid
    pub fn new(data: Vec<u8>) -> WhisperResult<Self> {
        validate_magic(&data)?;

        let header = AprHeader::parse(&data[4..])?;
        let n_tensors = header.n_tensors as usize;

        // Parse tensor index if present
        let index_start = 4 + HEADER_SIZE;
        let index_size = n_tensors * TENSOR_INDEX_ENTRY_SIZE;
        let tensor_data_offset = index_start + index_size;

        // For int8 models, scale table comes after tensor index
        let scale_table_size = if header.quantization == Quantization::Int8 {
            n_tensors * 4 // 4 bytes per scale
        } else {
            0
        };
        let tensor_data_offset = tensor_data_offset + scale_table_size;

        if n_tensors > 0 && data.len() < index_start + index_size {
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
    /// Handles both f32 and int8 quantized tensors automatically.
    ///
    /// # Errors
    /// Returns error if tensor not found or read fails
    pub fn load_tensor(&self, name: &str) -> WhisperResult<Vec<f32>> {
        let tensor_idx = self
            .tensors
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| WhisperError::Format(format!("tensor not found: {name}")))?;

        let desc = &self.tensors[tensor_idx];

        if self.header.quantization == Quantization::Int8 {
            // For int8 models, dequantize the tensor
            self.read_int8_tensor_dequantized(
                tensor_idx,
                desc.offset as usize,
                desc.n_elements as usize,
            )
        } else {
            self.read_f32_tensor(desc.offset as usize, desc.n_elements as usize)
        }
    }

    /// Read int8 tensor and dequantize to f32
    fn read_int8_tensor_dequantized(
        &self,
        tensor_idx: usize,
        offset: usize,
        count: usize,
    ) -> WhisperResult<Vec<f32>> {
        let n_tensors = self.tensors.len();

        // Scale table is between tensor index and tensor data
        let index_start = 4 + HEADER_SIZE;
        let index_size = n_tensors * TENSOR_INDEX_ENTRY_SIZE;
        let scale_table_start = index_start + index_size;

        // Read scale for this tensor
        let scale_offset = scale_table_start + tensor_idx * 4;
        if scale_offset + 4 > self.data.len() {
            return Err(WhisperError::Format("scale table out of bounds".into()));
        }
        let scale_bytes = &self.data[scale_offset..scale_offset + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read int8 data
        let start = self.tensor_data_offset + offset;
        let end = start + count;

        if end > self.data.len() {
            return Err(WhisperError::Format("tensor data out of bounds".into()));
        }

        let slice = &self.data[start..end];

        // Dequantize: f32_value = int8_value * scale
        let result: Vec<f32> = slice.iter().map(|&b| (b as i8) as f32 * scale).collect();

        Ok(result)
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

    /// Calculate total tensor data size
    fn total_tensor_data_size(&self) -> usize {
        self.tensors.iter().map(|t| t.size as usize).sum()
    }

    /// Read vocabulary from the file
    ///
    /// Returns None if no vocabulary is embedded or if parsing fails.
    #[must_use]
    pub fn read_vocabulary(&self) -> Option<crate::tokenizer::Vocabulary> {
        if !self.header.has_vocab {
            return None;
        }

        // Vocabulary section comes after tensor data
        // Format: u32 vocab_size + vocab_bytes
        let vocab_section_start = self.tensor_data_offset + self.total_tensor_data_size();

        if vocab_section_start + 4 > self.data.len() {
            return None;
        }

        // Read vocabulary size
        let vocab_size = u32::from_le_bytes([
            self.data[vocab_section_start],
            self.data[vocab_section_start + 1],
            self.data[vocab_section_start + 2],
            self.data[vocab_section_start + 3],
        ]) as usize;

        let vocab_data_start = vocab_section_start + 4;
        let vocab_data_end = vocab_data_start + vocab_size;

        if vocab_data_end > self.data.len() {
            return None;
        }

        // Parse vocabulary
        crate::tokenizer::Vocabulary::from_bytes(&self.data[vocab_data_start..vocab_data_end])
    }

    /// Check if vocabulary is embedded
    #[must_use]
    pub fn has_vocabulary(&self) -> bool {
        self.header.has_vocab
    }

    /// Read mel filterbank from model file if embedded
    ///
    /// Returns the slaney-normalized filterbank that matches OpenAI's implementation.
    pub fn read_mel_filterbank(&self) -> Option<MelFilterbankData> {
        if !self.header.has_filterbank {
            return None;
        }

        // Filterbank section comes after vocab section (if present)
        // Calculate vocab section end position
        let mut filterbank_section_start = self.tensor_data_offset + self.total_tensor_data_size();

        // Skip vocab section if present
        if self.header.has_vocab {
            if filterbank_section_start + 4 > self.data.len() {
                return None;
            }
            let vocab_size = u32::from_le_bytes([
                self.data[filterbank_section_start],
                self.data[filterbank_section_start + 1],
                self.data[filterbank_section_start + 2],
                self.data[filterbank_section_start + 3],
            ]) as usize;
            filterbank_section_start += 4 + vocab_size;
        }

        // Read filterbank section size
        if filterbank_section_start + 4 > self.data.len() {
            return None;
        }

        let filterbank_size = u32::from_le_bytes([
            self.data[filterbank_section_start],
            self.data[filterbank_section_start + 1],
            self.data[filterbank_section_start + 2],
            self.data[filterbank_section_start + 3],
        ]) as usize;

        let filterbank_data_start = filterbank_section_start + 4;
        let filterbank_data_end = filterbank_data_start + filterbank_size;

        if filterbank_data_end > self.data.len() {
            return None;
        }

        // Parse filterbank data
        MelFilterbankData::from_bytes(&self.data[filterbank_data_start..filterbank_data_end]).ok()
    }

    /// Check if mel filterbank is embedded
    #[must_use]
    pub fn has_mel_filterbank(&self) -> bool {
        self.header.has_filterbank
    }
}

/// Read u32 from little-endian bytes
fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

// =============================================================================
// .apr File Writer
// =============================================================================

/// Tensor with name and f32 data for writing
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Tensor name (max 31 chars)
    pub name: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Raw f32 data
    pub data: Vec<f32>,
}

impl TensorData {
    /// Create new tensor data
    #[must_use]
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            shape,
            data,
        }
    }

    /// Number of elements
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.data.len()
    }

    /// Size in bytes (f32 = 4 bytes each)
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len() * 4
    }
}

/// Quantized tensor with name and int8 data for writing
#[derive(Debug, Clone)]
pub struct QuantizedTensorData {
    /// Tensor name (max 31 chars)
    pub name: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Quantized int8 data
    pub data: Vec<i8>,
    /// Scale factor for dequantization: f32_value = int8_value * scale
    pub scale: f32,
}

impl QuantizedTensorData {
    /// Create quantized tensor from f32 data using per-tensor absmax quantization
    #[must_use]
    pub fn from_f32(name: impl Into<String>, shape: Vec<usize>, f32_data: &[f32]) -> Self {
        // Find absmax for scale
        let absmax = f32_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        // Avoid division by zero
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };

        // Quantize
        let data: Vec<i8> = f32_data
            .iter()
            .map(|&v| {
                let q = (v / scale).round();
                q.clamp(-127.0, 127.0) as i8
            })
            .collect();

        Self {
            name: name.into(),
            shape,
            data,
            scale,
        }
    }

    /// Number of elements
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.data.len()
    }

    /// Size in bytes (int8 = 1 byte each + 4 bytes for scale)
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len() + 4 // data + scale
    }

    /// Dequantize to f32
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&q| f32::from(q) * self.scale)
            .collect()
    }
}

/// .apr file writer for creating model files
#[derive(Debug)]
pub struct AprWriter {
    /// File header
    header: AprHeader,
    /// Tensors to write
    tensors: Vec<TensorData>,
    /// Optional vocabulary to embed
    vocabulary: Option<crate::tokenizer::Vocabulary>,
    /// Optional mel filterbank to embed (raw f32 data, flattened)
    mel_filterbank: Option<MelFilterbankData>,
}

/// Mel filterbank data for embedding in .apr files
#[derive(Debug, Clone)]
pub struct MelFilterbankData {
    /// Number of mel bands (80 or 128)
    pub n_mels: u32,
    /// Number of frequency bins (typically 201)
    pub n_freqs: u32,
    /// Raw f32 filterbank data (row-major: n_mels × n_freqs)
    pub data: Vec<f32>,
}

impl MelFilterbankData {
    /// Create new mel filterbank data
    ///
    /// # Panics
    /// Panics if data length doesn't match n_mels × n_freqs
    #[must_use]
    pub fn new(n_mels: u32, n_freqs: u32, data: Vec<f32>) -> Self {
        let expected = (n_mels * n_freqs) as usize;
        assert_eq!(
            data.len(),
            expected,
            "filterbank data length {} doesn't match {}×{}={}",
            data.len(),
            n_mels,
            n_freqs,
            expected
        );
        Self {
            n_mels,
            n_freqs,
            data,
        }
    }

    /// Create mel_80 filterbank (80 bands × 201 freqs)
    #[must_use]
    pub fn mel_80(data: Vec<f32>) -> Self {
        Self::new(80, 201, data)
    }

    /// Create mel_128 filterbank (128 bands × 201 freqs)
    #[must_use]
    pub fn mel_128(data: Vec<f32>) -> Self {
        Self::new(128, 201, data)
    }

    /// Byte size of filterbank section: header(8 bytes) + data
    #[must_use]
    pub fn byte_size(&self) -> usize {
        8 + self.data.len() * 4
    }

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.byte_size());
        bytes.extend_from_slice(&self.n_mels.to_le_bytes());
        bytes.extend_from_slice(&self.n_freqs.to_le_bytes());
        for &v in &self.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Parse from bytes
    ///
    /// # Errors
    /// Returns error if data is too short or malformed
    pub fn from_bytes(data: &[u8]) -> WhisperResult<Self> {
        if data.len() < 8 {
            return Err(WhisperError::Format("filterbank header too short".into()));
        }

        let n_mels = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let n_freqs = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let expected_bytes = (n_mels * n_freqs) as usize * 4;

        if data.len() < 8 + expected_bytes {
            return Err(WhisperError::Format(format!(
                "filterbank data too short: expected {} bytes, got {}",
                8 + expected_bytes,
                data.len()
            )));
        }

        let f32_data: Vec<f32> = data[8..8 + expected_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(Self {
            n_mels,
            n_freqs,
            data: f32_data,
        })
    }
}

impl AprWriter {
    /// Create new writer with header
    #[must_use]
    pub fn new(header: AprHeader) -> Self {
        Self {
            header,
            tensors: Vec::new(),
            vocabulary: None,
            mel_filterbank: None,
        }
    }

    /// Create writer for tiny model
    #[must_use]
    pub fn tiny() -> Self {
        Self::new(AprHeader::tiny())
    }

    /// Create writer for base model
    #[must_use]
    pub fn base() -> Self {
        Self::new(AprHeader::base())
    }

    /// Create writer from model config
    #[must_use]
    pub fn from_config(config: &ModelConfig) -> Self {
        Self::new(AprHeader::from_config(config, Quantization::F32, false))
    }

    /// Add a tensor to the model
    pub fn add_tensor(&mut self, tensor: TensorData) {
        self.tensors.push(tensor);
    }

    /// Add tensor with name, shape, and data
    pub fn add(&mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) {
        self.add_tensor(TensorData::new(name, shape, data));
    }

    /// Number of tensors
    #[must_use]
    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Set vocabulary to embed in the model file
    pub fn set_vocabulary(&mut self, vocab: crate::tokenizer::Vocabulary) {
        self.vocabulary = Some(vocab);
    }

    /// Check if vocabulary is set
    #[must_use]
    pub fn has_vocabulary(&self) -> bool {
        self.vocabulary.is_some()
    }

    /// Set mel filterbank to embed in the model file
    ///
    /// This embeds OpenAI's slaney-normalized mel filterbank directly in the model,
    /// ensuring exact numerical match with the reference implementation.
    pub fn set_mel_filterbank(&mut self, filterbank: MelFilterbankData) {
        self.mel_filterbank = Some(filterbank);
    }

    /// Check if mel filterbank is set
    #[must_use]
    pub fn has_mel_filterbank(&self) -> bool {
        self.mel_filterbank.is_some()
    }

    /// Write to bytes
    ///
    /// # Errors
    /// Returns error if writing fails
    pub fn to_bytes(&self) -> WhisperResult<Vec<u8>> {
        // Calculate sizes
        let index_size = self.tensors.len() * TENSOR_INDEX_ENTRY_SIZE;
        let data_size: usize = self.tensors.iter().map(TensorData::byte_size).sum();
        let vocab_bytes = self.vocabulary.as_ref().map(|v| v.to_bytes());
        let vocab_section_size = vocab_bytes.as_ref().map_or(0, |b| 4 + b.len());
        let filterbank_bytes = self.mel_filterbank.as_ref().map(MelFilterbankData::to_bytes);
        let filterbank_section_size = filterbank_bytes.as_ref().map_or(0, |b| 4 + b.len());
        let total_size = 4 + HEADER_SIZE + index_size + data_size + vocab_section_size + filterbank_section_size + 4;

        let mut bytes = Vec::with_capacity(total_size);

        // 1. Write magic
        bytes.extend_from_slice(&MAGIC);

        // 2. Write header (with n_tensors, has_vocab, has_filterbank set)
        let mut header = self.header.clone();
        header.n_tensors = self.tensors.len() as u16;
        header.has_vocab = self.vocabulary.is_some();
        header.has_filterbank = self.mel_filterbank.is_some();
        bytes.extend_from_slice(&header.to_bytes());

        // 3. Build and write tensor index
        let mut offset: u64 = 0;
        for tensor in &self.tensors {
            let desc = TensorDescriptor::new(
                &tensor.name,
                &tensor.shape,
                offset,
                tensor.byte_size() as u64,
            );
            bytes.extend_from_slice(&desc.to_bytes());
            offset += tensor.byte_size() as u64;
        }

        // 4. Write tensor data
        for tensor in &self.tensors {
            for &value in &tensor.data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }

        // 5. Write vocabulary section if present
        if let Some(vocab_data) = vocab_bytes {
            // Write vocabulary size (u32)
            bytes.extend_from_slice(&(vocab_data.len() as u32).to_le_bytes());
            // Write vocabulary data
            bytes.extend_from_slice(&vocab_data);
        }

        // 6. Write filterbank section if present
        if let Some(fb_data) = filterbank_bytes {
            // Write filterbank size (u32)
            bytes.extend_from_slice(&(fb_data.len() as u32).to_le_bytes());
            // Write filterbank data
            bytes.extend_from_slice(&fb_data);
        }

        // 7. Compute and write CRC32
        let crc = crc32(&bytes);
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

    /// Get header reference
    #[must_use]
    pub const fn header(&self) -> &AprHeader {
        &self.header
    }

    /// Get tensors reference
    #[must_use]
    pub fn tensors(&self) -> &[TensorData] {
        &self.tensors
    }
}

/// .apr file writer for int8 quantized models
#[derive(Debug)]
pub struct AprWriterInt8 {
    /// File header
    header: AprHeader,
    /// Quantized tensors to write
    tensors: Vec<QuantizedTensorData>,
    /// Optional vocabulary to embed
    vocabulary: Option<crate::tokenizer::Vocabulary>,
    /// Optional mel filterbank to embed
    mel_filterbank: Option<MelFilterbankData>,
}

impl AprWriterInt8 {
    /// Create new writer with header
    #[must_use]
    pub fn new(header: AprHeader) -> Self {
        let mut header = header;
        header.quantization = Quantization::Int8;
        Self {
            header,
            tensors: Vec::new(),
            vocabulary: None,
            mel_filterbank: None,
        }
    }

    /// Create writer for tiny model
    #[must_use]
    pub fn tiny() -> Self {
        Self::new(AprHeader::tiny())
    }

    /// Create writer from model config
    #[must_use]
    pub fn from_config(config: &ModelConfig) -> Self {
        Self::new(AprHeader::from_config(config, Quantization::Int8, false))
    }

    /// Add a quantized tensor from f32 data
    pub fn add_tensor_f32(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        let quantized = QuantizedTensorData::from_f32(name, shape, data);
        self.tensors.push(quantized);
    }

    /// Add a pre-quantized tensor
    pub fn add_tensor(&mut self, tensor: QuantizedTensorData) {
        self.tensors.push(tensor);
    }

    /// Number of tensors
    #[must_use]
    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Set vocabulary to embed in the model file
    pub fn set_vocabulary(&mut self, vocab: crate::tokenizer::Vocabulary) {
        self.vocabulary = Some(vocab);
    }

    /// Check if vocabulary is set
    #[must_use]
    pub fn has_vocabulary(&self) -> bool {
        self.vocabulary.is_some()
    }

    /// Set mel filterbank to embed in the model file
    pub fn set_mel_filterbank(&mut self, filterbank: MelFilterbankData) {
        self.mel_filterbank = Some(filterbank);
    }

    /// Check if mel filterbank is set
    #[must_use]
    pub fn has_mel_filterbank(&self) -> bool {
        self.mel_filterbank.is_some()
    }

    /// Write to bytes
    ///
    /// Format for int8:
    /// - magic (4 bytes)
    /// - header (48 bytes)
    /// - tensor index (64 bytes per tensor)
    /// - scale table (4 bytes per tensor)
    /// - tensor data (1 byte per element)
    /// - crc32 (4 bytes)
    ///
    /// # Errors
    /// Returns error if writing fails
    pub fn to_bytes(&self) -> WhisperResult<Vec<u8>> {
        // Calculate sizes
        let index_size = self.tensors.len() * TENSOR_INDEX_ENTRY_SIZE;
        let scale_table_size = self.tensors.len() * 4; // 4 bytes per scale
        let data_size: usize = self.tensors.iter().map(|t| t.data.len()).sum();
        let vocab_bytes = self.vocabulary.as_ref().map(|v| v.to_bytes());
        let vocab_section_size = vocab_bytes.as_ref().map_or(0, |b| 4 + b.len());
        let filterbank_bytes = self.mel_filterbank.as_ref().map(MelFilterbankData::to_bytes);
        let filterbank_section_size = filterbank_bytes.as_ref().map_or(0, |b| 4 + b.len());
        let total_size =
            4 + HEADER_SIZE + index_size + scale_table_size + data_size + vocab_section_size + filterbank_section_size + 4;

        let mut bytes = Vec::with_capacity(total_size);

        // 1. Write magic
        bytes.extend_from_slice(&MAGIC);

        // 2. Write header (with n_tensors, has_vocab, has_filterbank set)
        let mut header = self.header.clone();
        header.n_tensors = self.tensors.len() as u16;
        header.has_vocab = self.vocabulary.is_some();
        header.has_filterbank = self.mel_filterbank.is_some();
        bytes.extend_from_slice(&header.to_bytes());

        // 3. Build and write tensor index
        let mut offset: u64 = 0;

        for tensor in &self.tensors {
            // Create descriptor with adjusted offset
            // We store: scale at scale_table_start + i*4, data at data_start + offset
            let desc = TensorDescriptor {
                name: tensor.name.clone(),
                offset, // Offset relative to data section start
                size: tensor.data.len() as u64,
                n_elements: tensor.n_elements() as u64,
                shape: {
                    let mut shape_arr = [0u32; 4];
                    for (j, &dim) in tensor.shape.iter().take(4).enumerate() {
                        shape_arr[j] = dim as u32;
                    }
                    shape_arr
                },
                n_dims: tensor.shape.len().min(4) as u8,
            };
            bytes.extend_from_slice(&desc.to_bytes());
            offset += tensor.data.len() as u64;
        }

        // 4. Write scale table (f32 per tensor)
        for tensor in &self.tensors {
            bytes.extend_from_slice(&tensor.scale.to_le_bytes());
        }

        // 5. Write tensor data (int8)
        for tensor in &self.tensors {
            // Convert i8 to bytes safely
            let data_bytes: Vec<u8> = tensor.data.iter().map(|&v| v as u8).collect();
            bytes.extend_from_slice(&data_bytes);
        }

        // 6. Write vocabulary section if present
        if let Some(vocab_data) = vocab_bytes {
            // Write vocabulary size (u32)
            bytes.extend_from_slice(&(vocab_data.len() as u32).to_le_bytes());
            // Write vocabulary data
            bytes.extend_from_slice(&vocab_data);
        }

        // 7. Write filterbank section if present
        if let Some(fb_data) = filterbank_bytes {
            // Write filterbank size (u32)
            bytes.extend_from_slice(&(fb_data.len() as u32).to_le_bytes());
            // Write filterbank data
            bytes.extend_from_slice(&fb_data);
        }

        // 8. Compute and write CRC32
        let crc = crc32(&bytes);
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

    /// Get header reference
    #[must_use]
    pub const fn header(&self) -> &AprHeader {
        &self.header
    }

    /// Get tensors reference
    #[must_use]
    pub fn tensors(&self) -> &[QuantizedTensorData] {
        &self.tensors
    }
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
        let mut bytes = vec![0u8; 96];

        // Name: "encoder.pe" (0..48)
        let name = b"encoder.pe";
        bytes[..name.len()].copy_from_slice(name);

        // Offset: 1000 (48..56)
        bytes[48..56].copy_from_slice(&1000u64.to_le_bytes());
        // Size: 2000 (56..64)
        bytes[56..64].copy_from_slice(&2000u64.to_le_bytes());
        // n_elements: 500 (64..72)
        bytes[64..72].copy_from_slice(&500u64.to_le_bytes());
        // n_dims: 2 (88)
        bytes[88] = 2;

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

        assert_eq!(bytes.len(), 96);
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
        assert_eq!(
            DetectedModelSize::Tiny.to_model_type(),
            Some(ModelType::Tiny)
        );
        assert_eq!(
            DetectedModelSize::Base.to_model_type(),
            Some(ModelType::Base)
        );
        assert_eq!(
            DetectedModelSize::Small.to_model_type(),
            Some(ModelType::Small)
        );
        assert_eq!(
            DetectedModelSize::Medium.to_model_type(),
            Some(ModelType::Medium)
        );
        assert_eq!(
            DetectedModelSize::Large.to_model_type(),
            Some(ModelType::Large)
        );
        assert_eq!(DetectedModelSize::Unknown.to_model_type(), None);
    }

    #[test]
    fn test_model_size_estimation() {
        let tiny = estimate_model_memory_mb(&AprHeader::tiny());
        let base = estimate_model_memory_mb(&AprHeader::base());
        let small = estimate_model_memory_mb(&AprHeader::from_config(
            &ModelConfig::small(),
            Quantization::F32,
            false,
        ));
        let medium = estimate_model_memory_mb(&AprHeader::from_config(
            &ModelConfig::medium(),
            Quantization::F32,
            false,
        ));
        let large = estimate_model_memory_mb(&AprHeader::from_config(
            &ModelConfig::large(),
            Quantization::F32,
            false,
        ));

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
        assert!(
            int4_size < fp32_size,
            "int4={int4_size} should be < fp32={fp32_size}"
        );
        let ratio = int4_size as f32 / fp32_size as f32;
        assert!(ratio < 0.5, "Int4 ratio should be < 0.5, got {ratio}");
    }

    // =========================================================================
    // AprWriter Tests
    // =========================================================================

    #[test]
    fn test_apr_writer_new() {
        let writer = AprWriter::tiny();
        assert_eq!(writer.n_tensors(), 0);
        assert_eq!(writer.header().n_audio_state, 384);
    }

    #[test]
    fn test_apr_writer_base() {
        let writer = AprWriter::base();
        assert_eq!(writer.header().n_audio_state, 512);
    }

    #[test]
    fn test_apr_writer_from_config() {
        let config = ModelConfig::small();
        let writer = AprWriter::from_config(&config);
        assert_eq!(writer.header().n_audio_state, 768);
    }

    #[test]
    fn test_apr_writer_add_tensor() {
        let mut writer = AprWriter::tiny();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        writer.add("test.weight", vec![2, 2], data);
        assert_eq!(writer.n_tensors(), 1);
    }

    #[test]
    fn test_apr_writer_add_multiple_tensors() {
        let mut writer = AprWriter::tiny();
        writer.add("encoder.weight", vec![384, 80], vec![0.0; 384 * 80]);
        writer.add("decoder.weight", vec![384, 51865], vec![0.0; 384 * 51865]);
        assert_eq!(writer.n_tensors(), 2);
    }

    #[test]
    fn test_apr_writer_to_bytes() {
        let mut writer = AprWriter::tiny();
        writer.add("test.weight", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

        let bytes = writer.to_bytes().expect("should serialize");

        // Check magic
        assert_eq!(&bytes[0..4], &MAGIC);

        // Check file size: magic(4) + header(48) + index(96) + data(16) + crc(4)
        assert_eq!(bytes.len(), 4 + 48 + 96 + 16 + 4);
    }

    #[test]
    fn test_apr_writer_roundtrip() {
        let mut writer = AprWriter::tiny();
        writer.add("layer.0.weight", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        writer.add("layer.1.weight", vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

        let bytes = writer.to_bytes().expect("should serialize");

        // Read back with AprReader
        let reader = AprReader::with_tensors(bytes, 2).expect("should parse");
        assert_eq!(reader.n_tensors(), 2);
        assert_eq!(reader.header.n_audio_state, 384);

        // Verify tensor names
        assert!(reader.find_tensor("layer.0.weight").is_some());
        assert!(reader.find_tensor("layer.1.weight").is_some());

        // Verify data (first tensor)
        let tensor = reader
            .find_tensor("layer.0.weight")
            .expect("tensor should exist");
        let data = reader
            .read_f32_tensor(tensor.offset as usize, tensor.n_elements as usize)
            .expect("should read");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_data_new() {
        let tensor = TensorData::new("test", vec![3, 4], vec![0.0; 12]);
        assert_eq!(tensor.name, "test");
        assert_eq!(tensor.shape, vec![3, 4]);
        assert_eq!(tensor.n_elements(), 12);
        assert_eq!(tensor.byte_size(), 48);
    }

    #[test]
    fn test_apr_writer_empty() {
        let writer = AprWriter::tiny();
        let bytes = writer.to_bytes().expect("should serialize");

        // magic(4) + header(48) + index(0) + data(0) + crc(4)
        assert_eq!(bytes.len(), 4 + 48 + 0 + 0 + 4);
    }

    #[test]
    fn test_apr_writer_header_accessor() {
        let writer = AprWriter::tiny();
        assert_eq!(writer.header().version, FORMAT_VERSION);
    }

    #[test]
    fn test_apr_writer_tensors_accessor() {
        let mut writer = AprWriter::tiny();
        writer.add("a", vec![2], vec![1.0, 2.0]);
        writer.add("b", vec![3], vec![3.0, 4.0, 5.0]);
        let tensors = writer.tensors();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].name, "a");
        assert_eq!(tensors[1].name, "b");
    }

    // =========================================================================
    // Quantized Tensor Tests
    // =========================================================================

    #[test]
    fn test_quantized_tensor_from_f32() {
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let quantized = QuantizedTensorData::from_f32("test", vec![4], &data);

        assert_eq!(quantized.name, "test");
        assert_eq!(quantized.n_elements(), 4);
        // Scale should be 1.0 / 127 ≈ 0.00787
        assert!((quantized.scale - 1.0 / 127.0).abs() < 0.001);
        // Values should be quantized to [-127, 127]
        assert_eq!(quantized.data[0], 127); // 1.0 -> 127
        assert_eq!(quantized.data[1], -127); // -1.0 -> -127
    }

    #[test]
    fn test_quantized_tensor_dequantize() {
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let quantized = QuantizedTensorData::from_f32("test", vec![4], &data);
        let dequantized = quantized.to_f32();

        // Check roundtrip accuracy (within 1%)
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(
                error < 0.02,
                "Dequantization error too high: orig={orig}, deq={deq}"
            );
        }
    }

    #[test]
    fn test_quantized_tensor_zeros() {
        let data = vec![0.0; 100];
        let quantized = QuantizedTensorData::from_f32("zeros", vec![100], &data);

        // All zeros should remain zeros
        assert!(quantized.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantized_tensor_byte_size() {
        let data = vec![1.0; 1000];
        let quantized = QuantizedTensorData::from_f32("big", vec![1000], &data);

        // 1000 elements * 1 byte + 4 bytes for scale
        assert_eq!(quantized.byte_size(), 1004);
    }

    // =========================================================================
    // AprWriterInt8 Tests
    // =========================================================================

    #[test]
    fn test_apr_writer_int8_new() {
        let writer = AprWriterInt8::tiny();
        assert_eq!(writer.n_tensors(), 0);
        assert_eq!(writer.header().quantization, Quantization::Int8);
    }

    #[test]
    fn test_apr_writer_int8_from_config() {
        let config = ModelConfig::tiny();
        let writer = AprWriterInt8::from_config(&config);
        assert_eq!(writer.header().n_audio_state, 384);
        assert_eq!(writer.header().quantization, Quantization::Int8);
    }

    #[test]
    fn test_apr_writer_int8_add_tensor() {
        let mut writer = AprWriterInt8::tiny();
        writer.add_tensor_f32("test.weight", vec![4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(writer.n_tensors(), 1);
    }

    #[test]
    fn test_apr_writer_int8_to_bytes() {
        let mut writer = AprWriterInt8::tiny();
        writer.add_tensor_f32("test.weight", vec![4], &[1.0, 2.0, 3.0, 4.0]);

        let bytes = writer.to_bytes().expect("should serialize");

        // Check magic
        assert_eq!(&bytes[0..4], &MAGIC);

        // Check quantization flag in header (offset 4+3 = 7 after magic)
        assert_eq!(bytes[4 + 3], Quantization::Int8 as u8);
    }

    #[test]
    fn test_apr_writer_int8_size_reduction() {
        // Same tensor as f32 vs int8
        let data = vec![0.5; 1000];

        let mut f32_writer = AprWriter::tiny();
        f32_writer.add("tensor", vec![1000], data.clone());

        let mut int8_writer = AprWriterInt8::tiny();
        int8_writer.add_tensor_f32("tensor", vec![1000], &data);

        let f32_bytes = f32_writer.to_bytes().expect("f32 serialize");
        let int8_bytes = int8_writer.to_bytes().expect("int8 serialize");

        // Int8 should be significantly smaller (roughly 4x for data portion)
        assert!(
            int8_bytes.len() < f32_bytes.len(),
            "int8={} should be < f32={}",
            int8_bytes.len(),
            f32_bytes.len()
        );

        // More specifically: f32 data = 4000 bytes, int8 data = 1000 bytes + 4 scale
        // So int8 should be roughly 3000 bytes smaller for the data portion
        let savings = f32_bytes.len() - int8_bytes.len();
        assert!(
            savings > 2500,
            "Expected >2500 bytes savings, got {savings}"
        );
    }

    #[test]
    fn test_apr_writer_int8_multiple_tensors() {
        let mut writer = AprWriterInt8::tiny();
        writer.add_tensor_f32("a", vec![100], &[1.0; 100]);
        writer.add_tensor_f32("b", vec![200], &[2.0; 200]);
        writer.add_tensor_f32("c", vec![50], &[0.5; 50]);

        assert_eq!(writer.n_tensors(), 3);

        let bytes = writer.to_bytes().expect("serialize");
        assert!(validate_magic(&bytes).is_ok());
    }

    #[test]
    fn test_apr_writer_int8_header_accessor() {
        let writer = AprWriterInt8::tiny();
        assert_eq!(writer.header().version, FORMAT_VERSION);
        assert_eq!(writer.header().quantization, Quantization::Int8);
    }

    #[test]
    fn test_apr_writer_int8_tensors_accessor() {
        let mut writer = AprWriterInt8::tiny();
        writer.add_tensor_f32("x", vec![2], &[1.0, 2.0]);
        let tensors = writer.tensors();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "x");
    }
}

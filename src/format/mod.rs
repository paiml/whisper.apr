//! APR Model Format (v2)
//!
//! Canonical APR format implementation using `aprender::format::v2`.
//!
//! # Format Overview
//!
//! The APR format is designed for efficient streaming from network or disk
//! with 64-byte alignment for zero-copy mmap access.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (64 bytes, 64-byte aligned)                          │
//! │   - Magic: "APR\0" (4 bytes) - ONE format, no versioning    │
//! │   - Version: major.minor (2 bytes)                          │
//! │   - Flags (2 bytes)                                         │
//! │   - Tensor count (4 bytes)                                  │
//! │   - Metadata offset (8 bytes)                               │
//! │   - Metadata size (4 bytes)                                 │
//! │   - Tensor index offset (8 bytes)                           │
//! │   - Data offset (8 bytes)                                   │
//! │   - Checksum (4 bytes)                                      │
//! │   - Reserved (20 bytes, zero-padded)                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │ JSON Metadata (variable, padded to 64-byte boundary)        │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Tensor Index (sorted by name, 64-byte aligned entries)      │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Tensor Data (each tensor 64-byte aligned)                   │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Footer Checksum (4 bytes)                                   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example: Creating model with filterbank
//!
//! ```rust,ignore
//! use whisper_apr::format::{AprWriter, WhisperMetadata};
//!
//! let mut writer = AprWriter::new();
//! let metadata = WhisperMetadata::tiny();
//! writer.set_metadata(metadata);
//! writer.add_tensor_f32("encoder.conv1.weight", vec![384, 80, 3], &weights);
//! writer.write("model.apr")?;
//! ```

pub mod apr2;
pub mod checksum;
mod compress;
pub mod export;
#[cfg(feature = "cli")]
pub mod gguf_loader;
pub mod safetensors_loader;
pub mod validation;

// Re-export canonical APR v2 format from aprender
pub use aprender::format::v2::{
    align_64, align_up, AprV2Flags, AprV2Header, AprV2Metadata, AprV2Reader, AprV2ReaderRef,
    AprV2Writer, TensorDType, TensorIndexEntry, ALIGNMENT, HEADER_SIZE_V2, LZ4_BLOCK_SIZE,
    MAGIC_V2, MAX_METADATA_SIZE, MAX_TENSOR_NAME_LEN, VERSION_V2,
};

pub use apr2::{
    Apr2Header, Apr2Quantization, Apr2Reader, Apr2TensorData, Apr2TensorDescriptor, Apr2Writer,
    FfnActivation, LayerType, Lfm2Config, Lfm2WasmConfig, ModelFamily, QuantConfig, APR2_VERSION,
    MAGIC_APR2,
};
pub use checksum::{crc32, Crc32};
pub use compress::Decompressor;
#[cfg(feature = "cli")]
pub use gguf_loader::{load_gguf_whisper, map_gguf_whisper_tensor_name};
#[cfg(feature = "cli")]
pub use safetensors_loader::SafeTensorsLoader;
pub use safetensors_loader::{
    map_moonshine_tensor_name, map_tensor_name, ConversionStats, WeightMapping,
};
pub use validation::{
    quick_validate, validate_apr_bytes, AprValidator, TensorStats, ValidationCheck,
    ValidationReport,
};

// Implementation and types
#[path = "impl_generated.rs"]
#[allow(clippy::all)]
mod impl_;

pub use impl_::*;

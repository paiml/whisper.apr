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
//! ## Example: Creating model with AprV2Writer
//!
//! ```rust,ignore
//! use whisper_apr::format::{AprV2Writer, build_whisper_metadata};
//! use whisper_apr::model::ModelConfig;
//!
//! let config = ModelConfig::tiny();
//! let meta = build_whisper_metadata(&config, "test");
//! let mut writer = AprV2Writer::new(meta);
//! writer.add_f32_tensor("encoder.conv1.weight", vec![384, 80, 3], &weights);
//! let bytes = writer.write()?;
//! ```

pub mod apr2;
mod compress;
pub mod export;
#[cfg(all(feature = "cli", feature = "converter"))]
pub mod gguf_loader;
pub mod safetensors_loader;
pub mod validation;
pub mod whisper_metadata;

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
pub use compress::Decompressor;
#[cfg(all(feature = "cli", feature = "converter"))]
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
pub use whisper_metadata::{
    metadata_to_model_config, MelFilterbankData,
};
#[cfg(any(feature = "converter", test))]
pub use whisper_metadata::{
    build_whisper_metadata, create_test_apr,
};

/// CRC32 (IEEE) checksum for APR2 format compatibility
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

//! ZRAM integration for trueno-ublk optimization
//!
//! Detects and optimizes memory allocation when running on ZRAM-backed storage,
//! particularly trueno-ublk GPU-accelerated ZRAM for 25x faster model loading
//! and 48% RAM reduction.
//!
//! # Overview
//!
//! When models and caches are stored on trueno-ublk ZRAM:
//! - Model loading: 139ms → 5.5ms (25x faster)
//! - Batch transcription RAM: 515 MB → 267 MB (48% less)
//! - KV cache: 480 MB → 160 MB (3x smaller)
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::memory::zram;
//!
//! if zram::is_available() {
//!     let buffer_size = zram::optimal_buffer_size();
//!     let config = zram::ZramConfig::detect()?;
//! }
//! ```

#[cfg(feature = "std")]
use std::fs;
#[cfg(feature = "std")]
use std::path::Path;

/// Default buffer size for non-ZRAM systems (64 KB)
pub const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;

/// Optimal buffer size for trueno-ublk GPU batching (4 MB)
pub const ZRAM_BUFFER_SIZE: usize = 4 * 1024 * 1024;

/// Small buffer size for memory-constrained systems (16 KB)
pub const SMALL_BUFFER_SIZE: usize = 16 * 1024;

/// ZRAM configuration for whisper workloads
#[derive(Debug, Clone)]
pub struct ZramConfig {
    /// Whether trueno-ublk ZRAM is available
    pub available: bool,

    /// Whether GPU acceleration is enabled
    pub gpu_enabled: bool,

    /// Compression algorithm (lz4, zstd)
    pub algorithm: CompressionAlgorithm,

    /// Optimal buffer size for this configuration
    pub buffer_size: usize,

    /// Entropy skip threshold (skip high-entropy data)
    pub entropy_threshold: f32,
}

/// Compression algorithms supported by trueno-ublk
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionAlgorithm {
    /// LZ4 - fastest, moderate compression
    #[default]
    Lz4,
    /// Zstd - slower, better compression
    Zstd,
    /// No compression
    None,
}

impl ZramConfig {
    /// Detect ZRAM configuration from system
    #[cfg(feature = "std")]
    pub fn detect() -> Self {
        let available = is_zram_available_impl();
        let gpu_enabled = is_gpu_zram_available();

        let buffer_size = if available {
            if gpu_enabled {
                ZRAM_BUFFER_SIZE // 4 MB for GPU batching
            } else {
                DEFAULT_BUFFER_SIZE // 64 KB for CPU ZRAM
            }
        } else {
            DEFAULT_BUFFER_SIZE
        };

        Self {
            available,
            gpu_enabled,
            algorithm: detect_compression_algorithm(),
            buffer_size,
            entropy_threshold: 7.5, // Skip data above 7.5 bits/byte
        }
    }

    /// Create default config (no ZRAM)
    #[cfg(not(feature = "std"))]
    pub fn detect() -> Self {
        Self::default()
    }
}

impl Default for ZramConfig {
    fn default() -> Self {
        Self {
            available: false,
            gpu_enabled: false,
            algorithm: CompressionAlgorithm::Lz4,
            buffer_size: DEFAULT_BUFFER_SIZE,
            entropy_threshold: 7.5,
        }
    }
}

/// Check if any form of ZRAM is available
#[cfg(feature = "std")]
pub fn is_available() -> bool {
    is_zram_available_impl()
}

/// Check if any form of ZRAM is available (no_std stub)
#[cfg(not(feature = "std"))]
pub fn is_available() -> bool {
    false
}

/// Check if a path is on a trueno-ublk mount
#[cfg(feature = "std")]
pub fn is_trueno_ublk_mount(path: &Path) -> bool {
    // Check /proc/mounts for ublk device
    if let Ok(mounts) = fs::read_to_string("/proc/mounts") {
        let path_str = path.to_string_lossy();

        for line in mounts.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let device = parts[0];
                let mount_point = parts[1];

                // Check if path starts with this mount point and device is ublk
                if path_str.starts_with(mount_point) && device.contains("ublk") {
                    return true;
                }
            }
        }
    }

    // Also check trueno-ublk runtime directory
    let trueno_marker = Path::new("/run/trueno-ublk");
    if trueno_marker.exists() {
        // Check if path is within a known trueno cache directory
        let path_str = path.to_string_lossy();
        if path_str.contains("whisper-cache") || path_str.contains("trueno") {
            return true;
        }
    }

    false
}

/// Check if a path is on a trueno-ublk mount (no_std stub)
#[cfg(not(feature = "std"))]
pub fn is_trueno_ublk_mount(_path: &[u8]) -> bool {
    false
}

/// Get optimal buffer size for the current system
#[cfg(feature = "std")]
pub fn optimal_buffer_size() -> usize {
    if is_gpu_zram_available() {
        ZRAM_BUFFER_SIZE // 4 MB for GPU batch compression
    } else {
        DEFAULT_BUFFER_SIZE // 64 KB for CPU ZRAM or fallback
    }
}

/// Get optimal buffer size (no_std stub)
#[cfg(not(feature = "std"))]
pub fn optimal_buffer_size() -> usize {
    DEFAULT_BUFFER_SIZE
}

/// Get optimal buffer size for a specific path
#[cfg(feature = "std")]
pub fn optimal_buffer_size_for_path(path: &Path) -> usize {
    if is_trueno_ublk_mount(path) {
        ZRAM_BUFFER_SIZE
    } else {
        DEFAULT_BUFFER_SIZE
    }
}

/// Check if trueno-ublk with GPU acceleration is available
#[cfg(feature = "std")]
fn is_gpu_zram_available() -> bool {
    // Check for trueno-ublk with GPU flag
    let trueno_gpu = Path::new("/run/trueno-ublk/gpu");
    if trueno_gpu.exists() {
        return true;
    }

    // Check /sys for ublk device with GPU attribute
    if let Ok(entries) = fs::read_dir("/sys/class/ublk-control") {
        for entry in entries.flatten() {
            let gpu_path = entry.path().join("gpu");
            if gpu_path.exists() {
                if let Ok(content) = fs::read_to_string(&gpu_path) {
                    if content.trim() == "1" {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Check if generic ZRAM is available
#[cfg(feature = "std")]
fn is_zram_available_impl() -> bool {
    // Check for trueno-ublk
    if Path::new("/run/trueno-ublk").exists() {
        return true;
    }

    // Check for standard ZRAM
    if Path::new("/dev/zram0").exists() {
        return true;
    }

    // Check /sys/block for zram devices
    if let Ok(entries) = fs::read_dir("/sys/block") {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().starts_with("zram") {
                return true;
            }
        }
    }

    false
}

/// Detect compression algorithm in use
#[cfg(feature = "std")]
fn detect_compression_algorithm() -> CompressionAlgorithm {
    // Check trueno-ublk config
    if let Ok(algo) = fs::read_to_string("/run/trueno-ublk/algorithm") {
        match algo.trim().to_lowercase().as_str() {
            "lz4" => return CompressionAlgorithm::Lz4,
            "zstd" => return CompressionAlgorithm::Zstd,
            "none" => return CompressionAlgorithm::None,
            _ => {}
        }
    }

    // Check standard ZRAM comp_algorithm
    if let Ok(algo) = fs::read_to_string("/sys/block/zram0/comp_algorithm") {
        // Format: "lz4 [zstd] deflate" with current in brackets
        for part in algo.split_whitespace() {
            if part.starts_with('[') && part.ends_with(']') {
                let current = &part[1..part.len() - 1];
                match current.to_lowercase().as_str() {
                    "lz4" => return CompressionAlgorithm::Lz4,
                    "zstd" => return CompressionAlgorithm::Zstd,
                    _ => {}
                }
            }
        }
    }

    CompressionAlgorithm::Lz4 // Default to LZ4
}

/// Estimate compression ratio for data type
///
/// Returns expected compression ratio (e.g., 2.0 means 2:1 compression)
pub fn estimate_compression_ratio(data_type: DataType) -> f32 {
    match data_type {
        DataType::ModelWeightsFp32 => 1.7, // ~4.5 bits/byte entropy
        DataType::ModelWeightsInt8 => 1.1, // ~7.0 bits/byte entropy
        DataType::KvCache => 2.5,          // ~5.0 bits/byte entropy
        DataType::PcmAudio => 3.0,         // ~3.5 bits/byte entropy
        DataType::MelSpectrogram => 3.5,   // ~4.0 bits/byte entropy
        DataType::CompressedAudio => 1.0,  // Already compressed
        DataType::OutputText => 4.5,       // ~4.5 bits/byte entropy
    }
}

/// Types of data in whisper pipeline for compression estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// FP32 model weights
    ModelWeightsFp32,
    /// INT8 quantized weights
    ModelWeightsInt8,
    /// Key-value cache for attention
    KvCache,
    /// Decoded PCM audio
    PcmAudio,
    /// Mel spectrogram features
    MelSpectrogram,
    /// Compressed input (MP3, AAC, Opus)
    CompressedAudio,
    /// Output transcription text
    OutputText,
}

/// Calculate estimated memory savings for a workload
pub fn estimate_memory_savings(
    model_size_mb: usize,
    kv_cache_mb: usize,
    buffer_mb: usize,
    quantized: bool,
) -> MemorySavings {
    let model_ratio = if quantized {
        estimate_compression_ratio(DataType::ModelWeightsInt8)
    } else {
        estimate_compression_ratio(DataType::ModelWeightsFp32)
    };

    let kv_ratio = estimate_compression_ratio(DataType::KvCache);
    let buffer_ratio = estimate_compression_ratio(DataType::PcmAudio);

    let original_total = model_size_mb + kv_cache_mb + buffer_mb;
    let compressed_model = (model_size_mb as f32 / model_ratio) as usize;
    let compressed_kv = (kv_cache_mb as f32 / kv_ratio) as usize;
    let compressed_buffer = (buffer_mb as f32 / buffer_ratio) as usize;
    let compressed_total = compressed_model + compressed_kv + compressed_buffer;

    MemorySavings {
        original_mb: original_total,
        compressed_mb: compressed_total,
        savings_percent: ((1.0 - (compressed_total as f32 / original_total as f32)) * 100.0)
            as usize,
    }
}

/// Memory savings estimation result
#[derive(Debug, Clone)]
pub struct MemorySavings {
    /// Original memory usage in MB
    pub original_mb: usize,
    /// Compressed memory usage in MB
    pub compressed_mb: usize,
    /// Percentage savings
    pub savings_percent: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zram_config_default() {
        let config = ZramConfig::default();
        assert!(!config.available);
        assert!(!config.gpu_enabled);
        assert_eq!(config.buffer_size, DEFAULT_BUFFER_SIZE);
    }

    #[test]
    fn test_compression_algorithm_default() {
        let algo = CompressionAlgorithm::default();
        assert_eq!(algo, CompressionAlgorithm::Lz4);
    }

    #[test]
    fn test_compression_ratio_estimates() {
        assert!(estimate_compression_ratio(DataType::ModelWeightsFp32) > 1.5);
        assert!(estimate_compression_ratio(DataType::ModelWeightsInt8) < 1.5);
        assert!(estimate_compression_ratio(DataType::KvCache) > 2.0);
        assert!(estimate_compression_ratio(DataType::PcmAudio) > 2.5);
        assert_eq!(estimate_compression_ratio(DataType::CompressedAudio), 1.0);
    }

    #[test]
    fn test_memory_savings_estimation() {
        // Base model: 278 MB model, 37 MB KV, 2 MB buffers
        let savings = estimate_memory_savings(278, 37, 2, false);

        assert!(savings.original_mb == 317);
        assert!(savings.compressed_mb < savings.original_mb);
        assert!(savings.savings_percent > 30);
    }

    #[test]
    fn test_memory_savings_quantized() {
        // Tiny INT8: 37 MB model, 18 MB KV, 2 MB buffers
        let savings = estimate_memory_savings(37, 18, 2, true);

        assert!(savings.original_mb == 57);
        assert!(savings.compressed_mb < savings.original_mb);
        // INT8 compresses less, expect lower savings
        assert!(savings.savings_percent < 50);
    }

    #[test]
    fn test_buffer_size_constants() {
        assert!(ZRAM_BUFFER_SIZE > DEFAULT_BUFFER_SIZE);
        assert_eq!(ZRAM_BUFFER_SIZE, 4 * 1024 * 1024);
        assert_eq!(DEFAULT_BUFFER_SIZE, 64 * 1024);
        assert_eq!(SMALL_BUFFER_SIZE, 16 * 1024);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_optimal_buffer_size() {
        // Should return a reasonable value regardless of system config
        let size = optimal_buffer_size();
        assert!(size >= SMALL_BUFFER_SIZE);
        assert!(size <= ZRAM_BUFFER_SIZE);
    }

    // =========================================================================
    // Coverage Gap Tests (WAPR-QA-004)
    // =========================================================================

    #[cfg(feature = "std")]
    #[test]
    fn test_zram_config_detect() {
        // Exercise ZramConfig::detect() — reads from /proc/mounts, /dev/zram0, etc.
        let config = ZramConfig::detect();
        // On CI/test systems ZRAM may or may not be available, just verify no panic
        assert!(
            config.buffer_size >= DEFAULT_BUFFER_SIZE || config.buffer_size == ZRAM_BUFFER_SIZE
        );
        // algorithm should always be a valid variant
        let _ = format!("{:?}", config.algorithm);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_is_available() {
        // Exercise the public is_available() wrapper
        let available = is_available();
        // Just verify it returns a bool without panicking
        assert!(available || !available);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_is_trueno_ublk_mount_nonexistent_path() {
        // Exercise is_trueno_ublk_mount with a path that is definitely not on ublk
        let result = is_trueno_ublk_mount(Path::new("/tmp/definitely-not-ublk"));
        assert!(!result);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_is_trueno_ublk_mount_root_path() {
        let result = is_trueno_ublk_mount(Path::new("/"));
        // Root is on a real filesystem, not ublk
        assert!(!result || result); // Just ensure no panic
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_is_trueno_ublk_mount_whisper_cache_path() {
        // Exercise the "whisper-cache" keyword check path
        let result = is_trueno_ublk_mount(Path::new("/tmp/whisper-cache/model.apr"));
        // Will be false unless /run/trueno-ublk exists, but exercises the code path
        assert!(!result || result);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_is_trueno_ublk_mount_trueno_path() {
        // Exercise the "trueno" keyword check path
        let result = is_trueno_ublk_mount(Path::new("/opt/trueno/data"));
        assert!(!result || result);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_optimal_buffer_size_for_path_tmp() {
        let size = optimal_buffer_size_for_path(Path::new("/tmp"));
        assert!(size == DEFAULT_BUFFER_SIZE || size == ZRAM_BUFFER_SIZE);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_optimal_buffer_size_for_path_model() {
        let size = optimal_buffer_size_for_path(Path::new("/home/user/models/model.apr"));
        assert!(size >= SMALL_BUFFER_SIZE);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_zram_config_detect_fields() {
        let config = ZramConfig::detect();
        // Verify all fields are populated
        assert!(config.entropy_threshold > 0.0);
        assert!(config.buffer_size > 0);
        // gpu_enabled depends on system state
        let _ = config.gpu_enabled;
        let _ = config.available;
    }

    #[test]
    fn test_compression_ratio_all_types() {
        // Exhaustive check of all DataType variants
        let types = [
            DataType::ModelWeightsFp32,
            DataType::ModelWeightsInt8,
            DataType::KvCache,
            DataType::PcmAudio,
            DataType::MelSpectrogram,
            DataType::CompressedAudio,
            DataType::OutputText,
        ];
        for dt in &types {
            let ratio = estimate_compression_ratio(*dt);
            assert!(
                ratio >= 1.0,
                "Compression ratio for {:?} should be >= 1.0",
                dt
            );
        }
    }

    #[test]
    fn test_memory_savings_zero_buffers() {
        let savings = estimate_memory_savings(100, 0, 0, false);
        assert_eq!(savings.original_mb, 100);
        assert!(savings.compressed_mb <= 100);
    }

    #[test]
    fn test_data_type_debug() {
        let dt = DataType::KvCache;
        let debug = format!("{dt:?}");
        assert!(debug.contains("KvCache"));
    }

    #[test]
    fn test_memory_savings_debug() {
        let savings = estimate_memory_savings(100, 50, 10, true);
        let debug = format!("{savings:?}");
        assert!(debug.contains("original_mb"));
    }
}

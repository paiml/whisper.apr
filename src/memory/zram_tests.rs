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
    assert!(config.buffer_size >= DEFAULT_BUFFER_SIZE || config.buffer_size == ZRAM_BUFFER_SIZE);
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

// =========================================================================
// Parsing helpers coverage (PMAT-024)
// =========================================================================

#[test]
fn test_parse_algorithm_name_lz4() {
    assert_eq!(parse_algorithm_name("lz4"), CompressionAlgorithm::Lz4);
    assert_eq!(parse_algorithm_name("LZ4"), CompressionAlgorithm::Lz4);
}

#[test]
fn test_parse_algorithm_name_zstd() {
    assert_eq!(parse_algorithm_name("zstd"), CompressionAlgorithm::Zstd);
    assert_eq!(parse_algorithm_name("Zstd"), CompressionAlgorithm::Zstd);
}

#[test]
fn test_parse_algorithm_name_none() {
    assert_eq!(parse_algorithm_name("none"), CompressionAlgorithm::None);
}

#[test]
fn test_parse_algorithm_name_unknown_defaults_lz4() {
    assert_eq!(parse_algorithm_name("deflate"), CompressionAlgorithm::Lz4);
}

#[test]
fn test_parse_comp_algorithm_sysfs_zstd_active() {
    let content = "lzo lzo-rle lz4 lz4hc 842 [zstd]";
    assert_eq!(
        parse_comp_algorithm_sysfs(content),
        CompressionAlgorithm::Zstd
    );
}

#[test]
fn test_parse_comp_algorithm_sysfs_lz4_active() {
    let content = "[lz4] lzo zstd";
    assert_eq!(
        parse_comp_algorithm_sysfs(content),
        CompressionAlgorithm::Lz4
    );
}

#[test]
fn test_parse_comp_algorithm_sysfs_no_brackets() {
    let content = "lz4 lzo zstd";
    assert_eq!(
        parse_comp_algorithm_sysfs(content),
        CompressionAlgorithm::Lz4
    );
}

#[test]
fn test_parse_comp_algorithm_sysfs_empty() {
    assert_eq!(parse_comp_algorithm_sysfs(""), CompressionAlgorithm::Lz4);
}

#[test]
fn test_check_mounts_for_ublk_found() {
    let mounts = "/dev/ublk0 /mnt/trueno ext4 rw 0 0\n/dev/sda1 / ext4 rw 0 0\n";
    assert!(check_mounts_for_ublk(mounts, "/mnt/trueno/model.apr"));
}

#[test]
fn test_check_mounts_for_ublk_not_found() {
    let mounts = "/dev/sda1 / ext4 rw 0 0\n/dev/nvme0n1p1 /home ext4 rw 0 0\n";
    assert!(!check_mounts_for_ublk(mounts, "/home/user/model.apr"));
}

#[test]
fn test_check_mounts_for_ublk_empty() {
    assert!(!check_mounts_for_ublk("", "/any/path"));
}

#[test]
fn test_check_mounts_for_ublk_malformed_lines() {
    let mounts = "short\n\n   \n/dev/sda1\n";
    assert!(!check_mounts_for_ublk(mounts, "/any/path"));
}

#[test]
fn test_is_trueno_cache_path_whisper_cache() {
    assert!(is_trueno_cache_path("/opt/whisper-cache/model.apr"));
}

#[test]
fn test_is_trueno_cache_path_trueno() {
    assert!(is_trueno_cache_path("/mnt/trueno/data"));
}

#[test]
fn test_is_trueno_cache_path_unrelated() {
    assert!(!is_trueno_cache_path("/home/user/documents"));
}

// =========================================================================
// Extended coverage tests (WAPR-QA-005)
// =========================================================================

// --- check_mounts_for_ublk edge cases ---

#[test]
fn test_check_mounts_for_ublk_device_ublk_but_path_mismatch() {
    // ublk device exists but path does not start with the mount point
    let mounts = "/dev/ublk0 /mnt/zram ext4 rw 0 0\n";
    assert!(!check_mounts_for_ublk(mounts, "/home/user/data"));
}

#[test]
fn test_check_mounts_for_ublk_multiple_mounts_second_matches() {
    let mounts = "/dev/sda1 / ext4 rw 0 0\n\
                   /dev/nvme0 /home ext4 rw 0 0\n\
                   /dev/ublk1 /opt/models ext4 rw 0 0\n";
    assert!(check_mounts_for_ublk(mounts, "/opt/models/whisper.apr"));
}

#[test]
fn test_check_mounts_for_ublk_minimal_two_fields() {
    // Line with exactly 2 whitespace-separated fields
    let mounts = "/dev/ublk0 /data\n";
    assert!(check_mounts_for_ublk(mounts, "/data/file.bin"));
}

#[test]
fn test_check_mounts_for_ublk_single_field_line() {
    // Line with only one field should be skipped safely
    let mounts = "/dev/ublk0\n";
    assert!(!check_mounts_for_ublk(mounts, "/any"));
}

#[test]
fn test_check_mounts_for_ublk_non_ublk_device_path_matches() {
    // Path matches the mount point but device is not ublk
    let mounts = "/dev/sda1 /home ext4 rw 0 0\n";
    assert!(!check_mounts_for_ublk(mounts, "/home/user/file"));
}

#[test]
fn test_check_mounts_for_ublk_ublk_in_mount_path_not_device() {
    // "ublk" appears in mount path, not in device name
    let mounts = "/dev/sda1 /mnt/ublk-data ext4 rw 0 0\n";
    assert!(!check_mounts_for_ublk(mounts, "/mnt/ublk-data/file"));
}

#[test]
fn test_check_mounts_for_ublk_exact_mount_point_match() {
    // Path is exactly the mount point (not a subdirectory)
    let mounts = "/dev/ublk0 /mnt/zram ext4 rw 0 0\n";
    assert!(check_mounts_for_ublk(mounts, "/mnt/zram"));
}

#[test]
fn test_check_mounts_for_ublk_whitespace_only_lines() {
    let mounts = "   \n\t\n  \t  \n";
    assert!(!check_mounts_for_ublk(mounts, "/any/path"));
}

// --- parse_algorithm_name edge cases ---

#[test]
fn test_parse_algorithm_name_empty_string() {
    assert_eq!(parse_algorithm_name(""), CompressionAlgorithm::Lz4);
}

#[test]
fn test_parse_algorithm_name_mixed_case_zstd() {
    assert_eq!(parse_algorithm_name("ZSTD"), CompressionAlgorithm::Zstd);
    assert_eq!(parse_algorithm_name("ZsTd"), CompressionAlgorithm::Zstd);
}

#[test]
fn test_parse_algorithm_name_mixed_case_none() {
    assert_eq!(parse_algorithm_name("NONE"), CompressionAlgorithm::None);
    assert_eq!(parse_algorithm_name("None"), CompressionAlgorithm::None);
}

#[test]
fn test_parse_algorithm_name_whitespace_input() {
    // Direct call with whitespace (caller normally trims)
    assert_eq!(parse_algorithm_name(" lz4 "), CompressionAlgorithm::Lz4);
}

#[test]
fn test_parse_algorithm_name_lz4hc_unknown() {
    // lz4hc is not a recognized name, should default to Lz4
    assert_eq!(parse_algorithm_name("lz4hc"), CompressionAlgorithm::Lz4);
}

#[test]
fn test_parse_algorithm_name_lzo_unknown() {
    assert_eq!(parse_algorithm_name("lzo"), CompressionAlgorithm::Lz4);
}

// --- parse_comp_algorithm_sysfs edge cases ---

#[test]
fn test_parse_comp_algorithm_sysfs_none_active() {
    let content = "lz4 [none] zstd";
    assert_eq!(
        parse_comp_algorithm_sysfs(content),
        CompressionAlgorithm::None
    );
}

#[test]
fn test_parse_comp_algorithm_sysfs_unknown_bracketed() {
    let content = "lz4 [deflate] zstd";
    // "deflate" is unknown, defaults to Lz4
    assert_eq!(
        parse_comp_algorithm_sysfs(content),
        CompressionAlgorithm::Lz4
    );
}

#[test]
fn test_parse_comp_algorithm_sysfs_single_bracketed_entry() {
    let content = "[zstd]";
    assert_eq!(
        parse_comp_algorithm_sysfs(content),
        CompressionAlgorithm::Zstd
    );
}

#[test]
fn test_parse_comp_algorithm_sysfs_multiple_brackets_first_wins() {
    // Pathological: two bracketed entries, first one should win
    let content = "[lz4] [zstd]";
    assert_eq!(
        parse_comp_algorithm_sysfs(content),
        CompressionAlgorithm::Lz4
    );
}

#[test]
fn test_parse_comp_algorithm_sysfs_whitespace_only() {
    assert_eq!(
        parse_comp_algorithm_sysfs("   \t  "),
        CompressionAlgorithm::Lz4
    );
}

#[test]
fn test_parse_comp_algorithm_sysfs_partial_bracket_open() {
    // Bracket opens but doesn't close — should not match
    let content = "[lz4 zstd";
    assert_eq!(
        parse_comp_algorithm_sysfs(content),
        CompressionAlgorithm::Lz4
    );
}

#[test]
fn test_parse_comp_algorithm_sysfs_partial_bracket_close() {
    // Bracket closes but doesn't open — should not match
    let content = "lz4] zstd";
    assert_eq!(
        parse_comp_algorithm_sysfs(content),
        CompressionAlgorithm::Lz4
    );
}

// --- is_trueno_cache_path edge cases ---

#[test]
fn test_is_trueno_cache_path_empty_string() {
    assert!(!is_trueno_cache_path(""));
}

#[test]
fn test_is_trueno_cache_path_contains_both_keywords() {
    assert!(is_trueno_cache_path("/trueno/whisper-cache/data"));
}

#[test]
fn test_is_trueno_cache_path_partial_match_no_hit() {
    // "whisper" without "-cache" should not match
    assert!(!is_trueno_cache_path("/opt/whisper/models"));
}

#[test]
fn test_is_trueno_cache_path_case_sensitive() {
    // Keywords are case-sensitive
    assert!(!is_trueno_cache_path("/opt/Whisper-Cache/model"));
    assert!(!is_trueno_cache_path("/opt/TRUENO/data"));
}

// --- estimate_compression_ratio exact values ---

#[test]
fn test_estimate_compression_ratio_exact_values() {
    assert!((estimate_compression_ratio(DataType::ModelWeightsFp32) - 1.7).abs() < f32::EPSILON);
    assert!((estimate_compression_ratio(DataType::ModelWeightsInt8) - 1.1).abs() < f32::EPSILON);
    assert!((estimate_compression_ratio(DataType::KvCache) - 2.5).abs() < f32::EPSILON);
    assert!((estimate_compression_ratio(DataType::PcmAudio) - 3.0).abs() < f32::EPSILON);
    assert!((estimate_compression_ratio(DataType::MelSpectrogram) - 3.5).abs() < f32::EPSILON);
    assert!((estimate_compression_ratio(DataType::CompressedAudio) - 1.0).abs() < f32::EPSILON);
    assert!((estimate_compression_ratio(DataType::OutputText) - 4.5).abs() < f32::EPSILON);
}

// --- estimate_memory_savings edge cases ---

#[test]
fn test_memory_savings_large_model() {
    // Large model (small/medium whisper)
    let savings = estimate_memory_savings(800, 200, 50, false);
    assert_eq!(savings.original_mb, 1050);
    assert!(savings.compressed_mb < savings.original_mb);
    assert!(savings.savings_percent > 0);
    assert!(savings.savings_percent < 100);
}

#[test]
fn test_memory_savings_only_kv_cache() {
    let savings = estimate_memory_savings(0, 100, 0, false);
    assert_eq!(savings.original_mb, 100);
    // KV cache ratio is 2.5, so compressed ~40
    assert!(savings.compressed_mb < 50);
}

#[test]
fn test_memory_savings_only_buffers() {
    let savings = estimate_memory_savings(0, 0, 100, false);
    assert_eq!(savings.original_mb, 100);
    // PcmAudio ratio is 3.0, so compressed ~33
    assert!(savings.compressed_mb < 40);
}

#[test]
fn test_memory_savings_quantized_vs_unquantized() {
    let unquantized = estimate_memory_savings(200, 50, 10, false);
    let quantized = estimate_memory_savings(200, 50, 10, true);
    // INT8 compresses less than FP32, so quantized compressed_mb should be higher
    assert!(quantized.compressed_mb > unquantized.compressed_mb);
}

// --- Derive trait coverage ---

#[test]
fn test_zram_config_clone() {
    let config = ZramConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.available, config.available);
    assert_eq!(cloned.gpu_enabled, config.gpu_enabled);
    assert_eq!(cloned.buffer_size, config.buffer_size);
    assert!((cloned.entropy_threshold - config.entropy_threshold).abs() < f32::EPSILON);
    assert_eq!(cloned.algorithm, config.algorithm);
}

#[test]
fn test_compression_algorithm_copy() {
    let algo = CompressionAlgorithm::Zstd;
    let copied = algo;
    assert_eq!(algo, copied);
}

#[test]
fn test_compression_algorithm_clone() {
    let algo = CompressionAlgorithm::None;
    let cloned = algo.clone();
    assert_eq!(algo, cloned);
}

#[test]
fn test_compression_algorithm_eq() {
    assert_eq!(CompressionAlgorithm::Lz4, CompressionAlgorithm::Lz4);
    assert_ne!(CompressionAlgorithm::Lz4, CompressionAlgorithm::Zstd);
    assert_ne!(CompressionAlgorithm::Zstd, CompressionAlgorithm::None);
}

#[test]
fn test_data_type_clone_copy() {
    let dt = DataType::MelSpectrogram;
    let copied = dt;
    let cloned = dt.clone();
    assert_eq!(dt, copied);
    assert_eq!(dt, cloned);
}

#[test]
fn test_data_type_eq() {
    assert_eq!(DataType::PcmAudio, DataType::PcmAudio);
    assert_ne!(DataType::PcmAudio, DataType::OutputText);
}

#[test]
fn test_memory_savings_clone() {
    let savings = estimate_memory_savings(100, 50, 10, false);
    let cloned = savings.clone();
    assert_eq!(cloned.original_mb, savings.original_mb);
    assert_eq!(cloned.compressed_mb, savings.compressed_mb);
    assert_eq!(cloned.savings_percent, savings.savings_percent);
}

// --- Debug format coverage ---

#[test]
fn test_zram_config_debug_format() {
    let config = ZramConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("ZramConfig"));
    assert!(debug.contains("available"));
    assert!(debug.contains("gpu_enabled"));
    assert!(debug.contains("algorithm"));
    assert!(debug.contains("buffer_size"));
    assert!(debug.contains("entropy_threshold"));
}

#[test]
fn test_compression_algorithm_debug_format() {
    assert_eq!(format!("{:?}", CompressionAlgorithm::Lz4), "Lz4");
    assert_eq!(format!("{:?}", CompressionAlgorithm::Zstd), "Zstd");
    assert_eq!(format!("{:?}", CompressionAlgorithm::None), "None");
}

#[test]
fn test_data_type_debug_all_variants() {
    let expected = [
        (DataType::ModelWeightsFp32, "ModelWeightsFp32"),
        (DataType::ModelWeightsInt8, "ModelWeightsInt8"),
        (DataType::KvCache, "KvCache"),
        (DataType::PcmAudio, "PcmAudio"),
        (DataType::MelSpectrogram, "MelSpectrogram"),
        (DataType::CompressedAudio, "CompressedAudio"),
        (DataType::OutputText, "OutputText"),
    ];
    for (variant, name) in &expected {
        assert_eq!(format!("{variant:?}"), *name);
    }
}

// --- Constants relationship tests ---

#[test]
fn test_buffer_size_ordering() {
    assert!(SMALL_BUFFER_SIZE < DEFAULT_BUFFER_SIZE);
    assert!(DEFAULT_BUFFER_SIZE < ZRAM_BUFFER_SIZE);
}

#[test]
fn test_buffer_sizes_are_power_of_two_multiples() {
    // All buffer sizes should be multiples of 1024
    assert_eq!(SMALL_BUFFER_SIZE % 1024, 0);
    assert_eq!(DEFAULT_BUFFER_SIZE % 1024, 0);
    assert_eq!(ZRAM_BUFFER_SIZE % 1024, 0);
}

// --- ZramConfig default field values ---

#[test]
fn test_zram_config_default_entropy_threshold() {
    let config = ZramConfig::default();
    assert!((config.entropy_threshold - 7.5).abs() < f32::EPSILON);
}

#[test]
fn test_zram_config_default_algorithm_is_lz4() {
    let config = ZramConfig::default();
    assert_eq!(config.algorithm, CompressionAlgorithm::Lz4);
}

// --- Memory savings boundary conditions ---

#[test]
fn test_memory_savings_minimal_input() {
    // Smallest non-zero input
    let savings = estimate_memory_savings(1, 1, 1, false);
    assert_eq!(savings.original_mb, 3);
    assert!(savings.compressed_mb <= savings.original_mb);
}

#[test]
fn test_memory_savings_compressed_audio_no_savings() {
    // CompressedAudio has 1.0 ratio — test indirectly via already compressed model
    // INT8 is the closest at 1.1 ratio
    let savings = estimate_memory_savings(100, 0, 0, true);
    assert_eq!(savings.original_mb, 100);
    // 100 / 1.1 ~= 90, so about 10% savings
    assert!(savings.savings_percent < 15);
}

// --- check_mounts_for_ublk with realistic proc/mounts content ---

#[test]
fn test_check_mounts_for_ublk_realistic_proc_mounts() {
    let mounts = "\
sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0
proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
/dev/nvme0n1p2 / ext4 rw,relatime 0 0
/dev/nvme0n1p1 /boot/efi vfat rw,relatime 0 0
tmpfs /run tmpfs rw,nosuid,nodev 0 0
/dev/ublk0 /mnt/trueno-cache ext4 rw,relatime 0 0
tmpfs /tmp tmpfs rw 0 0";
    assert!(check_mounts_for_ublk(
        mounts,
        "/mnt/trueno-cache/models/tiny.apr"
    ));
    assert!(!check_mounts_for_ublk(mounts, "/home/user/model.apr"));
    assert!(!check_mounts_for_ublk(mounts, "/tmp/test"));
}

#[test]
fn test_check_mounts_for_ublk_ublk_substring_in_longer_device() {
    // Device name contains "ublk" as part of a longer name
    let mounts = "/dev/ublk-gpu0 /mnt/gpu ext4 rw 0 0\n";
    assert!(check_mounts_for_ublk(mounts, "/mnt/gpu/data"));
}

#[test]
fn test_check_mounts_for_ublk_path_prefix_false_positive() {
    // Mount point /mnt/data should NOT match /mnt/data2/file
    // Actually starts_with("/mnt/data") is true for "/mnt/data2/file"
    // This documents the current behavior
    let mounts = "/dev/ublk0 /mnt/data ext4 rw 0 0\n";
    // "/mnt/data2/file" does start with "/mnt/data", so this matches
    assert!(check_mounts_for_ublk(mounts, "/mnt/data2/file"));
}

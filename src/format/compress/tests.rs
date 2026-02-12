//! Tests for LZ4 compression/decompression

use super::*;

/// Roundtrip helper: compress_store → decompress_block → assert equality
fn assert_store_roundtrip(original: &[u8]) {
    let mut compressor = Compressor::new();
    let mut decompressor = Decompressor::new();
    let compressed = compressor.compress_store(original).expect("compress");
    let decompressed = decompressor
        .decompress_block(compressed, original.len())
        .expect("decompress");
    assert_eq!(decompressed, original);
}

/// Roundtrip helper: compress_block → decompress_block → assert equality
fn assert_block_roundtrip(original: &[u8]) {
    let mut compressor = Compressor::new();
    let mut decompressor = Decompressor::new();
    let compressed = compressor.compress_block(original).expect("compress");
    let decompressed = decompressor
        .decompress_block(compressed, original.len())
        .expect("decompress");
    assert_eq!(decompressed, original);
}

// =========================================================================
// Decompressor Construction Tests
// =========================================================================

#[test]
fn test_block_size() {
    assert_eq!(BLOCK_SIZE, 65536);
}

#[test]
fn test_max_block_size() {
    assert_eq!(MAX_BLOCK_SIZE, 4 * 1024 * 1024);
}

#[test]
fn test_lz4_magic() {
    assert_eq!(LZ4_MAGIC, 0x184D_2204);
}

#[test]
fn test_decompressor_new() {
    let decompressor = Decompressor::new();
    assert!(decompressor.capacity() >= BLOCK_SIZE);
    assert!(decompressor.is_empty());
}

#[test]
fn test_decompressor_with_capacity() {
    let decompressor = Decompressor::with_capacity(1024);
    assert!(decompressor.capacity() >= 1024);
}

#[test]
fn test_decompressor_default() {
    let decompressor = Decompressor::default();
    assert!(decompressor.capacity() >= BLOCK_SIZE);
}

#[test]
fn test_decompressor_reset() {
    let mut decompressor = Decompressor::new();
    // Store some data
    decompressor.buffer.push(1);
    decompressor.buffer.push(2);
    assert_eq!(decompressor.len(), 2);

    decompressor.reset();
    assert!(decompressor.is_empty());
}

// =========================================================================
// Compressor Construction Tests
// =========================================================================

#[test]
fn test_compressor_new() {
    let compressor = Compressor::new();
    assert!(compressor.buffer.capacity() >= BLOCK_SIZE);
}

#[test]
fn test_compressor_default() {
    let compressor = Compressor::default();
    assert!(compressor.buffer.capacity() >= BLOCK_SIZE);
}

// =========================================================================
// Roundtrip Tests (Compress → Decompress)
// =========================================================================

#[test]
fn test_roundtrip_empty() {
    let mut compressor = Compressor::new();
    let _decompressor = Decompressor::new(); // Available for empty roundtrip if needed

    let original: &[u8] = &[];
    let compressed = compressor.compress_store(original).expect("compress");

    // Empty data doesn't produce output
    assert!(compressed.is_empty());
}

#[test]
fn test_roundtrip_single_byte() {
    assert_store_roundtrip(&[0x42u8]);
}

#[test]
fn test_roundtrip_small_data() {
    assert_store_roundtrip(b"Hello, World!");
}

#[test]
fn test_roundtrip_larger_data() {
    let original: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    assert_store_roundtrip(&original);
}

#[test]
fn test_roundtrip_repeated_pattern() {
    assert_store_roundtrip(&vec![0xAB; 256]);
}

#[test]
fn test_roundtrip_binary_data() {
    let original: Vec<u8> = (0u8..=255).collect();
    assert_store_roundtrip(&original);
}

// =========================================================================
// Extended Length Tests
// =========================================================================

#[test]
fn test_roundtrip_long_literal() {
    // Data requiring extended literal length (> 15 bytes)
    let original: Vec<u8> = (0..100).map(|i| i as u8).collect();
    assert_store_roundtrip(&original);
}

#[test]
fn test_roundtrip_very_long_literal() {
    // Data requiring multiple extended length bytes (> 270 bytes)
    let original: Vec<u8> = (0..500).map(|i| (i % 256) as u8).collect();
    assert_store_roundtrip(&original);
}

// =========================================================================
// Error Handling Tests
// =========================================================================

#[test]
fn test_decompress_size_too_large() {
    let mut decompressor = Decompressor::new();
    let compressed = [0x10, 0x00]; // Simple token

    let result = decompressor.decompress_block(&compressed, MAX_BLOCK_SIZE + 1);
    assert!(result.is_err());
    assert!(result
        .expect_err("expected error for oversized output")
        .to_string()
        .contains("exceeds maximum"));
}

#[test]
fn test_decompress_truncated_data() {
    let mut decompressor = Decompressor::new();

    // Token says 1 literal byte but no data follows
    let compressed = [0x10]; // 1 literal, but no literal data
    let result = decompressor.decompress_block(&compressed, 1);
    assert!(result.is_err());
}

#[test]
fn test_decompress_invalid_offset() {
    let mut decompressor = Decompressor::new();

    // Manually crafted: token=0x01 (0 literals, match len 5), offset=0x0000
    // Zero offset is invalid
    let compressed = [0x01, 0x00, 0x00];
    let result = decompressor.decompress_block(&compressed, 5);
    assert!(result.is_err());
    assert!(result
        .expect_err("expected error for zero offset")
        .to_string()
        .contains("invalid offset"));
}

#[test]
fn test_decompress_offset_exceeds_buffer() {
    let mut decompressor = Decompressor::new();

    // Token: 1 literal, then match with offset 100 (but buffer only has 1 byte)
    let compressed = [0x11, 0x41, 0x64, 0x00]; // 1 literal 'A', offset 100
    let result = decompressor.decompress_block(&compressed, 10);
    assert!(result.is_err());
    assert!(result
        .expect_err("expected error for out-of-bounds offset")
        .to_string()
        .contains("offset"));
}

// =========================================================================
// Store Uncompressed Tests
// =========================================================================

#[test]
fn test_store_uncompressed() {
    let mut decompressor = Decompressor::new();
    let data = b"test data";

    let result = decompressor.store_uncompressed(data).expect("store");
    assert_eq!(result, data);
}

#[test]
fn test_store_uncompressed_clears_previous() {
    let mut decompressor = Decompressor::new();

    decompressor.store_uncompressed(b"first").expect("store");
    let result = decompressor.store_uncompressed(b"second").expect("store");

    assert_eq!(result, b"second");
    assert_eq!(decompressor.len(), 6);
}

// =========================================================================
// Match Compression Tests
// =========================================================================

#[test]
fn test_compress_with_matches() {
    // Data with repeated patterns that should compress well
    assert_block_roundtrip(b"ABCDABCDABCDABCD");
}

#[test]
fn test_compress_long_run() {
    let mut compressor = Compressor::new();
    let original: Vec<u8> = vec![0x55; 1000];
    let compressed = compressor.compress_block(&original).expect("compress");
    // Should compress significantly
    assert!(compressed.len() < original.len());
    // Verify roundtrip
    assert_block_roundtrip(&original);
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_decompress_exact_size() {
    assert_store_roundtrip(b"exact");
}

#[test]
fn test_compress_random_data() {
    // Pseudo-random data (hard to compress)
    let mut original = Vec::with_capacity(256);
    let mut state = 12345u32;
    for _ in 0..256 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        original.push((state >> 16) as u8);
    }
    assert_store_roundtrip(&original);
}

// =========================================================================
// Buffer State Tests
// =========================================================================

#[test]
fn test_len_and_is_empty() {
    let mut decompressor = Decompressor::new();

    assert!(decompressor.is_empty());
    assert_eq!(decompressor.len(), 0);

    decompressor.store_uncompressed(b"test").expect("store");

    assert!(!decompressor.is_empty());
    assert_eq!(decompressor.len(), 4);
}

#[test]
fn test_multiple_decompressions() {
    let mut compressor = Compressor::new();
    let mut decompressor = Decompressor::new();

    // First decompression
    let data1 = b"first block";
    let compressed1 = compressor.compress_store(data1).expect("compress");
    let result1 = decompressor
        .decompress_block(compressed1, data1.len())
        .expect("decompress");
    assert_eq!(result1, data1);

    // Second decompression (should work independently)
    let data2 = b"second block";
    let compressed2 = compressor.compress_store(data2).expect("compress");
    let result2 = decompressor
        .decompress_block(compressed2, data2.len())
        .expect("decompress");
    assert_eq!(result2, data2);
}

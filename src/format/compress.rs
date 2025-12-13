//! LZ4 streaming decompression
//!
//! Pure Rust implementation of LZ4 block decompression for WASM compatibility.
//! Supports both raw blocks and framed format.
//!
//! # LZ4 Block Format
//!
//! Each block consists of sequences:
//! - Token byte: high nibble = literal length, low nibble = match length - 4
//! - Optional literal length extension bytes (if literal length == 15)
//! - Literal bytes
//! - Match offset (2 bytes, little-endian)
//! - Optional match length extension bytes (if match length nibble == 15)
//!
//! # Example
//!
//! ```rust,ignore
//! use whisper_apr::format::compress::Decompressor;
//!
//! let mut decompressor = Decompressor::new();
//! let decompressed = decompressor.decompress_block(&compressed_data)?;
//! ```

use crate::error::{WhisperError, WhisperResult};

/// LZ4 block size for streaming decompression (64KB per spec)
pub const BLOCK_SIZE: usize = 64 * 1024;

/// Maximum supported block size (4MB - LZ4 limit)
pub const MAX_BLOCK_SIZE: usize = 4 * 1024 * 1024;

/// LZ4 frame magic number (used for framed format detection)
#[allow(dead_code)]
pub const LZ4_MAGIC: u32 = 0x184D_2204;

/// Streaming decompressor for .apr weights
#[derive(Debug)]
pub struct Decompressor {
    /// Internal buffer for decompressed data
    buffer: Vec<u8>,
}

impl Decompressor {
    /// Create a new decompressor
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(BLOCK_SIZE),
        }
    }

    /// Create decompressor with custom buffer capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Decompress a raw LZ4 block
    ///
    /// # Arguments
    /// * `compressed` - Compressed block data (raw LZ4, no frame header)
    /// * `decompressed_size` - Expected decompressed size (must be known)
    ///
    /// # Returns
    /// Reference to decompressed data in internal buffer
    ///
    /// # Errors
    /// Returns error if decompression fails or data is corrupted
    pub fn decompress_block(
        &mut self,
        compressed: &[u8],
        decompressed_size: usize,
    ) -> WhisperResult<&[u8]> {
        if decompressed_size > MAX_BLOCK_SIZE {
            return Err(WhisperError::Format(format!(
                "decompressed size {decompressed_size} exceeds maximum {MAX_BLOCK_SIZE}"
            )));
        }

        self.buffer.clear();
        self.buffer.reserve(decompressed_size);

        let mut src_pos = 0;
        let src = compressed;

        while self.buffer.len() < decompressed_size {
            if src_pos >= src.len() {
                return Err(WhisperError::Format(
                    "unexpected end of compressed data".into(),
                ));
            }

            // Read token
            let token = src[src_pos];
            src_pos += 1;

            let literal_len = (token >> 4) as usize;
            let match_len_base = (token & 0x0F) as usize;

            // Read extended literal length
            let literal_len = Self::read_extended_length(src, &mut src_pos, literal_len)?;

            // Copy literals
            if literal_len > 0 {
                if src_pos + literal_len > src.len() {
                    return Err(WhisperError::Format("literal overflow".into()));
                }
                self.buffer
                    .extend_from_slice(&src[src_pos..src_pos + literal_len]);
                src_pos += literal_len;
            }

            // Check if we've reached the end
            if self.buffer.len() >= decompressed_size {
                break;
            }

            // Read match offset (little-endian)
            if src_pos + 2 > src.len() {
                return Err(WhisperError::Format("missing match offset".into()));
            }
            let offset = u16::from_le_bytes([src[src_pos], src[src_pos + 1]]) as usize;
            src_pos += 2;

            if offset == 0 {
                return Err(WhisperError::Format("invalid zero offset".into()));
            }
            if offset > self.buffer.len() {
                return Err(WhisperError::Format(format!(
                    "offset {} exceeds buffer length {}",
                    offset,
                    self.buffer.len()
                )));
            }

            // Read extended match length (min match = 4)
            let match_len = Self::read_extended_length(src, &mut src_pos, match_len_base)? + 4;

            // Copy match (can overlap, so copy byte by byte)
            let match_start = self.buffer.len() - offset;
            for i in 0..match_len {
                let byte = self.buffer[match_start + (i % offset)];
                self.buffer.push(byte);
            }
        }

        // Truncate to exact size if we overshot
        self.buffer.truncate(decompressed_size);

        Ok(&self.buffer)
    }

    /// Read extended length value (used when nibble == 15)
    fn read_extended_length(src: &[u8], pos: &mut usize, base: usize) -> WhisperResult<usize> {
        let mut len = base;

        if base == 15 {
            loop {
                if *pos >= src.len() {
                    return Err(WhisperError::Format(
                        "unexpected end reading extended length".into(),
                    ));
                }
                let byte = src[*pos];
                *pos += 1;
                len += byte as usize;
                if byte != 255 {
                    break;
                }
            }
        }

        Ok(len)
    }

    /// Decompress uncompressed data (passthrough for uncompressed blocks)
    ///
    /// # Arguments
    /// * `data` - Uncompressed data
    ///
    /// # Returns
    /// Reference to data in internal buffer
    pub fn store_uncompressed(&mut self, data: &[u8]) -> WhisperResult<&[u8]> {
        self.buffer.clear();
        self.buffer.extend_from_slice(data);
        Ok(&self.buffer)
    }

    /// Get buffer capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Get current buffer length
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Reset decompressor state
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple LZ4 compressor for testing
///
/// This is a minimal implementation for creating test data.
/// Not optimized for production compression.
#[cfg(test)]
#[derive(Debug)]
struct Compressor {
    /// Output buffer
    buffer: Vec<u8>,
}

#[cfg(test)]
impl Compressor {
    /// Create a new compressor
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(BLOCK_SIZE),
        }
    }

    /// Compress data using LZ4 block format
    ///
    /// This is a simple implementation that creates valid LZ4 but may not
    /// achieve optimal compression ratios.
    pub fn compress_block(&mut self, data: &[u8]) -> WhisperResult<&[u8]> {
        self.buffer.clear();

        if data.is_empty() {
            return Ok(&self.buffer);
        }

        // Track position and pending literals
        let mut pos = 0;
        let mut literal_start = 0;

        while pos < data.len() {
            // Find best match in already-processed data
            let (match_offset, match_len) = Self::find_match(data, pos);

            if match_len >= 4 {
                // Found a match - encode pending literals and the match
                let literals = &data[literal_start..pos];
                self.encode_sequence(literals, match_offset, match_len)?;
                pos += match_len;
                literal_start = pos;
            } else {
                // No match - continue to next byte (will be part of literals)
                pos += 1;
            }
        }

        // Encode any remaining literals at the end
        if literal_start < data.len() {
            let remaining = &data[literal_start..];
            self.encode_literals_only(remaining)?;
        }

        Ok(&self.buffer)
    }

    /// Compress without any LZ4 encoding (store as literals)
    ///
    /// Creates valid LZ4 that decompresses to original data.
    pub fn compress_store(&mut self, data: &[u8]) -> WhisperResult<&[u8]> {
        self.buffer.clear();

        let mut pos = 0;
        while pos < data.len() {
            // Maximum literal run per token
            let chunk_size = std::cmp::min(data.len() - pos, MAX_BLOCK_SIZE);
            let chunk = &data[pos..pos + chunk_size];

            self.encode_literals_only(chunk)?;
            pos += chunk_size;
        }

        Ok(&self.buffer)
    }

    /// Find best match in window
    fn find_match(data: &[u8], pos: usize) -> (usize, usize) {
        let window_size = std::cmp::min(pos, 65535);
        let max_match = std::cmp::min(data.len() - pos, 255 + 15 + 4);

        let mut best_offset = 0;
        let mut best_len = 0;

        for offset in 1..=window_size {
            let match_start = pos - offset;
            let mut len = 0;

            while pos + len < data.len()
                && len < max_match
                && data[match_start + (len % offset)] == data[pos + len]
            {
                len += 1;
            }

            if len >= 4 && len > best_len {
                best_offset = offset;
                best_len = len;
            }
        }

        (best_offset, best_len)
    }

    /// Encode a sequence with literals and match
    fn encode_sequence(
        &mut self,
        literals: &[u8],
        offset: usize,
        match_len: usize,
    ) -> WhisperResult<()> {
        let literal_len = literals.len();
        let match_len_base = match_len.saturating_sub(4);

        // Token byte
        let lit_nibble = std::cmp::min(literal_len, 15);
        let match_nibble = std::cmp::min(match_len_base, 15);
        let token = ((lit_nibble as u8) << 4) | (match_nibble as u8);
        self.buffer.push(token);

        // Extended literal length
        if literal_len >= 15 {
            self.encode_extended_length(literal_len - 15);
        }

        // Literals
        self.buffer.extend_from_slice(literals);

        // Match offset (little-endian)
        self.buffer
            .extend_from_slice(&(offset as u16).to_le_bytes());

        // Extended match length
        if match_len_base >= 15 {
            self.encode_extended_length(match_len_base - 15);
        }

        Ok(())
    }

    /// Encode literals only (no match)
    fn encode_literals_only(&mut self, literals: &[u8]) -> WhisperResult<()> {
        if literals.is_empty() {
            return Ok(());
        }

        let literal_len = literals.len();

        // Token byte (no match, so match nibble doesn't matter for last sequence)
        let lit_nibble = std::cmp::min(literal_len, 15);
        let token = (lit_nibble as u8) << 4;
        self.buffer.push(token);

        // Extended literal length
        if literal_len >= 15 {
            self.encode_extended_length(literal_len - 15);
        }

        // Literals
        self.buffer.extend_from_slice(literals);

        Ok(())
    }

    /// Encode extended length value
    fn encode_extended_length(&mut self, mut extra: usize) {
        while extra >= 255 {
            self.buffer.push(255);
            extra -= 255;
        }
        self.buffer.push(extra as u8);
    }
}

#[cfg(test)]
impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        let original = [0x42u8];
        let compressed = compressor.compress_store(&original).expect("compress");

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_roundtrip_small_data() {
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        let original = b"Hello, World!";
        let compressed = compressor.compress_store(original).expect("compress");

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_roundtrip_larger_data() {
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        // 1KB of data
        let original: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let compressed = compressor.compress_store(&original).expect("compress");

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed, &original[..]);
    }

    #[test]
    fn test_roundtrip_repeated_pattern() {
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        // Highly compressible data
        let original: Vec<u8> = vec![0xAB; 256];
        let compressed = compressor.compress_store(&original).expect("compress");

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed, &original[..]);
    }

    #[test]
    fn test_roundtrip_binary_data() {
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        // Binary data with all byte values
        let original: Vec<u8> = (0u8..=255).collect();
        let compressed = compressor.compress_store(&original).expect("compress");

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed, &original[..]);
    }

    // =========================================================================
    // Extended Length Tests
    // =========================================================================

    #[test]
    fn test_roundtrip_long_literal() {
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        // Data requiring extended literal length (> 15 bytes)
        let original: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let compressed = compressor.compress_store(&original).expect("compress");

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed, &original[..]);
    }

    #[test]
    fn test_roundtrip_very_long_literal() {
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        // Data requiring multiple extended length bytes (> 270 bytes)
        let original: Vec<u8> = (0..500).map(|i| (i % 256) as u8).collect();
        let compressed = compressor.compress_store(&original).expect("compress");

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed, &original[..]);
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
            .contains("invalid zero offset"));
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
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        // Data with repeated patterns that should compress well
        let original = b"ABCDABCDABCDABCD";
        let compressed = compressor.compress_block(original).expect("compress");

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_compress_long_run() {
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        // Long run of same byte - very compressible
        let original: Vec<u8> = vec![0x55; 1000];
        let compressed = compressor.compress_block(&original).expect("compress");

        // Should compress significantly
        assert!(compressed.len() < original.len());

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed, &original[..]);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_decompress_exact_size() {
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        let original = b"exact";
        let compressed = compressor.compress_store(original).expect("compress");

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed.len(), original.len());
    }

    #[test]
    fn test_compress_random_data() {
        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        // Pseudo-random data (hard to compress)
        let mut original = Vec::with_capacity(256);
        let mut state = 12345u32;
        for _ in 0..256 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            original.push((state >> 16) as u8);
        }

        let compressed = compressor.compress_store(&original).expect("compress");

        let decompressed = decompressor
            .decompress_block(compressed, original.len())
            .expect("decompress");

        assert_eq!(decompressed, &original[..]);
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
}

//! CRC32 checksum implementation
//!
//! Pure Rust CRC-32 (IEEE 802.3 polynomial) for .apr file integrity verification.
//!
//! # Algorithm
//!
//! Uses the standard CRC-32 polynomial: 0xEDB88320 (reversed bit order)
//! Compatible with zlib, gzip, and most standard CRC-32 implementations.
//!
//! # Example
//!
//! ```rust
//! use whisper_apr::format::checksum::Crc32;
//!
//! let mut crc = Crc32::new();
//! crc.update(b"Hello, World!");
//! assert_eq!(crc.finalize(), 0xec4ac3d0);
//! ```

/// CRC-32 polynomial (IEEE 802.3) in reversed bit order
const CRC32_POLYNOMIAL: u32 = 0xEDB8_8320;

/// Pre-computed CRC32 lookup table
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// CRC-32 hasher for streaming checksum computation
///
/// Implements the IEEE 802.3 CRC-32 algorithm with a pre-computed lookup table
/// for efficient computation.
#[derive(Debug, Clone)]
pub struct Crc32 {
    /// Current CRC state (inverted)
    state: u32,
}

impl Crc32 {
    /// Create a new CRC-32 hasher
    #[must_use]
    pub const fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    /// Update the CRC with additional data
    ///
    /// Can be called multiple times for streaming computation.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let index = ((self.state ^ u32::from(byte)) & 0xFF) as usize;
            self.state = (self.state >> 8) ^ CRC32_TABLE[index];
        }
    }

    /// Finalize and return the CRC-32 value
    ///
    /// This can be called multiple times without resetting the state.
    #[must_use]
    pub const fn finalize(&self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }

    /// Reset the hasher to initial state
    pub fn reset(&mut self) {
        self.state = 0xFFFF_FFFF;
    }

    /// Compute CRC-32 of a single data buffer
    ///
    /// Convenience method for one-shot computation.
    #[must_use]
    pub fn compute(data: &[u8]) -> u32 {
        let mut crc = Self::new();
        crc.update(data);
        crc.finalize()
    }

    /// Verify data against expected CRC-32
    ///
    /// Returns true if the computed CRC matches expected.
    #[must_use]
    pub fn verify(data: &[u8], expected: u32) -> bool {
        Self::compute(data) == expected
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute CRC-32 checksum of data
///
/// Convenience function for one-shot computation.
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    Crc32::compute(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_crc32_new() {
        let crc = Crc32::new();
        assert_eq!(crc.state, 0xFFFF_FFFF);
    }

    #[test]
    fn test_crc32_default() {
        let crc = Crc32::default();
        assert_eq!(crc.state, 0xFFFF_FFFF);
    }

    // =========================================================================
    // Known Value Tests
    // =========================================================================

    #[test]
    fn test_empty_data() {
        let crc = crc32(&[]);
        assert_eq!(crc, 0x0000_0000);
    }

    #[test]
    fn test_hello_world() {
        // Standard test vector: "Hello, World!" -> 0xec4ac3d0
        let crc = crc32(b"Hello, World!");
        assert_eq!(crc, 0xec4a_c3d0);
    }

    #[test]
    fn test_check_string() {
        // Standard test: "123456789" -> 0xCBF43926
        let crc = crc32(b"123456789");
        assert_eq!(crc, 0xCBF4_3926);
    }

    #[test]
    fn test_single_byte() {
        // Single byte 'a' (0x61)
        let crc = crc32(b"a");
        assert_eq!(crc, 0xE8B7_BE43);
    }

    #[test]
    fn test_single_null() {
        // Single null byte
        let crc = crc32(&[0x00]);
        assert_eq!(crc, 0xD202_EF8D);
    }

    #[test]
    fn test_all_zeros() {
        let crc = crc32(&[0x00; 32]);
        assert_eq!(crc, 0x190A_55AD);
    }

    #[test]
    fn test_all_ones() {
        let crc = crc32(&[0xFF; 32]);
        assert_eq!(crc, 0xFF6C_AB0B);
    }

    // =========================================================================
    // Streaming Tests
    // =========================================================================

    #[test]
    fn test_streaming_single_update() {
        let mut crc = Crc32::new();
        crc.update(b"123456789");
        assert_eq!(crc.finalize(), 0xCBF4_3926);
    }

    #[test]
    fn test_streaming_multiple_updates() {
        let mut crc = Crc32::new();
        crc.update(b"123");
        crc.update(b"456");
        crc.update(b"789");
        assert_eq!(crc.finalize(), 0xCBF4_3926);
    }

    #[test]
    fn test_streaming_byte_by_byte() {
        let mut crc = Crc32::new();
        for byte in b"123456789" {
            crc.update(&[*byte]);
        }
        assert_eq!(crc.finalize(), 0xCBF4_3926);
    }

    #[test]
    fn test_streaming_equals_oneshot() {
        let data = b"The quick brown fox jumps over the lazy dog";

        let oneshot = crc32(data);

        let mut streaming = Crc32::new();
        for chunk in data.chunks(7) {
            streaming.update(chunk);
        }

        assert_eq!(streaming.finalize(), oneshot);
    }

    // =========================================================================
    // Reset Tests
    // =========================================================================

    #[test]
    fn test_reset() {
        let mut crc = Crc32::new();
        crc.update(b"some data");
        let first = crc.finalize();

        crc.reset();
        crc.update(b"other data");
        let second = crc.finalize();

        assert_ne!(first, second);

        // After reset, computing same data should give same result
        crc.reset();
        crc.update(b"some data");
        assert_eq!(crc.finalize(), first);
    }

    #[test]
    fn test_finalize_multiple_times() {
        let mut crc = Crc32::new();
        crc.update(b"test");

        let first = crc.finalize();
        let second = crc.finalize();

        assert_eq!(first, second);
    }

    // =========================================================================
    // Verify Tests
    // =========================================================================

    #[test]
    fn test_verify_success() {
        let data = b"123456789";
        assert!(Crc32::verify(data, 0xCBF4_3926));
    }

    #[test]
    fn test_verify_failure() {
        let data = b"123456789";
        assert!(!Crc32::verify(data, 0x1234_5678));
    }

    #[test]
    fn test_verify_corrupted_data() {
        let original = b"123456789";
        let mut corrupted = *original;
        corrupted[4] ^= 0x01; // Flip one bit

        let original_crc = crc32(original);
        assert!(!Crc32::verify(&corrupted, original_crc));
    }

    // =========================================================================
    // Table Tests
    // =========================================================================

    #[test]
    fn test_table_first_entries() {
        // First entry should be 0 (0 with 8 iterations gives 0)
        assert_eq!(CRC32_TABLE[0], 0x0000_0000);

        // Entry for 1 after 8 iterations of polynomial division
        assert_eq!(CRC32_TABLE[1], 0x7707_3096);

        // Entry for 2 (known value from standard tables)
        assert_eq!(CRC32_TABLE[2], 0xEE0E_612C);
    }

    #[test]
    fn test_table_length() {
        assert_eq!(CRC32_TABLE.len(), 256);
    }

    // =========================================================================
    // Binary Data Tests
    // =========================================================================

    #[test]
    fn test_binary_data() {
        // Test with all byte values
        let data: Vec<u8> = (0u8..=255).collect();
        let crc = crc32(&data);

        // Verify it's reproducible
        assert_eq!(crc, crc32(&data));
    }

    #[test]
    fn test_large_data() {
        // 64KB of pseudo-random data
        let mut data = vec![0u8; 65536];
        let mut state = 0x12345678u32;
        for byte in &mut data {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (state >> 16) as u8;
        }

        let crc1 = crc32(&data);
        let crc2 = crc32(&data);
        assert_eq!(crc1, crc2);

        // Modify one byte and verify CRC changes
        data[1000] ^= 0x80;
        let crc3 = crc32(&data);
        assert_ne!(crc1, crc3);
    }

    // =========================================================================
    // Clone Tests
    // =========================================================================

    #[test]
    fn test_clone() {
        let mut crc = Crc32::new();
        crc.update(b"partial");

        let cloned = crc.clone();

        crc.update(b"_more");
        let original_result = crc.finalize();

        // Clone should have the intermediate state
        let mut cloned = cloned;
        cloned.update(b"_more");
        assert_eq!(cloned.finalize(), original_result);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_repeated_pattern() {
        let data = b"aaaaaaaaaaaaaaaa";
        let crc = crc32(data);

        // Same pattern should give same result
        let data2 = b"aaaaaaaaaaaaaaaa";
        assert_eq!(crc, crc32(data2));

        // Different pattern should give different result
        let data3 = b"aaaaaaaaaaaaaaba";
        assert_ne!(crc, crc32(data3));
    }

    #[test]
    fn test_word_boundary() {
        // Test data that's exactly aligned to various boundaries
        for len in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32] {
            let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let crc1 = crc32(&data);
            let crc2 = Crc32::compute(&data);
            assert_eq!(crc1, crc2);
        }
    }
}

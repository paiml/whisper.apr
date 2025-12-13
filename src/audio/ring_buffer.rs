//! Lock-free ring buffer for audio streaming
//!
//! Provides a single-producer single-consumer (SPSC) ring buffer for transferring
//! audio samples between the AudioWorklet (real-time thread) and Whisper inference
//! (processing thread) without blocking.
//!
//! # Design (per spec 11.3)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Ring Buffer over SharedArrayBuffer             │
//! │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐   │
//! │  │     │     │     │  W  │     │     │  R  │     │     │   │
//! │  └─────┴─────┴─────┴──▲──┴─────┴─────┴──▲──┴─────┴─────┘   │
//! │                       │                  │                   │
//! │              write_index          read_index                 │
//! │                (atomic)            (atomic)                  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust
//! use whisper_apr::audio::RingBuffer;
//!
//! // Create a ring buffer with 2 seconds of audio at 16kHz
//! let mut buffer = RingBuffer::new(32000);
//!
//! // Producer (AudioWorklet) writes samples
//! let samples = vec![0.1, 0.2, 0.3];
//! buffer.write(&samples);
//!
//! // Consumer (Whisper) reads samples
//! let mut output = vec![0.0; 3];
//! let read = buffer.read(&mut output);
//! ```

use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Lock-free single-producer single-consumer ring buffer for audio samples
///
/// This implementation uses atomic operations for the read/write indices,
/// allowing one producer thread and one consumer thread to access the buffer
/// concurrently without locks.
///
/// # Thread Safety
///
/// - `write()` should only be called from one thread (producer/AudioWorklet)
/// - `read()` should only be called from one thread (consumer/inference)
/// - `available_read()` and `available_write()` can be called from either thread
#[derive(Debug)]
pub struct RingBuffer {
    /// Audio sample storage
    buffer: Vec<f32>,
    /// Write index (updated by producer)
    write_index: AtomicUsize,
    /// Read index (updated by consumer)
    read_index: AtomicUsize,
    /// Buffer capacity (power of 2 for efficient modulo)
    capacity: usize,
    /// Mask for fast modulo (capacity - 1)
    mask: usize,
}

impl RingBuffer {
    /// Create a new ring buffer with the specified capacity
    ///
    /// The capacity will be rounded up to the nearest power of 2 for efficiency.
    ///
    /// # Arguments
    /// * `min_capacity` - Minimum number of samples to hold
    ///
    /// # Example
    /// ```rust
    /// use whisper_apr::audio::RingBuffer;
    ///
    /// // 2 seconds at 16kHz
    /// let buffer = RingBuffer::new(32000);
    /// assert!(buffer.capacity() >= 32000);
    /// ```
    #[must_use]
    pub fn new(min_capacity: usize) -> Self {
        // Round up to power of 2 for efficient modulo
        let capacity = min_capacity.next_power_of_two();
        let mask = capacity - 1;

        Self {
            buffer: vec![0.0; capacity],
            write_index: AtomicUsize::new(0),
            read_index: AtomicUsize::new(0),
            capacity,
            mask,
        }
    }

    /// Create a ring buffer for a specific duration at a given sample rate
    ///
    /// # Arguments
    /// * `duration_seconds` - Duration in seconds
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Example
    /// ```rust
    /// use whisper_apr::audio::RingBuffer;
    ///
    /// // 2 minute buffer at 16kHz (handles inference lag)
    /// let buffer = RingBuffer::for_duration(120.0, 16000);
    /// assert!(buffer.capacity() >= 120 * 16000);
    /// ```
    #[must_use]
    pub fn for_duration(duration_seconds: f32, sample_rate: u32) -> Self {
        let min_capacity = (duration_seconds * sample_rate as f32) as usize;
        Self::new(min_capacity)
    }

    /// Get the buffer capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the number of samples available for reading
    #[must_use]
    pub fn available_read(&self) -> usize {
        let write = self.write_index.load(Ordering::Acquire);
        let read = self.read_index.load(Ordering::Acquire);
        write.wrapping_sub(read)
    }

    /// Get the number of samples that can be written
    #[must_use]
    pub fn available_write(&self) -> usize {
        self.capacity - self.available_read() - 1
    }

    /// Check if the buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.available_read() == 0
    }

    /// Check if the buffer is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.available_write() == 0
    }

    /// Write samples to the buffer (producer side)
    ///
    /// Returns the number of samples actually written. If the buffer is full,
    /// older samples may be overwritten (ring buffer behavior).
    ///
    /// # Arguments
    /// * `samples` - Audio samples to write
    ///
    /// # Returns
    /// Number of samples written (may be less than input if buffer is nearly full)
    pub fn write(&mut self, samples: &[f32]) -> usize {
        let available = self.available_write();
        let to_write = samples.len().min(available);

        if to_write == 0 {
            return 0;
        }

        let write = self.write_index.load(Ordering::Relaxed);

        for (i, &sample) in samples.iter().take(to_write).enumerate() {
            let idx = (write + i) & self.mask;
            self.buffer[idx] = sample;
        }

        // Release barrier ensures writes are visible before index update
        self.write_index
            .store(write.wrapping_add(to_write), Ordering::Release);

        to_write
    }

    /// Write samples, overwriting old data if buffer is full
    ///
    /// Unlike `write()`, this always writes all samples, advancing the read
    /// pointer if necessary to make room.
    ///
    /// # Arguments
    /// * `samples` - Audio samples to write
    pub fn write_overwrite(&mut self, samples: &[f32]) {
        // If we need to overwrite, advance read pointer
        if samples.len() > self.available_write() {
            let overflow = samples.len() - self.available_write();
            let read = self.read_index.load(Ordering::Relaxed);
            self.read_index
                .store(read.wrapping_add(overflow), Ordering::Release);
        }

        let write = self.write_index.load(Ordering::Relaxed);

        for (i, &sample) in samples.iter().enumerate() {
            let idx = (write + i) & self.mask;
            self.buffer[idx] = sample;
        }

        self.write_index
            .store(write.wrapping_add(samples.len()), Ordering::Release);
    }

    /// Read samples from the buffer (consumer side)
    ///
    /// Returns the number of samples actually read.
    ///
    /// # Arguments
    /// * `output` - Buffer to read samples into
    ///
    /// # Returns
    /// Number of samples read (may be less than output length if not enough data)
    pub fn read(&mut self, output: &mut [f32]) -> usize {
        let available = self.available_read();
        let to_read = output.len().min(available);

        if to_read == 0 {
            return 0;
        }

        let read = self.read_index.load(Ordering::Relaxed);

        for (i, sample) in output.iter_mut().take(to_read).enumerate() {
            let idx = (read + i) & self.mask;
            *sample = self.buffer[idx];
        }

        // Release barrier ensures reads complete before index update
        self.read_index
            .store(read.wrapping_add(to_read), Ordering::Release);

        to_read
    }

    /// Peek at samples without consuming them
    ///
    /// # Arguments
    /// * `output` - Buffer to copy samples into
    ///
    /// # Returns
    /// Number of samples peeked
    pub fn peek(&self, output: &mut [f32]) -> usize {
        let available = self.available_read();
        let to_peek = output.len().min(available);

        if to_peek == 0 {
            return 0;
        }

        let read = self.read_index.load(Ordering::Acquire);

        for (i, sample) in output.iter_mut().take(to_peek).enumerate() {
            let idx = (read + i) & self.mask;
            *sample = self.buffer[idx];
        }

        to_peek
    }

    /// Skip samples without reading them
    ///
    /// # Arguments
    /// * `count` - Number of samples to skip
    ///
    /// # Returns
    /// Number of samples actually skipped
    pub fn skip(&mut self, count: usize) -> usize {
        let available = self.available_read();
        let to_skip = count.min(available);

        if to_skip == 0 {
            return 0;
        }

        let read = self.read_index.load(Ordering::Relaxed);
        self.read_index
            .store(read.wrapping_add(to_skip), Ordering::Release);

        to_skip
    }

    /// Clear all samples from the buffer
    pub fn clear(&mut self) {
        let write = self.write_index.load(Ordering::Relaxed);
        self.read_index.store(write, Ordering::Release);
    }

    /// Get the duration of audio currently in the buffer
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Duration in seconds
    #[must_use]
    pub fn duration(&self, sample_rate: u32) -> f32 {
        self.available_read() as f32 / sample_rate as f32
    }
}

impl Default for RingBuffer {
    /// Create a default ring buffer with 2 minutes of audio at 16kHz
    fn default() -> Self {
        // 2 minutes at 16kHz = 1,920,000 samples
        // This handles typical inference lag (spec 11.3)
        Self::for_duration(120.0, 16000)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_new_rounds_to_power_of_two() {
        let buffer = RingBuffer::new(1000);
        assert_eq!(buffer.capacity(), 1024);

        let buffer = RingBuffer::new(1024);
        assert_eq!(buffer.capacity(), 1024);

        let buffer = RingBuffer::new(1025);
        assert_eq!(buffer.capacity(), 2048);
    }

    #[test]
    fn test_for_duration() {
        let buffer = RingBuffer::for_duration(1.0, 16000);
        assert!(buffer.capacity() >= 16000);

        let buffer = RingBuffer::for_duration(2.0, 44100);
        assert!(buffer.capacity() >= 88200);
    }

    #[test]
    fn test_default() {
        let buffer = RingBuffer::default();
        // 2 minutes at 16kHz = 1,920,000 samples
        assert!(buffer.capacity() >= 1_920_000);
    }

    // =========================================================================
    // Basic Operations Tests
    // =========================================================================

    #[test]
    fn test_empty_buffer() {
        let buffer = RingBuffer::new(1024);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.available_read(), 0);
        assert_eq!(buffer.available_write(), 1023); // capacity - 1
    }

    #[test]
    fn test_write_read_basic() {
        let mut buffer = RingBuffer::new(1024);

        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let written = buffer.write(&input);
        assert_eq!(written, 5);
        assert_eq!(buffer.available_read(), 5);

        let mut output = vec![0.0; 5];
        let read = buffer.read(&mut output);
        assert_eq!(read, 5);
        assert_eq!(output, input);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_write_read_partial() {
        let mut buffer = RingBuffer::new(1024);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.write(&input);

        let mut output = vec![0.0; 3];
        let read = buffer.read(&mut output);
        assert_eq!(read, 3);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
        assert_eq!(buffer.available_read(), 2);

        let read = buffer.read(&mut output);
        assert_eq!(read, 2);
        assert_eq!(output[..2], [4.0, 5.0]);
    }

    #[test]
    fn test_write_when_full() {
        let mut buffer = RingBuffer::new(8); // Capacity 8, can write 7

        let input = vec![1.0; 10];
        let written = buffer.write(&input);
        assert_eq!(written, 7); // Only 7 can be written (capacity - 1)
        assert!(buffer.is_full());

        // Can't write more when full
        let written = buffer.write(&[2.0]);
        assert_eq!(written, 0);
    }

    #[test]
    fn test_read_when_empty() {
        let mut buffer = RingBuffer::new(1024);
        let mut output = vec![0.0; 5];
        let read = buffer.read(&mut output);
        assert_eq!(read, 0);
    }

    // =========================================================================
    // Wraparound Tests
    // =========================================================================

    #[test]
    fn test_wraparound() {
        let mut buffer = RingBuffer::new(8);

        // Fill partially
        let input1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.write(&input1);

        // Read some
        let mut output = vec![0.0; 3];
        buffer.read(&mut output);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);

        // Write more (should wrap around)
        let input2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let written = buffer.write(&input2);
        assert_eq!(written, 5);

        // Read all
        let mut output = vec![0.0; 7];
        let read = buffer.read(&mut output);
        assert_eq!(read, 7);
        assert_eq!(output, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    // =========================================================================
    // Peek and Skip Tests
    // =========================================================================

    #[test]
    fn test_peek_does_not_consume() {
        let mut buffer = RingBuffer::new(1024);
        buffer.write(&[1.0, 2.0, 3.0]);

        let mut output = vec![0.0; 2];
        let peeked = buffer.peek(&mut output);
        assert_eq!(peeked, 2);
        assert_eq!(output, vec![1.0, 2.0]);
        assert_eq!(buffer.available_read(), 3); // Still 3

        // Read should still get all samples
        let mut output = vec![0.0; 3];
        buffer.read(&mut output);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_skip() {
        let mut buffer = RingBuffer::new(1024);
        buffer.write(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let skipped = buffer.skip(2);
        assert_eq!(skipped, 2);
        assert_eq!(buffer.available_read(), 3);

        let mut output = vec![0.0; 3];
        buffer.read(&mut output);
        assert_eq!(output, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_skip_more_than_available() {
        let mut buffer = RingBuffer::new(1024);
        buffer.write(&[1.0, 2.0, 3.0]);

        let skipped = buffer.skip(10);
        assert_eq!(skipped, 3);
        assert!(buffer.is_empty());
    }

    // =========================================================================
    // Clear Tests
    // =========================================================================

    #[test]
    fn test_clear() {
        let mut buffer = RingBuffer::new(1024);
        buffer.write(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(buffer.available_read(), 5);

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.available_read(), 0);
    }

    // =========================================================================
    // Overwrite Tests
    // =========================================================================

    #[test]
    fn test_write_overwrite() {
        let mut buffer = RingBuffer::new(8); // Capacity 8

        // Fill the buffer
        buffer.write(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert!(buffer.is_full());

        // Overwrite with new data
        buffer.write_overwrite(&[10.0, 11.0, 12.0]);

        // Should have overwritten oldest samples
        let mut output = vec![0.0; 7];
        let read = buffer.read(&mut output);
        assert_eq!(read, 7);
        // Old samples [1,2,3] should be gone, new ones added
        assert_eq!(output[4..], [10.0, 11.0, 12.0]);
    }

    // =========================================================================
    // Duration Tests
    // =========================================================================

    #[test]
    fn test_duration() {
        let mut buffer = RingBuffer::new(32000);

        // Write 1 second of audio at 16kHz
        let samples = vec![0.0; 16000];
        buffer.write(&samples);

        let duration = buffer.duration(16000);
        assert!((duration - 1.0).abs() < 0.001);
    }

    // =========================================================================
    // Property-Based Style Tests
    // =========================================================================

    #[test]
    fn test_write_read_preserves_data() {
        let mut buffer = RingBuffer::new(1024);

        // Generate test data
        let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        buffer.write(&input);

        let mut output = vec![0.0; 100];
        buffer.read(&mut output);

        for (i, (&expected, &actual)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < f32::EPSILON,
                "Mismatch at index {i}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_multiple_write_read_cycles() {
        let mut buffer = RingBuffer::new(64);

        for cycle in 0..10 {
            let input: Vec<f32> = (0..20).map(|i| (cycle * 20 + i) as f32).collect();
            let written = buffer.write(&input);
            assert_eq!(written, 20, "Cycle {cycle}: write failed");

            let mut output = vec![0.0; 20];
            let read = buffer.read(&mut output);
            assert_eq!(read, 20, "Cycle {cycle}: read failed");
            assert_eq!(output, input, "Cycle {cycle}: data mismatch");
        }
    }

    #[test]
    fn test_interleaved_write_read() {
        let mut buffer = RingBuffer::new(128);

        buffer.write(&[1.0, 2.0, 3.0]);
        let mut out = vec![0.0; 2];
        buffer.read(&mut out);
        assert_eq!(out, vec![1.0, 2.0]);

        buffer.write(&[4.0, 5.0, 6.0]);
        let mut out = vec![0.0; 4];
        let read = buffer.read(&mut out);
        assert_eq!(read, 4);
        assert_eq!(out, vec![3.0, 4.0, 5.0, 6.0]);
    }

    // =========================================================================
    // Property-Based Tests (WAPR-QA-002)
    // =========================================================================

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn property_write_read_preserves_data(
                capacity in 64usize..1024,
                data_len in 1usize..512
            ) {
                let capacity = capacity.max(data_len + 1);
                let mut buffer = RingBuffer::new(capacity);
                let data: Vec<f32> = (0..data_len).map(|i| i as f32).collect();

                buffer.write(&data);
                let mut output = vec![0.0; data_len];
                let read = buffer.read(&mut output);

                prop_assert_eq!(read, data_len);
                prop_assert_eq!(output, data);
            }

            #[test]
            fn property_available_matches_written(
                capacity in 64usize..512,
                write_count in 1usize..100
            ) {
                let mut buffer = RingBuffer::new(capacity);
                let data = vec![1.0; write_count.min(capacity - 1)];
                buffer.write(&data);

                prop_assert_eq!(buffer.available_read(), data.len());
                prop_assert!(buffer.available_read() <= buffer.capacity());
            }

            #[test]
            fn property_clear_empties_buffer(capacity in 32usize..256) {
                let mut buffer = RingBuffer::new(capacity);
                buffer.write(&[1.0, 2.0, 3.0]);
                buffer.clear();

                prop_assert!(buffer.is_empty());
                prop_assert_eq!(buffer.available_read(), 0);
            }

            #[test]
            fn property_peek_does_not_consume(capacity in 64usize..256) {
                let mut buffer = RingBuffer::new(capacity);
                let data = vec![1.0, 2.0, 3.0];
                buffer.write(&data);

                let len_before = buffer.available_read();
                let mut peek_output = vec![0.0; 2];
                let peeked = buffer.peek(&mut peek_output);
                let len_after = buffer.available_read();

                prop_assert_eq!(len_before, len_after);
                prop_assert_eq!(peeked, 2);
            }
        }
    }
}

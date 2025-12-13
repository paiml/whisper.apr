//! Memory pool for efficient tensor allocations
//!
//! Provides a reusable buffer pool to reduce allocation overhead during inference.
//! This is critical for WASM environments where allocation/deallocation can be expensive.
//!
//! # Architecture
//!
//! The pool maintains a set of pre-allocated buffers of various sizes. When a buffer
//! is requested, the pool returns the smallest available buffer that fits. When the
//! buffer is returned, it goes back to the pool for reuse.
//!
//! # Usage
//!
//! ```ignore
//! let pool = MemoryPool::new();
//! let buffer = pool.get(1024); // Get a buffer of at least 1024 floats
//! // Use buffer...
//! pool.return_buffer(buffer); // Return to pool for reuse
//! ```

use std::cell::RefCell;
use std::collections::BTreeMap;

/// Size class for buffer pooling
///
/// Buffers are allocated in power-of-2 sizes for efficient bucketing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SizeClass(usize);

impl SizeClass {
    /// Create a size class for the given number of elements
    #[must_use]
    pub fn for_size(size: usize) -> Self {
        if size == 0 {
            return Self(0);
        }
        // Round up to next power of 2
        let bits = usize::BITS - (size - 1).leading_zeros();
        Self(1 << bits)
    }

    /// Get the actual allocation size for this class
    #[must_use]
    pub const fn allocation_size(&self) -> usize {
        self.0
    }
}

/// A pooled buffer that can be returned to the pool
#[derive(Debug)]
pub struct PooledBuffer {
    /// Underlying data
    data: Vec<f32>,
    /// Size class this buffer belongs to
    size_class: SizeClass,
    /// Logical length (may be less than capacity)
    len: usize,
}

impl PooledBuffer {
    /// Create a new pooled buffer
    fn new(size_class: SizeClass) -> Self {
        Self {
            data: vec![0.0; size_class.allocation_size()],
            size_class,
            len: 0,
        }
    }

    /// Get the data as a slice
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data[..self.len]
    }

    /// Get the data as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data[..self.len]
    }

    /// Get the full capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Get the logical length
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Set the logical length (must be <= capacity)
    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.capacity(), "length exceeds capacity");
        self.len = len;
    }

    /// Fill with a value up to the logical length
    pub fn fill(&mut self, value: f32) {
        for v in &mut self.data[..self.len] {
            *v = value;
        }
    }

    /// Get the size class
    #[must_use]
    pub const fn size_class(&self) -> SizeClass {
        self.size_class
    }

    /// Convert to owned Vec (consumes the buffer)
    #[must_use]
    pub fn into_vec(mut self) -> Vec<f32> {
        self.data.truncate(self.len);
        self.data
    }

    /// Copy data from a slice
    pub fn copy_from_slice(&mut self, src: &[f32]) {
        assert!(src.len() <= self.capacity(), "source too large");
        self.data[..src.len()].copy_from_slice(src);
        self.len = src.len();
    }
}

/// Memory pool for tensor allocations
///
/// Maintains pools of buffers at different size classes for efficient reuse.
#[derive(Debug)]
pub struct MemoryPool {
    /// Pools of available buffers, keyed by size class
    pools: RefCell<BTreeMap<SizeClass, Vec<PooledBuffer>>>,
    /// Statistics
    stats: RefCell<PoolStats>,
    /// Maximum buffers per size class
    max_per_class: usize,
}

/// Pool statistics for monitoring
#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    /// Total allocations requested
    pub allocations: usize,
    /// Allocations served from pool (hits)
    pub hits: usize,
    /// Allocations that required new allocation (misses)
    pub misses: usize,
    /// Total returns to pool
    pub returns: usize,
    /// Returns that were dropped (pool full)
    pub dropped: usize,
}

impl PoolStats {
    /// Get hit rate as a percentage
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        if self.allocations == 0 {
            0.0
        } else {
            (self.hits as f32) / (self.allocations as f32) * 100.0
        }
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPool {
    /// Create a new memory pool with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            pools: RefCell::new(BTreeMap::new()),
            stats: RefCell::new(PoolStats::default()),
            max_per_class: 16,
        }
    }

    /// Create a memory pool with custom max buffers per size class
    #[must_use]
    pub fn with_max_per_class(max_per_class: usize) -> Self {
        Self {
            pools: RefCell::new(BTreeMap::new()),
            stats: RefCell::new(PoolStats::default()),
            max_per_class,
        }
    }

    /// Get a buffer of at least the requested size
    pub fn get(&self, size: usize) -> PooledBuffer {
        let size_class = SizeClass::for_size(size);
        self.stats.borrow_mut().allocations += 1;

        let mut pools = self.pools.borrow_mut();
        if let Some(pool) = pools.get_mut(&size_class) {
            if let Some(mut buffer) = pool.pop() {
                self.stats.borrow_mut().hits += 1;
                buffer.set_len(size);
                buffer.fill(0.0); // Zero out for safety
                return buffer;
            }
        }

        // Allocate new buffer
        self.stats.borrow_mut().misses += 1;
        let mut buffer = PooledBuffer::new(size_class);
        buffer.set_len(size);
        buffer
    }

    /// Get a buffer and fill it with data from a slice
    pub fn get_from_slice(&self, data: &[f32]) -> PooledBuffer {
        let mut buffer = self.get(data.len());
        buffer.copy_from_slice(data);
        buffer
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&self, buffer: PooledBuffer) {
        self.stats.borrow_mut().returns += 1;

        let mut pools = self.pools.borrow_mut();
        let pool = pools.entry(buffer.size_class).or_default();

        if pool.len() < self.max_per_class {
            pool.push(buffer);
        } else {
            self.stats.borrow_mut().dropped += 1;
            // Buffer is dropped, releasing memory
        }
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        self.stats.borrow().clone()
    }

    /// Clear all pooled buffers
    pub fn clear(&self) {
        self.pools.borrow_mut().clear();
    }

    /// Get total number of buffered allocations
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.pools.borrow().values().map(Vec::len).sum()
    }

    /// Get total bytes held in pool
    #[must_use]
    pub fn buffered_bytes(&self) -> usize {
        self.pools
            .borrow()
            .iter()
            .map(|(class, buffers)| class.allocation_size() * buffers.len() * 4)
            .sum()
    }

    /// Pre-allocate buffers of specific sizes
    pub fn preallocate(&self, sizes: &[usize]) {
        for &size in sizes {
            let buffer = self.get(size);
            self.return_buffer(buffer);
        }
    }
}

// Thread-local memory pool for convenience
thread_local! {
    static POOL: MemoryPool = MemoryPool::new();
}

/// Get a buffer from the thread-local pool
#[must_use]
pub fn get_buffer(size: usize) -> PooledBuffer {
    POOL.with(|pool| pool.get(size))
}

/// Return a buffer to the thread-local pool
pub fn return_buffer(buffer: PooledBuffer) {
    POOL.with(|pool| pool.return_buffer(buffer));
}

/// Get thread-local pool statistics
#[must_use]
pub fn pool_stats() -> PoolStats {
    POOL.with(|pool| pool.stats())
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SizeClass Tests
    // =========================================================================

    #[test]
    fn test_size_class_zero() {
        let class = SizeClass::for_size(0);
        assert_eq!(class.allocation_size(), 0);
    }

    #[test]
    fn test_size_class_one() {
        let class = SizeClass::for_size(1);
        assert_eq!(class.allocation_size(), 1);
    }

    #[test]
    fn test_size_class_power_of_two() {
        let class = SizeClass::for_size(64);
        assert_eq!(class.allocation_size(), 64);

        let class = SizeClass::for_size(1024);
        assert_eq!(class.allocation_size(), 1024);
    }

    #[test]
    fn test_size_class_rounds_up() {
        let class = SizeClass::for_size(65);
        assert_eq!(class.allocation_size(), 128);

        let class = SizeClass::for_size(1000);
        assert_eq!(class.allocation_size(), 1024);

        let class = SizeClass::for_size(1025);
        assert_eq!(class.allocation_size(), 2048);
    }

    // =========================================================================
    // PooledBuffer Tests
    // =========================================================================

    #[test]
    fn test_pooled_buffer_new() {
        let class = SizeClass::for_size(100);
        let buffer = PooledBuffer::new(class);

        assert_eq!(buffer.capacity(), 128);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_pooled_buffer_set_len() {
        let class = SizeClass::for_size(100);
        let mut buffer = PooledBuffer::new(class);

        buffer.set_len(50);
        assert_eq!(buffer.len(), 50);
        assert!(!buffer.is_empty());
    }

    #[test]
    #[should_panic(expected = "length exceeds capacity")]
    fn test_pooled_buffer_set_len_overflow() {
        let class = SizeClass::for_size(100);
        let mut buffer = PooledBuffer::new(class);
        buffer.set_len(200); // 200 > 128 capacity
    }

    #[test]
    fn test_pooled_buffer_fill() {
        let class = SizeClass::for_size(10);
        let mut buffer = PooledBuffer::new(class);
        buffer.set_len(10);
        buffer.fill(1.5);

        for &v in buffer.as_slice() {
            assert!((v - 1.5).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_pooled_buffer_copy_from_slice() {
        let class = SizeClass::for_size(10);
        let mut buffer = PooledBuffer::new(class);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.copy_from_slice(&data);

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.as_slice(), &data[..]);
    }

    #[test]
    fn test_pooled_buffer_into_vec() {
        let class = SizeClass::for_size(10);
        let mut buffer = PooledBuffer::new(class);

        let data = vec![1.0, 2.0, 3.0];
        buffer.copy_from_slice(&data);

        let vec = buffer.into_vec();
        assert_eq!(vec, data);
    }

    // =========================================================================
    // MemoryPool Tests
    // =========================================================================

    #[test]
    fn test_memory_pool_new() {
        let pool = MemoryPool::new();
        assert_eq!(pool.buffered_count(), 0);
    }

    #[test]
    fn test_memory_pool_get() {
        let pool = MemoryPool::new();
        let buffer = pool.get(100);

        assert_eq!(buffer.len(), 100);
        assert!(buffer.capacity() >= 100);
    }

    #[test]
    fn test_memory_pool_return() {
        let pool = MemoryPool::new();
        let buffer = pool.get(100);
        pool.return_buffer(buffer);

        assert_eq!(pool.buffered_count(), 1);
    }

    #[test]
    fn test_memory_pool_reuse() {
        let pool = MemoryPool::new();

        // Get and return a buffer
        let buffer = pool.get(100);
        let ptr1 = buffer.data.as_ptr();
        pool.return_buffer(buffer);

        // Get again - should reuse
        let buffer = pool.get(100);
        let ptr2 = buffer.data.as_ptr();

        // Same underlying allocation
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn test_memory_pool_stats() {
        let pool = MemoryPool::new();

        let b1 = pool.get(100);
        let _b2 = pool.get(100); // Count allocation
        pool.return_buffer(b1);
        let _b3 = pool.get(100); // Should hit

        let stats = pool.stats();
        assert_eq!(stats.allocations, 3);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.returns, 1);
    }

    #[test]
    fn test_memory_pool_hit_rate() {
        let pool = MemoryPool::new();

        // Miss
        let b1 = pool.get(100);
        pool.return_buffer(b1);

        // Hit
        let _b2 = pool.get(100);

        let stats = pool.stats();
        assert!((stats.hit_rate() - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_memory_pool_max_per_class() {
        let pool = MemoryPool::with_max_per_class(2);

        // Return 3 buffers
        let b1 = pool.get(100);
        let b2 = pool.get(100);
        let b3 = pool.get(100);

        pool.return_buffer(b1);
        pool.return_buffer(b2);
        pool.return_buffer(b3); // Should be dropped

        assert_eq!(pool.buffered_count(), 2);
        assert_eq!(pool.stats().dropped, 1);
    }

    #[test]
    fn test_memory_pool_clear() {
        let pool = MemoryPool::new();

        let b1 = pool.get(100);
        let b2 = pool.get(200);
        pool.return_buffer(b1);
        pool.return_buffer(b2);

        assert_eq!(pool.buffered_count(), 2);

        pool.clear();
        assert_eq!(pool.buffered_count(), 0);
    }

    #[test]
    fn test_memory_pool_preallocate() {
        let pool = MemoryPool::new();

        pool.preallocate(&[100, 200, 300]);

        assert_eq!(pool.buffered_count(), 3);
    }

    #[test]
    fn test_memory_pool_buffered_bytes() {
        let pool = MemoryPool::new();

        let b = pool.get(1024);
        pool.return_buffer(b);

        let bytes = pool.buffered_bytes();
        assert_eq!(bytes, 1024 * 4); // 1024 floats * 4 bytes each
    }

    #[test]
    fn test_memory_pool_get_from_slice() {
        let pool = MemoryPool::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let buffer = pool.get_from_slice(&data);

        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.as_slice(), &data[..]);
    }

    // =========================================================================
    // Thread-Local Pool Tests
    // =========================================================================

    #[test]
    fn test_thread_local_pool() {
        let buffer = get_buffer(100);
        assert_eq!(buffer.len(), 100);
        return_buffer(buffer);

        let stats = pool_stats();
        assert!(stats.allocations > 0);
    }

    #[test]
    fn test_different_size_classes() {
        let pool = MemoryPool::new();

        let b1 = pool.get(100); // class 128
        let b2 = pool.get(200); // class 256
        let b3 = pool.get(100); // class 128

        assert_eq!(b1.size_class(), b3.size_class());
        assert_ne!(b1.size_class(), b2.size_class());

        pool.return_buffer(b1);
        pool.return_buffer(b2);
        pool.return_buffer(b3);

        // 2 buffers in class 128, 1 in class 256
        assert_eq!(pool.buffered_count(), 3);
    }
}

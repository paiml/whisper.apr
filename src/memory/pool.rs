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

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

/// Size class for buffer pooling
///
/// Buffers are allocated in power-of-2 sizes for efficient bucketing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SizeClass(usize);

impl SizeClass {
    /// Create a size class for the given number of elements.
    ///
    /// Rounds `size` up to the next power of two for efficient pool bucketing.
    /// Returns `SizeClass(0)` when `size` is zero.
    #[must_use]
    pub fn for_size(size: usize) -> Self {
        if size == 0 {
            return Self(0);
        }
        // Round up to next power of 2
        let bits = usize::BITS - (size - 1).leading_zeros();
        let result = Self(1 << bits);
        debug_assert!(
            result.0.is_power_of_two(),
            "size class must be a power of 2"
        );
        debug_assert!(result.0 >= size, "size class must be >= requested size");
        result
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
    ///
    /// # Panics
    ///
    /// Panics if `len` exceeds capacity.
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
    ///
    /// # Panics
    ///
    /// Panics if source slice is larger than capacity.
    pub fn copy_from_slice(&mut self, src: &[f32]) {
        assert!(src.len() <= self.capacity(), "source too large");
        self.data[..src.len()].copy_from_slice(src);
        self.len = src.len();
    }
}

/// Statistics counters for pool operations (Cell for zero-borrow increment)
#[derive(Debug, Default)]
struct PoolCounters {
    allocations: Cell<usize>,
    hits: Cell<usize>,
    misses: Cell<usize>,
    returns: Cell<usize>,
    dropped: Cell<usize>,
}

impl PoolCounters {
    fn inc(counter: &Cell<usize>) {
        counter.set(counter.get() + 1);
    }

    fn to_stats(&self) -> PoolStats {
        PoolStats {
            allocations: self.allocations.get(),
            hits: self.hits.get(),
            misses: self.misses.get(),
            returns: self.returns.get(),
            dropped: self.dropped.get(),
        }
    }
}

/// Memory pool for tensor allocations
///
/// Maintains pools of buffers at different size classes for efficient reuse.
#[derive(Debug)]
pub struct MemoryPool {
    /// Pools of available buffers, keyed by size class
    pools: RefCell<BTreeMap<SizeClass, Vec<PooledBuffer>>>,
    /// Statistics counters
    counters: PoolCounters,
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
        Self::with_max_per_class(16)
    }

    /// Create a memory pool with custom max buffers per size class
    #[must_use]
    pub fn with_max_per_class(max_per_class: usize) -> Self {
        Self {
            pools: RefCell::new(BTreeMap::new()),
            counters: PoolCounters::default(),
            max_per_class,
        }
    }

    /// Get a buffer of at least the requested size
    pub fn get(&self, size: usize) -> PooledBuffer {
        let size_class = SizeClass::for_size(size);
        let cached = self
            .pools
            .borrow_mut()
            .get_mut(&size_class)
            .and_then(Vec::pop);
        PoolCounters::inc(&self.counters.allocations);

        if let Some(mut buffer) = cached {
            PoolCounters::inc(&self.counters.hits);
            buffer.set_len(size);
            buffer.fill(0.0);
            buffer
        } else {
            PoolCounters::inc(&self.counters.misses);
            let mut buffer = PooledBuffer::new(size_class);
            buffer.set_len(size);
            buffer
        }
    }

    /// Get a buffer and fill it with data from a slice
    pub fn get_from_slice(&self, data: &[f32]) -> PooledBuffer {
        let mut buffer = self.get(data.len());
        buffer.copy_from_slice(data);
        buffer
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&self, buffer: PooledBuffer) {
        PoolCounters::inc(&self.counters.returns);
        let mut pools = self.pools.borrow_mut();
        let pool = pools.entry(buffer.size_class).or_default();

        if pool.len() < self.max_per_class {
            pool.push(buffer);
        } else {
            PoolCounters::inc(&self.counters.dropped);
        }
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        self.counters.to_stats()
    }

    /// Clear all pooled buffers
    pub fn clear(&self) {
        self.pools.borrow_mut().clear();
    }

    /// Get total number of buffered allocations
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.with_pools(|pools| pools.values().map(Vec::len).sum())
    }

    /// Get total bytes held in pool
    #[must_use]
    pub fn buffered_bytes(&self) -> usize {
        self.with_pools(|pools| {
            pools
                .iter()
                .map(|(class, buffers)| class.allocation_size() * buffers.len() * 4)
                .sum()
        })
    }

    /// Execute a closure with a shared borrow of the pools
    fn with_pools<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&BTreeMap<SizeClass, Vec<PooledBuffer>>) -> R,
    {
        f(&self.pools.borrow())
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
#[path = "pool_tests.rs"]
mod tests;

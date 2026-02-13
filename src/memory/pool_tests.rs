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

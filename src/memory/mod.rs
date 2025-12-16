//! Memory management for Whisper inference
//!
//! Provides efficient memory allocation via pooling and memory-mapped file access
//! for large model weights.

pub mod mmap;
mod pool;

pub use mmap::{
    MemoryRegion, MmapConfig, MmapHandle, MmapMode, WeightDtype, WeightRegion, WeightType,
};
pub use pool::{
    get_buffer, pool_stats, return_buffer, MemoryPool, PoolStats, PooledBuffer, SizeClass,
};

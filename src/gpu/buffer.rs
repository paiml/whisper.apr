//! GPU buffer management (WAPR-121)
//!
//! Provides abstractions for GPU memory buffers used in compute operations.

use super::error::{GpuError, GpuResult};
use super::{BUFFER_ALIGNMENT, MAX_BUFFER_SIZE};

/// Buffer usage flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBufferUsage {
    /// Can be used as source for copy operations
    pub copy_src: bool,
    /// Can be used as destination for copy operations
    pub copy_dst: bool,
    /// Can be bound as storage buffer in compute shader
    pub storage: bool,
    /// Can be bound as uniform buffer
    pub uniform: bool,
    /// Can be mapped for CPU read
    pub map_read: bool,
    /// Can be mapped for CPU write
    pub map_write: bool,
}

impl Default for GpuBufferUsage {
    fn default() -> Self {
        Self {
            copy_src: false,
            copy_dst: false,
            storage: true,
            uniform: false,
            map_read: false,
            map_write: false,
        }
    }
}

impl GpuBufferUsage {
    /// Create storage buffer usage (for compute shaders)
    #[must_use]
    pub fn storage() -> Self {
        Self {
            storage: true,
            copy_src: true,
            copy_dst: true,
            ..Default::default()
        }
    }

    /// Create uniform buffer usage
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            uniform: true,
            copy_dst: true,
            storage: false,
            ..Default::default()
        }
    }

    /// Create staging buffer for upload (CPU -> GPU)
    #[must_use]
    pub fn staging_upload() -> Self {
        Self {
            copy_src: true,
            map_write: true,
            storage: false,
            ..Default::default()
        }
    }

    /// Create staging buffer for download (GPU -> CPU)
    #[must_use]
    pub fn staging_download() -> Self {
        Self {
            copy_dst: true,
            map_read: true,
            storage: false,
            ..Default::default()
        }
    }

    /// Create read-write storage buffer
    #[must_use]
    pub fn storage_read_write() -> Self {
        Self {
            storage: true,
            copy_src: true,
            copy_dst: true,
            ..Default::default()
        }
    }

    /// Add copy source capability
    #[must_use]
    pub fn with_copy_src(mut self) -> Self {
        self.copy_src = true;
        self
    }

    /// Add copy destination capability
    #[must_use]
    pub fn with_copy_dst(mut self) -> Self {
        self.copy_dst = true;
        self
    }

    /// Add map read capability
    #[must_use]
    pub fn with_map_read(mut self) -> Self {
        self.map_read = true;
        self
    }

    /// Add map write capability
    #[must_use]
    pub fn with_map_write(mut self) -> Self {
        self.map_write = true;
        self
    }
}

/// GPU buffer descriptor
#[derive(Debug, Clone)]
pub struct GpuBufferDescriptor {
    /// Buffer size in bytes
    pub size: u64,
    /// Buffer usage flags
    pub usage: GpuBufferUsage,
    /// Label for debugging
    pub label: Option<String>,
    /// Whether to map buffer on creation
    pub mapped_at_creation: bool,
}

impl GpuBufferDescriptor {
    /// Create a new buffer descriptor
    #[must_use]
    pub fn new(size: u64, usage: GpuBufferUsage) -> Self {
        Self {
            size,
            usage,
            label: None,
            mapped_at_creation: false,
        }
    }

    /// Create storage buffer descriptor
    #[must_use]
    pub fn storage(size: u64) -> Self {
        Self::new(size, GpuBufferUsage::storage())
    }

    /// Create uniform buffer descriptor
    #[must_use]
    pub fn uniform(size: u64) -> Self {
        Self::new(size, GpuBufferUsage::uniform())
    }

    /// Create staging upload buffer descriptor
    #[must_use]
    pub fn staging_upload(size: u64) -> Self {
        Self::new(size, GpuBufferUsage::staging_upload()).with_mapped(true)
    }

    /// Create staging download buffer descriptor
    #[must_use]
    pub fn staging_download(size: u64) -> Self {
        Self::new(size, GpuBufferUsage::staging_download())
    }

    /// Set buffer label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set mapped at creation
    #[must_use]
    pub fn with_mapped(mut self, mapped: bool) -> Self {
        self.mapped_at_creation = mapped;
        self
    }

    /// Validate the descriptor
    pub fn validate(&self) -> GpuResult<()> {
        if self.size == 0 {
            return Err(GpuError::buffer("Buffer size cannot be zero"));
        }

        if self.size > MAX_BUFFER_SIZE {
            return Err(GpuError::InvalidBufferSize {
                requested: self.size,
                max: MAX_BUFFER_SIZE,
            });
        }

        Ok(())
    }

    /// Get aligned size (rounded up to BUFFER_ALIGNMENT)
    #[must_use]
    pub fn aligned_size(&self) -> u64 {
        align_size(self.size, BUFFER_ALIGNMENT)
    }
}

/// GPU buffer handle
///
/// Represents a buffer allocated on the GPU. The actual buffer
/// data is managed by the GPU device.
#[derive(Debug)]
pub struct GpuBuffer {
    /// Buffer ID (for tracking)
    id: u64,
    /// Buffer size in bytes
    size: u64,
    /// Buffer usage
    usage: GpuBufferUsage,
    /// Buffer label
    label: Option<String>,
    /// Whether buffer is currently mapped
    is_mapped: bool,
}

impl GpuBuffer {
    /// Create a new buffer handle
    ///
    /// Note: This doesn't actually allocate GPU memory without the webgpu feature.
    /// Use `GpuDevice::create_buffer` for actual allocation.
    pub fn new(descriptor: GpuBufferDescriptor) -> GpuResult<Self> {
        descriptor.validate()?;

        static BUFFER_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Ok(Self {
            id: BUFFER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            size: descriptor.aligned_size(),
            usage: descriptor.usage,
            label: descriptor.label,
            is_mapped: descriptor.mapped_at_creation,
        })
    }

    /// Create a storage buffer
    pub fn storage(size: u64) -> GpuResult<Self> {
        Self::new(GpuBufferDescriptor::storage(size))
    }

    /// Create a uniform buffer
    pub fn uniform(size: u64) -> GpuResult<Self> {
        Self::new(GpuBufferDescriptor::uniform(size))
    }

    /// Get buffer ID
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get buffer size in bytes
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get buffer usage
    #[must_use]
    pub fn usage(&self) -> &GpuBufferUsage {
        &self.usage
    }

    /// Get buffer label
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Check if buffer is currently mapped
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.is_mapped
    }

    /// Check if buffer can be used as storage
    #[must_use]
    pub fn is_storage(&self) -> bool {
        self.usage.storage
    }

    /// Check if buffer can be used as uniform
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        self.usage.uniform
    }

    /// Check if buffer can be copied from
    #[must_use]
    pub fn can_copy_src(&self) -> bool {
        self.usage.copy_src
    }

    /// Check if buffer can be copied to
    #[must_use]
    pub fn can_copy_dst(&self) -> bool {
        self.usage.copy_dst
    }

    /// Get size in KB
    #[must_use]
    pub fn size_kb(&self) -> f32 {
        self.size as f32 / 1024.0
    }

    /// Get size in MB
    #[must_use]
    pub fn size_mb(&self) -> f32 {
        self.size as f32 / (1024.0 * 1024.0)
    }
}

/// Align a size to the given alignment
#[inline]
#[must_use]
pub fn align_size(size: u64, alignment: u64) -> u64 {
    ((size + alignment - 1) / alignment) * alignment
}

/// Check if an offset is properly aligned
#[inline]
#[must_use]
pub fn is_aligned(offset: u64, alignment: u64) -> bool {
    offset % alignment == 0
}

/// Calculate the number of elements that fit in a buffer
#[inline]
#[must_use]
pub fn elements_in_buffer<T>(buffer_size: u64) -> usize {
    (buffer_size as usize) / std::mem::size_of::<T>()
}

/// Calculate buffer size for a number of elements
#[inline]
#[must_use]
pub fn buffer_size_for<T>(count: usize) -> u64 {
    (count * std::mem::size_of::<T>()) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_usage_default() {
        let usage = GpuBufferUsage::default();
        assert!(usage.storage);
        assert!(!usage.copy_src);
        assert!(!usage.uniform);
    }

    #[test]
    fn test_buffer_usage_storage() {
        let usage = GpuBufferUsage::storage();
        assert!(usage.storage);
        assert!(usage.copy_src);
        assert!(usage.copy_dst);
    }

    #[test]
    fn test_buffer_usage_uniform() {
        let usage = GpuBufferUsage::uniform();
        assert!(usage.uniform);
        assert!(usage.copy_dst);
        assert!(!usage.storage);
    }

    #[test]
    fn test_buffer_usage_staging_upload() {
        let usage = GpuBufferUsage::staging_upload();
        assert!(usage.copy_src);
        assert!(usage.map_write);
        assert!(!usage.storage);
    }

    #[test]
    fn test_buffer_usage_staging_download() {
        let usage = GpuBufferUsage::staging_download();
        assert!(usage.copy_dst);
        assert!(usage.map_read);
    }

    #[test]
    fn test_buffer_usage_builders() {
        let usage = GpuBufferUsage::default()
            .with_copy_src()
            .with_copy_dst()
            .with_map_read();

        assert!(usage.copy_src);
        assert!(usage.copy_dst);
        assert!(usage.map_read);
    }

    #[test]
    fn test_buffer_descriptor_new() {
        let desc = GpuBufferDescriptor::new(1024, GpuBufferUsage::storage());
        assert_eq!(desc.size, 1024);
        assert!(desc.usage.storage);
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_buffer_descriptor_storage() {
        let desc = GpuBufferDescriptor::storage(2048);
        assert_eq!(desc.size, 2048);
        assert!(desc.usage.storage);
    }

    #[test]
    fn test_buffer_descriptor_with_label() {
        let desc = GpuBufferDescriptor::storage(1024).with_label("weights");
        assert_eq!(desc.label, Some("weights".to_string()));
    }

    #[test]
    fn test_buffer_descriptor_validate_zero_size() {
        let desc = GpuBufferDescriptor::new(0, GpuBufferUsage::storage());
        assert!(desc.validate().is_err());
    }

    #[test]
    fn test_buffer_descriptor_validate_too_large() {
        let desc = GpuBufferDescriptor::new(MAX_BUFFER_SIZE + 1, GpuBufferUsage::storage());
        assert!(desc.validate().is_err());
    }

    #[test]
    fn test_buffer_descriptor_aligned_size() {
        let desc = GpuBufferDescriptor::new(100, GpuBufferUsage::storage());
        assert_eq!(desc.aligned_size(), 256); // Aligned to 256
    }

    #[test]
    fn test_gpu_buffer_new() {
        let buffer = GpuBuffer::storage(1024).unwrap();
        assert!(buffer.id() > 0);
        assert_eq!(buffer.size(), 1024); // 1024 aligns to 1024
        assert!(buffer.is_storage());
    }

    #[test]
    fn test_gpu_buffer_uniform() {
        let buffer = GpuBuffer::uniform(512).unwrap();
        assert!(buffer.is_uniform());
        assert!(!buffer.is_storage());
    }

    #[test]
    fn test_gpu_buffer_unique_ids() {
        let b1 = GpuBuffer::storage(1024).unwrap();
        let b2 = GpuBuffer::storage(1024).unwrap();
        assert_ne!(b1.id(), b2.id());
    }

    #[test]
    fn test_gpu_buffer_size_conversions() {
        let buffer = GpuBuffer::storage(1024 * 1024).unwrap(); // 1 MB
        assert!((buffer.size_kb() - 1024.0).abs() < 1.0);
        assert!((buffer.size_mb() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_align_size() {
        assert_eq!(align_size(100, 256), 256);
        assert_eq!(align_size(256, 256), 256);
        assert_eq!(align_size(257, 256), 512);
        assert_eq!(align_size(1, 256), 256);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0, 256));
        assert!(is_aligned(256, 256));
        assert!(is_aligned(512, 256));
        assert!(!is_aligned(100, 256));
        assert!(!is_aligned(257, 256));
    }

    #[test]
    fn test_elements_in_buffer() {
        // f32 is 4 bytes
        assert_eq!(elements_in_buffer::<f32>(1024), 256);
        assert_eq!(elements_in_buffer::<f32>(4096), 1024);
    }

    #[test]
    fn test_buffer_size_for() {
        assert_eq!(buffer_size_for::<f32>(256), 1024);
        assert_eq!(buffer_size_for::<f32>(1024), 4096);
    }
}

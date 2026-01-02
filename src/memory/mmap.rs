//! Memory mapping for large models (WAPR-142)
//!
//! Provides memory-mapped file access for efficient loading of large model
//! weights that may exceed available RAM.

use crate::error::WhisperResult;

/// Memory mapping mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MmapMode {
    /// Read-only mapping (most common for inference)
    #[default]
    ReadOnly,
    /// Read-write mapping (for training/fine-tuning)
    ReadWrite,
    /// Copy-on-write mapping
    CopyOnWrite,
}

impl MmapMode {
    /// Get description
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::ReadOnly => "read-only",
            Self::ReadWrite => "read-write",
            Self::CopyOnWrite => "copy-on-write",
        }
    }

    /// Check if writable
    #[must_use]
    pub fn is_writable(&self) -> bool {
        matches!(self, Self::ReadWrite | Self::CopyOnWrite)
    }
}

/// Memory mapping configuration
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Mapping mode
    pub mode: MmapMode,
    /// Whether to prefetch data
    pub prefetch: bool,
    /// Page size hint (0 = system default)
    pub page_size_hint: usize,
    /// Whether to lock pages in memory
    pub lock_pages: bool,
    /// Whether to advise sequential access
    pub sequential_access: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            mode: MmapMode::default(),
            prefetch: false,
            page_size_hint: 0,
            lock_pages: false,
            sequential_access: true,
        }
    }
}

impl MmapConfig {
    /// Create config for inference (read-only, sequential)
    #[must_use]
    pub fn for_inference() -> Self {
        Self {
            mode: MmapMode::ReadOnly,
            prefetch: true,
            sequential_access: true,
            ..Default::default()
        }
    }

    /// Create config for random access
    #[must_use]
    pub fn random_access() -> Self {
        Self {
            sequential_access: false,
            ..Default::default()
        }
    }

    /// Set mode
    #[must_use]
    pub fn with_mode(mut self, mode: MmapMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable prefetching
    #[must_use]
    pub fn with_prefetch(mut self) -> Self {
        self.prefetch = true;
        self
    }

    /// Enable page locking
    #[must_use]
    pub fn with_locked_pages(mut self) -> Self {
        self.lock_pages = true;
        self
    }

    /// Set page size hint
    #[must_use]
    pub fn with_page_size(mut self, size: usize) -> Self {
        self.page_size_hint = size;
        self
    }
}

/// Memory region descriptor
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Offset in the file
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
    /// Alignment requirement
    pub alignment: u64,
    /// Label for debugging
    pub label: Option<String>,
}

impl MemoryRegion {
    /// Create a new memory region
    #[must_use]
    pub fn new(offset: u64, size: u64) -> Self {
        Self {
            offset,
            size,
            alignment: 1,
            label: None,
        }
    }

    /// Create region for entire file
    #[must_use]
    pub fn entire_file(size: u64) -> Self {
        Self::new(0, size)
    }

    /// Set alignment
    #[must_use]
    pub fn with_alignment(mut self, alignment: u64) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get end offset
    #[must_use]
    pub fn end(&self) -> u64 {
        self.offset + self.size
    }

    /// Check if offset is within region
    #[must_use]
    pub fn contains(&self, offset: u64) -> bool {
        offset >= self.offset && offset < self.end()
    }

    /// Get aligned offset (rounded down)
    #[must_use]
    pub fn aligned_offset(&self) -> u64 {
        (self.offset / self.alignment) * self.alignment
    }

    /// Get aligned size (rounded up)
    #[must_use]
    pub fn aligned_size(&self) -> u64 {
        let aligned_start = self.aligned_offset();
        let end = self.offset + self.size;
        let aligned_end = end.div_ceil(self.alignment) * self.alignment;
        aligned_end - aligned_start
    }
}

/// Memory mapped file handle (simulated without actual mmap)
#[derive(Debug)]
pub struct MmapHandle {
    /// Handle ID
    id: u64,
    /// Total mapped size
    size: u64,
    /// Configuration
    config: MmapConfig,
    /// Active regions
    regions: Vec<MemoryRegion>,
    /// Whether handle is valid
    valid: bool,
}

impl MmapHandle {
    /// Create a new simulated mmap handle
    pub fn new(size: u64, config: MmapConfig) -> WhisperResult<Self> {
        static HANDLE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Ok(Self {
            id: HANDLE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            size,
            config,
            regions: Vec::new(),
            valid: true,
        })
    }

    /// Create for inference
    pub fn for_inference(size: u64) -> WhisperResult<Self> {
        Self::new(size, MmapConfig::for_inference())
    }

    /// Get handle ID
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get total size
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &MmapConfig {
        &self.config
    }

    /// Check if valid
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Add a region to track
    pub fn add_region(&mut self, region: MemoryRegion) {
        self.regions.push(region);
    }

    /// Get number of regions
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Get total bytes in tracked regions
    #[must_use]
    pub fn tracked_bytes(&self) -> u64 {
        self.regions.iter().map(|r| r.size).sum()
    }

    /// Invalidate the handle
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Simulate reading from offset
    pub fn read_at(&self, offset: u64, size: usize) -> WhisperResult<Vec<u8>> {
        if !self.valid {
            return Err(crate::error::WhisperError::Model(
                "Memory map handle is invalid".to_string(),
            ));
        }

        if offset + size as u64 > self.size {
            return Err(crate::error::WhisperError::Model(
                "Read extends beyond mapped region".to_string(),
            ));
        }

        // Return zeroed buffer (simulated read)
        Ok(vec![0u8; size])
    }

    /// Simulate writing to offset
    pub fn write_at(&mut self, offset: u64, data: &[u8]) -> WhisperResult<()> {
        if !self.valid {
            return Err(crate::error::WhisperError::Model(
                "Memory map handle is invalid".to_string(),
            ));
        }

        if !self.config.mode.is_writable() {
            return Err(crate::error::WhisperError::Model(
                "Memory map is read-only".to_string(),
            ));
        }

        if offset + data.len() as u64 > self.size {
            return Err(crate::error::WhisperError::Model(
                "Write extends beyond mapped region".to_string(),
            ));
        }

        // Simulated write (no-op)
        Ok(())
    }
}

/// Model weight region
#[derive(Debug, Clone)]
pub struct WeightRegion {
    /// Layer name
    pub name: String,
    /// Parameter type (weight, bias, etc.)
    pub param_type: WeightType,
    /// Memory region
    pub region: MemoryRegion,
    /// Data type
    pub dtype: WeightDtype,
    /// Shape
    pub shape: Vec<usize>,
}

/// Weight parameter type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightType {
    /// Dense weight matrix
    Weight,
    /// Bias vector
    Bias,
    /// Layer norm scale (gamma)
    Scale,
    /// Layer norm offset (beta)
    Offset,
    /// Embedding matrix
    Embedding,
    /// Attention query projection
    QueryProj,
    /// Attention key projection
    KeyProj,
    /// Attention value projection
    ValueProj,
    /// Attention output projection
    OutProj,
}

impl WeightType {
    /// Get human-readable name
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Weight => "weight",
            Self::Bias => "bias",
            Self::Scale => "scale",
            Self::Offset => "offset",
            Self::Embedding => "embedding",
            Self::QueryProj => "query_proj",
            Self::KeyProj => "key_proj",
            Self::ValueProj => "value_proj",
            Self::OutProj => "out_proj",
        }
    }
}

/// Weight data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    /// 32-bit float
    F32,
    /// 16-bit float
    F16,
    /// 16-bit brain float
    Bf16,
    /// 8-bit integer (quantized)
    Int8,
    /// 4-bit integer (quantized)
    Int4,
}

impl WeightDtype {
    /// Get bytes per element
    #[must_use]
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Int8 | Self::Int4 => 1, // Int4 packed, but minimum addressable is 1 byte
        }
    }

    /// Get human-readable name
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::F32 => "float32",
            Self::F16 => "float16",
            Self::Bf16 => "bfloat16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }
}

impl WeightRegion {
    /// Create a new weight region
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        param_type: WeightType,
        region: MemoryRegion,
        dtype: WeightDtype,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            name: name.into(),
            param_type,
            region,
            dtype,
            shape,
        }
    }

    /// Get total elements
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get expected size in bytes
    #[must_use]
    pub fn expected_bytes(&self) -> usize {
        self.num_elements() * self.dtype.bytes_per_element()
    }

    /// Verify size matches expected
    #[must_use]
    pub fn size_matches(&self) -> bool {
        self.region.size as usize >= self.expected_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_mode_default() {
        assert_eq!(MmapMode::default(), MmapMode::ReadOnly);
    }

    #[test]
    fn test_mmap_mode_description() {
        assert!(MmapMode::ReadOnly.description().contains("read"));
        assert!(MmapMode::ReadWrite.description().contains("write"));
        assert!(MmapMode::CopyOnWrite.description().contains("copy"));
    }

    #[test]
    fn test_mmap_mode_is_writable() {
        assert!(!MmapMode::ReadOnly.is_writable());
        assert!(MmapMode::ReadWrite.is_writable());
        assert!(MmapMode::CopyOnWrite.is_writable());
    }

    #[test]
    fn test_mmap_config_default() {
        let config = MmapConfig::default();
        assert_eq!(config.mode, MmapMode::ReadOnly);
        assert!(!config.prefetch);
        assert!(config.sequential_access);
    }

    #[test]
    fn test_mmap_config_for_inference() {
        let config = MmapConfig::for_inference();
        assert_eq!(config.mode, MmapMode::ReadOnly);
        assert!(config.prefetch);
        assert!(config.sequential_access);
    }

    #[test]
    fn test_mmap_config_builders() {
        let config = MmapConfig::default()
            .with_mode(MmapMode::ReadWrite)
            .with_prefetch()
            .with_locked_pages()
            .with_page_size(4096);

        assert_eq!(config.mode, MmapMode::ReadWrite);
        assert!(config.prefetch);
        assert!(config.lock_pages);
        assert_eq!(config.page_size_hint, 4096);
    }

    #[test]
    fn test_memory_region_new() {
        let region = MemoryRegion::new(1024, 4096);
        assert_eq!(region.offset, 1024);
        assert_eq!(region.size, 4096);
        assert_eq!(region.alignment, 1);
    }

    #[test]
    fn test_memory_region_entire_file() {
        let region = MemoryRegion::entire_file(1_000_000);
        assert_eq!(region.offset, 0);
        assert_eq!(region.size, 1_000_000);
    }

    #[test]
    fn test_memory_region_builders() {
        let region = MemoryRegion::new(0, 1024)
            .with_alignment(256)
            .with_label("weights");

        assert_eq!(region.alignment, 256);
        assert_eq!(region.label, Some("weights".to_string()));
    }

    #[test]
    fn test_memory_region_end() {
        let region = MemoryRegion::new(1024, 4096);
        assert_eq!(region.end(), 5120);
    }

    #[test]
    fn test_memory_region_contains() {
        let region = MemoryRegion::new(1024, 4096);
        assert!(region.contains(1024));
        assert!(region.contains(2000));
        assert!(region.contains(5119));
        assert!(!region.contains(5120));
        assert!(!region.contains(0));
    }

    #[test]
    fn test_memory_region_aligned() {
        let region = MemoryRegion::new(100, 50).with_alignment(64);
        assert_eq!(region.aligned_offset(), 64);
        assert_eq!(region.aligned_size(), 128); // 64 to 192
    }

    #[test]
    fn test_mmap_handle_new() {
        let handle =
            MmapHandle::new(1024 * 1024, MmapConfig::default()).expect("Should create handle");
        assert!(handle.id() > 0);
        assert_eq!(handle.size(), 1024 * 1024);
        assert!(handle.is_valid());
    }

    #[test]
    fn test_mmap_handle_for_inference() {
        let handle = MmapHandle::for_inference(1024 * 1024).expect("Should create");
        assert!(handle.config().prefetch);
    }

    #[test]
    fn test_mmap_handle_regions() {
        let mut handle =
            MmapHandle::new(1024 * 1024, MmapConfig::default()).expect("Should create");

        assert_eq!(handle.region_count(), 0);
        assert_eq!(handle.tracked_bytes(), 0);

        handle.add_region(MemoryRegion::new(0, 1024));
        handle.add_region(MemoryRegion::new(1024, 2048));

        assert_eq!(handle.region_count(), 2);
        assert_eq!(handle.tracked_bytes(), 3072);
    }

    #[test]
    fn test_mmap_handle_invalidate() {
        let mut handle = MmapHandle::new(1024, MmapConfig::default()).expect("Should create");
        assert!(handle.is_valid());

        handle.invalidate();
        assert!(!handle.is_valid());
    }

    #[test]
    fn test_mmap_handle_read_at() {
        let handle = MmapHandle::new(1024, MmapConfig::default()).expect("Should create");

        let data = handle.read_at(0, 100).expect("Should read");
        assert_eq!(data.len(), 100);

        // Read past end should fail
        assert!(handle.read_at(1000, 100).is_err());
    }

    #[test]
    fn test_mmap_handle_write_at_readonly() {
        let mut handle = MmapHandle::new(1024, MmapConfig::default()).expect("Should create");

        // Write to read-only should fail
        assert!(handle.write_at(0, &[1, 2, 3]).is_err());
    }

    #[test]
    fn test_mmap_handle_write_at_readwrite() {
        let mut handle =
            MmapHandle::new(1024, MmapConfig::default().with_mode(MmapMode::ReadWrite))
                .expect("Should create");

        // Write should succeed
        assert!(handle.write_at(0, &[1, 2, 3]).is_ok());

        // Write past end should fail
        assert!(handle.write_at(1000, &[0; 100]).is_err());
    }

    #[test]
    fn test_weight_type_name() {
        assert_eq!(WeightType::Weight.name(), "weight");
        assert_eq!(WeightType::Bias.name(), "bias");
        assert_eq!(WeightType::QueryProj.name(), "query_proj");
    }

    #[test]
    fn test_weight_dtype_bytes() {
        assert_eq!(WeightDtype::F32.bytes_per_element(), 4);
        assert_eq!(WeightDtype::F16.bytes_per_element(), 2);
        assert_eq!(WeightDtype::Int8.bytes_per_element(), 1);
    }

    #[test]
    fn test_weight_dtype_name() {
        assert_eq!(WeightDtype::F32.name(), "float32");
        assert_eq!(WeightDtype::F16.name(), "float16");
        assert_eq!(WeightDtype::Int4.name(), "int4");
    }

    #[test]
    fn test_weight_region_new() {
        let region = WeightRegion::new(
            "encoder.layer_0.weight",
            WeightType::Weight,
            MemoryRegion::new(0, 768 * 768 * 4),
            WeightDtype::F32,
            vec![768, 768],
        );

        assert_eq!(region.name, "encoder.layer_0.weight");
        assert_eq!(region.param_type, WeightType::Weight);
        assert_eq!(region.num_elements(), 768 * 768);
    }

    #[test]
    fn test_weight_region_expected_bytes() {
        let region = WeightRegion::new(
            "test",
            WeightType::Weight,
            MemoryRegion::new(0, 1024),
            WeightDtype::F32,
            vec![16, 16],
        );

        assert_eq!(region.expected_bytes(), 16 * 16 * 4);
    }

    #[test]
    fn test_weight_region_size_matches() {
        let region = WeightRegion::new(
            "test",
            WeightType::Weight,
            MemoryRegion::new(0, 256 * 4),
            WeightDtype::F32,
            vec![16, 16],
        );

        assert!(region.size_matches());

        let too_small = WeightRegion::new(
            "test",
            WeightType::Weight,
            MemoryRegion::new(0, 100),
            WeightDtype::F32,
            vec![16, 16],
        );

        assert!(!too_small.size_matches());
    }
}

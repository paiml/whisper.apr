//! Memory mapping for large models (WAPR-142)
//!
//! Provides memory-mapped file access for efficient loading of large model
//! weights that may exceed available RAM.

#[cfg(test)]
mod tests;

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

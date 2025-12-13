//! GPU compute shader pipeline management (WAPR-122)
//!
//! Provides abstractions for creating and managing compute pipelines.

use super::error::{GpuError, GpuResult};
use super::DEFAULT_WORKGROUP_SIZE;

/// Shader source type
#[derive(Debug, Clone)]
pub enum ShaderSource {
    /// WGSL shader source code
    Wgsl(String),
    /// SPIR-V bytecode
    SpirV(Vec<u32>),
}

impl ShaderSource {
    /// Create WGSL shader source
    #[must_use]
    pub fn wgsl(source: impl Into<String>) -> Self {
        Self::Wgsl(source.into())
    }

    /// Create SPIR-V shader source
    #[must_use]
    pub fn spirv(bytecode: Vec<u32>) -> Self {
        Self::SpirV(bytecode)
    }

    /// Check if this is WGSL source
    #[must_use]
    pub fn is_wgsl(&self) -> bool {
        matches!(self, Self::Wgsl(_))
    }

    /// Check if this is SPIR-V bytecode
    #[must_use]
    pub fn is_spirv(&self) -> bool {
        matches!(self, Self::SpirV(_))
    }

    /// Get source length (characters for WGSL, words for SPIR-V)
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Wgsl(s) => s.len(),
            Self::SpirV(v) => v.len(),
        }
    }

    /// Check if source is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Shader module descriptor
#[derive(Debug, Clone)]
pub struct ShaderModuleDescriptor {
    /// Shader source
    pub source: ShaderSource,
    /// Label for debugging
    pub label: Option<String>,
}

impl ShaderModuleDescriptor {
    /// Create a new shader module descriptor from WGSL source
    #[must_use]
    pub fn wgsl(source: impl Into<String>) -> Self {
        Self {
            source: ShaderSource::wgsl(source),
            label: None,
        }
    }

    /// Create a new shader module descriptor from SPIR-V
    #[must_use]
    pub fn spirv(bytecode: Vec<u32>) -> Self {
        Self {
            source: ShaderSource::spirv(bytecode),
            label: None,
        }
    }

    /// Set the label for debugging
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Validate the descriptor
    pub fn validate(&self) -> GpuResult<()> {
        if self.source.is_empty() {
            return Err(GpuError::shader("Shader source cannot be empty"));
        }
        Ok(())
    }
}

/// Shader module handle
#[derive(Debug)]
pub struct ShaderModule {
    /// Module ID
    id: u64,
    /// Source type
    source_type: ShaderSourceType,
    /// Label
    label: Option<String>,
}

/// Type of shader source
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderSourceType {
    /// WGSL source
    Wgsl,
    /// SPIR-V bytecode
    SpirV,
}

impl ShaderModule {
    /// Create a new shader module
    pub fn new(descriptor: ShaderModuleDescriptor) -> GpuResult<Self> {
        descriptor.validate()?;

        static MODULE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        let source_type = if descriptor.source.is_wgsl() {
            ShaderSourceType::Wgsl
        } else {
            ShaderSourceType::SpirV
        };

        Ok(Self {
            id: MODULE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            source_type,
            label: descriptor.label,
        })
    }

    /// Get module ID
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get source type
    #[must_use]
    pub fn source_type(&self) -> ShaderSourceType {
        self.source_type
    }

    /// Get label
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

/// Bind group entry type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingType {
    /// Storage buffer (read-write)
    StorageBuffer,
    /// Read-only storage buffer
    ReadOnlyStorageBuffer,
    /// Uniform buffer
    UniformBuffer,
}

impl BindingType {
    /// Check if this is a storage type
    #[must_use]
    pub fn is_storage(&self) -> bool {
        matches!(self, Self::StorageBuffer | Self::ReadOnlyStorageBuffer)
    }

    /// Check if this is read-only
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        matches!(self, Self::ReadOnlyStorageBuffer | Self::UniformBuffer)
    }
}

/// Bind group layout entry
#[derive(Debug, Clone)]
pub struct BindGroupLayoutEntry {
    /// Binding index
    pub binding: u32,
    /// Binding type
    pub binding_type: BindingType,
    /// Whether this binding is optional
    pub optional: bool,
}

impl BindGroupLayoutEntry {
    /// Create a storage buffer binding
    #[must_use]
    pub fn storage_buffer(binding: u32) -> Self {
        Self {
            binding,
            binding_type: BindingType::StorageBuffer,
            optional: false,
        }
    }

    /// Create a read-only storage buffer binding
    #[must_use]
    pub fn read_only_storage_buffer(binding: u32) -> Self {
        Self {
            binding,
            binding_type: BindingType::ReadOnlyStorageBuffer,
            optional: false,
        }
    }

    /// Create a uniform buffer binding
    #[must_use]
    pub fn uniform_buffer(binding: u32) -> Self {
        Self {
            binding,
            binding_type: BindingType::UniformBuffer,
            optional: false,
        }
    }

    /// Mark as optional
    #[must_use]
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }
}

/// Bind group layout descriptor
#[derive(Debug, Clone)]
pub struct BindGroupLayoutDescriptor {
    /// Entries in the layout
    pub entries: Vec<BindGroupLayoutEntry>,
    /// Label for debugging
    pub label: Option<String>,
}

impl BindGroupLayoutDescriptor {
    /// Create a new bind group layout descriptor
    #[must_use]
    pub fn new(entries: Vec<BindGroupLayoutEntry>) -> Self {
        Self {
            entries,
            label: None,
        }
    }

    /// Set the label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the number of entries
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Validate the descriptor
    pub fn validate(&self) -> GpuResult<()> {
        // Check for duplicate bindings
        let mut seen = std::collections::HashSet::new();
        for entry in &self.entries {
            if !seen.insert(entry.binding) {
                return Err(GpuError::pipeline(format!(
                    "Duplicate binding index: {}",
                    entry.binding
                )));
            }
        }
        Ok(())
    }
}

/// Bind group layout handle
#[derive(Debug)]
pub struct BindGroupLayout {
    /// Layout ID
    id: u64,
    /// Number of entries
    entry_count: usize,
    /// Label
    label: Option<String>,
}

impl BindGroupLayout {
    /// Create a new bind group layout
    pub fn new(descriptor: BindGroupLayoutDescriptor) -> GpuResult<Self> {
        descriptor.validate()?;

        static LAYOUT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Ok(Self {
            id: LAYOUT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            entry_count: descriptor.entries.len(),
            label: descriptor.label,
        })
    }

    /// Get layout ID
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get entry count
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Get label
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

/// Compute pipeline descriptor
#[derive(Debug)]
pub struct ComputePipelineDescriptor {
    /// Shader module
    pub shader_module_id: u64,
    /// Entry point function name
    pub entry_point: String,
    /// Bind group layouts
    pub bind_group_layout_ids: Vec<u64>,
    /// Label for debugging
    pub label: Option<String>,
}

impl ComputePipelineDescriptor {
    /// Create a new compute pipeline descriptor
    #[must_use]
    pub fn new(shader_module_id: u64, entry_point: impl Into<String>) -> Self {
        Self {
            shader_module_id,
            entry_point: entry_point.into(),
            bind_group_layout_ids: Vec::new(),
            label: None,
        }
    }

    /// Add a bind group layout
    #[must_use]
    pub fn with_bind_group_layout(mut self, layout_id: u64) -> Self {
        self.bind_group_layout_ids.push(layout_id);
        self
    }

    /// Set the label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Validate the descriptor
    pub fn validate(&self) -> GpuResult<()> {
        if self.entry_point.is_empty() {
            return Err(GpuError::pipeline("Entry point cannot be empty"));
        }
        Ok(())
    }
}

/// Compute pipeline handle
#[derive(Debug)]
pub struct ComputePipeline {
    /// Pipeline ID
    id: u64,
    /// Shader module ID
    shader_module_id: u64,
    /// Entry point
    entry_point: String,
    /// Bind group layout count
    bind_group_count: usize,
    /// Label
    label: Option<String>,
}

impl ComputePipeline {
    /// Create a new compute pipeline
    pub fn new(descriptor: ComputePipelineDescriptor) -> GpuResult<Self> {
        descriptor.validate()?;

        static PIPELINE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Ok(Self {
            id: PIPELINE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            shader_module_id: descriptor.shader_module_id,
            entry_point: descriptor.entry_point,
            bind_group_count: descriptor.bind_group_layout_ids.len(),
            label: descriptor.label,
        })
    }

    /// Get pipeline ID
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get shader module ID
    #[must_use]
    pub fn shader_module_id(&self) -> u64 {
        self.shader_module_id
    }

    /// Get entry point
    #[must_use]
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Get bind group count
    #[must_use]
    pub fn bind_group_count(&self) -> usize {
        self.bind_group_count
    }

    /// Get label
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

/// Workgroup dimensions for compute dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupDimensions {
    /// X dimension
    pub x: u32,
    /// Y dimension
    pub y: u32,
    /// Z dimension
    pub z: u32,
}

impl Default for WorkgroupDimensions {
    fn default() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }
}

impl WorkgroupDimensions {
    /// Create 1D workgroup dimensions
    #[must_use]
    pub fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create 2D workgroup dimensions
    #[must_use]
    pub fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create 3D workgroup dimensions
    #[must_use]
    pub fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Get total number of workgroups
    #[must_use]
    pub fn total(&self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
    }

    /// Check if this is a 1D dispatch
    #[must_use]
    pub fn is_1d(&self) -> bool {
        self.y == 1 && self.z == 1
    }

    /// Check if this is a 2D dispatch
    #[must_use]
    pub fn is_2d(&self) -> bool {
        self.z == 1 && !self.is_1d()
    }

    /// Check if this is a 3D dispatch
    #[must_use]
    pub fn is_3d(&self) -> bool {
        !self.is_1d() && !self.is_2d()
    }
}

/// Compute dispatch configuration
#[derive(Debug, Clone)]
pub struct ComputeDispatch {
    /// Pipeline ID
    pub pipeline_id: u64,
    /// Workgroup dimensions
    pub workgroups: WorkgroupDimensions,
    /// Workgroup size (threads per workgroup)
    pub workgroup_size: u32,
}

impl ComputeDispatch {
    /// Create a new compute dispatch
    #[must_use]
    pub fn new(pipeline_id: u64, workgroups: WorkgroupDimensions) -> Self {
        Self {
            pipeline_id,
            workgroups,
            workgroup_size: DEFAULT_WORKGROUP_SIZE,
        }
    }

    /// Create a 1D dispatch for N elements
    #[must_use]
    pub fn for_elements(pipeline_id: u64, elements: u32) -> Self {
        let workgroups = (elements + DEFAULT_WORKGROUP_SIZE - 1) / DEFAULT_WORKGROUP_SIZE;
        Self {
            pipeline_id,
            workgroups: WorkgroupDimensions::new_1d(workgroups),
            workgroup_size: DEFAULT_WORKGROUP_SIZE,
        }
    }

    /// Set workgroup size
    #[must_use]
    pub fn with_workgroup_size(mut self, size: u32) -> Self {
        self.workgroup_size = size;
        self
    }

    /// Get total thread count
    #[must_use]
    pub fn total_threads(&self) -> u64 {
        self.workgroups.total() * u64::from(self.workgroup_size)
    }
}

/// Buffer binding for a bind group
#[derive(Debug, Clone)]
pub struct BufferBinding {
    /// Buffer ID
    pub buffer_id: u64,
    /// Offset into buffer
    pub offset: u64,
    /// Size to bind (None = entire buffer)
    pub size: Option<u64>,
}

impl BufferBinding {
    /// Create a new buffer binding
    #[must_use]
    pub fn new(buffer_id: u64) -> Self {
        Self {
            buffer_id,
            offset: 0,
            size: None,
        }
    }

    /// Create binding with offset and size
    #[must_use]
    pub fn with_range(buffer_id: u64, offset: u64, size: u64) -> Self {
        Self {
            buffer_id,
            offset,
            size: Some(size),
        }
    }

    /// Set offset
    #[must_use]
    pub fn at_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    /// Set size
    #[must_use]
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }
}

/// Bind group entry
#[derive(Debug, Clone)]
pub struct BindGroupEntry {
    /// Binding index
    pub binding: u32,
    /// Buffer binding
    pub resource: BufferBinding,
}

impl BindGroupEntry {
    /// Create a new bind group entry
    #[must_use]
    pub fn new(binding: u32, buffer_id: u64) -> Self {
        Self {
            binding,
            resource: BufferBinding::new(buffer_id),
        }
    }

    /// Create with a buffer binding
    #[must_use]
    pub fn with_buffer(binding: u32, resource: BufferBinding) -> Self {
        Self { binding, resource }
    }
}

/// Bind group descriptor
#[derive(Debug, Clone)]
pub struct BindGroupDescriptor {
    /// Layout ID
    pub layout_id: u64,
    /// Entries
    pub entries: Vec<BindGroupEntry>,
    /// Label for debugging
    pub label: Option<String>,
}

impl BindGroupDescriptor {
    /// Create a new bind group descriptor
    #[must_use]
    pub fn new(layout_id: u64, entries: Vec<BindGroupEntry>) -> Self {
        Self {
            layout_id,
            entries,
            label: None,
        }
    }

    /// Set the label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Validate the descriptor
    pub fn validate(&self) -> GpuResult<()> {
        // Check for duplicate bindings
        let mut seen = std::collections::HashSet::new();
        for entry in &self.entries {
            if !seen.insert(entry.binding) {
                return Err(GpuError::pipeline(format!(
                    "Duplicate binding in bind group: {}",
                    entry.binding
                )));
            }
        }
        Ok(())
    }
}

/// Bind group handle
#[derive(Debug)]
pub struct BindGroup {
    /// Bind group ID
    id: u64,
    /// Layout ID
    layout_id: u64,
    /// Number of entries
    entry_count: usize,
    /// Label
    label: Option<String>,
}

impl BindGroup {
    /// Create a new bind group
    pub fn new(descriptor: BindGroupDescriptor) -> GpuResult<Self> {
        descriptor.validate()?;

        static BIND_GROUP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Ok(Self {
            id: BIND_GROUP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            layout_id: descriptor.layout_id,
            entry_count: descriptor.entries.len(),
            label: descriptor.label,
        })
    }

    /// Get bind group ID
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get layout ID
    #[must_use]
    pub fn layout_id(&self) -> u64 {
        self.layout_id
    }

    /// Get entry count
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Get label
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_source_wgsl() {
        let source = ShaderSource::wgsl("@compute fn main() {}");
        assert!(source.is_wgsl());
        assert!(!source.is_spirv());
        assert!(!source.is_empty());
    }

    #[test]
    fn test_shader_source_spirv() {
        let source = ShaderSource::spirv(vec![0x07230203, 0x00010000]);
        assert!(source.is_spirv());
        assert!(!source.is_wgsl());
        assert_eq!(source.len(), 2);
    }

    #[test]
    fn test_shader_source_empty() {
        let source = ShaderSource::wgsl("");
        assert!(source.is_empty());
    }

    #[test]
    fn test_shader_module_descriptor() {
        let desc = ShaderModuleDescriptor::wgsl("@compute fn main() {}")
            .with_label("test_shader");
        assert!(desc.label.is_some());
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn test_shader_module_descriptor_empty_fails() {
        let desc = ShaderModuleDescriptor::wgsl("");
        assert!(desc.validate().is_err());
    }

    #[test]
    fn test_shader_module_new() {
        let desc = ShaderModuleDescriptor::wgsl("@compute fn main() {}");
        let module = ShaderModule::new(desc).expect("Should create module");
        assert!(module.id() > 0);
        assert_eq!(module.source_type(), ShaderSourceType::Wgsl);
    }

    #[test]
    fn test_shader_module_unique_ids() {
        let m1 = ShaderModule::new(ShaderModuleDescriptor::wgsl("test1")).expect("m1");
        let m2 = ShaderModule::new(ShaderModuleDescriptor::wgsl("test2")).expect("m2");
        assert_ne!(m1.id(), m2.id());
    }

    #[test]
    fn test_binding_type() {
        assert!(BindingType::StorageBuffer.is_storage());
        assert!(BindingType::ReadOnlyStorageBuffer.is_storage());
        assert!(!BindingType::UniformBuffer.is_storage());

        assert!(BindingType::ReadOnlyStorageBuffer.is_read_only());
        assert!(BindingType::UniformBuffer.is_read_only());
        assert!(!BindingType::StorageBuffer.is_read_only());
    }

    #[test]
    fn test_bind_group_layout_entry() {
        let entry = BindGroupLayoutEntry::storage_buffer(0);
        assert_eq!(entry.binding, 0);
        assert_eq!(entry.binding_type, BindingType::StorageBuffer);
        assert!(!entry.optional);

        let optional = entry.optional();
        assert!(optional.optional);
    }

    #[test]
    fn test_bind_group_layout_descriptor() {
        let desc = BindGroupLayoutDescriptor::new(vec![
            BindGroupLayoutEntry::storage_buffer(0),
            BindGroupLayoutEntry::uniform_buffer(1),
        ]).with_label("test_layout");

        assert_eq!(desc.entry_count(), 2);
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn test_bind_group_layout_duplicate_binding_fails() {
        let desc = BindGroupLayoutDescriptor::new(vec![
            BindGroupLayoutEntry::storage_buffer(0),
            BindGroupLayoutEntry::storage_buffer(0), // Duplicate!
        ]);

        assert!(desc.validate().is_err());
    }

    #[test]
    fn test_bind_group_layout_new() {
        let desc = BindGroupLayoutDescriptor::new(vec![
            BindGroupLayoutEntry::storage_buffer(0),
        ]);
        let layout = BindGroupLayout::new(desc).expect("Should create layout");
        assert!(layout.id() > 0);
        assert_eq!(layout.entry_count(), 1);
    }

    #[test]
    fn test_compute_pipeline_descriptor() {
        let desc = ComputePipelineDescriptor::new(1, "main")
            .with_bind_group_layout(1)
            .with_label("test_pipeline");

        assert_eq!(desc.shader_module_id, 1);
        assert_eq!(desc.entry_point, "main");
        assert_eq!(desc.bind_group_layout_ids.len(), 1);
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn test_compute_pipeline_descriptor_empty_entry_fails() {
        let desc = ComputePipelineDescriptor::new(1, "");
        assert!(desc.validate().is_err());
    }

    #[test]
    fn test_compute_pipeline_new() {
        let desc = ComputePipelineDescriptor::new(1, "main");
        let pipeline = ComputePipeline::new(desc).expect("Should create pipeline");
        assert!(pipeline.id() > 0);
        assert_eq!(pipeline.entry_point(), "main");
    }

    #[test]
    fn test_workgroup_dimensions_default() {
        let dims = WorkgroupDimensions::default();
        assert_eq!(dims.x, 1);
        assert_eq!(dims.y, 1);
        assert_eq!(dims.z, 1);
        assert!(dims.is_1d());
    }

    #[test]
    fn test_workgroup_dimensions_1d() {
        let dims = WorkgroupDimensions::new_1d(64);
        assert_eq!(dims.total(), 64);
        assert!(dims.is_1d());
        assert!(!dims.is_2d());
    }

    #[test]
    fn test_workgroup_dimensions_2d() {
        let dims = WorkgroupDimensions::new_2d(8, 8);
        assert_eq!(dims.total(), 64);
        assert!(dims.is_2d());
        assert!(!dims.is_1d());
    }

    #[test]
    fn test_workgroup_dimensions_3d() {
        let dims = WorkgroupDimensions::new_3d(4, 4, 4);
        assert_eq!(dims.total(), 64);
        assert!(dims.is_3d());
    }

    #[test]
    fn test_compute_dispatch() {
        let dispatch = ComputeDispatch::new(1, WorkgroupDimensions::new_1d(16));
        assert_eq!(dispatch.pipeline_id, 1);
        assert_eq!(dispatch.workgroups.x, 16);
        assert_eq!(dispatch.workgroup_size, DEFAULT_WORKGROUP_SIZE);
    }

    #[test]
    fn test_compute_dispatch_for_elements() {
        // 1000 elements / 256 workgroup size = 4 workgroups (rounded up)
        let dispatch = ComputeDispatch::for_elements(1, 1000);
        assert_eq!(dispatch.workgroups.x, 4);
        assert!(dispatch.total_threads() >= 1000);
    }

    #[test]
    fn test_compute_dispatch_with_workgroup_size() {
        let dispatch = ComputeDispatch::for_elements(1, 1000)
            .with_workgroup_size(128);
        assert_eq!(dispatch.workgroup_size, 128);
    }

    #[test]
    fn test_buffer_binding() {
        let binding = BufferBinding::new(1);
        assert_eq!(binding.buffer_id, 1);
        assert_eq!(binding.offset, 0);
        assert!(binding.size.is_none());

        let with_range = BufferBinding::with_range(2, 256, 1024);
        assert_eq!(with_range.offset, 256);
        assert_eq!(with_range.size, Some(1024));
    }

    #[test]
    fn test_buffer_binding_builders() {
        let binding = BufferBinding::new(1)
            .at_offset(512)
            .with_size(256);

        assert_eq!(binding.offset, 512);
        assert_eq!(binding.size, Some(256));
    }

    #[test]
    fn test_bind_group_entry() {
        let entry = BindGroupEntry::new(0, 1);
        assert_eq!(entry.binding, 0);
        assert_eq!(entry.resource.buffer_id, 1);
    }

    #[test]
    fn test_bind_group_descriptor() {
        let desc = BindGroupDescriptor::new(1, vec![
            BindGroupEntry::new(0, 1),
            BindGroupEntry::new(1, 2),
        ]).with_label("test_bind_group");

        assert_eq!(desc.layout_id, 1);
        assert_eq!(desc.entries.len(), 2);
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn test_bind_group_descriptor_duplicate_fails() {
        let desc = BindGroupDescriptor::new(1, vec![
            BindGroupEntry::new(0, 1),
            BindGroupEntry::new(0, 2), // Duplicate binding!
        ]);

        assert!(desc.validate().is_err());
    }

    #[test]
    fn test_bind_group_new() {
        let desc = BindGroupDescriptor::new(1, vec![
            BindGroupEntry::new(0, 1),
        ]);
        let group = BindGroup::new(desc).expect("Should create bind group");
        assert!(group.id() > 0);
        assert_eq!(group.layout_id(), 1);
        assert_eq!(group.entry_count(), 1);
    }

    #[test]
    fn test_bind_group_unique_ids() {
        let g1 = BindGroup::new(BindGroupDescriptor::new(1, vec![])).expect("g1");
        let g2 = BindGroup::new(BindGroupDescriptor::new(1, vec![])).expect("g2");
        assert_ne!(g1.id(), g2.id());
    }
}

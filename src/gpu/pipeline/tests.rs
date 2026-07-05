//! Tests for GPU compute shader pipeline management
#![allow(clippy::expect_used)]

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
    let desc = ShaderModuleDescriptor::wgsl("@compute fn main() {}").with_label("test_shader");
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
    ])
    .with_label("test_layout");

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
    let desc = BindGroupLayoutDescriptor::new(vec![BindGroupLayoutEntry::storage_buffer(0)]);
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
    let dispatch = ComputeDispatch::for_elements(1, 1000).with_workgroup_size(128);
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
    let binding = BufferBinding::new(1).at_offset(512).with_size(256);

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
    let desc = BindGroupDescriptor::new(
        1,
        vec![BindGroupEntry::new(0, 1), BindGroupEntry::new(1, 2)],
    )
    .with_label("test_bind_group");

    assert_eq!(desc.layout_id, 1);
    assert_eq!(desc.entries.len(), 2);
    assert!(desc.validate().is_ok());
}

#[test]
fn test_bind_group_descriptor_duplicate_fails() {
    let desc = BindGroupDescriptor::new(
        1,
        vec![
            BindGroupEntry::new(0, 1),
            BindGroupEntry::new(0, 2), // Duplicate binding!
        ],
    );

    assert!(desc.validate().is_err());
}

#[test]
fn test_bind_group_new() {
    let desc = BindGroupDescriptor::new(1, vec![BindGroupEntry::new(0, 1)]);
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

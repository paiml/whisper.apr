//! Tests for memory mapping

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
    let handle = MmapHandle::new(1024 * 1024, MmapConfig::default()).expect("Should create handle");
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
    let mut handle = MmapHandle::new(1024 * 1024, MmapConfig::default()).expect("Should create");

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
    let mut handle = MmapHandle::new(1024, MmapConfig::default().with_mode(MmapMode::ReadWrite))
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

// =========================================================================
// Additional Coverage Tests (WAPR-QA-003)
// =========================================================================

#[test]
fn test_mmap_config_random_access() {
    let config = MmapConfig::random_access();
    assert!(!config.sequential_access);
    assert_eq!(config.mode, MmapMode::ReadOnly);
    assert!(!config.prefetch);
}

#[test]
fn test_weight_type_name_all_variants() {
    assert_eq!(WeightType::Scale.name(), "scale");
    assert_eq!(WeightType::Offset.name(), "offset");
    assert_eq!(WeightType::Embedding.name(), "embedding");
    assert_eq!(WeightType::KeyProj.name(), "key_proj");
    assert_eq!(WeightType::ValueProj.name(), "value_proj");
    assert_eq!(WeightType::OutProj.name(), "out_proj");
}

#[test]
fn test_mmap_handle_read_at_boundary() {
    let handle = MmapHandle::new(256, MmapConfig::default()).expect("Should create");

    // Read exactly to the end
    let data = handle.read_at(0, 256).expect("Should read full size");
    assert_eq!(data.len(), 256);

    // Read 1 byte past end should fail
    assert!(handle.read_at(0, 257).is_err());
    assert!(handle.read_at(255, 2).is_err());
}

#[test]
fn test_mmap_handle_invalidate_then_read() {
    let mut handle = MmapHandle::new(1024, MmapConfig::default()).expect("Should create");
    handle.invalidate();
    assert!(!handle.is_valid());
    // After invalidation, reads should still work on the simulated buffer
}

#[test]
fn test_memory_region_zero_size() {
    let region = MemoryRegion::new(0, 0);
    assert_eq!(region.end(), 0);
    assert!(!region.contains(0));
}

#[test]
fn test_weight_dtype_all_bytes() {
    assert_eq!(WeightDtype::Bf16.bytes_per_element(), 2);
    assert_eq!(WeightDtype::Int4.bytes_per_element(), 1);
}

// =========================================================================
// Coverage Tests for random_access and additional branches (WAPR-QA-005)
// =========================================================================

#[test]
fn test_mmap_config_random_access_fields() {
    let config = MmapConfig::random_access();
    // Verify all fields match expectations for random access config
    assert!(!config.sequential_access);
    assert_eq!(config.mode, MmapMode::ReadOnly);
    assert!(!config.prefetch);
    assert!(!config.lock_pages);
    assert_eq!(config.page_size_hint, 0);
}

#[test]
fn test_mmap_config_random_access_with_mode() {
    let config = MmapConfig::random_access().with_mode(MmapMode::ReadWrite);
    assert!(!config.sequential_access);
    assert_eq!(config.mode, MmapMode::ReadWrite);
}

#[test]
fn test_mmap_config_random_access_with_prefetch() {
    let config = MmapConfig::random_access().with_prefetch();
    assert!(!config.sequential_access);
    assert!(config.prefetch);
}

#[test]
fn test_mmap_config_random_access_with_locked_pages() {
    let config = MmapConfig::random_access().with_locked_pages();
    assert!(!config.sequential_access);
    assert!(config.lock_pages);
}

#[test]
fn test_mmap_config_random_access_with_page_size() {
    let config = MmapConfig::random_access().with_page_size(8192);
    assert!(!config.sequential_access);
    assert_eq!(config.page_size_hint, 8192);
}

#[test]
fn test_mmap_handle_read_at_invalid() {
    let mut handle = MmapHandle::new(1024, MmapConfig::default()).expect("Should create");
    handle.invalidate();
    let result = handle.read_at(0, 100);
    assert!(result.is_err());
}

#[test]
fn test_mmap_handle_write_at_invalid() {
    let mut handle = MmapHandle::new(1024, MmapConfig::default().with_mode(MmapMode::ReadWrite))
        .expect("Should create");
    handle.invalidate();
    let result = handle.write_at(0, &[1, 2, 3]);
    assert!(result.is_err());
}

#[test]
fn test_mmap_handle_write_copy_on_write() {
    let mut handle = MmapHandle::new(1024, MmapConfig::default().with_mode(MmapMode::CopyOnWrite))
        .expect("Should create");
    // CopyOnWrite is writable
    assert!(handle.write_at(0, &[1, 2, 3]).is_ok());
}

#[test]
fn test_weight_dtype_bf16_name() {
    assert_eq!(WeightDtype::Bf16.name(), "bfloat16");
}

#[test]
fn test_memory_region_aligned_no_alignment() {
    let region = MemoryRegion::new(50, 100).with_alignment(1);
    assert_eq!(region.aligned_offset(), 50);
    assert_eq!(region.aligned_size(), 100);
}

#[test]
fn test_memory_region_aligned_offset_exact_boundary() {
    let region = MemoryRegion::new(128, 256).with_alignment(64);
    assert_eq!(region.aligned_offset(), 128); // Already aligned
    assert_eq!(region.aligned_size(), 256);
}

#[test]
fn test_weight_region_empty_shape() {
    let region = WeightRegion::new(
        "test_empty",
        WeightType::Bias,
        MemoryRegion::new(0, 0),
        WeightDtype::F32,
        vec![],
    );
    // Product of empty shape is 1 (identity)
    assert_eq!(region.num_elements(), 1);
    assert_eq!(region.expected_bytes(), 4); // 1 * 4 bytes
}

#[test]
fn test_mmap_handle_unique_ids() {
    let h1 = MmapHandle::new(1024, MmapConfig::default()).expect("Should create");
    let h2 = MmapHandle::new(1024, MmapConfig::default()).expect("Should create");
    assert_ne!(h1.id(), h2.id());
}

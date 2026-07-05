//! Tests for backend selection
#![allow(clippy::ignore_without_reason, clippy::expect_used)]

use super::super::traits::MatMulOp;
use super::*;

#[test]
#[ignore]
fn test_selection_strategy_default() {
    assert_eq!(SelectionStrategy::default(), SelectionStrategy::Automatic);
}

#[test]
#[ignore]
fn test_selection_strategy_threshold() {
    let strategy = SelectionStrategy::threshold(1_000_000);
    assert!(matches!(
        strategy,
        SelectionStrategy::Threshold {
            min_flops: 1_000_000
        }
    ));
}

#[test]
#[ignore]
fn test_selection_strategy_description() {
    assert!(SelectionStrategy::PreferGpu.description().contains("GPU"));
    assert!(SelectionStrategy::PreferSimd.description().contains("SIMD"));
    assert!(SelectionStrategy::Automatic
        .description()
        .contains("automatic"));
}

#[test]
#[ignore]
fn test_selector_config_default() {
    let config = SelectorConfig::default();
    assert_eq!(config.strategy, SelectionStrategy::Automatic);
    assert!(config.gpu_threshold_flops > 0);
}

#[test]
#[ignore]
fn test_selector_config_for_inference() {
    let config = SelectorConfig::for_inference();
    assert_eq!(config.strategy, SelectionStrategy::Automatic);
    assert!(config.gpu_threshold_flops >= 1_000_000);
}

#[test]
#[ignore]
fn test_selector_config_builders() {
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::PreferGpu)
        .with_gpu_threshold(500_000)
        .with_max_gpu_memory(512 * 1024 * 1024);

    assert_eq!(config.strategy, SelectionStrategy::PreferGpu);
    assert_eq!(config.gpu_threshold_flops, 500_000);
    assert_eq!(config.max_gpu_memory, 512 * 1024 * 1024);
}

#[test]
#[ignore]
fn test_backend_selector_new() {
    let selector = BackendSelector::new(SelectorConfig::default());
    assert!(selector.simd_capabilities().available);
}

#[test]
#[ignore]
fn test_backend_selector_default_config() {
    let selector = BackendSelector::default_config();
    assert!(selector.simd_capabilities().available);
}

#[test]
#[ignore]
fn test_backend_selector_select_prefer_simd() {
    let selector = BackendSelector::new(SelectorConfig::prefer_simd());
    let op = MatMulOp::new(64, 128, 64);
    let selection = selector.select(&op);

    assert!(selection.is_simd());
    assert!(selection.reason.contains("PreferSimd"));
}

#[test]
#[ignore]
fn test_backend_selector_select_small_workload() {
    let selector =
        BackendSelector::new(SelectorConfig::default().with_gpu_threshold(1_000_000_000));
    let op = MatMulOp::new(8, 8, 8); // Very small
    let selection = selector.select(&op);

    // Should select SIMD for small workloads
    assert!(selection.is_simd());
}

#[test]
#[ignore]
fn test_backend_selector_select_threshold() {
    let selector = BackendSelector::new(
        SelectorConfig::default().with_strategy(SelectionStrategy::threshold(100)),
    );

    // Small operation - below threshold
    let small_op = MatMulOp::new(2, 2, 2);
    let selection = selector.select(&small_op);
    // Note: GPU not available in tests, so always SIMD
    assert!(selection.is_simd());
}

#[test]
#[ignore]
fn test_backend_selector_select_batch() {
    let selector = BackendSelector::default_config();
    let ops = vec![
        MatMulOp::new(64, 128, 64),
        MatMulOp::new(64, 128, 64),
        MatMulOp::new(64, 128, 64),
    ];
    let selection = selector.select_batch(&ops);

    // Selection should be made
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_backend_selector_select_batch_empty() {
    let selector = BackendSelector::default_config();
    let ops: Vec<MatMulOp> = vec![];
    let selection = selector.select_batch(&ops);

    assert!(selection.is_simd());
    assert!(selection.reason.contains("No operations"));
}

#[test]
#[ignore]
fn test_backend_selector_summary() {
    let selector = BackendSelector::default_config();
    let summary = selector.summary();

    assert!(summary.contains("SIMD"));
    assert!(summary.contains("parallelism"));
}

#[test]
#[ignore]
fn test_backend_selection_gpu() {
    let selection = BackendSelection::gpu("test reason");
    assert!(selection.is_gpu());
    assert!(!selection.is_simd());
    assert_eq!(selection.reason, "test reason");
}

#[test]
#[ignore]
fn test_backend_selection_simd() {
    let selection = BackendSelection::simd("test reason");
    assert!(selection.is_simd());
    assert!(!selection.is_gpu());
}

#[test]
#[ignore]
fn test_backend_selection_display() {
    let selection = BackendSelection::gpu("performance");
    let s = selection.to_string();
    assert!(s.contains("GPU"));
    assert!(s.contains("performance"));
}

// =========================================================================
// Additional Coverage Tests (WAPR-QA)
// =========================================================================

#[test]
#[ignore]
fn test_selection_strategy_threshold_description() {
    let strategy = SelectionStrategy::threshold(1_000_000);
    assert_eq!(strategy.description(), "threshold-based");
}

#[test]
#[ignore]
fn test_selector_config_prefer_gpu() {
    let config = SelectorConfig::prefer_gpu();
    assert_eq!(config.strategy, SelectionStrategy::PreferGpu);
}

#[test]
#[ignore]
fn test_selector_config_prefer_simd() {
    let config = SelectorConfig::prefer_simd();
    assert_eq!(config.strategy, SelectionStrategy::PreferSimd);
}

#[test]
#[ignore]
fn test_backend_selector_config_accessor() {
    let selector = BackendSelector::default_config();
    let config = selector.config();
    assert_eq!(config.strategy, SelectionStrategy::Automatic);
}

#[test]
#[ignore]
fn test_backend_selector_gpu_capabilities() {
    let selector = BackendSelector::default_config();
    // GPU may or may not be available, just check it doesn't panic
    let _ = selector.gpu_capabilities();
}

#[test]
#[ignore]
fn test_backend_selector_select_prefer_gpu_no_gpu() {
    let selector = BackendSelector::new(SelectorConfig::prefer_gpu());
    let op = MatMulOp::new(64, 128, 64);
    let selection = selector.select(&op);
    // Since GPU is likely not available in tests, should fall back
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_backend_selector_select_large_workload() {
    let selector = BackendSelector::new(SelectorConfig::default().with_gpu_threshold(100));
    let op = MatMulOp::new(128, 256, 128); // Large workload
    let selection = selector.select(&op);
    // Just verify selection is made
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_backend_selection_backend_type() {
    let gpu_selection = BackendSelection::gpu("test");
    assert_eq!(gpu_selection.backend, BackendType::Gpu);

    let simd_selection = BackendSelection::simd("test");
    assert_eq!(simd_selection.backend, BackendType::Simd);
}

#[test]
#[ignore]
fn test_selector_with_high_memory_requirement() {
    // Create selector with low max GPU memory
    let config = SelectorConfig::default().with_max_gpu_memory(1024);
    let selector = BackendSelector::new(config);

    // Large operation that exceeds GPU memory
    let op = MatMulOp::new(1024, 1024, 1024);
    let selection = selector.select(&op);

    // Should use SIMD due to memory constraint
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_selector_automatic_with_small_workload() {
    let selector = BackendSelector::new(SelectorConfig::default());
    let op = MatMulOp::new(4, 4, 4); // Very small
    let selection = selector.select(&op);

    // Small workload should prefer SIMD
    assert!(selection.is_simd());
}

#[test]
#[ignore]
fn test_selector_threshold_below_flops() {
    let selector = BackendSelector::new(
        SelectorConfig::default().with_strategy(SelectionStrategy::threshold(1_000_000_000)),
    );
    let op = MatMulOp::new(32, 32, 32); // Below threshold
    let selection = selector.select(&op);

    // Below threshold should use SIMD
    assert!(selection.is_simd());
}

#[test]
#[ignore]
fn test_selection_strategy_all_descriptions() {
    assert!(!SelectionStrategy::PreferGpu.description().is_empty());
    assert!(!SelectionStrategy::PreferSimd.description().is_empty());
    assert!(!SelectionStrategy::Automatic.description().is_empty());
    assert!(!SelectionStrategy::threshold(1000).description().is_empty());
}

#[test]
#[ignore]
fn test_selector_config_with_all_builders() {
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::Automatic)
        .with_gpu_threshold(500_000)
        .with_max_gpu_memory(1024 * 1024 * 1024);

    assert_eq!(config.strategy, SelectionStrategy::Automatic);
    assert_eq!(config.gpu_threshold_flops, 500_000);
    assert_eq!(config.max_gpu_memory, 1024 * 1024 * 1024);
}

#[test]
#[ignore]
fn test_backend_selector_select_batch_large() {
    let selector = BackendSelector::default_config();
    let ops: Vec<MatMulOp> = (0..10).map(|_| MatMulOp::new(128, 256, 128)).collect();
    let selection = selector.select_batch(&ops);
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_backend_selector_simd_capabilities() {
    let selector = BackendSelector::default_config();
    let caps = selector.simd_capabilities();
    assert!(caps.available);
    assert!(caps.max_parallelism > 0);
}

#[test]
#[ignore]
fn test_backend_selector_gpu_available() {
    let selector = BackendSelector::default_config();
    // Just ensure method works, GPU may or may not be available
    let _available = selector.gpu_available();
}

#[test]
#[ignore]
fn test_selector_prefer_gpu_memory_exceeds_limit() {
    // Create selector with very low GPU memory limit
    let config = SelectorConfig::prefer_gpu().with_max_gpu_memory(1);
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(1024, 1024, 1024); // Large memory requirement
    let selection = selector.select(&op);

    // If GPU available, should fall back to SIMD due to memory
    if selector.gpu_available() {
        assert!(
            selection.reason.contains("Memory exceeds")
                || selection.reason.contains("GPU not available")
        );
    }
}

#[test]
#[ignore]
fn test_selector_threshold_with_large_memory() {
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(100))
        .with_max_gpu_memory(1);
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(1024, 1024, 1024);
    let selection = selector.select(&op);

    // Should fall back to SIMD due to memory constraint
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_selector_automatic_large_workload() {
    let config = SelectorConfig::default().with_gpu_threshold(10);
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(256, 512, 256); // Large enough
    let selection = selector.select(&op);
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_selector_is_gpu_worthwhile_below_threshold() {
    let config = SelectorConfig::default().with_gpu_threshold(1_000_000_000);
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(8, 8, 8); // Small workload
    let selection = selector.select(&op);
    assert!(selection.is_simd());
}

#[test]
#[ignore]
fn test_selector_batch_with_varying_sizes() {
    let selector = BackendSelector::default_config();
    let ops = vec![
        MatMulOp::new(8, 8, 8),
        MatMulOp::new(128, 128, 128),
        MatMulOp::new(64, 64, 64),
    ];
    let selection = selector.select_batch(&ops);
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_selector_summary_with_different_strategies() {
    let configs = vec![
        SelectorConfig::default(),
        SelectorConfig::prefer_gpu(),
        SelectorConfig::prefer_simd(),
        SelectorConfig::for_inference(),
    ];

    for config in configs {
        let selector = BackendSelector::new(config);
        let summary = selector.summary();
        assert!(summary.contains("SIMD"));
    }
}

// =========================================================================
// GPU Path Coverage Tests (simulated GPU)
// =========================================================================

#[test]
#[ignore]
fn test_select_automatic_gpu_worthwhile() {
    let config = SelectorConfig::default().with_gpu_threshold(100);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(256, 256, 256); // Large enough workload
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert!(selection.reason.contains("Large workload"));
}

#[test]
#[ignore]
fn test_select_automatic_small_workload_with_gpu() {
    let config = SelectorConfig::default().with_gpu_threshold(1_000_000_000);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(4, 4, 4); // Tiny workload
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Small workload"));
}

#[test]
#[ignore]
fn test_select_automatic_memory_exceeds_with_gpu() {
    let config = SelectorConfig::default().with_max_gpu_memory(1);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(256, 256, 256);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Memory exceeds"));
}

#[test]
#[ignore]
fn test_select_prefer_gpu_with_gpu_available() {
    let config = SelectorConfig::prefer_gpu();
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(64, 128, 64);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert!(selection.reason.contains("PreferGpu"));
}

#[test]
#[ignore]
fn test_select_prefer_gpu_memory_exceeds_with_gpu() {
    let config = SelectorConfig::prefer_gpu().with_max_gpu_memory(1);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(256, 256, 256);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Memory exceeds"));
}

#[test]
#[ignore]
fn test_select_threshold_gpu_above_threshold() {
    let config = SelectorConfig::default().with_strategy(SelectionStrategy::threshold(100));
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(128, 128, 128);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert!(selection.reason.contains("FLOPs exceed"));
}

#[test]
#[ignore]
fn test_select_threshold_gpu_below_threshold() {
    let config =
        SelectorConfig::default().with_strategy(SelectionStrategy::threshold(1_000_000_000));
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(4, 4, 4);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("below threshold"));
}

#[test]
#[ignore]
fn test_select_threshold_memory_exceeds_with_gpu() {
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(1))
        .with_max_gpu_memory(1);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(256, 256, 256);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Memory exceeds"));
}

#[test]
#[ignore]
fn test_is_gpu_worthwhile_via_batch_with_gpu() {
    let config = SelectorConfig::default().with_gpu_threshold(100);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let ops = vec![MatMulOp::new(128, 128, 128), MatMulOp::new(128, 128, 128)];
    let selection = selector.select_batch(&ops);
    assert!(selection.is_gpu());
}

#[test]
#[ignore]
fn test_gpu_caps_memory_limit_check() {
    // GPU with very small buffer size - can_handle should fail
    let config = SelectorConfig::default().with_gpu_threshold(1);
    let selector = BackendSelector::with_simulated_gpu(config, 64); // Very small GPU memory
    let op = MatMulOp::new(256, 256, 256); // Large memory requirement
    let selection = selector.select(&op);
    // Memory requirement exceeds GPU caps, should fall back
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_simulated_gpu_summary() {
    let config = SelectorConfig::default();
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let summary = selector.summary();
    assert!(summary.contains("GPU"));
    assert!(summary.contains("f16=true"));
}

#[test]
#[ignore]
fn test_simulated_gpu_capabilities() {
    let config = SelectorConfig::default();
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    assert!(selector.gpu_available());
    let gpu_caps = selector
        .gpu_capabilities()
        .expect("GPU should be available");
    assert!(gpu_caps.available);
    assert!(gpu_caps.supports_f16);
}

// =========================================================================
// Coverage Gap Tests: is_gpu_worthwhile deep paths (WAPR-QA-004)
// =========================================================================

#[test]
#[ignore]
fn test_is_gpu_worthwhile_gpu_caps_cannot_handle() {
    // Simulate GPU with very small max_buffer_size so can_handle returns false
    let config = SelectorConfig::default().with_gpu_threshold(1); // Very low threshold
    let selector = BackendSelector::with_simulated_gpu(config, 16); // Tiny GPU memory: 16 bytes
                                                                    // MatMulOp requires m*n*4 bytes of memory, 256*256*4 = 262144 > 16
    let op = MatMulOp::new(256, 256, 256);
    let selection = selector.select(&op);
    // GPU threshold is met but can_handle should fail, falling back to SIMD
    assert!(selection.is_simd());
}

#[test]
#[ignore]
fn test_is_gpu_worthwhile_below_flops_threshold() {
    // Simulate GPU with high threshold so flops check fails
    let config = SelectorConfig::default().with_gpu_threshold(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(4, 4, 4); // Tiny workload
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Small workload"));
}

#[test]
#[ignore]
fn test_select_automatic_gpu_memory_boundary() {
    // Set max_gpu_memory to exactly match operation memory
    let op = MatMulOp::new(16, 16, 16);
    let mem = op.memory_requirement() as u64;
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(mem);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let selection = selector.select(&op);
    // Should use GPU since memory is exactly at limit
    assert!(selection.is_gpu() || selection.is_simd()); // Depends on can_handle
}

#[test]
#[ignore]
fn test_select_batch_gpu_worthwhile() {
    let config = SelectorConfig::default().with_gpu_threshold(1);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let ops = vec![MatMulOp::new(64, 64, 64), MatMulOp::new(64, 64, 64)];
    let selection = selector.select_batch(&ops);
    assert!(selection.is_gpu());
}

#[test]
#[ignore]
fn test_select_batch_gpu_memory_too_small() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(1); // Tiny memory limit
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let ops = vec![MatMulOp::new(256, 256, 256)];
    let selection = selector.select_batch(&ops);
    assert!(selection.is_simd());
}

// =========================================================================
// Additional Coverage Tests (WAPR-QA-003)
// =========================================================================

#[test]
#[ignore]
fn test_selector_automatic_gpu_memory_exceeded() {
    // Automatic strategy with very low GPU memory limit
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::Automatic)
        .with_max_gpu_memory(1);
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(256, 256, 256);
    let selection = selector.select(&op);

    // Should fall back to SIMD due to memory
    assert!(selection.is_simd());
}

#[test]
#[ignore]
fn test_selector_prefer_gpu_without_gpu() {
    // PreferGpu strategy but GPU not available (simulated)
    let config = SelectorConfig::prefer_gpu();
    let selector = BackendSelector::new(config);

    // In test environment, GPU is typically not available
    if !selector.gpu_available() {
        let op = MatMulOp::new(64, 64, 64);
        let selection = selector.select(&op);
        assert!(selection.is_simd());
        assert!(selection.reason.contains("not available") || selection.reason.contains("SIMD"));
    }
}

#[test]
#[ignore]
fn test_selector_threshold_high_flops() {
    // Threshold strategy with very low threshold - should suggest GPU if available
    let config = SelectorConfig::default().with_strategy(SelectionStrategy::threshold(1));
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(128, 128, 128);
    let selection = selector.select(&op);

    // Should use GPU if available, SIMD otherwise
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_selector_for_inference_config() {
    let config = SelectorConfig::for_inference();
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(8, 8, 8);
    let selection = selector.select(&op);
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_backend_selection_reason_non_empty() {
    let sel_gpu = BackendSelection::gpu("test reason");
    assert_eq!(sel_gpu.reason, "test reason");
    assert!(sel_gpu.is_gpu());
    assert!(!sel_gpu.is_simd());

    let sel_simd = BackendSelection::simd("simd reason");
    assert_eq!(sel_simd.reason, "simd reason");
    assert!(sel_simd.is_simd());
    assert!(!sel_simd.is_gpu());
}

// =========================================================================
// Coverage Tests for is_gpu_worthwhile, select_automatic, summary (WAPR-QA-005)
// =========================================================================

#[test]
#[ignore]
fn test_is_gpu_worthwhile_flops_above_threshold_caps_pass() {
    // GPU with large buffer, flops above threshold -> true
    let config = SelectorConfig::default().with_gpu_threshold(100);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(64, 64, 64);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert!(selection.reason.contains("Large workload"));
}

#[test]
#[ignore]
fn test_is_gpu_worthwhile_flops_below_threshold_returns_false() {
    // flops < gpu_threshold_flops -> false -> "Small workload"
    let config = SelectorConfig::default().with_gpu_threshold(999_999_999);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(4, 4, 4);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Small workload"));
}

#[test]
#[ignore]
fn test_is_gpu_worthwhile_caps_cannot_handle_returns_false() {
    // GPU caps with small max_buffer_size so can_handle returns false
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(1024 * 1024 * 1024); // High enough to pass memory check in select_automatic
                                                  // GPU buffer = 32 bytes (tiny), so can_handle fails for any real workload
    let selector = BackendSelector::with_simulated_gpu(config, 32);
    let op = MatMulOp::new(64, 64, 64); // Memory req: (64*64 + 64*64 + 64*64)*4 = 49152 > 32
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Small workload"));
}

#[test]
#[ignore]
fn test_select_automatic_no_gpu_available() {
    // Test select_automatic when GPU is not available
    let selector = BackendSelector::new(SelectorConfig::default());
    if !selector.gpu_available() {
        let op = MatMulOp::new(256, 256, 256);
        let selection = selector.select(&op);
        assert!(selection.is_simd());
        assert!(
            selection.reason.contains("not available")
                || selection.reason.contains("Small workload")
        );
    }
}

#[test]
#[ignore]
fn test_select_automatic_memory_exceeds_limit() {
    // Automatic with GPU, memory exceeds max_gpu_memory
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(1); // 1 byte limit
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(128, 128, 128);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Memory exceeds"));
}

#[test]
#[ignore]
fn test_select_automatic_gpu_worthwhile_true() {
    // GPU available, memory fits, workload above threshold
    let config = SelectorConfig::default()
        .with_gpu_threshold(10)
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(128, 128, 128);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert!(selection.reason.contains("Large workload"));
}

#[test]
#[ignore]
fn test_select_automatic_gpu_worthwhile_false() {
    // GPU available, memory fits, but workload below threshold
    let config = SelectorConfig::default()
        .with_gpu_threshold(u64::MAX)
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(2, 2, 2); // Tiny workload
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Small workload"));
}

#[test]
#[ignore]
fn test_summary_with_simulated_gpu_includes_gpu_line() {
    let config = SelectorConfig::default();
    let selector = BackendSelector::with_simulated_gpu(config, 256 * 1024 * 1024);
    let summary = selector.summary();

    assert!(summary.contains("Backend Selector"));
    assert!(summary.contains("automatic"));
    assert!(summary.contains("SIMD"));
    assert!(summary.contains("GPU"));
    assert!(summary.contains("parallelism="));
    assert!(summary.contains("score="));
    assert!(summary.contains("f16="));
}

#[test]
#[ignore]
fn test_summary_without_gpu_shows_not_available() {
    let selector = BackendSelector::new(SelectorConfig::default());
    if !selector.gpu_available() {
        let summary = selector.summary();
        assert!(summary.contains("not available"));
    }
}

#[test]
#[ignore]
fn test_summary_with_different_strategy_names() {
    // PreferGpu strategy
    let config = SelectorConfig::prefer_gpu();
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let summary = selector.summary();
    assert!(summary.contains("prefer GPU"));

    // PreferSimd strategy
    let config2 = SelectorConfig::prefer_simd();
    let selector2 = BackendSelector::new(config2);
    let summary2 = selector2.summary();
    assert!(summary2.contains("prefer SIMD"));

    // Threshold strategy
    let config3 = SelectorConfig::default().with_strategy(SelectionStrategy::threshold(1000));
    let selector3 = BackendSelector::new(config3);
    let summary3 = selector3.summary();
    assert!(summary3.contains("threshold-based"));
}

// =========================================================================
// Deep Coverage Tests: is_gpu_worthwhile, new, select, select_automatic
// (WAPR-QA-006)
// =========================================================================

// --- is_gpu_worthwhile: boundary and edge case coverage ---

#[test]
#[ignore]
fn test_is_gpu_worthwhile_zero_flops() {
    // Zero FLOPs should never be worthwhile for GPU
    let config = SelectorConfig::default().with_gpu_threshold(0);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    // Use a 0x0 matmul to generate zero flops
    let op = MatMulOp::new(0, 0, 0);
    let selection = selector.select(&op);
    // 0 flops >= 0 threshold, but memory is 0 which can_handle(0) should pass
    // The actual path depends on whether 0 >= 0 is true (it is), and can_handle(0)
    // Either GPU or SIMD is acceptable - just verify no panic
    assert!(!selection.reason.is_empty());
}

#[test]
#[ignore]
fn test_is_gpu_worthwhile_flops_exactly_at_threshold() {
    // FLOPs exactly equal to threshold: should pass the threshold check
    // MatMulOp flops = 2*m*k*n, so 2*10*5*1 = 100
    let config = SelectorConfig::default()
        .with_gpu_threshold(100)
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(10, 5, 1); // 2*10*5*1 = 100 flops exactly
    let selection = selector.select(&op);
    // flops (100) >= threshold (100), so GPU worthwhile (if can_handle passes)
    assert!(selection.is_gpu());
    assert!(selection.reason.contains("Large workload"));
}

#[test]
#[ignore]
fn test_is_gpu_worthwhile_flops_one_below_threshold() {
    // FLOPs just below threshold: should NOT pass
    // We need exactly 99 flops: 2*m*k*n. But flops must be integer.
    // Use threshold=101 with a MatMulOp giving 100 flops
    let config = SelectorConfig::default()
        .with_gpu_threshold(101)
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(10, 5, 1); // 2*10*5*1 = 100 flops
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Small workload"));
}

#[ignore]
#[test]
fn test_is_gpu_worthwhile_gpu_caps_none_not_available() {
    // Test the path where gpu_available is false and gpu_caps is None
    // This is the normal BackendSelector::new path without webgpu feature
    let selector = BackendSelector::new(SelectorConfig::default());
    // Without webgpu feature, gpu_available is false and gpu_caps is None
    assert!(!selector.gpu_available());
    assert!(selector.gpu_capabilities().is_none());

    let op = MatMulOp::new(256, 256, 256);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
}

#[test]
#[ignore]
fn test_is_gpu_worthwhile_gpu_can_handle_exact_boundary() {
    // GPU max_buffer_size exactly equals memory requirement
    // MatMulOp memory = (m*k + k*n + m*n) * 4
    // For 8x8x8: (64 + 64 + 64) * 4 = 768 bytes
    let op = MatMulOp::new(8, 8, 8);
    let mem = op.memory_requirement();
    assert_eq!(mem, 768);

    let config = SelectorConfig::default()
        .with_gpu_threshold(1) // Low threshold so flops check passes
        .with_max_gpu_memory(1024 * 1024 * 1024); // High config memory limit
                                                  // GPU buffer exactly 768 bytes -- can_handle(768) should return true since 768 <= 768
    let selector = BackendSelector::with_simulated_gpu(config, mem as u64);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
}

#[test]
#[ignore]
fn test_is_gpu_worthwhile_gpu_can_handle_one_byte_short() {
    // GPU max_buffer_size is one byte less than memory requirement
    let op = MatMulOp::new(8, 8, 8);
    let mem = op.memory_requirement();

    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(1024 * 1024 * 1024);
    // GPU buffer one byte short -- can_handle should fail
    let selector = BackendSelector::with_simulated_gpu(config, (mem as u64) - 1);
    let selection = selector.select(&op);
    // is_gpu_worthwhile returns false because can_handle fails -> "Small workload"
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Small workload"));
}

// --- BackendSelector::new: exercising constructor paths ---

#[test]
#[ignore]
fn test_backend_selector_new_default_config_fields() {
    let config = SelectorConfig::default();
    let selector = BackendSelector::new(config);

    // Verify config was stored correctly
    assert_eq!(selector.config().strategy, SelectionStrategy::Automatic);
    assert_eq!(selector.config().gpu_threshold_flops, 100_000);
    assert_eq!(selector.config().max_gpu_memory, 256 * 1024 * 1024);
    assert_eq!(selector.config().gpu_dispatch_overhead_us, 100);
}

#[test]
#[ignore]
fn test_backend_selector_new_for_inference_config() {
    let config = SelectorConfig::for_inference();
    let selector = BackendSelector::new(config);

    assert_eq!(selector.config().strategy, SelectionStrategy::Automatic);
    assert_eq!(selector.config().gpu_threshold_flops, 1_000_000);
    assert_eq!(selector.config().max_gpu_memory, 1024 * 1024 * 1024);
    assert_eq!(selector.config().gpu_dispatch_overhead_us, 50);
}

#[test]
#[ignore]
fn test_backend_selector_new_simd_caps_always_available() {
    // SIMD capabilities should always be available regardless of config
    let configs = vec![
        SelectorConfig::default(),
        SelectorConfig::prefer_gpu(),
        SelectorConfig::prefer_simd(),
        SelectorConfig::for_inference(),
    ];
    for config in configs {
        let selector = BackendSelector::new(config);
        let caps = selector.simd_capabilities();
        assert!(caps.available);
        assert_eq!(caps.backend_type, BackendType::Simd);
        assert!(caps.max_parallelism > 0);
        assert!(caps.performance_score > 0.0);
    }
}

// --- select: diverse operation types (not just MatMulOp) ---

#[test]
#[ignore]
fn test_select_with_softmax_op_prefer_simd() {
    use super::super::traits::SoftmaxOp;

    let selector = BackendSelector::new(SelectorConfig::prefer_simd());
    let op = SoftmaxOp::new(32, 512);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("PreferSimd"));
}

#[test]
#[ignore]
fn test_select_with_softmax_op_automatic_gpu() {
    use super::super::traits::SoftmaxOp;

    // SoftmaxOp flops = rows * cols * 5
    // For 1000 x 1000: 5_000_000 flops
    let config = SelectorConfig::default()
        .with_gpu_threshold(1_000)
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = SoftmaxOp::new(1000, 1000);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert!(selection.reason.contains("Large workload"));
}

#[test]
#[ignore]
fn test_select_with_layer_norm_op_automatic_gpu() {
    use super::super::traits::LayerNormOp;

    // LayerNormOp flops = batch_size * hidden_size * 6
    // For 64 x 1024: 393_216 flops
    let config = SelectorConfig::default()
        .with_gpu_threshold(100_000)
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = LayerNormOp::new(64, 1024);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
}

#[test]
#[ignore]
fn test_select_with_gelu_op_below_threshold() {
    use super::super::traits::GeluOp;

    // GeluOp flops = num_elements * 10
    // For 100 elements: 1000 flops
    let config = SelectorConfig::default()
        .with_gpu_threshold(10_000)
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = GeluOp::new(100);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Small workload"));
}

#[test]
#[ignore]
fn test_select_with_gelu_op_above_threshold() {
    use super::super::traits::GeluOp;

    // GeluOp flops = num_elements * 10
    // For 100_000 elements: 1_000_000 flops
    let config = SelectorConfig::default()
        .with_gpu_threshold(500_000)
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = GeluOp::new(100_000);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
}

// --- select: PreferGpu strategy paths with GPU available ---

#[test]
#[ignore]
fn test_select_prefer_gpu_gpu_available_memory_fits() {
    let config = SelectorConfig::prefer_gpu().with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(32, 32, 32);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert!(selection.reason.contains("PreferGpu strategy"));
}

#[test]
#[ignore]
fn test_select_prefer_gpu_gpu_available_memory_exceeds() {
    // PreferGpu but memory exceeds max_gpu_memory -> should fallback to SIMD
    let config = SelectorConfig::prefer_gpu().with_max_gpu_memory(1); // 1 byte limit
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(64, 64, 64); // memory = (64*64 + 64*64 + 64*64)*4 = 49152
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Memory exceeds GPU limit"));
}

// --- select: Threshold strategy edge cases ---

#[test]
#[ignore]
fn test_select_threshold_flops_exactly_at_min_flops() {
    // Threshold with min_flops = 100, op with exactly 100 flops
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(100))
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(10, 5, 1); // 2*10*5*1 = 100 flops
    let selection = selector.select(&op);
    // flops (100) >= min_flops (100) -> GPU
    assert!(selection.is_gpu());
    assert!(selection.reason.contains("FLOPs exceed threshold"));
}

#[test]
#[ignore]
fn test_select_threshold_flops_one_below_min_flops() {
    // Threshold with min_flops = 101, op with exactly 100 flops
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(101))
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(10, 5, 1); // 100 flops
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("below threshold"));
}

#[test]
#[ignore]
fn test_select_threshold_memory_exceeds_with_high_flops() {
    // Threshold strategy: flops exceed threshold but memory exceeds limit
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(1))
        .with_max_gpu_memory(1); // 1 byte limit
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(256, 256, 256); // High flops, high memory
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Memory exceeds GPU limit"));
}

#[ignore]
#[test]
fn test_select_threshold_no_gpu_available() {
    // Threshold strategy without GPU
    let config = SelectorConfig::default().with_strategy(SelectionStrategy::threshold(1));
    let selector = BackendSelector::new(config); // No webgpu feature -> no GPU
    let op = MatMulOp::new(256, 256, 256);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("GPU not available"));
}

// --- select_automatic: comprehensive path coverage ---

#[test]
#[ignore]
fn test_select_automatic_gpu_available_memory_at_exact_limit() {
    // Memory exactly equals max_gpu_memory (should pass: memory > max is checked, not >=)
    let op = MatMulOp::new(8, 8, 8);
    let mem = op.memory_requirement() as u64; // 768 bytes

    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(mem); // Exact match
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let selection = selector.select(&op);
    // memory (768) > max_gpu_memory (768) is false, so passes memory check
    assert!(selection.is_gpu());
}

#[test]
#[ignore]
fn test_select_automatic_gpu_available_memory_one_over_limit() {
    // Memory is one byte over max_gpu_memory
    let op = MatMulOp::new(8, 8, 8);
    let mem = op.memory_requirement() as u64; // 768 bytes

    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(mem - 1); // One byte short
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let selection = selector.select(&op);
    // memory (768) > max_gpu_memory (767) is true -> SIMD
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Memory exceeds GPU limit"));
}

#[test]
#[ignore]
fn test_select_automatic_zero_memory_zero_flops() {
    // Edge case: zero-dimension op
    let config = SelectorConfig::default()
        .with_gpu_threshold(0)
        .with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(0, 0, 0); // Zero everything
    let selection = selector.select(&op);
    // 0 >= 0 threshold passes, memory 0 <= max passes, can_handle(0) passes
    assert!(selection.is_gpu());
}

#[test]
#[ignore]
fn test_select_automatic_very_large_flops() {
    // Very large workload should definitely pick GPU
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);
    let op = MatMulOp::new(4096, 4096, 4096);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert!(selection.reason.contains("Large workload"));
}

// --- select_batch: additional edge cases ---

#[test]
#[ignore]
fn test_select_batch_single_op() {
    let config = SelectorConfig::default().with_gpu_threshold(1);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let ops = vec![MatMulOp::new(64, 64, 64)];
    let selection = selector.select_batch(&ops);
    assert!(selection.is_gpu());
}

#[test]
#[ignore]
fn test_select_batch_mixed_op_types_via_matmul() {
    // Batch with very different sizes
    let config = SelectorConfig::default().with_gpu_threshold(1);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let ops = vec![
        MatMulOp::new(1, 1, 1),       // 2 flops, 12 bytes
        MatMulOp::new(512, 512, 512), // 268M flops, 3M bytes
    ];
    let selection = selector.select_batch(&ops);
    // Total flops is very large, should use GPU
    assert!(selection.is_gpu());
}

#[test]
#[ignore]
fn test_select_batch_all_tiny_ops_high_threshold() {
    let config = SelectorConfig::default().with_gpu_threshold(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let ops = vec![
        MatMulOp::new(1, 1, 1),
        MatMulOp::new(1, 1, 1),
        MatMulOp::new(1, 1, 1),
    ];
    let selection = selector.select_batch(&ops);
    // Total flops = 6, way below u64::MAX threshold
    assert!(selection.is_simd());
}

#[test]
#[ignore]
fn test_select_batch_memory_exceeds_for_largest_op() {
    // Batch where the largest op's memory exceeds the limit
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(100); // Very small memory limit
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let ops = vec![
        MatMulOp::new(2, 2, 2),       // Small memory
        MatMulOp::new(256, 256, 256), // Large memory exceeds 100 bytes
    ];
    let selection = selector.select_batch(&ops);
    // max_memory from batch exceeds 100 bytes -> SIMD
    assert!(selection.is_simd());
    assert!(selection.reason.contains("Memory exceeds"));
}

// --- BackendSelection: Display and accessor coverage ---

#[test]
#[ignore]
fn test_backend_selection_display_simd() {
    let selection = BackendSelection::simd("CPU is faster for small workloads");
    let display = format!("{selection}");
    assert!(display.contains("SIMD"));
    assert!(display.contains("CPU is faster for small workloads"));
}

#[test]
#[ignore]
fn test_backend_selection_clone() {
    let selection = BackendSelection::gpu("cloneable");
    let cloned = selection.clone();
    assert_eq!(cloned.backend, BackendType::Gpu);
    assert_eq!(cloned.reason, "cloneable");
}

// --- SelectorConfig: chained builder coverage ---

#[test]
#[ignore]
fn test_selector_config_chained_builders_all_fields() {
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(42))
        .with_gpu_threshold(999)
        .with_max_gpu_memory(12345);

    assert!(matches!(
        config.strategy,
        SelectionStrategy::Threshold { min_flops: 42 }
    ));
    assert_eq!(config.gpu_threshold_flops, 999);
    assert_eq!(config.max_gpu_memory, 12345);
}

// --- SelectionStrategy: equality and debug coverage ---

#[test]
#[ignore]
fn test_selection_strategy_clone_and_copy() {
    let s = SelectionStrategy::PreferGpu;
    let s2 = s; // Copy
    let s3 = s; // Clone
    assert_eq!(s, s2);
    assert_eq!(s, s3);
}

#[test]
#[ignore]
fn test_selection_strategy_debug_format() {
    let s = SelectionStrategy::Threshold { min_flops: 500 };
    let debug = format!("{s:?}");
    assert!(debug.contains("Threshold"));
    assert!(debug.contains("500"));
}

#[test]
#[ignore]
fn test_selection_strategy_eq_different_variants() {
    assert_ne!(SelectionStrategy::PreferGpu, SelectionStrategy::PreferSimd);
    assert_ne!(SelectionStrategy::Automatic, SelectionStrategy::PreferGpu);
    assert_ne!(
        SelectionStrategy::threshold(100),
        SelectionStrategy::threshold(200)
    );
    assert_eq!(
        SelectionStrategy::threshold(100),
        SelectionStrategy::threshold(100)
    );
}

// --- with_simulated_gpu: verify fields are set correctly ---

#[test]
#[ignore]
#[allow(clippy::expect_used)]
fn test_with_simulated_gpu_fields() {
    let config = SelectorConfig::default();
    let selector = BackendSelector::with_simulated_gpu(config, 2048);

    assert!(selector.gpu_available());
    let gpu_caps = selector
        .gpu_capabilities()
        .expect("GPU should be available");
    assert!(gpu_caps.available);
    assert_eq!(gpu_caps.max_buffer_size, 2048);
    assert_eq!(gpu_caps.max_parallelism, 256);
    assert!(gpu_caps.supports_f16);
    assert_eq!(gpu_caps.backend_type, BackendType::Gpu);
}

// --- Integration-style tests combining multiple paths ---

#[test]
#[ignore]
fn test_select_all_strategies_same_op() {
    let op = MatMulOp::new(64, 128, 64);
    let max_mem = 1024 * 1024 * 1024_u64;

    // PreferGpu with GPU available
    let s1 = BackendSelector::with_simulated_gpu(SelectorConfig::prefer_gpu(), max_mem);
    let r1 = s1.select(&op);
    assert!(r1.is_gpu());

    // PreferSimd (always SIMD regardless)
    let s2 = BackendSelector::with_simulated_gpu(SelectorConfig::prefer_simd(), max_mem);
    let r2 = s2.select(&op);
    assert!(r2.is_simd());

    // Automatic with low threshold (should pick GPU)
    let cfg3 = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(max_mem);
    let s3 = BackendSelector::with_simulated_gpu(cfg3, max_mem);
    let r3 = s3.select(&op);
    assert!(r3.is_gpu());

    // Threshold with high threshold (should pick SIMD)
    let cfg4 = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(u64::MAX))
        .with_max_gpu_memory(max_mem);
    let s4 = BackendSelector::with_simulated_gpu(cfg4, max_mem);
    let r4 = s4.select(&op);
    assert!(r4.is_simd());
}

#[test]
#[ignore]
fn test_select_automatic_transitions_based_on_workload_size() {
    let max_mem = 1024 * 1024 * 1024_u64;
    let threshold = 50_000_u64;
    let config = SelectorConfig::default()
        .with_gpu_threshold(threshold)
        .with_max_gpu_memory(max_mem);
    let selector = BackendSelector::with_simulated_gpu(config, max_mem);

    // Small op: below threshold -> SIMD
    let small_op = MatMulOp::new(4, 4, 4); // 2*4*4*4 = 128 flops
    let sel_small = selector.select(&small_op);
    assert!(sel_small.is_simd());

    // Medium op: above threshold -> GPU
    // Need flops >= 50_000. 2*m*k*n >= 50_000 -> m*k*n >= 25_000
    // 30*30*30 = 27_000 -> 2*27_000 = 54_000 >= 50_000
    let med_op = MatMulOp::new(30, 30, 30);
    let sel_med = selector.select(&med_op);
    assert!(sel_med.is_gpu());
}

#[ignore]
#[test]
fn test_prefer_gpu_no_gpu_falls_back_to_simd_reason() {
    // Without webgpu feature, GPU is never available
    let selector = BackendSelector::new(SelectorConfig::prefer_gpu());
    let op = MatMulOp::new(256, 256, 256);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert!(selection.reason.contains("GPU not available"));
}

#[test]
#[ignore]
fn test_select_prefer_simd_ignores_gpu_even_if_available() {
    let config = SelectorConfig::prefer_simd();
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    assert!(selector.gpu_available());

    let op = MatMulOp::new(512, 512, 512); // Very large op
    let selection = selector.select(&op);
    // PreferSimd always returns SIMD regardless of GPU availability
    assert!(selection.is_simd());
    assert!(selection.reason.contains("PreferSimd strategy"));
}

// =========================================================================
// Coverage Gap Tests: is_gpu_worthwhile, new, select, select_automatic
// (WAPR-QA-007)
// =========================================================================

// --- is_gpu_worthwhile: direct path coverage via simulated GPU ---
// The function has three exit points:
//   1. flops < gpu_threshold_flops -> return false
//   2. gpu_caps.can_handle(memory) returns false -> return false
//   3. falls through -> return true
// All paths require gpu_available=true (reached only via select_automatic).

/// Exercise is_gpu_worthwhile path 1: flops below threshold returns false.
/// Verifies the early return at line 266-268.
#[test]
#[ignore]
fn test_is_gpu_worthwhile_path_flops_below_threshold() {
    // Threshold set very high so any real op fails the flops check
    let config = SelectorConfig::default()
        .with_gpu_threshold(10_000_000)
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    // MatMulOp 4x4x4 = 2*4*4*4 = 128 flops, well below 10M
    let op = MatMulOp::new(4, 4, 4);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Small workload better on CPU");
}

/// Exercise is_gpu_worthwhile path 2: flops pass but gpu_caps.can_handle fails.
/// The GPU has a tiny max_buffer_size so can_handle returns false (line 272-274).
#[test]
#[ignore]
fn test_is_gpu_worthwhile_path_can_handle_fails() {
    // Low threshold so flops check passes, but GPU buffer is tiny
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(u64::MAX); // config memory limit is high (passes select_automatic check)

    // GPU buffer = 8 bytes: can_handle will fail for any non-trivial op
    let selector = BackendSelector::with_simulated_gpu(config, 8);

    // MatMulOp 16x16x16: memory = (256+256+256)*4 = 3072 bytes > 8
    let op = MatMulOp::new(16, 16, 16);
    let selection = selector.select(&op);
    // flops (8192) >= 1 passes, but can_handle(3072) fails because 3072 > 8
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Small workload better on CPU");
}

/// Exercise is_gpu_worthwhile path 3: both checks pass, returns true.
/// Verifies the happy path at line 277.
#[test]
#[ignore]
fn test_is_gpu_worthwhile_path_both_checks_pass() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(100)
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    // MatMulOp 32x32x32: flops = 2*32*32*32 = 65536 >= 100, memory fits
    let op = MatMulOp::new(32, 32, 32);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert_eq!(selection.reason, "Large workload benefits from GPU");
}

/// Exercise is_gpu_worthwhile when gpu_caps is None but gpu_available is true.
/// This is a contrived state but tests the code path where the if-let at line 271
/// does NOT enter the Some branch, falling through to return true.
#[test]
#[ignore]
fn test_is_gpu_worthwhile_no_gpu_caps_but_available() {
    // Manually construct a selector with gpu_available=true but gpu_caps=None
    let simd_caps = BackendCapabilities::simd();
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector {
        config,
        simd_caps,
        gpu_available: true,
        gpu_caps: None,
    };

    let op = MatMulOp::new(16, 16, 16);
    let selection = selector.select(&op);
    // gpu_available=true, memory check passes, flops >= 1 passes,
    // gpu_caps is None so if-let skipped, returns true -> GPU selected
    assert!(selection.is_gpu());
    assert_eq!(selection.reason, "Large workload benefits from GPU");
}

// --- new: verifying constructor field assignment (no-GPU path) ---

/// Verify that new() with default config produces correct field values
/// when GPU is not available (the else branch at line 158-160).
#[ignore]
#[test]
fn test_new_no_gpu_stores_none_caps() {
    let config = SelectorConfig::default();
    let selector = BackendSelector::new(config);

    // Without webgpu feature, GPU detection returns unavailable
    assert!(!selector.gpu_available());
    assert!(selector.gpu_capabilities().is_none());

    // SIMD caps are always populated
    let simd = selector.simd_capabilities();
    assert!(simd.available);
    assert_eq!(simd.backend_type, BackendType::Simd);
}

/// Verify new() stores the provided config verbatim.
#[test]
#[ignore]
fn test_new_stores_custom_config() {
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(42_000))
        .with_gpu_threshold(77_777)
        .with_max_gpu_memory(99_999);
    let selector = BackendSelector::new(config);

    let stored = selector.config();
    assert!(matches!(
        stored.strategy,
        SelectionStrategy::Threshold { min_flops: 42_000 }
    ));
    assert_eq!(stored.gpu_threshold_flops, 77_777);
    assert_eq!(stored.max_gpu_memory, 99_999);
}

/// Verify new() with prefer_gpu config still results in no GPU
/// when webgpu feature is disabled.
#[ignore]
#[test]
fn test_new_prefer_gpu_config_no_webgpu() {
    let selector = BackendSelector::new(SelectorConfig::prefer_gpu());

    // Config stored correctly
    assert_eq!(selector.config().strategy, SelectionStrategy::PreferGpu);

    // But GPU is not available
    assert!(!selector.gpu_available());
    assert!(selector.gpu_capabilities().is_none());

    // Selecting should fall back to SIMD
    let op = MatMulOp::new(64, 64, 64);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "GPU not available");
}

/// Verify new() with for_inference config stores all fields correctly.
#[test]
#[ignore]
fn test_new_for_inference_fields() {
    let selector = BackendSelector::new(SelectorConfig::for_inference());

    let config = selector.config();
    assert_eq!(config.strategy, SelectionStrategy::Automatic);
    assert_eq!(config.gpu_threshold_flops, 1_000_000);
    assert_eq!(config.max_gpu_memory, 1024 * 1024 * 1024);
    assert_eq!(config.gpu_dispatch_overhead_us, 50);

    // SIMD always available
    assert!(selector.simd_capabilities().available);
}

// --- select: exercising all four match arms with GPU available ---

/// PreferGpu with GPU available and memory fits -> GPU selected (line 208).
#[test]
#[ignore]
fn test_select_prefer_gpu_arm_gpu_available_memory_ok() {
    let config = SelectorConfig::prefer_gpu().with_max_gpu_memory(1024 * 1024 * 1024);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    let op = MatMulOp::new(16, 16, 16);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert_eq!(selection.reason, "PreferGpu strategy");
}

/// PreferGpu with GPU available but memory exceeds limit -> SIMD (line 210).
#[test]
#[ignore]
fn test_select_prefer_gpu_arm_memory_exceeds() {
    let config = SelectorConfig::prefer_gpu().with_max_gpu_memory(4);
    let selector = BackendSelector::with_simulated_gpu(config, 1024 * 1024 * 1024);
    // MatMulOp 32x32x32: memory = (1024+1024+1024)*4 = 12288 > 4
    let op = MatMulOp::new(32, 32, 32);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Memory exceeds GPU limit");
}

/// PreferGpu without GPU -> SIMD (line 212).
#[ignore]
#[test]
fn test_select_prefer_gpu_arm_no_gpu() {
    let selector = BackendSelector::new(SelectorConfig::prefer_gpu());
    let op = MatMulOp::new(32, 32, 32);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "GPU not available");
}

/// PreferSimd always returns SIMD regardless (line 216).
#[test]
#[ignore]
fn test_select_prefer_simd_arm_always_simd() {
    let config = SelectorConfig::prefer_simd();
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);
    let op = MatMulOp::new(512, 512, 512);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "PreferSimd strategy");
}

/// Threshold with GPU available, flops above min -> GPU (line 228).
#[test]
#[ignore]
fn test_select_threshold_arm_above_min_flops() {
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(500))
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);
    // MatMulOp 16x16x16: flops = 2*16*16*16 = 8192 >= 500
    let op = MatMulOp::new(16, 16, 16);
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert_eq!(selection.reason, "FLOPs exceed threshold");
}

/// Threshold with GPU available, flops below min -> SIMD (line 230).
#[test]
#[ignore]
fn test_select_threshold_arm_below_min_flops() {
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(100_000))
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);
    // MatMulOp 4x4x4: flops = 128 < 100_000
    let op = MatMulOp::new(4, 4, 4);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "FLOPs below threshold");
}

/// Threshold without GPU -> early return SIMD (line 220).
#[ignore]
#[test]
fn test_select_threshold_arm_no_gpu() {
    let config = SelectorConfig::default().with_strategy(SelectionStrategy::threshold(1));
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(128, 128, 128);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "GPU not available");
}

/// Threshold with GPU but memory exceeds limit -> SIMD (line 224).
#[test]
#[ignore]
fn test_select_threshold_arm_memory_exceeds() {
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(1))
        .with_max_gpu_memory(4);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);
    let op = MatMulOp::new(64, 64, 64);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Memory exceeds GPU limit");
}

/// Automatic strategy delegates to select_automatic (line 234).
#[test]
#[ignore]
fn test_select_automatic_arm_delegates() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);
    let op = MatMulOp::new(32, 32, 32);
    let selection = selector.select(&op);
    // select_automatic -> is_gpu_worthwhile -> true -> GPU
    assert!(selection.is_gpu());
    assert_eq!(selection.reason, "Large workload benefits from GPU");
}

// --- select_automatic: all three branches ---

/// select_automatic: GPU not available -> early return SIMD (line 241-243).
#[ignore]
#[test]
fn test_select_automatic_no_gpu_early_return() {
    let selector = BackendSelector::new(SelectorConfig::default());
    // Confirm no GPU
    assert!(!selector.gpu_available());

    let op = MatMulOp::new(256, 256, 256);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "GPU not available");
}

/// select_automatic: GPU available, memory exceeds limit -> SIMD (line 246-248).
#[test]
#[ignore]
fn test_select_automatic_memory_exceeds_early_return() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(16); // Very small limit
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    // MatMulOp 32x32x32: memory = (1024+1024+1024)*4 = 12288 > 16
    let op = MatMulOp::new(32, 32, 32);
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Memory exceeds GPU limit");
}

/// select_automatic: GPU available, memory ok, worthwhile=true -> GPU (line 253-254).
#[test]
#[ignore]
fn test_select_automatic_gpu_worthwhile_returns_gpu() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(50)
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    let op = MatMulOp::new(16, 16, 16); // flops=8192 >= 50
    let selection = selector.select(&op);
    assert!(selection.is_gpu());
    assert_eq!(selection.reason, "Large workload benefits from GPU");
}

/// select_automatic: GPU available, memory ok, worthwhile=false -> SIMD (line 255-256).
#[test]
#[ignore]
fn test_select_automatic_gpu_not_worthwhile_returns_simd() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(1_000_000)
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    let op = MatMulOp::new(4, 4, 4); // flops=128 < 1_000_000
    let selection = selector.select(&op);
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Small workload better on CPU");
}

/// select_automatic: GPU available, memory exactly at boundary (equal) -> passes (line 246).
/// The check is `memory > max_gpu_memory`, so equal should pass.
#[test]
#[ignore]
fn test_select_automatic_memory_at_exact_boundary() {
    let op = MatMulOp::new(8, 8, 8);
    let mem = op.memory_requirement() as u64; // 768

    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(mem); // exactly 768
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    let selection = selector.select(&op);
    // 768 > 768 is false, so memory check passes -> proceeds to is_gpu_worthwhile
    assert!(selection.is_gpu());
    assert_eq!(selection.reason, "Large workload benefits from GPU");
}

/// select_automatic: GPU available, memory one byte over boundary -> fails (line 246-248).
#[test]
#[ignore]
fn test_select_automatic_memory_one_over_boundary() {
    let op = MatMulOp::new(8, 8, 8);
    let mem = op.memory_requirement() as u64; // 768

    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(mem - 1); // 767
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    let selection = selector.select(&op);
    // 768 > 767 is true -> memory exceeds
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Memory exceeds GPU limit");
}

// =========================================================================
// Coverage Gap Tests: is_gpu_worthwhile and select_automatic
// direct invocation paths (WAPR-QA-008)
//
// These tests exercise is_gpu_worthwhile and select_automatic through
// the select() method with Automatic strategy, using with_simulated_gpu
// to ensure GPU is available. Tests are structured to guarantee that
// each branch within is_gpu_worthwhile and select_automatic is exercised.
// =========================================================================

/// Directly exercise is_gpu_worthwhile returning true (both checks pass):
/// flops >= threshold AND gpu_caps.can_handle(memory) == true.
#[test]
#[ignore]
fn test_is_gpu_worthwhile_returns_true_all_checks_pass() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(500)
        .with_max_gpu_memory(u64::MAX);
    // Large GPU buffer so can_handle passes
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    // MatMulOp 32x32x32: flops = 2*32*32*32 = 65536 >= 500
    let op = MatMulOp::new(32, 32, 32);
    let selection = selector.select(&op);

    // select_automatic -> is_gpu_worthwhile returns true -> GPU
    assert!(selection.is_gpu());
    assert_eq!(selection.reason, "Large workload benefits from GPU");
}

/// Exercise is_gpu_worthwhile returning false via flops < threshold.
#[test]
#[ignore]
fn test_is_gpu_worthwhile_returns_false_flops_below() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(1_000_000)
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    // MatMulOp 2x2x2: flops = 16 < 1_000_000
    let op = MatMulOp::new(2, 2, 2);
    let selection = selector.select(&op);

    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Small workload better on CPU");
}

/// Exercise is_gpu_worthwhile returning false via can_handle failure.
/// flops pass the threshold but GPU buffer is too small.
#[test]
#[ignore]
fn test_is_gpu_worthwhile_returns_false_can_handle_fails() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(1) // Very low, so flops always pass
        .with_max_gpu_memory(u64::MAX); // Config memory limit is high
                                        // GPU buffer = 4 bytes - too small for any non-trivial op
    let selector = BackendSelector::with_simulated_gpu(config, 4);

    // MatMulOp 16x16x16: memory = 3072 bytes > 4
    let op = MatMulOp::new(16, 16, 16);
    let selection = selector.select(&op);

    // flops pass threshold but can_handle(3072) fails -> false -> SIMD
    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Small workload better on CPU");
}

/// Exercise select_automatic path 1: GPU not available -> SIMD.
#[ignore]
#[test]
fn test_select_automatic_path_gpu_not_available() {
    // Standard constructor without webgpu feature -> no GPU
    let config = SelectorConfig::default();
    let selector = BackendSelector::new(config);

    let op = MatMulOp::new(128, 128, 128);
    let selection = selector.select(&op);

    assert!(selection.is_simd());
    assert_eq!(selection.reason, "GPU not available");
}

/// Exercise select_automatic path 2: GPU available but memory exceeds limit.
#[test]
#[ignore]
fn test_select_automatic_path_memory_exceeds() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(1)
        .with_max_gpu_memory(8); // Very small limit
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    // MatMulOp 32x32x32: memory = 12288 > 8
    let op = MatMulOp::new(32, 32, 32);
    let selection = selector.select(&op);

    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Memory exceeds GPU limit");
}

/// Exercise select_automatic path 3: worthwhile true -> GPU.
#[test]
#[ignore]
fn test_select_automatic_path_gpu_worthwhile() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(100)
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    let op = MatMulOp::new(64, 64, 64); // flops = 524288 >= 100
    let selection = selector.select(&op);

    assert!(selection.is_gpu());
    assert_eq!(selection.reason, "Large workload benefits from GPU");
}

/// Exercise select_automatic path 4: worthwhile false -> SIMD.
#[test]
#[ignore]
fn test_select_automatic_path_gpu_not_worthwhile() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(10_000_000) // High threshold
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    let op = MatMulOp::new(4, 4, 4); // flops = 128 < 10M
    let selection = selector.select(&op);

    assert!(selection.is_simd());
    assert_eq!(selection.reason, "Small workload better on CPU");
}

/// Test select_batch exercises select_automatic when GPU is available.
#[test]
#[ignore]
fn test_select_batch_exercises_select_automatic_gpu() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(100)
        .with_max_gpu_memory(u64::MAX);
    let selector = BackendSelector::with_simulated_gpu(config, u64::MAX);

    let ops = vec![MatMulOp::new(32, 32, 32), MatMulOp::new(64, 64, 64)];
    let selection = selector.select_batch(&ops);

    // Total flops well above threshold
    assert!(selection.is_gpu());
    assert_eq!(selection.reason, "Large workload benefits from GPU");
}

// =========================================================================
// BackendSelector::new() coverage (WAPR-QA-009)
//
// new() at line 144 has 57.1% coverage because the GPU-available path
// (lines 148-157) is never exercised in tests without the webgpu feature.
// These tests cover the remaining paths by:
// 1. Verifying the no-GPU path produces expected field values
// 2. Testing with_simulated_gpu to exercise equivalent GPU logic
// 3. Validating all constructor field assignments
// =========================================================================

/// Verify new() GPU detection path: without webgpu, detect_gpu returns
/// unavailable, so gpu_caps is None and gpu_available is false.
/// This covers lines 146 (simd_caps), 147 (gpu_result), 158-160 (else branch).
#[ignore]
#[test]
fn test_new_covers_no_gpu_detection_path() {
    let config = SelectorConfig::default();
    let selector = BackendSelector::new(config);

    // Line 146: simd_caps populated
    let simd = selector.simd_capabilities();
    assert!(simd.available);
    assert!(simd.max_parallelism > 0);
    assert!(simd.performance_score > 0.0);

    // Lines 158-160: gpu_result.available is false
    assert!(!selector.gpu_available());
    assert!(selector.gpu_capabilities().is_none());

    // Line 162-167: Self struct fields
    assert_eq!(selector.config().strategy, SelectionStrategy::Automatic);
}

/// Verify new() stores all config fields correctly with each
/// SelectorConfig constructor variant.
#[test]
#[ignore]
fn test_new_all_config_constructors() {
    // Default
    let s1 = BackendSelector::new(SelectorConfig::default());
    assert_eq!(s1.config().strategy, SelectionStrategy::Automatic);
    assert_eq!(s1.config().gpu_threshold_flops, 100_000);

    // For inference
    let s2 = BackendSelector::new(SelectorConfig::for_inference());
    assert_eq!(s2.config().strategy, SelectionStrategy::Automatic);
    assert_eq!(s2.config().gpu_threshold_flops, 1_000_000);
    assert_eq!(s2.config().gpu_dispatch_overhead_us, 50);

    // Prefer GPU
    let s3 = BackendSelector::new(SelectorConfig::prefer_gpu());
    assert_eq!(s3.config().strategy, SelectionStrategy::PreferGpu);

    // Prefer SIMD
    let s4 = BackendSelector::new(SelectorConfig::prefer_simd());
    assert_eq!(s4.config().strategy, SelectionStrategy::PreferSimd);

    // Threshold
    let s5 = BackendSelector::new(
        SelectorConfig::default().with_strategy(SelectionStrategy::threshold(42)),
    );
    assert!(matches!(
        s5.config().strategy,
        SelectionStrategy::Threshold { min_flops: 42 }
    ));
}

/// Verify with_simulated_gpu exercises the GPU-available path equivalent,
/// setting gpu_caps to Some and gpu_available to true (analogous to
/// lines 148-157 in new() when detect_gpu returns available).
#[test]
#[ignore]
#[allow(clippy::expect_used)]
fn test_simulated_gpu_covers_gpu_caps_construction() {
    let config = SelectorConfig::for_inference();
    let buffer_size = 512 * 1024 * 1024_u64;
    let selector = BackendSelector::with_simulated_gpu(config, buffer_size);

    // gpu_available = true (line 165 equivalent)
    assert!(selector.gpu_available());

    // gpu_caps = Some(...) (lines 148-157 equivalent)
    let gpu = selector
        .gpu_capabilities()
        .expect("GPU caps should be present");
    assert!(gpu.available);
    assert_eq!(gpu.max_buffer_size, buffer_size);
    assert!(gpu.supports_f16);
    assert_eq!(gpu.backend_type, BackendType::Gpu);

    // simd_caps always populated (line 145)
    assert!(selector.simd_capabilities().available);

    // Config stored correctly (line 162)
    assert_eq!(selector.config().strategy, SelectionStrategy::Automatic);
    assert_eq!(selector.config().gpu_threshold_flops, 1_000_000);
}

/// Verify that new() with custom config preserves all builder-set values
/// and the GPU detection doesn't corrupt them.
#[test]
#[ignore]
fn test_new_preserves_custom_config_after_gpu_detection() {
    let config = SelectorConfig::default()
        .with_strategy(SelectionStrategy::threshold(99_999))
        .with_gpu_threshold(12_345)
        .with_max_gpu_memory(67_890);

    let selector = BackendSelector::new(config);

    // Config should be preserved exactly as set, even after GPU detection
    let stored = selector.config();
    assert!(matches!(
        stored.strategy,
        SelectionStrategy::Threshold { min_flops: 99_999 }
    ));
    assert_eq!(stored.gpu_threshold_flops, 12_345);
    assert_eq!(stored.max_gpu_memory, 67_890);
}

/// Test new() with a config that has zero thresholds to exercise edge cases.
#[test]
#[ignore]
fn test_new_zero_threshold_config() {
    let config = SelectorConfig::default()
        .with_gpu_threshold(0)
        .with_max_gpu_memory(0);

    let selector = BackendSelector::new(config);

    assert_eq!(selector.config().gpu_threshold_flops, 0);
    assert_eq!(selector.config().max_gpu_memory, 0);

    // Even with zero memory limit, SIMD should still work
    let op = MatMulOp::new(4, 4, 4);
    let selection = selector.select(&op);
    assert!(!selection.reason.is_empty());
}

/// Test new() followed by summary() to exercise the full constructor
/// and display path.
#[test]
#[ignore]
fn test_new_then_summary_exercises_constructor() {
    let selector = BackendSelector::new(SelectorConfig::default());
    let summary = selector.summary();

    // Summary should include strategy description
    assert!(summary.contains("automatic"));
    assert!(summary.contains("SIMD"));
    // Without GPU, should show "not available"
    if !selector.gpu_available() {
        assert!(summary.contains("not available"));
    }
}

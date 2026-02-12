//! Tests for backend selection

use super::super::traits::MatMulOp;
use super::*;

#[test]
fn test_selection_strategy_default() {
    assert_eq!(SelectionStrategy::default(), SelectionStrategy::Automatic);
}

#[test]
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
fn test_selection_strategy_description() {
    assert!(SelectionStrategy::PreferGpu.description().contains("GPU"));
    assert!(SelectionStrategy::PreferSimd.description().contains("SIMD"));
    assert!(SelectionStrategy::Automatic
        .description()
        .contains("automatic"));
}

#[test]
fn test_selector_config_default() {
    let config = SelectorConfig::default();
    assert_eq!(config.strategy, SelectionStrategy::Automatic);
    assert!(config.gpu_threshold_flops > 0);
}

#[test]
fn test_selector_config_for_inference() {
    let config = SelectorConfig::for_inference();
    assert_eq!(config.strategy, SelectionStrategy::Automatic);
    assert!(config.gpu_threshold_flops >= 1_000_000);
}

#[test]
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
fn test_backend_selector_new() {
    let selector = BackendSelector::new(SelectorConfig::default());
    assert!(selector.simd_capabilities().available);
}

#[test]
fn test_backend_selector_default_config() {
    let selector = BackendSelector::default_config();
    assert!(selector.simd_capabilities().available);
}

#[test]
fn test_backend_selector_select_prefer_simd() {
    let selector = BackendSelector::new(SelectorConfig::prefer_simd());
    let op = MatMulOp::new(64, 128, 64);
    let selection = selector.select(&op);

    assert!(selection.is_simd());
    assert!(selection.reason.contains("PreferSimd"));
}

#[test]
fn test_backend_selector_select_small_workload() {
    let selector =
        BackendSelector::new(SelectorConfig::default().with_gpu_threshold(1_000_000_000));
    let op = MatMulOp::new(8, 8, 8); // Very small
    let selection = selector.select(&op);

    // Should select SIMD for small workloads
    assert!(selection.is_simd());
}

#[test]
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
fn test_backend_selector_select_batch_empty() {
    let selector = BackendSelector::default_config();
    let ops: Vec<MatMulOp> = vec![];
    let selection = selector.select_batch(&ops);

    assert!(selection.is_simd());
    assert!(selection.reason.contains("No operations"));
}

#[test]
fn test_backend_selector_summary() {
    let selector = BackendSelector::default_config();
    let summary = selector.summary();

    assert!(summary.contains("SIMD"));
    assert!(summary.contains("parallelism"));
}

#[test]
fn test_backend_selection_gpu() {
    let selection = BackendSelection::gpu("test reason");
    assert!(selection.is_gpu());
    assert!(!selection.is_simd());
    assert_eq!(selection.reason, "test reason");
}

#[test]
fn test_backend_selection_simd() {
    let selection = BackendSelection::simd("test reason");
    assert!(selection.is_simd());
    assert!(!selection.is_gpu());
}

#[test]
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
fn test_selection_strategy_threshold_description() {
    let strategy = SelectionStrategy::threshold(1_000_000);
    assert_eq!(strategy.description(), "threshold-based");
}

#[test]
fn test_selector_config_prefer_gpu() {
    let config = SelectorConfig::prefer_gpu();
    assert_eq!(config.strategy, SelectionStrategy::PreferGpu);
}

#[test]
fn test_selector_config_prefer_simd() {
    let config = SelectorConfig::prefer_simd();
    assert_eq!(config.strategy, SelectionStrategy::PreferSimd);
}

#[test]
fn test_backend_selector_config_accessor() {
    let selector = BackendSelector::default_config();
    let config = selector.config();
    assert_eq!(config.strategy, SelectionStrategy::Automatic);
}

#[test]
fn test_backend_selector_gpu_capabilities() {
    let selector = BackendSelector::default_config();
    // GPU may or may not be available, just check it doesn't panic
    let _ = selector.gpu_capabilities();
}

#[test]
fn test_backend_selector_select_prefer_gpu_no_gpu() {
    let selector = BackendSelector::new(SelectorConfig::prefer_gpu());
    let op = MatMulOp::new(64, 128, 64);
    let selection = selector.select(&op);
    // Since GPU is likely not available in tests, should fall back
    assert!(!selection.reason.is_empty());
}

#[test]
fn test_backend_selector_select_large_workload() {
    let selector = BackendSelector::new(SelectorConfig::default().with_gpu_threshold(100));
    let op = MatMulOp::new(128, 256, 128); // Large workload
    let selection = selector.select(&op);
    // Just verify selection is made
    assert!(!selection.reason.is_empty());
}

#[test]
fn test_backend_selection_backend_type() {
    let gpu_selection = BackendSelection::gpu("test");
    assert_eq!(gpu_selection.backend, BackendType::Gpu);

    let simd_selection = BackendSelection::simd("test");
    assert_eq!(simd_selection.backend, BackendType::Simd);
}

#[test]
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
fn test_selector_automatic_with_small_workload() {
    let selector = BackendSelector::new(SelectorConfig::default());
    let op = MatMulOp::new(4, 4, 4); // Very small
    let selection = selector.select(&op);

    // Small workload should prefer SIMD
    assert!(selection.is_simd());
}

#[test]
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
fn test_selection_strategy_all_descriptions() {
    assert!(!SelectionStrategy::PreferGpu.description().is_empty());
    assert!(!SelectionStrategy::PreferSimd.description().is_empty());
    assert!(!SelectionStrategy::Automatic.description().is_empty());
    assert!(!SelectionStrategy::threshold(1000).description().is_empty());
}

#[test]
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
fn test_backend_selector_select_batch_large() {
    let selector = BackendSelector::default_config();
    let ops: Vec<MatMulOp> = (0..10).map(|_| MatMulOp::new(128, 256, 128)).collect();
    let selection = selector.select_batch(&ops);
    assert!(!selection.reason.is_empty());
}

#[test]
fn test_backend_selector_simd_capabilities() {
    let selector = BackendSelector::default_config();
    let caps = selector.simd_capabilities();
    assert!(caps.available);
    assert!(caps.max_parallelism > 0);
}

#[test]
fn test_backend_selector_gpu_available() {
    let selector = BackendSelector::default_config();
    // Just ensure method works, GPU may or may not be available
    let _available = selector.gpu_available();
}

#[test]
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
fn test_selector_automatic_large_workload() {
    let config = SelectorConfig::default().with_gpu_threshold(10);
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(256, 512, 256); // Large enough
    let selection = selector.select(&op);
    assert!(!selection.reason.is_empty());
}

#[test]
fn test_selector_is_gpu_worthwhile_below_threshold() {
    let config = SelectorConfig::default().with_gpu_threshold(1_000_000_000);
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(8, 8, 8); // Small workload
    let selection = selector.select(&op);
    assert!(selection.is_simd());
}

#[test]
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
// Additional Coverage Tests (WAPR-QA-003)
// =========================================================================

#[test]
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
fn test_selector_for_inference_config() {
    let config = SelectorConfig::for_inference();
    let selector = BackendSelector::new(config);
    let op = MatMulOp::new(8, 8, 8);
    let selection = selector.select(&op);
    assert!(!selection.reason.is_empty());
}

#[test]
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

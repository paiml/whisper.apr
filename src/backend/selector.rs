//! Automatic backend selection (WAPR-141)
//!
//! Provides intelligent selection between SIMD and GPU backends
//! based on workload characteristics and available hardware.

use super::traits::{BackendCapabilities, BackendType, ComputeOp};
use crate::gpu::{detect_gpu, DetectionOptions};

/// Backend selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Always use GPU if available
    PreferGpu,
    /// Always use SIMD
    PreferSimd,
    /// Automatic selection based on workload
    Automatic,
    /// Use GPU for large workloads, SIMD for small
    Threshold {
        /// Minimum FLOPs to use GPU
        min_flops: u64,
    },
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        Self::Automatic
    }
}

impl SelectionStrategy {
    /// Create threshold strategy
    #[must_use]
    pub fn threshold(min_flops: u64) -> Self {
        Self::Threshold { min_flops }
    }

    /// Get description of the strategy
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::PreferGpu => "prefer GPU",
            Self::PreferSimd => "prefer SIMD",
            Self::Automatic => "automatic",
            Self::Threshold { .. } => "threshold-based",
        }
    }
}

/// Backend selector configuration
#[derive(Debug, Clone)]
pub struct SelectorConfig {
    /// Selection strategy
    pub strategy: SelectionStrategy,
    /// GPU detection options
    pub gpu_options: DetectionOptions,
    /// Minimum workload size to consider GPU (in FLOPs)
    pub gpu_threshold_flops: u64,
    /// Maximum memory to use on GPU (bytes)
    pub max_gpu_memory: u64,
    /// Overhead factor for GPU dispatch (microseconds)
    pub gpu_dispatch_overhead_us: u32,
}

impl Default for SelectorConfig {
    fn default() -> Self {
        Self {
            strategy: SelectionStrategy::default(),
            gpu_options: DetectionOptions::default(),
            gpu_threshold_flops: 100_000, // 100K FLOPs minimum for GPU
            max_gpu_memory: 256 * 1024 * 1024, // 256 MB
            gpu_dispatch_overhead_us: 100, // 100 microseconds
        }
    }
}

impl SelectorConfig {
    /// Create config for inference workloads
    #[must_use]
    pub fn for_inference() -> Self {
        Self {
            strategy: SelectionStrategy::Automatic,
            gpu_options: DetectionOptions::for_inference(),
            gpu_threshold_flops: 1_000_000,     // 1M FLOPs
            max_gpu_memory: 1024 * 1024 * 1024, // 1 GB
            gpu_dispatch_overhead_us: 50,
        }
    }

    /// Create config that prefers GPU
    #[must_use]
    pub fn prefer_gpu() -> Self {
        Self {
            strategy: SelectionStrategy::PreferGpu,
            ..Default::default()
        }
    }

    /// Create config that prefers SIMD
    #[must_use]
    pub fn prefer_simd() -> Self {
        Self {
            strategy: SelectionStrategy::PreferSimd,
            ..Default::default()
        }
    }

    /// Set strategy
    #[must_use]
    pub fn with_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set GPU threshold
    #[must_use]
    pub fn with_gpu_threshold(mut self, flops: u64) -> Self {
        self.gpu_threshold_flops = flops;
        self
    }

    /// Set max GPU memory
    #[must_use]
    pub fn with_max_gpu_memory(mut self, bytes: u64) -> Self {
        self.max_gpu_memory = bytes;
        self
    }
}

/// Backend selector
#[derive(Debug)]
pub struct BackendSelector {
    /// Configuration
    config: SelectorConfig,
    /// SIMD capabilities
    simd_caps: BackendCapabilities,
    /// GPU capabilities (if detected)
    gpu_caps: Option<BackendCapabilities>,
    /// Whether GPU was detected
    gpu_available: bool,
}

impl BackendSelector {
    /// Create a new backend selector
    #[must_use]
    pub fn new(config: SelectorConfig) -> Self {
        let simd_caps = BackendCapabilities::simd();
        let gpu_result = detect_gpu(&config.gpu_options);

        let gpu_caps = if gpu_result.available {
            Some(BackendCapabilities::gpu(
                true,
                gpu_result.capabilities.limits.max_buffer_size,
                gpu_result
                    .capabilities
                    .limits
                    .max_compute_invocations_per_workgroup,
                gpu_result.capabilities.supports_f16,
            ))
        } else {
            None
        };

        Self {
            config,
            simd_caps,
            gpu_available: gpu_result.available,
            gpu_caps,
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(SelectorConfig::default())
    }

    /// Get selector configuration
    #[must_use]
    pub fn config(&self) -> &SelectorConfig {
        &self.config
    }

    /// Check if GPU is available
    #[must_use]
    pub fn gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Get SIMD capabilities
    #[must_use]
    pub fn simd_capabilities(&self) -> &BackendCapabilities {
        &self.simd_caps
    }

    /// Get GPU capabilities (if available)
    #[must_use]
    pub fn gpu_capabilities(&self) -> Option<&BackendCapabilities> {
        self.gpu_caps.as_ref()
    }

    /// Select backend for a given operation
    pub fn select<O: ComputeOp>(&self, op: &O) -> BackendSelection {
        let flops = op.estimated_flops();
        let memory = op.memory_requirement() as u64;

        match self.config.strategy {
            SelectionStrategy::PreferGpu => {
                if self.gpu_available && memory <= self.config.max_gpu_memory {
                    BackendSelection::gpu("PreferGpu strategy")
                } else if self.gpu_available {
                    BackendSelection::simd("Memory exceeds GPU limit")
                } else {
                    BackendSelection::simd("GPU not available")
                }
            }

            SelectionStrategy::PreferSimd => BackendSelection::simd("PreferSimd strategy"),

            SelectionStrategy::Threshold { min_flops } => {
                if !self.gpu_available {
                    return BackendSelection::simd("GPU not available");
                }

                if memory > self.config.max_gpu_memory {
                    return BackendSelection::simd("Memory exceeds GPU limit");
                }

                if flops >= min_flops {
                    BackendSelection::gpu("FLOPs exceed threshold")
                } else {
                    BackendSelection::simd("FLOPs below threshold")
                }
            }

            SelectionStrategy::Automatic => self.select_automatic(flops, memory),
        }
    }

    /// Automatic backend selection
    fn select_automatic(&self, flops: u64, memory: u64) -> BackendSelection {
        // If GPU not available, always use SIMD
        if !self.gpu_available {
            return BackendSelection::simd("GPU not available");
        }

        // Check memory constraints
        if memory > self.config.max_gpu_memory {
            return BackendSelection::simd("Memory exceeds GPU limit");
        }

        // Check if workload is large enough to justify GPU dispatch overhead
        let gpu_worthwhile = self.is_gpu_worthwhile(flops, memory);

        if gpu_worthwhile {
            BackendSelection::gpu("Large workload benefits from GPU")
        } else {
            BackendSelection::simd("Small workload better on CPU")
        }
    }

    /// Check if GPU is worthwhile for given workload
    fn is_gpu_worthwhile(&self, flops: u64, memory: u64) -> bool {
        // Rough heuristic: GPU dispatch overhead is ~100us
        // At 10 TFLOPS, GPU processes 10M FLOPs in 1us
        // So for 100us overhead, need at least 1M FLOPs to break even

        if flops < self.config.gpu_threshold_flops {
            return false;
        }

        // Check if GPU has capacity
        if let Some(gpu_caps) = &self.gpu_caps {
            if !gpu_caps.can_handle(memory) {
                return false;
            }
        }

        true
    }

    /// Select backend for multiple operations (batched)
    pub fn select_batch<O: ComputeOp>(&self, ops: &[O]) -> BackendSelection {
        if ops.is_empty() {
            return BackendSelection::simd("No operations");
        }

        let total_flops: u64 = ops.iter().map(|o| o.estimated_flops()).sum();
        let max_memory: u64 = ops
            .iter()
            .map(|o| o.memory_requirement() as u64)
            .max()
            .unwrap_or(0);

        self.select_automatic(total_flops, max_memory)
    }

    /// Get a summary of available backends
    #[must_use]
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let mut s = format!(
            "Backend Selector ({})\n",
            self.config.strategy.description()
        );
        let _ = writeln!(
            s,
            "  SIMD: parallelism={}, score={:.1}",
            self.simd_caps.max_parallelism, self.simd_caps.performance_score
        );

        if let Some(gpu) = &self.gpu_caps {
            let _ = writeln!(
                s,
                "  GPU: parallelism={}, score={:.1}, f16={}",
                gpu.max_parallelism, gpu.performance_score, gpu.supports_f16
            );
        } else {
            s.push_str("  GPU: not available\n");
        }

        s
    }
}

/// Backend selection result
#[derive(Debug, Clone)]
pub struct BackendSelection {
    /// Selected backend
    pub backend: BackendType,
    /// Reason for selection
    pub reason: String,
}

impl BackendSelection {
    /// Create GPU selection
    #[must_use]
    pub fn gpu(reason: impl Into<String>) -> Self {
        Self {
            backend: BackendType::Gpu,
            reason: reason.into(),
        }
    }

    /// Create SIMD selection
    #[must_use]
    pub fn simd(reason: impl Into<String>) -> Self {
        Self {
            backend: BackendType::Simd,
            reason: reason.into(),
        }
    }

    /// Check if GPU was selected
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        self.backend.is_gpu()
    }

    /// Check if SIMD was selected
    #[must_use]
    pub fn is_simd(&self) -> bool {
        self.backend.is_cpu()
    }
}

impl std::fmt::Display for BackendSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.backend, self.reason)
    }
}

#[cfg(test)]
mod tests {
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
}

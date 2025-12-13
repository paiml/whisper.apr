//! GPU detection and capability queries (WAPR-123)
//!
//! Provides high-level API for detecting GPU capabilities and selecting backends.

use super::capabilities::{GpuBackend, GpuCapabilities, GpuLimits};
use super::error::{GpuError, GpuResult};

/// GPU detection result
#[derive(Debug, Clone)]
pub struct GpuDetectionResult {
    /// Whether a GPU was found
    pub available: bool,
    /// Detected capabilities
    pub capabilities: GpuCapabilities,
    /// Recommended backend
    pub recommended_backend: GpuBackend,
    /// Detection method used
    pub detection_method: DetectionMethod,
}

impl GpuDetectionResult {
    /// Create a result indicating no GPU available
    #[must_use]
    pub fn unavailable() -> Self {
        Self {
            available: false,
            capabilities: GpuCapabilities::default(),
            recommended_backend: GpuBackend::None,
            detection_method: DetectionMethod::NoGpu,
        }
    }

    /// Check if GPU is suitable for inference
    #[must_use]
    pub fn suitable_for_inference(&self) -> bool {
        self.available && self.capabilities.suitable_for_inference()
    }

    /// Get a human-readable summary
    #[must_use]
    pub fn summary(&self) -> String {
        if self.available {
            format!(
                "GPU Available: {} via {} ({})",
                self.capabilities.name,
                self.recommended_backend,
                self.detection_method
            )
        } else {
            "No GPU available".to_string()
        }
    }
}

/// How the GPU was detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMethod {
    /// wgpu native adapter request
    WgpuNative,
    /// WebGPU browser API
    WebGpuBrowser,
    /// Simulated for testing
    Simulated,
    /// No GPU detected
    NoGpu,
}

impl std::fmt::Display for DetectionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WgpuNative => write!(f, "wgpu native"),
            Self::WebGpuBrowser => write!(f, "WebGPU browser"),
            Self::Simulated => write!(f, "simulated"),
            Self::NoGpu => write!(f, "none"),
        }
    }
}

/// GPU detection options
#[derive(Debug, Clone)]
pub struct DetectionOptions {
    /// Prefer high-performance GPU over power-efficient
    pub prefer_high_performance: bool,
    /// Require compute shader support
    pub require_compute: bool,
    /// Minimum VRAM in bytes (0 = no minimum)
    pub min_vram: u64,
    /// Preferred backend (None = auto-select)
    pub preferred_backend: Option<GpuBackend>,
    /// Timeout for detection in milliseconds
    pub timeout_ms: u32,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            prefer_high_performance: true,
            require_compute: true,
            min_vram: 0,
            preferred_backend: None,
            timeout_ms: 5000,
        }
    }
}

impl DetectionOptions {
    /// Options for inference workloads
    #[must_use]
    pub fn for_inference() -> Self {
        Self {
            prefer_high_performance: true,
            require_compute: true,
            min_vram: 256 * 1024 * 1024, // 256 MB minimum
            preferred_backend: None,
            timeout_ms: 5000,
        }
    }

    /// Options for development/testing
    #[must_use]
    pub fn for_development() -> Self {
        Self {
            prefer_high_performance: false,
            require_compute: false,
            min_vram: 0,
            preferred_backend: None,
            timeout_ms: 10000,
        }
    }

    /// Set preferred backend
    #[must_use]
    pub fn with_backend(mut self, backend: GpuBackend) -> Self {
        self.preferred_backend = Some(backend);
        self
    }

    /// Set minimum VRAM
    #[must_use]
    pub fn with_min_vram(mut self, vram: u64) -> Self {
        self.min_vram = vram;
        self
    }

    /// Disable compute requirement
    #[must_use]
    pub fn without_compute_requirement(mut self) -> Self {
        self.require_compute = false;
        self
    }
}

/// Detect GPU capabilities
///
/// This is the main entry point for GPU detection. It queries available
/// GPU backends and returns information about the best available GPU.
pub fn detect_gpu(options: &DetectionOptions) -> GpuDetectionResult {
    // Without the webgpu feature, we can only return unavailable
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = options; // Silence unused warning
        GpuDetectionResult::unavailable()
    }

    #[cfg(feature = "webgpu")]
    {
        // TODO: Implement actual wgpu detection
        detect_gpu_wgpu(options)
    }
}

#[cfg(feature = "webgpu")]
fn detect_gpu_wgpu(_options: &DetectionOptions) -> GpuDetectionResult {
    // Placeholder for actual wgpu implementation
    GpuDetectionResult::unavailable()
}

/// Create a simulated GPU result for testing
#[must_use]
pub fn detect_gpu_simulated(config: SimulatedGpuConfig) -> GpuDetectionResult {
    let capabilities = GpuCapabilities {
        name: config.name,
        vendor: config.vendor,
        backend: config.backend,
        limits: config.limits,
        supports_f16: config.supports_f16,
        supports_timestamp_query: config.supports_timestamp_query,
        vram_bytes: config.vram_bytes,
    };

    GpuDetectionResult {
        available: config.backend != GpuBackend::None,
        capabilities,
        recommended_backend: config.backend,
        detection_method: DetectionMethod::Simulated,
    }
}

/// Configuration for simulated GPU
#[derive(Debug, Clone)]
pub struct SimulatedGpuConfig {
    /// GPU name
    pub name: String,
    /// GPU vendor
    pub vendor: String,
    /// Backend type
    pub backend: GpuBackend,
    /// Device limits
    pub limits: GpuLimits,
    /// F16 support
    pub supports_f16: bool,
    /// Timestamp query support
    pub supports_timestamp_query: bool,
    /// VRAM in bytes
    pub vram_bytes: u64,
}

impl Default for SimulatedGpuConfig {
    fn default() -> Self {
        Self {
            name: "Simulated GPU".to_string(),
            vendor: "Test".to_string(),
            backend: GpuBackend::Vulkan,
            limits: GpuLimits::default(),
            supports_f16: true,
            supports_timestamp_query: true,
            vram_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
        }
    }
}

impl SimulatedGpuConfig {
    /// Create high-end desktop GPU config
    #[must_use]
    pub fn high_end_desktop() -> Self {
        Self {
            name: "Simulated RTX 4090".to_string(),
            vendor: "NVIDIA".to_string(),
            backend: GpuBackend::Vulkan,
            limits: GpuLimits::desktop_high_end(),
            supports_f16: true,
            supports_timestamp_query: true,
            vram_bytes: 24 * 1024 * 1024 * 1024, // 24 GB
        }
    }

    /// Create Apple Silicon GPU config
    #[must_use]
    pub fn apple_silicon() -> Self {
        Self {
            name: "Simulated Apple M2".to_string(),
            vendor: "Apple".to_string(),
            backend: GpuBackend::Metal,
            limits: GpuLimits::default(),
            supports_f16: true,
            supports_timestamp_query: false,
            vram_bytes: 16 * 1024 * 1024 * 1024, // 16 GB unified
        }
    }

    /// Create mobile GPU config
    #[must_use]
    pub fn mobile() -> Self {
        Self {
            name: "Simulated Adreno 730".to_string(),
            vendor: "Qualcomm".to_string(),
            backend: GpuBackend::Vulkan,
            limits: GpuLimits::mobile(),
            supports_f16: true,
            supports_timestamp_query: false,
            vram_bytes: 512 * 1024 * 1024, // 512 MB
        }
    }

    /// Create browser WebGPU config
    #[must_use]
    pub fn browser_webgpu() -> Self {
        Self {
            name: "Browser GPU".to_string(),
            vendor: "Unknown".to_string(),
            backend: GpuBackend::BrowserWebGpu,
            limits: GpuLimits::default(),
            supports_f16: false, // Browser might not expose this
            supports_timestamp_query: false,
            vram_bytes: 0, // Unknown in browser
        }
    }

    /// Set GPU name
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set VRAM
    #[must_use]
    pub fn with_vram(mut self, vram_bytes: u64) -> Self {
        self.vram_bytes = vram_bytes;
        self
    }

    /// Set backend
    #[must_use]
    pub fn with_backend(mut self, backend: GpuBackend) -> Self {
        self.backend = backend;
        self
    }
}

/// Query specific GPU features
#[derive(Debug, Clone, Copy)]
pub struct GpuFeatureQuery {
    /// Requires compute shader support
    pub compute: bool,
    /// Requires F16 support
    pub f16: bool,
    /// Requires timestamp queries
    pub timestamp_query: bool,
    /// Minimum buffer size needed
    pub min_buffer_size: u64,
    /// Minimum VRAM needed
    pub min_vram: u64,
}

impl Default for GpuFeatureQuery {
    fn default() -> Self {
        Self {
            compute: false,
            f16: false,
            timestamp_query: false,
            min_buffer_size: 0,
            min_vram: 0,
        }
    }
}

impl GpuFeatureQuery {
    /// Query for inference workloads
    #[must_use]
    pub fn for_inference() -> Self {
        Self {
            compute: true,
            f16: false, // Preferred but not required
            timestamp_query: false,
            min_buffer_size: 256 * 1024 * 1024,
            min_vram: 256 * 1024 * 1024,
        }
    }

    /// Query for profiling
    #[must_use]
    pub fn for_profiling() -> Self {
        Self {
            compute: true,
            f16: false,
            timestamp_query: true,
            min_buffer_size: 64 * 1024 * 1024,
            min_vram: 0,
        }
    }

    /// Add compute requirement
    #[must_use]
    pub fn with_compute(mut self) -> Self {
        self.compute = true;
        self
    }

    /// Add F16 requirement
    #[must_use]
    pub fn with_f16(mut self) -> Self {
        self.f16 = true;
        self
    }

    /// Add timestamp query requirement
    #[must_use]
    pub fn with_timestamp_query(mut self) -> Self {
        self.timestamp_query = true;
        self
    }

    /// Check if capabilities satisfy this query
    #[must_use]
    pub fn satisfied_by(&self, caps: &GpuCapabilities) -> bool {
        if self.compute && !caps.supports_compute() {
            return false;
        }
        if self.f16 && !caps.supports_f16 {
            return false;
        }
        if self.timestamp_query && !caps.supports_timestamp_query {
            return false;
        }
        if self.min_buffer_size > caps.limits.max_buffer_size {
            return false;
        }
        if self.min_vram > 0 && caps.vram_bytes > 0 && self.min_vram > caps.vram_bytes {
            return false;
        }
        true
    }

    /// Get list of unsatisfied requirements
    #[must_use]
    pub fn unsatisfied_requirements(&self, caps: &GpuCapabilities) -> Vec<String> {
        let mut reqs = Vec::new();

        if self.compute && !caps.supports_compute() {
            reqs.push("compute shaders".to_string());
        }
        if self.f16 && !caps.supports_f16 {
            reqs.push("F16 support".to_string());
        }
        if self.timestamp_query && !caps.supports_timestamp_query {
            reqs.push("timestamp queries".to_string());
        }
        if self.min_buffer_size > caps.limits.max_buffer_size {
            reqs.push(format!(
                "buffer size (need {} MB, have {} MB)",
                self.min_buffer_size / 1024 / 1024,
                caps.limits.max_buffer_size / 1024 / 1024
            ));
        }
        if self.min_vram > 0 && caps.vram_bytes > 0 && self.min_vram > caps.vram_bytes {
            reqs.push(format!(
                "VRAM (need {} MB, have {} MB)",
                self.min_vram / 1024 / 1024,
                caps.vram_bytes / 1024 / 1024
            ));
        }

        reqs
    }
}

/// Recommend the best backend for the current platform
#[must_use]
pub fn recommend_backend() -> GpuBackend {
    #[cfg(target_os = "macos")]
    {
        GpuBackend::Metal
    }

    #[cfg(target_os = "windows")]
    {
        GpuBackend::Dx12
    }

    #[cfg(target_os = "linux")]
    {
        GpuBackend::Vulkan
    }

    #[cfg(target_arch = "wasm32")]
    {
        GpuBackend::BrowserWebGpu
    }

    #[cfg(not(any(
        target_os = "macos",
        target_os = "windows",
        target_os = "linux",
        target_arch = "wasm32"
    )))]
    {
        GpuBackend::Vulkan // Default fallback
    }
}

/// Check if GPU should be used for given workload size
#[must_use]
pub fn should_use_gpu(caps: &GpuCapabilities, workload_elements: usize) -> GpuRecommendation {
    const GPU_THRESHOLD: usize = 10_000; // Below this, CPU is likely faster
    const GPU_STRONGLY_RECOMMENDED: usize = 100_000;

    if !caps.is_available() {
        return GpuRecommendation::CpuOnly {
            reason: "No GPU available".to_string(),
        };
    }

    if !caps.supports_compute() {
        return GpuRecommendation::CpuOnly {
            reason: "GPU doesn't support compute shaders".to_string(),
        };
    }

    if workload_elements < GPU_THRESHOLD {
        return GpuRecommendation::CpuPreferred {
            reason: format!(
                "Workload size ({} elements) is small; CPU may be faster due to GPU overhead",
                workload_elements
            ),
        };
    }

    if workload_elements >= GPU_STRONGLY_RECOMMENDED {
        return GpuRecommendation::GpuStronglyRecommended {
            speedup_estimate: estimate_speedup(caps, workload_elements),
        };
    }

    GpuRecommendation::GpuRecommended {
        speedup_estimate: estimate_speedup(caps, workload_elements),
    }
}

/// GPU usage recommendation
#[derive(Debug, Clone)]
pub enum GpuRecommendation {
    /// CPU only (no GPU available or suitable)
    CpuOnly {
        /// Reason for CPU-only recommendation
        reason: String,
    },
    /// CPU preferred for this workload
    CpuPreferred {
        /// Reason CPU is preferred
        reason: String,
    },
    /// GPU recommended
    GpuRecommended {
        /// Estimated speedup factor
        speedup_estimate: f32,
    },
    /// GPU strongly recommended
    GpuStronglyRecommended {
        /// Estimated speedup factor
        speedup_estimate: f32,
    },
}

impl GpuRecommendation {
    /// Check if GPU is recommended
    #[must_use]
    pub fn use_gpu(&self) -> bool {
        matches!(
            self,
            Self::GpuRecommended { .. } | Self::GpuStronglyRecommended { .. }
        )
    }

    /// Get speedup estimate if GPU is recommended
    #[must_use]
    pub fn speedup(&self) -> Option<f32> {
        match self {
            Self::GpuRecommended { speedup_estimate } => Some(*speedup_estimate),
            Self::GpuStronglyRecommended { speedup_estimate } => Some(*speedup_estimate),
            _ => None,
        }
    }
}

/// Estimate speedup from using GPU
fn estimate_speedup(caps: &GpuCapabilities, elements: usize) -> f32 {
    // Very rough heuristic estimates
    let base_speedup = if caps.backend.is_high_performance() {
        10.0
    } else {
        5.0
    };

    // Scale with workload size (diminishing returns)
    let scale = (elements as f32 / 10_000.0).ln().max(1.0);

    (base_speedup * scale).min(100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_result_unavailable() {
        let result = GpuDetectionResult::unavailable();
        assert!(!result.available);
        assert!(!result.suitable_for_inference());
        assert_eq!(result.recommended_backend, GpuBackend::None);
    }

    #[test]
    fn test_detection_result_summary() {
        let result = GpuDetectionResult::unavailable();
        assert!(result.summary().contains("No GPU"));

        let simulated = detect_gpu_simulated(SimulatedGpuConfig::default());
        assert!(simulated.summary().contains("Simulated GPU"));
    }

    #[test]
    fn test_detection_method_display() {
        assert_eq!(DetectionMethod::WgpuNative.to_string(), "wgpu native");
        assert_eq!(DetectionMethod::WebGpuBrowser.to_string(), "WebGPU browser");
        assert_eq!(DetectionMethod::Simulated.to_string(), "simulated");
        assert_eq!(DetectionMethod::NoGpu.to_string(), "none");
    }

    #[test]
    fn test_detection_options_default() {
        let opts = DetectionOptions::default();
        assert!(opts.prefer_high_performance);
        assert!(opts.require_compute);
        assert_eq!(opts.min_vram, 0);
    }

    #[test]
    fn test_detection_options_for_inference() {
        let opts = DetectionOptions::for_inference();
        assert!(opts.prefer_high_performance);
        assert!(opts.require_compute);
        assert!(opts.min_vram > 0);
    }

    #[test]
    fn test_detection_options_builders() {
        let opts = DetectionOptions::default()
            .with_backend(GpuBackend::Metal)
            .with_min_vram(1024 * 1024 * 1024)
            .without_compute_requirement();

        assert_eq!(opts.preferred_backend, Some(GpuBackend::Metal));
        assert_eq!(opts.min_vram, 1024 * 1024 * 1024);
        assert!(!opts.require_compute);
    }

    #[test]
    fn test_detect_gpu_without_feature() {
        let result = detect_gpu(&DetectionOptions::default());
        // Without webgpu feature, should be unavailable
        #[cfg(not(feature = "webgpu"))]
        assert!(!result.available);
    }

    #[test]
    fn test_simulated_gpu_config_default() {
        let config = SimulatedGpuConfig::default();
        assert_eq!(config.name, "Simulated GPU");
        assert_eq!(config.backend, GpuBackend::Vulkan);
        assert!(config.supports_f16);
    }

    #[test]
    fn test_simulated_gpu_config_high_end() {
        let config = SimulatedGpuConfig::high_end_desktop();
        assert!(config.name.contains("RTX"));
        assert_eq!(config.vram_bytes, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_simulated_gpu_config_apple_silicon() {
        let config = SimulatedGpuConfig::apple_silicon();
        assert!(config.name.contains("Apple"));
        assert_eq!(config.backend, GpuBackend::Metal);
    }

    #[test]
    fn test_simulated_gpu_config_mobile() {
        let config = SimulatedGpuConfig::mobile();
        assert!(config.limits.max_buffer_size < GpuLimits::default().max_buffer_size);
    }

    #[test]
    fn test_simulated_gpu_config_browser() {
        let config = SimulatedGpuConfig::browser_webgpu();
        assert_eq!(config.backend, GpuBackend::BrowserWebGpu);
        assert!(!config.supports_f16);
    }

    #[test]
    fn test_simulated_gpu_config_builders() {
        let config = SimulatedGpuConfig::default()
            .with_name("Custom GPU")
            .with_vram(8 * 1024 * 1024 * 1024)
            .with_backend(GpuBackend::Metal);

        assert_eq!(config.name, "Custom GPU");
        assert_eq!(config.vram_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(config.backend, GpuBackend::Metal);
    }

    #[test]
    fn test_detect_gpu_simulated() {
        let result = detect_gpu_simulated(SimulatedGpuConfig::default());
        assert!(result.available);
        assert_eq!(result.detection_method, DetectionMethod::Simulated);
        assert!(result.suitable_for_inference());
    }

    #[test]
    fn test_gpu_feature_query_default() {
        let query = GpuFeatureQuery::default();
        assert!(!query.compute);
        assert!(!query.f16);
        assert_eq!(query.min_buffer_size, 0);
    }

    #[test]
    fn test_gpu_feature_query_for_inference() {
        let query = GpuFeatureQuery::for_inference();
        assert!(query.compute);
        assert!(query.min_buffer_size > 0);
    }

    #[test]
    fn test_gpu_feature_query_builders() {
        let query = GpuFeatureQuery::default()
            .with_compute()
            .with_f16()
            .with_timestamp_query();

        assert!(query.compute);
        assert!(query.f16);
        assert!(query.timestamp_query);
    }

    #[test]
    fn test_gpu_feature_query_satisfied_by() {
        let result = detect_gpu_simulated(SimulatedGpuConfig::default());
        let query = GpuFeatureQuery::for_inference();

        assert!(query.satisfied_by(&result.capabilities));
    }

    #[test]
    fn test_gpu_feature_query_unsatisfied() {
        let result = detect_gpu_simulated(SimulatedGpuConfig::default());
        let query = GpuFeatureQuery::default().with_timestamp_query();

        // Default config has timestamp_query = true
        assert!(query.satisfied_by(&result.capabilities));

        // Test with config that doesn't have timestamp query
        let no_ts = SimulatedGpuConfig::mobile();
        let result2 = detect_gpu_simulated(no_ts);
        let reqs = query.unsatisfied_requirements(&result2.capabilities);
        assert!(reqs.iter().any(|r| r.contains("timestamp")));
    }

    #[test]
    fn test_recommend_backend() {
        let backend = recommend_backend();
        // Should return a valid backend for the current platform
        #[cfg(target_os = "macos")]
        assert_eq!(backend, GpuBackend::Metal);
        #[cfg(target_os = "windows")]
        assert_eq!(backend, GpuBackend::Dx12);
        #[cfg(target_os = "linux")]
        assert_eq!(backend, GpuBackend::Vulkan);
    }

    #[test]
    fn test_should_use_gpu_unavailable() {
        let caps = GpuCapabilities::default(); // No GPU
        let rec = should_use_gpu(&caps, 100_000);
        assert!(!rec.use_gpu());
    }

    #[test]
    fn test_should_use_gpu_small_workload() {
        let result = detect_gpu_simulated(SimulatedGpuConfig::default());
        let rec = should_use_gpu(&result.capabilities, 1_000);
        assert!(!rec.use_gpu()); // Small workload, CPU preferred
    }

    #[test]
    fn test_should_use_gpu_large_workload() {
        let result = detect_gpu_simulated(SimulatedGpuConfig::default());
        let rec = should_use_gpu(&result.capabilities, 500_000);
        assert!(rec.use_gpu());
        assert!(rec.speedup().is_some());
    }

    #[test]
    fn test_gpu_recommendation_use_gpu() {
        assert!(!GpuRecommendation::CpuOnly { reason: "test".to_string() }.use_gpu());
        assert!(!GpuRecommendation::CpuPreferred { reason: "test".to_string() }.use_gpu());
        assert!(GpuRecommendation::GpuRecommended { speedup_estimate: 5.0 }.use_gpu());
        assert!(GpuRecommendation::GpuStronglyRecommended { speedup_estimate: 10.0 }.use_gpu());
    }

    #[test]
    fn test_gpu_recommendation_speedup() {
        assert!(GpuRecommendation::CpuOnly { reason: "test".to_string() }.speedup().is_none());
        assert_eq!(
            GpuRecommendation::GpuRecommended { speedup_estimate: 5.0 }.speedup(),
            Some(5.0)
        );
    }
}

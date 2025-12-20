//! WebGPU WASM bindings (WAPR-143)
//!
//! Provides JavaScript-friendly API for GPU detection and backend selection.
//!
//! # Usage
//!
//! ```javascript
//! import { GpuDetectionWasm, BackendSelectorWasm, SelectorConfigWasm } from 'whisper-apr';
//!
//! // Detect GPU capabilities
//! const detection = new GpuDetectionWasm();
//! console.log(`GPU available: ${detection.available}`);
//! console.log(`Backend: ${detection.backendName}`);
//!
//! // Configure backend selection
//! const config = SelectorConfigWasm.forInference();
//! const selector = new BackendSelectorWasm(config);
//!
//! // Select backend for a workload
//! const selection = selector.selectForMatMul(1024, 768, 1024);
//! console.log(`Selected: ${selection.backendName} - ${selection.reason}`);
//! ```

use wasm_bindgen::prelude::*;

use crate::backend::{
    BackendSelection, BackendSelector, BackendType, MatMulOp, SelectionStrategy, SelectorConfig,
};
use crate::gpu::{detect_gpu, DetectionOptions, GpuBackend, GpuCapabilities, GpuLimits};

// =============================================================================
// GPU Backend Type
// =============================================================================

/// WASM-friendly GPU backend type
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendWasm {
    /// Vulkan backend
    Vulkan = 0,
    /// Metal backend (macOS/iOS)
    Metal = 1,
    /// DirectX 12 backend (Windows)
    Dx12 = 2,
    /// WebGPU backend (browser)
    BrowserWebGpu = 3,
    /// OpenGL backend (legacy)
    Gl = 4,
    /// No GPU backend available
    None = 5,
}

impl From<GpuBackend> for GpuBackendWasm {
    fn from(backend: GpuBackend) -> Self {
        match backend {
            GpuBackend::Vulkan => Self::Vulkan,
            GpuBackend::Metal => Self::Metal,
            GpuBackend::Dx12 => Self::Dx12,
            GpuBackend::Dx11 => Self::Gl, // Map Dx11 to Gl slot for WASM
            GpuBackend::OpenGl => Self::Gl,
            GpuBackend::BrowserWebGpu => Self::BrowserWebGpu,
            GpuBackend::None => Self::None,
        }
    }
}

impl From<GpuBackendWasm> for GpuBackend {
    fn from(wasm: GpuBackendWasm) -> Self {
        match wasm {
            GpuBackendWasm::Vulkan => Self::Vulkan,
            GpuBackendWasm::Metal => Self::Metal,
            GpuBackendWasm::Dx12 => Self::Dx12,
            GpuBackendWasm::BrowserWebGpu => Self::BrowserWebGpu,
            GpuBackendWasm::Gl => Self::OpenGl,
            GpuBackendWasm::None => Self::None,
        }
    }
}

// =============================================================================
// GPU Limits
// =============================================================================

/// WASM-friendly GPU limits
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct GpuLimitsWasm {
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u32,
    max_uniform_buffer_binding_size: u32,
    max_compute_workgroup_size_x: u32,
    max_compute_workgroup_size_y: u32,
    max_compute_workgroup_size_z: u32,
    max_compute_invocations_per_workgroup: u32,
    max_compute_workgroups_per_dimension: u32,
}

#[wasm_bindgen]
impl GpuLimitsWasm {
    /// Get maximum buffer size in bytes
    #[wasm_bindgen(getter, js_name = maxBufferSize)]
    pub fn max_buffer_size(&self) -> u64 {
        self.max_buffer_size
    }

    /// Get maximum storage buffer binding size
    #[wasm_bindgen(getter, js_name = maxStorageBufferBindingSize)]
    pub fn max_storage_buffer_binding_size(&self) -> u32 {
        self.max_storage_buffer_binding_size
    }

    /// Get maximum uniform buffer binding size
    #[wasm_bindgen(getter, js_name = maxUniformBufferBindingSize)]
    pub fn max_uniform_buffer_binding_size(&self) -> u32 {
        self.max_uniform_buffer_binding_size
    }

    /// Get maximum compute workgroup size X
    #[wasm_bindgen(getter, js_name = maxComputeWorkgroupSizeX)]
    pub fn max_compute_workgroup_size_x(&self) -> u32 {
        self.max_compute_workgroup_size_x
    }

    /// Get maximum compute workgroup size Y
    #[wasm_bindgen(getter, js_name = maxComputeWorkgroupSizeY)]
    pub fn max_compute_workgroup_size_y(&self) -> u32 {
        self.max_compute_workgroup_size_y
    }

    /// Get maximum compute workgroup size Z
    #[wasm_bindgen(getter, js_name = maxComputeWorkgroupSizeZ)]
    pub fn max_compute_workgroup_size_z(&self) -> u32 {
        self.max_compute_workgroup_size_z
    }

    /// Get maximum compute invocations per workgroup
    #[wasm_bindgen(getter, js_name = maxComputeInvocationsPerWorkgroup)]
    pub fn max_compute_invocations_per_workgroup(&self) -> u32 {
        self.max_compute_invocations_per_workgroup
    }

    /// Get maximum compute workgroups per dimension
    #[wasm_bindgen(getter, js_name = maxComputeWorkgroupsPerDimension)]
    pub fn max_compute_workgroups_per_dimension(&self) -> u32 {
        self.max_compute_workgroups_per_dimension
    }

    /// Get maximum buffer size in MB
    #[wasm_bindgen(js_name = maxBufferSizeMb)]
    pub fn max_buffer_size_mb(&self) -> f32 {
        self.max_buffer_size as f32 / (1024.0 * 1024.0)
    }
}

impl From<GpuLimits> for GpuLimitsWasm {
    fn from(limits: GpuLimits) -> Self {
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }
}

// =============================================================================
// GPU Capabilities
// =============================================================================

/// WASM-friendly GPU capabilities
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct GpuCapabilitiesWasm {
    backend: GpuBackendWasm,
    device_name: String,
    vendor_name: String,
    driver_info: String,
    supports_f16: bool,
    supports_timestamp_query: bool,
    limits: GpuLimitsWasm,
}

#[wasm_bindgen]
impl GpuCapabilitiesWasm {
    /// Get the GPU backend type
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> GpuBackendWasm {
        self.backend
    }

    /// Get the device name
    #[wasm_bindgen(getter, js_name = deviceName)]
    pub fn device_name(&self) -> String {
        self.device_name.clone()
    }

    /// Get the vendor name
    #[wasm_bindgen(getter, js_name = vendorName)]
    pub fn vendor_name(&self) -> String {
        self.vendor_name.clone()
    }

    /// Get the driver info
    #[wasm_bindgen(getter, js_name = driverInfo)]
    pub fn driver_info(&self) -> String {
        self.driver_info.clone()
    }

    /// Check if F16 (half precision) is supported
    #[wasm_bindgen(getter, js_name = supportsF16)]
    pub fn supports_f16(&self) -> bool {
        self.supports_f16
    }

    /// Check if timestamp queries are supported
    #[wasm_bindgen(getter, js_name = supportsTimestampQuery)]
    pub fn supports_timestamp_query(&self) -> bool {
        self.supports_timestamp_query
    }

    /// Get GPU limits
    #[wasm_bindgen(getter)]
    pub fn limits(&self) -> GpuLimitsWasm {
        self.limits.clone()
    }

    /// Get backend name as string
    #[wasm_bindgen(js_name = backendName)]
    pub fn backend_name(&self) -> String {
        match self.backend {
            GpuBackendWasm::Vulkan => "Vulkan".to_string(),
            GpuBackendWasm::Metal => "Metal".to_string(),
            GpuBackendWasm::Dx12 => "DirectX 12".to_string(),
            GpuBackendWasm::BrowserWebGpu => "WebGPU".to_string(),
            GpuBackendWasm::Gl => "OpenGL".to_string(),
            GpuBackendWasm::None => "None".to_string(),
        }
    }

    /// Get a summary of capabilities
    #[wasm_bindgen]
    pub fn summary(&self) -> String {
        format!(
            "GPU: {} ({}) | Backend: {} | F16: {} | Max Buffer: {:.0}MB",
            self.device_name,
            self.vendor_name,
            self.backend_name(),
            if self.supports_f16 { "Yes" } else { "No" },
            self.limits.max_buffer_size_mb()
        )
    }
}

impl From<GpuCapabilities> for GpuCapabilitiesWasm {
    fn from(caps: GpuCapabilities) -> Self {
        Self {
            backend: caps.backend.into(),
            device_name: caps.name,
            vendor_name: caps.vendor,
            driver_info: String::new(), // Not available in native GpuCapabilities
            supports_f16: caps.supports_f16,
            supports_timestamp_query: caps.supports_timestamp_query,
            limits: caps.limits.into(),
        }
    }
}

// =============================================================================
// GPU Detection
// =============================================================================

/// WASM-friendly GPU detection result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct GpuDetectionWasm {
    available: bool,
    capabilities: Option<GpuCapabilitiesWasm>,
    error_message: Option<String>,
}

#[wasm_bindgen]
impl GpuDetectionWasm {
    /// Detect GPU with default options
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::detect_with_options(DetectionOptionsWasm::default())
    }

    /// Detect GPU with specific options
    #[wasm_bindgen(js_name = detectWithOptions)]
    pub fn detect_with_options(options: DetectionOptionsWasm) -> Self {
        let native_options: DetectionOptions = options.into();
        let result = detect_gpu(&native_options);

        Self {
            available: result.available,
            capabilities: if result.available {
                Some(result.capabilities.into())
            } else {
                None
            },
            error_message: None, // GpuDetectionResult doesn't have an error field
        }
    }

    /// Detect GPU for inference workloads
    #[wasm_bindgen(js_name = forInference)]
    pub fn for_inference() -> Self {
        Self::detect_with_options(DetectionOptionsWasm::for_inference())
    }

    /// Check if GPU is available
    #[wasm_bindgen(getter)]
    pub fn available(&self) -> bool {
        self.available
    }

    /// Get GPU capabilities (if available)
    #[wasm_bindgen(getter)]
    pub fn capabilities(&self) -> Option<GpuCapabilitiesWasm> {
        self.capabilities.clone()
    }

    /// Get error message (if detection failed)
    #[wasm_bindgen(getter, js_name = errorMessage)]
    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    /// Get the backend name
    #[wasm_bindgen(js_name = backendName)]
    pub fn backend_name(&self) -> String {
        self.capabilities
            .as_ref()
            .map_or_else(|| "None".to_string(), |c| c.backend_name())
    }

    /// Get the device name
    #[wasm_bindgen(js_name = deviceName)]
    pub fn device_name(&self) -> String {
        self.capabilities
            .as_ref()
            .map_or_else(|| "No GPU".to_string(), |c| c.device_name())
    }

    /// Check if F16 is supported
    #[wasm_bindgen(js_name = supportsF16)]
    pub fn supports_f16(&self) -> bool {
        self.capabilities.as_ref().is_some_and(|c| c.supports_f16)
    }

    /// Get a summary of the detection result
    #[wasm_bindgen]
    pub fn summary(&self) -> String {
        if let Some(caps) = &self.capabilities {
            caps.summary()
        } else if let Some(err) = &self.error_message {
            format!("GPU not available: {err}")
        } else {
            "GPU not available".to_string()
        }
    }
}

impl Default for GpuDetectionWasm {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Detection Options
// =============================================================================

/// WASM-friendly detection options
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DetectionOptionsWasm {
    prefer_high_performance: bool,
    require_f16: bool,
    timeout_ms: u32,
}

#[wasm_bindgen]
impl DetectionOptionsWasm {
    /// Create default detection options
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            prefer_high_performance: true,
            require_f16: false,
            timeout_ms: 5000,
        }
    }

    /// Create options for inference workloads
    #[wasm_bindgen(js_name = forInference)]
    pub fn for_inference() -> Self {
        Self {
            prefer_high_performance: true,
            require_f16: false,
            timeout_ms: 10000,
        }
    }

    /// Set high-performance preference
    #[wasm_bindgen(setter, js_name = preferHighPerformance)]
    pub fn set_prefer_high_performance(&mut self, value: bool) {
        self.prefer_high_performance = value;
    }

    /// Get high-performance preference
    #[wasm_bindgen(getter, js_name = preferHighPerformance)]
    pub fn prefer_high_performance(&self) -> bool {
        self.prefer_high_performance
    }

    /// Set F16 requirement
    #[wasm_bindgen(setter, js_name = requireF16)]
    pub fn set_require_f16(&mut self, value: bool) {
        self.require_f16 = value;
    }

    /// Get F16 requirement
    #[wasm_bindgen(getter, js_name = requireF16)]
    pub fn require_f16(&self) -> bool {
        self.require_f16
    }

    /// Set timeout in milliseconds
    #[wasm_bindgen(setter, js_name = timeoutMs)]
    pub fn set_timeout_ms(&mut self, value: u32) {
        self.timeout_ms = value;
    }

    /// Get timeout in milliseconds
    #[wasm_bindgen(getter, js_name = timeoutMs)]
    pub fn timeout_ms(&self) -> u32 {
        self.timeout_ms
    }
}

impl Default for DetectionOptionsWasm {
    fn default() -> Self {
        Self::new()
    }
}

impl From<DetectionOptionsWasm> for DetectionOptions {
    fn from(wasm: DetectionOptionsWasm) -> Self {
        if wasm.prefer_high_performance {
            Self::for_inference()
        } else {
            Self::default()
        }
    }
}

// =============================================================================
// Backend Selection
// =============================================================================

/// WASM-friendly backend type
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendTypeWasm {
    /// CPU with SIMD acceleration
    Simd = 0,
    /// GPU compute
    Gpu = 1,
    /// Plain CPU (no SIMD)
    Cpu = 2,
    /// Automatic selection
    Auto = 3,
}

impl From<BackendType> for BackendTypeWasm {
    fn from(backend: BackendType) -> Self {
        match backend {
            BackendType::Simd => Self::Simd,
            BackendType::Gpu => Self::Gpu,
            BackendType::Cpu => Self::Cpu,
            BackendType::Auto => Self::Auto,
        }
    }
}

impl From<BackendTypeWasm> for BackendType {
    fn from(wasm: BackendTypeWasm) -> Self {
        match wasm {
            BackendTypeWasm::Simd => Self::Simd,
            BackendTypeWasm::Gpu => Self::Gpu,
            BackendTypeWasm::Cpu => Self::Cpu,
            BackendTypeWasm::Auto => Self::Auto,
        }
    }
}

/// WASM-friendly selection strategy
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategyWasm {
    /// Always prefer GPU if available
    PreferGpu = 0,
    /// Always prefer SIMD
    PreferSimd = 1,
    /// Automatic selection based on workload
    Automatic = 2,
    /// Threshold-based selection
    Threshold = 3,
}

impl From<SelectionStrategy> for SelectionStrategyWasm {
    fn from(strategy: SelectionStrategy) -> Self {
        match strategy {
            SelectionStrategy::PreferGpu => Self::PreferGpu,
            SelectionStrategy::PreferSimd => Self::PreferSimd,
            SelectionStrategy::Automatic => Self::Automatic,
            SelectionStrategy::Threshold { .. } => Self::Threshold,
        }
    }
}

/// WASM-friendly backend selection result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct BackendSelectionWasm {
    backend: BackendTypeWasm,
    reason: String,
}

#[wasm_bindgen]
impl BackendSelectionWasm {
    /// Get the selected backend
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> BackendTypeWasm {
        self.backend
    }

    /// Get the reason for selection
    #[wasm_bindgen(getter)]
    pub fn reason(&self) -> String {
        self.reason.clone()
    }

    /// Get the backend name as string
    #[wasm_bindgen(js_name = backendName)]
    pub fn backend_name(&self) -> String {
        match self.backend {
            BackendTypeWasm::Simd => "SIMD".to_string(),
            BackendTypeWasm::Gpu => "GPU".to_string(),
            BackendTypeWasm::Cpu => "CPU".to_string(),
            BackendTypeWasm::Auto => "Auto".to_string(),
        }
    }

    /// Check if GPU was selected
    #[wasm_bindgen(js_name = isGpu)]
    pub fn is_gpu(&self) -> bool {
        matches!(self.backend, BackendTypeWasm::Gpu)
    }

    /// Check if SIMD was selected
    #[wasm_bindgen(js_name = isSimd)]
    pub fn is_simd(&self) -> bool {
        matches!(self.backend, BackendTypeWasm::Simd)
    }
}

impl From<BackendSelection> for BackendSelectionWasm {
    fn from(selection: BackendSelection) -> Self {
        Self {
            backend: selection.backend.into(),
            reason: selection.reason,
        }
    }
}

// =============================================================================
// Selector Configuration
// =============================================================================

/// WASM-friendly selector configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SelectorConfigWasm {
    strategy: SelectionStrategyWasm,
    gpu_threshold_flops: u64,
    max_gpu_memory: u64,
    gpu_dispatch_overhead_us: u32,
}

#[wasm_bindgen]
impl SelectorConfigWasm {
    /// Create default configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            strategy: SelectionStrategyWasm::Automatic,
            gpu_threshold_flops: 100_000,
            max_gpu_memory: 256 * 1024 * 1024,
            gpu_dispatch_overhead_us: 100,
        }
    }

    /// Create configuration for inference workloads
    #[wasm_bindgen(js_name = forInference)]
    pub fn for_inference() -> Self {
        Self {
            strategy: SelectionStrategyWasm::Automatic,
            gpu_threshold_flops: 1_000_000,
            max_gpu_memory: 1024 * 1024 * 1024,
            gpu_dispatch_overhead_us: 50,
        }
    }

    /// Create configuration that prefers GPU
    #[wasm_bindgen(js_name = preferGpu)]
    pub fn prefer_gpu() -> Self {
        Self {
            strategy: SelectionStrategyWasm::PreferGpu,
            ..Self::new()
        }
    }

    /// Create configuration that prefers SIMD
    #[wasm_bindgen(js_name = preferSimd)]
    pub fn prefer_simd() -> Self {
        Self {
            strategy: SelectionStrategyWasm::PreferSimd,
            ..Self::new()
        }
    }

    /// Get the selection strategy
    #[wasm_bindgen(getter)]
    pub fn strategy(&self) -> SelectionStrategyWasm {
        self.strategy
    }

    /// Set the selection strategy
    #[wasm_bindgen(setter)]
    pub fn set_strategy(&mut self, strategy: SelectionStrategyWasm) {
        self.strategy = strategy;
    }

    /// Get the GPU threshold in FLOPs
    #[wasm_bindgen(getter, js_name = gpuThresholdFlops)]
    pub fn gpu_threshold_flops(&self) -> u64 {
        self.gpu_threshold_flops
    }

    /// Set the GPU threshold in FLOPs
    #[wasm_bindgen(setter, js_name = gpuThresholdFlops)]
    pub fn set_gpu_threshold_flops(&mut self, value: u64) {
        self.gpu_threshold_flops = value;
    }

    /// Get the maximum GPU memory in bytes
    #[wasm_bindgen(getter, js_name = maxGpuMemory)]
    pub fn max_gpu_memory(&self) -> u64 {
        self.max_gpu_memory
    }

    /// Set the maximum GPU memory in bytes
    #[wasm_bindgen(setter, js_name = maxGpuMemory)]
    pub fn set_max_gpu_memory(&mut self, value: u64) {
        self.max_gpu_memory = value;
    }

    /// Get the maximum GPU memory in MB
    #[wasm_bindgen(js_name = maxGpuMemoryMb)]
    pub fn max_gpu_memory_mb(&self) -> f32 {
        self.max_gpu_memory as f32 / (1024.0 * 1024.0)
    }

    /// Set the maximum GPU memory in MB
    #[wasm_bindgen(js_name = setMaxGpuMemoryMb)]
    pub fn set_max_gpu_memory_mb(&mut self, mb: f32) {
        self.max_gpu_memory = (mb * 1024.0 * 1024.0) as u64;
    }

    /// Get the GPU dispatch overhead in microseconds
    #[wasm_bindgen(getter, js_name = gpuDispatchOverheadUs)]
    pub fn gpu_dispatch_overhead_us(&self) -> u32 {
        self.gpu_dispatch_overhead_us
    }

    /// Set the GPU dispatch overhead in microseconds
    #[wasm_bindgen(setter, js_name = gpuDispatchOverheadUs)]
    pub fn set_gpu_dispatch_overhead_us(&mut self, value: u32) {
        self.gpu_dispatch_overhead_us = value;
    }
}

impl Default for SelectorConfigWasm {
    fn default() -> Self {
        Self::new()
    }
}

impl From<SelectorConfigWasm> for SelectorConfig {
    fn from(wasm: SelectorConfigWasm) -> Self {
        let strategy = match wasm.strategy {
            SelectionStrategyWasm::PreferGpu => SelectionStrategy::PreferGpu,
            SelectionStrategyWasm::PreferSimd => SelectionStrategy::PreferSimd,
            SelectionStrategyWasm::Automatic => SelectionStrategy::Automatic,
            SelectionStrategyWasm::Threshold => {
                SelectionStrategy::threshold(wasm.gpu_threshold_flops)
            }
        };

        Self::default()
            .with_strategy(strategy)
            .with_gpu_threshold(wasm.gpu_threshold_flops)
            .with_max_gpu_memory(wasm.max_gpu_memory)
    }
}

// =============================================================================
// Backend Selector
// =============================================================================

/// WASM-friendly backend selector
#[wasm_bindgen]
pub struct BackendSelectorWasm {
    inner: BackendSelector,
}

#[wasm_bindgen]
impl BackendSelectorWasm {
    /// Create a new backend selector with the given configuration
    #[wasm_bindgen(constructor)]
    pub fn new(config: SelectorConfigWasm) -> Self {
        Self {
            inner: BackendSelector::new(config.into()),
        }
    }

    /// Create a selector with default configuration
    #[wasm_bindgen(js_name = default)]
    pub fn default_selector() -> Self {
        Self {
            inner: BackendSelector::default_config(),
        }
    }

    /// Create a selector for inference workloads
    #[wasm_bindgen(js_name = forInference)]
    pub fn for_inference() -> Self {
        Self {
            inner: BackendSelector::new(SelectorConfig::for_inference()),
        }
    }

    /// Check if GPU is available
    #[wasm_bindgen(js_name = gpuAvailable)]
    pub fn gpu_available(&self) -> bool {
        self.inner.gpu_available()
    }

    /// Select backend for matrix multiplication
    #[wasm_bindgen(js_name = selectForMatMul)]
    pub fn select_for_mat_mul(&self, m: u32, k: u32, n: u32) -> BackendSelectionWasm {
        let op = MatMulOp::new(m as usize, k as usize, n as usize);
        self.inner.select(&op).into()
    }

    /// Select backend for a given FLOPs count
    #[wasm_bindgen(js_name = selectForFlops)]
    pub fn select_for_flops(&self, flops: u64, _memory_bytes: u64) -> BackendSelectionWasm {
        // Create a synthetic MatMul that matches the FLOP count
        // For matmul: FLOPs = 2 * M * K * N, so for simplicity use M=K=N=cbrt(FLOPs/2)
        let dim = ((flops / 2) as f64).cbrt() as usize;
        let dim = dim.max(1);
        let op = MatMulOp::new(dim, dim, dim);

        // The selector will use the FLOP estimate from the op
        self.inner.select(&op).into()
    }

    /// Get a summary of available backends
    #[wasm_bindgen]
    pub fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Get the SIMD performance score
    #[wasm_bindgen(js_name = simdPerformanceScore)]
    pub fn simd_performance_score(&self) -> f32 {
        self.inner.simd_capabilities().performance_score
    }

    /// Get the GPU performance score (if available)
    #[wasm_bindgen(js_name = gpuPerformanceScore)]
    pub fn gpu_performance_score(&self) -> Option<f32> {
        self.inner.gpu_capabilities().map(|c| c.performance_score)
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get the recommended backend for a given model size
#[wasm_bindgen(js_name = recommendedBackendForModel)]
pub fn recommended_backend_for_model(model_type: &str) -> String {
    let params = match model_type.to_lowercase().as_str() {
        "tiny" | "tiny.en" => 39_000_000,
        "base" | "base.en" => 74_000_000,
        "small" | "small.en" => 244_000_000,
        "medium" | "medium.en" => 769_000_000,
        "large" | "large-v2" | "large-v3" => 1_550_000_000,
        _ => 39_000_000,
    };

    // For models with >100M parameters, prefer GPU if available
    if params > 100_000_000 {
        "GPU (if available) or SIMD".to_string()
    } else {
        "SIMD (GPU optional for larger batch sizes)".to_string()
    }
}

/// Estimate the memory required for a matrix multiplication
#[wasm_bindgen(js_name = estimateMatMulMemory)]
pub fn estimate_mat_mul_memory(m: u32, k: u32, n: u32, element_size: u32) -> u64 {
    let m = m as u64;
    let k = k as u64;
    let n = n as u64;
    let elem = element_size as u64;

    // A: M x K, B: K x N, C: M x N
    (m * k + k * n + m * n) * elem
}

/// Estimate the FLOPs for a matrix multiplication
#[wasm_bindgen(js_name = estimateMatMulFlops)]
pub fn estimate_mat_mul_flops(m: u32, k: u32, n: u32) -> u64 {
    // 2 * M * K * N (multiply-add counted as 2 FLOPs)
    2 * (m as u64) * (k as u64) * (n as u64)
}

/// Check if a workload is GPU-worthwhile
#[wasm_bindgen(js_name = isGpuWorthwhile)]
pub fn is_gpu_worthwhile(flops: u64, threshold: u64) -> bool {
    flops >= threshold
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // GpuBackendWasm Tests
    // =========================================================================

    #[test]
    fn test_gpu_backend_wasm_values() {
        assert_eq!(GpuBackendWasm::Vulkan as u32, 0);
        assert_eq!(GpuBackendWasm::Metal as u32, 1);
        assert_eq!(GpuBackendWasm::Dx12 as u32, 2);
        assert_eq!(GpuBackendWasm::BrowserWebGpu as u32, 3);
        assert_eq!(GpuBackendWasm::Gl as u32, 4);
        assert_eq!(GpuBackendWasm::None as u32, 5);
    }

    #[test]
    fn test_gpu_backend_wasm_conversion() {
        assert_eq!(
            GpuBackendWasm::from(GpuBackend::Vulkan),
            GpuBackendWasm::Vulkan
        );
        assert_eq!(
            GpuBackendWasm::from(GpuBackend::Metal),
            GpuBackendWasm::Metal
        );
        assert_eq!(
            GpuBackendWasm::from(GpuBackend::BrowserWebGpu),
            GpuBackendWasm::BrowserWebGpu
        );
    }

    #[test]
    fn test_gpu_backend_wasm_to_native() {
        assert_eq!(GpuBackend::from(GpuBackendWasm::Vulkan), GpuBackend::Vulkan);
        assert_eq!(GpuBackend::from(GpuBackendWasm::Metal), GpuBackend::Metal);
    }

    // =========================================================================
    // GpuLimitsWasm Tests
    // =========================================================================

    #[test]
    fn test_gpu_limits_wasm_from_native() {
        let native = GpuLimits {
            max_buffer_size: 1024 * 1024 * 1024,
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            max_uniform_buffer_binding_size: 64 * 1024,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 65535,
            max_bind_groups: 4,
        };

        let wasm: GpuLimitsWasm = native.into();

        assert_eq!(wasm.max_buffer_size(), 1024 * 1024 * 1024);
        assert_eq!(wasm.max_storage_buffer_binding_size(), 128 * 1024 * 1024);
        assert_eq!(wasm.max_uniform_buffer_binding_size(), 64 * 1024);
        assert_eq!(wasm.max_compute_workgroup_size_x(), 256);
        assert!((wasm.max_buffer_size_mb() - 1024.0).abs() < 0.01);
    }

    // =========================================================================
    // DetectionOptionsWasm Tests
    // =========================================================================

    #[test]
    fn test_detection_options_wasm_new() {
        let opts = DetectionOptionsWasm::new();
        assert!(opts.prefer_high_performance());
        assert!(!opts.require_f16());
        assert_eq!(opts.timeout_ms(), 5000);
    }

    #[test]
    fn test_detection_options_wasm_for_inference() {
        let opts = DetectionOptionsWasm::for_inference();
        assert!(opts.prefer_high_performance());
        assert_eq!(opts.timeout_ms(), 10000);
    }

    #[test]
    fn test_detection_options_wasm_setters() {
        let mut opts = DetectionOptionsWasm::new();
        opts.set_prefer_high_performance(false);
        opts.set_require_f16(true);
        opts.set_timeout_ms(15000);

        assert!(!opts.prefer_high_performance());
        assert!(opts.require_f16());
        assert_eq!(opts.timeout_ms(), 15000);
    }

    // =========================================================================
    // GpuDetectionWasm Tests
    // =========================================================================

    #[test]
    fn test_gpu_detection_wasm_new() {
        let detection = GpuDetectionWasm::new();
        // GPU may or may not be available depending on the environment
        assert!(!detection.summary().is_empty());
    }

    #[test]
    fn test_gpu_detection_wasm_for_inference() {
        let detection = GpuDetectionWasm::for_inference();
        assert!(!detection.summary().is_empty());
    }

    #[test]
    fn test_gpu_detection_wasm_supports_f16_no_gpu() {
        // When no GPU is available, supports_f16 should return false
        let detection = GpuDetectionWasm::new();
        if !detection.available() {
            assert!(!detection.supports_f16());
        }
    }

    // =========================================================================
    // BackendTypeWasm Tests
    // =========================================================================

    #[test]
    fn test_backend_type_wasm_values() {
        assert_eq!(BackendTypeWasm::Simd as u32, 0);
        assert_eq!(BackendTypeWasm::Gpu as u32, 1);
        assert_eq!(BackendTypeWasm::Cpu as u32, 2);
        assert_eq!(BackendTypeWasm::Auto as u32, 3);
    }

    #[test]
    fn test_backend_type_wasm_conversion() {
        assert_eq!(
            BackendTypeWasm::from(BackendType::Simd),
            BackendTypeWasm::Simd
        );
        assert_eq!(
            BackendTypeWasm::from(BackendType::Gpu),
            BackendTypeWasm::Gpu
        );
        assert_eq!(BackendType::from(BackendTypeWasm::Simd), BackendType::Simd);
        assert_eq!(BackendType::from(BackendTypeWasm::Gpu), BackendType::Gpu);
    }

    // =========================================================================
    // SelectionStrategyWasm Tests
    // =========================================================================

    #[test]
    fn test_selection_strategy_wasm_values() {
        assert_eq!(SelectionStrategyWasm::PreferGpu as u32, 0);
        assert_eq!(SelectionStrategyWasm::PreferSimd as u32, 1);
        assert_eq!(SelectionStrategyWasm::Automatic as u32, 2);
        assert_eq!(SelectionStrategyWasm::Threshold as u32, 3);
    }

    #[test]
    fn test_selection_strategy_wasm_from_native() {
        assert_eq!(
            SelectionStrategyWasm::from(SelectionStrategy::PreferGpu),
            SelectionStrategyWasm::PreferGpu
        );
        assert_eq!(
            SelectionStrategyWasm::from(SelectionStrategy::Automatic),
            SelectionStrategyWasm::Automatic
        );
        assert_eq!(
            SelectionStrategyWasm::from(SelectionStrategy::threshold(1000)),
            SelectionStrategyWasm::Threshold
        );
    }

    // =========================================================================
    // BackendSelectionWasm Tests
    // =========================================================================

    #[test]
    fn test_backend_selection_wasm_from_native() {
        let native = BackendSelection::gpu("Test reason");
        let wasm: BackendSelectionWasm = native.into();

        assert!(wasm.is_gpu());
        assert!(!wasm.is_simd());
        assert_eq!(wasm.backend_name(), "GPU");
        assert_eq!(wasm.reason(), "Test reason");
    }

    #[test]
    fn test_backend_selection_wasm_simd() {
        let native = BackendSelection::simd("SIMD fallback");
        let wasm: BackendSelectionWasm = native.into();

        assert!(wasm.is_simd());
        assert!(!wasm.is_gpu());
        assert_eq!(wasm.backend_name(), "SIMD");
    }

    // =========================================================================
    // SelectorConfigWasm Tests
    // =========================================================================

    #[test]
    fn test_selector_config_wasm_new() {
        let config = SelectorConfigWasm::new();
        assert_eq!(config.strategy(), SelectionStrategyWasm::Automatic);
        assert_eq!(config.gpu_threshold_flops(), 100_000);
        assert_eq!(config.max_gpu_memory(), 256 * 1024 * 1024);
    }

    #[test]
    fn test_selector_config_wasm_for_inference() {
        let config = SelectorConfigWasm::for_inference();
        assert_eq!(config.strategy(), SelectionStrategyWasm::Automatic);
        assert_eq!(config.gpu_threshold_flops(), 1_000_000);
        assert_eq!(config.max_gpu_memory(), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_selector_config_wasm_prefer_gpu() {
        let config = SelectorConfigWasm::prefer_gpu();
        assert_eq!(config.strategy(), SelectionStrategyWasm::PreferGpu);
    }

    #[test]
    fn test_selector_config_wasm_prefer_simd() {
        let config = SelectorConfigWasm::prefer_simd();
        assert_eq!(config.strategy(), SelectionStrategyWasm::PreferSimd);
    }

    #[test]
    fn test_selector_config_wasm_setters() {
        let mut config = SelectorConfigWasm::new();
        config.set_strategy(SelectionStrategyWasm::PreferGpu);
        config.set_gpu_threshold_flops(500_000);
        config.set_max_gpu_memory(512 * 1024 * 1024);
        config.set_gpu_dispatch_overhead_us(200);

        assert_eq!(config.strategy(), SelectionStrategyWasm::PreferGpu);
        assert_eq!(config.gpu_threshold_flops(), 500_000);
        assert_eq!(config.max_gpu_memory(), 512 * 1024 * 1024);
        assert_eq!(config.gpu_dispatch_overhead_us(), 200);
    }

    #[test]
    fn test_selector_config_wasm_memory_mb() {
        let mut config = SelectorConfigWasm::new();
        config.set_max_gpu_memory_mb(512.0);
        assert!((config.max_gpu_memory_mb() - 512.0).abs() < 0.01);
    }

    // =========================================================================
    // BackendSelectorWasm Tests
    // =========================================================================

    #[test]
    fn test_backend_selector_wasm_new() {
        let config = SelectorConfigWasm::new();
        let selector = BackendSelectorWasm::new(config);
        assert!(!selector.summary().is_empty());
    }

    #[test]
    fn test_backend_selector_wasm_default() {
        let selector = BackendSelectorWasm::default_selector();
        assert!(!selector.summary().is_empty());
    }

    #[test]
    fn test_backend_selector_wasm_for_inference() {
        let selector = BackendSelectorWasm::for_inference();
        assert!(!selector.summary().is_empty());
    }

    #[test]
    fn test_backend_selector_wasm_select_for_mat_mul() {
        let selector = BackendSelectorWasm::default_selector();
        let selection = selector.select_for_mat_mul(64, 128, 64);

        assert!(!selection.reason().is_empty());
        assert!(!selection.backend_name().is_empty());
    }

    #[test]
    fn test_backend_selector_wasm_select_for_flops() {
        let selector = BackendSelectorWasm::default_selector();
        let selection = selector.select_for_flops(1_000_000, 1024 * 1024);

        assert!(!selection.reason().is_empty());
    }

    #[test]
    fn test_backend_selector_wasm_simd_performance_score() {
        let selector = BackendSelectorWasm::default_selector();
        let score = selector.simd_performance_score();
        assert!(score > 0.0);
    }

    // =========================================================================
    // Utility Function Tests
    // =========================================================================

    #[test]
    fn test_recommended_backend_for_model_tiny() {
        let rec = recommended_backend_for_model("tiny");
        assert!(rec.contains("SIMD"));
    }

    #[test]
    fn test_recommended_backend_for_model_large() {
        let rec = recommended_backend_for_model("large");
        assert!(rec.contains("GPU"));
    }

    #[test]
    fn test_estimate_mat_mul_memory() {
        // 64x128 * 128x64 with f32 (4 bytes)
        let mem = estimate_mat_mul_memory(64, 128, 64, 4);
        // A: 64*128*4 = 32768, B: 128*64*4 = 32768, C: 64*64*4 = 16384
        // Total: 81920
        assert_eq!(mem, 81920);
    }

    #[test]
    fn test_estimate_mat_mul_flops() {
        // 64x128 * 128x64
        let flops = estimate_mat_mul_flops(64, 128, 64);
        // 2 * 64 * 128 * 64 = 1,048,576
        assert_eq!(flops, 1_048_576);
    }

    #[test]
    fn test_is_gpu_worthwhile() {
        assert!(is_gpu_worthwhile(1_000_000, 100_000));
        assert!(!is_gpu_worthwhile(50_000, 100_000));
        assert!(is_gpu_worthwhile(100_000, 100_000)); // Exactly at threshold
    }

    // =========================================================================
    // GpuCapabilitiesWasm Tests
    // =========================================================================

    #[test]
    fn test_gpu_capabilities_wasm_backend_name() {
        // Test all backend name variants
        let caps = GpuCapabilitiesWasm {
            backend: GpuBackendWasm::Vulkan,
            device_name: "Test GPU".to_string(),
            vendor_name: "Test Vendor".to_string(),
            driver_info: "1.0.0".to_string(),
            supports_f16: true,
            supports_timestamp_query: true,
            limits: GpuLimitsWasm {
                max_buffer_size: 1024 * 1024 * 1024,
                max_storage_buffer_binding_size: 128 * 1024 * 1024,
                max_uniform_buffer_binding_size: 64 * 1024,
                max_compute_workgroup_size_x: 256,
                max_compute_workgroup_size_y: 256,
                max_compute_workgroup_size_z: 64,
                max_compute_invocations_per_workgroup: 256,
                max_compute_workgroups_per_dimension: 65535,
            },
        };

        assert_eq!(caps.backend_name(), "Vulkan");
        assert_eq!(caps.device_name(), "Test GPU");
        assert_eq!(caps.vendor_name(), "Test Vendor");
        assert!(caps.supports_f16());
        assert!(caps.summary().contains("Test GPU"));
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_gpu_limits_wasm_fields() {
        let limits = GpuLimitsWasm {
            max_buffer_size: 1024,
            max_storage_buffer_binding_size: 512,
            max_uniform_buffer_binding_size: 256,
            max_compute_workgroup_size_x: 128,
            max_compute_workgroup_size_y: 128,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 65535,
        };

        assert_eq!(limits.max_buffer_size, 1024);
        assert_eq!(limits.max_storage_buffer_binding_size, 512);
        assert_eq!(limits.max_uniform_buffer_binding_size, 256);
        assert_eq!(limits.max_compute_workgroup_size_x, 128);
        assert_eq!(limits.max_compute_workgroup_size_y, 128);
        assert_eq!(limits.max_compute_workgroup_size_z, 64);
        assert_eq!(limits.max_compute_invocations_per_workgroup, 256);
        assert_eq!(limits.max_compute_workgroups_per_dimension, 65535);
    }

    #[test]
    fn test_gpu_capabilities_wasm_all_backends() {
        let backends = vec![
            GpuBackendWasm::Vulkan,
            GpuBackendWasm::Metal,
            GpuBackendWasm::Dx12,
            GpuBackendWasm::Gl,
            GpuBackendWasm::BrowserWebGpu,
            GpuBackendWasm::None,
        ];

        for backend in backends {
            let caps = GpuCapabilitiesWasm {
                backend,
                device_name: "Test".to_string(),
                vendor_name: "Vendor".to_string(),
                driver_info: "1.0".to_string(),
                supports_f16: false,
                supports_timestamp_query: false,
                limits: GpuLimitsWasm {
                    max_buffer_size: 1024,
                    max_storage_buffer_binding_size: 512,
                    max_uniform_buffer_binding_size: 256,
                    max_compute_workgroup_size_x: 64,
                    max_compute_workgroup_size_y: 64,
                    max_compute_workgroup_size_z: 64,
                    max_compute_invocations_per_workgroup: 256,
                    max_compute_workgroups_per_dimension: 65535,
                },
            };

            let name = caps.backend_name();
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_detection_options_wasm_timeout() {
        let mut options = DetectionOptionsWasm::new();
        options.set_timeout_ms(5000);
        options.set_prefer_high_performance(true);
        options.set_require_f16(true);

        assert!(options.prefer_high_performance);
        assert!(options.require_f16);
    }

    #[test]
    fn test_backend_selector_wasm_with_config() {
        let config = SelectorConfigWasm::new();
        let selector = BackendSelectorWasm::new(config);

        // Just verify creation succeeded
        let simd_score = selector.simd_performance_score();
        assert!(simd_score >= 0.0);
    }

    #[test]
    fn test_recommended_backend_for_model_all_sizes() {
        let sizes = vec!["tiny", "base", "small", "medium", "large"];

        for size in sizes {
            let rec = recommended_backend_for_model(size);
            assert!(!rec.is_empty());
        }
    }

    #[test]
    fn test_recommended_backend_for_model_unknown() {
        // Unknown models default to tiny parameters (39M), so SIMD is recommended
        let rec = recommended_backend_for_model("unknown");
        assert!(rec.contains("SIMD"));
    }

    #[test]
    fn test_gpu_backend_wasm_dx12_and_none_conversion() {
        use super::GpuBackend;
        // Test Dx12 to GpuBackendWasm conversion
        let wasm: GpuBackendWasm = GpuBackend::Dx12.into();
        assert_eq!(wasm, GpuBackendWasm::Dx12);

        // Test Dx11 maps to Gl
        let wasm2: GpuBackendWasm = GpuBackend::Dx11.into();
        assert_eq!(wasm2, GpuBackendWasm::Gl);

        // Test None conversion
        let wasm3: GpuBackendWasm = GpuBackend::None.into();
        assert_eq!(wasm3, GpuBackendWasm::None);
    }

    #[test]
    fn test_gpu_backend_wasm_to_native_extra() {
        use super::GpuBackend;
        // Test Dx12 from wasm back to native
        assert_eq!(GpuBackend::from(GpuBackendWasm::Dx12), GpuBackend::Dx12);
        // Test Gl maps to OpenGl
        assert_eq!(GpuBackend::from(GpuBackendWasm::Gl), GpuBackend::OpenGl);
        // Test BrowserWebGpu
        assert_eq!(
            GpuBackend::from(GpuBackendWasm::BrowserWebGpu),
            GpuBackend::BrowserWebGpu
        );
        // Test None
        assert_eq!(GpuBackend::from(GpuBackendWasm::None), GpuBackend::None);
    }

    #[test]
    fn test_gpu_limits_wasm_getters() {
        let limits = GpuLimitsWasm {
            max_buffer_size: 1024 * 1024,
            max_storage_buffer_binding_size: 512,
            max_uniform_buffer_binding_size: 256,
            max_compute_workgroup_size_x: 128,
            max_compute_workgroup_size_y: 64,
            max_compute_workgroup_size_z: 32,
            max_compute_invocations_per_workgroup: 512,
            max_compute_workgroups_per_dimension: 65535,
        };

        // Test all getter methods
        assert_eq!(limits.max_compute_workgroup_size_y(), 64);
        assert_eq!(limits.max_compute_workgroup_size_z(), 32);
        assert_eq!(limits.max_compute_invocations_per_workgroup(), 512);
        assert_eq!(limits.max_compute_workgroups_per_dimension(), 65535);
    }
}

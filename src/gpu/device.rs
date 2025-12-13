//! GPU device initialization and management

use super::capabilities::{GpuBackend, GpuCapabilities, GpuLimits};
use super::error::{GpuError, GpuResult};

/// GPU device configuration
#[derive(Debug, Clone)]
pub struct GpuDeviceConfig {
    /// Prefer high-performance GPU over power-efficient
    pub high_performance: bool,
    /// Required features (empty = use defaults)
    pub required_features: Vec<String>,
    /// Minimum buffer size in bytes
    pub min_buffer_size: u64,
    /// Enable debug validation (slower but catches errors)
    pub validation: bool,
}

impl Default for GpuDeviceConfig {
    fn default() -> Self {
        Self {
            high_performance: true,
            required_features: Vec::new(),
            min_buffer_size: 128 * 1024 * 1024, // 128 MB
            validation: cfg!(debug_assertions),
        }
    }
}

impl GpuDeviceConfig {
    /// Create config for inference workloads
    #[must_use]
    pub fn for_inference() -> Self {
        Self {
            high_performance: true,
            required_features: Vec::new(),
            min_buffer_size: 256 * 1024 * 1024, // 256 MB for model weights
            validation: false,
        }
    }

    /// Create config for development/debugging
    #[must_use]
    pub fn for_development() -> Self {
        Self {
            high_performance: false,
            required_features: Vec::new(),
            min_buffer_size: 64 * 1024 * 1024,
            validation: true,
        }
    }

    /// Set minimum buffer size
    #[must_use]
    pub fn with_min_buffer_size(mut self, size: u64) -> Self {
        self.min_buffer_size = size;
        self
    }

    /// Enable or disable validation
    #[must_use]
    pub fn with_validation(mut self, enable: bool) -> Self {
        self.validation = enable;
        self
    }
}

/// GPU device state for compute operations
///
/// This is the main entry point for GPU operations. It manages the
/// connection to the GPU and provides methods to create buffers and
/// execute compute shaders.
#[derive(Debug)]
pub struct GpuDevice {
    /// Device capabilities
    capabilities: GpuCapabilities,
    /// Device configuration
    config: GpuDeviceConfig,
    /// Whether device is initialized
    initialized: bool,
    /// Internal wgpu device (when feature enabled)
    #[cfg(feature = "webgpu")]
    device: Option<wgpu::Device>,
    #[cfg(feature = "webgpu")]
    queue: Option<wgpu::Queue>,
}

impl GpuDevice {
    /// Create a new GPU device with default configuration
    ///
    /// Returns an error if no suitable GPU is available.
    pub fn new() -> GpuResult<Self> {
        Self::with_config(GpuDeviceConfig::default())
    }

    /// Create a new GPU device with specific configuration
    pub fn with_config(config: GpuDeviceConfig) -> GpuResult<Self> {
        let capabilities = Self::detect_capabilities()?;

        // Verify minimum requirements
        if !capabilities.is_available() {
            return Err(GpuError::NotAvailable);
        }

        if capabilities.limits.max_buffer_size < config.min_buffer_size {
            return Err(GpuError::InvalidBufferSize {
                requested: config.min_buffer_size,
                max: capabilities.limits.max_buffer_size,
            });
        }

        Ok(Self {
            capabilities,
            config,
            initialized: true,
            #[cfg(feature = "webgpu")]
            device: None,
            #[cfg(feature = "webgpu")]
            queue: None,
        })
    }

    /// Create a simulated device for testing
    #[must_use]
    pub fn simulated() -> Self {
        Self {
            capabilities: GpuCapabilities {
                name: "Simulated GPU".to_string(),
                vendor: "Test".to_string(),
                backend: GpuBackend::None,
                limits: GpuLimits::default(),
                supports_f16: true,
                supports_timestamp_query: false,
                vram_bytes: 4 * 1024 * 1024 * 1024,
            },
            config: GpuDeviceConfig::default(),
            initialized: false,
            #[cfg(feature = "webgpu")]
            device: None,
            #[cfg(feature = "webgpu")]
            queue: None,
        }
    }

    /// Detect GPU capabilities without creating a device
    fn detect_capabilities() -> GpuResult<GpuCapabilities> {
        // In non-webgpu builds, return unavailable
        #[cfg(not(feature = "webgpu"))]
        {
            Ok(GpuCapabilities::default())
        }

        #[cfg(feature = "webgpu")]
        {
            // Will be implemented when wgpu is used
            Ok(GpuCapabilities::default())
        }
    }

    /// Check if GPU is available on this system
    #[must_use]
    pub fn is_available() -> bool {
        #[cfg(feature = "webgpu")]
        {
            // TODO: Implement actual detection
            false
        }
        #[cfg(not(feature = "webgpu"))]
        {
            false
        }
    }

    /// Get device capabilities
    #[must_use]
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    /// Get device configuration
    #[must_use]
    pub fn config(&self) -> &GpuDeviceConfig {
        &self.config
    }

    /// Check if device is initialized and ready
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the backend type
    #[must_use]
    pub fn backend(&self) -> GpuBackend {
        self.capabilities.backend
    }

    /// Check if this device supports compute shaders
    #[must_use]
    pub fn supports_compute(&self) -> bool {
        self.capabilities.supports_compute()
    }

    /// Check if float16 is supported
    #[must_use]
    pub fn supports_f16(&self) -> bool {
        self.capabilities.supports_f16
    }

    /// Get maximum buffer size
    #[must_use]
    pub fn max_buffer_size(&self) -> u64 {
        self.capabilities.limits.max_buffer_size
    }

    /// Calculate optimal workgroup size for a dimension
    #[must_use]
    pub fn optimal_workgroup_size(&self, elements: u32) -> u32 {
        self.capabilities.limits.optimal_workgroup_size(elements)
    }

    /// Get human-readable device info
    #[must_use]
    pub fn info(&self) -> String {
        format!(
            "GpuDevice: {} ({}) - {}",
            self.capabilities.name,
            self.capabilities.backend,
            if self.initialized { "Ready" } else { "Not initialized" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_config_default() {
        let config = GpuDeviceConfig::default();
        assert!(config.high_performance);
        assert!(config.required_features.is_empty());
        assert_eq!(config.min_buffer_size, 128 * 1024 * 1024);
    }

    #[test]
    fn test_device_config_for_inference() {
        let config = GpuDeviceConfig::for_inference();
        assert!(config.high_performance);
        assert!(!config.validation);
        assert_eq!(config.min_buffer_size, 256 * 1024 * 1024);
    }

    #[test]
    fn test_device_config_for_development() {
        let config = GpuDeviceConfig::for_development();
        assert!(!config.high_performance);
        assert!(config.validation);
    }

    #[test]
    fn test_device_config_builder() {
        let config = GpuDeviceConfig::default()
            .with_min_buffer_size(512 * 1024 * 1024)
            .with_validation(true);

        assert_eq!(config.min_buffer_size, 512 * 1024 * 1024);
        assert!(config.validation);
    }

    #[test]
    fn test_gpu_device_simulated() {
        let device = GpuDevice::simulated();
        assert!(!device.is_initialized());
        assert_eq!(device.capabilities().name, "Simulated GPU");
        assert!(device.capabilities().supports_f16);
    }

    #[test]
    fn test_gpu_device_capabilities() {
        let device = GpuDevice::simulated();
        let caps = device.capabilities();
        assert_eq!(caps.vram_bytes, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_gpu_device_config() {
        let device = GpuDevice::simulated();
        let config = device.config();
        assert!(config.high_performance);
    }

    #[test]
    fn test_gpu_device_backend() {
        let device = GpuDevice::simulated();
        assert_eq!(device.backend(), GpuBackend::None);
    }

    #[test]
    fn test_gpu_device_supports_compute() {
        let device = GpuDevice::simulated();
        assert!(!device.supports_compute()); // Simulated has None backend
    }

    #[test]
    fn test_gpu_device_supports_f16() {
        let device = GpuDevice::simulated();
        assert!(device.supports_f16());
    }

    #[test]
    fn test_gpu_device_max_buffer_size() {
        let device = GpuDevice::simulated();
        assert_eq!(device.max_buffer_size(), 256 * 1024 * 1024);
    }

    #[test]
    fn test_gpu_device_optimal_workgroup_size() {
        let device = GpuDevice::simulated();
        assert_eq!(device.optimal_workgroup_size(100), 128);
        assert_eq!(device.optimal_workgroup_size(256), 256);
    }

    #[test]
    fn test_gpu_device_info() {
        let device = GpuDevice::simulated();
        let info = device.info();
        assert!(info.contains("Simulated GPU"));
        assert!(info.contains("Not initialized"));
    }

    #[test]
    fn test_gpu_device_is_available() {
        // Without webgpu feature, should return false
        assert!(!GpuDevice::is_available());
    }
}

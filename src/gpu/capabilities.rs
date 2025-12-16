//! GPU capabilities detection and reporting

use std::fmt;

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// Vulkan backend (Linux, Windows, Android)
    Vulkan,
    /// Metal backend (macOS, iOS)
    Metal,
    /// DirectX 12 backend (Windows)
    Dx12,
    /// DirectX 11 backend (Windows fallback)
    Dx11,
    /// OpenGL backend (fallback)
    OpenGl,
    /// WebGPU browser backend
    BrowserWebGpu,
    /// No GPU available
    None,
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Metal => write!(f, "Metal"),
            Self::Dx12 => write!(f, "DirectX 12"),
            Self::Dx11 => write!(f, "DirectX 11"),
            Self::OpenGl => write!(f, "OpenGL"),
            Self::BrowserWebGpu => write!(f, "WebGPU (Browser)"),
            Self::None => write!(f, "None"),
        }
    }
}

impl GpuBackend {
    /// Check if this backend supports compute shaders
    #[must_use]
    pub const fn supports_compute(&self) -> bool {
        !matches!(self, Self::None | Self::OpenGl)
    }

    /// Check if this is a high-performance backend
    #[must_use]
    pub const fn is_high_performance(&self) -> bool {
        matches!(self, Self::Vulkan | Self::Metal | Self::Dx12)
    }

    /// Check if this is a browser backend
    #[must_use]
    pub const fn is_browser(&self) -> bool {
        matches!(self, Self::BrowserWebGpu)
    }
}

/// GPU device limits
#[derive(Debug, Clone)]
pub struct GpuLimits {
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Maximum storage buffer binding size
    pub max_storage_buffer_binding_size: u32,
    /// Maximum compute workgroup size X
    pub max_compute_workgroup_size_x: u32,
    /// Maximum compute workgroup size Y
    pub max_compute_workgroup_size_y: u32,
    /// Maximum compute workgroup size Z
    pub max_compute_workgroup_size_z: u32,
    /// Maximum compute invocations per workgroup
    pub max_compute_invocations_per_workgroup: u32,
    /// Maximum compute workgroups per dimension
    pub max_compute_workgroups_per_dimension: u32,
    /// Maximum bind groups
    pub max_bind_groups: u32,
    /// Maximum uniform buffer binding size
    pub max_uniform_buffer_binding_size: u32,
}

impl Default for GpuLimits {
    fn default() -> Self {
        // WebGPU minimum guaranteed limits
        Self {
            max_buffer_size: 256 * 1024 * 1024,                 // 256 MB
            max_storage_buffer_binding_size: 128 * 1024 * 1024, // 128 MB
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 65535,
            max_bind_groups: 4,
            max_uniform_buffer_binding_size: 64 * 1024, // 64 KB
        }
    }
}

impl GpuLimits {
    /// Create limits for high-end desktop GPUs
    #[must_use]
    pub fn desktop_high_end() -> Self {
        Self {
            max_buffer_size: 2 * 1024 * 1024 * 1024,             // 2 GB
            max_storage_buffer_binding_size: 1024 * 1024 * 1024, // 1 GB
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_size_y: 1024,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroups_per_dimension: 65535,
            max_bind_groups: 8,
            max_uniform_buffer_binding_size: 64 * 1024,
        }
    }

    /// Create limits for mobile GPUs
    #[must_use]
    pub fn mobile() -> Self {
        Self {
            max_buffer_size: 128 * 1024 * 1024,                // 128 MB
            max_storage_buffer_binding_size: 64 * 1024 * 1024, // 64 MB
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroups_per_dimension: 65535,
            max_bind_groups: 4,
            max_uniform_buffer_binding_size: 64 * 1024,
        }
    }

    /// Check if a buffer size is within limits
    #[must_use]
    pub fn buffer_size_ok(&self, size: u64) -> bool {
        size <= self.max_buffer_size
    }

    /// Calculate optimal workgroup size for a given dimension
    #[must_use]
    pub fn optimal_workgroup_size(&self, elements: u32) -> u32 {
        let max_size = self.max_compute_workgroup_size_x;
        if elements <= max_size {
            // Round up to power of 2 for efficiency
            let mut size = 1u32;
            while size < elements && size < max_size {
                size *= 2;
            }
            size
        } else {
            max_size
        }
    }
}

/// GPU capabilities and features
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// GPU device name
    pub name: String,
    /// GPU vendor
    pub vendor: String,
    /// Backend type
    pub backend: GpuBackend,
    /// Device limits
    pub limits: GpuLimits,
    /// Whether float16 is supported
    pub supports_f16: bool,
    /// Whether timestamp queries are supported
    pub supports_timestamp_query: bool,
    /// Estimated VRAM in bytes (0 if unknown)
    pub vram_bytes: u64,
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            vendor: "Unknown".to_string(),
            backend: GpuBackend::None,
            limits: GpuLimits::default(),
            supports_f16: false,
            supports_timestamp_query: false,
            vram_bytes: 0,
        }
    }
}

impl GpuCapabilities {
    /// Check if any GPU is available
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.backend != GpuBackend::None
    }

    /// Check if compute shaders are supported
    #[must_use]
    pub fn supports_compute(&self) -> bool {
        self.backend.supports_compute()
    }

    /// Check if this GPU is suitable for large model inference
    ///
    /// Requires:
    /// - Compute shader support
    /// - At least 256 MB buffer size
    /// - At least 128 MB storage buffer binding
    #[must_use]
    pub fn suitable_for_inference(&self) -> bool {
        self.supports_compute()
            && self.limits.max_buffer_size >= 256 * 1024 * 1024
            && self.limits.max_storage_buffer_binding_size >= 128 * 1024 * 1024
    }

    /// Estimate if a model of given size (in bytes) can fit
    #[must_use]
    pub fn can_fit_model(&self, model_bytes: u64) -> bool {
        // Model needs ~2x size for weights + activations
        let required = model_bytes * 2;
        if self.vram_bytes > 0 {
            required <= self.vram_bytes
        } else {
            // Unknown VRAM, use buffer limit as proxy
            required <= self.limits.max_buffer_size
        }
    }

    /// Get estimated VRAM in MB
    #[must_use]
    pub fn vram_mb(&self) -> f32 {
        self.vram_bytes as f32 / (1024.0 * 1024.0)
    }

    /// Get human-readable summary
    #[must_use]
    pub fn summary(&self) -> String {
        let vram = if self.vram_bytes > 0 {
            format!("{:.0} MB", self.vram_mb())
        } else {
            "Unknown".to_string()
        };

        format!(
            "GPU: {} ({}) - VRAM: {}, F16: {}, Compute: {}",
            self.name,
            self.backend,
            vram,
            if self.supports_f16 { "Yes" } else { "No" },
            if self.supports_compute() { "Yes" } else { "No" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_display() {
        assert_eq!(GpuBackend::Vulkan.to_string(), "Vulkan");
        assert_eq!(GpuBackend::Metal.to_string(), "Metal");
        assert_eq!(GpuBackend::Dx12.to_string(), "DirectX 12");
        assert_eq!(GpuBackend::BrowserWebGpu.to_string(), "WebGPU (Browser)");
        assert_eq!(GpuBackend::None.to_string(), "None");
    }

    #[test]
    fn test_gpu_backend_supports_compute() {
        assert!(GpuBackend::Vulkan.supports_compute());
        assert!(GpuBackend::Metal.supports_compute());
        assert!(GpuBackend::Dx12.supports_compute());
        assert!(GpuBackend::BrowserWebGpu.supports_compute());
        assert!(!GpuBackend::None.supports_compute());
        assert!(!GpuBackend::OpenGl.supports_compute());
    }

    #[test]
    fn test_gpu_backend_is_high_performance() {
        assert!(GpuBackend::Vulkan.is_high_performance());
        assert!(GpuBackend::Metal.is_high_performance());
        assert!(GpuBackend::Dx12.is_high_performance());
        assert!(!GpuBackend::OpenGl.is_high_performance());
        assert!(!GpuBackend::BrowserWebGpu.is_high_performance());
    }

    #[test]
    fn test_gpu_backend_is_browser() {
        assert!(GpuBackend::BrowserWebGpu.is_browser());
        assert!(!GpuBackend::Vulkan.is_browser());
        assert!(!GpuBackend::None.is_browser());
    }

    #[test]
    fn test_gpu_limits_default() {
        let limits = GpuLimits::default();
        assert_eq!(limits.max_buffer_size, 256 * 1024 * 1024);
        assert_eq!(limits.max_compute_workgroup_size_x, 256);
        assert_eq!(limits.max_bind_groups, 4);
    }

    #[test]
    fn test_gpu_limits_desktop_high_end() {
        let limits = GpuLimits::desktop_high_end();
        assert!(limits.max_buffer_size > GpuLimits::default().max_buffer_size);
        assert!(limits.max_compute_workgroup_size_x >= 1024);
    }

    #[test]
    fn test_gpu_limits_mobile() {
        let limits = GpuLimits::mobile();
        assert!(limits.max_buffer_size < GpuLimits::default().max_buffer_size);
    }

    #[test]
    fn test_gpu_limits_buffer_size_ok() {
        let limits = GpuLimits::default();
        assert!(limits.buffer_size_ok(1024));
        assert!(limits.buffer_size_ok(256 * 1024 * 1024));
        assert!(!limits.buffer_size_ok(512 * 1024 * 1024));
    }

    #[test]
    fn test_gpu_limits_optimal_workgroup_size() {
        let limits = GpuLimits::default();

        // Small elements -> round up to power of 2
        assert_eq!(limits.optimal_workgroup_size(50), 64);
        assert_eq!(limits.optimal_workgroup_size(100), 128);
        assert_eq!(limits.optimal_workgroup_size(200), 256);

        // At limit
        assert_eq!(limits.optimal_workgroup_size(256), 256);

        // Over limit -> cap at max
        assert_eq!(limits.optimal_workgroup_size(1000), 256);
    }

    #[test]
    fn test_gpu_capabilities_default() {
        let caps = GpuCapabilities::default();
        assert!(!caps.is_available());
        assert!(!caps.supports_compute());
        assert_eq!(caps.backend, GpuBackend::None);
    }

    #[test]
    fn test_gpu_capabilities_is_available() {
        let mut caps = GpuCapabilities::default();
        assert!(!caps.is_available());

        caps.backend = GpuBackend::Vulkan;
        assert!(caps.is_available());
    }

    #[test]
    fn test_gpu_capabilities_supports_compute() {
        let mut caps = GpuCapabilities::default();
        caps.backend = GpuBackend::Vulkan;
        assert!(caps.supports_compute());

        caps.backend = GpuBackend::OpenGl;
        assert!(!caps.supports_compute());
    }

    #[test]
    fn test_gpu_capabilities_suitable_for_inference() {
        let mut caps = GpuCapabilities::default();
        assert!(!caps.suitable_for_inference()); // No backend

        caps.backend = GpuBackend::Vulkan;
        assert!(caps.suitable_for_inference()); // Default limits are sufficient

        caps.limits.max_buffer_size = 64 * 1024 * 1024;
        assert!(!caps.suitable_for_inference()); // Buffer too small
    }

    #[test]
    fn test_gpu_capabilities_can_fit_model() {
        let mut caps = GpuCapabilities::default();
        caps.vram_bytes = 4 * 1024 * 1024 * 1024; // 4 GB

        // 1 GB model needs ~2 GB
        assert!(caps.can_fit_model(1024 * 1024 * 1024));

        // 3 GB model needs ~6 GB
        assert!(!caps.can_fit_model(3 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_gpu_capabilities_vram_mb() {
        let mut caps = GpuCapabilities::default();
        caps.vram_bytes = 4 * 1024 * 1024 * 1024;
        assert!((caps.vram_mb() - 4096.0).abs() < 1.0);
    }

    #[test]
    fn test_gpu_capabilities_summary() {
        let mut caps = GpuCapabilities::default();
        caps.name = "RTX 4090".to_string();
        caps.backend = GpuBackend::Vulkan;
        caps.vram_bytes = 24 * 1024 * 1024 * 1024;
        caps.supports_f16 = true;

        let summary = caps.summary();
        assert!(summary.contains("RTX 4090"));
        assert!(summary.contains("Vulkan"));
        assert!(summary.contains("Yes")); // F16 support
    }
}

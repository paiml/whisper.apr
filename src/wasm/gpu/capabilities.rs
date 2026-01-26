//! WASM GPU capabilities bindings

use wasm_bindgen::prelude::*;

use crate::gpu::GpuCapabilities;

use super::backend::GpuBackendWasm;
use super::limits::GpuLimitsWasm;

/// WASM-friendly GPU capabilities
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct GpuCapabilitiesWasm {
    pub(super) backend: GpuBackendWasm,
    pub(super) device_name: String,
    pub(super) vendor_name: String,
    pub(super) driver_info: String,
    pub(super) supports_f16: bool,
    pub(super) supports_timestamp_query: bool,
    pub(super) limits: GpuLimitsWasm,
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

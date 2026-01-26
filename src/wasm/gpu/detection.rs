//! WASM GPU detection bindings

use wasm_bindgen::prelude::*;

use crate::gpu::{detect_gpu, DetectionOptions};

use super::capabilities::GpuCapabilitiesWasm;

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

/// WASM-friendly detection options
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DetectionOptionsWasm {
    pub(super) prefer_high_performance: bool,
    pub(super) require_f16: bool,
    pub(super) timeout_ms: u32,
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

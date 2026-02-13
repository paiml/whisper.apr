//! WASM backend types and conversions

use wasm_bindgen::prelude::*;

use crate::backend::{BackendSelection, BackendType, SelectionStrategy};
use crate::gpu::GpuBackend;

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
    #[allow(clippy::match_same_arms)]
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

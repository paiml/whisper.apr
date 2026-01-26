//! WASM backend selector bindings

use wasm_bindgen::prelude::*;

use crate::backend::{BackendSelector, MatMulOp, SelectionStrategy, SelectorConfig};

use super::backend::{BackendSelectionWasm, SelectionStrategyWasm};

/// WASM-friendly selector configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SelectorConfigWasm {
    pub(super) strategy: SelectionStrategyWasm,
    pub(super) gpu_threshold_flops: u64,
    pub(super) max_gpu_memory: u64,
    pub(super) gpu_dispatch_overhead_us: u32,
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

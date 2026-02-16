//! WASM capability detection
//!
//! Provides runtime detection of browser capabilities for the twin-binary fallback strategy.
//!
//! # Twin-Binary Strategy (Spec 8.3)
//!
//! The build pipeline produces distinct artifacts:
//!
//! | Binary | SIMD | Threads | Use Case |
//! |--------|------|---------|----------|
//! | `simd-threaded` | Yes | Yes | Modern browsers with COOP/COEP headers |
//! | `simd-sequential` | Yes | No | Environments lacking SharedArrayBuffer |
//! | `scalar` | No | No | Legacy hardware, very restrictive environments |
//!
//! # Usage
//!
//! ```javascript
//! // JavaScript side capability detection
//! import { detectCapabilities, getBinaryName } from 'whisper-apr';
//!
//! const caps = await detectCapabilities();
//! console.log(caps.simd); // true/false
//! console.log(caps.threads); // true/false
//!
//! const binary = getBinaryName(); // e.g., "whisper-apr-simd-sequential.wasm"
//! ```

#[cfg(test)]
mod tests;

use wasm_bindgen::prelude::*;

/// Runtime capabilities of the current environment
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct Capabilities {
    /// WASM SIMD 128-bit support
    simd: bool,
    /// SharedArrayBuffer (threading) support
    threads: bool,
    /// Cross-origin isolated environment
    cross_origin_isolated: bool,
    /// WebGPU support (future)
    webgpu: bool,
    /// Available memory in MB (approximate)
    memory_mb: u32,
    /// Number of hardware threads
    hardware_concurrency: u32,
}

#[wasm_bindgen]
impl Capabilities {
    /// Create new capabilities with default values
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create capabilities with specific values (for testing)
    #[must_use]
    pub fn with_values(
        simd: bool,
        threads: bool,
        cross_origin_isolated: bool,
        hardware_concurrency: u32,
    ) -> Self {
        Self {
            simd,
            threads,
            cross_origin_isolated,
            webgpu: false,
            memory_mb: 0,
            hardware_concurrency,
        }
    }

    /// Check SIMD support
    #[wasm_bindgen(getter)]
    pub fn simd(&self) -> bool {
        self.simd
    }

    /// Check threading support
    #[wasm_bindgen(getter)]
    pub fn threads(&self) -> bool {
        self.threads
    }

    /// Check cross-origin isolation
    #[wasm_bindgen(getter, js_name = crossOriginIsolated)]
    pub fn cross_origin_isolated(&self) -> bool {
        self.cross_origin_isolated
    }

    /// Check WebGPU support
    #[wasm_bindgen(getter)]
    pub fn webgpu(&self) -> bool {
        self.webgpu
    }

    /// Get available memory in MB
    #[wasm_bindgen(getter, js_name = memoryMb)]
    pub fn memory_mb(&self) -> u32 {
        self.memory_mb
    }

    /// Get hardware concurrency (thread count)
    #[wasm_bindgen(getter, js_name = hardwareConcurrency)]
    pub fn hardware_concurrency(&self) -> u32 {
        self.hardware_concurrency
    }

    /// Set SIMD support
    #[wasm_bindgen(setter)]
    pub fn set_simd(&mut self, value: bool) {
        self.simd = value;
    }

    /// Set threads support
    #[wasm_bindgen(setter)]
    pub fn set_threads(&mut self, value: bool) {
        self.threads = value;
    }

    /// Set cross-origin isolated
    #[wasm_bindgen(setter, js_name = setCrossOriginIsolated)]
    pub fn set_cross_origin_isolated(&mut self, value: bool) {
        self.cross_origin_isolated = value;
    }

    /// Set WebGPU support
    #[wasm_bindgen(setter)]
    pub fn set_webgpu(&mut self, value: bool) {
        self.webgpu = value;
    }

    /// Set memory in MB
    #[wasm_bindgen(setter, js_name = setMemoryMb)]
    pub fn set_memory_mb(&mut self, value: u32) {
        self.memory_mb = value;
    }

    /// Set hardware concurrency
    #[wasm_bindgen(setter, js_name = setHardwareConcurrency)]
    pub fn set_hardware_concurrency(&mut self, value: u32) {
        self.hardware_concurrency = value;
    }

    /// Get the recommended binary name based on capabilities
    #[wasm_bindgen(js_name = getBinaryName)]
    pub fn get_binary_name(&self) -> String {
        if self.simd && self.threads && self.cross_origin_isolated {
            "whisper-apr-simd-threaded.wasm".to_string()
        } else if self.simd {
            "whisper-apr-simd-sequential.wasm".to_string()
        } else {
            "whisper-apr-scalar.wasm".to_string()
        }
    }

    /// Get the optimal thread count for this environment
    ///
    /// Per spec 10.3: N_threads = max(1, min(hardwareConcurrency - 1, N_limit))
    #[wasm_bindgen(js_name = optimalThreadCount)]
    pub fn optimal_thread_count(&self) -> u32 {
        if !self.threads {
            return 1;
        }

        let hw = self.hardware_concurrency;
        if hw <= 1 {
            return 1;
        }

        // Reserve 1 thread for UI/audio, cap at 8 for diminishing returns
        let available = hw.saturating_sub(1);
        available.clamp(1, 8)
    }

    /// Check if the environment can run the specified model
    #[wasm_bindgen(js_name = canRunModel)]
    #[allow(clippy::match_same_arms)]
    pub fn can_run_model(&self, model_type: &str) -> bool {
        let required_mb = match model_type {
            "tiny" | "tiny.en" => 200,
            "base" | "base.en" => 400,
            "small" | "small.en" => 900,
            "medium" | "medium.en" => 2500,
            "large" | "large-v2" | "large-v3" => 4000,
            "large-v3-turbo" => 2500,
            _ => 200, // Default to tiny requirements
        };

        // If we don't know memory, assume it's enough
        if self.memory_mb == 0 {
            return true;
        }

        self.memory_mb >= required_mb
    }

    /// Get performance tier (0-3)
    ///
    /// - 3: SIMD + Threads (best)
    /// - 2: SIMD only
    /// - 1: Threads only
    /// - 0: Scalar (baseline)
    #[wasm_bindgen(js_name = performanceTier)]
    pub fn performance_tier(&self) -> u8 {
        match (self.simd, self.threads) {
            (true, true) => 3,
            (true, false) => 2,
            (false, true) => 1,
            (false, false) => 0,
        }
    }

    /// Get human-readable description of capabilities
    #[wasm_bindgen(js_name = description)]
    pub fn description(&self) -> String {
        let mut parts = Vec::new();

        if self.simd {
            parts.push("SIMD");
        }
        if self.threads {
            parts.push("Threads");
        }
        if self.cross_origin_isolated {
            parts.push("CrossOriginIsolated");
        }
        if self.webgpu {
            parts.push("WebGPU");
        }

        if parts.is_empty() {
            "Scalar (no acceleration)".to_string()
        } else {
            parts.join(" + ")
        }
    }

    /// Get the execution mode for this capability set
    #[wasm_bindgen(js_name = executionMode)]
    pub fn execution_mode(&self) -> ExecutionMode {
        ExecutionMode::from(self)
    }

    /// Get the RTF (Real-Time Factor) multiplier
    ///
    /// Lower is better. 1.0 = real-time.
    #[wasm_bindgen(js_name = rtfMultiplier)]
    pub fn rtf_multiplier(&self) -> f32 {
        self.execution_mode().rtf_multiplier()
    }

    /// Get execution mode name
    #[wasm_bindgen(js_name = executionModeName)]
    pub fn execution_mode_name(&self) -> String {
        self.execution_mode().name()
    }
}

/// Execution mode based on detected capabilities
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// SIMD + multi-threading (best performance)
    SimdThreaded,
    /// SIMD only (good performance)
    SimdSequential,
    /// Scalar only (baseline)
    Scalar,
}

impl From<&Capabilities> for ExecutionMode {
    fn from(caps: &Capabilities) -> Self {
        if caps.simd && caps.threads && caps.cross_origin_isolated {
            Self::SimdThreaded
        } else if caps.simd {
            Self::SimdSequential
        } else {
            Self::Scalar
        }
    }
}

impl ExecutionMode {
    /// Get the RTF (Real-Time Factor) multiplier for this mode
    ///
    /// Lower is better. 1.0 = real-time.
    #[must_use]
    pub fn rtf_multiplier(self) -> f32 {
        match self {
            Self::SimdThreaded => 1.0,
            Self::SimdSequential => 1.5,
            Self::Scalar => 4.0,
        }
    }

    /// Get human-readable name
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::SimdThreaded => "High Performance (SIMD + Threads)".to_string(),
            Self::SimdSequential => "Compatibility (SIMD Sequential)".to_string(),
            Self::Scalar => "Fallback (Scalar)".to_string(),
        }
    }
}

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

// ============================================================================
// Tests (native Rust only - WASM tests would need wasm-bindgen-test)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Capabilities Tests
    // =========================================================================

    #[test]
    fn test_capabilities_default() {
        let caps = Capabilities::default();
        assert!(!caps.simd);
        assert!(!caps.threads);
        assert!(!caps.cross_origin_isolated);
        assert!(!caps.webgpu);
    }

    #[test]
    fn test_capabilities_with_values() {
        let caps = Capabilities::with_values(true, true, true, 8);
        assert!(caps.simd);
        assert!(caps.threads);
        assert!(caps.cross_origin_isolated);
        assert_eq!(caps.hardware_concurrency, 8);
    }

    #[test]
    fn test_capabilities_setters() {
        let mut caps = Capabilities::new();
        caps.set_simd(true);
        caps.set_threads(true);
        caps.set_cross_origin_isolated(true);
        caps.set_memory_mb(1024);
        caps.set_hardware_concurrency(4);

        assert!(caps.simd());
        assert!(caps.threads());
        assert!(caps.cross_origin_isolated());
        assert_eq!(caps.memory_mb(), 1024);
        assert_eq!(caps.hardware_concurrency(), 4);
    }

    // =========================================================================
    // Binary Name Tests
    // =========================================================================

    #[test]
    fn test_binary_name_simd_threaded() {
        let caps = Capabilities::with_values(true, true, true, 8);
        assert_eq!(caps.get_binary_name(), "whisper-apr-simd-threaded.wasm");
    }

    #[test]
    fn test_binary_name_simd_sequential() {
        let caps = Capabilities::with_values(true, false, false, 4);
        assert_eq!(caps.get_binary_name(), "whisper-apr-simd-sequential.wasm");
    }

    #[test]
    fn test_binary_name_scalar() {
        let caps = Capabilities::with_values(false, false, false, 2);
        assert_eq!(caps.get_binary_name(), "whisper-apr-scalar.wasm");
    }

    #[test]
    fn test_binary_name_threads_without_isolation() {
        // Has threads but not cross-origin isolated - falls back to SIMD sequential
        let mut caps = Capabilities::with_values(true, true, false, 4);
        assert_eq!(caps.get_binary_name(), "whisper-apr-simd-sequential.wasm");

        // Without SIMD, just scalar
        caps.set_simd(false);
        assert_eq!(caps.get_binary_name(), "whisper-apr-scalar.wasm");
    }

    // =========================================================================
    // Thread Count Tests
    // =========================================================================

    #[test]
    fn test_optimal_thread_count_no_threads() {
        let caps = Capabilities::with_values(true, false, false, 8);
        assert_eq!(caps.optimal_thread_count(), 1);
    }

    #[test]
    fn test_optimal_thread_count_single_core() {
        let caps = Capabilities::with_values(true, true, true, 1);
        assert_eq!(caps.optimal_thread_count(), 1);
    }

    #[test]
    fn test_optimal_thread_count_dual_core() {
        let caps = Capabilities::with_values(true, true, true, 2);
        assert_eq!(caps.optimal_thread_count(), 1);
    }

    #[test]
    fn test_optimal_thread_count_quad_core() {
        let caps = Capabilities::with_values(true, true, true, 4);
        assert_eq!(caps.optimal_thread_count(), 3);
    }

    #[test]
    fn test_optimal_thread_count_many_cores() {
        let caps = Capabilities::with_values(true, true, true, 16);
        assert_eq!(caps.optimal_thread_count(), 8); // Capped at 8
    }

    // =========================================================================
    // Model Compatibility Tests
    // =========================================================================

    #[test]
    fn test_can_run_model_unknown_memory() {
        let caps = Capabilities::default();
        assert!(caps.can_run_model("tiny"));
        assert!(caps.can_run_model("large"));
    }

    #[test]
    fn test_can_run_model_limited_memory() {
        let mut caps = Capabilities::default();
        caps.set_memory_mb(500);

        assert!(caps.can_run_model("tiny"));
        assert!(caps.can_run_model("base"));
        assert!(!caps.can_run_model("small"));
        assert!(!caps.can_run_model("large"));
    }

    #[test]
    fn test_can_run_model_ample_memory() {
        let mut caps = Capabilities::default();
        caps.set_memory_mb(5000);

        assert!(caps.can_run_model("tiny"));
        assert!(caps.can_run_model("base"));
        assert!(caps.can_run_model("small"));
        assert!(caps.can_run_model("large"));
    }

    // =========================================================================
    // Performance Tier Tests
    // =========================================================================

    #[test]
    fn test_performance_tier() {
        // Scalar
        let caps = Capabilities::with_values(false, false, false, 1);
        assert_eq!(caps.performance_tier(), 0);

        // Threads only
        let caps = Capabilities::with_values(false, true, true, 4);
        assert_eq!(caps.performance_tier(), 1);

        // SIMD only
        let caps = Capabilities::with_values(true, false, false, 4);
        assert_eq!(caps.performance_tier(), 2);

        // SIMD + Threads
        let caps = Capabilities::with_values(true, true, true, 8);
        assert_eq!(caps.performance_tier(), 3);
    }

    // =========================================================================
    // Description Tests
    // =========================================================================

    #[test]
    fn test_description_scalar() {
        let caps = Capabilities::default();
        assert_eq!(caps.description(), "Scalar (no acceleration)");
    }

    #[test]
    fn test_description_simd_only() {
        let caps = Capabilities::with_values(true, false, false, 4);
        assert_eq!(caps.description(), "SIMD");
    }

    #[test]
    fn test_description_full() {
        let mut caps = Capabilities::with_values(true, true, true, 8);
        caps.set_webgpu(true);
        assert!(caps.description().contains("SIMD"));
        assert!(caps.description().contains("Threads"));
        assert!(caps.description().contains("CrossOriginIsolated"));
        assert!(caps.description().contains("WebGPU"));
    }

    // =========================================================================
    // Execution Mode Tests
    // =========================================================================

    #[test]
    fn test_execution_mode_from_capabilities() {
        let caps = Capabilities::with_values(true, true, true, 8);
        assert_eq!(ExecutionMode::from(&caps), ExecutionMode::SimdThreaded);

        let caps = Capabilities::with_values(true, false, false, 4);
        assert_eq!(ExecutionMode::from(&caps), ExecutionMode::SimdSequential);

        let caps = Capabilities::with_values(false, false, false, 2);
        assert_eq!(ExecutionMode::from(&caps), ExecutionMode::Scalar);
    }

    #[test]
    fn test_execution_mode_rtf_multiplier() {
        assert!((ExecutionMode::SimdThreaded.rtf_multiplier() - 1.0).abs() < f32::EPSILON);
        assert!((ExecutionMode::SimdSequential.rtf_multiplier() - 1.5).abs() < f32::EPSILON);
        assert!((ExecutionMode::Scalar.rtf_multiplier() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_execution_mode_name() {
        assert!(ExecutionMode::SimdThreaded
            .name()
            .contains("High Performance"));
        assert!(ExecutionMode::SimdSequential
            .name()
            .contains("Compatibility"));
        assert!(ExecutionMode::Scalar.name().contains("Fallback"));
    }

    // =========================================================================
    // Cross-Browser Profile Tests (Spec WAPR-QA-002)
    // =========================================================================

    /// Chrome Desktop (2024+): Full support
    #[test]
    fn test_browser_profile_chrome_desktop() {
        let mut caps = Capabilities::new();
        caps.set_simd(true);
        caps.set_threads(true);
        caps.set_cross_origin_isolated(true);
        caps.set_hardware_concurrency(8);
        caps.set_memory_mb(4096);

        assert_eq!(caps.performance_tier(), 3);
        assert_eq!(caps.get_binary_name(), "whisper-apr-simd-threaded.wasm");
        assert!(caps.can_run_model("large"));
        assert_eq!(caps.optimal_thread_count(), 7);
    }

    /// Firefox Desktop (2024+): Full support with COOP/COEP headers
    #[test]
    fn test_browser_profile_firefox_desktop() {
        let mut caps = Capabilities::new();
        caps.set_simd(true);
        caps.set_threads(true);
        caps.set_cross_origin_isolated(true);
        caps.set_hardware_concurrency(8);
        caps.set_memory_mb(4096);

        assert_eq!(caps.performance_tier(), 3);
        assert_eq!(caps.get_binary_name(), "whisper-apr-simd-threaded.wasm");
        assert_eq!(ExecutionMode::from(&caps), ExecutionMode::SimdThreaded);
    }

    /// Safari Desktop (macOS 14+): SIMD support, limited threading
    #[test]
    fn test_browser_profile_safari_desktop() {
        let mut caps = Capabilities::new();
        caps.set_simd(true);
        caps.set_threads(false); // Safari historically more restrictive
        caps.set_cross_origin_isolated(false);
        caps.set_hardware_concurrency(8);
        caps.set_memory_mb(1024); // Safari stricter memory (spec 3.1)

        assert_eq!(caps.performance_tier(), 2);
        assert_eq!(caps.get_binary_name(), "whisper-apr-simd-sequential.wasm");
        assert!(caps.can_run_model("small"));
        assert!(!caps.can_run_model("large")); // Safari memory limit
        assert_eq!(ExecutionMode::from(&caps), ExecutionMode::SimdSequential);
    }

    /// Chrome Mobile (Android): SIMD but limited threads/memory
    #[test]
    fn test_browser_profile_chrome_mobile() {
        let mut caps = Capabilities::new();
        caps.set_simd(true);
        caps.set_threads(true);
        caps.set_cross_origin_isolated(true);
        caps.set_hardware_concurrency(4); // Mobile typically 4-8 cores
        caps.set_memory_mb(512); // Limited mobile memory

        assert!(caps.performance_tier() >= 2);
        assert!(caps.can_run_model("tiny"));
        assert!(caps.can_run_model("base"));
        assert!(!caps.can_run_model("small"));
        assert_eq!(caps.optimal_thread_count(), 3);
    }

    /// Safari Mobile (iOS): More restrictive environment
    #[test]
    fn test_browser_profile_safari_mobile() {
        let mut caps = Capabilities::new();
        caps.set_simd(true);
        caps.set_threads(false);
        caps.set_cross_origin_isolated(false);
        caps.set_hardware_concurrency(6);
        caps.set_memory_mb(256); // iOS very restricted

        assert_eq!(caps.performance_tier(), 2);
        assert_eq!(caps.get_binary_name(), "whisper-apr-simd-sequential.wasm");
        assert!(caps.can_run_model("tiny"));
        assert!(!caps.can_run_model("base")); // Too little memory
        assert_eq!(caps.optimal_thread_count(), 1);
    }

    /// Legacy browser (pre-SIMD): Scalar fallback
    #[test]
    fn test_browser_profile_legacy() {
        let caps = Capabilities::with_values(false, false, false, 2);

        assert_eq!(caps.performance_tier(), 0);
        assert_eq!(caps.get_binary_name(), "whisper-apr-scalar.wasm");
        assert_eq!(ExecutionMode::from(&caps), ExecutionMode::Scalar);
        assert!((caps.rtf_multiplier() - 4.0).abs() < f32::EPSILON);
    }

    /// Embedded WebView (restrictive): Often no SIMD
    #[test]
    fn test_browser_profile_webview() {
        let mut caps = Capabilities::new();
        caps.set_simd(false);
        caps.set_threads(false);
        caps.set_hardware_concurrency(4);
        caps.set_memory_mb(128);

        assert_eq!(caps.performance_tier(), 0);
        assert_eq!(caps.get_binary_name(), "whisper-apr-scalar.wasm");
        assert!(!caps.can_run_model("tiny")); // Insufficient memory
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    /// Threading without COOP/COEP should not enable threaded mode
    #[test]
    fn test_threads_without_isolation() {
        let mut caps = Capabilities::new();
        caps.set_simd(true);
        caps.set_threads(true);
        caps.set_cross_origin_isolated(false);

        // Should fall back to SIMD sequential
        assert_eq!(caps.get_binary_name(), "whisper-apr-simd-sequential.wasm");
        assert_eq!(ExecutionMode::from(&caps), ExecutionMode::SimdSequential);
    }

    /// Memory boundary tests
    #[test]
    fn test_memory_boundary_tiny() {
        let mut caps = Capabilities::new();
        caps.set_memory_mb(199); // Just below tiny requirement
        assert!(!caps.can_run_model("tiny"));

        caps.set_memory_mb(200); // Exactly at tiny requirement
        assert!(caps.can_run_model("tiny"));
    }

    /// Memory boundary tests for larger models
    #[test]
    fn test_memory_boundary_large() {
        let mut caps = Capabilities::new();
        caps.set_memory_mb(3999); // Just below large requirement
        assert!(!caps.can_run_model("large"));

        caps.set_memory_mb(4000); // Exactly at large requirement
        assert!(caps.can_run_model("large"));
    }

    /// Hardware concurrency edge cases
    #[test]
    fn test_hardware_concurrency_zero() {
        let caps = Capabilities::with_values(true, true, true, 0);
        assert_eq!(caps.optimal_thread_count(), 1);
    }

    /// Model variants should have same memory requirements
    #[test]
    fn test_model_variant_equivalence() {
        let mut caps = Capabilities::new();
        caps.set_memory_mb(500);

        // Standard and English-only variants should have same requirements
        assert_eq!(caps.can_run_model("tiny"), caps.can_run_model("tiny.en"));
        assert_eq!(caps.can_run_model("base"), caps.can_run_model("base.en"));
        assert_eq!(caps.can_run_model("small"), caps.can_run_model("small.en"));
    }

    /// RTF scaling across execution modes
    #[test]
    fn test_rtf_scaling() {
        let simd_threaded = Capabilities::with_values(true, true, true, 8);
        let simd_seq = Capabilities::with_values(true, false, false, 4);
        let scalar = Capabilities::with_values(false, false, false, 2);

        // SIMD threaded should be fastest
        assert!(simd_threaded.rtf_multiplier() < simd_seq.rtf_multiplier());
        assert!(simd_seq.rtf_multiplier() < scalar.rtf_multiplier());

        // Scalar should be 4x slower than SIMD threaded
        assert!(
            (scalar.rtf_multiplier() / simd_threaded.rtf_multiplier() - 4.0).abs() < f32::EPSILON
        );
    }

    // =========================================================================
    // WebGPU Future Capability Tests
    // =========================================================================

    #[test]
    fn test_webgpu_capability() {
        let mut caps = Capabilities::new();
        caps.set_webgpu(true);

        assert!(caps.webgpu());
        assert!(caps.description().contains("WebGPU"));
    }

    #[test]
    fn test_full_capability_description() {
        let mut caps = Capabilities::with_values(true, true, true, 8);
        caps.set_webgpu(true);

        let desc = caps.description();
        assert!(desc.contains("SIMD"));
        assert!(desc.contains("Threads"));
        assert!(desc.contains("CrossOriginIsolated"));
        assert!(desc.contains("WebGPU"));
    }
}

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

mod backend;
mod capabilities;
mod detection;
mod limits;
mod selector;

pub use backend::{BackendSelectionWasm, BackendTypeWasm, GpuBackendWasm, SelectionStrategyWasm};
pub use capabilities::GpuCapabilitiesWasm;
pub use detection::{DetectionOptionsWasm, GpuDetectionWasm};
pub use limits::GpuLimitsWasm;
pub use selector::{BackendSelectorWasm, SelectorConfigWasm};

use wasm_bindgen::prelude::*;

// =============================================================================
// Utility Functions
// =============================================================================

/// Get the recommended backend for a given model size
#[wasm_bindgen(js_name = recommendedBackendForModel)]
pub fn recommended_backend_for_model(model_type: &str) -> String {
    let params = match model_type.to_lowercase().as_str() {
        "base" | "base.en" => 74_000_000,
        "small" | "small.en" => 244_000_000,
        "medium" | "medium.en" => 769_000_000,
        "large" | "large-v2" | "large-v3" => 1_550_000_000,
        "large-v3-turbo" => 809_000_000,
        _ => 39_000_000, // tiny and unknown default to smallest
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
    use crate::backend::{BackendSelection, BackendType, SelectionStrategy};
    use crate::gpu::{GpuBackend, GpuLimits};

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

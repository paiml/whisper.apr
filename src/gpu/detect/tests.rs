//! Tests for GPU detection module

use super::*;

#[test]
fn test_detection_result_unavailable() {
    let result = GpuDetectionResult::unavailable();
    assert!(!result.available);
    assert!(!result.suitable_for_inference());
    assert_eq!(result.recommended_backend, GpuBackend::None);
}

#[test]
fn test_detection_result_summary() {
    let result = GpuDetectionResult::unavailable();
    assert!(result.summary().contains("No GPU"));

    let simulated = detect_gpu_simulated(SimulatedGpuConfig::default());
    assert!(simulated.summary().contains("Simulated GPU"));
}

#[test]
fn test_detection_method_display() {
    assert_eq!(DetectionMethod::WgpuNative.to_string(), "wgpu native");
    assert_eq!(DetectionMethod::WebGpuBrowser.to_string(), "WebGPU browser");
    assert_eq!(DetectionMethod::Simulated.to_string(), "simulated");
    assert_eq!(DetectionMethod::NoGpu.to_string(), "none");
}

#[test]
fn test_detection_options_default() {
    let opts = DetectionOptions::default();
    assert!(opts.prefer_high_performance);
    assert!(opts.require_compute);
    assert_eq!(opts.min_vram, 0);
}

#[test]
fn test_detection_options_for_inference() {
    let opts = DetectionOptions::for_inference();
    assert!(opts.prefer_high_performance);
    assert!(opts.require_compute);
    assert!(opts.min_vram > 0);
}

#[test]
fn test_detection_options_builders() {
    let opts = DetectionOptions::default()
        .with_backend(GpuBackend::Metal)
        .with_min_vram(1024 * 1024 * 1024)
        .without_compute_requirement();

    assert_eq!(opts.preferred_backend, Some(GpuBackend::Metal));
    assert_eq!(opts.min_vram, 1024 * 1024 * 1024);
    assert!(!opts.require_compute);
}

#[test]
fn test_detect_gpu_without_feature() {
    let result = detect_gpu(&DetectionOptions::default());
    // Without webgpu feature, should be unavailable
    #[cfg(not(feature = "webgpu"))]
    assert!(!result.available);
    #[cfg(feature = "webgpu")]
    let _ = result; // Silence unused variable warning when feature enabled
}

#[test]
fn test_simulated_gpu_config_default() {
    let config = SimulatedGpuConfig::default();
    assert_eq!(config.name, "Simulated GPU");
    assert_eq!(config.backend, GpuBackend::Vulkan);
    assert!(config.supports_f16);
}

#[test]
fn test_simulated_gpu_config_high_end() {
    let config = SimulatedGpuConfig::high_end_desktop();
    assert!(config.name.contains("RTX"));
    assert_eq!(config.vram_bytes, 24 * 1024 * 1024 * 1024);
}

#[test]
fn test_simulated_gpu_config_apple_silicon() {
    let config = SimulatedGpuConfig::apple_silicon();
    assert!(config.name.contains("Apple"));
    assert_eq!(config.backend, GpuBackend::Metal);
}

#[test]
fn test_simulated_gpu_config_mobile() {
    let config = SimulatedGpuConfig::mobile();
    assert!(config.limits.max_buffer_size < GpuLimits::default().max_buffer_size);
}

#[test]
fn test_simulated_gpu_config_browser() {
    let config = SimulatedGpuConfig::browser_webgpu();
    assert_eq!(config.backend, GpuBackend::BrowserWebGpu);
    assert!(!config.supports_f16);
}

#[test]
fn test_simulated_gpu_config_builders() {
    let config = SimulatedGpuConfig::default()
        .with_name("Custom GPU")
        .with_vram(8 * 1024 * 1024 * 1024)
        .with_backend(GpuBackend::Metal);

    assert_eq!(config.name, "Custom GPU");
    assert_eq!(config.vram_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(config.backend, GpuBackend::Metal);
}

#[test]
fn test_detect_gpu_simulated() {
    let result = detect_gpu_simulated(SimulatedGpuConfig::default());
    assert!(result.available);
    assert_eq!(result.detection_method, DetectionMethod::Simulated);
    assert!(result.suitable_for_inference());
}

#[test]
fn test_gpu_feature_query_default() {
    let query = GpuFeatureQuery::default();
    assert!(!query.compute);
    assert!(!query.f16);
    assert_eq!(query.min_buffer_size, 0);
}

#[test]
fn test_gpu_feature_query_for_inference() {
    let query = GpuFeatureQuery::for_inference();
    assert!(query.compute);
    assert!(query.min_buffer_size > 0);
}

#[test]
fn test_gpu_feature_query_builders() {
    let query = GpuFeatureQuery::default()
        .with_compute()
        .with_f16()
        .with_timestamp_query();

    assert!(query.compute);
    assert!(query.f16);
    assert!(query.timestamp_query);
}

#[test]
fn test_gpu_feature_query_satisfied_by() {
    let result = detect_gpu_simulated(SimulatedGpuConfig::default());
    let query = GpuFeatureQuery::for_inference();

    assert!(query.satisfied_by(&result.capabilities));
}

#[test]
fn test_gpu_feature_query_unsatisfied() {
    let result = detect_gpu_simulated(SimulatedGpuConfig::default());
    let query = GpuFeatureQuery::default().with_timestamp_query();

    // Default config has timestamp_query = true
    assert!(query.satisfied_by(&result.capabilities));

    // Test with config that doesn't have timestamp query
    let no_ts = SimulatedGpuConfig::mobile();
    let result2 = detect_gpu_simulated(no_ts);
    let reqs = query.unsatisfied_requirements(&result2.capabilities);
    assert!(reqs.iter().any(|r| r.contains("timestamp")));
}

#[test]
fn test_recommend_backend() {
    let backend = recommend_backend();
    // Should return a valid backend for the current platform
    #[cfg(target_os = "macos")]
    assert_eq!(backend, GpuBackend::Metal);
    #[cfg(target_os = "windows")]
    assert_eq!(backend, GpuBackend::Dx12);
    #[cfg(target_os = "linux")]
    assert_eq!(backend, GpuBackend::Vulkan);
}

#[test]
fn test_should_use_gpu_unavailable() {
    let caps = GpuCapabilities::default(); // No GPU
    let rec = should_use_gpu(&caps, 100_000);
    assert!(!rec.use_gpu());
}

#[test]
fn test_should_use_gpu_small_workload() {
    let result = detect_gpu_simulated(SimulatedGpuConfig::default());
    let rec = should_use_gpu(&result.capabilities, 1_000);
    assert!(!rec.use_gpu()); // Small workload, CPU preferred
}

#[test]
fn test_should_use_gpu_large_workload() {
    let result = detect_gpu_simulated(SimulatedGpuConfig::default());
    let rec = should_use_gpu(&result.capabilities, 500_000);
    assert!(rec.use_gpu());
    assert!(rec.speedup().is_some());
}

#[test]
fn test_gpu_recommendation_use_gpu() {
    assert!(!GpuRecommendation::CpuOnly {
        reason: "test".to_string()
    }
    .use_gpu());
    assert!(!GpuRecommendation::CpuPreferred {
        reason: "test".to_string()
    }
    .use_gpu());
    assert!(GpuRecommendation::GpuRecommended {
        speedup_estimate: 5.0
    }
    .use_gpu());
    assert!(GpuRecommendation::GpuStronglyRecommended {
        speedup_estimate: 10.0
    }
    .use_gpu());
}

#[test]
fn test_gpu_recommendation_speedup() {
    assert!(GpuRecommendation::CpuOnly {
        reason: "test".to_string()
    }
    .speedup()
    .is_none());
    assert_eq!(
        GpuRecommendation::GpuRecommended {
            speedup_estimate: 5.0
        }
        .speedup(),
        Some(5.0)
    );
}

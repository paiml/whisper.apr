//! WebGPU compute backend for accelerated inference (WAPR-120 to WAPR-143)
//!
//! This module provides GPU-accelerated operations using WebGPU, enabling
//! faster inference on large models when GPU hardware is available.
//!
//! # Features
//!
//! - Automatic device detection and initialization
//! - GPU buffer management with host-device transfers
//! - Compute shader pipeline for matrix operations
//! - Backend selection (SIMD vs GPU)
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::gpu::{GpuDevice, GpuCapabilities};
//!
//! // Check if WebGPU is available
//! if GpuCapabilities::is_available() {
//!     let device = GpuDevice::new().await?;
//!     let caps = device.capabilities();
//!     println!("GPU: {} ({})", caps.name, caps.backend);
//! }
//! ```

mod buffer;
mod capabilities;
mod detect;
mod device;
mod error;
mod pipeline;

pub use buffer::{GpuBuffer, GpuBufferUsage};
pub use capabilities::{GpuBackend, GpuCapabilities, GpuLimits};
pub use detect::{
    detect_gpu, detect_gpu_simulated, recommend_backend, should_use_gpu, DetectionMethod,
    DetectionOptions, GpuDetectionResult, GpuFeatureQuery, GpuRecommendation, SimulatedGpuConfig,
};
pub use device::{GpuDevice, GpuDeviceConfig};
pub use error::{GpuError, GpuResult};
pub use pipeline::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBinding, ComputeDispatch, ComputePipeline,
    ComputePipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
    ShaderSourceType, WorkgroupDimensions,
};

/// Default workgroup size for compute shaders (optimized for most GPUs)
pub const DEFAULT_WORKGROUP_SIZE: u32 = 256;

/// Maximum buffer size for single allocation (256 MB)
pub const MAX_BUFFER_SIZE: u64 = 256 * 1024 * 1024;

/// Minimum buffer alignment (256 bytes for WebGPU)
pub const BUFFER_ALIGNMENT: u64 = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_WORKGROUP_SIZE, 256);
        assert_eq!(MAX_BUFFER_SIZE, 256 * 1024 * 1024);
        assert_eq!(BUFFER_ALIGNMENT, 256);
    }

    #[test]
    fn test_module_exports() {
        // Verify all public types are accessible
        let _ = std::any::type_name::<GpuBuffer>();
        let _ = std::any::type_name::<GpuBufferUsage>();
        let _ = std::any::type_name::<GpuCapabilities>();
        let _ = std::any::type_name::<GpuDevice>();
        let _ = std::any::type_name::<GpuDeviceConfig>();
        let _ = std::any::type_name::<GpuError>();
    }
}

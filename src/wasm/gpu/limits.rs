//! WASM GPU limits bindings

use wasm_bindgen::prelude::*;

use crate::gpu::GpuLimits;

/// WASM-friendly GPU limits
#[wasm_bindgen]
#[derive(Debug, Clone)]
#[allow(clippy::struct_field_names)]
pub struct GpuLimitsWasm {
    pub(super) max_buffer_size: u64,
    pub(super) max_storage_buffer_binding_size: u32,
    pub(super) max_uniform_buffer_binding_size: u32,
    pub(super) max_compute_workgroup_size_x: u32,
    pub(super) max_compute_workgroup_size_y: u32,
    pub(super) max_compute_workgroup_size_z: u32,
    pub(super) max_compute_invocations_per_workgroup: u32,
    pub(super) max_compute_workgroups_per_dimension: u32,
}

#[wasm_bindgen]
impl GpuLimitsWasm {
    /// Get maximum buffer size in bytes
    #[wasm_bindgen(getter, js_name = maxBufferSize)]
    pub fn max_buffer_size(&self) -> u64 {
        self.max_buffer_size
    }

    /// Get maximum storage buffer binding size
    #[wasm_bindgen(getter, js_name = maxStorageBufferBindingSize)]
    pub fn max_storage_buffer_binding_size(&self) -> u32 {
        self.max_storage_buffer_binding_size
    }

    /// Get maximum uniform buffer binding size
    #[wasm_bindgen(getter, js_name = maxUniformBufferBindingSize)]
    pub fn max_uniform_buffer_binding_size(&self) -> u32 {
        self.max_uniform_buffer_binding_size
    }

    /// Get maximum compute workgroup size X
    #[wasm_bindgen(getter, js_name = maxComputeWorkgroupSizeX)]
    pub fn max_compute_workgroup_size_x(&self) -> u32 {
        self.max_compute_workgroup_size_x
    }

    /// Get maximum compute workgroup size Y
    #[wasm_bindgen(getter, js_name = maxComputeWorkgroupSizeY)]
    pub fn max_compute_workgroup_size_y(&self) -> u32 {
        self.max_compute_workgroup_size_y
    }

    /// Get maximum compute workgroup size Z
    #[wasm_bindgen(getter, js_name = maxComputeWorkgroupSizeZ)]
    pub fn max_compute_workgroup_size_z(&self) -> u32 {
        self.max_compute_workgroup_size_z
    }

    /// Get maximum compute invocations per workgroup
    #[wasm_bindgen(getter, js_name = maxComputeInvocationsPerWorkgroup)]
    pub fn max_compute_invocations_per_workgroup(&self) -> u32 {
        self.max_compute_invocations_per_workgroup
    }

    /// Get maximum compute workgroups per dimension
    #[wasm_bindgen(getter, js_name = maxComputeWorkgroupsPerDimension)]
    pub fn max_compute_workgroups_per_dimension(&self) -> u32 {
        self.max_compute_workgroups_per_dimension
    }

    /// Get maximum buffer size in MB
    #[wasm_bindgen(js_name = maxBufferSizeMb)]
    pub fn max_buffer_size_mb(&self) -> f32 {
        self.max_buffer_size as f32 / (1024.0 * 1024.0)
    }
}

impl From<GpuLimits> for GpuLimitsWasm {
    fn from(limits: GpuLimits) -> Self {
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }
}

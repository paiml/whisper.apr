//! GPU softmax operation (WAPR-131)
//!
//! Provides numerically stable softmax computation using compute shaders.
//! Optimized for attention score normalization.

use crate::gpu::error::{GpuError, GpuResult};

/// Softmax computation dimension
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxDimension {
    /// Apply softmax along rows (last dimension)
    Row,
    /// Apply softmax along columns
    Column,
    /// Apply softmax to entire tensor (flattened)
    All,
}

impl Default for SoftmaxDimension {
    fn default() -> Self {
        Self::Row
    }
}

impl SoftmaxDimension {
    /// Get the reduction axis description
    #[must_use]
    pub fn axis_description(&self) -> &str {
        match self {
            Self::Row => "last",
            Self::Column => "first",
            Self::All => "all",
        }
    }
}

/// Softmax configuration
#[derive(Debug, Clone)]
pub struct SoftmaxConfig {
    /// Dimension to apply softmax
    pub dimension: SoftmaxDimension,
    /// Number of rows
    pub rows: u32,
    /// Number of columns
    pub cols: u32,
    /// Temperature scaling (divide logits by this before softmax)
    pub temperature: f32,
    /// Whether to apply log-softmax instead
    pub log_softmax: bool,
    /// Workgroup size for reduction
    pub workgroup_size: u32,
    /// Label for debugging
    pub label: Option<String>,
}

impl Default for SoftmaxConfig {
    fn default() -> Self {
        Self {
            dimension: SoftmaxDimension::default(),
            rows: 1,
            cols: 1,
            temperature: 1.0,
            log_softmax: false,
            workgroup_size: 256,
            label: None,
        }
    }
}

impl SoftmaxConfig {
    /// Create softmax config for attention scores
    #[must_use]
    pub fn attention(seq_len: u32, num_heads: u32) -> Self {
        Self {
            dimension: SoftmaxDimension::Row,
            rows: num_heads,
            cols: seq_len,
            temperature: 1.0,
            log_softmax: false,
            workgroup_size: 256,
            label: Some("attention_softmax".to_string()),
        }
    }

    /// Create softmax config with custom dimensions
    #[must_use]
    pub fn new(rows: u32, cols: u32) -> Self {
        Self {
            rows,
            cols,
            ..Default::default()
        }
    }

    /// Set temperature scaling
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Enable log-softmax
    #[must_use]
    pub fn log_softmax(mut self) -> Self {
        self.log_softmax = true;
        self
    }

    /// Set dimension
    #[must_use]
    pub fn along(mut self, dimension: SoftmaxDimension) -> Self {
        self.dimension = dimension;
        self
    }

    /// Set workgroup size
    #[must_use]
    pub fn with_workgroup_size(mut self, size: u32) -> Self {
        self.workgroup_size = size;
        self
    }

    /// Set label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> GpuResult<()> {
        if self.rows == 0 || self.cols == 0 {
            return Err(GpuError::compute("Softmax dimensions cannot be zero"));
        }
        if self.temperature <= 0.0 {
            return Err(GpuError::compute("Temperature must be positive"));
        }
        if !self.workgroup_size.is_power_of_two() {
            return Err(GpuError::compute("Workgroup size must be power of two"));
        }
        Ok(())
    }

    /// Get total elements
    #[must_use]
    pub fn total_elements(&self) -> usize {
        (self.rows as usize) * (self.cols as usize)
    }

    /// Get reduction dimension size
    #[must_use]
    pub fn reduction_size(&self) -> u32 {
        match self.dimension {
            SoftmaxDimension::Row => self.cols,
            SoftmaxDimension::Column => self.rows,
            SoftmaxDimension::All => self.rows * self.cols,
        }
    }

    /// Get number of independent softmax operations
    #[must_use]
    pub fn num_reductions(&self) -> u32 {
        match self.dimension {
            SoftmaxDimension::Row => self.rows,
            SoftmaxDimension::Column => self.cols,
            SoftmaxDimension::All => 1,
        }
    }
}

/// GPU softmax operation
#[derive(Debug)]
pub struct GpuSoftmax {
    /// Operation ID
    id: u64,
    /// Configuration
    config: SoftmaxConfig,
    /// Whether executed
    executed: bool,
}

impl GpuSoftmax {
    /// Create a new softmax operation
    #[allow(clippy::items_after_statements)]
    pub fn new(config: SoftmaxConfig) -> GpuResult<Self> {
        config.validate()?;

        static OP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Ok(Self {
            id: OP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            config,
            executed: false,
        })
    }

    /// Create softmax for attention scores
    pub fn attention(seq_len: u32, num_heads: u32) -> GpuResult<Self> {
        Self::new(SoftmaxConfig::attention(seq_len, num_heads))
    }

    /// Get operation ID
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &SoftmaxConfig {
        &self.config
    }

    /// Check if executed
    #[must_use]
    pub fn is_executed(&self) -> bool {
        self.executed
    }

    /// Get memory requirement in bytes
    #[must_use]
    pub fn memory_requirement(&self) -> usize {
        // Input and output are same size
        self.config.total_elements() * 4 * 2
    }

    /// Calculate workgroups for dispatch
    #[must_use]
    pub fn workgroups(&self) -> (u32, u32, u32) {
        let num_reductions = self.config.num_reductions();
        (num_reductions, 1, 1)
    }

    /// Generate WGSL shader for this operation
    #[must_use]
    pub fn generate_shader(&self) -> String {
        let reduction_size = self.config.reduction_size();
        let workgroup_size = self.config.workgroup_size.min(reduction_size);
        let temperature = self.config.temperature;
        let is_log = self.config.log_softmax;

        format!(
            r"// Softmax shader ({log}softmax along {dim})
// Rows: {rows}, Cols: {cols}
// Temperature: {temp}

struct Params {{
    rows: u32,
    cols: u32,
    temperature: f32,
}}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> partial_max: array<f32, {wg_size}>;
var<workgroup> partial_sum: array<f32, {wg_size}>;

@compute @workgroup_size({wg_size}, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {{
    let row = workgroup_id.x;
    let tid = local_id.x;
    let reduction_size = params.cols;
    let row_offset = row * reduction_size;

    // Phase 1: Find max value for numerical stability
    var local_max: f32 = -1e38;
    for (var i = tid; i < reduction_size; i = i + {wg_size}u) {{
        let val = input[row_offset + i] / params.temperature;
        local_max = max(local_max, val);
    }}
    partial_max[tid] = local_max;
    workgroupBarrier();

    // Reduce to find global max
    for (var stride = {wg_size}u / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            partial_max[tid] = max(partial_max[tid], partial_max[tid + stride]);
        }}
        workgroupBarrier();
    }}

    if (tid == 0u) {{
        shared_max = partial_max[0];
    }}
    workgroupBarrier();

    // Phase 2: Compute exp(x - max) and sum
    var local_sum: f32 = 0.0;
    for (var i = tid; i < reduction_size; i = i + {wg_size}u) {{
        let val = input[row_offset + i] / params.temperature;
        local_sum = local_sum + exp(val - shared_max);
    }}
    partial_sum[tid] = local_sum;
    workgroupBarrier();

    // Reduce to find sum
    for (var stride = {wg_size}u / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            partial_sum[tid] = partial_sum[tid] + partial_sum[tid + stride];
        }}
        workgroupBarrier();
    }}

    if (tid == 0u) {{
        shared_sum = partial_sum[0];
    }}
    workgroupBarrier();

    // Phase 3: Normalize
    for (var i = tid; i < reduction_size; i = i + {wg_size}u) {{
        let val = input[row_offset + i] / params.temperature;
        {output_expr}
    }}
}}
",
            log = if is_log { "log-" } else { "" },
            dim = self.config.dimension.axis_description(),
            rows = self.config.rows,
            cols = self.config.cols,
            temp = temperature,
            wg_size = workgroup_size,
            output_expr = if is_log {
                "output[row_offset + i] = val - shared_max - log(shared_sum);"
            } else {
                "output[row_offset + i] = exp(val - shared_max) / shared_sum;"
            },
        )
    }
}

/// WGSL shader source for simple row-wise softmax
#[allow(dead_code)]
pub const SOFTMAX_SHADER_SIMPLE: &str = r"
struct Params {
    rows: u32,
    cols: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.rows) {
        return;
    }

    let row_offset = row * params.cols;

    // Find max for numerical stability
    var max_val: f32 = -1e38;
    for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
        max_val = max(max_val, input[row_offset + i]);
    }

    // Compute exp and sum
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
        let exp_val = exp(input[row_offset + i] - max_val);
        output[row_offset + i] = exp_val;
        sum = sum + exp_val;
    }

    // Normalize
    for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
        output[row_offset + i] = output[row_offset + i] / sum;
    }
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_dimension_default() {
        assert_eq!(SoftmaxDimension::default(), SoftmaxDimension::Row);
    }

    #[test]
    fn test_softmax_dimension_axis_description() {
        assert_eq!(SoftmaxDimension::Row.axis_description(), "last");
        assert_eq!(SoftmaxDimension::Column.axis_description(), "first");
        assert_eq!(SoftmaxDimension::All.axis_description(), "all");
    }

    #[test]
    fn test_softmax_config_default() {
        let config = SoftmaxConfig::default();
        assert_eq!(config.dimension, SoftmaxDimension::Row);
        assert_eq!(config.temperature, 1.0);
        assert!(!config.log_softmax);
    }

    #[test]
    fn test_softmax_config_attention() {
        let config = SoftmaxConfig::attention(512, 8);
        assert_eq!(config.rows, 8);
        assert_eq!(config.cols, 512);
        assert_eq!(config.dimension, SoftmaxDimension::Row);
    }

    #[test]
    fn test_softmax_config_builders() {
        let config = SoftmaxConfig::new(16, 64)
            .with_temperature(0.5)
            .log_softmax()
            .along(SoftmaxDimension::Column)
            .with_workgroup_size(128)
            .with_label("test_softmax");

        assert_eq!(config.rows, 16);
        assert_eq!(config.cols, 64);
        assert_eq!(config.temperature, 0.5);
        assert!(config.log_softmax);
        assert_eq!(config.dimension, SoftmaxDimension::Column);
        assert_eq!(config.workgroup_size, 128);
        assert_eq!(config.label, Some("test_softmax".to_string()));
    }

    #[test]
    fn test_softmax_config_validate() {
        assert!(SoftmaxConfig::new(16, 64).validate().is_ok());
        assert!(SoftmaxConfig::new(0, 64).validate().is_err());
        assert!(SoftmaxConfig::new(16, 0).validate().is_err());
        assert!(SoftmaxConfig::new(16, 64).with_temperature(0.0).validate().is_err());
        assert!(SoftmaxConfig::new(16, 64).with_workgroup_size(100).validate().is_err());
    }

    #[test]
    fn test_softmax_config_total_elements() {
        let config = SoftmaxConfig::new(16, 64);
        assert_eq!(config.total_elements(), 16 * 64);
    }

    #[test]
    fn test_softmax_config_reduction_size() {
        let config = SoftmaxConfig::new(16, 64);

        assert_eq!(config.reduction_size(), 64); // Row
        assert_eq!(config.clone().along(SoftmaxDimension::Column).reduction_size(), 16);
        assert_eq!(config.clone().along(SoftmaxDimension::All).reduction_size(), 16 * 64);
    }

    #[test]
    fn test_softmax_config_num_reductions() {
        let config = SoftmaxConfig::new(16, 64);

        assert_eq!(config.num_reductions(), 16); // Row: one per row
        assert_eq!(config.clone().along(SoftmaxDimension::Column).num_reductions(), 64);
        assert_eq!(config.clone().along(SoftmaxDimension::All).num_reductions(), 1);
    }

    #[test]
    fn test_gpu_softmax_new() {
        let softmax = GpuSoftmax::new(SoftmaxConfig::new(16, 64))
            .expect("Should create softmax");
        assert!(softmax.id() > 0);
        assert!(!softmax.is_executed());
    }

    #[test]
    fn test_gpu_softmax_attention() {
        let softmax = GpuSoftmax::attention(512, 8).expect("Should create");
        assert_eq!(softmax.config().rows, 8);
        assert_eq!(softmax.config().cols, 512);
    }

    #[test]
    fn test_gpu_softmax_memory_requirement() {
        let softmax = GpuSoftmax::new(SoftmaxConfig::new(16, 64))
            .expect("Should create");
        // 16 * 64 * 4 bytes * 2 (input + output)
        assert_eq!(softmax.memory_requirement(), 16 * 64 * 4 * 2);
    }

    #[test]
    fn test_gpu_softmax_workgroups() {
        let softmax = GpuSoftmax::new(SoftmaxConfig::new(16, 64))
            .expect("Should create");
        let (x, y, z) = softmax.workgroups();
        assert_eq!(x, 16); // One per row
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_gpu_softmax_generate_shader() {
        let softmax = GpuSoftmax::new(SoftmaxConfig::new(16, 64))
            .expect("Should create");
        let shader = softmax.generate_shader();

        assert!(shader.contains("@compute"));
        assert!(shader.contains("shared_max"));
        assert!(shader.contains("shared_sum"));
        assert!(shader.contains("workgroupBarrier"));
    }

    #[test]
    fn test_gpu_softmax_log_softmax_shader() {
        let softmax = GpuSoftmax::new(
            SoftmaxConfig::new(16, 64).log_softmax()
        ).expect("Should create");
        let shader = softmax.generate_shader();

        assert!(shader.contains("log-softmax"));
        assert!(shader.contains("log(shared_sum)"));
    }

    #[test]
    fn test_softmax_shader_simple() {
        assert!(SOFTMAX_SHADER_SIMPLE.contains("@compute"));
        assert!(SOFTMAX_SHADER_SIMPLE.contains("max_val"));
        assert!(SOFTMAX_SHADER_SIMPLE.contains("exp"));
    }

    #[test]
    fn test_gpu_softmax_unique_ids() {
        let s1 = GpuSoftmax::new(SoftmaxConfig::new(16, 64)).expect("s1");
        let s2 = GpuSoftmax::new(SoftmaxConfig::new(16, 64)).expect("s2");
        assert_ne!(s1.id(), s2.id());
    }
}

//! GPU layer normalization (WAPR-132)
//!
//! Provides layer normalization computation using compute shaders.
//! Essential for transformer layer outputs.

use crate::gpu::error::{GpuError, GpuResult};

/// Layer normalization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerNormMode {
    /// Standard layer norm: (x - mean) / sqrt(var + eps)
    Standard,
    /// RMS norm: x / sqrt(mean(x^2) + eps) (no centering)
    RmsNorm,
    /// Pre-norm variant (normalize before, not after)
    PreNorm,
}

impl Default for LayerNormMode {
    fn default() -> Self {
        Self::Standard
    }
}

impl LayerNormMode {
    /// Get description of the normalization mode
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Standard => "layer normalization",
            Self::RmsNorm => "RMS normalization",
            Self::PreNorm => "pre-layer normalization",
        }
    }

    /// Check if this mode computes mean (for centering)
    #[must_use]
    pub fn computes_mean(&self) -> bool {
        matches!(self, Self::Standard | Self::PreNorm)
    }
}

/// Layer normalization configuration
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    /// Normalization mode
    pub mode: LayerNormMode,
    /// Batch size
    pub batch_size: u32,
    /// Hidden dimension (normalization dimension)
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Whether to apply learned scale (gamma)
    pub use_scale: bool,
    /// Whether to apply learned bias (beta)
    pub use_bias: bool,
    /// Workgroup size
    pub workgroup_size: u32,
    /// Label for debugging
    pub label: Option<String>,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            mode: LayerNormMode::default(),
            batch_size: 1,
            hidden_size: 512,
            epsilon: 1e-5,
            use_scale: true,
            use_bias: true,
            workgroup_size: 256,
            label: None,
        }
    }
}

impl LayerNormConfig {
    /// Create config for transformer hidden states
    #[must_use]
    pub fn transformer(batch_size: u32, hidden_size: u32) -> Self {
        Self {
            batch_size,
            hidden_size,
            ..Default::default()
        }
    }

    /// Create RMS norm config (used in some models like LLaMA)
    #[must_use]
    pub fn rms_norm(batch_size: u32, hidden_size: u32) -> Self {
        Self {
            mode: LayerNormMode::RmsNorm,
            batch_size,
            hidden_size,
            use_bias: false, // RMS norm typically doesn't use bias
            ..Default::default()
        }
    }

    /// Create config with custom dimensions
    #[must_use]
    pub fn new(batch_size: u32, hidden_size: u32) -> Self {
        Self {
            batch_size,
            hidden_size,
            ..Default::default()
        }
    }

    /// Set normalization mode
    #[must_use]
    pub fn with_mode(mut self, mode: LayerNormMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set epsilon
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Disable scale parameter
    #[must_use]
    pub fn without_scale(mut self) -> Self {
        self.use_scale = false;
        self
    }

    /// Disable bias parameter
    #[must_use]
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
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
        if self.batch_size == 0 || self.hidden_size == 0 {
            return Err(GpuError::compute("Layer norm dimensions cannot be zero"));
        }
        if self.epsilon <= 0.0 {
            return Err(GpuError::compute("Epsilon must be positive"));
        }
        if !self.workgroup_size.is_power_of_two() {
            return Err(GpuError::compute("Workgroup size must be power of two"));
        }
        Ok(())
    }

    /// Get total elements
    #[must_use]
    pub fn total_elements(&self) -> usize {
        (self.batch_size as usize) * (self.hidden_size as usize)
    }

    /// Get number of normalization groups (one per batch item)
    #[must_use]
    pub fn num_groups(&self) -> u32 {
        self.batch_size
    }

    /// Get parameter count (scale + bias if used)
    #[must_use]
    pub fn param_count(&self) -> usize {
        let mut count = 0;
        if self.use_scale {
            count += self.hidden_size as usize;
        }
        if self.use_bias {
            count += self.hidden_size as usize;
        }
        count
    }
}

/// GPU layer normalization operation
#[derive(Debug)]
pub struct GpuLayerNorm {
    /// Operation ID
    id: u64,
    /// Configuration
    config: LayerNormConfig,
    /// Whether executed
    executed: bool,
}

impl GpuLayerNorm {
    /// Create a new layer norm operation
    #[allow(clippy::items_after_statements)]
    pub fn new(config: LayerNormConfig) -> GpuResult<Self> {
        config.validate()?;

        static OP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Ok(Self {
            id: OP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            config,
            executed: false,
        })
    }

    /// Create for transformer hidden states
    pub fn transformer(batch_size: u32, hidden_size: u32) -> GpuResult<Self> {
        Self::new(LayerNormConfig::transformer(batch_size, hidden_size))
    }

    /// Create RMS norm
    pub fn rms_norm(batch_size: u32, hidden_size: u32) -> GpuResult<Self> {
        Self::new(LayerNormConfig::rms_norm(batch_size, hidden_size))
    }

    /// Get operation ID
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &LayerNormConfig {
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
        let input_output = self.config.total_elements() * 4 * 2;
        let params = self.config.param_count() * 4;
        input_output + params
    }

    /// Calculate workgroups for dispatch
    #[must_use]
    pub fn workgroups(&self) -> (u32, u32, u32) {
        (self.config.batch_size, 1, 1)
    }

    /// Generate WGSL shader for this operation
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn generate_shader(&self) -> String {
        let hidden_size = self.config.hidden_size;
        let workgroup_size = self.config.workgroup_size.min(hidden_size);
        let epsilon = self.config.epsilon;
        let is_rms = matches!(self.config.mode, LayerNormMode::RmsNorm);

        let compute_variance = if is_rms {
            r"
    // Compute mean of squares (RMS norm)
    var local_sq_sum: f32 = 0.0;
    for (var i = tid; i < hidden_size; i = i + WORKGROUP_SIZE) {
        let val = input[row_offset + i];
        local_sq_sum = local_sq_sum + val * val;
    }
    partial_sum[tid] = local_sq_sum;
    workgroupBarrier();

    // Reduce
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            partial_sum[tid] = partial_sum[tid] + partial_sum[tid + stride];
        }
        workgroupBarrier();
    }

    let variance = partial_sum[0] / f32(hidden_size);
    let inv_std = 1.0 / sqrt(variance + params.epsilon);
    let mean: f32 = 0.0;  // RMS norm doesn't center"
        } else {
            r"
    // Compute mean
    var local_sum: f32 = 0.0;
    for (var i = tid; i < hidden_size; i = i + WORKGROUP_SIZE) {
        local_sum = local_sum + input[row_offset + i];
    }
    partial_sum[tid] = local_sum;
    workgroupBarrier();

    // Reduce for mean
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            partial_sum[tid] = partial_sum[tid] + partial_sum[tid + stride];
        }
        workgroupBarrier();
    }

    let mean = partial_sum[0] / f32(hidden_size);
    workgroupBarrier();

    // Compute variance
    var local_var: f32 = 0.0;
    for (var i = tid; i < hidden_size; i = i + WORKGROUP_SIZE) {
        let diff = input[row_offset + i] - mean;
        local_var = local_var + diff * diff;
    }
    partial_sum[tid] = local_var;
    workgroupBarrier();

    // Reduce for variance
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            partial_sum[tid] = partial_sum[tid] + partial_sum[tid + stride];
        }
        workgroupBarrier();
    }

    let variance = partial_sum[0] / f32(hidden_size);
    let inv_std = 1.0 / sqrt(variance + params.epsilon);"
        };

        let normalize_expr = if self.config.use_scale && self.config.use_bias {
            "output[row_offset + i] = gamma[i] * normalized + beta[i];"
        } else if self.config.use_scale {
            "output[row_offset + i] = gamma[i] * normalized;"
        } else if self.config.use_bias {
            "output[row_offset + i] = normalized + beta[i];"
        } else {
            "output[row_offset + i] = normalized;"
        };

        format!(
            r"// {mode} shader
// Batch: {batch}, Hidden: {hidden}
// Epsilon: {eps}

const WORKGROUP_SIZE: u32 = {wg_size}u;

struct Params {{
    batch_size: u32,
    hidden_size: u32,
    epsilon: f32,
}}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;

var<workgroup> partial_sum: array<f32, {wg_size}>;

@compute @workgroup_size({wg_size}, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {{
    let row = workgroup_id.x;
    let tid = local_id.x;
    let hidden_size = params.hidden_size;
    let row_offset = row * hidden_size;

    {compute_variance}

    // Normalize and apply scale/bias
    for (var i = tid; i < hidden_size; i = i + WORKGROUP_SIZE) {{
        let normalized = (input[row_offset + i] - mean) * inv_std;
        {normalize_expr}
    }}
}}
",
            mode = self.config.mode.description(),
            batch = self.config.batch_size,
            hidden = hidden_size,
            eps = epsilon,
            wg_size = workgroup_size,
            compute_variance = compute_variance,
            normalize_expr = normalize_expr,
        )
    }
}

/// WGSL shader source for simple layer normalization
#[allow(dead_code)]
pub const LAYERNORM_SHADER_SIMPLE: &str = r"
struct Params {
    batch_size: u32,
    hidden_size: u32,
    epsilon: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.batch_size) {
        return;
    }

    let row_offset = row * params.hidden_size;

    // Compute mean
    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
        mean = mean + input[row_offset + i];
    }
    mean = mean / f32(params.hidden_size);

    // Compute variance
    var variance: f32 = 0.0;
    for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
        let diff = input[row_offset + i] - mean;
        variance = variance + diff * diff;
    }
    variance = variance / f32(params.hidden_size);

    // Normalize
    let inv_std = 1.0 / sqrt(variance + params.epsilon);
    for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
        let normalized = (input[row_offset + i] - mean) * inv_std;
        output[row_offset + i] = gamma[i] * normalized + beta[i];
    }
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_mode_default() {
        assert_eq!(LayerNormMode::default(), LayerNormMode::Standard);
    }

    #[test]
    fn test_layer_norm_mode_description() {
        assert!(LayerNormMode::Standard.description().contains("layer"));
        assert!(LayerNormMode::RmsNorm.description().contains("RMS"));
        assert!(LayerNormMode::PreNorm.description().contains("pre"));
    }

    #[test]
    fn test_layer_norm_mode_computes_mean() {
        assert!(LayerNormMode::Standard.computes_mean());
        assert!(!LayerNormMode::RmsNorm.computes_mean());
        assert!(LayerNormMode::PreNorm.computes_mean());
    }

    #[test]
    fn test_layer_norm_config_default() {
        let config = LayerNormConfig::default();
        assert_eq!(config.mode, LayerNormMode::Standard);
        assert_eq!(config.epsilon, 1e-5);
        assert!(config.use_scale);
        assert!(config.use_bias);
    }

    #[test]
    fn test_layer_norm_config_transformer() {
        let config = LayerNormConfig::transformer(32, 768);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.hidden_size, 768);
    }

    #[test]
    fn test_layer_norm_config_rms_norm() {
        let config = LayerNormConfig::rms_norm(32, 768);
        assert_eq!(config.mode, LayerNormMode::RmsNorm);
        assert!(!config.use_bias); // RMS norm typically no bias
    }

    #[test]
    fn test_layer_norm_config_builders() {
        let config = LayerNormConfig::new(16, 512)
            .with_mode(LayerNormMode::RmsNorm)
            .with_epsilon(1e-6)
            .without_scale()
            .without_bias()
            .with_workgroup_size(128)
            .with_label("test_layernorm");

        assert_eq!(config.batch_size, 16);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.mode, LayerNormMode::RmsNorm);
        assert_eq!(config.epsilon, 1e-6);
        assert!(!config.use_scale);
        assert!(!config.use_bias);
        assert_eq!(config.workgroup_size, 128);
    }

    #[test]
    fn test_layer_norm_config_validate() {
        assert!(LayerNormConfig::new(16, 512).validate().is_ok());
        assert!(LayerNormConfig::new(0, 512).validate().is_err());
        assert!(LayerNormConfig::new(16, 0).validate().is_err());
        assert!(LayerNormConfig::new(16, 512)
            .with_epsilon(0.0)
            .validate()
            .is_err());
        assert!(LayerNormConfig::new(16, 512)
            .with_workgroup_size(100)
            .validate()
            .is_err());
    }

    #[test]
    fn test_layer_norm_config_total_elements() {
        let config = LayerNormConfig::new(16, 512);
        assert_eq!(config.total_elements(), 16 * 512);
    }

    #[test]
    fn test_layer_norm_config_param_count() {
        let config = LayerNormConfig::new(16, 512);
        assert_eq!(config.param_count(), 512 * 2); // gamma + beta

        let no_bias = config.clone().without_bias();
        assert_eq!(no_bias.param_count(), 512);

        let no_params = config.without_scale().without_bias();
        assert_eq!(no_params.param_count(), 0);
    }

    #[test]
    fn test_gpu_layer_norm_new() {
        let ln =
            GpuLayerNorm::new(LayerNormConfig::new(16, 512)).expect("Should create layer norm");
        assert!(ln.id() > 0);
        assert!(!ln.is_executed());
    }

    #[test]
    fn test_gpu_layer_norm_transformer() {
        let ln = GpuLayerNorm::transformer(32, 768).expect("Should create");
        assert_eq!(ln.config().batch_size, 32);
        assert_eq!(ln.config().hidden_size, 768);
    }

    #[test]
    fn test_gpu_layer_norm_rms_norm() {
        let ln = GpuLayerNorm::rms_norm(32, 768).expect("Should create");
        assert_eq!(ln.config().mode, LayerNormMode::RmsNorm);
    }

    #[test]
    fn test_gpu_layer_norm_memory_requirement() {
        let ln = GpuLayerNorm::new(LayerNormConfig::new(16, 512)).expect("Should create");
        // Input: 16*512*4, Output: 16*512*4, Params: 512*2*4
        let expected = (16 * 512 * 4 * 2) + (512 * 2 * 4);
        assert_eq!(ln.memory_requirement(), expected);
    }

    #[test]
    fn test_gpu_layer_norm_workgroups() {
        let ln = GpuLayerNorm::new(LayerNormConfig::new(16, 512)).expect("Should create");
        let (x, y, z) = ln.workgroups();
        assert_eq!(x, 16); // One per batch item
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_gpu_layer_norm_generate_shader() {
        let ln = GpuLayerNorm::new(LayerNormConfig::new(16, 512)).expect("Should create");
        let shader = ln.generate_shader();

        assert!(shader.contains("@compute"));
        assert!(shader.contains("partial_sum"));
        assert!(shader.contains("mean"));
        assert!(shader.contains("variance"));
        assert!(shader.contains("gamma"));
        assert!(shader.contains("beta"));
    }

    #[test]
    fn test_gpu_layer_norm_rms_shader() {
        let ln = GpuLayerNorm::rms_norm(16, 512).expect("Should create");
        let shader = ln.generate_shader();

        assert!(shader.contains("RMS normalization"));
        assert!(shader.contains("mean of squares"));
    }

    #[test]
    fn test_layer_norm_shader_simple() {
        assert!(LAYERNORM_SHADER_SIMPLE.contains("@compute"));
        assert!(LAYERNORM_SHADER_SIMPLE.contains("mean"));
        assert!(LAYERNORM_SHADER_SIMPLE.contains("variance"));
        assert!(LAYERNORM_SHADER_SIMPLE.contains("gamma"));
        assert!(LAYERNORM_SHADER_SIMPLE.contains("beta"));
    }

    #[test]
    fn test_gpu_layer_norm_unique_ids() {
        let l1 = GpuLayerNorm::new(LayerNormConfig::new(16, 512)).expect("l1");
        let l2 = GpuLayerNorm::new(LayerNormConfig::new(16, 512)).expect("l2");
        assert_ne!(l1.id(), l2.id());
    }
}

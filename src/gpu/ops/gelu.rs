//! GPU GELU activation (WAPR-133)
//!
//! Provides Gaussian Error Linear Unit activation using compute shaders.
//! Used in transformer feed-forward networks.

use crate::gpu::error::{GpuError, GpuResult};

/// GELU approximation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluApproximation {
    /// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    Exact,
    /// Tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Tanh,
    /// Sigmoid approximation: x * sigmoid(1.702 * x) (fastest)
    Sigmoid,
}

impl Default for GeluApproximation {
    fn default() -> Self {
        Self::Tanh // Most common in transformers
    }
}

impl GeluApproximation {
    /// Get description of the approximation
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Exact => "exact GELU",
            Self::Tanh => "tanh approximation",
            Self::Sigmoid => "sigmoid approximation",
        }
    }

    /// Get relative accuracy (1.0 = exact)
    #[must_use]
    pub fn accuracy(&self) -> f32 {
        match self {
            Self::Exact => 1.0,
            Self::Tanh => 0.999,
            Self::Sigmoid => 0.995,
        }
    }

    /// Get relative speed (1.0 = baseline)
    #[must_use]
    pub fn relative_speed(&self) -> f32 {
        match self {
            Self::Exact => 1.0,
            Self::Tanh => 1.5,
            Self::Sigmoid => 2.0,
        }
    }
}

/// GELU configuration
#[derive(Debug, Clone)]
pub struct GeluConfig {
    /// Approximation method
    pub approximation: GeluApproximation,
    /// Total number of elements
    pub num_elements: u32,
    /// Whether to apply in-place (input = output buffer)
    pub inplace: bool,
    /// Workgroup size
    pub workgroup_size: u32,
    /// Label for debugging
    pub label: Option<String>,
}

impl Default for GeluConfig {
    fn default() -> Self {
        Self {
            approximation: GeluApproximation::default(),
            num_elements: 1,
            inplace: false,
            workgroup_size: 256,
            label: None,
        }
    }
}

impl GeluConfig {
    /// Create config for given number of elements
    #[must_use]
    pub fn new(num_elements: u32) -> Self {
        Self {
            num_elements,
            ..Default::default()
        }
    }

    /// Create config for transformer FFN
    #[must_use]
    pub fn for_ffn(batch_size: u32, hidden_size: u32) -> Self {
        Self {
            num_elements: batch_size * hidden_size,
            approximation: GeluApproximation::Tanh,
            ..Default::default()
        }
    }

    /// Set approximation method
    #[must_use]
    pub fn with_approximation(mut self, approx: GeluApproximation) -> Self {
        self.approximation = approx;
        self
    }

    /// Use exact GELU
    #[must_use]
    pub fn exact(mut self) -> Self {
        self.approximation = GeluApproximation::Exact;
        self
    }

    /// Use tanh approximation
    #[must_use]
    pub fn tanh(mut self) -> Self {
        self.approximation = GeluApproximation::Tanh;
        self
    }

    /// Use sigmoid approximation (fastest)
    #[must_use]
    pub fn sigmoid(mut self) -> Self {
        self.approximation = GeluApproximation::Sigmoid;
        self
    }

    /// Enable in-place operation
    #[must_use]
    pub fn inplace(mut self) -> Self {
        self.inplace = true;
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
        if self.num_elements == 0 {
            return Err(GpuError::compute("Number of elements cannot be zero"));
        }
        if !self.workgroup_size.is_power_of_two() {
            return Err(GpuError::compute("Workgroup size must be power of two"));
        }
        Ok(())
    }

    /// Calculate number of workgroups needed
    #[must_use]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_elements + self.workgroup_size - 1) / self.workgroup_size
    }
}

/// GPU GELU operation
#[derive(Debug)]
pub struct GpuGelu {
    /// Operation ID
    id: u64,
    /// Configuration
    config: GeluConfig,
    /// Whether executed
    executed: bool,
}

impl GpuGelu {
    /// Create a new GELU operation
    pub fn new(config: GeluConfig) -> GpuResult<Self> {
        config.validate()?;

        static OP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Ok(Self {
            id: OP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            config,
            executed: false,
        })
    }

    /// Create simple GELU for given elements
    pub fn simple(num_elements: u32) -> GpuResult<Self> {
        Self::new(GeluConfig::new(num_elements))
    }

    /// Create for transformer FFN
    pub fn for_ffn(batch_size: u32, hidden_size: u32) -> GpuResult<Self> {
        Self::new(GeluConfig::for_ffn(batch_size, hidden_size))
    }

    /// Get operation ID
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &GeluConfig {
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
        let input = self.config.num_elements as usize * 4;
        if self.config.inplace {
            input // Same buffer for input/output
        } else {
            input * 2 // Separate input and output
        }
    }

    /// Calculate workgroups for dispatch
    #[must_use]
    pub fn workgroups(&self) -> (u32, u32, u32) {
        (self.config.num_workgroups(), 1, 1)
    }

    /// Generate WGSL shader for this operation
    #[must_use]
    pub fn generate_shader(&self) -> String {
        let workgroup_size = self.config.workgroup_size;

        let gelu_function = match self.config.approximation {
            GeluApproximation::Exact => r#"
// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
// We approximate erf using a polynomial
fn gelu(x: f32) -> f32 {
    let sqrt2_inv = 0.7071067811865475;
    let a = x * sqrt2_inv;

    // Polynomial approximation of erf
    let a2 = a * a;
    let a3 = a2 * a;
    let erf_approx = sign(a) * (1.0 - 1.0 / (1.0 + 0.278393 * abs(a) + 0.230389 * a2 + 0.000972 * a3 + 0.078108 * a2 * a2));

    return x * 0.5 * (1.0 + erf_approx);
}"#,
            GeluApproximation::Tanh => r#"
// Tanh approximation (most common in transformers)
// x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608028654;
    let coeff = 0.044715;
    let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return x * 0.5 * (1.0 + tanh(inner));
}"#,
            GeluApproximation::Sigmoid => r#"
// Sigmoid approximation (fastest)
// x * sigmoid(1.702 * x)
fn gelu(x: f32) -> f32 {
    let sigmoid_input = 1.702 * x;
    let sigmoid_val = 1.0 / (1.0 + exp(-sigmoid_input));
    return x * sigmoid_val;
}"#,
        };

        let output_binding = if self.config.inplace {
            "@group(0) @binding(1) var<storage, read_write> data: array<f32>;"
        } else {
            "@group(0) @binding(1) var<storage, read> input: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;"
        };

        let compute_body = if self.config.inplace {
            r#"
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    data[idx] = gelu(data[idx]);"#
        } else {
            r#"
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = gelu(input[idx]);"#
        };

        format!(
            r#"// GELU activation shader ({approx})
// Elements: {num_elements}
// Workgroup size: {wg_size}
{gelu_fn}

struct Params {{
    num_elements: u32,
}}

@group(0) @binding(0) var<uniform> params: Params;
{output_binding}

@compute @workgroup_size({wg_size}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{{compute_body}
}}
"#,
            approx = self.config.approximation.description(),
            num_elements = self.config.num_elements,
            wg_size = workgroup_size,
            gelu_fn = gelu_function,
            output_binding = output_binding,
            compute_body = compute_body,
        )
    }
}

/// WGSL shader source for GELU (tanh approximation)
pub const GELU_SHADER_TANH: &str = r#"
struct Params {
    num_elements: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Tanh approximation GELU
fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608028654;
    let coeff = 0.044715;
    let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return x * 0.5 * (1.0 + tanh(inner));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = gelu(input[idx]);
}
"#;

/// WGSL shader source for GELU (sigmoid approximation - fastest)
pub const GELU_SHADER_SIGMOID: &str = r#"
struct Params {
    num_elements: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Sigmoid approximation GELU (fastest)
fn gelu(x: f32) -> f32 {
    let sigmoid_val = 1.0 / (1.0 + exp(-1.702 * x));
    return x * sigmoid_val;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = gelu(input[idx]);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_approximation_default() {
        assert_eq!(GeluApproximation::default(), GeluApproximation::Tanh);
    }

    #[test]
    fn test_gelu_approximation_description() {
        assert!(GeluApproximation::Exact.description().contains("exact"));
        assert!(GeluApproximation::Tanh.description().contains("tanh"));
        assert!(GeluApproximation::Sigmoid.description().contains("sigmoid"));
    }

    #[test]
    fn test_gelu_approximation_accuracy() {
        assert_eq!(GeluApproximation::Exact.accuracy(), 1.0);
        assert!(GeluApproximation::Tanh.accuracy() > 0.99);
        assert!(GeluApproximation::Sigmoid.accuracy() > 0.99);
    }

    #[test]
    fn test_gelu_approximation_speed() {
        // Sigmoid should be fastest
        assert!(GeluApproximation::Sigmoid.relative_speed() > GeluApproximation::Tanh.relative_speed());
        assert!(GeluApproximation::Tanh.relative_speed() > GeluApproximation::Exact.relative_speed());
    }

    #[test]
    fn test_gelu_config_default() {
        let config = GeluConfig::default();
        assert_eq!(config.approximation, GeluApproximation::Tanh);
        assert!(!config.inplace);
        assert_eq!(config.workgroup_size, 256);
    }

    #[test]
    fn test_gelu_config_new() {
        let config = GeluConfig::new(1024);
        assert_eq!(config.num_elements, 1024);
    }

    #[test]
    fn test_gelu_config_for_ffn() {
        let config = GeluConfig::for_ffn(32, 3072);
        assert_eq!(config.num_elements, 32 * 3072);
        assert_eq!(config.approximation, GeluApproximation::Tanh);
    }

    #[test]
    fn test_gelu_config_builders() {
        let config = GeluConfig::new(1024)
            .exact()
            .inplace()
            .with_workgroup_size(128)
            .with_label("test_gelu");

        assert_eq!(config.approximation, GeluApproximation::Exact);
        assert!(config.inplace);
        assert_eq!(config.workgroup_size, 128);
        assert_eq!(config.label, Some("test_gelu".to_string()));
    }

    #[test]
    fn test_gelu_config_approximation_builders() {
        assert_eq!(GeluConfig::new(1024).exact().approximation, GeluApproximation::Exact);
        assert_eq!(GeluConfig::new(1024).tanh().approximation, GeluApproximation::Tanh);
        assert_eq!(GeluConfig::new(1024).sigmoid().approximation, GeluApproximation::Sigmoid);
    }

    #[test]
    fn test_gelu_config_validate() {
        assert!(GeluConfig::new(1024).validate().is_ok());
        assert!(GeluConfig::new(0).validate().is_err());
        assert!(GeluConfig::new(1024).with_workgroup_size(100).validate().is_err());
    }

    #[test]
    fn test_gelu_config_num_workgroups() {
        let config = GeluConfig::new(1000).with_workgroup_size(256);
        assert_eq!(config.num_workgroups(), 4); // ceil(1000/256)

        let config2 = GeluConfig::new(256).with_workgroup_size(256);
        assert_eq!(config2.num_workgroups(), 1);
    }

    #[test]
    fn test_gpu_gelu_new() {
        let gelu = GpuGelu::new(GeluConfig::new(1024)).expect("Should create GELU");
        assert!(gelu.id() > 0);
        assert!(!gelu.is_executed());
    }

    #[test]
    fn test_gpu_gelu_simple() {
        let gelu = GpuGelu::simple(1024).expect("Should create");
        assert_eq!(gelu.config().num_elements, 1024);
    }

    #[test]
    fn test_gpu_gelu_for_ffn() {
        let gelu = GpuGelu::for_ffn(32, 3072).expect("Should create");
        assert_eq!(gelu.config().num_elements, 32 * 3072);
    }

    #[test]
    fn test_gpu_gelu_memory_requirement() {
        let gelu = GpuGelu::new(GeluConfig::new(1024)).expect("Should create");
        assert_eq!(gelu.memory_requirement(), 1024 * 4 * 2); // input + output

        let inplace = GpuGelu::new(GeluConfig::new(1024).inplace()).expect("Should create");
        assert_eq!(inplace.memory_requirement(), 1024 * 4); // same buffer
    }

    #[test]
    fn test_gpu_gelu_workgroups() {
        let gelu = GpuGelu::new(GeluConfig::new(1000).with_workgroup_size(256))
            .expect("Should create");
        let (x, y, z) = gelu.workgroups();
        assert_eq!(x, 4);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_gpu_gelu_generate_shader_tanh() {
        let gelu = GpuGelu::new(GeluConfig::new(1024).tanh()).expect("Should create");
        let shader = gelu.generate_shader();

        assert!(shader.contains("@compute"));
        assert!(shader.contains("tanh"));
        assert!(shader.contains("0.044715"));
    }

    #[test]
    fn test_gpu_gelu_generate_shader_sigmoid() {
        let gelu = GpuGelu::new(GeluConfig::new(1024).sigmoid()).expect("Should create");
        let shader = gelu.generate_shader();

        assert!(shader.contains("sigmoid"));
        assert!(shader.contains("1.702"));
    }

    #[test]
    fn test_gpu_gelu_generate_shader_exact() {
        let gelu = GpuGelu::new(GeluConfig::new(1024).exact()).expect("Should create");
        let shader = gelu.generate_shader();

        assert!(shader.contains("exact"));
        assert!(shader.contains("erf"));
    }

    #[test]
    fn test_gpu_gelu_generate_shader_inplace() {
        let gelu = GpuGelu::new(GeluConfig::new(1024).inplace()).expect("Should create");
        let shader = gelu.generate_shader();

        assert!(shader.contains("data[idx]"));
        assert!(!shader.contains("input[idx]"));
    }

    #[test]
    fn test_gelu_shader_tanh() {
        assert!(GELU_SHADER_TANH.contains("@compute"));
        assert!(GELU_SHADER_TANH.contains("tanh"));
        assert!(GELU_SHADER_TANH.contains("0.044715"));
    }

    #[test]
    fn test_gelu_shader_sigmoid() {
        assert!(GELU_SHADER_SIGMOID.contains("@compute"));
        assert!(GELU_SHADER_SIGMOID.contains("1.702"));
    }

    #[test]
    fn test_gpu_gelu_unique_ids() {
        let g1 = GpuGelu::new(GeluConfig::new(1024)).expect("g1");
        let g2 = GpuGelu::new(GeluConfig::new(1024)).expect("g2");
        assert_ne!(g1.id(), g2.id());
    }
}

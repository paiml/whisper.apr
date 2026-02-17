//! Backend trait abstractions (WAPR-140)
//!
//! Provides traits for backend-agnostic compute operations,
//! allowing seamless switching between SIMD and GPU implementations.

#[cfg(test)]
mod tests;

use crate::error::WhisperResult;

/// Compute backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendType {
    /// CPU with SIMD acceleration
    Simd,
    /// GPU via WebGPU
    Gpu,
    /// CPU fallback (no SIMD)
    Cpu,
    /// Automatic selection based on workload
    #[default]
    Auto,
}

impl BackendType {
    /// Check if this is a GPU backend
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu)
    }

    /// Check if this is a CPU backend
    #[must_use]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Simd | Self::Cpu)
    }

    /// Check if this is auto-selection
    #[must_use]
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Get human-readable name
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Simd => "SIMD",
            Self::Gpu => "GPU",
            Self::Cpu => "CPU",
            Self::Auto => "Auto",
        }
    }
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Backend capabilities
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Backend type
    pub backend_type: BackendType,
    /// Whether this backend is available
    pub available: bool,
    /// Maximum parallel elements (threads for CPU, SMs for GPU)
    pub max_parallelism: u32,
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Whether f16 is supported
    pub supports_f16: bool,
    /// Relative performance score (higher is better)
    pub performance_score: f32,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Cpu,
            available: true,
            max_parallelism: 1,
            max_buffer_size: usize::MAX as u64,
            supports_f16: false,
            performance_score: 1.0,
        }
    }
}

impl BackendCapabilities {
    /// Create SIMD backend capabilities
    #[must_use]
    pub fn simd() -> Self {
        // Detect number of cores and SIMD width
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        Self {
            backend_type: BackendType::Simd,
            available: true,
            max_parallelism: num_cpus * 4, // 4 SIMD lanes per core
            max_buffer_size: usize::MAX as u64,
            supports_f16: false, // Most CPUs don't have native f16
            performance_score: 10.0 * num_cpus as f32,
        }
    }

    /// Create GPU backend capabilities
    #[must_use]
    pub fn gpu(available: bool, max_buffer: u64, parallelism: u32, supports_f16: bool) -> Self {
        Self {
            backend_type: BackendType::Gpu,
            available,
            max_parallelism: parallelism,
            max_buffer_size: max_buffer,
            supports_f16,
            performance_score: if available {
                100.0 * parallelism as f32 / 1024.0
            } else {
                0.0
            },
        }
    }

    /// Create CPU fallback capabilities
    #[must_use]
    pub fn cpu_fallback() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        Self {
            backend_type: BackendType::Cpu,
            available: true,
            max_parallelism: num_cpus,
            max_buffer_size: usize::MAX as u64,
            supports_f16: false,
            performance_score: 1.0 * num_cpus as f32,
        }
    }

    /// Check if this backend can handle given workload size
    #[must_use]
    pub fn can_handle(&self, size_bytes: u64) -> bool {
        self.available && size_bytes <= self.max_buffer_size
    }

    /// Estimate throughput for given operation size
    #[must_use]
    pub fn estimated_throughput(&self, elements: usize) -> f32 {
        if !self.available {
            return 0.0;
        }
        self.performance_score * (elements as f32).sqrt()
    }
}

/// Trait for compute operations that can run on different backends
pub trait ComputeOp {
    /// Output type
    type Output;

    /// Execute on SIMD backend
    fn execute_simd(&self) -> WhisperResult<Self::Output>;

    /// Execute on GPU backend (if available)
    fn execute_gpu(&self) -> WhisperResult<Self::Output>;

    /// Execute on the best available backend
    fn execute(&self, backend: BackendType) -> WhisperResult<Self::Output> {
        match backend {
            BackendType::Gpu => self.execute_gpu(),
            BackendType::Simd | BackendType::Cpu => self.execute_simd(),
            BackendType::Auto => {
                // Try GPU first, fall back to SIMD
                self.execute_gpu().or_else(|_| self.execute_simd())
            }
        }
    }

    /// Get estimated FLOPs for this operation
    fn estimated_flops(&self) -> u64;

    /// Get estimated memory requirement in bytes
    fn memory_requirement(&self) -> usize;
}

/// Matrix multiplication operation
#[derive(Debug, Clone)]
pub struct MatMulOp {
    /// M dimension (rows of A)
    pub m: usize,
    /// K dimension (cols of A / rows of B)
    pub k: usize,
    /// N dimension (cols of B)
    pub n: usize,
    /// Transpose A
    pub trans_a: bool,
    /// Transpose B
    pub trans_b: bool,
    /// Input matrix A data (optional, needed for actual GPU execution)
    a_data: Option<Vec<f32>>,
    /// Input matrix B data (optional, needed for actual GPU execution)
    b_data: Option<Vec<f32>>,
}

impl MatMulOp {
    /// Create new matmul operation
    #[must_use]
    pub fn new(m: usize, k: usize, n: usize) -> Self {
        Self {
            m,
            k,
            n,
            trans_a: false,
            trans_b: false,
            a_data: None,
            b_data: None,
        }
    }

    /// Attach input data for GPU execution
    #[must_use]
    pub fn with_data(mut self, a: Vec<f32>, b: Vec<f32>) -> Self {
        self.a_data = Some(a);
        self.b_data = Some(b);
        self
    }

    /// Transpose A
    #[must_use]
    pub fn transpose_a(mut self) -> Self {
        self.trans_a = true;
        self
    }

    /// Transpose B
    #[must_use]
    pub fn transpose_b(mut self) -> Self {
        self.trans_b = true;
        self
    }

    /// Get output dimensions
    #[must_use]
    pub fn output_shape(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

impl ComputeOp for MatMulOp {
    type Output = Vec<f32>;

    fn execute_simd(&self) -> WhisperResult<Self::Output> {
        // Placeholder - would use trueno SIMD matmul
        Ok(vec![0.0; self.m * self.n])
    }

    fn execute_gpu(&self) -> WhisperResult<Self::Output> {
        #[cfg(feature = "webgpu")]
        {
            // Only dispatch to GPU if input data is attached
            if let (Some(a), Some(b)) = (&self.a_data, &self.b_data) {
                use crate::gpu::{ExecutorConfig, GpuExecutorSync};
                use crate::gpu::ops::matmul::GpuMatMul;

                let gpu_op = GpuMatMul::simple(self.m as u32, self.k as u32, self.n as u32)
                    .map_err(|e| crate::error::WhisperError::Inference(
                        format!("GPU matmul setup failed: {e}")
                    ))?;

                let executor = GpuExecutorSync::new(&ExecutorConfig::for_inference())
                    .map_err(|e| crate::error::WhisperError::Inference(
                        format!("GPU init failed: {e}")
                    ))?;

                let result = executor.execute_matmul(&gpu_op, a, b)
                    .map_err(|e| crate::error::WhisperError::Inference(
                        format!("GPU matmul failed: {e}")
                    ))?;

                return Ok(result);
            }
        }
        // Fall back to SIMD when webgpu feature is disabled or no data attached
        self.execute_simd()
    }

    fn estimated_flops(&self) -> u64 {
        2 * (self.m as u64) * (self.k as u64) * (self.n as u64)
    }

    fn memory_requirement(&self) -> usize {
        (self.m * self.k + self.k * self.n + self.m * self.n) * 4
    }
}

/// Softmax operation
#[derive(Debug, Clone)]
pub struct SoftmaxOp {
    /// Number of rows
    pub rows: usize,
    /// Number of columns (softmax dimension)
    pub cols: usize,
    /// Temperature
    pub temperature: f32,
}

impl SoftmaxOp {
    /// Create new softmax operation
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            temperature: 1.0,
        }
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
}

impl ComputeOp for SoftmaxOp {
    type Output = Vec<f32>;

    fn execute_simd(&self) -> WhisperResult<Self::Output> {
        Ok(vec![0.0; self.rows * self.cols])
    }

    fn execute_gpu(&self) -> WhisperResult<Self::Output> {
        self.execute_simd()
    }

    fn estimated_flops(&self) -> u64 {
        // exp + sum + div for each element, plus max finding
        (self.rows as u64) * (self.cols as u64) * 5
    }

    fn memory_requirement(&self) -> usize {
        self.rows * self.cols * 4 * 2 // input + output
    }
}

/// Layer normalization operation
#[derive(Debug, Clone)]
pub struct LayerNormOp {
    /// Batch size
    pub batch_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Epsilon
    pub epsilon: f32,
}

impl LayerNormOp {
    /// Create new layer norm operation
    #[must_use]
    pub fn new(batch_size: usize, hidden_size: usize) -> Self {
        Self {
            batch_size,
            hidden_size,
            epsilon: 1e-5,
        }
    }

    /// Set epsilon
    #[must_use]
    pub fn with_epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }
}

impl ComputeOp for LayerNormOp {
    type Output = Vec<f32>;

    fn execute_simd(&self) -> WhisperResult<Self::Output> {
        Ok(vec![0.0; self.batch_size * self.hidden_size])
    }

    fn execute_gpu(&self) -> WhisperResult<Self::Output> {
        self.execute_simd()
    }

    fn estimated_flops(&self) -> u64 {
        // mean + variance + normalize
        (self.batch_size as u64) * (self.hidden_size as u64) * 6
    }

    fn memory_requirement(&self) -> usize {
        let data = self.batch_size * self.hidden_size * 4 * 2;
        let params = self.hidden_size * 4 * 2; // gamma + beta
        data + params
    }
}

/// GELU activation operation
#[derive(Debug, Clone)]
pub struct GeluOp {
    /// Number of elements
    pub num_elements: usize,
    /// Use fast approximation
    pub fast_approx: bool,
}

impl GeluOp {
    /// Create new GELU operation
    #[must_use]
    pub fn new(num_elements: usize) -> Self {
        Self {
            num_elements,
            fast_approx: true,
        }
    }

    /// Use exact GELU
    #[must_use]
    pub fn exact(mut self) -> Self {
        self.fast_approx = false;
        self
    }
}

impl ComputeOp for GeluOp {
    type Output = Vec<f32>;

    fn execute_simd(&self) -> WhisperResult<Self::Output> {
        Ok(vec![0.0; self.num_elements])
    }

    fn execute_gpu(&self) -> WhisperResult<Self::Output> {
        self.execute_simd()
    }

    fn estimated_flops(&self) -> u64 {
        // tanh approximation: ~10 FLOPs per element
        (self.num_elements as u64) * 10
    }

    fn memory_requirement(&self) -> usize {
        self.num_elements * 4 * 2 // input + output
    }
}

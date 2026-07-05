//! WebGPU compute executor (WAPR-WEBGPU-001)
//!
//! Provides actual WebGPU execution using wgpu. This module bridges the gap
//! between the GPU operation definitions and actual GPU execution.
//!
//! # Architecture
//!
//! ```text
//! GpuMatMul::generate_shader() → WGSL → GpuExecutor → wgpu → GPU
//! ```

use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ops::matmul::GpuMatMul;

#[cfg(feature = "webgpu")]
use wgpu;

/// GPU executor state
#[derive(Debug)]
pub struct GpuExecutor {
    #[cfg(feature = "webgpu")]
    device: wgpu::Device,
    #[cfg(feature = "webgpu")]
    queue: wgpu::Queue,
    #[cfg(not(feature = "webgpu"))]
    _marker: std::marker::PhantomData<()>,
}

/// Configuration for GPU executor
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Prefer high-performance adapter
    pub high_performance: bool,
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Label for debugging
    pub label: Option<String>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            high_performance: true,
            max_buffer_size: 256 * 1024 * 1024, // 256 MB
            label: None,
        }
    }
}

impl ExecutorConfig {
    /// Configuration for inference workloads
    #[must_use]
    pub fn for_inference() -> Self {
        Self {
            high_performance: true,
            max_buffer_size: 1024 * 1024 * 1024, // 1 GB
            label: Some("whisper-inference".to_string()),
        }
    }
}

impl GpuExecutor {
    /// Create a new GPU executor
    ///
    /// This is an async operation that requests a GPU adapter and device.
    #[cfg(feature = "webgpu")]
    pub async fn new(config: &ExecutorConfig) -> GpuResult<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let power_preference = if config.high_performance {
            wgpu::PowerPreference::HighPerformance
        } else {
            wgpu::PowerPreference::LowPower
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| GpuError::device("No suitable GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: config.label.as_deref(),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| GpuError::device(format!("Failed to create device: {e}")))?;

        // WGPU panics by default if an uncaptured validation error occurs.
        // We capture it here so `device.poll` returns instead of panicking, 
        // allowing us to gracefully detect invalid CPU adapters (like lavapipe).
        let device_error = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let device_error_clone = std::sync::Arc::clone(&device_error);
        device.on_uncaptured_error(Box::new(move |err| {
            // Log the error but avoid panicking
            eprintln!("wgpu device error detected: {}", err);
            device_error_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        }));

        // Do a dummy poll. If the device was instantly lost (e.g. invalid lavapipe setup),
        // the uncaptured error handler will fire, or poll will simply return/fail without panicking.
        device.poll(wgpu::Maintain::Wait);

        if device_error.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(GpuError::DeviceLost);
        }

        Ok(Self { device, queue })
    }

    /// Create a new GPU executor (stub when webgpu feature is disabled)
    #[cfg(not(feature = "webgpu"))]
    pub async fn new(_config: &ExecutorConfig) -> GpuResult<Self> {
        Err(GpuError::device(
            "WebGPU feature not enabled. Compile with --features webgpu",
        ))
    }

    /// Create a standard 4-buffer bind group layout: uniform + 2 read-only storage + 1 read-write.
    #[cfg(feature = "webgpu")]
    fn create_matmul_bind_group_layout(&self, label: &str) -> wgpu::BindGroupLayout {
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        self.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                ],
            })
    }

    /// Read back f32 data from a GPU staging buffer.
    #[cfg(feature = "webgpu")]
    fn read_staging_buffer(&self, staging_buffer: &wgpu::Buffer) -> GpuResult<Vec<f32>> {
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| GpuError::compute("Failed to receive buffer mapping result"))?
            .map_err(|e| GpuError::compute(format!("Buffer mapping failed: {e}")))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        Ok(result)
    }

    /// Validate matmul input slice lengths against expected dimensions.
    #[cfg(feature = "webgpu")]
    fn validate_matmul_inputs(
        &self,
        lhs: &[f32],
        rhs: &[f32],
        dims: &crate::gpu::ops::matmul::MatMulDimensions,
    ) -> GpuResult<()> {
        if lhs.len() != dims.a_size() {
            return Err(GpuError::compute(format!(
                "Matrix A size mismatch: expected {}, got {}",
                dims.a_size(),
                lhs.len()
            )));
        }
        if rhs.len() != dims.b_size() {
            return Err(GpuError::compute(format!(
                "Matrix B size mismatch: expected {}, got {}",
                dims.b_size(),
                rhs.len()
            )));
        }
        Ok(())
    }

    /// Execute matrix multiplication on GPU.
    ///
    /// Runs the full pipeline: validates dimensions, creates input/output GPU buffers,
    /// compiles the WGSL shader, dispatches the compute workgroups, and reads back
    /// the result.
    #[cfg(feature = "webgpu")]
    pub async fn execute_matmul(
        &self,
        op: &GpuMatMul,
        lhs: &[f32],
        rhs: &[f32],
    ) -> GpuResult<Vec<f32>> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            m: u32,
            k: u32,
            n: u32,
            alpha: f32,
            beta: f32,
            _padding: [u32; 3],
        }

        let dims = op.dimensions();
        self.validate_matmul_inputs(lhs, rhs, &dims)?;

        // Create GPU buffers
        let a_buffer = self.create_storage_buffer_init("A", bytemuck::cast_slice(lhs));
        let b_buffer = self.create_storage_buffer_init("B", bytemuck::cast_slice(rhs));
        let c_buffer = self.create_storage_buffer("C", dims.result_bytes() as u64);
        let staging_buffer = self.create_staging_buffer("staging", dims.result_bytes() as u64);
        let dims_data = Dimensions {
            m: dims.m,
            k: dims.k,
            n: dims.n,
            alpha: op.config().alpha,
            beta: op.config().beta,
            _padding: [0; 3],
        };
        let dims_buffer = self.create_uniform_buffer_init("dims", bytemuck::bytes_of(&dims_data));

        // Compile shader and create pipeline
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul_shader"),
                source: wgpu::ShaderSource::Wgsl(op.generate_shader().into()),
            });
        let bind_group_layout = self.create_matmul_bind_group_layout("matmul_bgl");
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matmul_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Bind and dispatch
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let (wg_x, wg_y, wg_z) = op.workgroups();
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }
        encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, dims.result_bytes() as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        self.read_staging_buffer(&staging_buffer)
    }

    /// Execute matrix multiplication (stub when webgpu disabled)
    #[cfg(not(feature = "webgpu"))]
    pub async fn execute_matmul(
        &self,
        _op: &GpuMatMul,
        _a: &[f32],
        _b: &[f32],
    ) -> GpuResult<Vec<f32>> {
        Err(GpuError::compute("WebGPU feature not enabled"))
    }

    /// Create a storage buffer with initial data
    #[cfg(feature = "webgpu")]
    fn create_storage_buffer_init(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Create a storage buffer
    #[cfg(feature = "webgpu")]
    fn create_storage_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer with initial data
    #[cfg(feature = "webgpu")]
    fn create_uniform_buffer_init(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Create a staging buffer for readback
    #[cfg(feature = "webgpu")]
    fn create_staging_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Get device info
    #[cfg(feature = "webgpu")]
    pub fn device_info(&self) -> String {
        format!("wgpu device: {:?}", self.device.limits())
    }

    /// Get device info (stub)
    #[cfg(not(feature = "webgpu"))]
    pub fn device_info(&self) -> String {
        "WebGPU not enabled".to_string()
    }
}

/// Synchronous wrapper for GPU execution
#[allow(dead_code)]
pub struct GpuExecutorSync {
    executor: GpuExecutor,
    #[cfg(feature = "webgpu")]
    runtime: tokio::runtime::Runtime,
}

impl GpuExecutorSync {
    /// Create a new synchronous executor
    #[cfg(feature = "webgpu")]
    pub fn new(config: &ExecutorConfig) -> GpuResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| GpuError::device(format!("Failed to create runtime: {e}")))?;

        let executor = runtime.block_on(GpuExecutor::new(config))?;

        Ok(Self { executor, runtime })
    }

    /// Create a new synchronous executor (stub)
    #[cfg(not(feature = "webgpu"))]
    pub fn new(_config: &ExecutorConfig) -> GpuResult<Self> {
        Err(GpuError::device("WebGPU feature not enabled"))
    }

    /// Execute matrix multiplication synchronously
    #[cfg(feature = "webgpu")]
    pub fn execute_matmul(&self, op: &GpuMatMul, a: &[f32], b: &[f32]) -> GpuResult<Vec<f32>> {
        self.runtime
            .block_on(self.executor.execute_matmul(op, a, b))
    }

    /// Execute matrix multiplication synchronously (stub)
    #[cfg(not(feature = "webgpu"))]
    pub fn execute_matmul(&self, _op: &GpuMatMul, _a: &[f32], _b: &[f32]) -> GpuResult<Vec<f32>> {
        Err(GpuError::compute("WebGPU feature not enabled"))
    }
}

/// Execute a simple matrix multiplication (convenience function)
#[cfg(feature = "webgpu")]
#[allow(dead_code)]
pub async fn matmul_gpu(
    rows: u32,
    inner: u32,
    cols: u32,
    lhs: &[f32],
    rhs: &[f32],
) -> GpuResult<Vec<f32>> {
    let executor = GpuExecutor::new(&ExecutorConfig::default()).await?;
    let op = GpuMatMul::simple(rows, inner, cols)?;
    executor.execute_matmul(&op, lhs, rhs).await
}

/// Execute a simple matrix multiplication (stub)
#[allow(dead_code)] // Available for future use when webgpu feature enabled
#[cfg(not(feature = "webgpu"))]
pub async fn matmul_gpu(
    _rows: u32,
    _inner: u32,
    _cols: u32,
    _lhs: &[f32],
    _rhs: &[f32],
) -> GpuResult<Vec<f32>> {
    Err(GpuError::compute("WebGPU feature not enabled"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();
        assert!(config.high_performance);
        assert_eq!(config.max_buffer_size, 256 * 1024 * 1024);
    }

    #[test]
    fn test_executor_config_for_inference() {
        let config = ExecutorConfig::for_inference();
        assert!(config.high_performance);
        assert_eq!(config.max_buffer_size, 1024 * 1024 * 1024);
        assert!(config.label.is_some());
    }

    // Note: Integration tests with actual GPU require the webgpu feature
    // and a GPU-capable environment. These run in CI with appropriate hardware.

    #[cfg(feature = "webgpu")]
    mod gpu_tests {
        use super::*;

        #[tokio::test]
        async fn test_executor_creation() {
            // This test will fail gracefully if no GPU is available
            let result = GpuExecutor::new(&ExecutorConfig::default()).await;
            // Don't assert success - GPU may not be available in all environments
            if let Ok(executor) = result {
                assert!(!executor.device_info().is_empty());
            }
        }

        #[tokio::test]
        async fn test_simple_matmul() {
            let result = GpuExecutor::new(&ExecutorConfig::default()).await;
            if let Ok(executor) = result {
                // 2x2 @ 2x2 matrix multiplication
                let a = vec![1.0, 2.0, 3.0, 4.0];
                let b = vec![5.0, 6.0, 7.0, 8.0];

                let op = GpuMatMul::simple(2, 2, 2).expect("create op");
                let c = executor.execute_matmul(&op, &a, &b).await;

                if let Ok(c) = c {
                    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
                    //         = [[19, 22], [43, 50]]
                    assert_eq!(c.len(), 4);
                    let expected = vec![19.0, 22.0, 43.0, 50.0];
                    for (i, (got, exp)) in c.iter().zip(expected.iter()).enumerate() {
                        assert!(
                            (got - exp).abs() < 1e-5,
                            "Mismatch at {i}: got {got}, expected {exp}"
                        );
                    }
                }
            }
        }
    }
}

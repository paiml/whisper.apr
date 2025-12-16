//! GPU matrix multiplication (WAPR-130)
//!
//! Provides tiled matrix multiplication using compute shaders.
//! Optimized for transformer attention and feed-forward operations.

use crate::gpu::error::{GpuError, GpuResult};

/// Tile size for matrix multiplication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileSize {
    /// 8x8 tiles (small matrices, limited hardware)
    Tile8x8,
    /// 16x16 tiles (medium matrices)
    Tile16x16,
    /// 32x32 tiles (large matrices, high-end hardware)
    Tile32x32,
}

impl Default for TileSize {
    fn default() -> Self {
        Self::Tile16x16
    }
}

impl TileSize {
    /// Get the numeric tile dimension
    #[must_use]
    pub fn dimension(&self) -> u32 {
        match self {
            Self::Tile8x8 => 8,
            Self::Tile16x16 => 16,
            Self::Tile32x32 => 32,
        }
    }

    /// Get workgroup size for this tile size
    #[must_use]
    pub fn workgroup_size(&self) -> u32 {
        let dim = self.dimension();
        dim * dim
    }

    /// Select optimal tile size for given matrix dimensions
    #[must_use]
    pub fn optimal_for(m: u32, n: u32, max_workgroup: u32) -> Self {
        if max_workgroup >= 1024 && m >= 32 && n >= 32 {
            Self::Tile32x32
        } else if max_workgroup >= 256 && m >= 16 && n >= 16 {
            Self::Tile16x16
        } else {
            Self::Tile8x8
        }
    }

    /// Check if tile size is compatible with workgroup limits
    #[must_use]
    pub fn compatible_with(&self, max_workgroup: u32) -> bool {
        self.workgroup_size() <= max_workgroup
    }
}

/// Matrix multiplication dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulDimensions {
    /// Rows of A (and result)
    pub m: u32,
    /// Columns of A / Rows of B (shared dimension)
    pub k: u32,
    /// Columns of B (and result)
    pub n: u32,
}

impl MatMulDimensions {
    /// Create new dimensions
    #[must_use]
    pub fn new(m: u32, k: u32, n: u32) -> Self {
        Self { m, k, n }
    }

    /// Create for square matrices
    #[must_use]
    pub fn square(size: u32) -> Self {
        Self::new(size, size, size)
    }

    /// Validate dimensions
    pub fn validate(&self) -> GpuResult<()> {
        if self.m == 0 || self.k == 0 || self.n == 0 {
            return Err(GpuError::compute("Matrix dimensions cannot be zero"));
        }
        Ok(())
    }

    /// Get result matrix size (m x n)
    #[must_use]
    pub fn result_size(&self) -> usize {
        (self.m as usize) * (self.n as usize)
    }

    /// Get result matrix size in bytes (assuming f32)
    #[must_use]
    pub fn result_bytes(&self) -> usize {
        self.result_size() * 4
    }

    /// Get A matrix size
    #[must_use]
    pub fn a_size(&self) -> usize {
        (self.m as usize) * (self.k as usize)
    }

    /// Get B matrix size
    #[must_use]
    pub fn b_size(&self) -> usize {
        (self.k as usize) * (self.n as usize)
    }

    /// Calculate number of workgroups needed for given tile size
    #[must_use]
    pub fn workgroups(&self, tile: TileSize) -> (u32, u32) {
        let dim = tile.dimension();
        let x = self.n.div_ceil(dim);
        let y = self.m.div_ceil(dim);
        (x, y)
    }

    /// Get total FLOPs for this multiplication
    #[must_use]
    pub fn flops(&self) -> u64 {
        // Each element requires K multiply-adds = 2K FLOPs
        2 * (self.m as u64) * (self.k as u64) * (self.n as u64)
    }

    /// Check if dimensions are amenable to tiled multiplication
    #[must_use]
    pub fn is_tileable(&self, tile: TileSize) -> bool {
        let dim = tile.dimension();
        self.m >= dim && self.n >= dim
    }
}

/// Matrix multiplication configuration
#[derive(Debug, Clone)]
pub struct MatMulConfig {
    /// Tile size for the computation
    pub tile_size: TileSize,
    /// Whether to transpose A
    pub transpose_a: bool,
    /// Whether to transpose B
    pub transpose_b: bool,
    /// Optional alpha scalar (C = alpha * A @ B + beta * C)
    pub alpha: f32,
    /// Optional beta scalar for accumulation
    pub beta: f32,
    /// Label for debugging
    pub label: Option<String>,
}

impl Default for MatMulConfig {
    fn default() -> Self {
        Self {
            tile_size: TileSize::default(),
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
            label: None,
        }
    }
}

impl MatMulConfig {
    /// Create default config
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tile size
    #[must_use]
    pub fn with_tile_size(mut self, tile_size: TileSize) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Transpose A matrix
    #[must_use]
    pub fn transpose_a(mut self) -> Self {
        self.transpose_a = true;
        self
    }

    /// Transpose B matrix
    #[must_use]
    pub fn transpose_b(mut self) -> Self {
        self.transpose_b = true;
        self
    }

    /// Set GEMM scalars
    #[must_use]
    pub fn with_gemm(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Check if this is a simple matmul (no transpose, alpha=1, beta=0)
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn is_simple(&self) -> bool {
        !self.transpose_a && !self.transpose_b && self.alpha == 1.0 && self.beta == 0.0
    }
}

/// GPU matrix multiplication operation
#[derive(Debug)]
pub struct GpuMatMul {
    /// Operation ID
    id: u64,
    /// Dimensions
    dimensions: MatMulDimensions,
    /// Configuration
    config: MatMulConfig,
    /// Whether the operation has been executed
    executed: bool,
}

impl GpuMatMul {
    /// Create a new matrix multiplication operation
    #[allow(clippy::items_after_statements)]
    pub fn new(dimensions: MatMulDimensions, config: MatMulConfig) -> GpuResult<Self> {
        dimensions.validate()?;

        static OP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Ok(Self {
            id: OP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            dimensions,
            config,
            executed: false,
        })
    }

    /// Create with default configuration
    pub fn simple(m: u32, k: u32, n: u32) -> GpuResult<Self> {
        Self::new(MatMulDimensions::new(m, k, n), MatMulConfig::default())
    }

    /// Get operation ID
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get dimensions
    #[must_use]
    pub fn dimensions(&self) -> &MatMulDimensions {
        &self.dimensions
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &MatMulConfig {
        &self.config
    }

    /// Check if executed
    #[must_use]
    pub fn is_executed(&self) -> bool {
        self.executed
    }

    /// Get estimated memory requirement in bytes
    #[must_use]
    pub fn memory_requirement(&self) -> usize {
        let a = self.dimensions.a_size() * 4;
        let b = self.dimensions.b_size() * 4;
        let c = self.dimensions.result_bytes();
        a + b + c
    }

    /// Get estimated FLOPs
    #[must_use]
    pub fn flops(&self) -> u64 {
        self.dimensions.flops()
    }

    /// Calculate workgroups for dispatch
    #[must_use]
    pub fn workgroups(&self) -> (u32, u32, u32) {
        let (x, y) = self.dimensions.workgroups(self.config.tile_size);
        (x, y, 1)
    }

    /// Generate WGSL shader for this operation
    #[must_use]
    pub fn generate_shader(&self) -> String {
        let tile = self.config.tile_size.dimension();
        let _workgroup_size = self.config.tile_size.workgroup_size();

        format!(
            r"// Matrix multiplication shader
// Dimensions: {}x{} @ {}x{} = {}x{}
// Tile size: {}x{}

struct Dimensions {{
    M: u32,
    K: u32,
    N: u32,
    alpha: f32,
    beta: f32,
}}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

var<workgroup> tile_A: array<f32, {tile_mem}>;
var<workgroup> tile_B: array<f32, {tile_mem}>;

@compute @workgroup_size({tile}, {tile}, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {{
    let row = workgroup_id.y * {tile}u + local_id.y;
    let col = workgroup_id.x * {tile}u + local_id.x;

    var sum: f32 = 0.0;

    let num_tiles = (dims.K + {tile}u - 1u) / {tile}u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {{
        let tile_row = t * {tile}u + local_id.y;
        let tile_col = t * {tile}u + local_id.x;

        // Load tile from A
        if (row < dims.M && tile_col < dims.K) {{
            tile_A[local_id.y * {tile}u + local_id.x] = A[row * dims.K + tile_col];
        }} else {{
            tile_A[local_id.y * {tile}u + local_id.x] = 0.0;
        }}

        // Load tile from B
        if (tile_row < dims.K && col < dims.N) {{
            tile_B[local_id.y * {tile}u + local_id.x] = B[tile_row * dims.N + col];
        }} else {{
            tile_B[local_id.y * {tile}u + local_id.x] = 0.0;
        }}

        workgroupBarrier();

        // Compute partial sum
        for (var k: u32 = 0u; k < {tile}u; k = k + 1u) {{
            sum = sum + tile_A[local_id.y * {tile}u + k] * tile_B[k * {tile}u + local_id.x];
        }}

        workgroupBarrier();
    }}

    // Write result
    if (row < dims.M && col < dims.N) {{
        let idx = row * dims.N + col;
        C[idx] = dims.alpha * sum + dims.beta * C[idx];
    }}
}}
",
            self.dimensions.m,
            self.dimensions.k,
            self.dimensions.k,
            self.dimensions.n,
            self.dimensions.m,
            self.dimensions.n,
            tile,
            tile,
            tile = tile,
            tile_mem = tile * tile,
        )
    }
}

/// WGSL shader source for simple matrix multiplication
#[allow(dead_code)]
pub const MATMUL_SHADER_SIMPLE: &str = r"
struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
        sum = sum + A[row * dims.K + k] * B[k * dims.N + col];
    }

    C[row * dims.N + col] = sum;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_size_dimension() {
        assert_eq!(TileSize::Tile8x8.dimension(), 8);
        assert_eq!(TileSize::Tile16x16.dimension(), 16);
        assert_eq!(TileSize::Tile32x32.dimension(), 32);
    }

    #[test]
    fn test_tile_size_workgroup_size() {
        assert_eq!(TileSize::Tile8x8.workgroup_size(), 64);
        assert_eq!(TileSize::Tile16x16.workgroup_size(), 256);
        assert_eq!(TileSize::Tile32x32.workgroup_size(), 1024);
    }

    #[test]
    fn test_tile_size_optimal_for() {
        assert_eq!(TileSize::optimal_for(256, 256, 1024), TileSize::Tile32x32);
        assert_eq!(TileSize::optimal_for(64, 64, 256), TileSize::Tile16x16);
        assert_eq!(TileSize::optimal_for(8, 8, 64), TileSize::Tile8x8);
    }

    #[test]
    fn test_tile_size_compatible_with() {
        assert!(TileSize::Tile8x8.compatible_with(64));
        assert!(TileSize::Tile16x16.compatible_with(256));
        assert!(!TileSize::Tile32x32.compatible_with(256));
    }

    #[test]
    fn test_matmul_dimensions_new() {
        let dims = MatMulDimensions::new(64, 128, 64);
        assert_eq!(dims.m, 64);
        assert_eq!(dims.k, 128);
        assert_eq!(dims.n, 64);
    }

    #[test]
    fn test_matmul_dimensions_square() {
        let dims = MatMulDimensions::square(256);
        assert_eq!(dims.m, 256);
        assert_eq!(dims.k, 256);
        assert_eq!(dims.n, 256);
    }

    #[test]
    fn test_matmul_dimensions_validate() {
        assert!(MatMulDimensions::new(64, 128, 64).validate().is_ok());
        assert!(MatMulDimensions::new(0, 128, 64).validate().is_err());
        assert!(MatMulDimensions::new(64, 0, 64).validate().is_err());
        assert!(MatMulDimensions::new(64, 128, 0).validate().is_err());
    }

    #[test]
    fn test_matmul_dimensions_sizes() {
        let dims = MatMulDimensions::new(64, 128, 32);
        assert_eq!(dims.a_size(), 64 * 128);
        assert_eq!(dims.b_size(), 128 * 32);
        assert_eq!(dims.result_size(), 64 * 32);
        assert_eq!(dims.result_bytes(), 64 * 32 * 4);
    }

    #[test]
    fn test_matmul_dimensions_workgroups() {
        let dims = MatMulDimensions::new(64, 128, 64);
        let (x, y) = dims.workgroups(TileSize::Tile16x16);
        assert_eq!(x, 4); // 64 / 16
        assert_eq!(y, 4); // 64 / 16
    }

    #[test]
    fn test_matmul_dimensions_flops() {
        let dims = MatMulDimensions::new(64, 128, 64);
        // 2 * M * K * N = 2 * 64 * 128 * 64 = 1,048,576
        assert_eq!(dims.flops(), 2 * 64 * 128 * 64);
    }

    #[test]
    fn test_matmul_config_default() {
        let config = MatMulConfig::default();
        assert_eq!(config.tile_size, TileSize::Tile16x16);
        assert!(!config.transpose_a);
        assert!(!config.transpose_b);
        assert_eq!(config.alpha, 1.0);
        assert_eq!(config.beta, 0.0);
    }

    #[test]
    fn test_matmul_config_builders() {
        let config = MatMulConfig::new()
            .with_tile_size(TileSize::Tile32x32)
            .transpose_a()
            .transpose_b()
            .with_gemm(0.5, 0.5)
            .with_label("test_matmul");

        assert_eq!(config.tile_size, TileSize::Tile32x32);
        assert!(config.transpose_a);
        assert!(config.transpose_b);
        assert_eq!(config.alpha, 0.5);
        assert_eq!(config.beta, 0.5);
        assert_eq!(config.label, Some("test_matmul".to_string()));
    }

    #[test]
    fn test_matmul_config_is_simple() {
        assert!(MatMulConfig::default().is_simple());
        assert!(!MatMulConfig::default().transpose_a().is_simple());
        assert!(!MatMulConfig::default().with_gemm(0.5, 0.0).is_simple());
    }

    #[test]
    fn test_gpu_matmul_new() {
        let dims = MatMulDimensions::new(64, 128, 64);
        let matmul = GpuMatMul::new(dims, MatMulConfig::default()).expect("Should create matmul");
        assert!(matmul.id() > 0);
        assert!(!matmul.is_executed());
    }

    #[test]
    fn test_gpu_matmul_simple() {
        let matmul = GpuMatMul::simple(64, 128, 64).expect("Should create");
        assert_eq!(matmul.dimensions().m, 64);
        assert_eq!(matmul.dimensions().k, 128);
        assert_eq!(matmul.dimensions().n, 64);
    }

    #[test]
    fn test_gpu_matmul_memory_requirement() {
        let matmul = GpuMatMul::simple(64, 128, 64).expect("Should create");
        // A: 64*128*4 = 32768
        // B: 128*64*4 = 32768
        // C: 64*64*4 = 16384
        // Total: 81920
        assert_eq!(matmul.memory_requirement(), 81920);
    }

    #[test]
    fn test_gpu_matmul_workgroups() {
        let matmul = GpuMatMul::simple(64, 128, 64).expect("Should create");
        let (x, y, z) = matmul.workgroups();
        assert_eq!(x, 4);
        assert_eq!(y, 4);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_gpu_matmul_generate_shader() {
        let matmul = GpuMatMul::simple(64, 128, 64).expect("Should create");
        let shader = matmul.generate_shader();

        assert!(shader.contains("@compute"));
        assert!(shader.contains("@workgroup_size"));
        assert!(shader.contains("tile_A"));
        assert!(shader.contains("tile_B"));
        assert!(shader.contains("workgroupBarrier"));
    }

    #[test]
    fn test_matmul_shader_simple() {
        assert!(MATMUL_SHADER_SIMPLE.contains("@compute"));
        assert!(MATMUL_SHADER_SIMPLE.contains("Dimensions"));
        assert!(MATMUL_SHADER_SIMPLE.contains("@workgroup_size(16, 16, 1)"));
    }

    #[test]
    fn test_gpu_matmul_unique_ids() {
        let m1 = GpuMatMul::simple(64, 128, 64).expect("m1");
        let m2 = GpuMatMul::simple(64, 128, 64).expect("m2");
        assert_ne!(m1.id(), m2.id());
    }

    #[test]
    fn test_matmul_dimensions_is_tileable() {
        let dims = MatMulDimensions::new(64, 128, 64);
        assert!(dims.is_tileable(TileSize::Tile16x16));
        assert!(dims.is_tileable(TileSize::Tile32x32));

        let small = MatMulDimensions::new(8, 8, 8);
        assert!(small.is_tileable(TileSize::Tile8x8));
        assert!(!small.is_tileable(TileSize::Tile16x16));
    }
}

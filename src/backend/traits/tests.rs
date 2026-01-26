//! Tests for backend trait abstractions

use super::*;

#[test]
fn test_backend_type_default() {
    assert_eq!(BackendType::default(), BackendType::Auto);
}

#[test]
fn test_backend_type_is_gpu() {
    assert!(BackendType::Gpu.is_gpu());
    assert!(!BackendType::Simd.is_gpu());
    assert!(!BackendType::Cpu.is_gpu());
}

#[test]
fn test_backend_type_is_cpu() {
    assert!(BackendType::Simd.is_cpu());
    assert!(BackendType::Cpu.is_cpu());
    assert!(!BackendType::Gpu.is_cpu());
}

#[test]
fn test_backend_type_display() {
    assert_eq!(BackendType::Simd.to_string(), "SIMD");
    assert_eq!(BackendType::Gpu.to_string(), "GPU");
    assert_eq!(BackendType::Cpu.to_string(), "CPU");
    assert_eq!(BackendType::Auto.to_string(), "Auto");
}

#[test]
fn test_backend_capabilities_simd() {
    let caps = BackendCapabilities::simd();
    assert_eq!(caps.backend_type, BackendType::Simd);
    assert!(caps.available);
    assert!(caps.max_parallelism >= 4);
}

#[test]
fn test_backend_capabilities_gpu() {
    let caps = BackendCapabilities::gpu(true, 256 * 1024 * 1024, 4096, true);
    assert_eq!(caps.backend_type, BackendType::Gpu);
    assert!(caps.available);
    assert!(caps.supports_f16);
    assert!(caps.performance_score > 0.0);
}

#[test]
fn test_backend_capabilities_gpu_unavailable() {
    let caps = BackendCapabilities::gpu(false, 0, 0, false);
    assert!(!caps.available);
    assert_eq!(caps.performance_score, 0.0);
}

#[test]
fn test_backend_capabilities_can_handle() {
    let caps = BackendCapabilities::gpu(true, 256 * 1024 * 1024, 4096, true);
    assert!(caps.can_handle(128 * 1024 * 1024));
    assert!(caps.can_handle(256 * 1024 * 1024));
    assert!(!caps.can_handle(512 * 1024 * 1024));
}

#[test]
fn test_backend_capabilities_estimated_throughput() {
    let available = BackendCapabilities::gpu(true, 256 * 1024 * 1024, 4096, true);
    assert!(available.estimated_throughput(1024) > 0.0);

    let unavailable = BackendCapabilities::gpu(false, 0, 0, false);
    assert_eq!(unavailable.estimated_throughput(1024), 0.0);
}

#[test]
fn test_matmul_op_new() {
    let op = MatMulOp::new(64, 128, 64);
    assert_eq!(op.m, 64);
    assert_eq!(op.k, 128);
    assert_eq!(op.n, 64);
    assert!(!op.trans_a);
    assert!(!op.trans_b);
}

#[test]
fn test_matmul_op_transpose() {
    let op = MatMulOp::new(64, 128, 64).transpose_a().transpose_b();
    assert!(op.trans_a);
    assert!(op.trans_b);
}

#[test]
fn test_matmul_op_output_shape() {
    let op = MatMulOp::new(64, 128, 32);
    assert_eq!(op.output_shape(), (64, 32));
}

#[test]
fn test_matmul_op_flops() {
    let op = MatMulOp::new(64, 128, 64);
    assert_eq!(op.estimated_flops(), 2 * 64 * 128 * 64);
}

#[test]
fn test_matmul_op_memory() {
    let op = MatMulOp::new(64, 128, 64);
    // A: 64*128*4, B: 128*64*4, C: 64*64*4
    let expected = (64 * 128 + 128 * 64 + 64 * 64) * 4;
    assert_eq!(op.memory_requirement(), expected);
}

#[test]
fn test_matmul_op_execute_simd() {
    let op = MatMulOp::new(8, 8, 8);
    let result = op.execute_simd().expect("Should execute");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_softmax_op_new() {
    let op = SoftmaxOp::new(16, 64);
    assert_eq!(op.rows, 16);
    assert_eq!(op.cols, 64);
    assert_eq!(op.temperature, 1.0);
}

#[test]
fn test_softmax_op_temperature() {
    let op = SoftmaxOp::new(16, 64).with_temperature(0.5);
    assert_eq!(op.temperature, 0.5);
}

#[test]
fn test_softmax_op_execute() {
    let op = SoftmaxOp::new(4, 8);
    let result = op.execute_simd().expect("Should execute");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_layer_norm_op_new() {
    let op = LayerNormOp::new(32, 768);
    assert_eq!(op.batch_size, 32);
    assert_eq!(op.hidden_size, 768);
    assert_eq!(op.epsilon, 1e-5);
}

#[test]
fn test_layer_norm_op_epsilon() {
    let op = LayerNormOp::new(32, 768).with_epsilon(1e-6);
    assert_eq!(op.epsilon, 1e-6);
}

#[test]
fn test_layer_norm_op_execute() {
    let op = LayerNormOp::new(4, 64);
    let result = op.execute_simd().expect("Should execute");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_gelu_op_new() {
    let op = GeluOp::new(1024);
    assert_eq!(op.num_elements, 1024);
    assert!(op.fast_approx);
}

#[test]
fn test_gelu_op_exact() {
    let op = GeluOp::new(1024).exact();
    assert!(!op.fast_approx);
}

#[test]
fn test_gelu_op_execute() {
    let op = GeluOp::new(64);
    let result = op.execute_simd().expect("Should execute");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_compute_op_execute_auto() {
    let op = MatMulOp::new(8, 8, 8);
    let result = op.execute(BackendType::Auto).expect("Should execute");
    assert_eq!(result.len(), 64);
}

// =========================================================================
// Additional Coverage Tests (WAPR-QA)
// =========================================================================

#[test]
fn test_backend_type_is_auto() {
    assert!(BackendType::Auto.is_auto());
    assert!(!BackendType::Simd.is_auto());
    assert!(!BackendType::Gpu.is_auto());
    assert!(!BackendType::Cpu.is_auto());
}

#[test]
fn test_backend_type_name() {
    assert_eq!(BackendType::Simd.name(), "SIMD");
    assert_eq!(BackendType::Gpu.name(), "GPU");
    assert_eq!(BackendType::Cpu.name(), "CPU");
    assert_eq!(BackendType::Auto.name(), "Auto");
}

#[test]
fn test_backend_capabilities_default() {
    let caps = BackendCapabilities::default();
    assert_eq!(caps.backend_type, BackendType::Cpu);
    assert!(caps.available);
    assert_eq!(caps.max_parallelism, 1);
}

#[test]
fn test_matmul_op_execute_gpu() {
    let op = MatMulOp::new(8, 8, 8);
    let result = op.execute_gpu().expect("Should execute");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_softmax_op_execute_simd() {
    let op = SoftmaxOp::new(4, 8);
    let result = op.execute_simd().expect("Should execute");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_softmax_op_execute_gpu() {
    let op = SoftmaxOp::new(4, 8);
    let result = op.execute_gpu().expect("Should execute");
    assert_eq!(result.len(), 32);
}

#[test]
fn test_softmax_op_flops() {
    let op = SoftmaxOp::new(4, 8);
    assert!(op.estimated_flops() > 0);
}

#[test]
fn test_softmax_op_memory() {
    let op = SoftmaxOp::new(4, 8);
    let mem = op.memory_requirement();
    assert_eq!(mem, 4 * 8 * 4 * 2); // rows * cols * sizeof(f32) * 2
}

#[test]
fn test_layer_norm_op_execute_gpu() {
    let op = LayerNormOp::new(4, 64);
    let result = op.execute_gpu().expect("Should execute");
    assert_eq!(result.len(), 256);
}

#[test]
fn test_layer_norm_op_flops() {
    let op = LayerNormOp::new(4, 64);
    assert!(op.estimated_flops() > 0);
}

#[test]
fn test_layer_norm_op_memory() {
    let op = LayerNormOp::new(4, 64);
    let mem = op.memory_requirement();
    assert!(mem > 0);
}

#[test]
fn test_gelu_op_execute_gpu() {
    let op = GeluOp::new(64);
    let result = op.execute_gpu().expect("Should execute");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_gelu_op_flops() {
    let op = GeluOp::new(64);
    assert!(op.estimated_flops() > 0);
}

#[test]
fn test_gelu_op_memory() {
    let op = GeluOp::new(64);
    let mem = op.memory_requirement();
    assert_eq!(mem, 64 * 4 * 2); // input + output
}

#[test]
fn test_compute_op_execute_simd_backend() {
    let op = MatMulOp::new(8, 8, 8);
    let result = op.execute(BackendType::Simd).expect("Should execute");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_compute_op_execute_gpu_backend() {
    let op = MatMulOp::new(8, 8, 8);
    let result = op.execute(BackendType::Gpu).expect("Should execute");
    assert_eq!(result.len(), 64);
}

#[test]
fn test_compute_op_execute_cpu_backend() {
    let op = MatMulOp::new(8, 8, 8);
    let result = op.execute(BackendType::Cpu).expect("Should execute");
    assert_eq!(result.len(), 64);
}

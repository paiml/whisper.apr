//! SIMD operations example
//!
//! Demonstrates SIMD-accelerated operations in whisper.apr.
//!
//! Run with: `cargo run --example simd_operations`

use std::time::Instant;
use whisper_apr::simd;

fn main() {
    println!("=== Whisper.apr SIMD Operations Example ===\n");

    // Check SIMD availability
    println!("SIMD backend: {}", simd::backend_name());
    println!("SIMD available: {}", simd::simd_available());
    println!("Best backend: {:?}", simd::best_backend());
    println!();

    // Benchmark dot product
    println!("=== Dot Product ===");
    let sizes = [64, 256, 1024, 4096];

    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.002).cos()).collect();

        // Warm up
        let _ = simd::dot(&a, &b);

        // Benchmark
        let start = Instant::now();
        let iterations = 10000;
        let mut result = 0.0;
        for _ in 0..iterations {
            result = simd::dot(&a, &b);
        }
        let elapsed = start.elapsed();

        println!(
            "  Size {}: result={:.6}, time={:.2}us/iter",
            size,
            result,
            elapsed.as_nanos() as f64 / iterations as f64 / 1000.0
        );
    }
    println!();

    // Demonstrate softmax
    println!("=== Softmax ===");
    let logits = vec![2.0, 1.0, 0.1, -1.0, -2.0];
    println!("  Input:  {:?}", logits);

    let probs = simd::softmax(&logits);
    println!(
        "  Output: {:?}",
        probs
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    let sum: f32 = probs.iter().sum();
    println!("  Sum: {:.6} (should be ~1.0)", sum);
    println!();

    // Demonstrate numerical stability of softmax
    println!("=== Softmax Numerical Stability ===");
    let large_logits = vec![1000.0, 1001.0, 1002.0]; // Would overflow naive impl
    println!("  Input:  {:?}", large_logits);

    let large_probs = simd::softmax(&large_logits);
    println!(
        "  Output: {:?}",
        large_probs
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!("  (No overflow - stable implementation)");
    println!();

    // Demonstrate GELU activation
    println!("=== GELU Activation ===");
    let inputs = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    println!("  Input: {:?}", inputs);

    let outputs = simd::gelu(&inputs);
    println!(
        "  Output: {:?}",
        outputs
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!("  Note: GELU(-x) ~ -0 for negative x, GELU(x) ~ x for positive x");
    println!();

    // Demonstrate layer normalization
    println!("=== Layer Normalization ===");
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0; data.len()]; // Scale
    let beta = vec![0.0; data.len()]; // Shift
    println!("  Input: {:?}", data);

    let normalized = simd::layer_norm(&data, &gamma, &beta, 1e-5);

    // Verify mean ~ 0 and std ~ 1
    let mean = simd::mean(&normalized);
    let std = simd::std_dev(&normalized);

    println!(
        "  Output: {:?}",
        normalized
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!("  Mean: {:.6} (should be ~0)", mean);
    println!("  Std:  {:.6} (should be ~1)", std);
    println!();

    // Matrix multiplication
    println!("=== Matrix Multiplication ===");
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2 matrix
    let c = simd::matmul(&a, &b, 2, 3, 2);
    println!("  A (2x3): {:?}", a);
    println!("  B (3x2): {:?}", b);
    println!("  C = A @ B (2x2): {:?}", c);
    println!();

    // Performance comparison with larger data
    println!("=== Layer Norm Performance ===");
    let sizes = [512, 768, 1024, 1280]; // Common transformer dimensions

    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        let gamma = vec![1.0; size];
        let beta = vec![0.0; size];

        let start = Instant::now();
        let iterations = 10000;
        for _ in 0..iterations {
            let _ = simd::layer_norm(&data, &gamma, &beta, 1e-5);
        }
        let elapsed = start.elapsed();

        println!(
            "  d_model={}: {:.2}us/iter",
            size,
            elapsed.as_nanos() as f64 / iterations as f64 / 1000.0
        );
    }
    println!();

    println!("=== Example Complete ===");
}

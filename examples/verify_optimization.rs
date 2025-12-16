//! Verify trueno optimization is working
//!
//! Tests that the vector-matrix multiply optimization is active

use std::time::Instant;

fn main() {
    println!("=== TRUENO OPTIMIZATION VERIFICATION ===\n");

    // Whisper vocab projection dimensions
    let rows = 1;
    let inner = 384;
    let cols = 51865;

    let a: Vec<f32> = (0..rows * inner).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..inner * cols).map(|i| (i as f32) * 0.0001).collect();

    println!("Test: {}×{} @ {}×{}", rows, inner, inner, cols);
    println!("Total elements: {} million\n", (inner * cols) as f64 / 1e6);

    // Test whisper_apr::simd::matmul
    println!("Method 1: whisper_apr::simd::matmul");

    // Warmup
    for _ in 0..3 {
        let _ = whisper_apr::simd::matmul(&a, &b, rows, inner, cols);
    }

    let start = Instant::now();
    for _ in 0..10 {
        let _ = whisper_apr::simd::matmul(&a, &b, rows, inner, cols);
    }
    let time1 = start.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("  Time: {:.1}ms", time1);

    // Test direct trueno Matrix::matmul
    println!("\nMethod 2: trueno::Matrix::matmul directly");

    use trueno::Matrix;
    let ma = Matrix::from_vec(rows, inner, a.clone()).expect("example assertion");
    let mb = Matrix::from_vec(inner, cols, b.clone()).expect("example assertion");

    // Warmup
    for _ in 0..3 {
        let _ = ma.matmul(&mb);
    }

    let start = Instant::now();
    for _ in 0..10 {
        let _ = ma.matmul(&mb).expect("example assertion");
    }
    let time2 = start.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("  Time: {:.1}ms", time2);

    // Summary
    println!("\n=== RESULTS ===");
    println!("whisper_apr::simd::matmul: {:.1}ms", time1);
    println!("trueno::Matrix::matmul:    {:.1}ms", time2);

    if time2 < 10.0 {
        println!("\n✓ Optimization is WORKING (trueno < 10ms)");
    } else {
        println!("\n✗ Optimization may NOT be working (trueno > 10ms)");
    }

    if time1 > time2 * 2.0 {
        println!(
            "⚠ whisper_apr::simd::matmul has overhead ({}x slower)",
            time1 / time2
        );
        println!("  This is due to a.to_vec() and b.to_vec() allocations in simd::matmul");
    }
}

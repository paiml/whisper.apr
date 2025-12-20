//! Parallel inference example
//!
//! Demonstrates multi-threaded attention head computation using the `parallel` feature.
//!
//! Run with:
//!   cargo run --example parallel_inference --features parallel
//!
//! Compare single-threaded vs multi-threaded:
//!   cargo run --example parallel_inference           # Sequential
//!   cargo run --example parallel_inference --features parallel  # Parallel

use std::time::Instant;

use whisper_apr::parallel::{configure_thread_pool, is_parallel_available, thread_count};

fn main() {
    println!("=== Whisper.apr Parallel Inference Example ===\n");

    // Check if parallel execution is available
    println!("Parallel execution available: {}", is_parallel_available());
    println!("Thread count: {}", thread_count());
    println!();

    // Configure thread pool (simulating --threads CLI flag)
    println!("Configuring thread pool...");

    // Test with different thread counts
    for threads in [1, 2, 4, 8] {
        match configure_thread_pool(Some(threads)) {
            Ok(actual) => {
                println!("  Requested {} threads, got {} threads", threads, actual);
            }
            Err(e) => {
                println!("  Failed to configure {} threads: {}", threads, e);
            }
        }
        // Note: Rayon only allows configuring global pool once
        // Subsequent calls return current thread count
        break;
    }
    println!();

    // Demonstrate parallel_map behavior
    demonstrate_parallel_map();

    println!("\n=== Parallel Inference Configuration ===");
    println!();
    println!("CLI Usage:");
    println!("  whisper-apr transcribe -f audio.wav --threads 4");
    println!();
    println!("Cargo Features:");
    println!("  cargo build --features parallel     # Enable parallel inference");
    println!("  cargo build                         # Sequential fallback");
    println!();
    println!("WASM (requires special build flags):");
    println!("  RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \\");
    println!("    wasm-pack build --features parallel -- -Z build-std=std,panic_abort");
    println!();

    // Show Amdahl's Law analysis
    println!("=== Amdahl's Law Analysis ===");
    println!();
    println!("Whisper attention head parallelization:");
    println!("  Parallel fraction (P): ~25% (attention head computation)");
    println!("  Sequential fraction (S): ~75% (FFN, projections, etc.)");
    println!();
    println!("Expected speedup = 1 / (S + P/N)");
    println!();
    for threads in [1, 2, 4, 8] {
        let s = 0.75;
        let p = 0.25;
        let speedup = 1.0 / (s + p / threads as f64);
        println!("  {} threads: {:.2}x speedup", threads, speedup);
    }
    println!();
    println!("Measured results (1.5s audio):");
    println!("  1 thread:  6.61x RTF (baseline)");
    println!("  2 threads: 5.98x RTF (1.11x speedup)");
    println!("  4 threads: 5.42x RTF (1.22x speedup)");

    println!("\n=== Example Complete ===");
}

fn demonstrate_parallel_map() {
    use whisper_apr::parallel::parallel_map;

    println!("Demonstrating parallel_map...");

    // Simulate attention head computation
    let n_heads = 6; // Whisper tiny has 6 heads
    let work_per_head = 1_000_000; // Simulated work

    let start = Instant::now();
    let results: Vec<u64> = parallel_map(0..n_heads, |head| {
        // Simulate attention computation for each head
        let mut sum: u64 = 0;
        for i in 0..work_per_head {
            sum = sum.wrapping_add((head as u64).wrapping_mul(i as u64));
        }
        sum
    });
    let elapsed = start.elapsed();

    println!("  Computed {} attention heads in {:?}", n_heads, elapsed);
    println!("  Results (checksums): {:?}", results);

    #[cfg(feature = "parallel")]
    println!("  Mode: Parallel (rayon)");

    #[cfg(not(feature = "parallel"))]
    println!("  Mode: Sequential (fallback)");
}

#![allow(clippy::unwrap_used)]
//! Encoder profiling for WAPR-PERF-011 verification matrix
//!
//! Implements Dr. Popper's falsification methodology:
//! 1. Brick Profiling - kernel execution times
//! 2. Tile Analysis - occupancy for batch=6
//! 3. Step Function Tracing - TRANSFORMER_BLOCK gaps
//! 4. Layer Tracing - Σ(Kernel) vs Layer time

#[cfg(feature = "realizar-gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::inference_trace::TraceConfig;
    use trueno_gpu::memory::resident::{
        kernel_cache_hits, kernel_cache_misses, reset_transfer_counters, TransferStats,
    };

    // Load audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();
    println!(
        "Audio samples: {} ({:.2}s)",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    // Load model
    println!("\n=== Loading Model ===");
    let load_start = Instant::now();
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let model = WhisperApr::load_from_apr(&model_bytes)?;
    println!(
        "CPU model loaded in {:.1}ms",
        load_start.elapsed().as_millis()
    );

    // Compute mel on CPU first
    let mel_start = Instant::now();
    let mel = model.compute_mel(&samples)?;
    println!(
        "Mel computed in {:.1}ms ({} frames)",
        mel_start.elapsed().as_millis(),
        mel.len() / 80
    );

    // Convert to CUDA
    println!("\n=== Converting to CUDA ===");
    let cuda_start = Instant::now();
    let mut cuda_model = model.into_cuda(0)?;
    println!(
        "CUDA model created in {:.1}ms",
        cuda_start.elapsed().as_millis()
    );

    // Upload weights
    let upload_start = Instant::now();
    cuda_model.upload_encoder_weights()?;
    println!(
        "Weights uploaded in {:.1}ms",
        upload_start.elapsed().as_millis()
    );
    println!("Device: {}", cuda_model.device_name());

    // Enable tracing (WAPR-PERF-004)
    cuda_model.enable_tracing(TraceConfig::enabled());
    println!("\n=== Tracing Enabled ===");

    // Reset kernel cache counters
    reset_transfer_counters();

    // Warmup run (compiles kernels)
    println!("\n=== Warmup Run (Kernel Compilation) ===");
    let warmup_start = Instant::now();
    let _ = cuda_model.encode_gpu_total_offload(&mel)?;
    let warmup_time = warmup_start.elapsed();
    println!(
        "Warmup: {:.1}ms (includes kernel JIT)",
        warmup_time.as_millis()
    );

    // Get cache stats after warmup
    let hits_after_warmup = kernel_cache_hits();
    let misses_after_warmup = kernel_cache_misses();
    println!(
        "Kernel cache: {} hits, {} misses (compiles)",
        hits_after_warmup, misses_after_warmup
    );

    // Reset for timed runs
    cuda_model.reset_tracer();
    reset_transfer_counters();

    // === VERIFICATION MATRIX ===
    println!("\n{}", "=".repeat(60));
    println!("=== WAPR-PERF-011 VERIFICATION MATRIX ===");
    println!("{}", "=".repeat(60));

    // Timed runs with tracing
    let n_runs = 5;
    let mut times = Vec::with_capacity(n_runs);
    let mut layer_times: Vec<Vec<u128>> = Vec::new();

    for i in 0..n_runs {
        cuda_model.reset_tracer();
        cuda_model.enable_tracing(TraceConfig::enabled());

        let start = Instant::now();
        let features = cuda_model.encode_gpu_total_offload(&mel)?;
        let elapsed = start.elapsed();
        times.push(elapsed.as_micros() as f64);

        // Collect per-layer times from tracer
        let events = cuda_model.tracer().events();
        let run_layer_times: Vec<u128> = events
            .iter()
            .filter(|e| e.step.name() == "TRANSFORMER_BLOCK")
            .map(|e| e.duration_us as u128)
            .collect();

        println!(
            "  Run {}: {:.1}ms ({} features, {} trace events)",
            i + 1,
            elapsed.as_millis(),
            features.len(),
            events.len()
        );

        layer_times.push(run_layer_times);
    }

    // === 1. BRICK PROFILING (Kernel Execution) ===
    println!("\n--- 1. BRICK PROFILING (Kernel Execution) ---");
    let avg_time_us = times.iter().sum::<f64>() / n_runs as f64;
    let min_time_us = times.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("Total encoder time:");
    println!(
        "  Average: {:.1}ms ({:.0}µs)",
        avg_time_us / 1000.0,
        avg_time_us
    );
    println!(
        "  Best:    {:.1}ms ({:.0}µs)",
        min_time_us / 1000.0,
        min_time_us
    );

    // Kernel cache analysis
    let final_hits = kernel_cache_hits();
    let final_misses = kernel_cache_misses();
    let hits_per_run = (final_hits - hits_after_warmup) / n_runs as u64;
    println!("  Cache hits/run: {} (kernel reuse)", hits_per_run);

    // === 2. TILE ANALYSIS (Occupancy) ===
    println!("\n--- 2. TILE ANALYSIS (Occupancy) ---");
    println!("Batched GEMM configuration:");
    println!("  batch = 6 heads (Whisper tiny)");
    println!("  WMMA tile = 16x16x16");
    println!("  Grid: (n+15)/16 × (m+15)/16 × batch");
    println!("  Block: 32 threads (1 warp) per tile");
    println!("");
    println!("Theoretical occupancy analysis:");
    println!("  6 batches × ceil(1500/16) × ceil(64/16) = 6 × 94 × 4 = 2256 blocks");
    println!("  RTX 4090 has 128 SMs, max 32 blocks/SM");
    println!("  Expected: ~18 waves (2256 / 128 = 17.6)");
    println!("  Latency hiding: INSUFFICIENT for memory-bound workload");

    // === 3. STEP FUNCTION TRACING (State Gaps) ===
    println!("\n--- 3. STEP FUNCTION TRACING (State Gaps) ---");
    let last_events = cuda_model.tracer().events();

    let mut step_times: std::collections::HashMap<&str, Vec<u128>> =
        std::collections::HashMap::new();
    for event in last_events.iter() {
        step_times
            .entry(event.step.name())
            .or_default()
            .push(event.duration_us as u128);
    }

    let mut total_traced_us: u128 = 0;
    for (step, times) in &step_times {
        let sum: u128 = times.iter().sum();
        total_traced_us += sum;
        let avg = sum as f64 / times.len() as f64;
        println!(
            "  {}: {} events, total {:.1}ms, avg {:.0}µs/event",
            step,
            times.len(),
            sum as f64 / 1000.0,
            avg
        );
    }

    println!("");
    println!(
        "Total traced time: {:.1}ms",
        total_traced_us as f64 / 1000.0
    );
    println!("Wall clock time:   {:.1}ms", min_time_us / 1000.0);

    let unaccounted_us = min_time_us as i128 - total_traced_us as i128;
    let unaccounted_pct = (unaccounted_us as f64 / min_time_us) * 100.0;
    println!(
        "DARK MATTER:       {:.1}ms ({:.1}%)",
        unaccounted_us as f64 / 1000.0,
        unaccounted_pct
    );

    // === 4. LAYER TRACING (Sum of Parts) ===
    println!("\n--- 4. LAYER TRACING (Σ Kernel vs Layer) ---");
    if let Some(last_layer_times) = layer_times.last() {
        let sum_layers: u128 = last_layer_times.iter().sum();
        let transformer_events: Vec<_> = last_events
            .iter()
            .filter(|e| e.step.name() == "TRANSFORMER_BLOCK")
            .collect();

        println!("  TRANSFORMER_BLOCK events: {}", transformer_events.len());
        println!("  Σ(layer times): {:.1}ms", sum_layers as f64 / 1000.0);

        if !last_layer_times.is_empty() {
            let avg_layer = sum_layers as f64 / last_layer_times.len() as f64;
            println!("  Avg per layer: {:.0}µs", avg_layer);

            // Check for variance (hidden sync would cause spikes)
            let max_layer = last_layer_times.iter().max().unwrap_or(&0);
            let min_layer = last_layer_times.iter().min().unwrap_or(&0);
            let variance_ratio = *max_layer as f64 / (*min_layer as f64 + 1.0);
            println!("  Layer variance: {:.1}x (max/min)", variance_ratio);

            if variance_ratio > 2.0 {
                println!("  ⚠️  HIGH VARIANCE: Possible hidden synchronization");
            }
        }
    }

    // === FALSIFICATION VERDICT ===
    println!("\n{}", "=".repeat(60));
    println!("=== FALSIFICATION VERDICT ===");
    println!("{}", "=".repeat(60));

    let target_us = 166_000.0; // 166ms target
    let gap = min_time_us / target_us;

    println!("Current: {:.1}ms", min_time_us / 1000.0);
    println!("Target:  166ms (2x whisper.cpp @ 83ms)");
    println!("Gap:     {:.1}x", gap);
    println!("");

    if unaccounted_pct.abs() > 20.0 {
        println!("🔍 HYPOTHESIS: \"Hidden Synchronization\" (Point 149)");
        println!(
            "   {:.1}% of time is unaccounted for in trace events.",
            unaccounted_pct.abs()
        );
        println!("   Investigate: cudaDeviceSynchronize, memory allocations in loop");
    } else if gap > 2.0 {
        println!("🔍 HYPOTHESIS: \"Memory Bound\" (Point 144)");
        println!("   Compute is traced, but still {:.1}x from target.", gap);
        println!("   Investigate: Roofline position, memory bandwidth utilization");
    }

    Ok(())
}

#[cfg(not(feature = "realizar-gpu"))]
fn main() {
    eprintln!("This example requires the 'realizar-gpu' feature.");
    eprintln!("Run with: cargo run --release --features realizar-gpu --example profile_encoder");
}

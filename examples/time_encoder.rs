#![allow(clippy::unwrap_used)]
//! Encoder timing benchmark (WAPR-PERF-010)

#[cfg(feature = "realizar-gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load audio first (need it for mel computation)
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
    println!("\nLoading model...");
    let load_start = Instant::now();
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let model = WhisperApr::load_from_apr(&model_bytes)?;
    println!(
        "CPU model loaded in {:.1}ms",
        load_start.elapsed().as_millis()
    );

    // Compute mel spectrogram on CPU model (before consuming it)
    let mel_start = Instant::now();
    let mel = model.compute_mel(&samples)?;
    println!(
        "Mel computed in {:.1}ms ({} frames)",
        mel_start.elapsed().as_millis(),
        mel.len() / 80
    );

    // Convert to CUDA model (consumes CPU model)
    println!("\nConverting to CUDA model...");
    let cuda_start = Instant::now();
    let mut cuda_model = model.into_cuda(0)?;
    println!(
        "CUDA model created in {:.1}ms",
        cuda_start.elapsed().as_millis()
    );

    // Upload weights to GPU
    println!("Uploading encoder weights...");
    let upload_start = Instant::now();
    cuda_model.upload_encoder_weights()?;
    println!(
        "Weights uploaded in {:.1}ms",
        upload_start.elapsed().as_millis()
    );
    println!("Device: {}", cuda_model.device_name());

    // Memory info
    let (free, total) = cuda_model.memory_info();
    println!(
        "GPU Memory: {:.0}MB free / {:.0}MB total",
        free as f64 / 1024.0 / 1024.0,
        total as f64 / 1024.0 / 1024.0
    );

    // Warmup
    println!("\nWarmup (GPU Total Offload encoder)...");
    let _ = cuda_model.encode_gpu_total_offload(&mel);

    // Time GPU encoder with total offload
    println!("\n=== GPU Encoder Total Offload timing (5 runs) ===");
    let n_runs = 5;
    let mut times = Vec::with_capacity(n_runs);
    for i in 0..n_runs {
        let start = Instant::now();
        let features = cuda_model.encode_gpu_total_offload(&mel)?;
        let elapsed = start.elapsed();
        times.push(elapsed.as_millis() as f64);
        println!(
            "  Run {}: {:.1}ms ({} features)",
            i + 1,
            elapsed.as_millis(),
            features.len()
        );
    }

    let avg = times.iter().sum::<f64>() / n_runs as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("\n=== Summary ===");
    println!("Average:    {:.1}ms", avg);
    println!("Best:       {:.1}ms", min);
    println!("Target:     166ms (2x whisper.cpp @ 83ms)");
    println!("Gap:        {:.1}x from target", avg / 166.0);

    Ok(())
}

#[cfg(not(feature = "realizar-gpu"))]
fn main() {
    eprintln!("This example requires the 'realizar-gpu' feature.");
    eprintln!("Run with: cargo run --release --features realizar-gpu --example time_encoder");
}

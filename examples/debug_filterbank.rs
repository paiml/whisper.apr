//! Debug Filterbank Weights
//!
//! Compare our loaded filterbank with ground truth to find scale mismatch.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FILTERBANK DEBUG ===\n");

    // Ground truth from test_data/ref_b_filterbank.json (whisper.cpp)
    let gt_max = 0.0259_f32;
    let gt_mean = 0.000124_f32;
    let gt_std = 0.00114_f32;
    let gt_nonzero = 391_usize;
    let gt_shape = (80, 201);

    println!("Ground Truth Filterbank (whisper.cpp):");
    println!("  shape: {:?}", gt_shape);
    println!("  max:   {:.6}", gt_max);
    println!("  mean:  {:.6}", gt_mean);
    println!("  std:   {:.6}", gt_std);
    println!("  nonzero: {}", gt_nonzero);

    // Load model and get filterbank
    let model_path = Path::new("models/whisper-tiny.apr");
    if !model_path.exists() {
        println!("\nERROR: Model not found");
        return Ok(());
    }

    let model_bytes = std::fs::read(model_path)?;
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Get filterbank stats (we need to access the internal filterbank)
    // Since we can't directly access it, let's use the mel computation
    // and work backwards

    // Actually, let's compute our own filterbank with the same parameters
    // and compare with what the model uses
    let our_fb = whisper_apr::audio::MelFilterbank::new(80, 400, 16000);
    let our_filters = our_fb.filters();

    let our_max = our_filters
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let our_min = our_filters.iter().cloned().fold(f32::INFINITY, f32::min);
    let our_mean = our_filters.iter().sum::<f32>() / our_filters.len() as f32;
    let our_var = our_filters
        .iter()
        .map(|x| (x - our_mean).powi(2))
        .sum::<f32>()
        / our_filters.len() as f32;
    let our_std = our_var.sqrt();
    let our_nonzero = our_filters.iter().filter(|&&x| x > 1e-10).count();

    println!("\nOur Computed Filterbank (MelFilterbank::new):");
    println!("  shape: ({}, {})", our_fb.n_mels(), our_fb.n_freqs());
    println!("  max:   {:.6}", our_max);
    println!("  mean:  {:.6}", our_mean);
    println!("  std:   {:.6}", our_std);
    println!("  min:   {:.6}", our_min);
    println!("  nonzero: {}", our_nonzero);

    // Compare
    println!("\n=== COMPARISON ===");
    let max_ratio = our_max / gt_max;
    let mean_ratio = our_mean / gt_mean;
    println!("Max ratio (our/GT): {:.2}x", max_ratio);
    println!("Mean ratio (our/GT): {:.2}x", mean_ratio);

    if max_ratio > 10.0 {
        println!("\n⚠️  Our filterbank weights are {:.0}x larger!", max_ratio);
        println!("This could explain the mel spectrogram scale issue.");
        println!("\nPossible causes:");
        println!("  1. Missing Slaney normalization in our filterbank computation");
        println!("  2. Different frequency-to-bin mapping");
        println!("  3. Different triangle filter normalization");
    }

    // Check if we're using the model's filterbank or computing our own
    println!("\n=== FILTERBANK SOURCE CHECK ===");
    println!("whisper_apr::WhisperApr should use filterbank from .apr file");
    println!("But MelFilterbank::new() computes its own filterbank");
    println!("\nThe model might be using MelFilterbank::from_apr_data() with");
    println!("pre-computed slaney-normalized weights from the model file.");

    // Slaney normalization explanation
    println!("\n=== SLANEY NORMALIZATION ===");
    println!("Slaney ('librosa') normalization divides each filter by its bandwidth.");
    println!("This makes filters covering wider frequency ranges have smaller weights.");
    println!("Without Slaney normalization, our filters are NOT scaled correctly.");

    // Check if applying Slaney normalization would fix it
    println!("\n=== MANUAL SLANEY CHECK ===");

    // Mel frequencies for the filterbank
    let n_mels = 80;
    let n_fft = 400;
    let sample_rate = 16000;
    let n_freqs = n_fft / 2 + 1;

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate as f32 / 2.0);

    // Get center frequencies
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Compute bandwidths (difference between successive center frequencies)
    let bandwidths: Vec<f32> = (0..n_mels)
        .map(|i| hz_points[i + 2] - hz_points[i])
        .collect();

    let avg_bandwidth = bandwidths.iter().sum::<f32>() / bandwidths.len() as f32;
    let max_bandwidth = bandwidths.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_bandwidth = bandwidths.iter().cloned().fold(f32::INFINITY, f32::min);

    println!("Bandwidth stats:");
    println!("  avg: {:.2} Hz", avg_bandwidth);
    println!("  min: {:.2} Hz", min_bandwidth);
    println!("  max: {:.2} Hz", max_bandwidth);

    // What Slaney normalization factor would be
    println!("\nSlaney normalization divides by (2 / bandwidth)");
    println!(
        "For avg bandwidth {:.2} Hz, scale factor ≈ {:.4}",
        avg_bandwidth,
        2.0 / avg_bandwidth
    );

    // Scale our max by typical Slaney factor
    let slaney_scale = 2.0 / avg_bandwidth;
    println!("\nIf we multiply our max by Slaney scale:");
    println!(
        "  {:.6} * {:.4} = {:.6}",
        our_max,
        slaney_scale,
        our_max * slaney_scale
    );
    println!("  GT max: {:.6}", gt_max);

    Ok(())
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

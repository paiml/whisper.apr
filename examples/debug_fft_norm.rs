//! Debug FFT normalization difference

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

fn main() {
    println!("=== FFT NORMALIZATION DEBUG ===\n");

    let n_fft = 400;

    // Create a simple test signal - sine wave
    let freq = 1000.0; // Hz
    let sample_rate = 16000.0;
    let samples: Vec<f32> = (0..n_fft)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect();

    // Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
        .collect();

    // Windowed samples
    let windowed: Vec<f32> = samples
        .iter()
        .zip(window.iter())
        .map(|(s, w)| s * w)
        .collect();

    // FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut fft_input: Vec<Complex<f32>> = windowed.iter().map(|&s| Complex::new(s, 0.0)).collect();
    fft.process(&mut fft_input);

    // Power spectrum (unnormalized)
    let power_unnorm: Vec<f32> = fft_input.iter().map(|c| c.norm_sqr()).collect();

    // Power spectrum (normalized by N)
    let power_norm_n: Vec<f32> = fft_input
        .iter()
        .map(|c| c.norm_sqr() / (n_fft as f32))
        .collect();

    // Power spectrum (normalized by N²)
    let power_norm_n2: Vec<f32> = fft_input
        .iter()
        .map(|c| c.norm_sqr() / ((n_fft * n_fft) as f32))
        .collect();

    // Find peak (should be near bin 25 for 1kHz at 16kHz sample rate with n_fft=400)
    // bin = freq * n_fft / sample_rate = 1000 * 400 / 16000 = 25
    let peak_bin = 25;

    println!("Peak power at bin {} (1kHz):", peak_bin);
    println!("  Unnormalized:     {:.6}", power_unnorm[peak_bin]);
    println!("  Normalized by N:  {:.6}", power_norm_n[peak_bin]);
    println!("  Normalized by N²: {:.6}", power_norm_n2[peak_bin]);

    println!("\nLog10 of peak power:");
    println!("  Unnormalized:     {:+.6}", power_unnorm[peak_bin].log10());
    println!("  Normalized by N:  {:+.6}", power_norm_n[peak_bin].log10());
    println!(
        "  Normalized by N²: {:+.6}",
        power_norm_n2[peak_bin].log10()
    );

    println!("\nDifferences:");
    println!("  log10(N) = {:.4}", (n_fft as f32).log10());
    println!("  2*log10(N) = {:.4}", 2.0 * (n_fft as f32).log10());

    // Check total energy
    let sum_unnorm: f32 = power_unnorm.iter().sum();
    let sum_norm_n: f32 = power_norm_n.iter().sum();
    let sum_norm_n2: f32 = power_norm_n2.iter().sum();

    println!("\nTotal power (sum):");
    println!("  Unnormalized:     {:.6}", sum_unnorm);
    println!("  Normalized by N:  {:.6}", sum_norm_n);
    println!("  Normalized by N²: {:.6}", sum_norm_n2);

    // Parseval's theorem: sum of |x|² in time = (1/N) * sum of |X|² in freq
    let time_energy: f32 = windowed.iter().map(|x| x * x).sum();
    println!("\nParseval check:");
    println!("  Time domain energy: {:.6}", time_energy);
    println!("  Freq/N:             {:.6} (should match)", sum_norm_n);
}

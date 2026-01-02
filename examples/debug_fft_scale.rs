//! Debug FFT Scaling
//!
//! Compare FFT output magnitude with expected values to find missing scale factor.
//! Uses APR-VERIFY ground truth (NO PYTHON).

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

fn main() {
    println!("=== FFT SCALE DEBUG ===\n");

    // Test with a simple sine wave where we know the expected FFT magnitude
    let n_fft = 400;
    let sample_rate = 16000.0;
    let freq = 1000.0; // 1kHz tone

    // Generate exactly one FFT window of sine wave
    let audio: Vec<f32> = (0..n_fft)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect();

    // Apply Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / n_fft as f32).cos()))
        .collect();

    let windowed: Vec<f32> = audio
        .iter()
        .zip(window.iter())
        .map(|(a, w)| a * w)
        .collect();

    // Compute FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut fft_input: Vec<Complex<f32>> = windowed.iter().map(|&x| Complex::new(x, 0.0)).collect();

    fft.process(&mut fft_input);

    // Find the bin with maximum magnitude
    let n_freqs = n_fft / 2 + 1;
    let magnitudes: Vec<f32> = fft_input.iter().take(n_freqs).map(|c| c.norm()).collect();
    let powers: Vec<f32> = fft_input
        .iter()
        .take(n_freqs)
        .map(|c| c.norm_sqr())
        .collect();

    let max_mag_idx = magnitudes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let freq_per_bin = sample_rate / n_fft as f32;
    let peak_freq = max_mag_idx as f32 * freq_per_bin;

    println!("Input: 1kHz sine wave, {} samples", n_fft);
    println!("Peak bin: {} (frequency: {:.1} Hz)", max_mag_idx, peak_freq);
    println!("Peak magnitude: {:.4}", magnitudes[max_mag_idx]);
    println!("Peak power: {:.4}", powers[max_mag_idx]);
    println!("log10(power): {:.4}", powers[max_mag_idx].log10());

    // For a Hann-windowed sine wave, the expected DFT magnitude is approximately N/4
    // (due to windowing reducing the effective energy)
    let expected_mag_unwindowed = n_fft as f32 / 2.0; // Half due to being split between +/- freq
    let expected_mag_windowed = expected_mag_unwindowed * 0.5; // Hann window has gain of 0.5

    println!("\nExpected values:");
    println!(
        "  Unwindowed sine DFT magnitude: ~{:.1}",
        expected_mag_unwindowed
    );
    println!("  Hann-windowed magnitude: ~{:.1}", expected_mag_windowed);

    // What torch.stft produces
    println!("\n=== TORCH.STFT COMPARISON ===");
    println!("torch.stft with normalized=False gives raw DFT output (same as rustfft)");
    println!("torch.stft with normalized='window' normalizes by window.sum()");

    let window_sum: f32 = window.iter().sum();
    println!("\nOur window sum: {:.4}", window_sum);
    println!(
        "If we normalize by window sum: {:.4}",
        magnitudes[max_mag_idx] / window_sum
    );
    println!(
        "Power normalized by window_sum^2: {:.6}",
        powers[max_mag_idx] / (window_sum * window_sum)
    );

    // Whisper's specific normalization
    println!("\n=== WHISPER MEL NORMALIZATION ===");

    // In HuggingFace Whisper, the mel spectrogram uses:
    // magnitudes = stft.abs() ** 2  (power spectrum)
    // mel_spec = filters @ magnitudes
    //
    // The filters are slaney-normalized, which affects the scale.
    // But importantly, the log values should be in a certain range.

    // From our GT data:
    // GT max mel energy → log10 = 0.9368 → mel_energy = 8.64
    // Our max mel energy → log10 = 2.675 → mel_energy = 473
    // Ratio: 473 / 8.64 = 54.7

    println!(
        "GT max mel_energy (10^0.9368) = {:.2}",
        10.0_f32.powf(0.9368)
    );
    println!(
        "Our max mel_energy (10^2.675) = {:.2}",
        10.0_f32.powf(2.675)
    );
    println!(
        "Ratio: {:.1}x",
        10.0_f32.powf(2.675) / 10.0_f32.powf(0.9368)
    );

    // Possible scale factors
    println!("\n=== POSSIBLE SCALE FACTORS ===");
    println!("N_FFT = {}", n_fft);
    println!("N_FFT^2 = {}", n_fft * n_fft);
    println!("sqrt(N_FFT) = {:.2}", (n_fft as f32).sqrt());

    // If we're missing a 1/N normalization on the FFT:
    // Our values would be N times larger in magnitude, N^2 times larger in power
    // log10(N^2) = 2 * log10(N) = 2 * log10(400) = 5.2
    // But our difference is 1.74, which is log10(55)

    // What about the filterbank normalization?
    // Slaney normalization divides each filter by its bandwidth
    // This typically reduces the filter weights

    println!(
        "\nlog10(55) = {:.4} (matches our 1.74 difference)",
        55.0_f32.log10()
    );
    println!("This could be from:");
    println!("  1. Missing filterbank weight normalization");
    println!("  2. Different FFT normalization convention");
    println!("  3. Different power spectrum computation");

    // Try computing what scale factor would fix it
    let scale_factor = 10.0_f32.powf(1.74); // ~55
    println!(
        "\nTo fix: divide mel_energy by {:.1} before log10",
        scale_factor
    );
    println!("Or: subtract {:.4} from log10(mel_energy)", 1.74);
    println!(
        "After (x+4)/4 normalization, this means subtract {:.4}",
        1.74 / 4.0
    );
}

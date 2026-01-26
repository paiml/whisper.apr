//! Tests for audio resampling

use super::*;

// =========================================================================
// Construction Tests
// =========================================================================

#[test]
fn test_resampler_new() {
    let resampler = SincResampler::new(44100, 16000);
    assert!(resampler.is_ok());
    let r = resampler.expect("resampler should be valid");
    assert_eq!(r.source_rate(), 44100);
    assert_eq!(r.target_rate(), 16000);
}

#[test]
fn test_resampler_with_params() {
    let resampler = SincResampler::with_params(48000, 16000, 32, 8.0);
    assert!(resampler.is_ok());
    let r = resampler.expect("resampler should be valid");
    assert_eq!(r.kernel_half_len(), 32);
}

#[test]
fn test_resampler_invalid_source_rate() {
    let resampler = SincResampler::new(0, 16000);
    assert!(resampler.is_err());
    let err = resampler.expect_err("expected error for invalid source rate");
    assert!(matches!(err, WhisperError::Audio(_)));
}

#[test]
fn test_resampler_invalid_target_rate() {
    let resampler = SincResampler::new(44100, 0);
    assert!(resampler.is_err());
}

#[test]
fn test_resampler_invalid_kernel_half_len() {
    let resampler = SincResampler::with_params(44100, 16000, 0, 6.0);
    assert!(resampler.is_err());
}

#[test]
fn test_resampler_ratio() {
    let resampler = SincResampler::new(48000, 16000).expect("valid");
    let expected_ratio = 16000.0 / 48000.0;
    assert!((resampler.ratio() - expected_ratio).abs() < 1e-10);
}

// =========================================================================
// Resampling Tests
// =========================================================================

#[test]
fn test_resample_same_rate() {
    let resampler = SincResampler::new(16000, 16000).expect("valid resampler");
    let audio = vec![1.0, 2.0, 3.0, 4.0];
    let result = resampler.resample(&audio);
    assert!(result.is_ok());
    assert_eq!(result.expect("valid result"), audio);
}

#[test]
fn test_resample_empty_audio() {
    let resampler = SincResampler::new(44100, 16000).expect("valid resampler");
    let audio: Vec<f32> = vec![];
    let result = resampler.resample(&audio);
    assert!(result.is_err());
}

#[test]
fn test_resample_downsample_44100_to_16000() {
    let resampler = SincResampler::new(44100, 16000).expect("valid resampler");
    let audio = vec![0.5; 44100]; // 1 second at 44.1kHz
    let result = resampler.resample(&audio);
    assert!(result.is_ok());
    let output = result.expect("valid result");
    // Output should be approximately 16000 samples
    assert!(output.len() >= 15900 && output.len() <= 16100);
}

#[test]
fn test_resample_downsample_48000_to_16000() {
    let resampler = SincResampler::new(48000, 16000).expect("valid resampler");
    let audio = vec![0.5; 48000]; // 1 second at 48kHz
    let result = resampler.resample(&audio);
    assert!(result.is_ok());
    let output = result.expect("valid result");
    // Output should be approximately 16000 samples (3x downsample)
    assert!(output.len() >= 15900 && output.len() <= 16100);
}

#[test]
fn test_resample_upsample_8000_to_16000() {
    let resampler = SincResampler::new(8000, 16000).expect("valid resampler");
    let audio = vec![0.5; 8000]; // 1 second at 8kHz
    let result = resampler.resample(&audio);
    assert!(result.is_ok());
    let output = result.expect("valid result");
    // Output should be approximately 16000 samples (2x upsample)
    assert!(output.len() >= 15900 && output.len() <= 16100);
}

#[test]
fn test_resample_preserves_dc_signal() {
    let resampler = SincResampler::new(44100, 16000).expect("valid");
    let dc_value = 0.5_f32;
    let audio = vec![dc_value; 4410]; // 100ms at 44.1kHz
    let output = resampler.resample(&audio).expect("valid");

    // DC signal should be approximately preserved (within some tolerance due to edge effects)
    let mid_samples: Vec<f32> = output
        .iter()
        .skip(output.len() / 4)
        .take(output.len() / 2)
        .copied()
        .collect();
    let avg: f32 = mid_samples.iter().sum::<f32>() / mid_samples.len() as f32;
    assert!(
        (avg - dc_value).abs() < 0.1,
        "DC signal not preserved: expected {}, got {}",
        dc_value,
        avg
    );
}

#[test]
fn test_resample_sine_wave_downsample() {
    let resampler = SincResampler::new(48000, 16000).expect("valid");
    let freq = 440.0_f32; // 440 Hz sine wave
    let duration = 0.1; // 100ms
    let n_samples = (48000.0 * duration) as usize;

    // Generate sine wave
    let audio: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 48000.0).sin())
        .collect();

    let output = resampler.resample(&audio).expect("valid");

    // Output should have correct length
    let expected_len = (16000.0 * duration) as usize;
    assert!(
        (output.len() as i32 - expected_len as i32).abs() <= 2,
        "Expected ~{} samples, got {}",
        expected_len,
        output.len()
    );

    // Verify output amplitude is reasonable (not all zeros)
    let max_amp = output.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    assert!(max_amp > 0.1, "Output amplitude too low: {}", max_amp);
}

#[test]
fn test_resample_sine_wave_upsample() {
    let resampler = SincResampler::new(16000, 48000).expect("valid");
    let freq = 440.0_f32;
    let duration = 0.1;
    let n_samples = (16000.0 * duration) as usize;

    let audio: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 16000.0).sin())
        .collect();

    let output = resampler.resample(&audio).expect("valid");

    // Output should have correct length
    let expected_len = (48000.0 * duration) as usize;
    assert!(
        (output.len() as i32 - expected_len as i32).abs() <= 2,
        "Expected ~{} samples, got {}",
        expected_len,
        output.len()
    );
}

// =========================================================================
// Bessel Function Tests
// =========================================================================

#[test]
fn test_bessel_i0_at_zero() {
    let result = bessel_i0(0.0);
    assert!((result - 1.0).abs() < 1e-10, "I0(0) should be 1");
}

#[test]
fn test_bessel_i0_known_values() {
    // Known values from mathematical tables
    // I0(1) ≈ 1.2660658777520084
    let result = bessel_i0(1.0);
    assert!(
        (result - 1.2660658777520084).abs() < 1e-10,
        "I0(1) incorrect: {}",
        result
    );

    // I0(2) ≈ 2.2795853023360673
    let result = bessel_i0(2.0);
    assert!(
        (result - 2.2795853023360673).abs() < 1e-10,
        "I0(2) incorrect: {}",
        result
    );
}

#[test]
fn test_bessel_i0_symmetry() {
    // I0 is even function: I0(x) = I0(-x)
    let x = 3.5;
    let pos = bessel_i0(x);
    let neg = bessel_i0(-x);
    assert!((pos - neg).abs() < 1e-10);
}

// =========================================================================
// Kaiser Window Tests
// =========================================================================

#[test]
fn test_kaiser_window_center() {
    let resampler = SincResampler::new(44100, 16000).expect("valid");
    // Kaiser window should be 1.0 at center (x=0)
    let val = resampler.kaiser_window(0.0);
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn test_kaiser_window_edges() {
    let resampler = SincResampler::new(44100, 16000).expect("valid");
    // Kaiser window should be small at edges (x = ±1)
    // With beta=6.0: I0(0)/I0(6) = 1/67.23 ≈ 0.0149
    let val_pos = resampler.kaiser_window(1.0);
    let val_neg = resampler.kaiser_window(-1.0);
    assert!(
        val_pos < 0.02,
        "Window should be small at edge, got {}",
        val_pos
    );
    assert!(
        (val_pos - val_neg).abs() < 1e-10,
        "Window should be symmetric"
    );
}

#[test]
fn test_kaiser_window_outside_range() {
    let resampler = SincResampler::new(44100, 16000).expect("valid");
    // Kaiser window should be 0 outside [-1, 1]
    let val = resampler.windowed_sinc(100.0, 1.0);
    assert!((val).abs() < 1e-10);
}

// =========================================================================
// Sinc Function Tests
// =========================================================================

#[test]
fn test_windowed_sinc_at_zero() {
    let resampler = SincResampler::new(44100, 16000).expect("valid");
    // sinc(0) = 1
    let val = resampler.windowed_sinc(0.0, 1.0);
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn test_windowed_sinc_at_integers() {
    let resampler = SincResampler::with_params(44100, 16000, 32, 6.0).expect("valid");
    // sinc(n) = 0 for non-zero integers n
    for n in 1..10 {
        let val = resampler.windowed_sinc(n as f64, 1.0);
        assert!(
            val.abs() < 0.1,
            "sinc({}) should be near zero, got {}",
            n,
            val
        );
    }
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_resample_single_sample() {
    let resampler = SincResampler::new(44100, 16000).expect("valid");
    let audio = vec![0.5_f32];
    let output = resampler.resample(&audio).expect("valid");
    assert!(!output.is_empty());
}

#[test]
fn test_resample_very_short_audio() {
    let resampler = SincResampler::new(44100, 16000).expect("valid");
    let audio = vec![0.5_f32; 10];
    let output = resampler.resample(&audio).expect("valid");
    assert!(!output.is_empty());
}

#[test]
fn test_resample_high_frequency_rejection() {
    // When downsampling, frequencies above Nyquist should be attenuated
    let resampler = SincResampler::new(48000, 16000).expect("valid");

    // Generate 10kHz sine wave (above 8kHz Nyquist of 16kHz output)
    let freq = 10000.0_f32;
    let n_samples = 4800; // 100ms
    let audio: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 48000.0).sin())
        .collect();

    let output = resampler.resample(&audio).expect("valid");

    // High frequency should be significantly attenuated
    // Skip edge samples and check middle
    let mid_start = output.len() / 4;
    let mid_end = 3 * output.len() / 4;
    let max_mid_amp = output[mid_start..mid_end]
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f32, f32::max);

    // With proper anti-aliasing, the 10kHz component should be attenuated
    // (not eliminated due to windowing, but reduced)
    assert!(
        max_mid_amp < 0.8,
        "High frequency not sufficiently attenuated: {}",
        max_mid_amp
    );
}

// =========================================================================
// Legacy Alias Test
// =========================================================================

#[test]
fn test_resampler_alias() {
    // Verify the type alias works
    let resampler: Resampler = Resampler::new(44100, 16000).expect("valid");
    assert_eq!(resampler.source_rate(), 44100);
}

// =========================================================================
// Property-Based Tests
// =========================================================================

#[test]
fn test_resample_output_length_property() {
    // Property: output length should be proportional to ratio
    for (src, tgt) in [
        (44100, 16000),
        (48000, 16000),
        (8000, 16000),
        (22050, 16000),
    ] {
        let resampler = SincResampler::new(src, tgt).expect("valid");
        let input_len = 1000_usize;
        let audio = vec![0.5_f32; input_len];
        let output = resampler.resample(&audio).expect("valid");

        let expected_len = (input_len as f64 * tgt as f64 / src as f64).ceil() as usize;
        assert!(
            (output.len() as i32 - expected_len as i32).abs() <= 1,
            "For {}→{}: expected ~{}, got {}",
            src,
            tgt,
            expected_len,
            output.len()
        );
    }
}

#[test]
fn test_resample_bounded_output_property() {
    // Property: if input is bounded by [-1, 1], output should be approximately bounded
    let resampler = SincResampler::new(44100, 16000).expect("valid");
    let audio: Vec<f32> = (0..4410)
        .map(|i| (i as f32 * 0.01).sin()) // All values in [-1, 1]
        .collect();

    let output = resampler.resample(&audio).expect("valid");

    let max_output = output.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    // Due to sinc interpolation, slight overshoot is possible (Gibbs phenomenon)
    // but should be bounded
    assert!(
        max_output < 1.5,
        "Output exceeded expected bounds: {}",
        max_output
    );
}

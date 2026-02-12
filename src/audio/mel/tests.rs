//! Tests for mel filterbank computation

use super::*;
use crate::format::MelFilterbankData;
use std::f32::consts::PI;

// ============================================================
// UNIT TESTS: Mel scale conversion
// ============================================================

#[test]
fn test_hz_to_mel_zero() {
    let mel = MelFilterbank::hz_to_mel(0.0);
    assert!((mel - 0.0).abs() < 1e-5, "0 Hz should map to 0 mel");
}

#[test]
fn test_hz_to_mel_1000hz() {
    // 1000 Hz is approximately 1000 mel (by design of the mel scale)
    let mel = MelFilterbank::hz_to_mel(1000.0);
    assert!(
        (mel - 1000.0).abs() < 50.0,
        "1000 Hz should be close to 1000 mel, got {mel}"
    );
}

#[test]
fn test_mel_to_hz_roundtrip() {
    let frequencies = [0.0, 100.0, 500.0, 1000.0, 4000.0, 8000.0];
    for &hz in &frequencies {
        let mel = MelFilterbank::hz_to_mel(hz);
        let recovered = MelFilterbank::mel_to_hz(mel);
        assert!(
            (hz - recovered).abs() < 0.1,
            "Roundtrip failed for {hz} Hz: got {recovered}"
        );
    }
}

#[test]
fn test_mel_scale_monotonic() {
    let mut prev_mel = -1.0_f32;
    for hz in (0..8000).step_by(100) {
        let mel = MelFilterbank::hz_to_mel(hz as f32);
        assert!(
            mel > prev_mel,
            "Mel scale should be monotonically increasing"
        );
        prev_mel = mel;
    }
}

// ============================================================
// UNIT TESTS: Filterbank creation
// ============================================================

#[test]
fn test_mel_filterbank_new() {
    let mel = MelFilterbank::new(80, 400, 16000);
    assert_eq!(mel.n_mels(), 80);
    assert_eq!(mel.n_fft(), 400);
    assert_eq!(mel.sample_rate(), 16000);
    assert_eq!(mel.n_freqs(), 201); // 400/2 + 1
}

#[test]
fn test_mel_filterbank_filters_shape() {
    let mel = MelFilterbank::new(80, 400, 16000);
    assert_eq!(mel.filters.len(), 80 * 201);
}

#[test]
fn test_mel_filterbank_filters_nonnegative() {
    let mel = MelFilterbank::new(80, 400, 16000);
    for &f in &mel.filters {
        assert!(f >= 0.0, "Filter values should be non-negative");
    }
}

#[test]
fn test_mel_filterbank_filters_bounded() {
    let mel = MelFilterbank::new(80, 400, 16000);
    for &f in &mel.filters {
        assert!(f <= 1.0, "Filter values should be at most 1.0");
    }
}

#[test]
fn test_mel_filterbank_window_size() {
    let mel = MelFilterbank::new(80, 400, 16000);
    assert_eq!(mel.window.len(), 400);
}

#[test]
fn test_hann_window_endpoints() {
    let window = MelFilterbank::hann_window(100);
    // Hann window should be close to 0 at endpoints
    assert!(window[0] < 0.01, "Hann window should start near 0");
    assert!(
        window[99] < 0.01,
        "Hann window should end near 0, got {}",
        window[99]
    );
}

#[test]
fn test_hann_window_peak() {
    let window = MelFilterbank::hann_window(100);
    // Hann window should peak in the middle
    let mid = window[50];
    assert!(
        mid > 0.9,
        "Hann window should peak near 1.0 in the middle, got {mid}"
    );
}

// ============================================================
// UNIT TESTS: Spectrogram computation
// ============================================================

#[test]
fn test_mel_compute_empty() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let result = mel.compute(&[], 160);
    assert!(result.is_ok());
    assert!(result.map_or(false, |v| v.is_empty()));
}

#[test]
fn test_mel_compute_short_audio() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let audio = vec![0.0; 100]; // Too short for even one frame
    let result = mel.compute(&audio, 160);
    assert!(result.is_ok());
    assert!(result.map_or(false, |v| v.is_empty()));
}

#[test]
fn test_mel_compute_exact_one_frame() {
    let mel = MelFilterbank::new(80, 400, 16000);
    // With center padding, n_frames = n_samples / hop_length
    // 160 samples -> 160 / 160 = 1 frame
    let audio = vec![0.0; 160];
    let result = mel.compute(&audio, 160).expect("compute should succeed");
    // Should have exactly 1 frame
    assert_eq!(result.len(), 80 * 1);
}

#[test]
fn test_mel_compute_multiple_frames() {
    let mel = MelFilterbank::new(80, 400, 16000);
    // 16000 samples = 1 second at 16kHz
    // With center padding and hop_length=160: 16000 / 160 = 100 frames
    let audio = vec![0.0; 16000];
    let result = mel.compute(&audio, 160).expect("compute should succeed");
    let n_frames = result.len() / 80;
    assert_eq!(n_frames, 100);
}

#[test]
fn test_mel_compute_sine_wave() {
    let mel = MelFilterbank::new(80, 400, 16000);

    // Generate 1 second of 440 Hz sine wave
    let sample_rate = 16000.0;
    let freq = 440.0;
    let audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect();

    let result = mel.compute(&audio, 160).expect("compute should succeed");

    // Should have reasonable energy (not all zeros or infinities)
    let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);

    assert!(max_val.is_finite(), "Max should be finite");
    assert!(min_val.is_finite(), "Min should be finite");
    assert!(max_val > min_val, "Should have some variation in output");
}

#[test]
fn test_mel_compute_zero_hop_length() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let audio = vec![0.0; 1600];
    let result = mel.compute(&audio, 0);
    assert!(result.is_err());
}

// ============================================================
// UNIT TESTS: Normalization
// ============================================================

#[test]
fn test_normalize_empty() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let mut data: Vec<f32> = vec![];
    mel.normalize(&mut data); // Should not panic
    assert!(data.is_empty());
}

#[test]
fn test_normalize_single_value() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let mut data = vec![5.0];
    mel.normalize(&mut data);
    // Single value normalized should be 0 (x - mean = 0)
    assert!((data[0]).abs() < 1e-5);
}

#[test]
fn test_normalize_mean_zero() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    mel.normalize(&mut data);

    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 1e-5, "Mean after normalization should be ~0");
}

#[test]
fn test_normalize_std_one() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    mel.normalize(&mut data);

    let variance: f32 = data.iter().map(|&x| x.powi(2)).sum::<f32>() / data.len() as f32;
    let std = variance.sqrt();
    assert!(
        (std - 1.0).abs() < 1e-5,
        "Std after normalization should be ~1, got {std}"
    );
}

// ============================================================
// PROPERTY TESTS (with proptest)
// ============================================================

#[test]
fn test_filterbank_energy_conservation() {
    // Each frequency bin should be covered by at least one filter
    // (except possibly edge bins)
    let mel = MelFilterbank::new(80, 400, 16000);
    let n_freqs = mel.n_freqs();

    for freq_idx in 10..n_freqs - 10 {
        // Skip edge bins
        let total_weight: f32 = (0..80)
            .map(|mel_idx| mel.filters[mel_idx * n_freqs + freq_idx])
            .sum();
        assert!(
            total_weight > 0.0,
            "Frequency bin {freq_idx} should be covered by filters"
        );
    }
}

#[test]
fn test_output_shape_consistency() {
    let mel = MelFilterbank::new(80, 400, 16000);

    for audio_len in [400, 800, 1600, 8000, 16000] {
        let audio = vec![0.0; audio_len];
        let result = mel.compute(&audio, 160).expect("compute should succeed");

        // Output should always be a multiple of n_mels
        assert_eq!(
            result.len() % 80,
            0,
            "Output length {} should be multiple of 80",
            result.len()
        );
    }
}

#[test]
fn test_silence_produces_low_energy() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let silence = vec![0.0; 16000];
    let result = mel.compute(&silence, 160).expect("compute should succeed");

    // Silence should produce very low (negative) log mel energies
    for &val in &result {
        assert!(
            val < 0.0,
            "Silence should produce negative log mel values, got {val}"
        );
    }
}

// ============================================================
// UNIT TESTS: SIMD optimized methods
// ============================================================

#[test]
fn test_apply_filterbank_simd_matches_scalar() {
    let mel = MelFilterbank::new(80, 400, 16000);

    // Generate a test power spectrum
    let power_spec: Vec<f32> = (0..mel.n_freqs())
        .map(|i| (i as f32 * 0.01).sin().powi(2))
        .collect();

    let simd_result = mel.apply_filterbank_simd(&power_spec);
    let scalar_result = mel.apply_filterbank_scalar(&power_spec);

    assert_eq!(simd_result.len(), scalar_result.len());
    for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
        assert!(
            (simd_val - scalar_val).abs() < 1e-5,
            "SIMD and scalar should match: {} vs {}",
            simd_val,
            scalar_val
        );
    }
}

#[test]
fn test_apply_filterbank_simd_output_shape() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let power_spec = vec![1.0; mel.n_freqs()];

    let result = mel.apply_filterbank_simd(&power_spec);
    assert_eq!(result.len(), 80);
}

#[test]
fn test_apply_filterbank_simd_dimension_mismatch() {
    let mel = MelFilterbank::new(80, 400, 16000);

    // Wrong size power spectrum - should fall back to scalar
    let power_spec = vec![1.0; 100]; // Wrong size
    let result = mel.apply_filterbank_simd(&power_spec);

    // Should still work via scalar fallback
    assert_eq!(result.len(), 80);
}

#[test]
fn test_normalize_simd_matches_scalar() {
    let mel = MelFilterbank::new(80, 400, 16000);

    // Generate test data
    let mut simd_data: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut scalar_data = simd_data.clone();

    mel.normalize_simd(&mut simd_data);
    mel.normalize(&mut scalar_data);

    for (simd_val, scalar_val) in simd_data.iter().zip(scalar_data.iter()) {
        assert!(
            (simd_val - scalar_val).abs() < 1e-5,
            "SIMD and scalar normalization should match: {} vs {}",
            simd_val,
            scalar_val
        );
    }
}

#[test]
fn test_normalize_simd_empty() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let mut data: Vec<f32> = vec![];
    mel.normalize_simd(&mut data); // Should not panic
    assert!(data.is_empty());
}

#[test]
fn test_normalize_simd_mean_zero() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    mel.normalize_simd(&mut data);

    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(
        mean.abs() < 1e-5,
        "Mean after SIMD normalization should be ~0, got {mean}"
    );
}

#[test]
fn test_normalize_simd_std_one() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    mel.normalize_simd(&mut data);

    let variance: f32 = data.iter().map(|&x| x.powi(2)).sum::<f32>() / data.len() as f32;
    let std = variance.sqrt();
    assert!(
        (std - 1.0).abs() < 1e-5,
        "Std after SIMD normalization should be ~1, got {std}"
    );
}

#[test]
fn test_compute_simd_matches_compute() {
    let mel = MelFilterbank::new(80, 400, 16000);

    // Generate test audio (sine wave)
    let audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();

    let compute_result = mel.compute(&audio, 160).expect("compute should succeed");
    let simd_result = mel
        .compute_simd(&audio, 160)
        .expect("compute_simd should succeed");

    assert_eq!(compute_result.len(), simd_result.len());
    for (compute_val, simd_val) in compute_result.iter().zip(simd_result.iter()) {
        assert!(
            (compute_val - simd_val).abs() < 1e-4,
            "compute and compute_simd should match: {} vs {}",
            compute_val,
            simd_val
        );
    }
}

#[test]
fn test_compute_simd_empty() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let result = mel.compute_simd(&[], 160);
    assert!(result.is_ok());
    assert!(result.map_or(false, |v| v.is_empty()));
}

#[test]
fn test_compute_simd_zero_hop() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let audio = vec![0.0; 1600];
    let result = mel.compute_simd(&audio, 0);
    assert!(result.is_err());
}

#[test]
fn test_compute_simd_short_audio() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let audio = vec![0.0; 100]; // Too short
    let result = mel.compute_simd(&audio, 160);
    assert!(result.is_ok());
    assert!(result.map_or(false, |v| v.is_empty()));
}

#[test]
fn test_compute_simd_one_frame() {
    let mel = MelFilterbank::new(80, 400, 16000);
    // With center padding: 160 samples -> 160 / 160 = 1 frame
    let audio = vec![0.0; 160];
    let result = mel.compute_simd(&audio, 160).expect("should succeed");
    assert_eq!(result.len(), 80);
}

// ============================================================
// ACCURACY TESTS: Reference validation (WAPR-013)
// ============================================================
// These tests validate mel spectrogram computation against
// known reference values and properties from Whisper's original
// implementation.

#[test]
fn test_mel_scale_matches_reference() {
    // Reference values from librosa/Whisper mel scale
    // mel(f) = 2595 * log10(1 + f/700)
    let test_cases = [
        (0.0, 0.0),
        (700.0, 781.9),   // mel(700) ≈ 782
        (1000.0, 999.99), // mel(1000) ≈ 1000 (by design)
        (4000.0, 2146.1), // mel(4000) ≈ 2146
        (8000.0, 2840.0), // mel(8000) ≈ 2840
    ];

    for (hz, expected_mel) in test_cases {
        let computed = MelFilterbank::hz_to_mel(hz);
        let error = (computed - expected_mel).abs();
        assert!(
            error < 10.0,
            "Mel conversion error for {hz} Hz: expected ~{expected_mel}, got {computed}"
        );
    }
}

#[test]
fn test_filterbank_triangular_shape() {
    // Verify mel filters have proper triangular shape
    let mel = MelFilterbank::new(40, 512, 16000);
    let n_freqs = mel.n_freqs();

    for mel_idx in 1..39 {
        // Check filter peaks in middle (roughly)
        let filter_row = &mel.filters[mel_idx * n_freqs..(mel_idx + 1) * n_freqs];
        let max_val = filter_row.iter().fold(0.0_f32, |a, &b| a.max(b));

        // Each filter should have a peak
        assert!(max_val > 0.0, "Filter {mel_idx} should have positive peak");

        // Peak should not exceed 1.0
        assert!(
            max_val <= 1.0,
            "Filter {mel_idx} peak {} exceeds 1.0",
            max_val
        );
    }
}

#[test]
fn test_whisper_standard_params() {
    // Whisper uses: n_mels=80, n_fft=400, sample_rate=16000, hop=160
    let mel = MelFilterbank::new(80, 400, 16000);

    // Verify standard parameters
    assert_eq!(mel.n_mels(), 80);
    assert_eq!(mel.n_fft(), 400);
    assert_eq!(mel.sample_rate(), 16000);
    assert_eq!(mel.n_freqs(), 201); // n_fft/2 + 1

    // Standard Whisper: 10ms frame shift (160 samples at 16kHz)
    // With center padding: n_frames = n_samples / hop_length
    let hop_length = 160;
    let audio_1s = vec![0.0; 16000];
    let result = mel.compute(&audio_1s, hop_length).expect("should work");

    // 1 second -> 16000 / 160 = 100 frames (with center padding)
    let n_frames = result.len() / 80;
    assert_eq!(n_frames, 100, "1s audio should produce 100 frames");
}

#[test]
fn test_tone_produces_localized_energy() {
    // A pure tone should produce energy in specific mel bands
    let mel = MelFilterbank::new(80, 400, 16000);

    // Generate 440 Hz sine wave (A4 note)
    let sample_rate = 16000.0;
    let freq = 440.0;
    let audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect();

    let result = mel.compute(&audio, 160).expect("compute should succeed");

    // Average energy per mel band across all frames
    // Data layout is [frame][mel], so index as: result[frame * 80 + mel_idx]
    let n_frames = result.len() / 80;
    let mut avg_energy = vec![0.0_f32; 80];
    for frame in 0..n_frames {
        for mel_idx in 0..80 {
            avg_energy[mel_idx] += result[frame * 80 + mel_idx];
        }
    }
    for e in &mut avg_energy {
        *e /= n_frames as f32;
    }

    // Find the mel band with maximum energy
    let max_mel = avg_energy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .expect("avg_energy should not be empty");

    // 440 Hz corresponds to mel ~550, which maps to around mel bin 15-25
    // for 80 mel bands spanning 0-8kHz
    assert!(
        max_mel >= 10 && max_mel <= 35,
        "440 Hz should produce peak in lower mel bands, got bin {max_mel}"
    );
}

#[test]
fn test_high_tone_energy_location() {
    // A high frequency tone should produce energy in higher mel bands
    let mel = MelFilterbank::new(80, 400, 16000);

    // Generate 4000 Hz sine wave
    let sample_rate = 16000.0;
    let freq = 4000.0;
    let audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect();

    let result = mel.compute(&audio, 160).expect("compute should succeed");

    // Average energy per mel band
    // Data layout is [frame][mel], so index as: result[frame * 80 + mel_idx]
    let n_frames = result.len() / 80;
    let mut avg_energy = vec![0.0_f32; 80];
    for frame in 0..n_frames {
        for mel_idx in 0..80 {
            avg_energy[mel_idx] += result[frame * 80 + mel_idx];
        }
    }
    for e in &mut avg_energy {
        *e /= n_frames as f32;
    }

    let max_mel = avg_energy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .expect("avg_energy should not be empty");

    // 4000 Hz corresponds to mel ~2146, in upper half of spectrum
    assert!(
        max_mel >= 40,
        "4000 Hz should produce peak in higher mel bands, got bin {max_mel}"
    );
}

#[test]
fn test_energy_increases_with_amplitude() {
    let mel = MelFilterbank::new(80, 400, 16000);

    let freq = 1000.0;
    let sample_rate = 16000.0;

    // Generate tones with different amplitudes
    let quiet: Vec<f32> = (0..16000)
        .map(|i| 0.1 * (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect();
    let loud: Vec<f32> = (0..16000)
        .map(|i| 1.0 * (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect();

    let result_quiet = mel.compute(&quiet, 160).expect("should work");
    let result_loud = mel.compute(&loud, 160).expect("should work");

    // Compute total energy
    let energy_quiet: f32 = result_quiet.iter().sum();
    let energy_loud: f32 = result_loud.iter().sum();

    // Loud signal should have more energy (in log domain, larger values)
    assert!(
        energy_loud > energy_quiet,
        "Louder signal should have higher energy: {} vs {}",
        energy_loud,
        energy_quiet
    );
}

#[test]
fn test_output_range_is_reasonable() {
    // Log mel values should be in a reasonable range
    let mel = MelFilterbank::new(80, 400, 16000);

    // Generate typical speech-like signal (mix of frequencies)
    let audio: Vec<f32> = (0..16000)
        .map(|i| {
            let t = i as f32 / 16000.0;
            0.3 * (2.0 * PI * 200.0 * t).sin()
                + 0.2 * (2.0 * PI * 500.0 * t).sin()
                + 0.1 * (2.0 * PI * 1000.0 * t).sin()
        })
        .collect();

    let result = mel.compute(&audio, 160).expect("should work");

    // Check that values are in reasonable log domain range
    let min_val = result.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = result.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Log mel values typically range from about -15 to +5
    assert!(min_val > -30.0, "Min log mel value {} is too low", min_val);
    assert!(max_val < 20.0, "Max log mel value {} is too high", max_val);
    assert!(max_val > min_val, "Should have variation in output");
}

#[test]
fn test_numerical_precision_consistency() {
    // Running the same computation multiple times should give identical results
    let mel = MelFilterbank::new(80, 400, 16000);

    let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

    let result1 = mel.compute(&audio, 160).expect("should work");
    let result2 = mel.compute(&audio, 160).expect("should work");

    for (v1, v2) in result1.iter().zip(result2.iter()) {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "Results should be identical: {} vs {}",
            v1,
            v2
        );
    }
}

#[test]
fn test_simd_accuracy_matches_scalar() {
    // SIMD implementation should match scalar within floating point tolerance
    let mel = MelFilterbank::new(80, 400, 16000);

    let audio: Vec<f32> = (0..16000)
        .map(|i| {
            let t = i as f32 / 16000.0;
            (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 880.0 * t).sin()
        })
        .collect();

    let scalar_result = mel.compute(&audio, 160).expect("scalar should work");
    let simd_result = mel.compute_simd(&audio, 160).expect("simd should work");

    assert_eq!(scalar_result.len(), simd_result.len());

    let max_diff = scalar_result
        .iter()
        .zip(simd_result.iter())
        .map(|(s, si)| (s - si).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "SIMD and scalar results differ by {}, should match closely",
        max_diff
    );
}

// =========================================================================
// Property-Based Tests (WAPR-QA-002)
// =========================================================================

mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn property_mel_scale_monotonic(freq in 0.0f32..20000.0) {
            // Mel scale should be monotonically increasing
            let mel1 = MelFilterbank::hz_to_mel(freq);
            let mel2 = MelFilterbank::hz_to_mel(freq + 1.0);
            prop_assert!(mel2 >= mel1, "mel scale should be monotonic");
        }

        #[test]
        fn property_mel_hz_roundtrip(freq in 20.0f32..15000.0) {
            // hz -> mel -> hz should be close to original
            let mel = MelFilterbank::hz_to_mel(freq);
            let back = MelFilterbank::mel_to_hz(mel);
            let error = (freq - back).abs() / freq.max(1.0);
            prop_assert!(error < 0.01, "roundtrip error {} too large for freq {}", error, freq);
        }

        #[test]
        fn property_filterbank_nonnegative(n_mels in 20usize..128, n_fft in 256usize..1024) {
            let sample_rate = 16000;
            let mel = MelFilterbank::new(n_mels, n_fft, sample_rate);
            for val in &mel.filters {
                prop_assert!(*val >= 0.0, "filterbank values must be non-negative");
            }
        }

        #[test]
        fn property_normalize_mean_zero(len in 10usize..1000) {
            // Create a mel filterbank and some test data
            let mel = MelFilterbank::new(80, 400, 16000);
            let mut data: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();

            // Normalize in place
            mel.normalize(&mut data);

            if data.len() > 1 {
                let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                prop_assert!(mean.abs() < 1e-5, "normalized mean {} should be ~0", mean);
            }
        }

        #[test]
        fn property_simd_matches_scalar(audio_len in 1600usize..8000) {
            let mel = MelFilterbank::new(80, 400, 16000);
            let audio: Vec<f32> = (0..audio_len)
                .map(|i| (i as f32 * 0.01).sin() * 0.5)
                .collect();

            if let (Ok(scalar), Ok(simd)) = (mel.compute(&audio, 160), mel.compute_simd(&audio, 160)) {
                prop_assert_eq!(scalar.len(), simd.len(), "output lengths must match");
                for (s, si) in scalar.iter().zip(simd.iter()) {
                    prop_assert!((s - si).abs() < 1e-3, "scalar {} vs simd {} mismatch", s, si);
                }
            }
        }
    }
}

// ============================================================
// ADDITIONAL COVERAGE TESTS
// ============================================================

#[test]
fn test_filters_accessor() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let filters = mel.filters();
    assert_eq!(filters.len(), 80 * 201);
    // All filter values should be non-negative
    assert!(filters.iter().all(|&f| f >= 0.0));
}

#[test]
fn test_from_apr_data() {
    // Create a MelFilterbankData with known values
    let n_mels = 80u32;
    let n_freqs = 201u32;
    let data = vec![0.01_f32; (n_mels * n_freqs) as usize];

    let filterbank_data = MelFilterbankData {
        n_mels,
        n_freqs,
        data,
    };

    let mel = MelFilterbank::from_apr_data(filterbank_data, 16000);
    assert_eq!(mel.n_mels(), 80);
    assert_eq!(mel.n_freqs(), 201);
    assert_eq!(mel.n_fft(), 400);
    assert_eq!(mel.sample_rate(), 16000);
}

#[test]
fn test_from_filters() {
    let n_mels = 40;
    let n_fft = 256;
    let n_freqs = n_fft / 2 + 1;
    let filters = vec![0.1_f32; n_mels * n_freqs];

    let mel = MelFilterbank::from_filters(filters, n_mels, n_fft, 16000);
    assert_eq!(mel.n_mels(), n_mels);
    assert_eq!(mel.n_fft(), n_fft);
    assert_eq!(mel.n_freqs(), n_freqs);
}

#[test]
fn test_window_accessor() {
    let mel = MelFilterbank::new(80, 400, 16000);
    // Window length should match n_fft
    assert_eq!(mel.window.len(), 400);
    // Window values should be in [0, 1]
    assert!(mel.window.iter().all(|&w| w >= 0.0 && w <= 1.0));
}

#[test]
fn test_compute_with_varying_hop_lengths() {
    let mel = MelFilterbank::new(80, 400, 16000);
    let audio = vec![0.0; 16000];

    for hop_length in [80, 160, 320, 400] {
        let result = mel.compute(&audio, hop_length);
        assert!(result.is_ok());
        let spec = result.unwrap();
        let expected_frames = 16000 / hop_length;
        assert_eq!(spec.len(), 80 * expected_frames);
    }
}

#[test]
fn test_mel_to_hz_edge_cases() {
    // Zero mel should map to zero Hz
    let hz = MelFilterbank::mel_to_hz(0.0);
    assert!(hz.abs() < 1e-5);

    // High mel value - mel(3000) should give around 8000 Hz
    let hz_high = MelFilterbank::mel_to_hz(3000.0);
    assert!(
        hz_high > 5000.0,
        "3000 mel should be > 5000 Hz, got {hz_high}"
    );
}

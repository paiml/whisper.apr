//! Debug Mel Spectrogram Normalization
//!
//! Traces through each step of mel computation to find the sign flip.
//! Uses APR-VERIFY ground truth (NO PYTHON).
//!
//! Run with: cargo run --example debug_mel_normalization

use std::f32::consts::PI;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MEL NORMALIZATION DEBUG ===\n");

    // Ground truth from test_data/ref_c_mel_numpy.json (pre-extracted, NO PYTHON)
    let gt_mean = -0.2148_f32;
    let gt_std = 0.4479_f32;
    let gt_min = -0.7658_f32;
    let gt_max = 1.2342_f32;

    println!("Ground Truth (pre-extracted JSON):");
    println!("  mean: {:+.4}", gt_mean);
    println!("  std:  {:.4}", gt_std);
    println!("  min:  {:+.4}", gt_min);
    println!("  max:  {:+.4}", gt_max);

    // Load model
    let model_path = Path::new("models/whisper-tiny.apr");
    if !model_path.exists() {
        println!("\nERROR: Model not found. Run: cargo run --example download_model");
        return Ok(());
    }

    let model_bytes = std::fs::read(model_path)?;
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load test audio
    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    let audio_bytes = std::fs::read(audio_path)?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    // Compute mel
    let mel_full = model.compute_mel(&samples)?;
    let n_mels = 80;
    let n_frames_gt = 148;
    let mel_subset: Vec<f32> = mel_full
        .iter()
        .take(n_frames_gt * n_mels)
        .cloned()
        .collect();

    // Our stats
    let our_mean = mel_subset.iter().sum::<f32>() / mel_subset.len() as f32;
    let our_variance = mel_subset
        .iter()
        .map(|x| (x - our_mean).powi(2))
        .sum::<f32>()
        / mel_subset.len() as f32;
    let our_std = our_variance.sqrt();
    let our_min = mel_subset.iter().cloned().fold(f32::INFINITY, f32::min);
    let our_max = mel_subset.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\nOur Values:");
    println!("  mean: {:+.4}", our_mean);
    println!("  std:  {:.4}", our_std);
    println!("  min:  {:+.4}", our_min);
    println!("  max:  {:+.4}", our_max);

    // Delta analysis
    println!("\n=== DELTA ANALYSIS ===");
    println!("  Mean offset: {:+.4} (should be 0)", our_mean - gt_mean);
    println!("  Std delta:   {:+.4} (should be 0)", our_std - gt_std);
    println!("  Min delta:   {:+.4}", our_min - gt_min);
    println!("  Max delta:   {:+.4}", our_max - gt_max);

    // Reverse engineer the normalization
    // Normalization: y = (x + 4) / 4
    // Inverse: x = y * 4 - 4
    println!("\n=== REVERSE ENGINEERING NORMALIZATION ===");

    // GT before normalization
    let gt_log_max = gt_max * 4.0 - 4.0;
    let gt_log_min_clamped = gt_min * 4.0 - 4.0;
    let gt_log_mean_approx = gt_mean * 4.0 - 4.0;

    println!("GT (before (x+4)/4 normalization):");
    println!("  log_max:  {:+.4}", gt_log_max);
    println!("  log_min (clamped): {:+.4}", gt_log_min_clamped);
    println!("  log_mean (approx): {:+.4}", gt_log_mean_approx);
    println!(
        "  clamp check: log_max - 8 = {:+.4} (should equal log_min)",
        gt_log_max - 8.0
    );

    // Our before normalization
    let our_log_max = our_max * 4.0 - 4.0;
    let our_log_min_clamped = our_min * 4.0 - 4.0;
    let our_log_mean_approx = our_mean * 4.0 - 4.0;

    println!("\nOur (before (x+4)/4 normalization):");
    println!("  log_max:  {:+.4}", our_log_max);
    println!("  log_min (clamped): {:+.4}", our_log_min_clamped);
    println!("  log_mean (approx): {:+.4}", our_log_mean_approx);
    println!("  clamp check: log_max - 8 = {:+.4}", our_log_max - 8.0);

    // Log base analysis
    println!("\n=== LOG BASE ANALYSIS ===");

    // If they use ln and we use log10, the relationship is:
    // ln(x) = log10(x) * ln(10) = log10(x) * 2.303
    // So: log10(x) = ln(x) / 2.303
    //
    // If GT uses log10 with max=0.936, our log10_max should also be 0.936
    // But if GT uses ln, then: ln_max = 0.936, so log10_max would be 0.936/2.303 = 0.406

    println!("If GT uses log10:");
    println!("  GT log10_max = {:.4}", gt_log_max);
    println!("  Our log10_max = {:.4}", our_log_max);
    println!("  Difference = {:+.4}", our_log_max - gt_log_max);

    // What if we need to scale by ln(10)?
    let ln10 = 10.0_f32.ln();
    println!("\nln(10) = {:.4}", ln10);

    // If our log10 values need to be multiplied by ln(10) to get ln values:
    let our_if_ln = our_log_mean_approx * ln10;
    let gt_if_ln = gt_log_mean_approx; // assume GT is already ln

    println!("\nIf GT uses ln (natural log) and we use log10:");
    println!(
        "  Our mean * ln(10) = {:.4} (would be our ln equivalent)",
        our_if_ln
    );
    println!("  If GT mean is ln: {:.4}", gt_if_ln);

    // What if the constants are different?
    println!("\n=== ALTERNATIVE NORMALIZATION FORMULAS ===");

    // Try different normalizations to see which matches
    // Standard Whisper: (log_mel + 4) / 4
    // librosa default: log_mel / 80.0 (per-mel normalization)
    // or: (log_mel - min) / (max - min)

    // What constant C would make our mean match GT mean?
    // (our_log + C) / 4 = gt_mean
    // our_log + C = gt_mean * 4
    // C = gt_mean * 4 - our_log_mean_approx
    let c_needed = gt_mean * 4.0 - our_log_mean_approx;
    println!("Constant C needed to match: {:.4}", c_needed);
    println!("  Standard Whisper uses C = 4.0");
    println!("  We would need C = {:.4}", c_needed);

    // Sample values analysis
    println!("\n=== SAMPLE VALUES (first 5 of each mel bin) ===");
    println!("mel[frame][bin] | Our | (Our * 4 - 4)");
    for bin in 0..3 {
        for frame in 0..5 {
            let idx = frame * n_mels + bin;
            let val = mel_subset[idx];
            let before_norm = val * 4.0 - 4.0;
            println!(
                "  [{:3}][{:2}] | {:+.4} | {:+.4}",
                frame, bin, val, before_norm
            );
        }
    }

    // Check if values are just offset by a constant
    println!("\n=== CONSTANT OFFSET TEST ===");
    let offset = our_mean - gt_mean;
    println!("Mean offset: {:+.4}", offset);

    // If we subtract this offset from all our values, would stats match?
    let adjusted: Vec<f32> = mel_subset.iter().map(|x| x - offset).collect();
    let adj_mean = adjusted.iter().sum::<f32>() / adjusted.len() as f32;
    let adj_var =
        adjusted.iter().map(|x| (x - adj_mean).powi(2)).sum::<f32>() / adjusted.len() as f32;
    let adj_std = adj_var.sqrt();
    let adj_min = adjusted.iter().cloned().fold(f32::INFINITY, f32::min);
    let adj_max = adjusted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\nOur values after subtracting offset {:+.4}:", offset);
    println!("  mean: {:+.4} (GT: {:+.4})", adj_mean, gt_mean);
    println!("  std:  {:.4} (GT: {:.4})", adj_std, gt_std);
    println!("  min:  {:+.4} (GT: {:+.4})", adj_min, gt_min);
    println!("  max:  {:+.4} (GT: {:+.4})", adj_max, gt_max);

    println!("\n=== CONCLUSION ===");
    if (adj_std - gt_std).abs() < 0.01 {
        println!("✓ Std matches after offset correction!");
        println!("  The issue is a CONSTANT OFFSET of {:+.4}", offset);
        println!("\nPossible causes:");
        println!("  1. Wrong normalization constant (using +4.0 when should be different)");
        println!("  2. Missing or incorrect global normalization step");
        println!("  3. Different log base handling in the +4.0 constant");
    } else {
        println!("✗ Std doesn't match even after offset correction");
        println!("  The issue is in the distribution shape, not just offset");
    }

    Ok(())
}

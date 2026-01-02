//! H25: Conv Stem Smear Hypothesis Falsification
//!
//! Tests whether the convolutional stem (Conv1 + GELU + Conv2 + GELU) differentiates
//! audio from padding regions BEFORE transformer blocks process them.
//!
//! Falsification logic:
//! - If conv_output[audio_region].std ≈ conv_output[padding_region].std → H25 VERIFIED (conv smears)
//! - If conv_output[audio_region].std >> conv_output[padding_region].std → H25 FALSIFIED (transformer bug)
//!
//! Run: cargo run --example debug_conv_stem_h25

use std::path::Path;

fn stats(data: &[f32]) -> (f32, f32, f32, f32) {
    let n = data.len() as f32;
    let sum: f32 = data.iter().sum();
    let mean = sum / n;
    let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min, max)
}

fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     H25: CONV STEM SMEAR HYPOTHESIS - FALSIFICATION PROBE        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Load model
    let model_path = Path::new("models/whisper-tiny.apr");
    if !model_path.exists() {
        eprintln!("ERROR: Model not found at {:?}", model_path);
        eprintln!("Run: ./scripts/download-model.sh tiny");
        std::process::exit(1);
    }

    let model_bytes = std::fs::read(model_path)?;
    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load test audio (1.5s speech)
    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    let audio_bytes = std::fs::read(audio_path)?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    println!("[INPUT]");
    println!(
        "  Audio: {} samples ({:.2}s @ 16kHz)",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    // Step 1: Compute mel spectrogram
    println!("\n[STEP 1: MEL SPECTROGRAM]");
    let mel = model.compute_mel(&samples)?;
    let n_mels = 80;
    let n_frames = mel.len() / n_mels;

    // Estimate audio frames (1.5s audio → ~148 frames at 100fps after padding to 30s → 3000 frames)
    let audio_frames = (samples.len() as f32 / 16000.0 * 100.0).ceil() as usize;
    println!("  Mel shape: {} frames × {} mels", n_frames, n_mels);
    println!(
        "  Audio region: frames 0..{} ({:.1}s)",
        audio_frames,
        audio_frames as f32 / 100.0
    );
    println!("  Padding region: frames {}..{}", audio_frames, n_frames);

    // Step 2: Run through conv frontend ONLY (this is what we're testing)
    println!("\n[STEP 2: CONV STEM OUTPUT]");
    println!("  Pipeline: mel → Conv1 → GELU → Conv2(stride=2) → GELU → conv_output\n");

    // Access conv_frontend through encoder_mut (returns &mut but we only need to read)
    let encoder = model.encoder_mut();
    let conv_output = encoder.conv_frontend().forward(&mel)?;

    let d_model = encoder.d_model();
    let conv_len = conv_output.len() / d_model;

    // After stride-2 conv, frames are halved
    let audio_positions_in_conv = audio_frames / 2;
    println!(
        "  Conv output shape: {} positions × {} d_model",
        conv_len, d_model
    );
    println!(
        "  Audio region in conv: positions 0..{}",
        audio_positions_in_conv
    );
    println!(
        "  Padding region in conv: positions {}..{}",
        audio_positions_in_conv, conv_len
    );

    // Step 3: Compare statistics
    println!("\n[STEP 3: REGION STATISTICS]\n");

    // Sample positions in audio region
    let audio_probes = [0, 10, 20, 30, 40, 50, 60, 70];
    // Sample positions in padding region
    let padding_probes = [700, 800, 900, 1000, 1100, 1200, 1300, 1400];

    println!("  AUDIO REGION (positions with speech):");
    let mut audio_stds = Vec::new();
    let mut audio_norms = Vec::new();
    for &pos in &audio_probes {
        if pos < conv_len {
            let start = pos * d_model;
            let end = start + d_model;
            let (mean, std, min, max) = stats(&conv_output[start..end]);
            let norm = l2_norm(&conv_output[start..end]);
            println!(
                "    Pos {:4}: mean={:+8.5}  std={:.5}  L2={:.4}  range=[{:.3}, {:.3}]",
                pos, mean, std, norm, min, max
            );
            audio_stds.push(std);
            audio_norms.push(norm);
        }
    }

    println!("\n  PADDING REGION (positions with zeros):");
    let mut padding_stds = Vec::new();
    let mut padding_norms = Vec::new();
    for &pos in &padding_probes {
        if pos < conv_len {
            let start = pos * d_model;
            let end = start + d_model;
            let (mean, std, min, max) = stats(&conv_output[start..end]);
            let norm = l2_norm(&conv_output[start..end]);
            println!(
                "    Pos {:4}: mean={:+8.5}  std={:.5}  L2={:.4}  range=[{:.3}, {:.3}]",
                pos, mean, std, norm, min, max
            );
            padding_stds.push(std);
            padding_norms.push(norm);
        }
    }

    // Step 4: Aggregate comparison
    println!("\n[STEP 4: AGGREGATE ANALYSIS]\n");

    let audio_std_avg: f32 = audio_stds.iter().sum::<f32>() / audio_stds.len() as f32;
    let padding_std_avg: f32 = padding_stds.iter().sum::<f32>() / padding_stds.len() as f32;
    let audio_norm_avg: f32 = audio_norms.iter().sum::<f32>() / audio_norms.len() as f32;
    let padding_norm_avg: f32 = padding_norms.iter().sum::<f32>() / padding_norms.len() as f32;

    println!("  Average statistics:");
    println!(
        "    Audio region   - std: {:.5}, L2 norm: {:.4}",
        audio_std_avg, audio_norm_avg
    );
    println!(
        "    Padding region - std: {:.5}, L2 norm: {:.4}",
        padding_std_avg, padding_norm_avg
    );

    let std_ratio = audio_std_avg / padding_std_avg.max(1e-10);
    let norm_ratio = audio_norm_avg / padding_norm_avg.max(1e-10);

    println!("\n  Differentiation metrics:");
    println!("    Std ratio (audio/padding):  {:.4}", std_ratio);
    println!("    Norm ratio (audio/padding): {:.4}", norm_ratio);

    // Step 5: Falsification verdict
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      FALSIFICATION VERDICT                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Thresholds for falsification
    let threshold = 1.1; // If ratio > 1.1, conv is differentiating

    if std_ratio > threshold && norm_ratio > threshold {
        println!("  ✓ H25 FALSIFIED");
        println!("");
        println!("  Conv stem output DOES differentiate audio from padding:");
        println!("    - Audio region has {:.1}x higher std", std_ratio);
        println!("    - Audio region has {:.1}x higher L2 norm", norm_ratio);
        println!("");
        println!("  The 'Blind Encoder' bug is NOT in the conv stem.");
        println!("  → Problem is in TRANSFORMER BLOCKS (attention or FFN)");
        println!("  → Next: Investigate per-block output to localize the collapse");
    } else if std_ratio < 0.9 || norm_ratio < 0.9 {
        println!("  ✗ H25 VERIFIED");
        println!("");
        println!("  Conv stem produces SIMILAR output for audio and padding:");
        println!("    - Std ratio: {:.4} (should be >> 1.0)", std_ratio);
        println!("    - Norm ratio: {:.4} (should be >> 1.0)", norm_ratio);
        println!("");
        println!("  The conv stem is 'smearing' - input layout or weights are wrong.");
        println!("  → Next: Check Conv1d weight loading and mel layout (H23)");
    } else {
        println!("  ⚠ INCONCLUSIVE");
        println!("");
        println!("  Ratios are close to 1.0 - weak differentiation:");
        println!("    - Std ratio: {:.4}", std_ratio);
        println!("    - Norm ratio: {:.4}", norm_ratio);
        println!("");
        println!("  Conv stem may be partially working but signal is weak.");
        println!("  → Need deeper investigation");
    }

    // Bonus: Check if bias is dominating
    println!("\n[BONUS: BIAS CHECK]\n");
    let conv1_bias = &encoder.conv_frontend().conv1.bias;
    let conv2_bias = &encoder.conv_frontend().conv2.bias;

    let (b1_mean, b1_std, b1_min, b1_max) = stats(conv1_bias);
    let (b2_mean, b2_std, b2_min, b2_max) = stats(conv2_bias);

    println!(
        "  Conv1 bias: mean={:+.5}  std={:.5}  range=[{:.4}, {:.4}]",
        b1_mean, b1_std, b1_min, b1_max
    );
    println!(
        "  Conv2 bias: mean={:+.5}  std={:.5}  range=[{:.4}, {:.4}]",
        b2_mean, b2_std, b2_min, b2_max
    );

    if b1_std < 0.01 || b2_std < 0.01 {
        println!("\n  ⚠ WARNING: Bias has very low variance - may not be loaded correctly");
    } else {
        println!("\n  ✓ Bias appears loaded (non-zero variance)");
    }

    Ok(())
}

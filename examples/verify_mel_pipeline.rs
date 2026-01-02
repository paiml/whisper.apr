//! APR-VERIFY Pipeline Verification Example
//!
//! Uses aprender::verify to systematically compare mel spectrogram
//! against pre-extracted ground truth (NO PYTHON - Rust only).
//!
//! Run with: cargo run --example verify_mel_pipeline
//!
//! Ground truth source: test_data/reference_summary.json

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           APR-VERIFY: MEL SPECTROGRAM PIPELINE VERIFICATION                  ║");
    println!("║                    Rust-Only Ground Truth Comparison                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════════════════════
    // Load pre-extracted ground truth from JSON (NO PYTHON)
    // ═══════════════════════════════════════════════════════════════════════════════

    // Ground truth values from test_data/reference_summary.json
    // Extracted once, used forever - NO PYTHON NEEDED
    let gt_audio = aprender::verify::GroundTruth::from_stats(0.000178, 0.0696);
    let gt_mel = aprender::verify::GroundTruth::from_stats(-0.2148, 0.4479);

    println!("Ground Truth (pre-extracted, stored in test_data/):");
    println!(
        "  Audio: mean={:+.6}, std={:.4}",
        gt_audio.mean(),
        gt_audio.std()
    );
    println!(
        "  Mel:   mean={:+.4}, std={:.4}",
        gt_mel.mean(),
        gt_mel.std()
    );

    // ═══════════════════════════════════════════════════════════════════════════════
    // Build APR-VERIFY Pipeline
    // ═══════════════════════════════════════════════════════════════════════════════

    let pipeline = aprender::verify::Pipeline::builder("whisper-tiny-mel")
        .stage("audio")
        .ground_truth(gt_audio.clone())
        .tolerance(aprender::verify::Tolerance::percent(5.0))
        .description("16kHz audio samples")
        .build_stage()
        .stage("mel")
        .ground_truth(gt_mel.clone())
        .tolerance(aprender::verify::Tolerance::stats(0.05, 0.05))
        .description("80-bin mel spectrogram")
        .build_stage()
        .build()?;

    println!(
        "\nPipeline '{}' with {} stages configured.",
        pipeline.name(),
        pipeline.stages().len()
    );

    // ═══════════════════════════════════════════════════════════════════════════════
    // Load Model and Audio
    // ═══════════════════════════════════════════════════════════════════════════════

    let model_path = Path::new("models/whisper-tiny.apr");
    if !model_path.exists() {
        println!("\nERROR: Model not found at {:?}", model_path);
        println!("       Run: cargo run --example download_model");
        return Ok(());
    }

    let model_bytes = std::fs::read(model_path)?;
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    let audio_bytes = std::fs::read(audio_path)?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    // ═══════════════════════════════════════════════════════════════════════════════
    // Compute Our Values
    // ═══════════════════════════════════════════════════════════════════════════════

    let our_audio = aprender::verify::GroundTruth::from_slice(&samples);

    let mel_full = model.compute_mel(&samples)?;
    let n_mels = 80;
    let n_frames_gt = 148; // Match ground truth frame count
    let mel_subset: Vec<f32> = mel_full
        .iter()
        .take(n_frames_gt * n_mels)
        .cloned()
        .collect();
    let our_mel = aprender::verify::GroundTruth::from_slice(&mel_subset);

    println!("\nOur Values (computed by whisper.apr):");
    println!(
        "  Audio: mean={:+.6}, std={:.4}",
        our_audio.mean(),
        our_audio.std()
    );
    println!(
        "  Mel:   mean={:+.4}, std={:.4}",
        our_mel.mean(),
        our_mel.std()
    );

    // ═══════════════════════════════════════════════════════════════════════════════
    // Run APR-VERIFY Pipeline
    // ═══════════════════════════════════════════════════════════════════════════════

    let report = pipeline.verify(|stage_name| match stage_name {
        "audio" => Some(our_audio.clone()),
        "mel" => Some(our_mel.clone()),
        _ => None,
    });

    // Print the visual report
    println!("{}", report);

    // ═══════════════════════════════════════════════════════════════════════════════
    // Detailed Delta Analysis
    // ═══════════════════════════════════════════════════════════════════════════════

    println!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│                         DELTA ANALYSIS                                       │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");

    let delta = aprender::verify::Delta::compute(&our_mel, &gt_mel);

    println!("│ Mel Stage:                                                                   │");
    println!(
        "│   Mean delta: {:+.4} (our {:+.4} vs GT {:+.4})                          │",
        our_mel.mean() - gt_mel.mean(),
        our_mel.mean(),
        gt_mel.mean()
    );
    println!(
        "│   Std delta:  {:+.4} (our {:.4} vs GT {:.4})                            │",
        our_mel.std() - gt_mel.std(),
        our_mel.std(),
        gt_mel.std()
    );
    println!(
        "│   Percent:    {:.1}%                                                         │",
        delta.percent()
    );

    if delta.is_sign_flipped() {
        println!(
            "│                                                                              │"
        );
        println!(
            "│   ⚠️  SIGN FLIP DETECTED: Our mean is {:+.4}, GT is {:+.4}              │",
            our_mel.mean(),
            gt_mel.mean()
        );
        println!(
            "│                                                                              │"
        );
        println!("│   ROOT CAUSE CANDIDATES:                                                    │");
        println!("│   1. Wrong log base: log10 vs ln                                            │");
        println!("│   2. Wrong normalization: +4.0/4.0 vs different constants                   │");
        println!("│   3. Wrong clamp: max-8.0 threshold                                         │");
        println!("│   4. Filterbank: magnitude (norm) vs power (norm_sqr)                       │");
    }

    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════════════════

    println!("\n{}", report.summary());

    if report.all_passed() {
        println!("\n✅ All stages passed verification!");
    } else {
        println!("\n❌ Verification failed. See DELTA ANALYSIS above for root cause.");
        std::process::exit(1);
    }

    Ok(())
}

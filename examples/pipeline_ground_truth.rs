//! Pipeline Ground Truth Visual Simulation
//!
//! Compares each pipeline step against reference implementation.
//! Run with: cargo run --example pipeline_ground_truth
//!
//! Similar to trueno and bashrs ground truth visualizations.

use std::path::Path;

/// Stats for a tensor
#[derive(Debug)]
struct TensorStats {
    shape: Vec<usize>,
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
}

impl TensorStats {
    fn from_slice(data: &[f32], shape: Vec<usize>) -> Self {
        let n = data.len() as f32;
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = data.iter().sum::<f32>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();
        Self {
            shape,
            min,
            max,
            mean,
            std,
        }
    }

    fn delta_percent(&self, other: &Self) -> f32 {
        let mean_delta = (self.mean - other.mean).abs();
        let std_delta = (self.std - other.std).abs();
        // Use std as reference since mean can be near zero
        let ref_val = other.std.abs().max(0.001);
        ((mean_delta + std_delta) / ref_val) * 100.0
    }
}

/// Reference stats from ground truth
struct GroundTruth {
    step_a_audio: TensorStats,
    step_c_mel: TensorStats,
}

impl GroundTruth {
    fn load() -> Self {
        // From test_data/reference_summary.json
        Self {
            step_a_audio: TensorStats {
                shape: vec![24000],
                min: -0.198,
                max: 0.298,
                mean: 0.000178,
                std: 0.0696,
            },
            step_c_mel: TensorStats {
                shape: vec![148, 80],
                min: -0.766,
                max: 1.234,
                mean: -0.215,
                std: 0.448,
            },
        }
    }
}

/// Visual status indicator
fn status_icon(delta: f32) -> &'static str {
    if delta < 1.0 {
        "✓"
    } else if delta < 5.0 {
        "~"
    } else if delta < 20.0 {
        "?"
    } else {
        "✗"
    }
}

/// Color code for delta
fn delta_color(delta: f32) -> &'static str {
    if delta < 1.0 {
        "\x1b[32m"
    }
    // Green
    else if delta < 5.0 {
        "\x1b[33m"
    }
    // Yellow
    else if delta < 20.0 {
        "\x1b[33m"
    }
    // Yellow
    else {
        "\x1b[31m"
    } // Red
}

const RESET: &str = "\x1b[0m";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           WHISPER.APR PIPELINE GROUND TRUTH VERIFICATION                     ║");
    println!("║                    Stop-the-Line Debug Mode                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let gt = GroundTruth::load();

    // Load model and audio
    let model_path = Path::new("models/whisper-tiny.apr");
    if !model_path.exists() {
        println!("ERROR: Model not found at {:?}", model_path);
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

    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step │ Name              │ Status │ Our Stats          │ GT Stats           │");
    println!("├──────┼───────────────────┼────────┼────────────────────┼────────────────────┤");

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP A: Audio Input
    // ═══════════════════════════════════════════════════════════════════════════════
    let our_audio = TensorStats::from_slice(&samples, vec![samples.len()]);
    let delta_a = our_audio.delta_percent(&gt.step_a_audio);

    println!(
        "│  A   │ Audio Input       │   {}    │ mean={:+.4} std={:.4} │ mean={:+.4} std={:.4} │",
        status_icon(delta_a),
        our_audio.mean,
        our_audio.std,
        gt.step_a_audio.mean,
        gt.step_a_audio.std
    );

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP C: Mel Spectrogram (only first 148 frames to match GT)
    // ═══════════════════════════════════════════════════════════════════════════════
    let mel_full = model.compute_mel(&samples)?;
    let n_mels = 80;
    let n_frames_gt = 148;

    // Extract first 148 frames to compare with ground truth
    let mel_subset: Vec<f32> = mel_full
        .iter()
        .take(n_frames_gt * n_mels)
        .cloned()
        .collect();
    let our_mel = TensorStats::from_slice(&mel_subset, vec![n_frames_gt, n_mels]);
    let delta_c = our_mel.delta_percent(&gt.step_c_mel);

    println!("│  C   │ Mel Spectrogram   │   {}{}{}    │ mean={:+.4} std={:.4} │ mean={:+.4} std={:.4} │",
             delta_color(delta_c), status_icon(delta_c), RESET,
             our_mel.mean, our_mel.std,
             gt.step_c_mel.mean, gt.step_c_mel.std);

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP D-E: Conv Frontend (no ground truth yet)
    // ═══════════════════════════════════════════════════════════════════════════════
    println!("│  D   │ Conv1 Output      │   ?    │ (no ground truth)  │ (need extraction)  │");
    println!("│  E   │ Conv2 Output      │   ?    │ (no ground truth)  │ (need extraction)  │");

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP G: Encoder Output
    // ═══════════════════════════════════════════════════════════════════════════════
    let encoded = model.encode(&mel_full)?;
    let d_model = 384;
    let enc_len = encoded.len() / d_model;
    let our_encoder = TensorStats::from_slice(&encoded, vec![enc_len, d_model]);

    // No ground truth for encoder yet, but we can show our stats
    println!(
        "│  G   │ Encoder Output    │   ?    │ mean={:+.4} std={:.4} │ (need extraction)  │",
        our_encoder.mean, our_encoder.std
    );

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP H: Cross-Attention K (check audio vs padding variance)
    // ═══════════════════════════════════════════════════════════════════════════════
    // Audio region (0-75) should have different stats than padding (1400+)
    let audio_region: Vec<f32> = (0..75)
        .flat_map(|p| encoded[p * d_model..(p + 1) * d_model].to_vec())
        .collect();
    let padding_region: Vec<f32> = (1400..1500.min(enc_len))
        .flat_map(|p| encoded[p * d_model..(p + 1) * d_model].to_vec())
        .collect();

    let audio_stats = TensorStats::from_slice(&audio_region, vec![75, d_model]);
    let padding_stats = TensorStats::from_slice(&padding_region, vec![100, d_model]);

    let enc_diff = (audio_stats.std - padding_stats.std).abs();
    let enc_status = if enc_diff > 0.1 { "✓" } else { "✗" };

    println!(
        "│  H   │ Enc Audio Region  │   {}    │ mean={:+.4} std={:.4} │ (should differ)    │",
        enc_status, audio_stats.mean, audio_stats.std
    );
    println!(
        "│      │ Enc Padding Rgn   │        │ mean={:+.4} std={:.4} │ (from audio)       │",
        padding_stats.mean, padding_stats.std
    );

    println!("└──────┴───────────────────┴────────┴────────────────────┴────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════════════════
    println!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│                              DIAGNOSIS                                       │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");

    if delta_c > 5.0 {
        println!(
            "│ {}✗ STEP C: Mel spectrogram diverges from ground truth by {:.1}%{}              │",
            "\x1b[31m", delta_c, RESET
        );
        println!("│   → Check mel filterbank, log normalization, padding                        │");
    } else {
        println!(
            "│ {}✓ STEP C: Mel spectrogram matches ground truth (delta={:.1}%){}               │",
            "\x1b[32m", delta_c, RESET
        );
    }

    if enc_diff < 0.1 {
        println!(
            "│ {}✗ STEP H: Encoder produces SAME output for audio and padding{}             │",
            "\x1b[31m", RESET
        );
        println!("│   → Conv frontend may not be distinguishing content from silence            │");
        println!("│   → This explains why attention prefers padding positions                   │");
    } else {
        println!(
            "│ {}✓ STEP H: Encoder differentiates audio from padding (diff={:.2}){}            │",
            "\x1b[32m", enc_diff, RESET
        );
    }

    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════════════
    // Action Items
    // ═══════════════════════════════════════════════════════════════════════════════
    println!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│                            NEXT ACTIONS                                      │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│ 1. Extract conv1/conv2 ground truth from whisper.cpp                        │");
    println!("│ 2. Extract encoder output ground truth from HuggingFace                     │");
    println!("│ 3. Compare conv weight application vs PyTorch                               │");
    println!("│ 4. Verify mel-to-encoder data flow matches reference                        │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘\n");

    Ok(())
}

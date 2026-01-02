//! Compare our mel spectrogram with ground truth binary

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MEL EXACT COMPARISON ===\n");

    // Load reference mel from binary file
    let ref_bytes = std::fs::read("test_data/ref_c_mel_numpy.bin")?;
    let ref_mel: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    println!(
        "Reference mel: {} values ({} frames × 80 mels)",
        ref_mel.len(),
        ref_mel.len() / 80
    );

    // Load model and compute our mel
    let model_bytes = std::fs::read("models/whisper-tiny.apr")?;
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load test audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let our_mel = model.compute_mel(&samples)?;
    println!(
        "Our mel: {} values ({} frames × 80 mels)",
        our_mel.len(),
        our_mel.len() / 80
    );

    // Compare first 148 frames (to match reference)
    let n_compare = ref_mel.len().min(148 * 80);

    println!("\n=== First 10 values comparison ===");
    for i in 0..10 {
        let ref_val = ref_mel[i];
        let our_val = our_mel[i];
        let diff = our_val - ref_val;
        println!(
            "  [{}] ref={:+.6}  our={:+.6}  diff={:+.6}",
            i, ref_val, our_val, diff
        );
    }

    // Compute statistics
    let mut sum_diff = 0.0;
    let mut sum_diff_sq = 0.0;
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;

    for i in 0..n_compare {
        let diff = our_mel[i] - ref_mel[i];
        sum_diff += diff;
        sum_diff_sq += diff * diff;
        if diff.abs() > max_diff.abs() {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    let mean_diff = sum_diff / n_compare as f32;
    let std_diff = (sum_diff_sq / n_compare as f32 - mean_diff * mean_diff).sqrt();

    println!("\n=== Difference Statistics ===");
    println!("  Mean difference:  {:+.6}", mean_diff);
    println!("  Std of difference: {:.6}", std_diff);
    println!(
        "  Max difference:   {:+.6} at index {}",
        max_diff, max_diff_idx
    );

    // Check if there's a constant offset
    println!("\n=== Offset Analysis ===");
    // If we subtract mean_diff from our values, how close do we get?
    let mut corrected_mse = 0.0;
    for i in 0..n_compare {
        let corrected = our_mel[i] - mean_diff;
        let diff = corrected - ref_mel[i];
        corrected_mse += diff * diff;
    }
    let corrected_rmse = (corrected_mse / n_compare as f32).sqrt();
    println!("  After subtracting mean offset:");
    println!("  Corrected RMSE: {:.6}", corrected_rmse);

    // Value range comparison
    let ref_min = ref_mel
        .iter()
        .take(n_compare)
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let ref_max = ref_mel
        .iter()
        .take(n_compare)
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let our_min = our_mel
        .iter()
        .take(n_compare)
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let our_max = our_mel
        .iter()
        .take(n_compare)
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    println!("\n=== Range Comparison ===");
    println!("  Reference: [{:+.4}, {:+.4}]", ref_min, ref_max);
    println!("  Ours:      [{:+.4}, {:+.4}]", our_min, our_max);
    println!(
        "  Range diff: [{:+.4}, {:+.4}]",
        our_min - ref_min,
        our_max - ref_max
    );

    Ok(())
}

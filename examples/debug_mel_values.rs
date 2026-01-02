//! Debug mel spectrogram values at each stage

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MEL SPECTROGRAM DEBUG ===\n");

    // Load model
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

    println!(
        "Audio: {} samples, mean={:.6}, std={:.6}",
        samples.len(),
        samples.iter().sum::<f32>() / samples.len() as f32,
        (samples.iter().map(|x| x.powi(2)).sum::<f32>() / samples.len() as f32).sqrt()
    );

    // Compute mel spectrogram
    let mel = model.compute_mel(&samples)?;

    let n_mels = 80;
    let n_frames = mel.len() / n_mels;
    println!(
        "\nMel shape: {} frames x {} mels = {} total",
        n_frames,
        n_mels,
        mel.len()
    );

    // Stats for first 148 frames (matching ground truth)
    let gt_frames = 148;
    let mel_subset: Vec<f32> = mel.iter().take(gt_frames * n_mels).cloned().collect();

    let mean = mel_subset.iter().sum::<f32>() / mel_subset.len() as f32;
    let variance =
        mel_subset.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / mel_subset.len() as f32;
    let std = variance.sqrt();
    let min = mel_subset.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = mel_subset.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\nMel stats (first {} frames):", gt_frames);
    println!("  Our mean: {:+.4}", mean);
    println!("  Our std:  {:.4}", std);
    println!("  Our min:  {:+.4}", min);
    println!("  Our max:  {:+.4}", max);

    println!("\nGround Truth (from HuggingFace):");
    println!("  GT mean:  {:+.4}", -0.215);
    println!("  GT std:   {:.4}", 0.448);
    println!("  GT min:   {:+.4}", -0.766);
    println!("  GT max:   {:+.4}", 1.234);

    println!("\nDelta:");
    println!("  Mean diff: {:+.4}", mean - (-0.215));
    println!("  Std diff:  {:+.4}", std - 0.448);

    // Check value distribution
    let count_negative = mel_subset.iter().filter(|&&x| x < 0.0).count();
    let count_positive = mel_subset.iter().filter(|&&x| x > 0.0).count();
    println!("\nValue distribution:");
    println!(
        "  Negative values: {} ({:.1}%)",
        count_negative,
        100.0 * count_negative as f32 / mel_subset.len() as f32
    );
    println!(
        "  Positive values: {} ({:.1}%)",
        count_positive,
        100.0 * count_positive as f32 / mel_subset.len() as f32
    );

    // Sample values from first frame
    println!("\nSample values (first frame, first 10 mels):");
    for i in 0..10 {
        println!("  mel[0][{}] = {:+.4}", i, mel[i]);
    }

    // Sample values from silence region (frame 1400)
    println!("\nSample values (frame 1400, first 10 mels) - should be padding:");
    for i in 0..10 {
        println!("  mel[1400][{}] = {:+.4}", i, mel[1400 * n_mels + i]);
    }

    // Check if there's a constant offset
    // If our values are consistently higher, the normalization might be wrong
    println!("\nAnalysis:");
    if mean > 0.0 && -0.215 < 0.0 {
        println!("  SIGN FLIP DETECTED: Our mean is positive, GT is negative");
        println!("  Possible causes:");
        println!("    1. Wrong log base (log10 vs ln)");
        println!("    2. Wrong normalization constants (+4, /4)");
        println!("    3. Wrong clamp threshold (max - 8.0)");
        println!("    4. Different mel filterbank weights");
    }

    Ok(())
}

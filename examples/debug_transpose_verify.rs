//! Verify that mel transpose is happening

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TRANSPOSE VERIFICATION ===\n");

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

    // Compute mel
    let mel = model.compute_mel(&samples)?;
    println!("Mel shape: {} (expected 240000 = 3000 × 80)", mel.len());

    // Check mel layout by looking at frame 0 vs frame 1400
    // If [frames, mels]: frame 0 at mel[0..80], frame 1400 at mel[112000..112080]
    let frame0_frames_first: f32 = mel[0..80].iter().map(|x| x.abs()).sum::<f32>() / 80.0;
    let frame1400_frames_first: f32 = mel[112000..112080].iter().map(|x| x.abs()).sum::<f32>() / 80.0;

    println!("\nIf mel is [frames, mels]:");
    println!("  Frame 0 mean abs: {:.4}", frame0_frames_first);
    println!("  Frame 1400 mean abs: {:.4} (should be ~0.33 if padding)", frame1400_frames_first);

    // Now encode - this should internally transpose mel
    let encoded = model.encode(&mel)?;
    println!("\nEncoder output: {} values", encoded.len());

    let d_model = 384;
    let enc_len = encoded.len() / d_model;
    println!("  = {} positions × {} dims", enc_len, d_model);

    // Check encoder output at different positions
    fn stats(data: &[f32]) -> (f32, f32) {
        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        (mean, std)
    }

    println!("\nEncoder output stats:");
    for pos in [0, 37, 74, 100, 750, 1400] {
        let slice = &encoded[pos * d_model..(pos + 1) * d_model];
        let (mean, std) = stats(slice);
        let region = if pos <= 75 { "AUDIO" } else { "PADDING" };
        println!("  Pos {:4} [{}]: mean={:+.4}, std={:.4}", pos, region, mean, std);
    }

    println!("\n=== KEY QUESTION ===");
    println!("Does encoder output differ between audio (0-75) and padding (76+)?");

    let audio_stds: Vec<f32> = (0..75)
        .map(|p| stats(&encoded[p * d_model..(p + 1) * d_model]).1)
        .collect();
    let padding_stds: Vec<f32> = (1350..1500)
        .map(|p| stats(&encoded[p * d_model..(p + 1) * d_model]).1)
        .collect();

    let audio_avg_std = audio_stds.iter().sum::<f32>() / audio_stds.len() as f32;
    let padding_avg_std = padding_stds.iter().sum::<f32>() / padding_stds.len() as f32;

    println!("  Audio region (0-74) avg std: {:.4}", audio_avg_std);
    println!("  Padding region (1350-1499) avg std: {:.4}", padding_avg_std);

    if (audio_avg_std - padding_avg_std).abs() > 0.1 {
        println!("  ✓ Regions have different variance - encoder sees the difference");
    } else {
        println!("  ✗ Regions have similar variance - encoder may not be processing correctly");
    }

    Ok(())
}

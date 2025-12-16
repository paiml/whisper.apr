//! Debug: Check if weights are being loaded correctly

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== WEIGHT LOADING DEBUG ===\n");

    let model_path = Path::new("models/whisper-tiny-int8.apr");
    let model_bytes = std::fs::read(model_path)?;

    let reader = whisper_apr::format::AprReader::new(model_bytes.clone())?;

    // Check specific tensors that should be loaded
    println!("=== KEY TENSOR VALUES ===\n");

    // Check decoder token embedding
    if let Ok(te) = reader.load_tensor("decoder.token_embedding") {
        let sum: f32 = te.iter().take(1000).sum();
        let first_10: Vec<f32> = te.iter().take(10).copied().collect();
        println!("decoder.token_embedding:");
        println!("  Total elements: {}", te.len());
        println!("  First 10: {:?}", first_10);
        println!("  Sum of first 1000: {:.6}", sum);

        // Check if values are reasonable (not all zeros)
        let non_zero = te.iter().filter(|&&v| v.abs() > 1e-6).count();
        println!(
            "  Non-zero elements: {} ({:.1}%)",
            non_zero,
            non_zero as f32 / te.len() as f32 * 100.0
        );
    } else {
        println!("❌ decoder.token_embedding NOT FOUND");
    }

    // Check encoder conv1 weight
    if let Ok(w) = reader.load_tensor("encoder.conv1.weight") {
        let sum: f32 = w.iter().sum();
        let first_10: Vec<f32> = w.iter().take(10).copied().collect();
        println!("\nencoder.conv1.weight:");
        println!("  Total elements: {}", w.len());
        println!("  First 10: {:?}", first_10);
        println!("  Sum: {:.6}", sum);
    }

    // Check a decoder layer self-attention weight
    if let Ok(w) = reader.load_tensor("decoder.layers.0.self_attn.q_proj.weight") {
        let sum: f32 = w.iter().sum();
        let first_10: Vec<f32> = w.iter().take(10).copied().collect();
        println!("\ndecoder.layers.0.self_attn.q_proj.weight:");
        println!("  Total elements: {}", w.len());
        println!("  First 10: {:?}", first_10);
        println!("  Sum: {:.6}", sum);
    } else {
        println!("\n❌ decoder.layers.0.self_attn.q_proj.weight NOT FOUND");
    }

    // Now load the model and check if weights were loaded
    println!("\n=== LOADING MODEL ===\n");
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load a test audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    println!(
        "Audio samples: {} ({:.2}s)",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    // Check mel spectrogram
    let mel = model.compute_mel(&samples)?;
    let mel_sum: f32 = mel.iter().sum();
    let mel_nonzero = mel.iter().filter(|&&v| v.abs() > 1e-6).count();
    println!("\nMel spectrogram:");
    println!("  Elements: {}", mel.len());
    println!("  Sum: {:.2}", mel_sum);
    println!(
        "  Non-zero: {} ({:.1}%)",
        mel_nonzero,
        mel_nonzero as f32 / mel.len() as f32 * 100.0
    );

    // Check encoder output
    let encoded = model.encode(&mel)?;
    let enc_sum: f32 = encoded.iter().sum();
    let enc_nonzero = encoded.iter().filter(|&&v| v.abs() > 1e-6).count();
    let enc_first_10: Vec<f32> = encoded.iter().take(10).copied().collect();
    println!("\nEncoder output:");
    println!("  Elements: {}", encoded.len());
    println!("  Sum: {:.6}", enc_sum);
    println!("  First 10: {:?}", enc_first_10);
    println!(
        "  Non-zero: {} ({:.1}%)",
        enc_nonzero,
        enc_nonzero as f32 / encoded.len() as f32 * 100.0
    );

    // Check variance of encoder output
    let mean: f32 = enc_sum / encoded.len() as f32;
    let variance: f32 =
        encoded.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / encoded.len() as f32;
    println!("  Mean: {:.6}", mean);
    println!("  Variance: {:.6}", variance);

    // Try transcription
    println!("\n=== TRANSCRIPTION ===\n");
    let result = model.transcribe(&samples, whisper_apr::TranscribeOptions::default())?;
    println!("Text length: {} chars", result.text.len());
    println!("Text: {:?}", result.text);
    println!("Segments: {}", result.segments.len());

    Ok(())
}

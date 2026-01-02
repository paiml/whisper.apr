//! Check if model has embedded filterbank

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MODEL FILTERBANK CHECK ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    if !model_path.exists() {
        println!("ERROR: Model not found");
        return Ok(());
    }

    let model_bytes = std::fs::read(model_path)?;

    // Use the format reader directly
    let reader = whisper_apr::format::AprReader::new(model_bytes)?;

    println!("Model header:");
    println!("  has_filterbank: {}", reader.has_mel_filterbank());

    if let Some(fb) = reader.read_mel_filterbank() {
        println!("\nEmbedded filterbank found!");
        println!("  n_mels: {}", fb.n_mels);
        println!("  n_freqs: {}", fb.n_freqs);
        println!("  data length: {}", fb.data.len());

        // Stats
        let max = fb.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = fb.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let mean = fb.data.iter().sum::<f32>() / fb.data.len() as f32;
        let var = fb.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / fb.data.len() as f32;
        let std = var.sqrt();
        let nonzero = fb.data.iter().filter(|&&x| x > 1e-10).count();

        println!("\nEmbedded filterbank stats:");
        println!("  max:   {:.6}", max);
        println!("  min:   {:.6}", min);
        println!("  mean:  {:.6}", mean);
        println!("  std:   {:.6}", std);
        println!("  nonzero: {}", nonzero);

        // Compare with GT
        println!("\nGround Truth (whisper.cpp):");
        println!("  max:   0.025900");
        println!("  mean:  0.000124");
        println!("  std:   0.001140");
        println!("  nonzero: 391");

        let max_ratio = max / 0.0259;
        println!("\nRatio (embedded/GT): {:.2}x", max_ratio);

        if max_ratio > 10.0 {
            println!("\n⚠️  Embedded filterbank weights are TOO LARGE");
        } else if (max_ratio - 1.0).abs() < 0.1 {
            println!("\n✓ Embedded filterbank matches GT scale");
        }
    } else {
        println!("\n⚠️  NO EMBEDDED FILTERBANK FOUND!");
        println!("Model will use MelFilterbank::new() which lacks Slaney normalization.");
        println!("\nThis is the ROOT CAUSE of the mel sign flip!");
        println!("\nFIX: Add Slaney normalization to MelFilterbank::compute_filterbank()");
    }

    Ok(())
}

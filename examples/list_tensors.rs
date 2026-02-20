//! List all tensors in an .apr model file

use std::path::Path;
use whisper_apr::format::{metadata_to_model_config, AprV2ReaderRef};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== APR MODEL TENSOR LIST ===\n");

    let model_path = Path::new("models/whisper-tiny-int8.apr");
    if !model_path.exists() {
        eprintln!("Model not found: {}", model_path.display());
        return Ok(());
    }

    let model_bytes = std::fs::read(model_path)?;

    // Use the format module directly
    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;

    let config = metadata_to_model_config(reader.metadata());
    println!("Model: {:?}", config.model_type);
    println!("Tensor count: {}\n", reader.tensor_names().len());

    println!("Tensors:");
    println!("{:-<80}", "");

    // Group tensors by prefix
    let mut encoder_tensors = Vec::new();
    let mut decoder_tensors = Vec::new();
    let mut other_tensors = Vec::new();

    for name in reader.tensor_names() {
        let entry = reader.get_tensor(name).expect("tensor must exist for listed name");
        let size = entry.element_count();

        if name.starts_with("encoder") {
            encoder_tensors.push((name.to_string(), entry.shape.clone(), size));
        } else if name.starts_with("decoder") {
            decoder_tensors.push((name.to_string(), entry.shape.clone(), size));
        } else {
            other_tensors.push((name.to_string(), entry.shape.clone(), size));
        }
    }

    println!("\n=== ENCODER TENSORS ({}) ===", encoder_tensors.len());
    for (name, dims, size) in &encoder_tensors[..encoder_tensors.len().min(10)] {
        println!("  {:<45} {:?} ({})", name, dims, size);
    }
    if encoder_tensors.len() > 10 {
        println!("  ... and {} more", encoder_tensors.len() - 10);
    }

    println!("\n=== DECODER TENSORS ({}) ===", decoder_tensors.len());
    for (name, dims, size) in &decoder_tensors[..decoder_tensors.len().min(15)] {
        println!("  {:<45} {:?} ({})", name, dims, size);
    }
    if decoder_tensors.len() > 15 {
        println!("  ... and {} more", decoder_tensors.len() - 15);
    }

    // Look specifically for embedding tensors
    println!("\n=== EMBEDDING/VOCAB TENSORS ===");
    for (name, dims, size) in other_tensors
        .iter()
        .chain(decoder_tensors.iter())
        .chain(encoder_tensors.iter())
    {
        if name.contains("embed") || name.contains("token") || name.contains("position") {
            println!("  {:<45} {:?} ({})", name, dims, size);
        }
    }

    println!("\n=== OTHER TENSORS ({}) ===", other_tensors.len());
    for (name, dims, size) in &other_tensors {
        println!("  {:<45} {:?} ({})", name, dims, size);
    }

    // Check if specific expected tensors exist
    println!("\n=== EXPECTED TENSOR CHECK ===");
    let expected = [
        "decoder.token_embedding",
        "decoder.positional_embedding",
        "encoder.conv1.weight",
        "encoder.conv1.bias",
        "decoder.blocks.0.attn.query.weight",
    ];

    for name in expected {
        match reader.get_tensor_as_f32(name) {
            Some(data) => println!("  {} ({} values)", name, data.len()),
            None => println!("  {} NOT FOUND", name),
        }
    }

    Ok(())
}

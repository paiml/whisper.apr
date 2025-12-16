//! List all tensors in an .apr model file

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== APR MODEL TENSOR LIST ===\n");

    let model_path = Path::new("models/whisper-tiny-int8.apr");
    if !model_path.exists() {
        eprintln!("Model not found: {}", model_path.display());
        return Ok(());
    }

    let model_bytes = std::fs::read(model_path)?;

    // Use the format module directly
    let reader = whisper_apr::format::AprReader::new(model_bytes)?;

    println!("Model: {:?}", reader.header.model_type);
    println!("Quantization: {:?}", reader.header.quantization);
    println!("Tensor count: {}\n", reader.tensors.len());

    println!("Tensors:");
    println!("{:-<80}", "");

    // Group tensors by prefix
    let mut encoder_tensors = Vec::new();
    let mut decoder_tensors = Vec::new();
    let mut other_tensors = Vec::new();

    for tensor in &reader.tensors {
        let name = tensor.name.trim_end_matches('\0');
        let dims: Vec<_> = tensor.shape.iter().take(tensor.n_dims as usize).collect();
        let size = tensor.n_elements;

        if name.starts_with("encoder") {
            encoder_tensors.push((name.to_string(), dims.clone(), size));
        } else if name.starts_with("decoder") {
            decoder_tensors.push((name.to_string(), dims.clone(), size));
        } else {
            other_tensors.push((name.to_string(), dims.clone(), size));
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
        match reader.load_tensor(name) {
            Ok(data) => println!("  ✅ {} ({} values)", name, data.len()),
            Err(_) => println!("  ❌ {} NOT FOUND", name),
        }
    }

    Ok(())
}

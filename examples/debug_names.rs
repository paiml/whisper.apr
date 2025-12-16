//! Debug tensor name matching

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TENSOR NAME DEBUG ===\n");

    let model_path = Path::new("models/whisper-tiny-int8.apr");
    let model_bytes = std::fs::read(model_path)?;

    let reader = whisper_apr::format::AprReader::new(model_bytes)?;

    println!("Total tensors: {}", reader.tensors.len());

    // Find tensors with "token" in name
    let search_term = "token";
    println!("\nTensors containing '{}':", search_term);

    for tensor in &reader.tensors {
        if tensor.name.contains(search_term) {
            println!("  Name: {:?}", tensor.name);
            println!("  Name bytes: {:?}", tensor.name.as_bytes());
            println!("  Name len: {}", tensor.name.len());
            println!();
        }
    }

    // Try exact match
    let exact = "decoder.token_embedding";
    println!("Searching for exact match: {:?}", exact);
    match reader.find_tensor(exact) {
        Some(t) => println!("  FOUND: {:?}", t.name),
        None => {
            println!("  NOT FOUND");

            // Check for close matches
            for tensor in &reader.tensors {
                if tensor.name.starts_with("decoder.token") {
                    println!(
                        "  Close match: {:?} (len {})",
                        tensor.name,
                        tensor.name.len()
                    );
                    // Show difference
                    println!("    Expected bytes: {:?}", exact.as_bytes());
                    println!("    Actual bytes:   {:?}", tensor.name.as_bytes());
                }
            }
        }
    }

    Ok(())
}

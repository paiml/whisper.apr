//! Debug tensor name matching

use std::path::Path;
use whisper_apr::format::AprV2ReaderRef;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TENSOR NAME DEBUG ===\n");

    let model_path = Path::new("models/whisper-tiny-int8.apr");
    let model_bytes = std::fs::read(model_path)?;

    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;

    println!("Total tensors: {}", reader.tensor_names().len());

    // Find tensors with "token" in name
    let search_term = "token";
    println!("\nTensors containing '{}':", search_term);

    for name in reader.tensor_names() {
        if name.contains(search_term) {
            println!("  Name: {:?}", name);
            println!("  Name bytes: {:?}", name.as_bytes());
            println!("  Name len: {}", name.len());
            println!();
        }
    }

    // Try exact match
    let exact = "decoder.token_embedding";
    println!("Searching for exact match: {:?}", exact);
    match reader.get_tensor(exact) {
        Some(t) => println!("  FOUND: {:?}", t.name),
        None => {
            println!("  NOT FOUND");

            // Check for close matches
            for name in reader.tensor_names() {
                if name.starts_with("decoder.token") {
                    println!(
                        "  Close match: {:?} (len {})",
                        name,
                        name.len()
                    );
                    // Show difference
                    println!("    Expected bytes: {:?}", exact.as_bytes());
                    println!("    Actual bytes:   {:?}", name.as_bytes());
                }
            }
        }
    }

    Ok(())
}

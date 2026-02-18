//! List ALL tensors in an .apr model file (no limit)

use std::path::Path;
use whisper_apr::format::{AprV2ReaderRef, metadata_to_model_config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== APR MODEL FULL TENSOR LIST ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    if !model_path.exists() {
        eprintln!("Model not found: {}", model_path.display());
        return Ok(());
    }

    let model_bytes = std::fs::read(model_path)?;
    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;

    let config = metadata_to_model_config(reader.metadata());
    println!("Model: {:?}", config.model_type);
    println!("Tensor count: {}\n", reader.tensor_names().len());

    // Count by type
    let mut cross_attn_k_count = 0;
    let mut cross_attn_v_count = 0;
    let mut cross_attn_q_count = 0;

    println!("=== ALL ENCODER_ATTN TENSORS ===");
    for name in reader.tensor_names() {
        if name.contains("encoder_attn") {
            let entry = reader.get_tensor(name).unwrap();
            println!("  {:<55} {:?}", name, entry.shape);

            if name.contains("k_proj") {
                cross_attn_k_count += 1;
            }
            if name.contains("v_proj") {
                cross_attn_v_count += 1;
            }
            if name.contains("q_proj") {
                cross_attn_q_count += 1;
            }
        }
    }

    println!("\n=== CROSS-ATTENTION SUMMARY ===");
    println!("Q projection tensors: {}", cross_attn_q_count);
    println!("K projection tensors: {}", cross_attn_k_count);
    println!("V projection tensors: {}", cross_attn_v_count);

    if cross_attn_k_count == 0 {
        println!("\nWARNING: NO CROSS-ATTENTION K_PROJ TENSORS FOUND!");
        println!("   This means decoder cannot attend to encoder output properly.");
        println!("   Cross-attention Keys will be uninitialized (zeros).");
    }

    Ok(())
}

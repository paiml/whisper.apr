//! List ALL tensors in an .apr model file (no limit)

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== APR MODEL FULL TENSOR LIST ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    if !model_path.exists() {
        eprintln!("Model not found: {}", model_path.display());
        return Ok(());
    }

    let model_bytes = std::fs::read(model_path)?;
    let reader = whisper_apr::format::AprReader::new(model_bytes)?;

    println!("Model: {:?}", reader.header.model_type);
    println!("Tensor count: {}\n", reader.tensors.len());

    // Count by type
    let mut cross_attn_k_count = 0;
    let mut cross_attn_v_count = 0;
    let mut cross_attn_q_count = 0;

    println!("=== ALL ENCODER_ATTN TENSORS ===");
    for tensor in &reader.tensors {
        let name = tensor.name.trim_end_matches('\0');
        if name.contains("encoder_attn") {
            let dims: Vec<_> = tensor.shape.iter().take(tensor.n_dims as usize).collect();
            println!("  {:<55} {:?}", name, dims);

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
        println!("\n⚠️  WARNING: NO CROSS-ATTENTION K_PROJ TENSORS FOUND!");
        println!("   This means decoder cannot attend to encoder output properly.");
        println!("   Cross-attention Keys will be uninitialized (zeros).");
    }

    Ok(())
}

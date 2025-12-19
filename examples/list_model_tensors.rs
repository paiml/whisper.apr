//! List all tensors in a whisper.apr model file
//!
//! Useful for debugging model structure and weight issues.
//!
//! Usage:
//!   cargo run --example list_model_tensors

use whisper_apr::format::AprReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/whisper-tiny-fb.apr".to_string());

    println!("=== Model Tensors: {} ===\n", model_path);

    let model_bytes = std::fs::read(&model_path)?;
    let reader = AprReader::new(model_bytes)?;

    println!("Total tensors: {}\n", reader.tensors.len());

    // Group by prefix for organization
    let mut by_prefix: std::collections::BTreeMap<String, Vec<_>> =
        std::collections::BTreeMap::new();

    for tensor in &reader.tensors {
        let prefix = tensor.name.split('.').next().unwrap_or("other").to_string();
        by_prefix.entry(prefix).or_default().push(tensor);
    }

    for (prefix, tensors) in &by_prefix {
        println!("=== {} ({} tensors) ===", prefix, tensors.len());
        for tensor in tensors {
            let shape: Vec<_> = tensor.shape().iter().map(|d| d.to_string()).collect();
            let shape_str = format!("[{}]", shape.join(", "));
            let size_kb = tensor.size as f64 / 1024.0;

            println!(
                "  {} {:>20} {:>10.1} KB ({} elements)",
                tensor.name, shape_str, size_kb, tensor.n_elements
            );
        }
        println!();
    }

    // Summary
    let total_bytes: u64 = reader.tensors.iter().map(|t| t.size).sum();
    let total_elements: u64 = reader.tensors.iter().map(|t| t.n_elements).sum();

    println!("=== Summary ===");
    println!("Total tensors: {}", reader.tensors.len());
    println!("Total size: {:.2} MB", total_bytes as f64 / 1024.0 / 1024.0);
    println!(
        "Total elements: {} ({:.2}M)",
        total_elements,
        total_elements as f64 / 1_000_000.0
    );

    // Check for any suspicious tensors (layer_norm weights should have mean ~1.0)
    println!("\n=== Layer Norm Weight Check ===");
    for tensor in &reader.tensors {
        if tensor.name.contains("layer_norm") && tensor.name.contains("weight") {
            if let Ok(values) = reader.load_tensor(&tensor.name) {
                let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
                let flag = if mean.abs() > 5.0 {
                    " <-- LARGE MEAN!"
                } else {
                    ""
                };
                println!("  {}: mean={:.4}{}", tensor.name, mean, flag);
            }
        }
    }

    Ok(())
}

//! List all tensors in a whisper.apr model file
//!
//! Useful for debugging model structure and weight issues.
//!
//! Usage:
//!   cargo run --example list_model_tensors

use whisper_apr::format::AprV2ReaderRef;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/whisper-tiny-fb.apr".to_string());

    println!("=== Model Tensors: {} ===\n", model_path);

    let model_bytes = std::fs::read(&model_path)?;
    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;

    println!("Total tensors: {}\n", reader.tensor_names().len());

    // Group by prefix for organization
    let mut by_prefix: std::collections::BTreeMap<String, Vec<(&str, &[usize], u64, usize)>> =
        std::collections::BTreeMap::new();

    for name in reader.tensor_names() {
        let entry = reader.get_tensor(name).expect("tensor must exist for listed name");
        let prefix = name.split('.').next().unwrap_or("other").to_string();
        let n_elements = entry.element_count();
        by_prefix
            .entry(prefix)
            .or_default()
            .push((name, &entry.shape, entry.size, n_elements));
    }

    for (prefix, tensors) in &by_prefix {
        println!("=== {} ({} tensors) ===", prefix, tensors.len());
        for (name, shape, size, n_elements) in tensors {
            let shape_strs: Vec<_> = shape.iter().map(|d| d.to_string()).collect();
            let shape_str = format!("[{}]", shape_strs.join(", "));
            let size_kb = *size as f64 / 1024.0;

            println!(
                "  {} {:>20} {:>10.1} KB ({} elements)",
                name, shape_str, size_kb, n_elements
            );
        }
        println!();
    }

    // Summary
    let mut total_bytes: u64 = 0;
    let mut total_elements: usize = 0;
    for name in reader.tensor_names() {
        let entry = reader.get_tensor(name).expect("tensor must exist for listed name");
        total_bytes += entry.size;
        total_elements += entry.element_count();
    }

    println!("=== Summary ===");
    println!("Total tensors: {}", reader.tensor_names().len());
    println!("Total size: {:.2} MB", total_bytes as f64 / 1024.0 / 1024.0);
    println!(
        "Total elements: {} ({:.2}M)",
        total_elements,
        total_elements as f64 / 1_000_000.0
    );

    // Check for any suspicious tensors (layer_norm weights should have mean ~1.0)
    println!("\n=== Layer Norm Weight Check ===");
    for name in reader.tensor_names() {
        if name.contains("layer_norm") && name.contains("weight") {
            if let Some(values) = reader.get_tensor_as_f32(name) {
                let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
                let flag = if mean.abs() > 5.0 {
                    " <-- LARGE MEAN!"
                } else {
                    ""
                };
                println!("  {}: mean={:.4}{}", name, mean, flag);
            }
        }
    }

    Ok(())
}

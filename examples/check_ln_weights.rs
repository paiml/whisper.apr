//! Check layer norm weights in the model file
//!
//! The gamma weights should be close to 1.0, not 11.0!

use whisper_apr::format::AprReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LAYER NORM WEIGHT CHECK ===\n");

    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let reader = AprReader::new(model_bytes)?;

    // Find all layer norm weight tensors
    println!("=== Decoder Final Layer Norm ===\n");

    for tensor in &reader.tensors {
        if tensor.name.contains("decoder") && tensor.name.contains("ln") && !tensor.name.contains("bias") {
            if let Ok(values) = reader.load_tensor(&tensor.name) {
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let l2 = values.iter().map(|v| v * v).sum::<f32>().sqrt();

                println!("{}", tensor.name);
                println!("  shape: {:?}, len: {}", tensor.shape, values.len());
                println!("  mean: {:.6}, min: {:.4}, max: {:.4}, L2: {:.4}", mean, min, max, l2);

                // Check if it looks like weights (near 1.0) or bias (near 0.0)
                if mean.abs() > 2.0 {
                    println!("  WARNING: Mean far from 1.0 - possible weight loading bug!");
                }
                println!();
            }
        }
    }

    // Also check decoder.layer_norm specifically
    println!("=== decoder.layer_norm.weight specifically ===\n");

    if let Ok(values) = reader.load_tensor("decoder.layer_norm.weight") {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("decoder.layer_norm.weight:");
        println!("  len: {}", values.len());
        println!("  mean: {:.6}", mean);
        println!("  range: [{:.4}, {:.4}]", min, max);
        println!("  first 10: {:?}", &values[..10].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());

        if mean.abs() > 2.0 {
            println!("\n  ERROR: Layer norm gamma should have mean ~1.0!");
            println!("  This is the root cause of the positive logit shift!");
        }
    }

    // Compare with encoder layer norm
    println!("\n=== Encoder Layer Norm for comparison ===\n");

    if let Ok(values) = reader.load_tensor("encoder.layer_norm.weight") {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        println!("encoder.layer_norm.weight: mean={:.6}", mean);
    }

    // Check all layer norm weights
    println!("\n=== All LayerNorm weights ===\n");

    for tensor in &reader.tensors {
        if tensor.name.contains("layer_norm") && tensor.name.contains("weight") {
            if let Ok(values) = reader.load_tensor(&tensor.name) {
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let flag = if mean.abs() > 2.0 { " <-- BAD!" } else { "" };
                println!("{}: mean={:.4}{}", tensor.name, mean, flag);
            }
        }
    }

    // Also check ln variants (different naming)
    println!("\n=== All LN weights (alt naming) ===\n");

    for tensor in &reader.tensors {
        if (tensor.name.ends_with("_ln.weight") || tensor.name.contains(".ln.") || tensor.name.contains(".ln_"))
           && !tensor.name.contains("bias") {
            if let Ok(values) = reader.load_tensor(&tensor.name) {
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let flag = if mean.abs() > 2.0 { " <-- BAD!" } else { "" };
                println!("{}: mean={:.4}{}", tensor.name, mean, flag);
            }
        }
    }

    Ok(())
}

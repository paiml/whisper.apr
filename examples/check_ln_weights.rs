//! Check layer norm weights in the model file
//!
//! The gamma weights should be close to 1.0, not 11.0!

use whisper_apr::format::AprV2ReaderRef;

fn print_ln_stats(reader: &AprV2ReaderRef<'_>, name: &str) {
    if let Some(values) = reader.get_tensor_as_f32(name) {
        let entry = reader
            .get_tensor(name)
            .expect("tensor must exist since get_tensor_as_f32 succeeded");
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let l2 = values.iter().map(|v| v * v).sum::<f32>().sqrt();

        println!("{name}");
        println!("  shape: {:?}, len: {}", entry.shape, values.len());
        println!("  mean: {mean:.6}, min: {min:.4}, max: {max:.4}, L2: {l2:.4}");

        if mean.abs() > 2.0 {
            println!("  WARNING: Mean far from 1.0 - possible weight loading bug!");
        }
        println!();
    }
}

fn print_ln_mean(reader: &AprV2ReaderRef<'_>, name: &str) {
    if let Some(values) = reader.get_tensor_as_f32(name) {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let flag = if mean.abs() > 2.0 { " <-- BAD!" } else { "" };
        println!("{name}: mean={mean:.4}{flag}");
    }
}

fn check_specific_ln(reader: &AprV2ReaderRef<'_>) {
    println!("=== decoder.layer_norm.weight specifically ===\n");
    if let Some(values) = reader.get_tensor_as_f32("decoder.layer_norm.weight") {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("decoder.layer_norm.weight:");
        println!("  len: {}", values.len());
        println!("  mean: {mean:.6}");
        println!("  range: [{min:.4}, {max:.4}]");
        println!(
            "  first 10: {:?}",
            &values[..10]
                .iter()
                .map(|x| format!("{x:.4}"))
                .collect::<Vec<_>>()
        );

        if mean.abs() > 2.0 {
            println!("\n  ERROR: Layer norm gamma should have mean ~1.0!");
            println!("  This is the root cause of the positive logit shift!");
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LAYER NORM WEIGHT CHECK ===\n");

    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;

    println!("=== Decoder Final Layer Norm ===\n");
    for name in reader.tensor_names() {
        if name.contains("decoder") && name.contains("ln") && !name.contains("bias") {
            print_ln_stats(&reader, name);
        }
    }

    check_specific_ln(&reader);

    println!("\n=== Encoder Layer Norm for comparison ===\n");
    if let Some(values) = reader.get_tensor_as_f32("encoder.layer_norm.weight") {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        println!("encoder.layer_norm.weight: mean={mean:.6}");
    }

    println!("\n=== All LayerNorm weights ===\n");
    for name in reader.tensor_names() {
        if name.contains("layer_norm") && name.contains("weight") {
            print_ln_mean(&reader, name);
        }
    }

    println!("\n=== All LN weights (alt naming) ===\n");
    for name in reader.tensor_names() {
        if (name.ends_with("_ln.weight") || name.contains(".ln.") || name.contains(".ln_"))
            && !name.contains("bias")
        {
            print_ln_mean(&reader, name);
        }
    }

    Ok(())
}

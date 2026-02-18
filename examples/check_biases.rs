//! Check decoder FFN and LayerNorm bias terms
//!
//! Identifies if biases are causing the positive shift.

use whisper_apr::format::AprV2ReaderRef;

fn print_tensor_bias_stats(reader: &AprV2ReaderRef<'_>, name: &str) -> Option<(f64, &'static str)> {
    let values = reader.get_tensor_as_f32(name)?;
    let entry = reader.get_tensor(name)?;
    let sum: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
    let mean = sum / values.len() as f64;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("{name}");
    println!(
        "  shape: {:?}, sum: {:.4}, mean: {:.6}, range: [{:.4}, {:.4}]",
        entry.shape, sum, mean, min, max
    );

    let category = if name.contains("mlp.0") || name.contains("fc1") {
        "fc1"
    } else if name.contains("mlp.2") || name.contains("fc2") {
        "fc2"
    } else if name.contains("ln") || name.contains("layer_norm") {
        "ln"
    } else if name.contains("attn") {
        "attn"
    } else {
        "other"
    };
    Some((sum, category))
}

fn print_bias_distribution(reader: &AprV2ReaderRef<'_>, name: &str) {
    if let Some(values) = reader.get_tensor_as_f32(name) {
        let sum: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
        let mean = sum / values.len() as f64;
        let positive = values.iter().filter(|&&v| v > 0.0).count();
        let negative = values.iter().filter(|&&v| v < 0.0).count();
        println!("{name}: sum={sum:.4}, mean={mean:.6}");
        println!("  positive: {positive}, negative: {negative}");
    }
}

fn print_bias_sum(reader: &AprV2ReaderRef<'_>, name: &str) {
    if let Some(values) = reader.get_tensor_as_f32(name) {
        let sum: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
        let mean = sum / values.len() as f64;
        println!("{name}: sum={sum:.4}, mean={mean:.6}");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIAS ANALYSIS ===\n");

    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;

    println!("=== Decoder Tensors with 'bias' ===\n");

    let mut totals = [0.0f64; 4]; // fc1, fc2, ln, attn

    for name in reader.tensor_names() {
        if name.contains("decoder") && name.contains("bias") {
            if let Some((sum, cat)) = print_tensor_bias_stats(&reader, name) {
                match cat {
                    "fc1" => totals[0] += sum,
                    "fc2" => totals[1] += sum,
                    "ln" => totals[2] += sum,
                    "attn" => totals[3] += sum,
                    _ => {}
                }
            }
        }
    }

    println!("\n=== Bias Sum Totals (across all layers) ===\n");
    println!("FFN fc1 bias sum:   {:.4}", totals[0]);
    println!("FFN fc2 bias sum:   {:.4}", totals[1]);
    println!("LayerNorm bias sum: {:.4}", totals[2]);
    println!("Attention bias sum: {:.4}", totals[3]);

    println!("\n=== Detailed fc2 biases (output projection) ===\n");
    for name in reader.tensor_names() {
        if name.contains("decoder") && (name.contains("mlp.2.bias") || name.contains("fc2.bias")) {
            print_bias_distribution(&reader, name);
        }
    }

    println!("\n=== LayerNorm post (final) ===\n");
    for name in reader.tensor_names() {
        if name.contains("decoder.ln") || name.contains("decoder_ln") {
            if let Some(values) = reader.get_tensor_as_f32(name) {
                let sum: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
                let mean = sum / values.len() as f64;
                println!("{name}: sum={sum:.4}, mean={mean:.6}, len={}", values.len());
            }
        }
    }

    println!("\n=== Encoder fc2 biases for comparison ===\n");
    for name in reader.tensor_names() {
        if name.contains("encoder") && (name.contains("mlp.2.bias") || name.contains("fc2.bias")) {
            print_bias_sum(&reader, name);
        }
    }

    Ok(())
}

//! Check decoder FFN and LayerNorm bias terms
//!
//! Identifies if biases are causing the positive shift.

use whisper_apr::format::AprReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BIAS ANALYSIS ===\n");

    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let reader = AprReader::new(model_bytes.clone())?;

    // Find all decoder bias tensors
    println!("=== Decoder Tensors with 'bias' ===\n");

    let mut total_fc1_bias_sum = 0.0f64;
    let mut total_fc2_bias_sum = 0.0f64;
    let mut total_ln_bias_sum = 0.0f64;
    let mut total_attn_bias_sum = 0.0f64;

    for tensor in &reader.tensors {
        if tensor.name.contains("decoder") && tensor.name.contains("bias") {
            let data = reader.load_tensor(&tensor.name);

            if let Ok(values) = data {
                let sum: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
                let mean = sum / values.len() as f64;
                let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                println!("{}", tensor.name);
                println!("  shape: {:?}, sum: {:.4}, mean: {:.6}, range: [{:.4}, {:.4}]",
                         tensor.shape, sum, mean, min, max);

                // Categorize
                if tensor.name.contains("mlp.0") || tensor.name.contains("fc1") {
                    total_fc1_bias_sum += sum;
                } else if tensor.name.contains("mlp.2") || tensor.name.contains("fc2") {
                    total_fc2_bias_sum += sum;
                } else if tensor.name.contains("ln") || tensor.name.contains("layer_norm") {
                    total_ln_bias_sum += sum;
                } else if tensor.name.contains("attn") {
                    total_attn_bias_sum += sum;
                }
            }
        }
    }

    println!("\n=== Bias Sum Totals (across all layers) ===\n");
    println!("FFN fc1 bias sum:   {:.4}", total_fc1_bias_sum);
    println!("FFN fc2 bias sum:   {:.4}", total_fc2_bias_sum);
    println!("LayerNorm bias sum: {:.4}", total_ln_bias_sum);
    println!("Attention bias sum: {:.4}", total_attn_bias_sum);

    println!("\n=== Detailed fc2 biases (output projection) ===\n");

    for tensor in &reader.tensors {
        if tensor.name.contains("decoder") &&
           (tensor.name.contains("mlp.2.bias") || tensor.name.contains("fc2.bias")) {
            if let Ok(values) = reader.load_tensor(&tensor.name) {
                let sum: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
                let mean = sum / values.len() as f64;

                println!("{}: sum={:.4}, mean={:.6}", tensor.name, sum, mean);

                // Show distribution
                let positive = values.iter().filter(|&&v| v > 0.0).count();
                let negative = values.iter().filter(|&&v| v < 0.0).count();
                println!("  positive: {}, negative: {}", positive, negative);
            }
        }
    }

    println!("\n=== LayerNorm post (final) ===\n");

    for tensor in &reader.tensors {
        if tensor.name.contains("decoder.ln") || tensor.name.contains("decoder_ln") {
            if let Ok(values) = reader.load_tensor(&tensor.name) {
                let sum: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
                let mean = sum / values.len() as f64;
                println!("{}: sum={:.4}, mean={:.6}, len={}", tensor.name, sum, mean, values.len());
            }
        }
    }

    // Check encoder biases too for comparison
    println!("\n=== Encoder fc2 biases for comparison ===\n");

    for tensor in &reader.tensors {
        if tensor.name.contains("encoder") &&
           (tensor.name.contains("mlp.2.bias") || tensor.name.contains("fc2.bias")) {
            if let Ok(values) = reader.load_tensor(&tensor.name) {
                let sum: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
                let mean = sum / values.len() as f64;
                println!("{}: sum={:.4}, mean={:.6}", tensor.name, sum, mean);
            }
        }
    }

    Ok(())
}

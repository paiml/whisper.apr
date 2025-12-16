//! Inspect layer norm weights directly from APR file

use whisper_apr::format::AprReader;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_bytes = fs::read("models/whisper-tiny-fb.apr")?;
    let reader = AprReader::new(model_bytes)?;
    
    println!("=== Layer Norm Weights from APR File ===\n");
    
    // Check decoder final layer norm
    if let Ok(weight) = reader.load_tensor("decoder.layer_norm.weight") {
        println!("decoder.layer_norm.weight:");
        println!("  Length: {}", weight.len());
        let mean: f32 = weight.iter().sum::<f32>() / weight.len() as f32;
        let min = weight.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  Mean: {:.4}", mean);
        println!("  Min: {:.4}, Max: {:.4}", min, max);
        println!("  First 10: {:?}", &weight[..10.min(weight.len())]);
    } else {
        println!("decoder.layer_norm.weight: NOT FOUND");
    }
    
    // Check encoder final layer norm
    if let Ok(weight) = reader.load_tensor("encoder.layer_norm.weight") {
        println!("\nencoder.layer_norm.weight:");
        println!("  Length: {}", weight.len());
        let mean: f32 = weight.iter().sum::<f32>() / weight.len() as f32;
        println!("  Mean: {:.4}", mean);
        println!("  First 10: {:?}", &weight[..10.min(weight.len())]);
    }
    
    // Check a decoder block layer norm
    if let Ok(weight) = reader.load_tensor("decoder.layers.0.self_attn_layer_norm.weight") {
        println!("\ndecoder.layers.0.self_attn_layer_norm.weight:");
        println!("  Length: {}", weight.len());
        let mean: f32 = weight.iter().sum::<f32>() / weight.len() as f32;
        println!("  Mean: {:.4}", mean);
        println!("  First 10: {:?}", &weight[..10.min(weight.len())]);
    }
    
    // List tensors
    println!("\n=== Looking for layer_norm tensors ===");
    let tensor_names = [
        "decoder.layer_norm.weight",
        "decoder.layer_norm.bias",
        "encoder.layer_norm.weight", 
        "encoder.layer_norm.bias",
        "decoder.layers.0.self_attn_layer_norm.weight",
        "decoder.layers.0.encoder_attn_layer_norm.weight",
        "decoder.layers.0.final_layer_norm.weight",
    ];
    
    for name in tensor_names {
        match reader.load_tensor(name) {
            Ok(t) => {
                let mean: f32 = t.iter().sum::<f32>() / t.len() as f32;
                println!("  {}: len={}, mean={:.4}", name, t.len(), mean);
            },
            Err(_) => println!("  {}: NOT FOUND", name),
        }
    }
    
    Ok(())
}

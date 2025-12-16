//! Compare layer norm weights between APR files

use whisper_apr::format::AprReader;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comparing APR files ===\n");
    
    // Load both
    let bytes_orig = fs::read("models/whisper-tiny.apr")?;
    let bytes_fb = fs::read("models/whisper-tiny-fb.apr")?;
    
    let reader_orig = AprReader::new(bytes_orig)?;
    let reader_fb = AprReader::new(bytes_fb)?;
    
    // Compare decoder.layer_norm.weight
    let orig = reader_orig.load_tensor("decoder.layer_norm.weight")?;
    let fb = reader_fb.load_tensor("decoder.layer_norm.weight")?;
    
    println!("decoder.layer_norm.weight:");
    println!("  whisper-tiny.apr: mean={:.4}", orig.iter().sum::<f32>() / orig.len() as f32);
    println!("  whisper-tiny-fb.apr: mean={:.4}", fb.iter().sum::<f32>() / fb.len() as f32);
    
    // Compare encoder.layer_norm.weight
    let orig = reader_orig.load_tensor("encoder.layer_norm.weight")?;
    let fb = reader_fb.load_tensor("encoder.layer_norm.weight")?;
    
    println!("\nencoder.layer_norm.weight:");
    println!("  whisper-tiny.apr: mean={:.4}", orig.iter().sum::<f32>() / orig.len() as f32);
    println!("  whisper-tiny-fb.apr: mean={:.4}", fb.iter().sum::<f32>() / fb.len() as f32);
    
    // Compare decoder.layers.0.self_attn_layer_norm.weight
    let orig = reader_orig.load_tensor("decoder.layers.0.self_attn_layer_norm.weight")?;
    let fb = reader_fb.load_tensor("decoder.layers.0.self_attn_layer_norm.weight")?;
    
    println!("\ndecoder.layers.0.self_attn_layer_norm.weight:");
    println!("  whisper-tiny.apr: mean={:.4}", orig.iter().sum::<f32>() / orig.len() as f32);
    println!("  whisper-tiny-fb.apr: mean={:.4}", fb.iter().sum::<f32>() / fb.len() as f32);
    
    Ok(())
}

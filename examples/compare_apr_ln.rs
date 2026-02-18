//! Compare layer norm weights between APR files

use std::fs;
use whisper_apr::format::AprV2ReaderRef;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comparing APR files ===\n");

    // Load both
    let bytes_orig = fs::read("models/whisper-tiny.apr")?;
    let bytes_fb = fs::read("models/whisper-tiny-fb.apr")?;

    let reader_orig = AprV2ReaderRef::from_bytes(&bytes_orig)?;
    let reader_fb = AprV2ReaderRef::from_bytes(&bytes_fb)?;

    // Compare decoder.layer_norm.weight
    let orig = reader_orig.get_tensor_as_f32("decoder.layer_norm.weight")
        .ok_or("decoder.layer_norm.weight not found in original")?;
    let fb = reader_fb.get_tensor_as_f32("decoder.layer_norm.weight")
        .ok_or("decoder.layer_norm.weight not found in fb")?;

    println!("decoder.layer_norm.weight:");
    println!(
        "  whisper-tiny.apr: mean={:.4}",
        orig.iter().sum::<f32>() / orig.len() as f32
    );
    println!(
        "  whisper-tiny-fb.apr: mean={:.4}",
        fb.iter().sum::<f32>() / fb.len() as f32
    );

    // Compare encoder.layer_norm.weight
    let orig = reader_orig.get_tensor_as_f32("encoder.layer_norm.weight")
        .ok_or("encoder.layer_norm.weight not found in original")?;
    let fb = reader_fb.get_tensor_as_f32("encoder.layer_norm.weight")
        .ok_or("encoder.layer_norm.weight not found in fb")?;

    println!("\nencoder.layer_norm.weight:");
    println!(
        "  whisper-tiny.apr: mean={:.4}",
        orig.iter().sum::<f32>() / orig.len() as f32
    );
    println!(
        "  whisper-tiny-fb.apr: mean={:.4}",
        fb.iter().sum::<f32>() / fb.len() as f32
    );

    // Compare decoder.layers.0.self_attn_layer_norm.weight
    let orig = reader_orig.get_tensor_as_f32("decoder.layers.0.self_attn_layer_norm.weight")
        .ok_or("decoder.layers.0.self_attn_layer_norm.weight not found in original")?;
    let fb = reader_fb.get_tensor_as_f32("decoder.layers.0.self_attn_layer_norm.weight")
        .ok_or("decoder.layers.0.self_attn_layer_norm.weight not found in fb")?;

    println!("\ndecoder.layers.0.self_attn_layer_norm.weight:");
    println!(
        "  whisper-tiny.apr: mean={:.4}",
        orig.iter().sum::<f32>() / orig.len() as f32
    );
    println!(
        "  whisper-tiny-fb.apr: mean={:.4}",
        fb.iter().sum::<f32>() / fb.len() as f32
    );

    Ok(())
}

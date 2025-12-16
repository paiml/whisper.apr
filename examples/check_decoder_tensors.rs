//! Check decoder tensor statistics

use whisper_apr::format::AprReader;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_bytes = fs::read("models/whisper-tiny-fb.apr")?;
    let reader = AprReader::new(model_bytes)?;
    
    println!("=== Decoder Tensor Statistics ===\n");
    
    let tensors = [
        "decoder.token_embedding",
        "decoder.positional_embedding",
        "decoder.layer_norm.weight",
        "decoder.layer_norm.bias",
        "decoder.layers.0.self_attn.q_proj.weight",
        "decoder.layers.0.self_attn.q_proj.bias",
        "decoder.layers.0.self_attn.k_proj.weight",
        "decoder.layers.0.self_attn.v_proj.weight",
        "decoder.layers.0.self_attn.out_proj.weight",
        "decoder.layers.0.fc1.weight",
        "decoder.layers.0.fc2.weight",
    ];
    
    for name in tensors {
        match reader.load_tensor(name) {
            Ok(t) => {
                let mean: f32 = t.iter().sum::<f32>() / t.len() as f32;
                let std: f32 = {
                    let var = t.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / t.len() as f32;
                    var.sqrt()
                };
                let min = t.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = t.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                println!("{}", name);
                println!("  len={}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}", 
                    t.len(), mean, std, min, max);
            },
            Err(_) => println!("{}: NOT FOUND", name),
        }
    }
    
    Ok(())
}

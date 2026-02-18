//! Check decoder tensor statistics

use std::fs;
use whisper_apr::format::AprV2ReaderRef;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_bytes = fs::read("models/whisper-tiny-fb.apr")?;
    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;

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
        match reader.get_tensor_as_f32(name) {
            Some(t) => {
                let mean: f32 = t.iter().sum::<f32>() / t.len() as f32;
                let std: f32 = {
                    let var = t.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / t.len() as f32;
                    var.sqrt()
                };
                let min = t.iter().copied().fold(f32::INFINITY, f32::min);
                let max = t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                println!("{name}");
                println!(
                    "  len={}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
                    t.len(),
                    mean,
                    std,
                    min,
                    max
                );
            }
            None => println!("{name}: NOT FOUND"),
        }
    }

    Ok(())
}

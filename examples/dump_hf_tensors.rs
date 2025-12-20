//! Dump tensor names from HuggingFace safetensors file
//!
//! Usage: cargo run --release --example dump_hf_tensors -- <path_to_safetensors>

use realizar::safetensors::{SafetensorsDtype, SafetensorsModel};
use std::fs;

fn f16_to_f32(bits: u16) -> f32 {
    // Simple f16 to f32 conversion
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;

    if exp == 0 {
        // Subnormal or zero
        let val = (frac as f32) / 1024.0 * 2.0_f32.powi(-14);
        if sign == 1 {
            -val
        } else {
            val
        }
    } else if exp == 31 {
        // Inf or NaN
        if frac == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        // Normal
        let val = (1.0 + (frac as f32) / 1024.0) * 2.0_f32.powi(exp as i32 - 15);
        if sign == 1 {
            -val
        } else {
            val
        }
    }
}

fn get_tensor_mean(model: &SafetensorsModel, name: &str) -> Option<f32> {
    let info = model.tensors.get(name)?;
    let start = info.data_offsets[0];
    let end = info.data_offsets[1];
    let data = &model.data[start..end];

    match info.dtype {
        SafetensorsDtype::F16 => {
            let vals: Vec<f32> = data
                .chunks(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f16_to_f32(bits)
                })
                .collect();
            Some(vals.iter().sum::<f32>() / vals.len() as f32)
        }
        SafetensorsDtype::F32 => {
            let vals: Vec<f32> = data
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Some(vals.iter().sum::<f32>() / vals.len() as f32)
        }
        _ => None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/noah/.cache/huggingface/hub/models--openai--whisper-tiny/snapshots/169d4a4341b33bc18d8881c4b69c2e104e1cc0af/model.safetensors".to_string()
    });

    println!("=== HuggingFace Tensor Names ===");
    println!("File: {}\n", path);

    let data = fs::read(&path)?;
    let model = SafetensorsModel::from_bytes(&data)?;

    let mut names: Vec<&String> = model.tensors.keys().collect();
    names.sort();

    println!("Total tensors: {}\n", names.len());

    // Filter for layer norm tensors specifically
    println!("=== Layer Norm Tensors ===\n");
    for name in names.iter() {
        if name.contains("layer_norm") || name.contains("ln_") || name.contains(".ln.") {
            if let Some(info) = model.tensors.get(*name) {
                let shape: Vec<_> = info.shape.iter().map(|d| d.to_string()).collect();
                let mean = get_tensor_mean(&model, name).unwrap_or(0.0);
                let flag = if mean.abs() > 5.0 { " <-- CHECK!" } else { "" };
                println!("{}: [{}] mean={:.4}{}", name, shape.join(", "), mean, flag);
            }
        }
    }

    println!("\n=== Final LayerNorm (encoder/decoder) ===\n");
    for name in names.iter() {
        // Look for the final layer norms specifically
        if (name.contains("encoder.layer_norm")
            || name.contains("decoder.layer_norm")
            || name.ends_with(".ln.weight")
            || name.ends_with(".ln.bias")
            || name.ends_with("ln_post.weight")
            || name.ends_with("ln_post.bias"))
            && !name.contains("layers")
        {
            if let Some(info) = model.tensors.get(*name) {
                let shape: Vec<_> = info.shape.iter().map(|d| d.to_string()).collect();
                let mean = get_tensor_mean(&model, name).unwrap_or(0.0);
                println!("{}: [{}] mean={:.6}", name, shape.join(", "), mean);
            }
        }
    }

    Ok(())
}

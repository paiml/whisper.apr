//! Show all decoder layer 0 tensor names

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_bytes = std::fs::read("models/whisper-tiny-int8.apr")?;
    let reader = whisper_apr::format::AprReader::new(model_bytes)?;

    println!("=== DECODER LAYER 0 TENSORS ===\n");

    for tensor in &reader.tensors {
        if tensor.name.contains("decoder.layers.0") {
            println!("  {:50} shape={:?}", tensor.name, tensor.shape());
        }
    }

    println!("\n=== ENCODER LAYER 0 TENSORS ===\n");

    for tensor in &reader.tensors {
        if tensor.name.contains("encoder.layers.0") {
            println!("  {:50} shape={:?}", tensor.name, tensor.shape());
        }
    }

    Ok(())
}

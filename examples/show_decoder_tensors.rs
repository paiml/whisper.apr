//! Show all decoder layer 0 tensor names

use whisper_apr::format::AprV2ReaderRef;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_bytes = std::fs::read("models/whisper-tiny-int8.apr")?;
    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;

    println!("=== DECODER LAYER 0 TENSORS ===\n");

    for name in reader.tensor_names() {
        if name.contains("decoder.layers.0") {
            let entry = reader.get_tensor(name).expect("tensor must exist for listed name");
            println!("  {:50} shape={:?}", name, &entry.shape);
        }
    }

    println!("\n=== ENCODER LAYER 0 TENSORS ===\n");

    for name in reader.tensor_names() {
        if name.contains("encoder.layers.0") {
            let entry = reader.get_tensor(name).expect("tensor must exist for listed name");
            println!("  {:50} shape={:?}", name, &entry.shape);
        }
    }

    Ok(())
}

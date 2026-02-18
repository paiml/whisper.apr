//! Compare layer norm weights between original and filterbank models

use whisper_apr::format::AprV2ReaderRef;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MODEL COMPARISON ===\n");

    // Load both models
    let original_bytes = std::fs::read("models/whisper-tiny.apr")?;
    let fb_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;

    let original = AprV2ReaderRef::from_bytes(&original_bytes)?;
    let fb = AprV2ReaderRef::from_bytes(&fb_bytes)?;

    println!("Original model: {} tensors", original.tensor_names().len());
    println!("FB model:       {} tensors", fb.tensor_names().len());

    // Compare decoder.layer_norm.weight
    println!("\n=== decoder.layer_norm.weight ===\n");

    let orig_ln = original.get_tensor_as_f32("decoder.layer_norm.weight")
        .ok_or("decoder.layer_norm.weight not found in original")?;
    let fb_ln = fb.get_tensor_as_f32("decoder.layer_norm.weight")
        .ok_or("decoder.layer_norm.weight not found in fb")?;

    let orig_mean = orig_ln.iter().sum::<f32>() / orig_ln.len() as f32;
    let fb_mean = fb_ln.iter().sum::<f32>() / fb_ln.len() as f32;

    println!(
        "Original: mean={:.6}, first 5: {:?}",
        orig_mean,
        orig_ln[..5]
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "With FB:  mean={:.6}, first 5: {:?}",
        fb_mean,
        fb_ln[..5]
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    // Check if they're the same
    let max_diff = orig_ln
        .iter()
        .zip(fb_ln.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max diff: {:.10}", max_diff);

    // Compare encoder.layer_norm.weight
    println!("\n=== encoder.layers.3.final_layer_norm.weight ===\n");

    if let (Some(orig_enc), Some(fb_enc)) = (
        original.get_tensor_as_f32("encoder.layers.3.final_layer_norm.weight"),
        fb.get_tensor_as_f32("encoder.layers.3.final_layer_norm.weight"),
    ) {
        let orig_mean = orig_enc.iter().sum::<f32>() / orig_enc.len() as f32;
        let fb_mean = fb_enc.iter().sum::<f32>() / fb_enc.len() as f32;

        println!("Original: mean={:.6}", orig_mean);
        println!("With FB:  mean={:.6}", fb_mean);
    }

    println!("\n=== Conclusion ===\n");
    if orig_mean.abs() > 2.0 {
        println!("BOTH models have wrong layer norm weights!");
        println!("The bug is in the ORIGINAL model conversion.");
    } else if fb_mean.abs() > 2.0 {
        println!("Only the FB model has wrong weights.");
        println!("The bug was introduced when adding filterbank.");
    } else {
        println!("Both models have correct layer norm weights (~1.0)");
    }

    Ok(())
}

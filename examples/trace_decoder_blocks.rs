//! Trace decoder block by block to find where positive bias is introduced
//!
//! Uses forward_traced to get L2 norms at each layer.

use whisper_apr::tokenizer::special_tokens;
use whisper_apr::WhisperApr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DECODER BLOCK TRACE ===\n");

    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    // Load audio and compute encoder output
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples)?;
    let encoded = model.encode(&mel)?;

    println!(
        "Encoder output: mean={:.6}, std={:.4}, min={:.4}, max={:.4}",
        stats_mean(&encoded),
        stats_std(&encoded),
        stats_min(&encoded),
        stats_max(&encoded)
    );

    // Initial tokens
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    println!("\nInitial tokens: {:?}", initial_tokens);

    // Use forward_traced to get L2 norms at each stage
    println!("\n=== Forward Traced ===\n");

    let (logits, trace) = model
        .decoder_mut()
        .forward_traced(&initial_tokens, &encoded, None)?;

    // Print trace
    println!("L2 norms at each stage:");
    for (name, value) in &trace {
        println!("  {:20}: {:.4}", name, value);
    }

    // Analyze logits
    let last_logits = &logits[3 * 51865..]; // Last position's logits
    println!("\nLast position logits:");
    println!("  mean: {:.4}", stats_mean(last_logits));
    println!("  std:  {:.4}", stats_std(last_logits));
    println!("  min:  {:.4}", stats_min(last_logits));
    println!("  max:  {:.4}", stats_max(last_logits));

    // Key observation from the trace:
    // The ln_mean trace shows the mean before final layer norm
    // This tells us if the hidden state has shifted positive
    println!("\n=== Key Observations ===");
    for (name, value) in &trace {
        if name == "ln_mean" {
            println!("\nHidden state mean before final LN: {:.6}", value);
            if value.abs() > 1.0 {
                println!("  WARNING: Hidden state has large mean shift!");
                println!("  This explains why all logits are shifted positive.");
            }
        }
    }

    Ok(())
}

fn stats_mean(x: &[f32]) -> f32 {
    x.iter().sum::<f32>() / x.len() as f32
}

fn stats_std(x: &[f32]) -> f32 {
    let mean = stats_mean(x);
    (x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32).sqrt()
}

fn stats_min(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::INFINITY, f32::min)
}

fn stats_max(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

//! Compare hidden states with HuggingFace
//!
//! The +13 logit shift comes from different hidden state magnitudes.
//! Let's trace where the divergence begins.

use std::fs::File;
use std::io::Read;
use whisper_apr::tokenizer::special_tokens;
use whisper_apr::WhisperApr;

fn load_npy_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    if &buf[0..6] != b"\x93NUMPY" {
        return Err("Not a numpy file".into());
    }

    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let data_start = 10 + header_len;
    let header = std::str::from_utf8(&buf[10..data_start])?;

    let is_f64 = header.contains("float64") || header.contains("<f8");
    let is_f32 = header.contains("float32") || header.contains("<f4");

    let data = &buf[data_start..];

    if is_f64 {
        let f64_values: Vec<f64> = data.chunks(8)
            .map(|chunk| f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7]
            ]))
            .collect();
        Ok(f64_values.iter().map(|&x| x as f32).collect())
    } else if is_f32 {
        Ok(data.chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    } else {
        Err(format!("Unknown dtype: {}", header).into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HIDDEN STATE COMPARISON ===\n");

    // Load HF hidden state
    let hf_hidden = load_npy_f32("/tmp/hf_last_hidden.npy")?;
    let hf_l2: f32 = hf_hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    let hf_mean: f32 = hf_hidden.iter().sum::<f32>() / hf_hidden.len() as f32;
    let hf_std: f32 = {
        let variance: f32 = hf_hidden.iter().map(|x| (x - hf_mean).powi(2)).sum::<f32>() / hf_hidden.len() as f32;
        variance.sqrt()
    };

    println!("HuggingFace last hidden state:");
    println!("  shape: {}", hf_hidden.len());
    println!("  L2:    {:.4}", hf_l2);
    println!("  mean:  {:.4}", hf_mean);
    println!("  std:   {:.4}", hf_std);

    // Load our model
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

    // Run decoder with tracing
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    let (logits, trace) = model
        .decoder_mut()
        .forward_traced(&initial_tokens, &encoded)?;

    println!("\nOur implementation:");
    for (name, value) in &trace {
        if name == "last_hidden" || name == "ln_mean" || name == "ln_var" || name == "ln_std"
            || name == "ln_weight_mean" || name == "ln_bias_l2" {
            println!("  {:20}: {:.4}", name, value);
        }
    }

    // Get our last hidden state (need to access it directly)
    // The trace shows L2, but we need the actual values for comparison
    println!("\n=== L2 Comparison ===");
    println!("HF last hidden L2:  {:.4}", hf_l2);
    for (name, value) in &trace {
        if name == "last_hidden" {
            println!("Our last hidden L2: {:.4}", value);
            println!("RATIO: {:.4}x", value / hf_l2);
        }
    }

    // The ratio should explain the logit shift
    // If our hidden is 1.25x larger and mean shift is +13, we need to find why
    println!("\n=== Logit Analysis ===");
    let last_logits = &logits[3 * 51865..];
    let our_logits_mean: f32 = last_logits.iter().sum::<f32>() / last_logits.len() as f32;
    println!("Our logits mean:  {:.4}", our_logits_mean);
    println!("HF logits mean:   8.9304 (from Python)");
    println!("SHIFT:            {:.4}", our_logits_mean - 8.9304);

    // Theory: The shift comes from the hidden state magnitude difference
    // when projected through token embeddings
    println!("\n=== Hypothesis ===");
    println!("Hidden L2 ratio: {:.4}x", trace.iter().find(|(n,_)| n == "last_hidden").map(|(_,v)| v / hf_l2).unwrap_or(1.0));
    println!("If token embeddings have positive bias, larger hidden = larger shift");

    // Check token embedding statistics
    let token_emb = model.decoder_mut().token_embedding();
    let emb_mean: f32 = token_emb.iter().sum::<f32>() / token_emb.len() as f32;
    let emb_l2: f32 = token_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nToken embedding stats:");
    println!("  Total elements: {}", token_emb.len());
    println!("  Mean: {:.6}", emb_mean);
    println!("  L2:   {:.4}", emb_l2);

    Ok(())
}

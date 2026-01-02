#![allow(clippy::unwrap_used)]
//! Compare our outputs with HuggingFace at each stage
//!
//! Run after: uv run --with transformers --with torch --with librosa tools/compare_logits_hf.py

use std::fs::File;
use std::io::Read;
use whisper_apr::tokenizer::special_tokens;
use whisper_apr::WhisperApr;

fn load_npy_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    // NumPy .npy format:
    // - 6 bytes magic: \x93NUMPY
    // - 2 bytes version
    // - 2 bytes header len (little endian)
    // - header (dict with dtype, shape, etc)
    // - data

    if &buf[0..6] != b"\x93NUMPY" {
        return Err("Not a numpy file".into());
    }

    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let data_start = 10 + header_len;

    // Parse header to get shape and dtype
    let header = std::str::from_utf8(&buf[10..data_start])?;
    println!("NPY header: {}", header.trim());

    // Check if float32 or float64
    let is_f64 = header.contains("float64") || header.contains("<f8");
    let is_f32 = header.contains("float32") || header.contains("<f4");

    let data = &buf[data_start..];

    if is_f64 {
        // Convert f64 to f32
        let f64_values: Vec<f64> = data
            .chunks(8)
            .map(|chunk| {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();
        Ok(f64_values.iter().map(|&x| x as f32).collect())
    } else if is_f32 {
        Ok(data
            .chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    } else {
        Err(format!("Unknown dtype in header: {}", header).into())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== COMPARING WITH HUGGINGFACE ===\n");

    // Load HuggingFace outputs
    println!("Loading HuggingFace outputs from /tmp/...");

    let hf_mel = load_npy_f32("/tmp/hf_mel_audio.npy")?;
    println!("HF mel: {} values", hf_mel.len());

    let hf_encoder = load_npy_f32("/tmp/hf_encoder_output.npy")?;
    println!("HF encoder: {} values", hf_encoder.len());

    let hf_logits = load_npy_f32("/tmp/hf_logits.npy")?;
    println!("HF logits: {} values", hf_logits.len());

    let hf_hidden = load_npy_f32("/tmp/hf_last_hidden.npy")?;
    println!("HF hidden: {} values", hf_hidden.len());

    // Load our model
    println!("\nLoading our model...");
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    // Load audio and compute mel
    println!("\n=== MEL COMPARISON ===\n");

    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let our_mel = model.compute_mel(&samples)?;
    println!(
        "Our mel: {} values ({} frames)",
        our_mel.len(),
        our_mel.len() / 80
    );

    // Compare mel (first 150 frames)
    let our_mel_flat: Vec<f32> = our_mel.iter().take(150 * 80).copied().collect();
    let hf_mel_flat: Vec<f32> = hf_mel.iter().take(150 * 80).copied().collect();

    if !our_mel_flat.is_empty() && !hf_mel_flat.is_empty() {
        let mel_sim = cosine_similarity(&our_mel_flat, &hf_mel_flat);
        println!("Mel cosine similarity: {:.6}", mel_sim);

        let our_mean: f32 = our_mel_flat.iter().sum::<f32>() / our_mel_flat.len() as f32;
        let hf_mean: f32 = hf_mel_flat.iter().sum::<f32>() / hf_mel_flat.len() as f32;
        println!("Our mel mean: {:.4}, HF mel mean: {:.4}", our_mean, hf_mean);
    }

    // Encode with our mel
    println!("\n=== ENCODER COMPARISON ===\n");

    let encoded = model.encode(&our_mel)?;

    println!("Our encoder output: {} values", encoded.len());
    println!("HF encoder output: {} values", hf_encoder.len());

    let our_enc_mean: f32 = encoded.iter().sum::<f32>() / encoded.len() as f32;
    let our_enc_std: f32 = {
        let variance: f32 = encoded
            .iter()
            .map(|x| (x - our_enc_mean).powi(2))
            .sum::<f32>()
            / encoded.len() as f32;
        variance.sqrt()
    };

    let hf_enc_mean: f32 = hf_encoder.iter().sum::<f32>() / hf_encoder.len() as f32;
    let hf_enc_std: f32 = {
        let variance: f32 = hf_encoder
            .iter()
            .map(|x| (x - hf_enc_mean).powi(2))
            .sum::<f32>()
            / hf_encoder.len() as f32;
        variance.sqrt()
    };

    println!(
        "Our encoder: mean={:.4}, std={:.4}",
        our_enc_mean, our_enc_std
    );
    println!(
        "HF encoder:  mean={:.4}, std={:.4}",
        hf_enc_mean, hf_enc_std
    );

    // Compare encoder outputs if same size
    if encoded.len() == hf_encoder.len() {
        let enc_sim = cosine_similarity(&encoded, &hf_encoder);
        println!("Encoder cosine similarity: {:.6}", enc_sim);
    }

    // Decode
    println!("\n=== DECODER COMPARISON ===\n");

    // Initial tokens: SOT, lang_en, transcribe, notimestamps
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE, // en
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    let (logits, _trace) = model
        .decoder_mut()
        .forward_traced(&initial_tokens, &encoded)?;

    // Get last position logits
    let last_logits: &[f32] = &logits[3 * 51865..];

    println!("Our logits: {} values", last_logits.len());
    println!("HF logits:  {} values", hf_logits.len());

    let our_logits_mean: f32 = last_logits.iter().sum::<f32>() / last_logits.len() as f32;
    let our_logits_std: f32 = {
        let variance: f32 = last_logits
            .iter()
            .map(|x| (x - our_logits_mean).powi(2))
            .sum::<f32>()
            / last_logits.len() as f32;
        variance.sqrt()
    };

    let hf_logits_mean: f32 = hf_logits.iter().sum::<f32>() / hf_logits.len() as f32;
    let hf_logits_std: f32 = {
        let variance: f32 = hf_logits
            .iter()
            .map(|x| (x - hf_logits_mean).powi(2))
            .sum::<f32>()
            / hf_logits.len() as f32;
        variance.sqrt()
    };

    println!(
        "Our logits: mean={:.4}, std={:.4}",
        our_logits_mean, our_logits_std
    );
    println!(
        "HF logits:  mean={:.4}, std={:.4}",
        hf_logits_mean, hf_logits_std
    );
    println!("SHIFT: {:.4}", our_logits_mean - hf_logits_mean);

    let logits_sim = cosine_similarity(last_logits, &hf_logits);
    println!("\nLogits cosine similarity: {:.6}", logits_sim);

    // Top tokens comparison
    println!("\n=== TOP TOKENS ===\n");

    let mut our_indexed: Vec<(usize, f32)> = last_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    our_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut hf_indexed: Vec<(usize, f32)> =
        hf_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    hf_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Top 10 tokens:");
    println!(
        "{:>6} {:>12} | {:>6} {:>12}",
        "Ours", "logit", "HF", "logit"
    );
    println!("{}", "-".repeat(45));
    for i in 0..10 {
        println!(
            "{:>6} {:>12.4} | {:>6} {:>12.4}",
            our_indexed[i].0, our_indexed[i].1, hf_indexed[i].0, hf_indexed[i].1
        );
    }

    // Specific tokens
    println!("\n=== KEY TOKENS ===\n");
    let key_tokens = [
        (440, " The"),
        (464, "The"),
        (220, " "),
        (50256, "EOT"),
        (50257, "SOT"),
    ];
    for (tok, name) in key_tokens {
        println!(
            "Token {} '{}': our={:.4}, hf={:.4}, diff={:.4}",
            tok,
            name,
            last_logits[tok],
            hf_logits[tok],
            last_logits[tok] - hf_logits[tok]
        );
    }

    // Check hidden state if we can get it
    println!("\n=== HIDDEN STATE ===\n");
    println!(
        "HF last hidden: mean={:.4}, L2={:.4}",
        hf_hidden.iter().sum::<f32>() / hf_hidden.len() as f32,
        hf_hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    Ok(())
}

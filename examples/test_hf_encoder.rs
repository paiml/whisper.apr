//! Test decoder with HuggingFace's encoder output
//!
//! Hypothesis: Using HF's encoder output should eliminate the +13 shift

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
    println!("Loading {}: header = {}", path, header.trim());

    let is_f64 = header.contains("float64") || header.contains("<f8");
    let is_f32 = header.contains("float32") || header.contains("<f4");

    let data = &buf[data_start..];

    if is_f64 {
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
        Err(format!("Unknown dtype: {}", header).into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TEST: DECODER WITH HF ENCODER OUTPUT ===\n");

    // Load HuggingFace encoder output (1500 x 384)
    let hf_encoder = load_npy_f32("/tmp/hf_encoder_output.npy")?;
    println!(
        "HF encoder output: {} values ({} positions x 384)",
        hf_encoder.len(),
        hf_encoder.len() / 384
    );

    let hf_enc_mean: f32 = hf_encoder.iter().sum::<f32>() / hf_encoder.len() as f32;
    let hf_enc_l2: f32 = hf_encoder.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("HF encoder: mean={:.4}, L2={:.4}\n", hf_enc_mean, hf_enc_l2);

    // Load HF logits for comparison
    let hf_logits = load_npy_f32("/tmp/hf_logits.npy")?;
    let hf_logits_mean: f32 = hf_logits.iter().sum::<f32>() / hf_logits.len() as f32;

    // Load our model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    // TEST 1: Use HF encoder output directly
    println!("=== TEST 1: Decoder with HF Encoder Output ===\n");

    let (logits_hf_enc, trace) = model
        .decoder_mut()
        .forward_traced(&initial_tokens, &hf_encoder, None)?;

    let last_logits_hf_enc = &logits_hf_enc[3 * 51865..];
    let our_mean_hf_enc: f32 =
        last_logits_hf_enc.iter().sum::<f32>() / last_logits_hf_enc.len() as f32;

    println!("With HF encoder output:");
    println!("  Our logits mean:  {:.4}", our_mean_hf_enc);
    println!("  HF logits mean:   {:.4}", hf_logits_mean);
    println!(
        "  SHIFT:            {:.4}",
        our_mean_hf_enc - hf_logits_mean
    );

    // Show trace
    println!("\nTrace (key values):");
    for (name, value) in &trace {
        if name == "last_hidden" || name.starts_with("ln_") {
            println!("  {:20}: {:.4}", name, value);
        }
    }

    // TEST 2: Use our encoder output
    println!("\n=== TEST 2: Decoder with Our Encoder Output ===\n");

    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples)?;
    let our_encoder = model.encode(&mel)?;

    println!(
        "Our encoder output: {} values ({} positions x 384)",
        our_encoder.len(),
        our_encoder.len() / 384
    );

    let our_enc_mean: f32 = our_encoder.iter().sum::<f32>() / our_encoder.len() as f32;
    let our_enc_l2: f32 = our_encoder.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "Our encoder: mean={:.4}, L2={:.4}\n",
        our_enc_mean, our_enc_l2
    );

    let (logits_our_enc, trace2) = model
        .decoder_mut()
        .forward_traced(&initial_tokens, &our_encoder, None)?;

    let last_logits_our_enc = &logits_our_enc[3 * 51865..];
    let our_mean_our_enc: f32 =
        last_logits_our_enc.iter().sum::<f32>() / last_logits_our_enc.len() as f32;

    println!("With our encoder output:");
    println!("  Our logits mean:  {:.4}", our_mean_our_enc);
    println!("  HF logits mean:   {:.4}", hf_logits_mean);
    println!(
        "  SHIFT:            {:.4}",
        our_mean_our_enc - hf_logits_mean
    );

    // Comparison
    println!("\n=== CONCLUSION ===");
    println!(
        "Shift with HF encoder:  {:.4}",
        our_mean_hf_enc - hf_logits_mean
    );
    println!(
        "Shift with our encoder: {:.4}",
        our_mean_our_enc - hf_logits_mean
    );

    if (our_mean_hf_enc - hf_logits_mean).abs() < 1.0 {
        println!("\n✓ Using HF encoder eliminates the shift!");
        println!("  → The bug is in our ENCODER, not the decoder.");
    } else if (our_mean_hf_enc - hf_logits_mean).abs() < (our_mean_our_enc - hf_logits_mean).abs() {
        println!("\n~ Using HF encoder reduces the shift.");
        println!("  → The bug is partially in encoder, partially in decoder.");
    } else {
        println!("\n✗ Using HF encoder doesn't help.");
        println!("  → The bug is in our DECODER, not the encoder.");
    }

    Ok(())
}

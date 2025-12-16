#![allow(clippy::unwrap_used)]
//! Debug: Examine decoder logits to understand token 11 issue

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DECODER LOGITS DEBUG ===\n");

    let model_path = Path::new("models/whisper-tiny-int8.apr");
    let model_bytes = std::fs::read(model_path)?;

    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load a test audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    println!(
        "Audio samples: {} ({:.2}s)\n",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    // 1. Compute mel spectrogram
    let mel = model.compute_mel(&samples)?;
    println!("Mel spectrogram: {} elements", mel.len());

    // 2. Encode
    let encoded = model.encode(&mel)?;
    println!("Encoder output: {} elements", encoded.len());

    // Check encoder output stats
    let enc_sum: f32 = encoded.iter().sum();
    let enc_mean = enc_sum / encoded.len() as f32;
    let enc_var: f32 =
        encoded.iter().map(|x| (x - enc_mean).powi(2)).sum::<f32>() / encoded.len() as f32;
    let enc_max = encoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let enc_min = encoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let enc_nans = encoded.iter().filter(|&&x| x.is_nan()).count();
    let enc_infs = encoded.iter().filter(|&&x| x.is_infinite()).count();

    println!("\nEncoder output stats:");
    println!("  Mean: {:.6}", enc_mean);
    println!("  Variance: {:.6}", enc_var);
    println!("  Min: {:.6}", enc_min);
    println!("  Max: {:.6}", enc_max);
    println!("  NaNs: {}", enc_nans);
    println!("  Infs: {}", enc_infs);
    println!("  First 10: {:?}", &encoded[..10.min(encoded.len())]);

    // 3. Now test the decoder directly
    println!("\n=== DECODER FORWARD TEST ===\n");

    // Get initial tokens - use LANG_BASE for English (50259)
    use whisper_apr::tokenizer::special_tokens;
    let lang_en = special_tokens::LANG_BASE; // English is at LANG_BASE
    let initial_tokens = vec![
        special_tokens::SOT,
        lang_en,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    println!("Initial tokens: {:?}", initial_tokens);
    println!("  SOT: {}", special_tokens::SOT);
    println!("  LANG_EN: {}", lang_en);
    println!("  TRANSCRIBE: {}", special_tokens::TRANSCRIBE);
    println!("  NO_TIMESTAMPS: {}", special_tokens::NO_TIMESTAMPS);

    // Do a single forward pass with all initial tokens
    let logits = model.decoder_mut().forward(&initial_tokens, &encoded)?;

    println!("\nLogits from decoder.forward():");
    println!("  Total elements: {}", logits.len());

    // The logits should be (seq_len x n_vocab), we want the LAST token's logits
    let n_vocab = 51865;
    let seq_len = initial_tokens.len();
    let expected_logits = seq_len * n_vocab;

    println!(
        "  Expected: {} (seq_len={} x n_vocab={})",
        expected_logits, seq_len, n_vocab
    );

    if logits.len() != expected_logits {
        println!("  WARNING: Logits size mismatch!");
    }

    // Get last token's logits
    let last_logits_start = (seq_len - 1) * n_vocab;
    let last_logits = &logits[last_logits_start..last_logits_start + n_vocab];

    // Check logits stats
    let log_sum: f32 = last_logits.iter().sum();
    let log_mean = log_sum / last_logits.len() as f32;
    let log_var: f32 = last_logits
        .iter()
        .map(|x| (x - log_mean).powi(2))
        .sum::<f32>()
        / last_logits.len() as f32;
    let log_max = last_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let log_min = last_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let log_nans = last_logits.iter().filter(|&&x: &&f32| x.is_nan()).count();
    let log_infs = last_logits
        .iter()
        .filter(|&&x: &&f32| x.is_infinite())
        .count();

    println!("\nLast token's logits stats:");
    println!("  Mean: {:.6}", log_mean);
    println!("  Variance: {:.6}", log_var);
    println!("  Min: {:.6}", log_min);
    println!("  Max: {:.6}", log_max);
    println!("  NaNs: {}", log_nans);
    println!("  Infs: {}", log_infs);
    println!(
        "  First 20: {:?}",
        &last_logits[..20.min(last_logits.len())]
    );

    // Find top 10 tokens
    let mut indexed: Vec<(usize, f32)> = last_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 10 tokens by logit:");
    for (i, (token_id, logit)) in indexed.iter().take(10).enumerate() {
        // Try to decode the token
        let token_str = match *token_id {
            id if id == special_tokens::SOT as usize => "<SOT>".to_string(),
            id if id == special_tokens::EOT as usize => "<EOT>".to_string(),
            id if id == special_tokens::TRANSLATE as usize => "<translate>".to_string(),
            id if id == special_tokens::TRANSCRIBE as usize => "<transcribe>".to_string(),
            id if id == special_tokens::NO_TIMESTAMPS as usize => "<notimestamps>".to_string(),
            id if id < 256 => format!("byte:{}", id),
            _ => format!("token:{}", token_id),
        };
        println!(
            "  {}: token {} ({}) = {:.4}",
            i + 1,
            token_id,
            token_str,
            logit
        );
    }

    // Check EOT token logit (50256)
    let eot_logit = last_logits[special_tokens::EOT as usize];
    println!("\nEOT (50256) logit: {:.4}", eot_logit);

    // Find EOT rank
    let eot_rank = indexed
        .iter()
        .position(|(id, _)| *id == special_tokens::EOT as usize)
        .map(|r| r + 1)
        .unwrap_or(0);
    println!("EOT rank: {} out of {}", eot_rank, n_vocab);

    // Check if token 11 is among top
    let token_11_logit = last_logits[11];
    println!("Token 11 logit: {:.4}", token_11_logit);

    // Check the argmax
    let argmax = indexed[0].0;
    println!("Argmax token: {} (logit: {:.4})", argmax, indexed[0].1);

    // Check if all logits are the same (would indicate weight loading issue)
    let unique_values: std::collections::HashSet<u32> =
        last_logits.iter().map(|&x: &f32| x.to_bits()).collect();
    println!("\nUnique logit values: {}", unique_values.len());

    if unique_values.len() < 100 {
        println!("  WARNING: Very few unique values - likely weight loading issue!");
    }

    Ok(())
}

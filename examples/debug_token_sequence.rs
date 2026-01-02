#![allow(clippy::unwrap_used)]
//! Debug: See the actual token sequence generated and how it's decoded

use std::cell::RefCell;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TOKEN SEQUENCE DEBUG ===\n");

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

    // Get mel and encoder output
    let mel = model.compute_mel(&samples)?;
    let encoded = model.encode(&mel)?;

    use whisper_apr::model::DecoderKVCache;
    use whisper_apr::tokenizer::special_tokens;

    let lang_en = special_tokens::LANG_BASE;
    let initial_tokens = vec![
        special_tokens::SOT,
        lang_en,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    println!("Initial tokens: {:?}", initial_tokens);

    // Get config for decoder
    let n_vocab = 51865;
    let max_tokens = 448;
    let d_model = 384; // tiny model
    let n_layers = 4; // tiny model

    // Create KV cache
    let cache = RefCell::new(DecoderKVCache::new(n_layers, d_model, max_tokens));
    let processed_count = RefCell::new(0usize);

    // Logits function (copied from lib.rs decode function)
    let mut logits_fn = |tokens: &[u32]| -> whisper_apr::error::WhisperResult<Vec<f32>> {
        let seq_len = tokens.len();
        let already_processed = *processed_count.borrow();

        let mut logits = vec![f32::NEG_INFINITY; n_vocab];

        for i in already_processed..seq_len {
            let token = tokens[i];
            logits = model
                .decoder_mut()
                .forward_one(token, &encoded, &mut cache.borrow_mut())?;
        }

        *processed_count.borrow_mut() = seq_len;
        Ok(logits)
    };

    // Do greedy decoding manually and track tokens
    let eot = special_tokens::EOT;
    let mut tokens = initial_tokens.clone();

    println!("\n=== DECODING LOOP ===\n");

    for step in 0..20 {
        // Limit to 20 steps
        let logits = logits_fn(&tokens)?;

        // Find argmax
        let (argmax_token, argmax_logit) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("example assertion"))
            .map(|(idx, val)| (idx as u32, *val))
            .expect("example assertion");

        // Check for NaNs/Infs
        let nans = logits.iter().filter(|&&x| x.is_nan()).count();
        let infs = logits.iter().filter(|&&x| x.is_infinite()).count();

        println!(
            "Step {}: argmax token {} (logit {:.4}), NaNs={}, Infs={}",
            step, argmax_token, argmax_logit, nans, infs
        );

        // Show top 5 tokens
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!(
            "  Top 5: {:?}",
            indexed
                .iter()
                .take(5)
                .map(|(i, v)| (*i as u32, format!("{:.2}", v)))
                .collect::<Vec<_>>()
        );

        // Stop if EOT
        if argmax_token == eot {
            tokens.push(argmax_token);
            println!("  -> EOT, stopping");
            break;
        }

        tokens.push(argmax_token);
    }

    println!("\n=== FINAL TOKEN SEQUENCE ===\n");
    println!("All tokens ({} total): {:?}", tokens.len(), tokens);

    // Try to decode each token individually
    println!("\n=== DECODING TOKENS ===\n");
    for (i, &token) in tokens.iter().enumerate() {
        let token_name = match token {
            t if t == special_tokens::SOT => "<SOT>".to_string(),
            t if t == special_tokens::EOT => "<EOT>".to_string(),
            t if t == special_tokens::TRANSLATE => "<translate>".to_string(),
            t if t == special_tokens::TRANSCRIBE => "<transcribe>".to_string(),
            t if t == special_tokens::NO_TIMESTAMPS => "<notimestamps>".to_string(),
            t if t == special_tokens::LANG_BASE => "<en>".to_string(),
            t if t < 256 => format!("byte:{} (0x{:02x})", t, t),
            _ => format!("token:{}", token),
        };
        println!("  Token {}: {} -> {}", i, token, token_name);
    }

    // Now decode using the tokenizer
    println!("\n=== TOKENIZER DECODE ===\n");
    let decoded = model.tokenizer().decode(&tokens)?;
    println!("Decoded text: {:?}", decoded);
    println!("Decoded bytes: {:?}", decoded.as_bytes());

    // Filter out special tokens and decode just content
    let content_tokens: Vec<u32> = tokens
        .iter()
        .copied()
        .filter(|&t| t < 50257) // Skip special tokens
        .collect();
    println!("\nContent tokens (< 50257): {:?}", content_tokens);

    if !content_tokens.is_empty() {
        let content_decoded = model.tokenizer().decode(&content_tokens)?;
        println!("Content decoded: {:?}", content_decoded);
    }

    Ok(())
}

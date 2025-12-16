#![allow(clippy::unwrap_used)]
//! Debug: Trace the actual transcribe path

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TRANSCRIBE DEBUG ===\n");

    let model_path = Path::new("models/whisper-tiny-fb.apr");
    let model_bytes = std::fs::read(model_path)?;

    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

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

    // Check what the decoder outputs with manual token suppression
    use std::cell::RefCell;
    use whisper_apr::model::DecoderKVCache;
    use whisper_apr::tokenizer::special_tokens;

    let mel = model.compute_mel(&samples)?;
    let encoded = model.encode(&mel)?;

    let n_vocab = 51865;
    let max_tokens = 448;
    let d_model = 384;
    let n_layers = 4;

    // Initial tokens with CORRECT IDs
    let initial_tokens = vec![
        special_tokens::SOT,           // 50257
        special_tokens::LANG_BASE,     // 50258 (English)
        special_tokens::TRANSCRIBE,    // 50358
        special_tokens::NO_TIMESTAMPS, // 50362
    ];

    println!("Initial tokens: {:?}", initial_tokens);

    let cache = RefCell::new(DecoderKVCache::new(n_layers, d_model, max_tokens));
    let processed_count = RefCell::new(0usize);
    let mut model = model;

    // Logits function with token suppression (matching lib.rs)
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

        // Suppress special tokens
        logits[special_tokens::SOT as usize] = f32::NEG_INFINITY;
        logits[special_tokens::NO_SPEECH as usize] = f32::NEG_INFINITY;
        logits[special_tokens::TRANSLATE as usize] = f32::NEG_INFINITY;
        logits[special_tokens::TRANSCRIBE as usize] = f32::NEG_INFINITY;
        logits[special_tokens::PREV as usize] = f32::NEG_INFINITY;
        logits[special_tokens::SPEAKER_TURN as usize] = f32::NEG_INFINITY;
        logits[special_tokens::NO_TIMESTAMPS as usize] = f32::NEG_INFINITY;

        // Suppress all language tokens (50258 to 50357)
        for i in special_tokens::LANG_BASE..special_tokens::TRANSLATE {
            logits[i as usize] = f32::NEG_INFINITY;
        }

        // Suppress timestamps
        for i in special_tokens::TIMESTAMP_BASE as usize..n_vocab {
            logits[i] = f32::NEG_INFINITY;
        }

        Ok(logits)
    };

    // Manual greedy decoding
    let eot = special_tokens::EOT;
    let mut tokens = initial_tokens.clone();

    println!("\n=== DECODING WITH SUPPRESSION ===\n");

    for step in 0..20 {
        let logits = logits_fn(&tokens)?;

        // Find argmax
        let (argmax_token, argmax_logit) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("assertion"))
            .map(|(idx, val)| (idx as u32, *val))
            .expect("assertion");

        println!(
            "Step {}: argmax token {} (logit {:.4})",
            step, argmax_token, argmax_logit
        );

        // Show top 5 AFTER suppression
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

        // Check if this token is in valid range for decoding
        if argmax_token > 255 && argmax_token < special_tokens::EOT {
            println!(
                "  WARNING: Token {} is a BPE merge token (not in base vocabulary)",
                argmax_token
            );
        }

        if argmax_token == eot {
            tokens.push(argmax_token);
            println!("  -> EOT, stopping");
            break;
        }

        tokens.push(argmax_token);
    }

    println!("\n=== FINAL TOKENS ===\n");
    println!("All tokens: {:?}", tokens);

    // Try to decode
    println!("\n=== TOKENIZER DECODE ===\n");
    println!("Vocab size: {}", model.tokenizer().vocab_size());

    for &token in &tokens {
        if token < 256 {
            println!(
                "  Token {}: byte '{}'",
                token,
                char::from_u32(token).unwrap_or('?')
            );
        } else if token < special_tokens::EOT {
            println!("  Token {}: BPE merge (NOT IN BASE VOCAB!)", token);
        } else {
            let name = match token {
                t if t == special_tokens::EOT => "<EOT>",
                t if t == special_tokens::SOT => "<SOT>",
                t if t == special_tokens::TRANSLATE => "<translate>",
                t if t == special_tokens::TRANSCRIBE => "<transcribe>",
                t if t == special_tokens::NO_TIMESTAMPS => "<notimestamps>",
                t if t >= special_tokens::LANG_BASE && t < special_tokens::TRANSLATE => "<lang>",
                _ => "<special>",
            };
            println!("  Token {}: {}", token, name);
        }
    }

    Ok(())
}

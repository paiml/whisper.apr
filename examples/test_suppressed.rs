#![allow(clippy::unwrap_used)]
//! Test decoding with SOT suppressed

use std::fs::File;
use std::io::Read;
use whisper_apr::tokenizer::special_tokens;
use whisper_apr::WhisperApr;

fn load_npy_f32(path: &str) -> Vec<f32> {
    let mut file = File::open(path).expect("open file");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("read file");

    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let data_start = 10 + header_len;

    let data = &buf[data_start..];
    data.chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TEST: Decoding with Suppression ===\n");

    // Load HF encoder output
    let hf_enc = load_npy_f32("/tmp/hf_encoder_output.npy");
    println!("Encoder: {} values", hf_enc.len());

    // Load model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    // Tokens to suppress (all special tokens)
    let suppress: Vec<u32> = (50257..51865).collect(); // All special tokens

    println!("Initial tokens: {:?}", initial_tokens);
    println!(
        "Suppressing {} special tokens (50257-51864)",
        suppress.len()
    );

    let mut tokens = initial_tokens.clone();

    for step in 0..15 {
        let logits = model.decoder_mut().forward(&tokens, &hf_enc)?;
        let mut last_logits: Vec<f32> =
            logits[(tokens.len() - 1) * 51865..tokens.len() * 51865].to_vec();

        // Suppress special tokens
        for &s in &suppress {
            if (s as usize) < last_logits.len() {
                last_logits[s as usize] = f32::NEG_INFINITY;
            }
        }

        // Argmax
        let (argmax, max_logit) = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        // Top 5
        let mut indexed: Vec<_> = last_logits.iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        println!("Step {}: token {} (logit {:.2})", step, argmax, max_logit);
        print!("  Top 5: ");
        for (i, (tok, logit)) in indexed.iter().take(5).enumerate() {
            if !logit.is_infinite() {
                print!("{}: {:.2}", tok, logit);
                if i < 4 {
                    print!(", ");
                }
            }
        }
        println!();

        tokens.push(argmax as u32);

        if argmax as u32 >= 50257 {
            break; // Should not happen with suppression
        }
    }

    // Decode tokens
    println!("\nAll tokens: {:?}", &tokens);

    let text_tokens: Vec<u32> = tokens
        .iter()
        .skip(4)
        .filter(|&&t| t < 50257)
        .cloned()
        .collect();

    println!("Text tokens: {:?}", &text_tokens);

    if !text_tokens.is_empty() {
        if let Ok(text) = model.tokenizer().decode(&text_tokens) {
            println!("Text: \"{}\"", text);
        }
    }

    Ok(())
}

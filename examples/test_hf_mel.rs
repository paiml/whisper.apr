#![allow(clippy::unwrap_used)]
//! Test transcription with HF's mel spectrogram

use std::fs::File;
use std::io::Read;
use whisper_apr::WhisperApr;

fn load_npy_f32(path: &str) -> Vec<f32> {
    let mut file = File::open(path).expect("open file");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("read file");

    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let data_start = 10 + header_len;
    let header = std::str::from_utf8(&buf[10..data_start]).unwrap_or("");
    println!("Loading {}: header = {}", path, header.trim());

    let data = &buf[data_start..];
    data.chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TEST: Transcription with HF Mel Spectrogram ===\n");

    // Load HF mel (80 x 3000)
    let hf_mel_raw = load_npy_f32("/tmp/hf_mel_full.npy");
    println!("HF mel: {} values", hf_mel_raw.len());

    // HF mel is [80, 3000] - transpose to [3000, 80]
    let n_mels = 80;
    let n_frames = 3000;
    let mut hf_mel = vec![0.0_f32; n_frames * n_mels];
    for frame in 0..n_frames {
        for mel in 0..n_mels {
            hf_mel[frame * n_mels + mel] = hf_mel_raw[mel * n_frames + frame];
        }
    }

    // Load model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    // Encode with HF mel
    let encoder_output = model.encode(&hf_mel)?;
    println!(
        "Encoder output: {} values ({} positions)",
        encoder_output.len(),
        encoder_output.len() / 384
    );

    let enc_mean: f32 = encoder_output.iter().sum::<f32>() / encoder_output.len() as f32;
    let enc_l2: f32 = encoder_output.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Encoder: mean={:.4}, L2={:.4}", enc_mean, enc_l2);

    // Compare with HF encoder output
    let hf_enc = load_npy_f32("/tmp/hf_encoder_output.npy");
    let hf_enc_mean: f32 = hf_enc.iter().sum::<f32>() / hf_enc.len() as f32;
    let hf_enc_l2: f32 = hf_enc.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nHF encoder: mean={:.4}, L2={:.4}", hf_enc_mean, hf_enc_l2);
    println!(
        "Difference: mean={:.4}, L2={:.4}",
        enc_mean - hf_enc_mean,
        enc_l2 - hf_enc_l2
    );

    // Decode
    use whisper_apr::tokenizer::special_tokens;
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    println!("\n=== DECODING ===\n");
    let mut tokens = initial_tokens.clone();

    for step in 0..20 {
        let logits = model.decoder_mut().forward(&tokens, &encoder_output, None)?;
        let last_logits = &logits[(tokens.len() - 1) * 51865..tokens.len() * 51865];

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
            print!("{}: {:.2}", tok, logit);
            if i < 4 {
                print!(", ");
            }
        }
        println!();

        tokens.push(argmax as u32);

        if argmax as u32 == special_tokens::EOT {
            println!("\nReached EOT");
            break;
        }
    }

    // Decode tokens
    let text_tokens: Vec<u32> = tokens
        .iter()
        .skip(4) // Skip special tokens
        .filter(|&&t| t < 50257)
        .cloned()
        .collect();

    println!("\nAll tokens: {:?}", &tokens);
    println!("Text tokens: {:?}", &text_tokens);

    if let Ok(text) = model.tokenizer().decode(&text_tokens) {
        println!("Text: \"{}\"", text);
    }

    Ok(())
}

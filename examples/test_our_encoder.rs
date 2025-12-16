#![allow(clippy::unwrap_used)]
//! Test transcription with our encoder output

use whisper_apr::WhisperApr;
use whisper_apr::tokenizer::special_tokens;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TEST: Transcription with Our Encoder ===\n");
    
    // Load model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;
    
    // Load audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();
    
    println!("Audio samples: {} ({:.2}s)", samples.len(), samples.len() as f32 / 16000.0);
    
    // Compute mel and encode
    let mel = model.compute_mel(&samples)?;
    println!("Mel: {} values ({} frames)", mel.len(), mel.len() / 80);
    
    let encoder_output = model.encode(&mel)?;
    println!("Encoder: {} values ({} positions)", encoder_output.len(), encoder_output.len() / 384);
    
    let enc_mean: f32 = encoder_output.iter().sum::<f32>() / encoder_output.len() as f32;
    let enc_l2: f32 = encoder_output.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Encoder stats: mean={:.4}, L2={:.4}", enc_mean, enc_l2);
    
    // Test decoding
    println!("\n=== Test 1: Standard prompt (no prefix) ===\n");
    test_decode(&mut model, &encoder_output, &[])?;
    
    println!("\n=== Test 2: Force \" The\" (440) ===\n");
    test_decode(&mut model, &encoder_output, &[440])?;
    
    println!("\n=== Test 3: Force \" The birds\" (440, 9009) ===\n");
    test_decode(&mut model, &encoder_output, &[440, 9009])?;
    
    Ok(())
}

fn test_decode(model: &mut WhisperApr, encoder_output: &[f32], prefix: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];
    tokens.extend_from_slice(prefix);
    
    println!("Starting tokens: {:?}", tokens);
    
    let suppress: Vec<u32> = (50257..51865).collect();
    
    for step in 0..6 {
        let logits = model.decoder_mut().forward(&tokens, encoder_output)?;
        let mut last_logits: Vec<f32> = logits[(tokens.len() - 1) * 51865..tokens.len() * 51865]
            .to_vec();
        
        for &s in &suppress {
            if (s as usize) < last_logits.len() {
                last_logits[s as usize] = f32::NEG_INFINITY;
            }
        }
        
        let (argmax, max_logit) = last_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let mut indexed: Vec<_> = last_logits.iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        
        print!("Step {}: {} ({:.2}) | ", step, argmax, max_logit);
        for (tok, logit) in indexed.iter().take(3) {
            if !logit.is_infinite() {
                print!("{}: {:.1}, ", tok, logit);
            }
        }
        println!();
        
        tokens.push(argmax as u32);
    }
    
    let text_tokens: Vec<u32> = tokens.iter()
        .skip(4)
        .filter(|&&t| t < 50257)
        .cloned()
        .collect();
    
    if let Ok(text) = model.tokenizer().decode(&text_tokens) {
        println!("Result: \"{}\"", text);
    }
    
    Ok(())
}

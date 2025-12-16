#![allow(clippy::unwrap_used)]
//! Test with encoder layer norm weights copied to decoder

use whisper_apr::WhisperApr;
use whisper_apr::tokenizer::special_tokens;
use whisper_apr::format::AprReader;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load APR to get encoder weights
    let model_bytes_copy = fs::read("models/whisper-tiny-fb.apr")?;
    let reader = AprReader::new(model_bytes_copy)?;
    
    let enc_ln_weight = reader.load_tensor("encoder.layer_norm.weight")?;
    let enc_ln_bias = reader.load_tensor("encoder.layer_norm.bias")?;
    
    println!("Encoder LN weight mean: {:.4}", enc_ln_weight.iter().sum::<f32>() / enc_ln_weight.len() as f32);
    println!("Encoder LN bias mean: {:.4}", enc_ln_bias.iter().sum::<f32>() / enc_ln_bias.len() as f32);
    
    // Load model
    let model_bytes = fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;
    
    // Copy encoder LN to decoder LN
    let ln = model.decoder_mut().ln_post_mut();
    let old_mean: f32 = ln.weight.iter().sum::<f32>() / ln.weight.len() as f32;
    println!("\nOriginal decoder LN weight mean: {:.4}", old_mean);
    
    ln.weight.copy_from_slice(&enc_ln_weight);
    ln.bias.copy_from_slice(&enc_ln_bias);
    
    let new_mean: f32 = ln.weight.iter().sum::<f32>() / ln.weight.len() as f32;
    println!("New decoder LN weight mean: {:.4}", new_mean);
    
    // Load audio
    let audio_bytes = fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();
    
    let mel = model.compute_mel(&samples)?;
    let encoder_output = model.encode(&mel)?;
    
    println!("\n=== Decoding with encoder LN weights ===\n");
    
    let mut tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];
    
    let suppress: Vec<u32> = (50257..51865).collect();
    
    for step in 0..15 {
        let logits = model.decoder_mut().forward(&tokens, &encoder_output)?;
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
    
    println!("\nTokens: {:?}", text_tokens);
    
    if let Ok(text) = model.tokenizer().decode(&text_tokens) {
        println!("Transcription: \"{}\"", text);
    }
    
    Ok(())
}

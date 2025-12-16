#![allow(clippy::unwrap_used)]
//! Test with forced prefix to see if model can continue correctly

use whisper_apr::WhisperApr;
use whisper_apr::tokenizer::special_tokens;
use std::fs::File;
use std::io::Read;

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
    println!("=== TEST: Forced Prefix Decoding ===\n");
    
    // Load HF encoder output  
    let hf_enc = load_npy_f32("/tmp/hf_encoder_output.npy");
    
    // Load model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;
    
    // Expected transcription: "The birds can use"
    // Tokens: " The" = 440, " birds" = 9009, " can" = 393
    
    // Test 1: Just the initial prompt
    println!("=== Test 1: Standard prompt (no prefix) ===\n");
    test_decode(&mut model, &hf_enc, &[])?;
    
    // Test 2: Force " The" (440)
    println!("\n=== Test 2: Force \" The\" (440) ===\n");
    test_decode(&mut model, &hf_enc, &[440])?;
    
    // Test 3: Force " The birds" (440, 9009)
    println!("\n=== Test 3: Force \" The birds\" (440, 9009) ===\n");
    test_decode(&mut model, &hf_enc, &[440, 9009])?;
    
    // Test 4: Force " The birds can" (440, 9009, 393)
    println!("\n=== Test 4: Force \" The birds can\" (440, 9009, 393) ===\n");
    test_decode(&mut model, &hf_enc, &[440, 9009, 393])?;
    
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

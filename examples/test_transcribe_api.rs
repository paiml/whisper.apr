#![allow(clippy::unwrap_used)]
//! Test the public transcribe API

use std::path::Path;
use whisper_apr::{DecodingStrategy, TranscribeOptions, WhisperApr};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TRANSCRIBE API TEST ===\n");

    let model_path = Path::new("models/whisper-tiny-fb.apr");
    let model_bytes = std::fs::read(model_path)?;
    let model = WhisperApr::load_from_apr(&model_bytes)?;

    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    println!(
        "Audio: {} samples ({:.2}s)\n",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    // Test with greedy decoding
    println!("=== GREEDY DECODING ===\n");
    let options_greedy = TranscribeOptions {
        language: Some("en".to_string()),
        strategy: DecodingStrategy::Greedy,
        ..Default::default()
    };

    match model.transcribe(&samples, options_greedy) {
        Ok(result) => {
            println!("Text: '{}'", result.text);
            println!("Segments: {:?}", result.segments);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    println!("\n=== BEAM SEARCH DECODING (beam_size=5) ===\n");
    let options_beam = TranscribeOptions {
        language: Some("en".to_string()),
        strategy: DecodingStrategy::BeamSearch {
            beam_size: 5,
            temperature: 0.0,
            patience: 1.0,
        },
        ..Default::default()
    };

    match model.transcribe(&samples, options_beam) {
        Ok(result) => {
            println!("Text: '{}'", result.text);
            println!("Segments: {:?}", result.segments);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    println!("\n=== GROUND TRUTH ===\n");
    println!("whisper.cpp: 'The birds can use.'");

    Ok(())
}

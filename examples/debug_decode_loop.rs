//! Debug: Trace the decode loop step by step

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DECODE LOOP DEBUG ===\n");

    let model_path = Path::new("models/whisper-tiny-int8.apr");
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

    // Now transcribe with default options and see what we get
    println!("=== RUNNING TRANSCRIBE ===\n");

    let result = model.transcribe(&samples, whisper_apr::TranscribeOptions::default())?;

    println!("Result text: {:?}", result.text);
    println!("Text length: {} chars", result.text.len());
    println!("Segments: {}", result.segments.len());

    // Show the raw bytes of the text
    println!("\nText bytes: {:?}", result.text.as_bytes());

    // Count occurrences of each byte
    let mut byte_counts = std::collections::HashMap::new();
    for b in result.text.bytes() {
        *byte_counts.entry(b).or_insert(0) += 1;
    }
    println!("\nByte frequency:");
    let mut sorted_counts: Vec<_> = byte_counts.into_iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(&a.1));
    for (byte, count) in sorted_counts.iter().take(10) {
        println!(
            "  0x{:02x} ({:?}): {}",
            byte,
            char::from_u32(*byte as u32),
            count
        );
    }

    // Try with beam search to see if it's different
    println!("\n=== BEAM SEARCH TRANSCRIBE ===\n");

    let beam_options = whisper_apr::TranscribeOptions {
        strategy: whisper_apr::DecodingStrategy::BeamSearch {
            beam_size: 5,
            temperature: 0.0,
            patience: 1.0,
        },
        ..Default::default()
    };

    let beam_result = model.transcribe(&samples, beam_options)?;

    println!("Beam result text: {:?}", beam_result.text);
    println!("Beam text length: {} chars", beam_result.text.len());

    Ok(())
}

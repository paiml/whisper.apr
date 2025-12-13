//! Basic transcription example
//!
//! Demonstrates the core whisper.apr API for transcribing audio.
//!
//! Run with: `cargo run --example basic_transcription`

use whisper_apr::{
    audio::MelFilterbank, model::ModelConfig, DecodingStrategy, Task, TranscribeOptions, WhisperApr,
};

fn main() {
    println!("=== Whisper.apr Basic Transcription Example ===\n");

    // Create a tiny model (fastest, smallest)
    let config = ModelConfig::tiny();
    println!("Model configuration:");
    println!("  Type: {:?}", config.model_type);
    println!("  Encoder layers: {}", config.n_audio_layer);
    println!("  Decoder layers: {}", config.n_text_layer);
    println!("  Audio state dim: {}", config.n_audio_state);
    println!("  Text state dim: {}", config.n_text_state);
    println!("  Attention heads: {}", config.n_audio_head);
    println!("  Vocabulary size: {}", config.n_vocab);
    println!();

    // Create WhisperApr instance
    let whisper = WhisperApr::tiny();
    println!("WhisperApr instance created: {:?}", whisper.model_type());
    println!();

    // Create transcription options
    let options = TranscribeOptions {
        language: Some("en".to_string()),
        task: Task::Transcribe,
        strategy: DecodingStrategy::Greedy,
        word_timestamps: false,
    };

    println!("Transcription options:");
    println!("  Task: {:?}", options.task);
    println!("  Language: {:?}", options.language);
    println!("  Strategy: {:?}", options.strategy);
    println!("  Word timestamps: {}", options.word_timestamps);
    println!();

    // Generate synthetic audio (1 second of 440Hz sine wave)
    let sample_rate = 16000;
    let duration_secs = 1.0;
    let frequency = 440.0;
    let audio: Vec<f32> = (0..((sample_rate as f32 * duration_secs) as usize))
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect();

    println!(
        "Generated {} samples of {}Hz tone at {}Hz sample rate",
        audio.len(),
        frequency,
        sample_rate
    );
    println!();

    // Compute mel spectrogram
    let mel = MelFilterbank::new(80, 400, sample_rate);
    match mel.compute(&audio, 160) {
        Ok(mel_spec) => {
            println!("Mel spectrogram computed:");
            println!("  Shape: {} mel frames x 80 mel bins", mel_spec.len() / 80);
            println!("  Total values: {}", mel_spec.len());

            // Calculate basic statistics
            let mean: f32 = mel_spec.iter().sum::<f32>() / mel_spec.len() as f32;
            let max = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = mel_spec.iter().copied().fold(f32::INFINITY, f32::min);
            println!("  Mean: {:.4}", mean);
            println!("  Range: [{:.4}, {:.4}]", min, max);
        }
        Err(e) => {
            println!("Error computing mel spectrogram: {}", e);
        }
    }

    println!("\n=== Example Complete ===");
    println!("Note: Full transcription requires loading a .apr model file with weights.");
    println!("See docs/specifications/whisper.apr-wasm-first-spec.md for details.");
}

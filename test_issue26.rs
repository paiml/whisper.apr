use std::path::Path;
use whisper_apr::{WhisperApr, TranscribeOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let whisper = WhisperApr::tiny();
    let audio = vec![0.0; 16000];
    let result = whisper.transcribe(&audio, TranscribeOptions::default())?;
    println!("{}", result.text);
    Ok(())
}

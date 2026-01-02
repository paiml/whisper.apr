fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();
    let model_bytes = std::fs::read("models/whisper-tiny-int8-fb.apr")?;
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;
    let result = model.transcribe(&samples, whisper_apr::TranscribeOptions::default())?;
    println!("Text: '{}' ({} chars)", result.text, result.text.len());
    Ok(())
}

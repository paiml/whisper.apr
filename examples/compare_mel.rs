//! Compare our mel spectrogram with HuggingFace's

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
    // Load HF mel
    let hf_mel = load_npy_f32("/tmp/hf_mel_full.npy");
    println!("HF mel: {} values", hf_mel.len());

    // HF mel is [80, 3000] in C-order (frame-major in their code)
    // Let me check the actual shape
    let n_mels = 80;
    let n_frames = hf_mel.len() / n_mels;
    println!("HF shape: {} mels x {} frames", n_mels, n_frames);

    // Compute our mel
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let model = WhisperApr::load_from_apr(&model_bytes)?;

    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let our_mel = model.compute_mel(&samples)?;
    println!("\nOur mel: {} values", our_mel.len());
    let our_frames = our_mel.len() / n_mels;
    println!("Our shape: {} mels x {} frames", n_mels, our_frames);

    // Stats comparison
    let hf_mean: f32 = hf_mel.iter().sum::<f32>() / hf_mel.len() as f32;
    let hf_min = hf_mel.iter().cloned().fold(f32::INFINITY, f32::min);
    let hf_max = hf_mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let our_mean: f32 = our_mel.iter().sum::<f32>() / our_mel.len() as f32;
    let our_min = our_mel.iter().cloned().fold(f32::INFINITY, f32::min);
    let our_max = our_mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\nStats comparison:");
    println!("         HF        Ours      Diff");
    println!(
        "Mean:    {:.4}    {:.4}    {:.4}",
        hf_mean,
        our_mean,
        our_mean - hf_mean
    );
    println!(
        "Min:     {:.4}    {:.4}    {:.4}",
        hf_min,
        our_min,
        our_min - hf_min
    );
    println!(
        "Max:     {:.4}    {:.4}    {:.4}",
        hf_max,
        our_max,
        our_max - hf_max
    );

    // HF mel is stored as [80, 3000] in C-order
    // Index as hf_mel[mel_idx * 3000 + frame_idx]
    // Our mel is stored as [3000, 80] (frame-major)
    // Index as our_mel[frame_idx * 80 + mel_idx]

    println!("\nFirst frame comparison (mel 0-9):");
    println!("HF:  ");
    for mel_idx in 0..10 {
        print!("{:.3} ", hf_mel[mel_idx * n_frames + 0]); // HF: [mel][frame]
    }
    println!();
    println!("Ours:");
    for mel_idx in 0..10 {
        print!("{:.3} ", our_mel[0 * n_mels + mel_idx]); // Ours: [frame][mel]
    }
    println!();

    // Per-position difference
    let mut total_diff = 0.0_f32;
    let mut max_diff = 0.0_f32;
    let mut max_diff_pos = (0, 0);

    for frame in 0..n_frames.min(our_frames) {
        for mel in 0..n_mels {
            let hf_val = hf_mel[mel * n_frames + frame]; // HF: [mel][frame]
            let our_val = our_mel[frame * n_mels + mel]; // Ours: [frame][mel]
            let diff = (hf_val - our_val).abs();
            total_diff += diff;
            if diff > max_diff {
                max_diff = diff;
                max_diff_pos = (frame, mel);
            }
        }
    }

    let avg_diff = total_diff / (n_frames.min(our_frames) * n_mels) as f32;
    println!("\nPer-element comparison:");
    println!("Average absolute difference: {:.4}", avg_diff);
    println!(
        "Maximum difference: {:.4} at frame={}, mel={}",
        max_diff, max_diff_pos.0, max_diff_pos.1
    );

    // Compare specific positions
    println!("\nFrame 0 detailed comparison (all 80 mels):");
    let mut diffs = Vec::new();
    for mel in 0..n_mels {
        let hf_val = hf_mel[mel * n_frames + 0];
        let our_val = our_mel[0 * n_mels + mel];
        let diff = hf_val - our_val;
        diffs.push((mel, hf_val, our_val, diff));
    }

    // Show first 10 and last 10 mels
    println!("Mel  HF        Ours      Diff");
    for (mel, hf, ours, diff) in &diffs[..10] {
        println!("{:3}  {:+.4}    {:+.4}    {:+.4}", mel, hf, ours, diff);
    }
    println!("...");
    for (mel, hf, ours, diff) in &diffs[70..] {
        println!("{:3}  {:+.4}    {:+.4}    {:+.4}", mel, hf, ours, diff);
    }

    Ok(())
}

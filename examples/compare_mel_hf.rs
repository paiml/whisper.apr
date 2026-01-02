#![allow(clippy::unwrap_used)]
//! Compare our mel spectrogram against HuggingFace

use std::fs::File;
use std::io::Read;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MEL COMPARISON ===\n");

    // Load HF mel output (properly transposed)
    let hf_mel = load_npy("/tmp/hf_mel_proper.npy")?;
    println!(
        "HF mel shape: {} values ({} frames x 80)",
        hf_mel.len(),
        hf_mel.len() / 80
    );

    // Load our model and compute mel
    let model_path = Path::new("models/whisper-tiny-fb.apr");
    let model_bytes = std::fs::read(model_path)?;
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let our_mel = model.compute_mel(&samples)?;
    println!(
        "Our mel shape: {} values ({} frames x 80)\n",
        our_mel.len(),
        our_mel.len() / 80
    );

    // Statistics
    let hf_mean: f32 = hf_mel.iter().sum::<f32>() / hf_mel.len() as f32;
    let our_mean: f32 = our_mel.iter().sum::<f32>() / our_mel.len() as f32;

    let hf_std: f32 =
        (hf_mel.iter().map(|x| (x - hf_mean).powi(2)).sum::<f32>() / hf_mel.len() as f32).sqrt();
    let our_std: f32 =
        (our_mel.iter().map(|x| (x - our_mean).powi(2)).sum::<f32>() / our_mel.len() as f32).sqrt();

    println!("Statistics:");
    println!("  HF  mean: {:.6}, std: {:.6}", hf_mean, hf_std);
    println!("  Ours mean: {:.6}, std: {:.6}", our_mean, our_std);
    println!(
        "  HF  min: {:.6}, max: {:.6}",
        hf_mel.iter().cloned().fold(f32::INFINITY, f32::min),
        hf_mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  Ours min: {:.6}, max: {:.6}",
        our_mel.iter().cloned().fold(f32::INFINITY, f32::min),
        our_mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Compare specific frames
    let n_mels = 80;
    let frames = [0, 100, 1500, 2999];

    println!("\nSample frame values (first 4 mel bins):");
    for &frame in &frames {
        let hf_start = frame * n_mels;
        let our_start = frame * n_mels;

        if hf_start + 4 > hf_mel.len() || our_start + 4 > our_mel.len() {
            continue;
        }

        let hf_slice = &hf_mel[hf_start..hf_start + 4];
        let our_slice = &our_mel[our_start..our_start + 4];

        println!("\n  Frame {}:", frame);
        println!(
            "    HF:   [{:.4}, {:.4}, {:.4}, {:.4}]",
            hf_slice[0], hf_slice[1], hf_slice[2], hf_slice[3]
        );
        println!(
            "    Ours: [{:.4}, {:.4}, {:.4}, {:.4}]",
            our_slice[0], our_slice[1], our_slice[2], our_slice[3]
        );

        let diff: f32 = hf_slice
            .iter()
            .zip(our_slice.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 4.0;
        println!("    Avg diff: {:.6}", diff);
    }

    // Correlation
    let compare_len = hf_mel.len().min(our_mel.len());
    let correlation = compute_correlation(&hf_mel[..compare_len], &our_mel[..compare_len]);
    println!("\nOverall Pearson correlation: {:.6}", correlation);

    // Find max diff
    let mut max_diff: f32 = 0.0;
    let mut max_diff_pos = 0;
    for i in 0..compare_len {
        let diff = (hf_mel[i] - our_mel[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_pos = i;
        }
    }
    println!(
        "Max diff: {:.6} at position {} (frame {}, mel bin {})",
        max_diff,
        max_diff_pos,
        max_diff_pos / 80,
        max_diff_pos % 80
    );

    Ok(())
}

fn load_npy(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    if &data[0..6] != b"\x93NUMPY" {
        return Err("Not a valid .npy file".into());
    }

    let version_major = data[6];
    let header_len = if version_major == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };

    let header_start = if version_major == 1 { 10 } else { 12 };
    let data_start = header_start + header_len;

    let float_data: Vec<f32> = data[data_start..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(float_data)
}

fn compute_correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let mean_a: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_b: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n;

    let mut cov: f64 = 0.0;
    let mut var_a: f64 = 0.0;
    let mut var_b: f64 = 0.0;

    for i in 0..a.len() {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    cov / (var_a.sqrt() * var_b.sqrt())
}

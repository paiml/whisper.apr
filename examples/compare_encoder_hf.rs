#![allow(clippy::unwrap_used)]
//! Compare our encoder output against HuggingFace

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ENCODER COMPARISON ===\n");

    // Load HF encoder output (saved as .npy)
    println!("Loading HuggingFace encoder output...");
    let hf_encoder = load_npy("/tmp/hf_encoder_output.npy")?;
    println!(
        "HF encoder shape: {} values ({} positions x 384)",
        hf_encoder.len(),
        hf_encoder.len() / 384
    );

    // Load our model and compute encoder output
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

    let mel = model.compute_mel(&samples)?;
    let our_encoder = model.encode(&mel)?;

    println!(
        "Our encoder shape: {} values ({} positions x 384)\n",
        our_encoder.len(),
        our_encoder.len() / 384
    );

    // Compare statistics
    let hf_mean: f32 = hf_encoder.iter().sum::<f32>() / hf_encoder.len() as f32;
    let our_mean: f32 = our_encoder.iter().sum::<f32>() / our_encoder.len() as f32;

    let hf_std: f32 = (hf_encoder
        .iter()
        .map(|x| (x - hf_mean).powi(2))
        .sum::<f32>()
        / hf_encoder.len() as f32)
        .sqrt();
    let our_std: f32 = (our_encoder
        .iter()
        .map(|x| (x - our_mean).powi(2))
        .sum::<f32>()
        / our_encoder.len() as f32)
        .sqrt();

    println!("Statistics:");
    println!("  HF  mean: {:.6}, std: {:.6}", hf_mean, hf_std);
    println!("  Ours mean: {:.6}, std: {:.6}", our_mean, our_std);

    // Compare specific positions
    let d_model = 384;
    let positions = [0, 100, 750, 1499];

    println!("\nSample values comparison:");
    for &pos in &positions {
        let hf_slice = &hf_encoder[pos * d_model..pos * d_model + 4];
        let our_slice = &our_encoder[pos * d_model..pos * d_model + 4];

        println!("\n  Position {}:", pos);
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

    // Overall difference
    println!("\nOverall comparison:");
    let mut total_diff: f64 = 0.0;
    let mut max_diff: f32 = 0.0;
    let mut max_diff_pos = 0;

    let compare_len = hf_encoder.len().min(our_encoder.len());
    for i in 0..compare_len {
        let diff = (hf_encoder[i] - our_encoder[i]).abs();
        total_diff += diff as f64;
        if diff > max_diff {
            max_diff = diff;
            max_diff_pos = i;
        }
    }

    let avg_diff = total_diff / compare_len as f64;
    println!("  Average abs diff: {:.6}", avg_diff);
    println!("  Max diff: {:.6} at position {}", max_diff, max_diff_pos);

    // Correlation
    let correlation = compute_correlation(&hf_encoder[..compare_len], &our_encoder[..compare_len]);
    println!("  Pearson correlation: {:.6}", correlation);

    if correlation > 0.99 {
        println!("\n  ✓ Encoder outputs are highly correlated!");
    } else if correlation > 0.9 {
        println!("\n  ⚠ Encoder outputs have moderate correlation");
    } else {
        println!("\n  ✗ Encoder outputs diverge significantly!");
    }

    Ok(())
}

fn load_npy(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use std::io::Read;

    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    // Parse NumPy .npy format
    // Magic: \x93NUMPY
    if &data[0..6] != b"\x93NUMPY" {
        return Err("Not a valid .npy file".into());
    }

    let version_major = data[6];
    let version_minor = data[7];

    let header_len = if version_major == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };

    let header_start = if version_major == 1 { 10 } else { 12 };
    let data_start = header_start + header_len;

    // Parse as f32 array
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

//! H23: Test Conv1d input layout hypothesis
//!
//! PyTorch Conv1d expects [batch, channels, time] = [1, 80, 3000]
//! Our Conv1d receives [time, channels] = [3000, 80]
//!
//! Question: Is our mel data correctly oriented for the conv weights?
//!
//! If weights were trained expecting mel[channel][time] but we pass mel[time][channel],
//! the conv will compute wrong features, explaining why Q prefers padding K.

use std::path::Path;

fn stats(data: &[f32]) -> (f32, f32, f32, f32) {
    let n = data.len() as f32;
    let sum: f32 = data.iter().sum();
    let mean = sum / n;
    let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min, max)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if na > 1e-10 && nb > 1e-10 {
        dot / (na * nb)
    } else {
        0.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== H23: CONV1D INPUT LAYOUT VERIFICATION ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    let model_bytes = std::fs::read(model_path)?;
    let mut model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Load test audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    println!("[STEP 1: Compute mel spectrogram]\n");
    let mel = model.compute_mel(&samples)?;
    let n_mels = 80;
    let n_frames = mel.len() / n_mels;
    println!("  Mel shape: {} total values", mel.len());
    println!("  Interpretation as [frames, mels]: {} × {}", n_frames, n_mels);
    println!("  Interpretation as [mels, frames]: {} × {}", n_mels, n_frames);

    // Check current layout: is mel[i] = mel[frame * n_mels + mel_idx] or mel[mel_idx * n_frames + frame]?
    println!("\n[STEP 2: Verify mel layout]\n");

    // Frame 0 should have audio content, frame 1400 should be padding
    // Audio region should have higher variance per-frame

    // If layout is [frames, mels]: mel[frame * 80 .. (frame+1) * 80] is one frame
    let frame0_if_frames_first = &mel[0..n_mels];
    let frame100_if_frames_first = &mel[100 * n_mels..(100 + 1) * n_mels];
    let frame1400_if_frames_first = &mel[1400 * n_mels..(1400 + 1) * n_mels];

    let (f0_mean, f0_std, _, _) = stats(frame0_if_frames_first);
    let (f100_mean, f100_std, _, _) = stats(frame100_if_frames_first);
    let (f1400_mean, f1400_std, _, _) = stats(frame1400_if_frames_first);

    println!("  If [frames, mels] layout:");
    println!("    Frame 0 (0ms):    mean={:+.4}  std={:.4}", f0_mean, f0_std);
    println!("    Frame 100 (2s):   mean={:+.4}  std={:.4}", f100_mean, f100_std);
    println!("    Frame 1400 (28s): mean={:+.4}  std={:.4}", f1400_mean, f1400_std);

    // If layout is [mels, frames]: mel[mel_idx * n_frames .. (mel_idx+1) * n_frames] is one mel bin across all frames
    // Frame 0 would be mel[0], mel[n_frames], mel[2*n_frames], ...
    let mut frame0_if_mels_first = Vec::with_capacity(n_mels);
    let mut frame100_if_mels_first = Vec::with_capacity(n_mels);
    let mut frame1400_if_mels_first = Vec::with_capacity(n_mels);

    for mel_idx in 0..n_mels {
        frame0_if_mels_first.push(mel[mel_idx * n_frames + 0]);
        frame100_if_mels_first.push(mel[mel_idx * n_frames + 100]);
        frame1400_if_mels_first.push(mel[mel_idx * n_frames + 1400]);
    }

    let (f0_mean_alt, f0_std_alt, _, _) = stats(&frame0_if_mels_first);
    let (f100_mean_alt, f100_std_alt, _, _) = stats(&frame100_if_mels_first);
    let (f1400_mean_alt, f1400_std_alt, _, _) = stats(&frame1400_if_mels_first);

    println!("\n  If [mels, frames] layout:");
    println!("    Frame 0 (0ms):    mean={:+.4}  std={:.4}", f0_mean_alt, f0_std_alt);
    println!("    Frame 100 (2s):   mean={:+.4}  std={:.4}", f100_mean_alt, f100_std_alt);
    println!("    Frame 1400 (28s): mean={:+.4}  std={:.4}", f1400_mean_alt, f1400_std_alt);

    // The correct layout should show:
    // - Audio frames (0-75) have higher variance/different mean than padding frames (>75)
    // - Padding frames should be nearly constant (low std)

    println!("\n[STEP 3: Layout determination]\n");

    let frames_first_variance_diff = (f0_std - f1400_std).abs();
    let mels_first_variance_diff = (f0_std_alt - f1400_std_alt).abs();

    println!("  Variance difference (audio vs padding):");
    println!("    [frames, mels] layout: {:.4}", frames_first_variance_diff);
    println!("    [mels, frames] layout: {:.4}", mels_first_variance_diff);

    if frames_first_variance_diff > mels_first_variance_diff {
        println!("\n  → Layout appears to be [frames, mels] (frames first)");
        println!("    This means mel[i*80 + j] = frame i, mel bin j");
    } else {
        println!("\n  → Layout appears to be [mels, frames] (mels first)");
        println!("    This means mel[i*3000 + j] = mel bin i, frame j");
    }

    // Now check what PyTorch Conv1d expects
    println!("\n[STEP 4: Conv1d weight analysis]\n");
    println!("  PyTorch Conv1d input shape: [batch=1, channels=80, time=3000]");
    println!("  PyTorch memory layout: contiguous over time dimension first");
    println!("  → For each channel c, values mel[0,c,0], mel[0,c,1], ... are contiguous");
    println!("  → This is effectively [mels, frames] in 2D terms\n");

    println!("  Our Conv1d expects: [time, channels] = [3000, 80]");
    println!("  → For each time t, values mel[t,0], mel[t,1], ... are contiguous");
    println!("  → This is [frames, mels] in 2D terms");

    println!("\n[STEP 5: Mismatch implications]\n");

    // If our mel is [frames, mels] but PyTorch expects [mels, frames]:
    // The conv1d will read wrong data!
    // When conv tries to read mel bin 0 across time, it actually reads frame 0 across mels

    println!("  If our mel is [frames, mels] but conv weights expect [mels, frames]:");
    println!("    - Conv kernel for mel bin 0 across 3 timesteps would read:");
    println!("      EXPECTED: mel[0,0], mel[0,1], mel[0,2] (same mel bin, consecutive frames)");
    println!("      ACTUAL:   mel[0,0], mel[0,1], mel[0,2] = mel bin 0,1,2 of frame 0!");
    println!("    - This is a 90° rotation of the data!");

    // Test by running encoder and checking Q·K alignment
    println!("\n[STEP 6: Current encoder behavior]\n");
    let encoded = model.encode(&mel)?;
    let d_model = 384;
    let enc_len = encoded.len() / d_model;

    // Check variance at audio vs padding positions
    let audio_positions = [0, 10, 20, 30, 37];
    let padding_positions = [1300, 1400, 1450, 1487];

    println!("  Encoder output variance at audio positions (0-75):");
    for &pos in &audio_positions {
        let start = pos * d_model;
        let end = start + d_model;
        let (mean, std, _, _) = stats(&encoded[start..end]);
        println!("    Pos {:4}: mean={:+.4}  std={:.4}", pos, mean, std);
    }

    println!("\n  Encoder output variance at padding positions (>75):");
    for &pos in &padding_positions {
        if pos < enc_len {
            let start = pos * d_model;
            let end = start + d_model;
            let (mean, std, _, _) = stats(&encoded[start..end]);
            println!("    Pos {:4}: mean={:+.4}  std={:.4}", pos, mean, std);
        }
    }

    // Compare cosine similarity between audio and padding encoder outputs
    println!("\n[STEP 7: Encoder output similarity analysis]\n");

    let enc_pos0 = &encoded[0..d_model];
    let enc_pos37 = &encoded[37 * d_model..38 * d_model];
    let enc_pos1400 = &encoded[1400 * d_model..1401 * d_model];
    let enc_pos1487 = &encoded[1487 * d_model..1488 * d_model];

    println!("  Cosine similarities:");
    println!("    Audio  pos0 ↔ pos37:   {:.4}", cosine_sim(enc_pos0, enc_pos37));
    println!("    Padding pos1400 ↔ pos1487: {:.4}", cosine_sim(enc_pos1400, enc_pos1487));
    println!("    Cross: pos0 ↔ pos1400: {:.4}", cosine_sim(enc_pos0, enc_pos1400));
    println!("    Cross: pos37 ↔ pos1487: {:.4}", cosine_sim(enc_pos37, enc_pos1487));

    // High similarity within padding region + low cross-similarity suggests
    // encoder is producing uniform/collapsed output for padding
    // But decoder Q preferring padding K suggests the audio region output is "wrong"

    println!("\n=== DIAGNOSIS ===\n");

    // Final determination
    if frames_first_variance_diff > mels_first_variance_diff {
        println!("  Our mel layout: [frames, mels] (frame-major)");
        println!("  PyTorch expects: [mels, frames] (channel-major)");
        println!("\n  ⚠️  LAYOUT MISMATCH DETECTED!");
        println!("  The conv1d weights expect input where:");
        println!("    - Consecutive memory positions = same mel bin, different times");
        println!("  But our mel has:");
        println!("    - Consecutive memory positions = same time, different mel bins");
        println!("\n  FIX: Transpose mel from [3000, 80] to [80, 3000] before conv");
    } else {
        println!("  Layout appears correct (both [mels, frames])");
        println!("  The issue is elsewhere...");
    }

    Ok(())
}

#![allow(clippy::unwrap_used)]
//! Compare with HuggingFace using padded mel (30 sec)
//!
//! Test hypothesis: padding mel to 3000 frames fixes the +13 shift

use std::fs::File;
use std::io::Read;
use whisper_apr::tokenizer::special_tokens;
use whisper_apr::WhisperApr;

fn load_npy_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    if &buf[0..6] != b"\x93NUMPY" {
        return Err("Not a numpy file".into());
    }

    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let data_start = 10 + header_len;
    let header = std::str::from_utf8(&buf[10..data_start])?;

    let is_f64 = header.contains("float64") || header.contains("<f8");
    let is_f32 = header.contains("float32") || header.contains("<f4");

    let data = &buf[data_start..];

    if is_f64 {
        let f64_values: Vec<f64> = data.chunks(8)
            .map(|chunk| f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7]
            ]))
            .collect();
        Ok(f64_values.iter().map(|&x| x as f32).collect())
    } else if is_f32 {
        Ok(data.chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    } else {
        Err(format!("Unknown dtype: {}", header).into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== COMPARING WITH HF (PADDED MEL) ===\n");

    // Load HuggingFace full mel (80, 3000)
    let hf_mel_full = load_npy_f32("/tmp/hf_mel_full.npy")?;
    println!("HF full mel: {} values ({} frames)", hf_mel_full.len(), hf_mel_full.len() / 80);

    let hf_encoder = load_npy_f32("/tmp/hf_encoder_output.npy")?;
    let hf_logits = load_npy_f32("/tmp/hf_logits.npy")?;

    // Load our model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    // Load audio and compute mel
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let our_mel = model.compute_mel(&samples)?;
    println!("Our mel: {} values ({} frames)", our_mel.len(), our_mel.len() / 80);

    // Pad our mel to 3000 frames like HuggingFace
    // HuggingFace uses -1.0 as padding value (log_spec padding)
    let n_mels = 80;
    let target_frames = 3000;
    let mut padded_mel = vec![-1.0f32; target_frames * n_mels];

    // Our mel is stored as [frame0_mel0, frame0_mel1, ..., frame0_mel79, frame1_mel0, ...]
    // HF mel is stored as [mel0_frame0, mel0_frame1, ..., mel0_frame2999, mel1_frame0, ...]
    // So HF is (80, 3000) and ours is (frames, 80)

    // We need to transpose our mel to match HF format
    let our_frames = our_mel.len() / n_mels;
    for mel_idx in 0..n_mels {
        for frame_idx in 0..our_frames {
            let our_idx = frame_idx * n_mels + mel_idx;
            let padded_idx = mel_idx * target_frames + frame_idx;
            padded_mel[padded_idx] = our_mel[our_idx];
        }
        // Rest remains -1.0 (padding)
    }

    println!("Padded mel: {} values ({} frames)", padded_mel.len(), padded_mel.len() / n_mels);

    // Compare mel values at same positions
    println!("\n=== MEL COMPARISON (FIRST 10 FRAMES) ===\n");
    for frame in 0..10 {
        let hf_idx = 0 * 3000 + frame; // First mel bin, frame N
        let our_idx = 0 * 3000 + frame; // Same
        println!("Frame {} mel[0]: HF={:.4}, ours={:.4}", frame, hf_mel_full[hf_idx], padded_mel[our_idx]);
    }

    // Compare mel statistics
    println!("\n=== MEL STATS (NON-PADDED REGION) ===\n");
    let mut hf_audio_region = Vec::new();
    let mut our_audio_region = Vec::new();
    for m in 0..n_mels {
        for f in 0..148 {
            hf_audio_region.push(hf_mel_full[m * 3000 + f]);
            our_audio_region.push(padded_mel[m * 3000 + f]);
        }
    }

    let hf_mean: f32 = hf_audio_region.iter().sum::<f32>() / hf_audio_region.len() as f32;
    let our_mean: f32 = our_audio_region.iter().sum::<f32>() / our_audio_region.len() as f32;
    println!("HF audio region mean:  {:.4}", hf_mean);
    println!("Our audio region mean: {:.4}", our_mean);

    // Encode with padded mel
    println!("\n=== ENCODER (WITH PADDED MEL) ===\n");

    // For the encoder, the input should be (80, 3000) which is what we have in padded_mel
    // But our encoder expects (frames, 80) format
    // So transpose back
    let mut padded_for_encoder = vec![0.0f32; 3000 * 80];
    for frame in 0..3000 {
        for mel in 0..80 {
            let src_idx = mel * 3000 + frame;  // (80, 3000) format
            let dst_idx = frame * 80 + mel;     // (3000, 80) format
            padded_for_encoder[dst_idx] = padded_mel[src_idx];
        }
    }

    let encoded = model.encode(&padded_for_encoder)?;
    println!("Our encoder (padded): {} values", encoded.len());
    println!("HF encoder: {} values", hf_encoder.len());

    if encoded.len() == hf_encoder.len() {
        let enc_dot: f32 = encoded.iter().zip(hf_encoder.iter()).map(|(a, b)| a * b).sum();
        let enc_norm_a: f32 = encoded.iter().map(|x| x * x).sum::<f32>().sqrt();
        let enc_norm_b: f32 = hf_encoder.iter().map(|x| x * x).sum::<f32>().sqrt();
        let enc_sim = enc_dot / (enc_norm_a * enc_norm_b);
        println!("Encoder cosine similarity: {:.6}", enc_sim);
    }

    let our_enc_mean: f32 = encoded.iter().sum::<f32>() / encoded.len() as f32;
    let hf_enc_mean: f32 = hf_encoder.iter().sum::<f32>() / hf_encoder.len() as f32;
    println!("Our encoder mean: {:.4}, HF encoder mean: {:.4}", our_enc_mean, hf_enc_mean);

    // Decode
    println!("\n=== DECODER (WITH PADDED MEL ENCODER OUTPUT) ===\n");

    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    let (logits, _trace) = model
        .decoder_mut()
        .forward_traced(&initial_tokens, &encoded)?;

    let last_logits: &[f32] = &logits[3 * 51865..];

    let our_logits_mean: f32 = last_logits.iter().sum::<f32>() / last_logits.len() as f32;
    let hf_logits_mean: f32 = hf_logits.iter().sum::<f32>() / hf_logits.len() as f32;

    println!("Our logits mean: {:.4}", our_logits_mean);
    println!("HF logits mean:  {:.4}", hf_logits_mean);
    println!("SHIFT: {:.4}", our_logits_mean - hf_logits_mean);

    // Top tokens
    let mut our_indexed: Vec<(usize, f32)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    our_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut hf_indexed: Vec<(usize, f32)> = hf_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    hf_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 tokens:");
    println!("{:>6} {:>12} | {:>6} {:>12}", "Ours", "logit", "HF", "logit");
    for i in 0..5 {
        println!("{:>6} {:>12.4} | {:>6} {:>12.4}",
            our_indexed[i].0, our_indexed[i].1,
            hf_indexed[i].0, hf_indexed[i].1);
    }

    Ok(())
}

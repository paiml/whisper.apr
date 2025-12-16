#![allow(clippy::unwrap_used)]
//! First-principles layer-by-layer comparison for falsification
//!
//! Hypothesis testing at each layer:
//! H0: WAV samples match reference
//! H1: Mel spectrogram matches reference (within tolerance)
//! H2: Encoder output has expected statistics
//! H3: Decoder logits produce meaningful tokens

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   FIRST-PRINCIPLES LAYER COMPARISON (FALSIFICATION)        ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════════
    // LAYER 0: RAW AUDIO SAMPLES
    // ═══════════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("LAYER 0: RAW AUDIO SAMPLES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let audio_path = "demos/test-audio/test-speech-1.5s.wav";
    let audio_bytes = std::fs::read(audio_path)?;

    // Parse WAV header
    let riff = &audio_bytes[0..4];
    let wave = &audio_bytes[8..12];
    let fmt_chunk_size = u32::from_le_bytes([audio_bytes[16], audio_bytes[17], audio_bytes[18], audio_bytes[19]]);
    let audio_format = u16::from_le_bytes([audio_bytes[20], audio_bytes[21]]);
    let num_channels = u16::from_le_bytes([audio_bytes[22], audio_bytes[23]]);
    let sample_rate = u32::from_le_bytes([audio_bytes[24], audio_bytes[25], audio_bytes[26], audio_bytes[27]]);
    let bits_per_sample = u16::from_le_bytes([audio_bytes[34], audio_bytes[35]]);

    println!("WAV Header Analysis:");
    println!("  RIFF marker: {:?} (expected: RIFF)", String::from_utf8_lossy(riff));
    println!("  WAVE marker: {:?} (expected: WAVE)", String::from_utf8_lossy(wave));
    println!("  Format chunk size: {} (expected: 16 for PCM)", fmt_chunk_size);
    println!("  Audio format: {} (expected: 1 for PCM)", audio_format);
    println!("  Channels: {} (expected: 1 for mono)", num_channels);
    println!("  Sample rate: {} Hz (expected: 16000)", sample_rate);
    println!("  Bits per sample: {} (expected: 16)", bits_per_sample);

    // Extract samples (skip header - find data chunk)
    let mut data_offset = 12;
    while data_offset < audio_bytes.len() - 8 {
        let chunk_id = &audio_bytes[data_offset..data_offset+4];
        let chunk_size = u32::from_le_bytes([
            audio_bytes[data_offset+4], audio_bytes[data_offset+5],
            audio_bytes[data_offset+6], audio_bytes[data_offset+7]
        ]);
        if chunk_id == b"data" {
            data_offset += 8;
            break;
        }
        data_offset += 8 + chunk_size as usize;
    }

    let samples: Vec<f32> = audio_bytes[data_offset..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    // Statistical analysis of samples
    let n = samples.len() as f64;
    let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / n;
    let variance: f64 = samples.iter().map(|&x| ((x as f64) - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let min = samples.iter().fold(f32::INFINITY, |a: f32, &b| a.min(b));
    let max = samples.iter().fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));

    println!("\nSample Statistics:");
    println!("  Count: {} samples ({:.3}s at 16kHz)", samples.len(), samples.len() as f64 / 16000.0);
    println!("  Mean: {:.6} (expected: ~0 for AC audio)", mean);
    println!("  Std Dev: {:.6} (expected: >0.01 for non-silent)", std_dev);
    println!("  Min: {:.6}, Max: {:.6}", min, max);
    println!("  Dynamic range: {:.2} dB", 20.0 * (max / min.abs()).log10());

    // H0 falsification test
    let h0_pass = sample_rate == 16000
        && num_channels == 1
        && bits_per_sample == 16
        && std_dev > 0.01;
    println!("\n✓ H0 (Valid Audio): {} - {}",
        if h0_pass { "PASS" } else { "FAIL" },
        if h0_pass { "Audio samples are valid" } else { "Audio may be corrupt or silent" }
    );

    // ═══════════════════════════════════════════════════════════════════
    // LAYER 1: MEL SPECTROGRAM
    // ═══════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("LAYER 1: MEL SPECTROGRAM");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Load model for mel computation
    let model_path = Path::new("models/whisper-tiny-int8.apr");
    let model_bytes = std::fs::read(model_path)?;
    let model = whisper_apr::WhisperApr::load_from_apr(&model_bytes)?;

    // Constants for Whisper tiny
    let n_mels = 80usize;
    let d_model = 384usize;

    let mel_data = model.compute_mel(&samples)?;
    let n_frames = mel_data.len() / n_mels;

    println!("Mel Spectrogram Dimensions:");
    println!("  N_mels: {} (expected: 80)", n_mels);
    println!("  N_frames: {} (expected: ~150 for 1.5s @ 100fps)", n_frames);
    println!("  Total values: {}", mel_data.len());

    // Mel statistics
    let mel_n = mel_data.len() as f64;
    let mel_mean: f64 = mel_data.iter().map(|&x| x as f64).sum::<f64>() / mel_n;
    let mel_var: f64 = mel_data.iter().map(|&x| ((x as f64) - mel_mean).powi(2)).sum::<f64>() / mel_n;
    let mel_std = mel_var.sqrt();
    let mel_min = mel_data.iter().fold(f32::INFINITY, |a: f32, &b| a.min(b));
    let mel_max = mel_data.iter().fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));

    println!("\nMel Statistics:");
    println!("  Mean: {:.6}", mel_mean);
    println!("  Std Dev: {:.6}", mel_std);
    println!("  Min: {:.6}, Max: {:.6}", mel_min, mel_max);

    // Check for whisper.cpp-style normalization
    // Expected: values in [-1, 1] range after (x + 4) / 4 normalization
    let in_normalized_range = mel_min >= -2.0 && mel_max <= 2.0;
    println!("\nNormalization Check:");
    println!("  In [-2, 2] range: {} (whisper.cpp uses (x+4)/4 → [-1,1])", in_normalized_range);

    // Check mel layout by examining first frame vs first mel bin
    println!("\nLayout Analysis (first 10 values):");
    println!("  Raw order: {:?}", &mel_data[..10.min(mel_data.len())]);

    // Check variance across frames (time axis) - should vary for speech
    let mut frame_means = Vec::new();
    for frame in 0..n_frames.min(10) {
        let frame_start = frame * n_mels;
        let frame_end = frame_start + n_mels;
        if frame_end <= mel_data.len() {
            let frame_mean: f32 = mel_data[frame_start..frame_end].iter().sum::<f32>() / n_mels as f32;
            frame_means.push(frame_mean);
        }
    }
    println!("  Frame means (first 10): {:?}", frame_means);

    // Check variance across mel bins - should show formant structure
    let mut mel_bin_means = Vec::new();
    for bin in 0..n_mels.min(10) {
        let mut bin_sum = 0.0f32;
        for frame in 0..n_frames {
            let idx = frame * n_mels + bin;
            if idx < mel_data.len() {
                bin_sum += mel_data[idx];
            }
        }
        mel_bin_means.push(bin_sum / n_frames as f32);
    }
    println!("  Mel bin means (first 10): {:?}", mel_bin_means);

    // H1: Mel spectrogram validity
    let h1_pass = n_mels == 80
        && n_frames > 100
        && mel_std > 0.01
        && !mel_data.iter().any(|&x: &f32| x.is_nan() || x.is_infinite());
    println!("\n✓ H1 (Valid Mel): {} - {}",
        if h1_pass { "PASS" } else { "FAIL" },
        if h1_pass { "Mel spectrogram has valid shape and variance" } else { "Mel spectrogram may be malformed" }
    );

    // Dump first frame for external comparison
    println!("\n📁 Dumping mel spectrogram for external comparison...");
    dump_mel_to_file(&mel_data, n_mels, n_frames, "/tmp/whisper_apr_mel.bin")?;
    println!("  Saved to /tmp/whisper_apr_mel.bin (raw f32, frame-major)");

    // ═══════════════════════════════════════════════════════════════════
    // LAYER 2: ENCODER OUTPUT
    // ═══════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("LAYER 2: ENCODER OUTPUT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let enc_data = model.encode(&mel_data)?;
    let enc_seq_len = enc_data.len() / d_model;

    println!("Encoder Output Dimensions:");
    println!("  Shape: {} x {}", enc_seq_len, d_model);
    println!("  Total values: {}", enc_data.len());

    // Encoder statistics
    let enc_n = enc_data.len() as f64;
    let enc_mean: f64 = enc_data.iter().map(|&x| x as f64).sum::<f64>() / enc_n;
    let enc_var: f64 = enc_data.iter().map(|&x| ((x as f64) - enc_mean).powi(2)).sum::<f64>() / enc_n;
    let enc_std = enc_var.sqrt();
    let enc_min = enc_data.iter().fold(f32::INFINITY, |a: f32, &b| a.min(b));
    let enc_max = enc_data.iter().fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));

    println!("\nEncoder Statistics:");
    println!("  Mean: {:.6} (expected: ~0 for layer-normed)", enc_mean);
    println!("  Std Dev: {:.6} (expected: ~1 for layer-normed)", enc_std);
    println!("  Min: {:.6}, Max: {:.6}", enc_min, enc_max);

    // Check for layer norm signature (mean ~0, std ~1)
    let layer_normed = enc_mean.abs() < 0.5 && enc_std > 0.1 && enc_std < 10.0;
    println!("\n  Layer norm signature: {}", if layer_normed { "YES" } else { "NO (SUSPICIOUS)" });

    // Check for collapsed representations (all same value)
    let unique_approx = enc_data.iter()
        .map(|&x| (x * 1000.0) as i32)
        .collect::<std::collections::HashSet<_>>()
        .len();
    println!("  Unique values (approx): {} (expected: many thousands)", unique_approx);

    // H2: Encoder validity
    let h2_pass = layer_normed && unique_approx > 1000;
    println!("\n✓ H2 (Valid Encoder): {} - {}",
        if h2_pass { "PASS" } else { "FAIL" },
        if h2_pass { "Encoder output has valid statistics" } else { "Encoder output may be collapsed/invalid" }
    );

    // Dump encoder output
    dump_encoder_to_file(&enc_data, enc_seq_len, d_model, "/tmp/whisper_apr_encoder.bin")?;
    println!("  Saved to /tmp/whisper_apr_encoder.bin");

    // ═══════════════════════════════════════════════════════════════════
    // LAYER 3: DECODER FIRST TOKEN LOGITS
    // ═══════════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("LAYER 3: DECODER FIRST TOKEN LOGITS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    use std::cell::RefCell;
    use whisper_apr::model::DecoderKVCache;
    use whisper_apr::tokenizer::special_tokens;

    let n_vocab = 51865usize;
    let max_tokens = 448usize;
    let n_layers = 4usize;

    let initial_tokens = vec![
        special_tokens::SOT,           // 50257
        special_tokens::LANG_BASE,     // 50258 (English)
        special_tokens::TRANSCRIBE,    // 50358
        special_tokens::NO_TIMESTAMPS, // 50362
    ];

    let cache = RefCell::new(DecoderKVCache::new(n_layers, d_model, max_tokens));
    let mut model = model;

    // Get raw logits (no suppression)
    let mut raw_logits = vec![f32::NEG_INFINITY; n_vocab];
    for &token in &initial_tokens {
        raw_logits = model
            .decoder_mut()
            .forward_one(token, &enc_data, &mut cache.borrow_mut())?;
    }

    // Logit statistics
    let logit_n = raw_logits.len() as f64;
    let logit_mean: f64 = raw_logits.iter().map(|&x| x as f64).sum::<f64>() / logit_n;
    let logit_var: f64 = raw_logits.iter().map(|&x| ((x as f64) - logit_mean).powi(2)).sum::<f64>() / logit_n;
    let logit_std = logit_var.sqrt();
    let logit_min = raw_logits.iter().fold(f32::INFINITY, |a: f32, &b| a.min(b));
    let logit_max = raw_logits.iter().fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));

    println!("Raw Logit Statistics (before suppression):");
    println!("  Mean: {:.6}", logit_mean);
    println!("  Std Dev: {:.6}", logit_std);
    println!("  Min: {:.6}, Max: {:.6}", logit_min, logit_max);

    // Top 10 tokens
    let mut indexed: Vec<(usize, f32)> = raw_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 10 Tokens (raw logits):");
    for (rank, (token_id, logit)) in indexed.iter().take(10).enumerate() {
        let token_type = if *token_id < 256 {
            format!("byte '{}'", (*token_id as u8) as char)
        } else if *token_id < 50257 {
            "BPE".to_string()
        } else {
            "special".to_string()
        };
        println!("  {:2}. Token {:5} ({:8}): {:.4}", rank+1, token_id, token_type, logit);
    }

    // Check if token 265 is abnormally high
    let token_265_rank = indexed.iter().position(|(id, _)| *id == 265).unwrap_or(n_vocab);
    let token_265_logit = raw_logits[265];
    println!("\nToken 265 ('re') Analysis:");
    println!("  Rank: {} / {}", token_265_rank + 1, n_vocab);
    println!("  Logit: {:.4}", token_265_logit);
    println!("  Gap to top: {:.4}", indexed[0].1 - token_265_logit);

    // Expected first token for "The birds can use" is likely "The" or " The"
    // Whisper tokens: " The" = 440, "The" = 464
    println!("\nExpected tokens for 'The birds can use':");
    println!("  Token 440 (' The') logit: {:.4}", raw_logits[440]);
    println!("  Token 464 ('The') logit: {:.4}", raw_logits[464]);
    println!("  Token 383 (' the') logit: {:.4}", raw_logits[383]);

    // H3: Decoder validity
    let h3_pass = token_265_rank > 5 || raw_logits[440] > raw_logits[265] || raw_logits[383] > raw_logits[265];
    println!("\n✓ H3 (Valid Decoder): {} - {}",
        if h3_pass { "PASS" } else { "FAIL" },
        if h3_pass { "Decoder produces sensible logits" } else { "Decoder is stuck on token 265" }
    );

    // Dump logits
    std::fs::write("/tmp/whisper_apr_logits.bin",
        raw_logits.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>())?;
    println!("  Saved to /tmp/whisper_apr_logits.bin");

    // ═══════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║   FALSIFICATION SUMMARY                                    ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("  H0 (Audio):   {}", if h0_pass { "✓ PASS" } else { "✗ FAIL ← INVESTIGATE" });
    println!("  H1 (Mel):     {}", if h1_pass { "✓ PASS" } else { "✗ FAIL ← INVESTIGATE" });
    println!("  H2 (Encoder): {}", if h2_pass { "✓ PASS" } else { "✗ FAIL ← INVESTIGATE" });
    println!("  H3 (Decoder): {}", if h3_pass { "✓ PASS" } else { "✗ FAIL ← INVESTIGATE" });

    if !h3_pass {
        println!("\n⚠️  DECODER FAILURE DETECTED");
        println!("   The decoder consistently predicts token 265 ('re')");
        println!("   This suggests either:");
        println!("   1. Mel spectrogram layout/values are wrong");
        println!("   2. Encoder cross-attention is not using audio features");
        println!("   3. Model weights are corrupted");
        println!("\n   Next step: Compare mel spectrogram with whisper.cpp output");
    }

    Ok(())
}

fn dump_mel_to_file(data: &[f32], n_mels: usize, n_frames: usize, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    // Write header: n_mels, n_frames
    file.write_all(&(n_mels as u32).to_le_bytes())?;
    file.write_all(&(n_frames as u32).to_le_bytes())?;

    // Write data
    for &val in data {
        file.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

fn dump_encoder_to_file(data: &[f32], seq_len: usize, hidden_dim: usize, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    // Write header
    file.write_all(&(seq_len as u32).to_le_bytes())?;
    file.write_all(&(hidden_dim as u32).to_le_bytes())?;

    // Write data
    for &val in data {
        file.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

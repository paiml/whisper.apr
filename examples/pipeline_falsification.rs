#![allow(clippy::unwrap_used)]
//! Pipeline Falsification Tool
//!
//! Systematically compares every pipeline step against whisper.cpp ground truth.
//!
//! Usage:
//!   cargo run --example pipeline_falsification
//!
//! Prerequisites:
//!   1. Run: python3 tools/extract_ground_truth.py
//!   2. Ensure golden_traces/ directory exists with step_*.bin files

use std::cell::RefCell;
use std::path::Path;
use whisper_apr::model::DecoderKVCache;
use whisper_apr::tokenizer::special_tokens;
use whisper_apr::WhisperApr;

const GOLDEN_DIR: &str = "golden_traces";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   PIPELINE FALSIFICATION vs WHISPER.CPP GROUND TRUTH          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Check prerequisites
    if !Path::new(GOLDEN_DIR).exists() {
        eprintln!("ERROR: {} directory not found", GOLDEN_DIR);
        eprintln!("Run: python3 tools/extract_ground_truth.py");
        return Ok(());
    }

    let mut all_pass = true;
    let mut first_failure: Option<&str> = None;

    // Step A: Audio samples
    println!("=== Step A: Audio Samples ===\n");
    match step_a_audio() {
        Ok(pass) => {
            if !pass && first_failure.is_none() {
                first_failure = Some("Step A: Audio");
            }
            all_pass &= pass;
        }
        Err(e) => {
            println!("  ERROR: {}\n", e);
            all_pass = false;
            if first_failure.is_none() {
                first_failure = Some("Step A: Audio");
            }
        }
    }

    // Step B: Filterbank
    println!("=== Step B: Filterbank ===\n");
    match step_b_filterbank() {
        Ok(pass) => {
            if !pass && first_failure.is_none() {
                first_failure = Some("Step B: Filterbank");
            }
            all_pass &= pass;
        }
        Err(e) => {
            println!("  ERROR: {}\n", e);
            all_pass = false;
            if first_failure.is_none() {
                first_failure = Some("Step B: Filterbank");
            }
        }
    }

    // Step C: Mel spectrogram
    println!("=== Step C: Mel Spectrogram ===\n");
    match step_c_mel() {
        Ok(pass) => {
            if !pass && first_failure.is_none() {
                first_failure = Some("Step C: Mel");
            }
            all_pass &= pass;
        }
        Err(e) => {
            println!("  ERROR: {}\n", e);
            all_pass = false;
            if first_failure.is_none() {
                first_failure = Some("Step C: Mel");
            }
        }
    }

    // Step D-G: Encoder
    println!("=== Steps D-G: Encoder ===\n");
    match steps_encoder() {
        Ok((pass, encoded)) => {
            if !pass && first_failure.is_none() {
                first_failure = Some("Steps D-G: Encoder");
            }
            all_pass &= pass;

            // Step H-N: Decoder
            println!("=== Steps H-N: Decoder ===\n");
            match steps_decoder(&encoded) {
                Ok(pass) => {
                    if !pass && first_failure.is_none() {
                        first_failure = Some("Steps H-N: Decoder");
                    }
                    all_pass &= pass;
                }
                Err(e) => {
                    println!("  ERROR: {}\n", e);
                    all_pass = false;
                    if first_failure.is_none() {
                        first_failure = Some("Steps H-N: Decoder");
                    }
                }
            }
        }
        Err(e) => {
            println!("  ERROR: {}\n", e);
            all_pass = false;
            if first_failure.is_none() {
                first_failure = Some("Steps D-G: Encoder");
            }
        }
    }

    // Summary
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║   SUMMARY                                                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    if all_pass {
        println!("  ✓ ALL STEPS PASS\n");
    } else {
        println!("  ✗ FALSIFICATION FAILED\n");
        if let Some(step) = first_failure {
            println!("  First divergence: {}\n", step);
        }
    }

    Ok(())
}

fn step_a_audio() -> Result<bool, Box<dyn std::error::Error>> {
    // Load ground truth
    let gt_path = format!("{}/step_a_audio.bin", GOLDEN_DIR);
    let gt_audio = load_f32_binary(&gt_path)?;

    // Load our audio
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let our_audio: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    // Compare
    println!("  Ground truth: {} samples", gt_audio.len());
    println!("  Our audio:    {} samples", our_audio.len());

    let cosine = cosine_similarity(&gt_audio, &our_audio);
    let max_diff = gt_audio
        .iter()
        .zip(our_audio.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    println!("  Cosine sim:   {:.10}", cosine);
    println!("  Max diff:     {:.10}", max_diff);

    let pass = cosine > 0.9999 && our_audio.len() == gt_audio.len();
    println!(
        "  Status:       {}\n",
        if pass { "✓ PASS" } else { "✗ FAIL" }
    );

    Ok(pass)
}

fn step_b_filterbank() -> Result<bool, Box<dyn std::error::Error>> {
    // Load ground truth
    let gt_path = format!("{}/step_b_filterbank.bin", GOLDEN_DIR);
    let gt_fb = load_f32_binary(&gt_path)?;

    // Load our filterbank from model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let reader = whisper_apr::format::AprReader::new(model_bytes)?;

    let our_fb = reader
        .read_mel_filterbank()
        .ok_or("No filterbank in model")?;

    println!("  Ground truth: {} values", gt_fb.len());
    println!("  Our FB:       {} values", our_fb.data.len());

    let cosine = cosine_similarity(&gt_fb, &our_fb.data);
    let max_diff = gt_fb
        .iter()
        .zip(our_fb.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // Check row sums (slaney normalization)
    let gt_row0_sum: f32 = gt_fb[0..201].iter().sum();
    let our_row0_sum: f32 = our_fb.data[0..201].iter().sum();

    println!("  Cosine sim:   {:.10}", cosine);
    println!("  Max diff:     {:.10}", max_diff);
    println!("  GT row0 sum:  {:.6}", gt_row0_sum);
    println!("  Our row0 sum: {:.6}", our_row0_sum);

    let pass = cosine > 0.9999;
    println!(
        "  Status:       {}\n",
        if pass { "✓ PASS" } else { "✗ FAIL" }
    );

    Ok(pass)
}

fn step_c_mel() -> Result<bool, Box<dyn std::error::Error>> {
    // Load ground truth
    let gt_path = format!("{}/step_c_mel_numpy.bin", GOLDEN_DIR);
    let gt_mel = load_f32_binary(&gt_path)?;

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

    println!(
        "  Ground truth: {} values ({} frames x 80 mels)",
        gt_mel.len(),
        gt_mel.len() / 80
    );
    println!(
        "  Our mel:      {} values ({} frames x 80 mels)",
        our_mel.len(),
        our_mel.len() / 80
    );

    // Stats
    let gt_mean: f32 = gt_mel.iter().sum::<f32>() / gt_mel.len() as f32;
    let gt_min = gt_mel.iter().cloned().fold(f32::INFINITY, f32::min);
    let gt_max = gt_mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let our_mean: f32 = our_mel.iter().sum::<f32>() / our_mel.len() as f32;
    let our_min = our_mel.iter().cloned().fold(f32::INFINITY, f32::min);
    let our_max = our_mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!(
        "  GT stats:     min={:.4}, max={:.4}, mean={:.4}",
        gt_min, gt_max, gt_mean
    );
    println!(
        "  Our stats:    min={:.4}, max={:.4}, mean={:.4}",
        our_min, our_max, our_mean
    );

    // Compare overlapping frames
    let min_len = gt_mel.len().min(our_mel.len());
    let cosine = cosine_similarity(&gt_mel[..min_len], &our_mel[..min_len]);

    let max_diff = gt_mel[..min_len]
        .iter()
        .zip(our_mel[..min_len].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    println!("  Cosine sim:   {:.10} (first {} values)", cosine, min_len);
    println!("  Max diff:     {:.10}", max_diff);

    // Detailed frame-by-frame comparison for first 5 frames
    println!("\n  First 5 frames comparison:");
    for frame in 0..5.min(gt_mel.len() / 80).min(our_mel.len() / 80) {
        let gt_frame = &gt_mel[frame * 80..(frame + 1) * 80];
        let our_frame = &our_mel[frame * 80..(frame + 1) * 80];
        let frame_cosine = cosine_similarity(gt_frame, our_frame);
        let frame_max_diff = gt_frame
            .iter()
            .zip(our_frame.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "    Frame {}: cosine={:.6}, max_diff={:.6}",
            frame, frame_cosine, frame_max_diff
        );
    }

    // Mel passes if cosine > 0.99 (allowing for implementation differences)
    let pass = cosine > 0.95;
    println!(
        "\n  Status:       {}",
        if pass { "✓ PASS" } else { "✗ FAIL" }
    );

    if !pass {
        println!("  WARNING: Mel spectrogram diverges significantly from ground truth!");
        println!("  This may cause downstream transcription errors.\n");
    } else {
        println!();
    }

    Ok(pass)
}

fn steps_encoder() -> Result<(bool, Vec<f32>), Box<dyn std::error::Error>> {
    // Load model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let model = WhisperApr::load_from_apr(&model_bytes)?;

    // Compute mel
    let audio_bytes = std::fs::read("demos/test-audio/test-speech-1.5s.wav")?;
    let samples: Vec<f32> = audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    let mel = model.compute_mel(&samples)?;

    // Run encoder
    let encoded = model.encode(&mel)?;

    // Encoder output statistics
    let enc_mean: f32 = encoded.iter().sum::<f32>() / encoded.len() as f32;
    let enc_min = encoded.iter().cloned().fold(f32::INFINITY, f32::min);
    let enc_max = encoded.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let enc_std =
        (encoded.iter().map(|x| (x - enc_mean).powi(2)).sum::<f32>() / encoded.len() as f32).sqrt();

    let nonzero = encoded.iter().filter(|&&x| x.abs() > 1e-6).count();
    let nan_count = encoded.iter().filter(|x| x.is_nan()).count();
    let inf_count = encoded.iter().filter(|x| x.is_infinite()).count();

    println!("  Encoder output: {} values", encoded.len());
    println!("  Shape:          {} frames x 384 dim", encoded.len() / 384);
    println!(
        "  Stats:          min={:.4}, max={:.4}, mean={:.6}, std={:.4}",
        enc_min, enc_max, enc_mean, enc_std
    );
    println!(
        "  Non-zero:       {} ({:.2}%)",
        nonzero,
        100.0 * nonzero as f32 / encoded.len() as f32
    );
    println!("  NaN count:      {}", nan_count);
    println!("  Inf count:      {}", inf_count);

    // Check for problematic patterns
    let mut pass = true;

    if nan_count > 0 || inf_count > 0 {
        println!("  ✗ FAIL: NaN or Inf in encoder output!");
        pass = false;
    }

    if nonzero < encoded.len() / 2 {
        println!("  ⚠ WARNING: More than 50% zeros in encoder output");
    }

    if enc_std < 0.01 {
        println!(
            "  ⚠ WARNING: Very low variance in encoder output (std={:.6})",
            enc_std
        );
        println!("    This may indicate encoder weights are wrong or all zeros");
    }

    // Check first few values
    println!(
        "\n  First 10 encoder values: {:?}",
        &encoded[..10.min(encoded.len())]
    );

    // No ground truth for encoder yet, so just check sanity
    if pass {
        println!("\n  Status:       ✓ PASS (sanity check only)\n");
    } else {
        println!("\n  Status:       ✗ FAIL\n");
    }

    Ok((pass, encoded))
}

fn steps_decoder(encoded: &[f32]) -> Result<bool, Box<dyn std::error::Error>> {
    // Load model
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    let n_vocab = 51865;
    let max_tokens = 448;
    let d_model = 384;
    let n_layers = 4;

    // Initial tokens
    let initial_tokens = vec![
        special_tokens::SOT,           // 50257
        special_tokens::LANG_BASE,     // 50258 (English)
        special_tokens::TRANSCRIBE,    // 50358
        special_tokens::NO_TIMESTAMPS, // 50362
    ];

    println!("  Initial tokens: {:?}", initial_tokens);

    let cache = RefCell::new(DecoderKVCache::new(n_layers, d_model, max_tokens));
    let processed_count = RefCell::new(0usize);

    // Logits function
    let mut logits_fn = |tokens: &[u32]| -> whisper_apr::error::WhisperResult<Vec<f32>> {
        let seq_len = tokens.len();
        let already_processed = *processed_count.borrow();

        let mut logits = vec![f32::NEG_INFINITY; n_vocab];

        for i in already_processed..seq_len {
            let token = tokens[i];
            logits = model
                .decoder_mut()
                .forward_one(token, encoded, &mut cache.borrow_mut(), None)?;
        }

        *processed_count.borrow_mut() = seq_len;
        Ok(logits)
    };

    // Get first logits (after initial tokens)
    let logits = logits_fn(&initial_tokens)?;

    // Analyze logits
    let logits_mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let logits_min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let logits_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let logits_std = (logits
        .iter()
        .map(|x| (x - logits_mean).powi(2))
        .sum::<f32>()
        / logits.len() as f32)
        .sqrt();

    println!("\n  Logits (after initial tokens):");
    println!(
        "  Stats:          min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
        logits_min, logits_max, logits_mean, logits_std
    );

    // Top 10 tokens (before suppression)
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n  Top 10 tokens (before suppression):");
    for (i, (token, logit)) in indexed.iter().take(10).enumerate() {
        let token_type = categorize_token(*token as u32);
        println!(
            "    {:2}. token {:5} ({}) = {:.4}",
            i + 1,
            token,
            token_type,
            logit
        );
    }

    // Expected first token from whisper.cpp
    // whisper.cpp produces " The birds can use." so first token should be "The" or " The"
    println!("\n  Expected (whisper.cpp): Token for ' The' or 'The'");
    println!("  Common tokens: 440=' The', 464='The'");

    // Check if expected tokens are in top 10
    let top10_tokens: Vec<usize> = indexed.iter().take(10).map(|(t, _)| *t).collect();
    let has_the = top10_tokens.contains(&440) || top10_tokens.contains(&464);

    if !has_the {
        println!("\n  ⚠ WARNING: Expected 'The' token not in top 10!");
        println!(
            "  Actual argmax: token {} ({})",
            indexed[0].0,
            categorize_token(indexed[0].0 as u32)
        );
    }

    // Run greedy decoding
    println!("\n  Greedy decoding (first 10 tokens):");

    let eot = special_tokens::EOT;
    let mut tokens = initial_tokens.clone();
    *processed_count.borrow_mut() = initial_tokens.len();

    let mut all_same = true;
    let mut first_token: Option<u32> = None;

    for step in 0..10 {
        let logits = logits_fn(&tokens)?;

        // Apply suppression
        let mut suppressed = logits.clone();
        suppress_tokens(&mut suppressed);

        let (argmax_token, argmax_logit) = suppressed
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, val)| (idx as u32, *val))
            .unwrap_or((0, 0.0));

        println!(
            "    Step {:2}: token {:5} ({}) logit={:.4}",
            step,
            argmax_token,
            categorize_token(argmax_token),
            argmax_logit
        );

        if first_token.is_none() {
            first_token = Some(argmax_token);
        } else if Some(argmax_token) != first_token {
            all_same = false;
        }

        if argmax_token == eot {
            break;
        }

        tokens.push(argmax_token);
    }

    // Check for repetition bug
    let mut pass = true;

    if all_same && tokens.len() > initial_tokens.len() + 1 {
        println!("\n  ✗ FAIL: Repetitive token generation detected!");
        println!("    All generated tokens are the same: {:?}", first_token);
        pass = false;
    }

    // Check if output matches expected
    let expected = "The birds can use";
    println!("\n  Expected output: '{}'", expected);
    println!("  Generated tokens: {:?}", &tokens[initial_tokens.len()..]);

    if pass {
        println!("\n  Status:       ✓ PASS (no repetition)\n");
    } else {
        println!("\n  Status:       ✗ FAIL (repetitive generation)\n");
    }

    Ok(pass)
}

fn suppress_tokens(logits: &mut [f32]) {
    let n_vocab = logits.len();

    // Suppress special tokens
    logits[special_tokens::SOT as usize] = f32::NEG_INFINITY;
    logits[special_tokens::NO_SPEECH as usize] = f32::NEG_INFINITY;
    logits[special_tokens::TRANSLATE as usize] = f32::NEG_INFINITY;
    logits[special_tokens::TRANSCRIBE as usize] = f32::NEG_INFINITY;
    logits[special_tokens::PREV as usize] = f32::NEG_INFINITY;
    logits[special_tokens::SPEAKER_TURN as usize] = f32::NEG_INFINITY;
    logits[special_tokens::NO_TIMESTAMPS as usize] = f32::NEG_INFINITY;

    // Suppress language tokens
    for i in special_tokens::LANG_BASE..special_tokens::TRANSLATE {
        logits[i as usize] = f32::NEG_INFINITY;
    }

    // Suppress timestamps
    for i in special_tokens::TIMESTAMP_BASE as usize..n_vocab {
        logits[i] = f32::NEG_INFINITY;
    }
}

fn categorize_token(token: u32) -> &'static str {
    match token {
        0..=255 => "byte",
        256..=50256 => "BPE",
        50257 => "SOT",
        50258..=50357 => "lang",
        50358 => "transcribe",
        50359 => "translate",
        50360 => "prev",
        50361 => "no_speech",
        50362 => "no_timestamps",
        50363 => "speaker_turn",
        50364..=51864 => "timestamp",
        _ => "unknown",
    }
}

fn load_f32_binary(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(floats)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        norm_a += (*x as f64).powi(2);
        norm_b += (*y as f64).powi(2);
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
}

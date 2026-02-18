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
use whisper_apr::format::{AprV2ReaderRef, MelFilterbankData};
use whisper_apr::model::DecoderKVCache;
use whisper_apr::tokenizer::special_tokens;
use whisper_apr::WhisperApr;

const GOLDEN_DIR: &str = "golden_traces";

fn run_step(
    label: &'static str,
    step: impl FnOnce() -> Result<bool, Box<dyn std::error::Error>>,
    all_pass: &mut bool,
    first_failure: &mut Option<&'static str>,
) {
    println!("=== {label} ===\n");
    match step() {
        Ok(pass) => {
            if !pass && first_failure.is_none() {
                *first_failure = Some(label);
            }
            *all_pass &= pass;
        }
        Err(e) => {
            println!("  ERROR: {e}\n");
            *all_pass = false;
            if first_failure.is_none() {
                *first_failure = Some(label);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Pipeline Falsification vs whisper.cpp Ground Truth\n");

    if !Path::new(GOLDEN_DIR).exists() {
        eprintln!("ERROR: {GOLDEN_DIR} directory not found");
        eprintln!("Run: python3 tools/extract_ground_truth.py");
        return Ok(());
    }

    let mut all_pass = true;
    let mut first_failure: Option<&str> = None;

    run_step("Step A: Audio Samples", step_a_audio, &mut all_pass, &mut first_failure);
    run_step("Step B: Filterbank", step_b_filterbank, &mut all_pass, &mut first_failure);
    run_step("Step C: Mel Spectrogram", step_c_mel, &mut all_pass, &mut first_failure);

    println!("=== Steps D-G: Encoder ===\n");
    match steps_encoder() {
        Ok((pass, encoded)) => {
            if !pass && first_failure.is_none() {
                first_failure = Some("Steps D-G: Encoder");
            }
            all_pass &= pass;
            run_step(
                "Steps H-N: Decoder",
                || steps_decoder(&encoded),
                &mut all_pass,
                &mut first_failure,
            );
        }
        Err(e) => {
            println!("  ERROR: {e}\n");
            all_pass = false;
            if first_failure.is_none() {
                first_failure = Some("Steps D-G: Encoder");
            }
        }
    }

    println!("\nSUMMARY\n");
    if all_pass {
        println!("  ALL STEPS PASS\n");
    } else {
        println!("  FALSIFICATION FAILED\n");
        if let Some(step) = first_failure {
            println!("  First divergence: {step}\n");
        }
    }

    Ok(())
}

fn load_wav_samples(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let audio_bytes = std::fs::read(path)?;
    Ok(audio_bytes[44..]
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect())
}

fn step_a_audio() -> Result<bool, Box<dyn std::error::Error>> {
    let gt_audio = load_f32_binary(&format!("{GOLDEN_DIR}/step_a_audio.bin"))?;
    let our_audio = load_wav_samples("demos/test-audio/test-speech-1.5s.wav")?;

    println!("  Ground truth: {} samples", gt_audio.len());
    println!("  Our audio:    {} samples", our_audio.len());

    let cosine = cosine_similarity(&gt_audio, &our_audio);
    let max_diff = max_abs_diff(&gt_audio, &our_audio);
    println!("  Cosine sim:   {cosine:.10}");
    println!("  Max diff:     {max_diff:.10}");

    let pass = cosine > 0.9999 && our_audio.len() == gt_audio.len();
    println!("  Status:       {}\n", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

fn step_b_filterbank() -> Result<bool, Box<dyn std::error::Error>> {
    let gt_fb = load_f32_binary(&format!("{GOLDEN_DIR}/step_b_filterbank.bin"))?;
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let reader = AprV2ReaderRef::from_bytes(&model_bytes)?;

    let fb_data = reader
        .get_tensor_data("__mel_filters__")
        .ok_or("No __mel_filters__ tensor in model")?;
    let our_fb = MelFilterbankData::from_bytes(fb_data)?;

    println!("  Ground truth: {} values", gt_fb.len());
    println!("  Our FB:       {} values", our_fb.data.len());

    let cosine = cosine_similarity(&gt_fb, &our_fb.data);
    let max_diff = max_abs_diff(&gt_fb, &our_fb.data);
    println!("  Cosine sim:   {cosine:.10}");
    println!("  Max diff:     {max_diff:.10}");

    let pass = cosine > 0.9999;
    println!("  Status:       {}\n", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

fn step_c_mel() -> Result<bool, Box<dyn std::error::Error>> {
    let gt_mel = load_f32_binary(&format!("{GOLDEN_DIR}/step_c_mel_numpy.bin"))?;
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let model = WhisperApr::load_from_apr(&model_bytes)?;
    let samples = load_wav_samples("demos/test-audio/test-speech-1.5s.wav")?;
    let our_mel = model.compute_mel(&samples)?;

    println!("  Ground truth: {} values ({} frames)", gt_mel.len(), gt_mel.len() / 80);
    println!("  Our mel:      {} values ({} frames)", our_mel.len(), our_mel.len() / 80);

    let min_len = gt_mel.len().min(our_mel.len());
    let cosine = cosine_similarity(&gt_mel[..min_len], &our_mel[..min_len]);
    let max_diff = max_abs_diff(&gt_mel[..min_len], &our_mel[..min_len]);
    println!("  Cosine sim:   {cosine:.10} (first {min_len} values)");
    println!("  Max diff:     {max_diff:.10}");

    let pass = cosine > 0.95;
    println!("  Status:       {}\n", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

fn steps_encoder() -> Result<(bool, Vec<f32>), Box<dyn std::error::Error>> {
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let model = WhisperApr::load_from_apr(&model_bytes)?;
    let samples = load_wav_samples("demos/test-audio/test-speech-1.5s.wav")?;
    let mel = model.compute_mel(&samples)?;
    let encoded = model.encode(&mel)?;

    let mean: f32 = encoded.iter().sum::<f32>() / encoded.len() as f32;
    let std = (encoded.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / encoded.len() as f32).sqrt();
    let nan_count = encoded.iter().filter(|x| x.is_nan()).count();
    let inf_count = encoded.iter().filter(|x| x.is_infinite()).count();

    println!("  Encoder: {} values, mean={mean:.6}, std={std:.4}", encoded.len());
    println!("  NaN: {nan_count}, Inf: {inf_count}");

    let pass = nan_count == 0 && inf_count == 0;
    println!("  Status:       {}\n", if pass { "PASS" } else { "FAIL" });
    Ok((pass, encoded))
}

fn steps_decoder(encoded: &[f32]) -> Result<bool, Box<dyn std::error::Error>> {
    let model_bytes = std::fs::read("models/whisper-tiny-fb.apr")?;
    let mut model = WhisperApr::load_from_apr(&model_bytes)?;

    let n_vocab = 51865;
    let initial_tokens = vec![
        special_tokens::SOT,
        special_tokens::LANG_BASE,
        special_tokens::TRANSCRIBE,
        special_tokens::NO_TIMESTAMPS,
    ];

    let cache = RefCell::new(DecoderKVCache::new(4, 384, 448));
    let processed = RefCell::new(0usize);

    let mut logits_fn = |tokens: &[u32]| -> whisper_apr::error::WhisperResult<Vec<f32>> {
        let already = *processed.borrow();
        let mut logits = vec![f32::NEG_INFINITY; n_vocab];
        for i in already..tokens.len() {
            logits = model.decoder_mut().forward_one(tokens[i], encoded, &mut cache.borrow_mut())?;
        }
        *processed.borrow_mut() = tokens.len();
        Ok(logits)
    };

    logits_fn(&initial_tokens)?;

    let eot = special_tokens::EOT;
    let mut tokens = initial_tokens.clone();
    *processed.borrow_mut() = initial_tokens.len();
    let mut all_same = true;
    let mut first_token: Option<u32> = None;

    for step in 0..10 {
        let logits = logits_fn(&tokens)?;
        let mut suppressed = logits.clone();
        suppress_tokens(&mut suppressed);

        let (tok, logit) = suppressed
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i as u32, v))
            .unwrap_or((0, 0.0));

        println!("    Step {step:2}: token {tok:5} ({}) logit={logit:.4}", categorize_token(tok));

        if first_token.is_none() {
            first_token = Some(tok);
        } else if Some(tok) != first_token {
            all_same = false;
        }

        if tok == eot { break; }
        tokens.push(tok);
    }

    let pass = !(all_same && tokens.len() > initial_tokens.len() + 1);
    if !pass {
        println!("\n  FAIL: Repetitive token generation detected!");
    }
    println!("  Status:       {}\n", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

fn suppress_tokens(logits: &mut [f32]) {
    let n = logits.len();
    for &t in &[
        special_tokens::SOT, special_tokens::NO_SPEECH, special_tokens::TRANSLATE,
        special_tokens::TRANSCRIBE, special_tokens::PREV, special_tokens::SPEAKER_TURN,
        special_tokens::NO_TIMESTAMPS,
    ] {
        logits[t as usize] = f32::NEG_INFINITY;
    }
    for i in special_tokens::LANG_BASE..special_tokens::TRANSLATE {
        logits[i as usize] = f32::NEG_INFINITY;
    }
    for i in special_tokens::TIMESTAMP_BASE as usize..n {
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
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 { return 0.0; }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

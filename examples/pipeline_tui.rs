//! Encode/Decode Pipeline TUI - Visual performance and correctness simulation
//!
//! Simulates the entire whisper pipeline and identifies theoretical issues:
//! - Audio preprocessing (mel spectrogram)
//! - Encoder forward pass
//! - Decoder forward pass (with KV cache)
//! - Vocab projection
//! - Token generation
//!
//! Run with: cargo run --release --example pipeline_tui

use std::time::Instant;

const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

fn color_for_time(time_ms: f64, good: f64, warn: f64) -> &'static str {
    if time_ms <= good {
        GREEN
    } else if time_ms <= warn {
        YELLOW
    } else {
        RED
    }
}

fn bar(value: f64, max: f64, width: usize) -> String {
    let filled = ((value / max) * width as f64).min(width as f64) as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "{}{}{}",
        "\u{2588}".repeat(filled),
        DIM,
        "\u{2591}".repeat(empty)
    )
}

fn check_pass(passed: bool) -> &'static str {
    if passed {
        "\u{2713}"
    } else {
        "\u{2717}"
    }
}

fn main() {
    println!("\n{BOLD}{CYAN}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}{RESET}");
    println!(
        "{BOLD}{CYAN}          WHISPER PIPELINE TUI - ENCODE/DECODE SIMULATION          {RESET}"
    );
    println!("{BOLD}{CYAN}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}{RESET}\n");

    // Try to load model, fallback to synthetic tests if not available
    let model_result = std::fs::read("models/whisper-tiny-int8.apr")
        .map(|data| whisper_apr::WhisperApr::load_from_apr(&data));

    match model_result {
        Ok(Ok(mut model)) => run_with_model(&mut model),
        _ => run_synthetic_tests(),
    }

    println!("\n{BOLD}{CYAN}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}{RESET}");
    println!("{DIM}Legend: {GREEN}FAST{RESET} {YELLOW}OK{RESET} {RED}SLOW{RESET}");
    println!("{BOLD}{CYAN}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}{RESET}\n");
}

fn run_with_model(model: &mut whisper_apr::WhisperApr) {
    let config = model.config();
    let d_model = config.n_text_state as usize;
    let n_vocab = config.n_vocab as usize;

    println!("{BOLD}{CYAN}\u{25b6} MODEL CONFIGURATION{RESET}");
    println!("{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}");
    println!("  d_model:       {d_model}");
    println!("  n_vocab:       {n_vocab}");
    println!("  n_text_layer:  {}", config.n_text_layer);
    println!("  n_audio_layer: {}", config.n_audio_layer);
    println!(
        "  Vocab matrix:  {:.1}MB",
        (d_model * n_vocab * 4) as f64 / 1e6
    );

    // =========================================================================
    // AUDIO PREPROCESSING
    // =========================================================================
    println!("\n{BOLD}{CYAN}\u{25b6} AUDIO PREPROCESSING{RESET}");
    println!("{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}");

    let audio_durations = [1.0, 5.0, 10.0, 30.0];
    let mut mel_times = Vec::new();

    for duration in audio_durations {
        let samples = (16000.0 * duration) as usize;
        let audio: Vec<f32> = (0..samples)
            .map(|i| ((i as f32) * 0.01).sin() * 0.3)
            .collect();

        // Warmup
        let _ = model.compute_mel(&audio);

        let start = Instant::now();
        let _mel = model.compute_mel(&audio).expect("mel");
        let time_ms = start.elapsed().as_secs_f64() * 1000.0;
        mel_times.push(time_ms);

        let expected = duration * 20.0; // Should be much faster than real-time
        let color = color_for_time(time_ms, expected * 0.5, expected);
        let rtf = time_ms / (duration * 1000.0);
        println!(
            "  {:.0}s audio -> mel   {:>7.1}ms   RTF={:.3}   {} {}{RESET}",
            duration,
            time_ms,
            rtf,
            bar(time_ms, expected, 15),
            color
        );
    }

    // =========================================================================
    // ENCODER FORWARD
    // =========================================================================
    println!("\n{BOLD}{CYAN}\u{25b6} ENCODER FORWARD{RESET}");
    println!("{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}");

    // Use 1s audio for encoder test
    let audio: Vec<f32> = (0..16000)
        .map(|i| ((i as f32) * 0.01).sin() * 0.3)
        .collect();
    let mel = model.compute_mel(&audio).expect("mel");

    // Warmup
    for _ in 0..3 {
        let _ = model.encoder_mut().forward_mel(&mel);
    }

    let start = Instant::now();
    let iterations = 5;
    for _ in 0..iterations {
        let _ = model.encoder_mut().forward_mel(&mel).expect("encode");
    }
    let enc_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    let enc_output = model.encoder_mut().forward_mel(&mel).expect("encode");
    let enc_len = enc_output.len() / d_model;

    let color = color_for_time(enc_time, 50.0, 100.0);
    println!(
        "  Encoder forward       {:>7.1}ms   output: {}x{} {} {}{RESET}",
        enc_time,
        enc_len,
        d_model,
        bar(enc_time, 100.0, 15),
        color
    );

    // =========================================================================
    // DECODER FORWARD (Batch)
    // =========================================================================
    println!("\n{BOLD}{CYAN}\u{25b6} DECODER FORWARD (Batch){RESET}");
    println!("{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}");

    let seq_lengths = [1, 5, 10, 20];

    for seq_len in seq_lengths {
        let tokens: Vec<u32> = (0..seq_len).map(|i| (50258 + i) as u32).collect();

        // Warmup
        for _ in 0..3 {
            let _ = model.decoder_mut().forward(&tokens, &enc_output);
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = model
                .decoder_mut()
                .forward(&tokens, &enc_output)
                .expect("dec");
        }
        let dec_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        let expected = 10.0 + seq_len as f64 * 5.0;
        let color = color_for_time(dec_time, expected, expected * 2.0);
        println!(
            "  Batch forward (seq={:>2})  {:>6.1}ms   {} {}{RESET}",
            seq_len,
            dec_time,
            bar(dec_time, expected * 2.0, 20),
            color
        );
    }

    // =========================================================================
    // DECODER FORWARD_ONE (KV Cache)
    // =========================================================================
    println!("\n{BOLD}{CYAN}\u{25b6} DECODER FORWARD_ONE (KV Cache){RESET}");
    println!("{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}");

    let mut cache = model.decoder_mut().create_kv_cache();

    // First token (cold cache)
    cache.clear();
    let start = Instant::now();
    let _ = model
        .decoder_mut()
        .forward_one(50258, &enc_output, &mut cache)
        .expect("dec");
    let first_token_time = start.elapsed().as_secs_f64() * 1000.0;

    let color = color_for_time(first_token_time, 30.0, 50.0);
    println!(
        "  First token (cold)     {:>6.1}ms   {} {}{RESET}",
        first_token_time,
        bar(first_token_time, 50.0, 20),
        color
    );

    // Subsequent tokens (warm cache)
    let mut subsequent_times = Vec::new();
    for i in 0..10 {
        let start = Instant::now();
        let _ = model
            .decoder_mut()
            .forward_one(50259 + i, &enc_output, &mut cache)
            .expect("dec");
        subsequent_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let avg_subsequent = subsequent_times.iter().sum::<f64>() / subsequent_times.len() as f64;

    let color = color_for_time(avg_subsequent, 10.0, 20.0);
    println!(
        "  Subsequent tokens      {:>6.1}ms   {} {}{RESET}",
        avg_subsequent,
        bar(avg_subsequent, 20.0, 20),
        color
    );

    let cache_benefit = first_token_time / avg_subsequent;
    println!(
        "  Cache speedup          {:>6.1}x    {}{RESET}",
        cache_benefit,
        if cache_benefit > 2.0 { GREEN } else { YELLOW }
    );

    // =========================================================================
    // VOCAB PROJECTION (Isolated)
    // =========================================================================
    println!("\n{BOLD}{CYAN}\u{25b6} VOCAB PROJECTION{RESET}");
    println!("{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}");

    // Test simd::matmul wrapper (with copy overhead)
    let hidden = vec![0.1_f32; d_model];
    let embedding_t = model.decoder_mut().token_embedding().to_vec();
    let embedding_t = whisper_apr::simd::transpose(&embedding_t, n_vocab, d_model);

    // Warmup
    for _ in 0..3 {
        let _ = whisper_apr::simd::matmul(&hidden, &embedding_t, 1, d_model, n_vocab);
    }

    let start = Instant::now();
    for _ in 0..10 {
        let _ = whisper_apr::simd::matmul(&hidden, &embedding_t, 1, d_model, n_vocab);
    }
    let wrapper_time = start.elapsed().as_secs_f64() * 1000.0 / 10.0;

    // Test trueno Matrix directly
    use trueno::Matrix;
    let ma = Matrix::from_slice(1, d_model, &hidden).expect("example assertion");
    let mb = Matrix::from_vec(d_model, n_vocab, embedding_t.clone()).expect("example assertion");

    // Warmup
    for _ in 0..3 {
        let _ = ma.matmul(&mb);
    }

    let start = Instant::now();
    for _ in 0..10 {
        let _ = ma.matmul(&mb).expect("example assertion");
    }
    let direct_time = start.elapsed().as_secs_f64() * 1000.0 / 10.0;

    let color = color_for_time(wrapper_time, 10.0, 30.0);
    println!(
        "  simd::matmul wrapper   {:>6.1}ms   {} {}{RESET}",
        wrapper_time,
        bar(wrapper_time, 50.0, 20),
        color
    );

    let color = color_for_time(direct_time, 5.0, 10.0);
    println!(
        "  trueno Matrix direct   {:>6.1}ms   {} {}{RESET}",
        direct_time,
        bar(direct_time, 50.0, 20),
        color
    );

    let speedup = wrapper_time / direct_time;
    println!(
        "  Direct speedup         {:>6.1}x    {}{} (decoder uses direct){RESET}",
        speedup,
        if speedup > 5.0 { GREEN } else { YELLOW },
        check_pass(speedup > 5.0)
    );

    // =========================================================================
    // CORRECTNESS CHECKS
    // =========================================================================
    println!("\n{BOLD}{CYAN}\u{25b6} CORRECTNESS CHECKS{RESET}");
    println!("{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}");

    // Check encoder output is finite
    let enc_finite = enc_output.iter().all(|x| x.is_finite());
    println!(
        "  Encoder output finite  {} {}{RESET}",
        check_pass(enc_finite),
        if enc_finite { GREEN } else { RED }
    );

    // Check decoder output is finite
    let tokens = vec![50258_u32];
    let dec_output = model
        .decoder_mut()
        .forward(&tokens, &enc_output)
        .expect("dec");
    let dec_finite = dec_output.iter().all(|x| x.is_finite());
    println!(
        "  Decoder output finite  {} {}{RESET}",
        check_pass(dec_finite),
        if dec_finite { GREEN } else { RED }
    );

    // Check softmax sums to 1
    let softmax_out = whisper_apr::simd::softmax(&dec_output);
    let softmax_sum: f32 = softmax_out.iter().sum();
    let softmax_ok = (softmax_sum - 1.0).abs() < 0.001;
    println!(
        "  Softmax sums to 1      {} (sum={:.6}) {}{RESET}",
        check_pass(softmax_ok),
        softmax_sum,
        if softmax_ok { GREEN } else { RED }
    );

    // Check no NaN in logits
    let no_nan = !dec_output.iter().any(|x| x.is_nan());
    println!(
        "  No NaN in logits       {} {}{RESET}",
        check_pass(no_nan),
        if no_nan { GREEN } else { RED }
    );

    // Check argmax is valid token
    let argmax = whisper_apr::simd::argmax(&dec_output);
    let argmax_valid = argmax < n_vocab;
    println!(
        "  Argmax valid token     {} (token={}) {}{RESET}",
        check_pass(argmax_valid),
        argmax,
        if argmax_valid { GREEN } else { RED }
    );

    // =========================================================================
    // END-TO-END THROUGHPUT
    // =========================================================================
    println!("\n{BOLD}{CYAN}\u{25b6} END-TO-END THROUGHPUT{RESET}");
    println!("{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}");

    // Simulate generating 20 tokens
    let num_tokens = 20;
    cache.clear();

    let start = Instant::now();
    for i in 0..num_tokens {
        let _ = model
            .decoder_mut()
            .forward_one(50258 + i, &enc_output, &mut cache)
            .expect("dec");
    }
    let total_gen_time = start.elapsed().as_secs_f64() * 1000.0;
    let tokens_per_sec = num_tokens as f64 / (total_gen_time / 1000.0);

    let color = color_for_time(total_gen_time, 200.0, 500.0);
    println!(
        "  Generate {num_tokens} tokens      {:>6.1}ms   {:.1} tok/s {} {}{RESET}",
        total_gen_time,
        tokens_per_sec,
        bar(tokens_per_sec, 200.0, 15),
        color
    );

    // Calculate theoretical RTF for 1s audio
    let total_pipeline = mel_times[0] + enc_time + total_gen_time;
    let rtf = total_pipeline / 1000.0;
    let color = if rtf < 2.0 {
        GREEN
    } else if rtf < 4.0 {
        YELLOW
    } else {
        RED
    };
    println!(
        "  1s audio pipeline      {:>6.1}ms   RTF={:.2} {} {}{RESET}",
        total_pipeline,
        rtf,
        bar(rtf, 5.0, 15),
        color
    );
}

fn run_synthetic_tests() {
    println!("{YELLOW}Model not found. Running synthetic tests...{RESET}\n");

    use whisper_apr::model::ModelConfig;

    let config = ModelConfig::tiny();
    let d_model = config.n_text_state as usize;
    let n_vocab = config.n_vocab as usize;

    println!("{BOLD}{CYAN}\u{25b6} SYNTHETIC PIPELINE TEST{RESET}");
    println!("{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}");

    // Test trueno vector-matrix multiply (the optimized path)
    use trueno::Matrix;

    let hidden: Vec<f32> = (0..d_model).map(|i| (i as f32 * 0.01).sin()).collect();
    let embedding: Vec<f32> = (0..d_model * n_vocab)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();

    let ma = Matrix::from_slice(1, d_model, &hidden).expect("example assertion");
    let mb = Matrix::from_vec(d_model, n_vocab, embedding).expect("example assertion");

    // Warmup
    for _ in 0..3 {
        let _ = ma.matmul(&mb);
    }

    let start = Instant::now();
    for _ in 0..10 {
        let _ = ma.matmul(&mb).expect("example assertion");
    }
    let direct_time = start.elapsed().as_secs_f64() * 1000.0 / 10.0;

    let color = color_for_time(direct_time, 5.0, 10.0);
    println!(
        "  Vocab projection (1x{d_model}x{n_vocab}): {:>6.1}ms {} {}{RESET}",
        direct_time,
        bar(direct_time, 20.0, 20),
        color
    );

    // Test correctness
    let result = ma.matmul(&mb).expect("example assertion");
    let finite = result.as_slice().iter().all(|x| x.is_finite());
    println!(
        "  Output finite          {} {}{RESET}",
        check_pass(finite),
        if finite { GREEN } else { RED }
    );

    // Test softmax
    let softmax_out = whisper_apr::simd::softmax(result.as_slice());
    let sum: f32 = softmax_out.iter().sum();
    let ok = (sum - 1.0).abs() < 0.001;
    println!(
        "  Softmax sums to 1      {} (sum={:.6}) {}{RESET}",
        check_pass(ok),
        sum,
        if ok { GREEN } else { RED }
    );
}

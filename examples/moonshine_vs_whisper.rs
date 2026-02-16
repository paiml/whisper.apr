#![allow(missing_docs)]
//! Moonshine vs Whisper architecture comparison benchmark
//!
//! Compares construction, encoder, decoder, and memory between
//! Whisper tiny (MHA + GELU + LayerNorm + mel) and
//! Moonshine tiny (GQA + SwiGLU + RmsNorm + ConvStem + RoPE).
//!
//! Run: cargo run --example moonshine_vs_whisper --release

use std::time::Instant;
use whisper_apr::model::{Decoder, Encoder, ModelConfig};

/// Number of warmup iterations
const WARMUP: usize = 3;
/// Number of timed iterations
const ITERS: usize = 10;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         Moonshine tiny vs Whisper tiny — Architecture Bench     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Config comparison ─────────────────────────────────────────────
    let w_cfg = ModelConfig::tiny();
    let m_cfg = ModelConfig::moonshine_tiny();

    println!("┌─────────────────────┬──────────────┬──────────────┐");
    println!("│ Parameter           │ Whisper tiny │ Moonshine    │");
    println!("├─────────────────────┼──────────────┼──────────────┤");
    println!(
        "│ d_model             │ {:>12} │ {:>12} │",
        w_cfg.n_audio_state, m_cfg.n_audio_state
    );
    println!(
        "│ encoder layers      │ {:>12} │ {:>12} │",
        w_cfg.n_audio_layer, m_cfg.n_audio_layer
    );
    println!(
        "│ decoder layers      │ {:>12} │ {:>12} │",
        w_cfg.n_text_layer, m_cfg.n_text_layer
    );
    println!(
        "│ attention heads     │ {:>12} │ {:>12} │",
        w_cfg.n_audio_head, m_cfg.n_audio_head
    );
    println!(
        "│ attention type      │ {:>12} │ {:>12} │",
        "MHA", "GQA (2 KV)"
    );
    println!(
        "│ FFN activation      │ {:>12} │ {:>12} │",
        "GELU", "SwiGLU"
    );
    println!(
        "│ positional enc      │ {:>12} │ {:>12} │",
        "Sinusoidal", "RoPE"
    );
    println!(
        "│ audio frontend      │ {:>12} │ {:>12} │",
        "Mel (80)", "Learned Conv"
    );
    println!(
        "│ vocab size          │ {:>12} │ {:>12} │",
        w_cfg.n_vocab, m_cfg.n_vocab
    );
    println!(
        "│ audio context       │ {:>12} │ {:>12} │",
        w_cfg.n_audio_ctx, "variable"
    );
    println!("└─────────────────────┴──────────────┴──────────────┘");
    println!();

    // ── 1. Construction time ──────────────────────────────────────────
    println!("── 1. Model Construction ──────────────────────────────────────");

    let w_construct = bench(WARMUP, ITERS, || {
        let _ = std::hint::black_box(Encoder::new(&w_cfg));
        let _ = std::hint::black_box(Decoder::new(&w_cfg));
    });
    let m_construct = bench(WARMUP, ITERS, || {
        let _ = std::hint::black_box(Encoder::new(&m_cfg));
        let _ = std::hint::black_box(Decoder::new(&m_cfg));
    });
    print_row("Construction (enc+dec)", w_construct, m_construct);
    println!();

    // ── 2. Encoder forward pass ───────────────────────────────────────
    println!("── 2. Encoder Forward Pass ────────────────────────────────────");

    let w_enc = Encoder::new(&w_cfg);
    let m_enc = Encoder::new(&m_cfg);

    for &seq_len in &[7_usize, 50, 200] {
        let w_input = vec![0.1_f32; seq_len * w_cfg.n_audio_state as usize];
        let m_input = vec![0.1_f32; seq_len * m_cfg.n_audio_state as usize];

        let w_time = bench(WARMUP, ITERS, || {
            let _ = std::hint::black_box(w_enc.forward(std::hint::black_box(&w_input)));
        });
        let m_time = bench(WARMUP, ITERS, || {
            let _ = std::hint::black_box(m_enc.forward(std::hint::black_box(&m_input)));
        });

        let label = format!("Encoder fwd (seq={})", seq_len);
        print_row(&label, w_time, m_time);
    }
    println!();

    // ── 3. Decoder full forward ───────────────────────────────────────
    println!("── 3. Decoder Full Forward ────────────────────────────────────");

    let w_dec = Decoder::new(&w_cfg);
    let m_dec = Decoder::new(&m_cfg);
    let enc_seq = 50;

    let w_enc_out = vec![0.1_f32; enc_seq * w_cfg.n_audio_state as usize];
    let m_enc_out = vec![0.1_f32; enc_seq * m_cfg.n_audio_state as usize];

    for &n_tokens in &[2_usize, 5, 16] {
        // Use small token IDs valid for both vocabs
        let tokens: Vec<u32> = (1..=n_tokens as u32).collect();

        let w_time = bench(WARMUP, ITERS, || {
            let _ = std::hint::black_box(
                w_dec.forward(std::hint::black_box(&tokens), std::hint::black_box(&w_enc_out)),
            );
        });
        let m_time = bench(WARMUP, ITERS, || {
            let _ = std::hint::black_box(
                m_dec.forward(std::hint::black_box(&tokens), std::hint::black_box(&m_enc_out)),
            );
        });

        let label = format!("Decoder fwd (tok={})", n_tokens);
        print_row(&label, w_time, m_time);
    }
    println!();

    // ── 4. Decoder cached forward_one ─────────────────────────────────
    println!("── 4. Decoder Cached forward_one ──────────────────────────────");
    println!("  Whisper: MHA incremental KV cache (always O(n))");
    println!("  Moonshine NEW: GQA incremental KV cache (WAPR-MOONSHINE-010)");
    println!("  Moonshine OLD: O(n^2) full-recompute (pre-MOONSHINE-010)");
    println!();
    println!(
        "  {:<22} {:>10} {:>10} {:>10}",
        "", "Whisper", "Moon NEW", "Moon OLD"
    );
    println!("  {}", "-".repeat(56));

    for &n_gen in &[10_usize, 20, 40] {
        // Whisper: standard KV-cached incremental decode
        let w_cached = bench_decode(&w_dec, &w_enc_out, n_gen, WARMUP, ITERS, false);

        // Moonshine NEW: incremental GQA KV cache (uses create_kv_cache → new_gqa)
        let m_cached_new = bench_decode(&m_dec, &m_enc_out, n_gen, WARMUP, ITERS, false);

        // Moonshine OLD: O(n²) full-recompute path via decoder.forward() each step
        let m_cached_old = bench_decode_old_moonshine(&m_dec, &m_enc_out, n_gen, WARMUP, ITERS);

        let label = format!("{}-token total", n_gen);
        print_row_3(&label, w_cached, m_cached_new, m_cached_old);

        let w_per = w_cached / n_gen as f64;
        let m_new_per = m_cached_new / n_gen as f64;
        let m_old_per = m_cached_old / n_gen as f64;
        print_row_3("  per-token avg", w_per, m_new_per, m_old_per);

        if m_old_per > 0.001 {
            println!(
                "  >>> Moonshine speedup: {:.1}x (old {:.2}ms → new {:.2}ms per token)",
                m_old_per / m_new_per,
                m_old_per,
                m_new_per,
            );
        }
        println!();
    }

    // ── 5. Memory estimate ────────────────────────────────────────────
    println!("── 5. Approximate Parameter Count (zero-weight) ─────────────");

    let w_enc_params = estimate_encoder_params(&w_cfg);
    let m_enc_params = estimate_encoder_params(&m_cfg);
    let w_dec_params = estimate_decoder_params(&w_cfg);
    let m_dec_params = estimate_decoder_params(&m_cfg);

    println!(
        "  {:25} {:>12.1}K {:>12.1}K",
        "Encoder params",
        w_enc_params as f64 / 1000.0,
        m_enc_params as f64 / 1000.0,
    );
    println!(
        "  {:25} {:>12.1}K {:>12.1}K",
        "Decoder params",
        w_dec_params as f64 / 1000.0,
        m_dec_params as f64 / 1000.0,
    );
    println!(
        "  {:25} {:>12.1}M {:>12.1}M",
        "Total (f32 bytes)",
        (w_enc_params + w_dec_params) as f64 * 4.0 / 1_000_000.0,
        (m_enc_params + m_dec_params) as f64 * 4.0 / 1_000_000.0,
    );
    let w_total = w_enc_params + w_dec_params;
    let m_total = m_enc_params + m_dec_params;
    let ratio = m_total as f64 / w_total as f64;
    println!("  Moonshine/Whisper param ratio: {ratio:.2}x");
    println!();

    // ── Summary ───────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Notes:                                                        ║");
    println!("║  • Moonshine uses d_model=288 (vs 384) → smaller per-layer     ║");
    println!("║  • Moonshine has 6 layers (vs 4) → deeper but narrower         ║");
    println!("║  • GQA (2 KV heads) reduces KV memory vs MHA (6 heads)         ║");
    println!("║  • SwiGLU uses gated activation (2 projections vs 1 for GELU)  ║");
    println!("║  • WAPR-MOONSHINE-010: Moonshine now uses incremental KV cache ║");
    println!("║  • Variable-length input: no 30s padding waste                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

/// Benchmark decoder forward_one with proper KV cache (new incremental path)
fn bench_decode(
    dec: &Decoder,
    enc_out: &[f32],
    n_gen: usize,
    warmup: usize,
    iters: usize,
    _hidden_only: bool,
) -> f64 {
    for _ in 0..warmup {
        let mut cache = dec.create_kv_cache();
        for t in 0..n_gen {
            let _ = dec.forward_one((t + 1) as u32, enc_out, &mut cache);
        }
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut cache = dec.create_kv_cache();
        let start = Instant::now();
        for t in 0..n_gen {
            let _ = dec.forward_one((t + 1) as u32, enc_out, &mut cache);
        }
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    median(&mut times)
}

/// Benchmark old O(n²) Moonshine path: accumulate tokens, run full forward each step
fn bench_decode_old_moonshine(
    dec: &Decoder,
    enc_out: &[f32],
    n_gen: usize,
    warmup: usize,
    iters: usize,
) -> f64 {
    // Simulate the old path: accumulate token embeddings, run full forward each step
    for _ in 0..warmup {
        let mut all_tokens: Vec<u32> = Vec::new();
        for t in 0..n_gen {
            all_tokens.push((t + 1) as u32);
            let _ = dec.forward(&all_tokens, enc_out);
        }
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut all_tokens: Vec<u32> = Vec::new();
        let start = Instant::now();
        for t in 0..n_gen {
            all_tokens.push((t + 1) as u32);
            let _ = dec.forward(&all_tokens, enc_out);
        }
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    median(&mut times)
}

/// Print a 3-column comparison row
fn print_row_3(label: &str, whisper_ms: f64, moon_new_ms: f64, moon_old_ms: f64) {
    println!(
        "  {:<22} {:>8.2}ms {:>8.2}ms {:>8.2}ms",
        label, whisper_ms, moon_new_ms, moon_old_ms,
    );
}

/// Benchmark a closure, returning median time in ms
fn bench<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    median(&mut times)
}

/// Compute median of a slice
fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

/// Print a comparison row: label | whisper time | moonshine time | speedup
fn print_row(label: &str, whisper_ms: f64, moonshine_ms: f64) {
    let speedup = whisper_ms / moonshine_ms;
    let indicator = if speedup > 1.05 {
        "faster"
    } else if speedup < 0.95 {
        "slower"
    } else {
        "~same"
    };
    println!(
        "  {:<30} {:>8.2}ms  {:>8.2}ms  {:.2}x {}",
        label, whisper_ms, moonshine_ms, speedup, indicator
    );
}

/// Rough parameter count for encoder
fn estimate_encoder_params(cfg: &ModelConfig) -> usize {
    let d = cfg.n_audio_state as usize;
    let n_layers = cfg.n_audio_layer as usize;
    let n_heads = cfg.n_audio_head as usize;
    let d_ff = d * 4;

    match cfg.attention_type {
        whisper_apr::model::AttentionType::Mha => {
            // Conv frontend: conv1(n_mels*d + d) + conv2(d*d + d)
            let conv = cfg.n_mels as usize * d + d + d * d + d;
            // Per block: 4 attn projections (d*d each) + 2 FFN (d*d_ff + d_ff*d) + norms
            let per_block = 4 * d * d + 2 * d * d_ff + 4 * d;
            // Final LN
            let ln = 2 * d;
            // Positional embedding
            let pe = cfg.n_audio_ctx as usize * d;
            conv + n_layers * per_block + ln + pe
        }
        whisper_apr::model::AttentionType::Gqa { kv_heads } => {
            let kv = kv_heads as usize;
            let head_dim = d / n_heads;
            // Per block: Q(d*d) + K(d*kv*head_dim) + V(d*kv*head_dim) + O(d*d)
            // + SwiGLU: gate(d*d_ff) + up(d*d_ff) + down(d_ff*d) + 2 norms(d each)
            let intermediate = (d * 8) / 3;
            let attn = d * d + 2 * d * kv * head_dim + d * d;
            let ffn = 2 * d * intermediate + intermediate * d;
            let norms = 2 * d;
            let per_block = attn + ffn + norms;
            // Final RmsNorm
            let ln = d;
            n_layers * per_block + ln
        }
    }
}

/// Rough parameter count for decoder
fn estimate_decoder_params(cfg: &ModelConfig) -> usize {
    let d = cfg.n_text_state as usize;
    let n_layers = cfg.n_text_layer as usize;
    let n_heads = cfg.n_text_head as usize;
    let d_ff = d * 4;
    let n_vocab = cfg.n_vocab as usize;

    // Token embedding
    let emb = n_vocab * d;
    // Positional embedding (Whisper only)
    let pe = cfg.n_text_ctx as usize * d;

    match cfg.attention_type {
        whisper_apr::model::AttentionType::Mha => {
            // Per block: self-attn(4*d*d) + cross-attn(4*d*d) + FFN(2*d*d_ff) + 3 norms(6*d)
            let per_block = 8 * d * d + 2 * d * d_ff + 6 * d;
            let ln = 2 * d;
            emb + pe + n_layers * per_block + ln
        }
        whisper_apr::model::AttentionType::Gqa { kv_heads } => {
            let kv = kv_heads as usize;
            let head_dim = d / n_heads;
            let intermediate = (d * 8) / 3;
            // Self-attn + cross-attn (both GQA)
            let attn = 2 * (d * d + 2 * d * kv * head_dim + d * d);
            let ffn = 2 * d * intermediate + intermediate * d;
            let norms = 3 * d; // ln1 + ln_cross + ln2
            let per_block = attn + ffn + norms;
            let ln = d; // final RmsNorm
            emb + n_layers * per_block + ln
        }
    }
}

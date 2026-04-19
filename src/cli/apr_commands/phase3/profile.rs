//! Tier C — Profiling (renacer integration + BrickProfiler).
//!
//! Provides instrumented transcription, thread-scaling sweeps, Amdahl's law
//! metrics, roofline classification, and renacer/Chrome Trace output.

use std::fs;
use std::time::Instant;

use super::super::super::apr_args::AprProfileArgs;
use super::super::super::commands::{CliError, CliResult, CommandResult};
use super::super::rtf_tier_label;
use super::emit_output;

/// Human-readable label for bottleneck diagnosis code
fn bottleneck_label(code: u8) -> &'static str {
    match code {
        1 => "memory-bound",
        2 => "compute-bound",
        3 => "throttled",
        4 => "balanced",
        _ => "insufficient data",
    }
}

/// WAPR-PROFILE-001 Gap 3: Thread scaling sweep
///
/// Runs transcription at each thread count and reports speedup, efficiency,
/// and Amdahl serial fraction. Computes `s` from `T(N) = T(1) * (s + (1-s)/N)`.
#[allow(clippy::too_many_arguments)]
struct SweepResult {
    threads: u32,
    enc_ms: f64,
    dec_ms: f64,
    total_ms: f64,
}

/// Amdahl's law metrics for a thread sweep data point.
fn amdahl_metrics(threads: u32, total_ms: f64, base_total: f64) -> (f64, f64, f64) {
    let speedup = base_total / total_ms;
    let eff = speedup / threads as f64 * 100.0;
    let n = threads as f64;
    let serial = if n > 1.0 {
        (1.0 / speedup - 1.0 / n) / (1.0 - 1.0 / n)
    } else {
        0.0
    };
    (speedup, eff, serial)
}

/// Format sweep results as JSON or text table. Returns `Some(string)` for JSON,
/// `None` for text (printed directly via `emit_output`).
fn format_sweep_output(
    results: &[SweepResult],
    baseline: &SweepResult,
    base_total: f64,
    audio_duration_s: f64,
    args: &AprProfileArgs,
    global: &super::super::super::args::Args,
    hw: &trueno::HardwareCapability,
) -> Option<String> {
    if args.format == "json" {
        let entries: Vec<String> = results
            .iter()
            .map(|r| {
                let (speedup, eff, serial) = amdahl_metrics(r.threads, r.total_ms, base_total);
                format!(
                    concat!(
                        "{{\"threads\":{},\"enc_ms\":{:.1},\"dec_ms\":{:.1},",
                        "\"total_ms\":{:.1},\"speedup\":{:.2},",
                        "\"efficiency_pct\":{:.1},\"amdahl_serial_pct\":{:.1}}}"
                    ),
                    r.threads,
                    r.enc_ms,
                    r.dec_ms,
                    r.total_ms,
                    speedup,
                    eff,
                    serial * 100.0,
                )
            })
            .collect();
        Some(format!(
            "{{\"sweep\":{{\"audio_duration_s\":{:.3},\"hw_cores\":{},\"hw_simd\":\"{}\",\"results\":[{}]}}}}",
            audio_duration_s,
            hw.cpu.cores,
            format!("{:?}", hw.cpu.simd),
            entries.join(","),
        ))
    } else {
        emit_output(
            global,
            || {},
            || {
                println!("Thread Scaling Sweep ({} runs each):", args.runs);
                println!(
                    "  Hardware: {} cores, {:?}, {:.0} GFLOP/s peak",
                    hw.cpu.cores, hw.cpu.simd, hw.cpu.peak_gflops,
                );
                println!();
                println!(
                    "  {:>7}  {:>9}  {:>9}  {:>9}  {:>7}  {:>6}  {:>8}",
                    "Threads", "Encoder", "Decoder", "Total", "Speedup", "Eff%", "Serial%"
                );
                println!(
                    "  {:>7}  {:>9}  {:>9}  {:>9}  {:>7}  {:>6}  {:>8}",
                    "───────",
                    "─────────",
                    "─────────",
                    "─────────",
                    "───────",
                    "──────",
                    "────────"
                );
                for r in results {
                    let (speedup, eff, serial) = amdahl_metrics(r.threads, r.total_ms, base_total);
                    if r.threads == baseline.threads {
                        println!(
                            "  {:>7}  {:>7.1}ms  {:>7.1}ms  {:>7.1}ms  {:>6.2}x  {:>5.0}%  {:>7}",
                            r.threads, r.enc_ms, r.dec_ms, r.total_ms, speedup, eff, "—"
                        );
                    } else {
                        println!(
                            "  {:>7}  {:>7.1}ms  {:>7.1}ms  {:>7.1}ms  {:>6.2}x  {:>5.0}%  {:>6.1}%",
                            r.threads, r.enc_ms, r.dec_ms, r.total_ms, speedup, eff, serial * 100.0,
                        );
                    }
                }
                let last = results.last().unwrap();
                println!();
                println!(
                    "  Encoder: {:.2}x ({}→{} threads)",
                    baseline.enc_ms / last.enc_ms,
                    baseline.threads,
                    last.threads,
                );
                println!(
                    "  Decoder: {:.2}x ({}→{} threads)",
                    baseline.dec_ms / last.dec_ms.max(0.001),
                    baseline.threads,
                    last.threads,
                );
            },
        );
        None
    }
}

/// Run N transcriptions and accumulate profiling timings.
fn measure_sweep_runs(
    whisper: &crate::WhisperApr,
    samples: &[f32],
    options: &crate::TranscribeOptions,
    runs: usize,
) -> CliResult<(f64, f64, f64)> {
    let mut enc_total = 0.0;
    let mut dec_total = 0.0;
    let mut total_total = 0.0;
    for _ in 0..runs {
        let result = whisper
            .transcribe(samples, options.clone())
            .map_err(|e| CliError::InvalidArgument(format!("Transcribe: {e}")))?;
        if let Some(ref prof) = result.profiling {
            enc_total += prof.breakdown.get("encoder_ms").copied().unwrap_or(0.0);
            dec_total += prof.breakdown.get("decoder_ms").copied().unwrap_or(0.0)
                - prof.breakdown.get("mel_ms").copied().unwrap_or(0.0);
            total_total += prof.total_ms;
        }
    }
    Ok((enc_total, dec_total, total_total))
}

fn run_sweep_threads(
    sweep_str: &str,
    whisper: &crate::WhisperApr,
    samples: &[f32],
    audio_duration_s: f64,
    args: &AprProfileArgs,
    global: &super::super::super::args::Args,
    hw: &trueno::HardwareCapability,
) -> CliResult<CommandResult> {
    use crate::TranscribeOptions;

    let thread_counts: Vec<u32> = sweep_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|&n| n > 0)
        .collect();

    if thread_counts.is_empty() {
        return Err(CliError::InvalidArgument(
            "No valid thread counts in --sweep-threads".into(),
        ));
    }

    let mut results: Vec<SweepResult> = Vec::new();
    let mut options = TranscribeOptions::default();
    options.profile = true;

    for &tc in &thread_counts {
        // Reconfigure rayon thread pool for this sweep point
        let actual = crate::parallel::configure_thread_pool(Some(tc))
            .map_err(|e| CliError::InvalidArgument(format!("Thread pool: {e}")))?;
        if global.verbose {
            eprintln!("[SWEEP] Threads: {actual}");
        }

        // Warmup
        for _ in 0..args.warmup {
            let _ = whisper.transcribe(samples, options.clone());
        }

        // Measure
        let (enc_total, dec_total, total_total) =
            measure_sweep_runs(whisper, samples, &options, args.runs)?;
        let n = args.runs.max(1) as f64;
        results.push(SweepResult {
            threads: tc,
            enc_ms: enc_total / n,
            dec_ms: dec_total / n,
            total_ms: total_total / n,
        });
    }

    // Compute speedups relative to minimum thread count result
    let baseline = results
        .iter()
        .min_by(|a, b| a.threads.cmp(&b.threads))
        .unwrap();
    let base_total = baseline.total_ms;

    let output = format_sweep_output(
        &results,
        baseline,
        base_total,
        audio_duration_s,
        args,
        global,
        hw,
    );
    if let Some(out_str) = output {
        if let Some(ref out) = args.output {
            fs::write(out, &out_str)
                .map_err(|e| CliError::InvalidArgument(format!("Write: {e}")))?;
        } else {
            println!("{out_str}");
        }
    }

    Ok(CommandResult::success("Thread sweep complete"))
}

/// Run instrumented transcription with per-step timing breakdown.
///
/// Uses `TranscribeOptions { profile: true }` so mel, encoder, and decoder
/// timings come from direct instrumentation inside `transcribe_single_chunk`
/// rather than approximate subtraction. Outputs text, JSON, or renacer
/// (Chrome Trace Event) format.
pub(in super::super) fn run_profile(
    args: &AprProfileArgs,
    global: &super::super::super::args::Args,
) -> CliResult<CommandResult> {
    use crate::{TranscribeOptions, WhisperApr};

    // Configure thread pool: user override or smart default (in configure_thread_pool)
    let thread_count = crate::parallel::configure_thread_pool(args.threads)
        .map_err(|e| CliError::InvalidArgument(format!("Thread pool: {e}")))?;
    if global.verbose {
        eprintln!("[INFO] Using {thread_count} thread(s) for inference");
    }

    // Load model
    let load_start = Instant::now();
    let model_bytes =
        fs::read(&args.model).map_err(|e| CliError::InvalidArgument(format!("Model: {e}")))?;
    let whisper = WhisperApr::load_from_apr(&model_bytes)
        .map_err(|e| CliError::InvalidArgument(format!("Model load: {e}")))?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // WAPR-PROFILE-001 Gap 2: Hardware roofline detection
    let hw = trueno::HardwareCapability::detect();
    if global.verbose {
        eprintln!(
            "[INFO] Hardware: {} cores, {:?} SIMD, {:.0} GFLOP/s peak, {:.1} GB/s BW, AI balance: {:.1} F/B",
            hw.cpu.cores,
            hw.cpu.simd,
            hw.cpu.peak_gflops,
            hw.cpu.memory_bw_gbps,
            hw.roofline.cpu_arithmetic_intensity,
        );
    }

    // Load audio
    let audio_bytes =
        fs::read(&args.audio).map_err(|e| CliError::InvalidArgument(format!("Audio: {e}")))?;
    let samples =
        super::super::super::commands::load_audio_samples(args.audio.as_path(), &audio_bytes)?;
    let audio_duration_s = samples.len() as f64 / 16000.0;

    // WAPR-PROFILE-001 Gap 3: Thread scaling sweep
    if let Some(ref sweep_str) = args.sweep_threads {
        return run_sweep_threads(
            sweep_str,
            &whisper,
            &samples,
            audio_duration_s,
            args,
            global,
            &hw,
        );
    }

    let total_runs = args.warmup + args.runs;
    let mut run_results: Vec<ProfileRun> = Vec::with_capacity(args.runs);

    let mut options = TranscribeOptions::default();
    options.profile = true;

    for run_idx in 0..total_runs {
        let is_warmup = run_idx < args.warmup;

        // Single transcribe() call with profile: true — no redundant mel+encode
        let run_start = Instant::now();
        let result = whisper
            .transcribe(&samples, options.clone())
            .map_err(|e| CliError::InvalidArgument(format!("Transcribe: {e}")))?;
        let wall_ms = run_start.elapsed().as_secs_f64() * 1000.0;

        // Extract directly-instrumented timings from ProfilingStats breakdown
        let (mel_ms, enc_ms, dec_ms, total_ms) = if let Some(ref prof) = result.profiling {
            let mel = prof.breakdown.get("mel_ms").copied().unwrap_or(0.0);
            let enc = prof.breakdown.get("encoder_ms").copied().unwrap_or(0.0);
            let dec = prof.breakdown.get("decoder_ms").copied().unwrap_or(0.0);
            (mel, enc, dec, prof.total_ms)
        } else {
            (0.0, 0.0, 0.0, wall_ms)
        };

        // WAPR-PROFILE-001 Gap 1: Extract BrickProfiler category breakdown
        let brick_detail = result
            .profiling
            .as_ref()
            .and_then(|prof| extract_brick_detail(prof, &hw));

        let token_count: usize = result.segments.iter().map(|s| s.tokens.len()).sum();

        if !is_warmup {
            run_results.push(ProfileRun {
                mel_ms,
                encode_ms: enc_ms,
                decode_ms: dec_ms,
                total_ms,
                rtf: total_ms / 1000.0 / audio_duration_s,
                token_count,
                text: result.text.clone(),
                brick_detail,
                trace_json: result.profiling.as_ref().and_then(|p| p.trace_json.clone()),
            });
        }
    }

    // Compute averages
    let n = run_results.len().max(1) as f64;
    let summary = ProfileSummary {
        load_ms,
        avg_mel: run_results.iter().map(|r| r.mel_ms).sum::<f64>() / n,
        avg_enc: run_results.iter().map(|r| r.encode_ms).sum::<f64>() / n,
        avg_dec: run_results.iter().map(|r| r.decode_ms).sum::<f64>() / n,
        avg_total: run_results.iter().map(|r| r.total_ms).sum::<f64>() / n,
        avg_rtf: run_results.iter().map(|r| r.rtf).sum::<f64>() / n,
        avg_tokens: run_results.iter().map(|r| r.token_count).sum::<usize>()
            / run_results.len().max(1),
        text: run_results.last().map_or("", |r| r.text.as_str()),
        audio_duration_s,
        avg_brick_detail: compute_avg_brick_detail(&run_results),
        // Gap 5: Use last run's InferenceTracer JSON
        trace_json: run_results.last().and_then(|r| r.trace_json.clone()),
        // Gap 2: Hardware info
        hw_cores: hw.cpu.cores,
        hw_simd: format!("{:?}", hw.cpu.simd),
        hw_peak_gflops: hw.cpu.peak_gflops,
        hw_bw_gbps: hw.cpu.memory_bw_gbps,
        hw_balance_point: hw.roofline.cpu_arithmetic_intensity,
    };

    let output_str = match args.format.as_str() {
        "json" => summary.format_json(args),
        "renacer" => summary.format_renacer(args),
        _ => {
            // Text output — use emit_output for quiet/json global flags
            emit_output(global, || {}, || summary.print_table(args));
            return Ok(CommandResult::success("Profile complete"));
        }
    };

    if let Some(ref out) = args.output {
        fs::write(out, &output_str)
            .map_err(|e| CliError::InvalidArgument(format!("Write: {e}")))?;
    } else {
        println!("{output_str}");
    }

    Ok(CommandResult::success("Profile complete"))
}

/// Classify roofline bound from achieved vs peak GFLOP/s
fn classify_roofline(blis_total_gflops: f64, peak_gflops: f64) -> (&'static str, f64) {
    let util_pct = if peak_gflops > 0.0 && blis_total_gflops > 0.0 {
        blis_total_gflops / peak_gflops * 100.0
    } else {
        0.0
    };
    let bound = if blis_total_gflops > 0.0 {
        if util_pct > 50.0 {
            "compute (efficient)"
        } else if util_pct > 10.0 {
            "compute (low util)"
        } else {
            "memory"
        }
    } else {
        "unknown"
    };
    (bound, util_pct)
}

/// Extract BrickProfiler category breakdown from profiling stats.
fn extract_brick_detail(
    prof: &crate::ProfilingStats,
    hw: &trueno::HardwareCapability,
) -> Option<BrickDetail> {
    let get = |key: &str| prof.breakdown.get(key).copied().unwrap_or(0.0);

    let norm = get("brick_norm_ms");
    let attn = get("brick_attn_ms");
    let ffn = get("brick_ffn_ms");
    let other = get("brick_other_ms");

    if norm == 0.0 && attn == 0.0 && ffn == 0.0 {
        return None;
    }

    let blis_total_gflops = get("blis_total_gflops");
    let (roofline_bound, roofline_util_pct) =
        classify_roofline(blis_total_gflops, hw.cpu.peak_gflops);

    Some(BrickDetail {
        norm_ms: norm,
        attn_ms: attn,
        ffn_ms: ffn,
        other_ms: other,
        page_faults_minor: get("page_faults_minor") as u64,
        page_faults_major: get("page_faults_major") as u64,
        ln_bottleneck: get("brick_LayerNorm_bottleneck") as u8,
        attn_bottleneck: get("brick_AttentionScore_bottleneck") as u8,
        ffn_bottleneck: get("brick_GateProjection_bottleneck") as u8,
        ln_cycles_per_elem: get("brick_LayerNorm_cycles_per_elem"),
        attn_cycles_per_elem: get("brick_AttentionScore_cycles_per_elem"),
        ffn_cycles_per_elem: get("brick_GateProjection_cycles_per_elem"),
        blis_total_gflops,
        blis_macro_gflops: get("blis_macro_gflops"),
        blis_micro_gflops: get("blis_micro_gflops"),
        blis_pack_pct: get("blis_pack_pct"),
        blis_macro_calls: get("blis_macro_calls") as u64,
        roofline_bound,
        roofline_util_pct,
    })
}

/// Compute averaged BrickDetail across multiple profiling runs.
fn compute_avg_brick_detail(runs: &[ProfileRun]) -> Option<AvgBrickDetail> {
    if runs.is_empty() || !runs.iter().all(|r| r.brick_detail.is_some()) {
        return None;
    }
    let n = runs.len() as f64;
    let avg = |f: fn(&BrickDetail) -> f64| -> f64 {
        runs.iter()
            .map(|r| f(r.brick_detail.as_ref().unwrap()))
            .sum::<f64>()
            / n
    };
    let last = runs.last().unwrap().brick_detail.as_ref().unwrap();
    Some(AvgBrickDetail {
        norm_ms: avg(|b| b.norm_ms),
        attn_ms: avg(|b| b.attn_ms),
        ffn_ms: avg(|b| b.ffn_ms),
        other_ms: avg(|b| b.other_ms),
        page_faults_minor: last.page_faults_minor,
        page_faults_major: last.page_faults_major,
        ln_bottleneck: last.ln_bottleneck,
        attn_bottleneck: last.attn_bottleneck,
        ffn_bottleneck: last.ffn_bottleneck,
        ln_cycles_per_elem: avg(|b| b.ln_cycles_per_elem),
        attn_cycles_per_elem: avg(|b| b.attn_cycles_per_elem),
        ffn_cycles_per_elem: avg(|b| b.ffn_cycles_per_elem),
        blis_total_gflops: avg(|b| b.blis_total_gflops),
        blis_macro_gflops: avg(|b| b.blis_macro_gflops),
        blis_micro_gflops: avg(|b| b.blis_micro_gflops),
        blis_pack_pct: avg(|b| b.blis_pack_pct),
        blis_macro_calls: last.blis_macro_calls,
        roofline_bound: last.roofline_bound,
        roofline_util_pct: avg(|b| b.roofline_util_pct),
    })
}

/// BrickProfiler category breakdown per run (WAPR-PROFILE-001 Gap 1)
#[derive(Debug, Clone)]
struct BrickDetail {
    norm_ms: f64,
    attn_ms: f64,
    ffn_ms: f64,
    other_ms: f64,
    page_faults_minor: u64,
    page_faults_major: u64,
    /// Bottleneck diagnosis (0=insufficient, 1=memory, 2=compute, 3=throttled, 4=balanced)
    ln_bottleneck: u8,
    attn_bottleneck: u8,
    ffn_bottleneck: u8,
    /// Cycles per element (frequency-invariant)
    ln_cycles_per_elem: f64,
    attn_cycles_per_elem: f64,
    ffn_cycles_per_elem: f64,
    /// WAPR-PROFILE-001 Gap 4: BLIS GEMM hierarchy stats
    blis_total_gflops: f64,
    blis_macro_gflops: f64,
    blis_micro_gflops: f64,
    blis_pack_pct: f64,
    blis_macro_calls: u64,
    /// WAPR-PROFILE-001 Gap 2: Roofline classification
    roofline_bound: &'static str,
    roofline_util_pct: f64,
}

/// Timing data for a single profiling run
struct ProfileRun {
    mel_ms: f64,
    encode_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    rtf: f64,
    token_count: usize,
    text: String,
    /// BrickProfiler category breakdown (WAPR-PROFILE-001 Gap 1)
    brick_detail: Option<BrickDetail>,
    /// WAPR-PROFILE-001 Gap 5: Structured Chrome Trace JSON from InferenceTracer
    trace_json: Option<String>,
}

/// Averaged BrickProfiler category breakdown
#[derive(Debug, Clone)]
struct AvgBrickDetail {
    norm_ms: f64,
    attn_ms: f64,
    ffn_ms: f64,
    other_ms: f64,
    page_faults_minor: u64,
    page_faults_major: u64,
    ln_bottleneck: u8,
    attn_bottleneck: u8,
    ffn_bottleneck: u8,
    ln_cycles_per_elem: f64,
    attn_cycles_per_elem: f64,
    ffn_cycles_per_elem: f64,
    /// WAPR-PROFILE-001 Gap 4: BLIS GEMM hierarchy stats
    blis_total_gflops: f64,
    blis_macro_gflops: f64,
    blis_micro_gflops: f64,
    blis_pack_pct: f64,
    blis_macro_calls: u64,
    /// WAPR-PROFILE-001 Gap 2: Roofline classification
    roofline_bound: &'static str,
    roofline_util_pct: f64,
}

/// Aggregated profile summary for output formatting
struct ProfileSummary<'text> {
    load_ms: f64,
    avg_mel: f64,
    avg_enc: f64,
    avg_dec: f64,
    avg_total: f64,
    avg_rtf: f64,
    avg_tokens: usize,
    text: &'text str,
    audio_duration_s: f64,
    /// BrickProfiler category breakdown averaged across runs (WAPR-PROFILE-001 Gap 1)
    avg_brick_detail: Option<AvgBrickDetail>,
    /// WAPR-PROFILE-001 Gap 5: Structured Chrome Trace JSON from last run's InferenceTracer
    trace_json: Option<String>,
    /// WAPR-PROFILE-001 Gap 2: Hardware roofline info
    hw_cores: usize,
    hw_simd: String,
    hw_peak_gflops: f64,
    hw_bw_gbps: f64,
    hw_balance_point: f64,
}

impl ProfileSummary<'_> {
    fn format_json(&self, args: &AprProfileArgs) -> String {
        // WAPR-PROFILE-001 Gap 1: BrickProfiler category breakdown in JSON
        let mut brick_json = String::new();
        if let Some(ref bd) = self.avg_brick_detail {
            let total_brick = bd.norm_ms + bd.attn_ms + bd.ffn_ms + bd.other_ms;
            let pct = |ms: f64| {
                if total_brick > 0.0 {
                    ms / total_brick * 100.0
                } else {
                    0.0
                }
            };
            brick_json = format!(
                concat!(
                    ",\"brick_profile\":{{",
                    "\"norm_ms\":{:.2},\"attn_ms\":{:.2},\"ffn_ms\":{:.2},\"other_ms\":{:.2},",
                    "\"norm_pct\":{:.1},\"attn_pct\":{:.1},\"ffn_pct\":{:.1},",
                    "\"page_faults\":{{\"minor\":{},\"major\":{}}},",
                    "\"cycles_per_elem\":{{\"ln\":{:.1},\"attn\":{:.1},\"ffn\":{:.1}}},",
                    "\"bottleneck\":{{\"ln\":{},\"attn\":{},\"ffn\":{}}},",
                    "\"blis\":{{\"total_gflops\":{:.2},\"macro_gflops\":{:.2},",
                    "\"micro_gflops\":{:.2},\"pack_pct\":{:.1},\"macro_calls\":{}}},",
                    "\"roofline\":{{\"bound\":\"{}\",\"util_pct\":{:.1}}}}}"
                ),
                bd.norm_ms,
                bd.attn_ms,
                bd.ffn_ms,
                bd.other_ms,
                pct(bd.norm_ms),
                pct(bd.attn_ms),
                pct(bd.ffn_ms),
                bd.page_faults_minor,
                bd.page_faults_major,
                bd.ln_cycles_per_elem,
                bd.attn_cycles_per_elem,
                bd.ffn_cycles_per_elem,
                bd.ln_bottleneck,
                bd.attn_bottleneck,
                bd.ffn_bottleneck,
                bd.blis_total_gflops,
                bd.blis_macro_gflops,
                bd.blis_micro_gflops,
                bd.blis_pack_pct,
                bd.blis_macro_calls,
                bd.roofline_bound,
                bd.roofline_util_pct,
            );
        }
        // Gap 2: Hardware info in JSON
        let hw_json = format!(
            concat!(
                ",\"hardware\":{{",
                "\"cores\":{},\"simd\":\"{}\",\"peak_gflops\":{:.1},",
                "\"bw_gbps\":{:.1},\"balance_point\":{:.1}}}"
            ),
            self.hw_cores,
            self.hw_simd,
            self.hw_peak_gflops,
            self.hw_bw_gbps,
            self.hw_balance_point,
        );
        format!(
            concat!(
                "{{\"model\":\"{}\",\"audio\":\"{}\",\"audio_duration_s\":{:.3},",
                "\"warmup\":{},\"runs\":{},",
                "\"avg_ms\":{{\"load\":{:.1},\"mel\":{:.1},\"encode\":{:.1},",
                "\"decode\":{:.1},\"total\":{:.1}}}{}{},",
                "\"rtf\":{:.3},\"tokens\":{},\"text\":\"{}\"}}"
            ),
            args.model.display(),
            args.audio.display(),
            self.audio_duration_s,
            args.warmup,
            args.runs,
            self.load_ms,
            self.avg_mel,
            self.avg_enc,
            self.avg_dec,
            self.avg_total,
            brick_json,
            hw_json,
            self.avg_rtf,
            self.avg_tokens,
            self.text.replace('"', "\\\"")
        )
    }

    /// Format as Chrome Trace Event JSON (renacer-compatible).
    ///
    /// Produces a `traceEvents` array with duration ("X") events for each
    /// pipeline stage. Timestamps are in microseconds. Compatible with
    /// `chrome://tracing`, Perfetto UI, and `renacer --format json`.
    fn format_renacer(&self, args: &AprProfileArgs) -> String {
        // WAPR-PROFILE-001 Gap 5: Prefer InferenceTracer's structured trace when available
        if let Some(ref trace) = self.trace_json {
            return trace.clone();
        }

        let mut events = Vec::new();
        let mut ts_us: f64 = 0.0;

        // Model load
        let load_dur = self.load_ms * 1000.0;
        events.push(format!(
            concat!(
                "{{\"name\":\"model_load\",\"cat\":\"apr_profile\",\"ph\":\"X\",",
                "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":1,",
                "\"args\":{{\"model\":\"{}\"}}}}"
            ),
            ts_us,
            load_dur,
            args.model.display()
        ));
        ts_us += load_dur;

        // Mel spectrogram
        let mel_dur = self.avg_mel * 1000.0;
        events.push(format!(
            concat!(
                "{{\"name\":\"mel_spectrogram\",\"cat\":\"apr_profile\",\"ph\":\"X\",",
                "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":1}}"
            ),
            ts_us, mel_dur
        ));
        ts_us += mel_dur;

        // Encoder (parent span)
        let enc_dur = self.avg_enc * 1000.0;
        events.push(format!(
            concat!(
                "{{\"name\":\"encoder\",\"cat\":\"apr_profile\",\"ph\":\"X\",",
                "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":1}}"
            ),
            ts_us, enc_dur
        ));
        // Encoder BrickProfiler sub-spans (nested on tid 2)
        if let Some(ref bd) = self.avg_brick_detail {
            let mut sub_ts = ts_us;
            // Other (conv_frontend) first
            let other_dur = bd.other_ms * 1000.0;
            if other_dur > 0.0 {
                events.push(format!(
                    concat!(
                        "{{\"name\":\"conv_frontend\",\"cat\":\"brick_profile\",\"ph\":\"X\",",
                        "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":2}}"
                    ),
                    sub_ts, other_dur
                ));
                sub_ts += other_dur;
            }
            // Norm
            let norm_dur = bd.norm_ms * 1000.0;
            events.push(format!(
                concat!(
                    "{{\"name\":\"norm\",\"cat\":\"brick_profile\",\"ph\":\"X\",",
                    "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":2,",
                    "\"args\":{{\"cycles_per_elem\":{:.1},\"bottleneck\":{}}}}}"
                ),
                sub_ts, norm_dur, bd.ln_cycles_per_elem, bd.ln_bottleneck
            ));
            sub_ts += norm_dur;
            // Attention
            let attn_dur = bd.attn_ms * 1000.0;
            events.push(format!(
                concat!(
                    "{{\"name\":\"attention\",\"cat\":\"brick_profile\",\"ph\":\"X\",",
                    "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":2,",
                    "\"args\":{{\"cycles_per_elem\":{:.1},\"bottleneck\":{}}}}}"
                ),
                sub_ts, attn_dur, bd.attn_cycles_per_elem, bd.attn_bottleneck
            ));
            sub_ts += attn_dur;
            // FFN
            let ffn_dur = bd.ffn_ms * 1000.0;
            events.push(format!(
                concat!(
                    "{{\"name\":\"ffn\",\"cat\":\"brick_profile\",\"ph\":\"X\",",
                    "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":2,",
                    "\"args\":{{\"cycles_per_elem\":{:.1},\"bottleneck\":{}}}}}"
                ),
                sub_ts, ffn_dur, bd.ffn_cycles_per_elem, bd.ffn_bottleneck
            ));
            // WAPR-PROFILE-001 Gap 4: BLIS GEMM hierarchy on tid 3
            if bd.blis_macro_calls > 0 {
                // BLIS spans the full encoder duration on tid 3
                let enc_start_us = ts_us;
                events.push(format!(
                    concat!(
                        "{{\"name\":\"blis_gemm\",\"cat\":\"blis_profile\",\"ph\":\"X\",",
                        "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":3,",
                        "\"args\":{{\"total_gflops\":{:.2},\"macro_gflops\":{:.2},",
                        "\"micro_gflops\":{:.2},\"pack_pct\":{:.1},\"calls\":{}}}}}"
                    ),
                    enc_start_us,
                    enc_dur,
                    bd.blis_total_gflops,
                    bd.blis_macro_gflops,
                    bd.blis_micro_gflops,
                    bd.blis_pack_pct,
                    bd.blis_macro_calls,
                ));
            }
        }
        ts_us += enc_dur;

        // Decoder
        let dec_dur = self.avg_dec * 1000.0;
        events.push(format!(
            concat!(
                "{{\"name\":\"decoder\",\"cat\":\"apr_profile\",\"ph\":\"X\",",
                "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":1,",
                "\"args\":{{\"tokens\":{},\"ms_per_token\":{:.1}}}}}"
            ),
            ts_us,
            dec_dur,
            self.avg_tokens,
            if self.avg_tokens > 0 {
                self.avg_dec / self.avg_tokens as f64
            } else {
                0.0
            }
        ));

        // Metadata event
        let meta = format!(
            concat!(
                "{{\"name\":\"process_name\",\"cat\":\"__metadata\",\"ph\":\"M\",",
                "\"ts\":0,\"pid\":1,\"tid\":0,",
                "\"args\":{{\"name\":\"apr profile ({} runs)\"}}}}"
            ),
            args.runs
        );

        format!(
            "{{\"traceEvents\":[{},{}],\"metadata\":{{\"audio\":\"{}\",\"audio_duration_s\":{:.3},\"rtf\":{:.3}}}}}",
            meta,
            events.join(","),
            args.audio.display(),
            self.audio_duration_s,
            self.avg_rtf,
        )
    }

    fn print_table(&self, args: &AprProfileArgs) {
        println!(
            "Pipeline Profile: {} runs (+ {} warmup)",
            args.runs, args.warmup
        );
        println!("  Model:    {}", args.model.display());
        println!(
            "  Audio:    {} ({:.2}s)",
            args.audio.display(),
            self.audio_duration_s
        );
        // WAPR-PROFILE-001 Gap 2: Hardware roofline info
        println!(
            "  Hardware: {} cores, {}, {:.0} GFLOP/s peak, {:.1} GB/s BW",
            self.hw_cores, self.hw_simd, self.hw_peak_gflops, self.hw_bw_gbps,
        );
        println!();
        println!("  Step          Avg (ms)    % of total");
        println!("  ────────────  ──────────  ──────────");
        println!("  Model load    {:>8.1}    (excluded)", self.load_ms);
        println!(
            "  Mel spec      {:>8.1}    {:>5.1}%",
            self.avg_mel,
            self.avg_mel / self.avg_total * 100.0
        );
        println!(
            "  Encoder       {:>8.1}    {:>5.1}%",
            self.avg_enc,
            self.avg_enc / self.avg_total * 100.0
        );
        if let Some(ref bd) = self.avg_brick_detail {
            let total_brick = bd.norm_ms + bd.attn_ms + bd.ffn_ms + bd.other_ms;
            let pct = |ms: f64| {
                if total_brick > 0.0 {
                    ms / total_brick * 100.0
                } else {
                    0.0
                }
            };
            if bd.other_ms > 0.0 {
                println!(
                    "    Conv frontend {:>6.1}    {:>5.1}%",
                    bd.other_ms,
                    bd.other_ms / self.avg_total * 100.0
                );
            }
            println!(
                "    Norm         {:>7.1}    {:>5.1}%  ({:.0}% of encoder)",
                bd.norm_ms,
                bd.norm_ms / self.avg_total * 100.0,
                pct(bd.norm_ms)
            );
            println!(
                "    Attention    {:>7.1}    {:>5.1}%  ({:.0}% of encoder)",
                bd.attn_ms,
                bd.attn_ms / self.avg_total * 100.0,
                pct(bd.attn_ms)
            );
            println!(
                "    FFN          {:>7.1}    {:>5.1}%  ({:.0}% of encoder)",
                bd.ffn_ms,
                bd.ffn_ms / self.avg_total * 100.0,
                pct(bd.ffn_ms)
            );
            // Cycles-per-element and bottleneck diagnosis
            println!("  ────────────  ──────────  ──────────");
            println!("  BrickProfiler Diagnosis:");
            println!(
                "    Norm:      {:.1} cyc/elem  {}",
                bd.ln_cycles_per_elem,
                bottleneck_label(bd.ln_bottleneck)
            );
            println!(
                "    Attention: {:.1} cyc/elem  {}",
                bd.attn_cycles_per_elem,
                bottleneck_label(bd.attn_bottleneck)
            );
            println!(
                "    FFN:       {:.1} cyc/elem  {}",
                bd.ffn_cycles_per_elem,
                bottleneck_label(bd.ffn_bottleneck)
            );
            if bd.page_faults_minor > 0 || bd.page_faults_major > 0 {
                println!(
                    "  Page faults:  {} minor, {} major",
                    bd.page_faults_minor, bd.page_faults_major
                );
            }
            // WAPR-PROFILE-001 Gap 4: BLIS GEMM hierarchy
            if bd.blis_macro_calls > 0 {
                println!("  ────────────  ──────────  ──────────");
                println!("  BLIS GEMM Hierarchy ({} calls):", bd.blis_macro_calls);
                println!("    Macro:   {:.1} GFLOP/s", bd.blis_macro_gflops);
                println!("    Micro:   {:.1} GFLOP/s", bd.blis_micro_gflops);
                println!("    Pack:    {:.1}% of GEMM time", bd.blis_pack_pct);
                println!("    Total:   {:.1} GFLOP/s", bd.blis_total_gflops);
                // Gap 2: Roofline classification
                println!(
                    "    Roofline: {} ({:.1}% of {:.0} GFLOP/s peak)",
                    bd.roofline_bound, bd.roofline_util_pct, self.hw_peak_gflops,
                );
            }
        }
        println!(
            "  Decoder       {:>8.1}    {:>5.1}%",
            self.avg_dec,
            self.avg_dec / self.avg_total * 100.0
        );
        println!("  ────────────  ──────────  ──────────");
        println!("  Total         {:>8.1}    100.0%", self.avg_total);
        println!();
        println!("  RTF:    {:.2}x", self.avg_rtf);
        println!("  Tokens: {}", self.avg_tokens);
        if args.per_token && self.avg_tokens > 0 {
            println!(
                "  ms/token (decode): {:.1}",
                self.avg_dec / self.avg_tokens as f64
            );
        }
        println!("  Text:   \"{}\"", self.text.trim());
        Self::print_rtf_indicator(self.avg_rtf);
    }

    fn print_rtf_indicator(rtf: f64) {
        println!("{}", rtf_tier_label(rtf));
    }
}

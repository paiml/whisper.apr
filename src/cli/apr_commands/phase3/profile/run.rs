#![allow(clippy::unwrap_used)]
// pre-existing surfaced by #55
// `cli` is now in default features (#55); items below are reachable only under
// the converter / phase3-encryption feature combos and lint as dead code
// when only `cli` is on. This is pre-existing technical debt — file follow-up.
#![allow(dead_code)]
#![allow(clippy::all, clippy::pedantic)] // pre-existing tech debt surfaced by #55

//! `run_profile` entry point + BrickDetail extraction + roofline classification.

use std::fs;
use std::time::Instant;

use super::super::super::super::apr_args::AprProfileArgs;
use super::super::super::super::commands::{CliError, CliResult, CommandResult};
use super::super::emit_output;
use super::sweep::run_sweep_threads;
use super::types::{AvgBrickDetail, BrickDetail, ProfileRun, ProfileSummary};

/// Run instrumented transcription with per-step timing breakdown.
///
/// Uses `TranscribeOptions { profile: true }` so mel, encoder, and decoder
/// timings come from direct instrumentation inside `transcribe_single_chunk`
/// rather than approximate subtraction. Outputs text, JSON, or renacer
/// (Chrome Trace Event) format.
pub(in super::super::super) fn run_profile(
    args: &AprProfileArgs,
    global: &super::super::super::super::args::Args,
) -> CliResult<CommandResult> {
    use crate::WhisperApr;

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
    let samples = super::super::super::super::commands::load_audio_samples(
        args.audio.as_path(),
        &audio_bytes,
    )?;
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

    let run_results = collect_runs(&whisper, &samples, args, audio_duration_s, &hw)?;
    let summary = summarize(run_results, load_ms, audio_duration_s, &hw);
    emit_summary(&summary, args, global)
}

/// Run warmup + measurement iterations and collect timings for each measured run.
fn collect_runs(
    whisper: &crate::WhisperApr,
    samples: &[f32],
    args: &AprProfileArgs,
    audio_duration_s: f64,
    hw: &trueno::HardwareCapability,
) -> CliResult<Vec<ProfileRun>> {
    use crate::TranscribeOptions;

    let total_runs = args.warmup + args.runs;
    let mut run_results: Vec<ProfileRun> = Vec::with_capacity(args.runs);

    let mut options = TranscribeOptions::default();
    options.profile = true;

    for run_idx in 0..total_runs {
        let is_warmup = run_idx < args.warmup;

        // Single transcribe() call with profile: true — no redundant mel+encode
        let run_start = Instant::now();
        let result = whisper
            .transcribe(samples, options.clone())
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
            .and_then(|prof| extract_brick_detail(prof, hw));

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

    Ok(run_results)
}

/// Build a `ProfileSummary` from collected runs.
fn summarize<'a>(
    run_results: Vec<ProfileRun>,
    load_ms: f64,
    audio_duration_s: f64,
    hw: &trueno::HardwareCapability,
) -> OwnedSummary {
    let n = run_results.len().max(1) as f64;
    let text = run_results.last().map_or(String::new(), |r| r.text.clone());
    let trace_json = run_results.last().and_then(|r| r.trace_json.clone());
    let avg_brick_detail = compute_avg_brick_detail(&run_results);
    let avg_mel = run_results.iter().map(|r| r.mel_ms).sum::<f64>() / n;
    let avg_enc = run_results.iter().map(|r| r.encode_ms).sum::<f64>() / n;
    let avg_dec = run_results.iter().map(|r| r.decode_ms).sum::<f64>() / n;
    let avg_total = run_results.iter().map(|r| r.total_ms).sum::<f64>() / n;
    let avg_rtf = run_results.iter().map(|r| r.rtf).sum::<f64>() / n;
    let avg_tokens =
        run_results.iter().map(|r| r.token_count).sum::<usize>() / run_results.len().max(1);
    OwnedSummary {
        load_ms,
        avg_mel,
        avg_enc,
        avg_dec,
        avg_total,
        avg_rtf,
        avg_tokens,
        text,
        audio_duration_s,
        avg_brick_detail,
        trace_json,
        hw_cores: hw.cpu.cores,
        hw_simd: format!("{:?}", hw.cpu.simd),
        hw_peak_gflops: hw.cpu.peak_gflops,
        hw_bw_gbps: hw.cpu.memory_bw_gbps,
        hw_balance_point: hw.roofline.cpu_arithmetic_intensity,
    }
}

/// Owned version of `ProfileSummary` (borrows the text string). Used because
/// `ProfileSummary<'text>` borrows the final text, but we need to keep it alive
/// across `emit_summary`.
struct OwnedSummary {
    load_ms: f64,
    avg_mel: f64,
    avg_enc: f64,
    avg_dec: f64,
    avg_total: f64,
    avg_rtf: f64,
    avg_tokens: usize,
    text: String,
    audio_duration_s: f64,
    avg_brick_detail: Option<AvgBrickDetail>,
    trace_json: Option<String>,
    hw_cores: usize,
    hw_simd: String,
    hw_peak_gflops: f64,
    hw_bw_gbps: f64,
    hw_balance_point: f64,
}

impl OwnedSummary {
    fn as_summary(&self) -> ProfileSummary<'_> {
        ProfileSummary {
            load_ms: self.load_ms,
            avg_mel: self.avg_mel,
            avg_enc: self.avg_enc,
            avg_dec: self.avg_dec,
            avg_total: self.avg_total,
            avg_rtf: self.avg_rtf,
            avg_tokens: self.avg_tokens,
            text: self.text.as_str(),
            audio_duration_s: self.audio_duration_s,
            avg_brick_detail: self.avg_brick_detail.clone(),
            trace_json: self.trace_json.clone(),
            hw_cores: self.hw_cores,
            hw_simd: self.hw_simd.clone(),
            hw_peak_gflops: self.hw_peak_gflops,
            hw_bw_gbps: self.hw_bw_gbps,
            hw_balance_point: self.hw_balance_point,
        }
    }
}

/// Emit the summary in the user-requested format.
fn emit_summary(
    summary: &OwnedSummary,
    args: &AprProfileArgs,
    global: &super::super::super::super::args::Args,
) -> CliResult<CommandResult> {
    let view = summary.as_summary();
    let output_str = match args.format.as_str() {
        "json" => view.format_json(args),
        "renacer" => view.format_renacer(args),
        _ => {
            // Text output — use emit_output for quiet/json global flags
            emit_output(global, || {}, || view.print_table(args));
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

#![allow(clippy::unwrap_used)] // pre-existing surfaced by #55
// `cli` is now in default features (#55); items below are reachable only under
// the converter / phase3-encryption feature combos and lint as dead code
// when only `cli` is on. This is pre-existing technical debt — file follow-up.
#![allow(dead_code)]
#![allow(clippy::all, clippy::pedantic)] // pre-existing tech debt surfaced by #55

//! Thread-scaling sweep (WAPR-PROFILE-001 Gap 3).
//!
//! Runs transcription at each thread count and reports speedup, efficiency,
//! and Amdahl serial fraction. Computes `s` from `T(N) = T(1) * (s + (1-s)/N)`.

use std::fs;

use super::super::super::super::apr_args::AprProfileArgs;
use super::super::super::super::commands::{CliError, CliResult, CommandResult};
use super::super::emit_output;

#[allow(clippy::too_many_arguments)]
pub(super) struct SweepResult {
    pub threads: u32,
    pub enc_ms: f64,
    pub dec_ms: f64,
    pub total_ms: f64,
}

/// Amdahl's law metrics for a thread sweep data point.
pub(super) fn amdahl_metrics(threads: u32, total_ms: f64, base_total: f64) -> (f64, f64, f64) {
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
    global: &super::super::super::super::args::Args,
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
            || print_sweep_table(results, baseline, base_total, args, hw),
        );
        None
    }
}

/// Print the human-readable sweep table.
fn print_sweep_table(
    results: &[SweepResult],
    baseline: &SweepResult,
    base_total: f64,
    args: &AprProfileArgs,
    hw: &trueno::HardwareCapability,
) {
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
        "───────", "─────────", "─────────", "─────────", "───────", "──────", "────────"
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
                r.threads,
                r.enc_ms,
                r.dec_ms,
                r.total_ms,
                speedup,
                eff,
                serial * 100.0,
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

/// Entry point: run a thread-count sweep and emit results.
pub(super) fn run_sweep_threads(
    sweep_str: &str,
    whisper: &crate::WhisperApr,
    samples: &[f32],
    audio_duration_s: f64,
    args: &AprProfileArgs,
    global: &super::super::super::super::args::Args,
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

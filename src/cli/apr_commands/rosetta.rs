//! Rosetta stone subcommands for cross-format model operations

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use aprender::format::hexdump::{statistics_table, TensorStatistics};
use aprender::format::{diff_models, DiffOptions, RosettaStone};

use super::super::apr_args::{
    RosettaAction, RosettaArgs, RosettaConvertArgs, RosettaDiffArgs, RosettaFingerprintArgs,
    RosettaInspectArgs, RosettaVerifyArgs,
};
use super::super::commands::{CliError, CliResult, CommandResult};
use super::inspect_model;

pub(super) fn run_rosetta(
    args: &RosettaArgs,
    global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    match &args.action {
        RosettaAction::Inspect(a) => run_rosetta_inspect(a, global),
        RosettaAction::Convert(a) => run_rosetta_convert(a, global),
        RosettaAction::Verify(a) => run_rosetta_verify(a, global),
        RosettaAction::Diff(a) => run_rosetta_diff(a, global),
        RosettaAction::Fingerprint(a) => run_rosetta_fingerprint(a, global),
    }
}

pub(super) fn run_rosetta_inspect(
    args: &RosettaInspectArgs,
    global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    let report = inspect_model(&args.file)?;

    if global.json {
        let tensors: Vec<serde_json::Value> = report
            .tensors
            .iter()
            .map(|t| {
                let mut obj = serde_json::json!({
                    "name": t.name,
                    "dtype": t.dtype,
                    "shape": t.shape,
                    "size_bytes": t.size_bytes,
                });
                if let Some(stats) = &t.stats {
                    obj["stats"] = serde_json::json!({
                        "min": stats.min,
                        "max": stats.max,
                        "mean": stats.mean,
                        "std": stats.std,
                    });
                }
                obj
            })
            .collect();

        let json = serde_json::json!({
            "format": format!("{}", report.format),
            "file_size": report.file_size,
            "total_params": report.total_params,
            "architecture": report.architecture,
            "quantization": report.quantization,
            "metadata": report.metadata,
            "tensors": tensors,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        print!("{report}");
    }

    Ok(CommandResult::success(format!(
        "Rosetta inspect: {} tensors",
        report.tensors.len()
    )))
}

pub(super) fn run_rosetta_convert(
    args: &RosettaConvertArgs,
    global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    let start = Instant::now();
    let rosetta = RosettaStone::new();

    if !global.quiet {
        eprintln!(
            "Converting {} -> {}",
            args.source.display(),
            args.dest.display()
        );
    }

    let report = rosetta
        .convert(&args.source, &args.dest, None)
        .map_err(|e| CliError::InvalidArgument(e.to_string()))?;

    let elapsed = start.elapsed();

    if args.verify {
        if !global.quiet {
            eprintln!("Verifying conversion...");
        }
        let options = DiffOptions::default();
        let diff = diff_models(&args.source, &args.dest, options)
            .map_err(|e| CliError::InvalidArgument(e.to_string()))?;

        if !diff.is_identical() && !global.quiet {
            eprintln!("Warning: {} differences detected", diff.diff_count());
        }
    }

    if global.json {
        let json = serde_json::json!({
            "lossless": report.is_lossless(),
            "tensor_counts_match": report.tensor_counts_match(),
            "warnings": report.warnings,
            "modified_tensors": report.modified_tensors,
            "dropped_tensors": report.dropped_tensors,
            "duration_ms": elapsed.as_millis(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!(
            "Converted {} -> {}",
            args.source.display(),
            args.dest.display()
        );
        println!("Lossless: {}", report.is_lossless());
        if !report.warnings.is_empty() {
            println!("Warnings:");
            for w in &report.warnings {
                println!("  - {w}");
            }
        }
        println!("Duration: {:.1}s", elapsed.as_secs_f64());
    }

    Ok(CommandResult::success("Conversion complete"))
}

pub(super) fn run_rosetta_verify(
    args: &RosettaVerifyArgs,
    global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    let report = inspect_model(&args.file)?;

    let intermediate = aprender::format::FormatType::Apr;
    let verification = RosettaStone::new()
        .verify_roundtrip(&args.file, intermediate)
        .map_err(|e| CliError::InvalidArgument(e.to_string()))?;

    let passes = verification.passes_with_tolerance(args.tolerance);

    if global.json {
        let json = serde_json::json!({
            "is_equivalent": verification.is_equivalent,
            "passes_tolerance": passes,
            "tolerance": args.tolerance,
            "max_diff": verification.max_diff,
            "mean_diff": verification.mean_diff,
            "failed_tensors": verification.failed_tensors,
            "changed_metadata": verification.changed_metadata,
            "source_format": format!("{}", report.format),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!(
            "Round-trip verification (tolerance: {:.0e})",
            args.tolerance
        );
        println!("  Source format: {}", report.format);
        println!("  Equivalent: {}", verification.is_equivalent);
        println!("  Passes tolerance: {passes}");
        println!("  Max diff: {:.6e}", verification.max_diff);
        println!("  Mean diff: {:.6e}", verification.mean_diff);
        if !verification.failed_tensors.is_empty() {
            println!("  Failed tensors: {}", verification.failed_tensors.len());
            for t in &verification.failed_tensors {
                println!("    - {t}");
            }
        }
    }

    let status = if passes { "PASS" } else { "FAIL" };
    Ok(CommandResult::success(format!(
        "Verification {status} (max_diff={:.6e})",
        verification.max_diff
    )))
}

pub(super) fn run_rosetta_diff(
    args: &RosettaDiffArgs,
    global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    let options = DiffOptions::default().with_stats();

    let report = diff_models(&args.file1, &args.file2, options)
        .map_err(|e| CliError::InvalidArgument(e.to_string()))?;

    if global.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
    } else {
        println!(
            "Rosetta diff: {} vs {}",
            args.file1.display(),
            args.file2.display()
        );
        println!("Format: {} vs {}", report.format1, report.format2);

        if report.is_identical() {
            println!("Result: IDENTICAL");
        } else {
            let max_display = 20;
            let total = report.differences.len();
            println!("Differences: {total}");

            for entry in report.differences.iter().take(max_display) {
                println!(
                    "  [{:?}] {}: {} vs {}",
                    entry.category, entry.field, entry.value1, entry.value2
                );
            }

            if total > max_display {
                println!(
                    "\n  ... and {} more differences (use --json for full output)",
                    total - max_display
                );
            }
        }
    }

    Ok(CommandResult::success(format!(
        "Rosetta diff: {} differences",
        report.diff_count()
    )))
}

pub(super) fn run_rosetta_fingerprint(
    args: &RosettaFingerprintArgs,
    global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    let report = inspect_model(&args.file)?;

    // Build per-tensor statistics — parallel tensor loading via Rayon (WAPR-APR-CLI-030)
    let tensor_names: Vec<String> = report.tensors.iter().map(|t| t.name.clone()).collect();
    let skipped = AtomicUsize::new(0);

    let stats_list: Vec<TensorStatistics> = {
        use rayon::prelude::*;
        tensor_names
            .par_iter()
            .filter_map(|name| {
                let r = RosettaStone::new(); // thread-local instance
                if let Ok(data) = r.load_tensor_f32(&args.file, name) {
                    let shape = report
                        .tensors
                        .iter()
                        .find(|t| &t.name == name)
                        .map(|t| t.shape.clone())
                        .unwrap_or_default();
                    Some(TensorStatistics::from_f32(name, shape, &data))
                } else {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    None
                }
            })
            .collect()
    };

    let skipped_count = skipped.load(Ordering::Relaxed);
    if skipped_count > 0 && !global.quiet {
        eprintln!("  Skipped {skipped_count} tensors (cannot load as f32)");
    }

    if global.json {
        let tensors: Vec<serde_json::Value> = stats_list
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "shape": s.shape,
                    "dtype": s.dtype,
                    "min": s.min,
                    "max": s.max,
                    "mean": s.mean,
                    "std": s.std,
                    "nan_count": s.nan_count,
                    "inf_count": s.inf_count,
                    "zero_count": s.zero_count,
                    "has_anomalies": s.has_anomalies(),
                })
            })
            .collect();

        let json = serde_json::json!({
            "file": args.file.display().to_string(),
            "format": format!("{}", report.format),
            "total_params": report.total_params,
            "fingerprinted_tensors": stats_list.len(),
            "skipped_tensors": skipped_count,
            "tensors": tensors,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!("Fingerprint: {}", args.file.display());
        println!("Format: {}", report.format);
        println!(
            "Tensors: {} fingerprinted, {} skipped",
            stats_list.len(),
            skipped_count
        );
        println!();
        println!("{}", statistics_table(&stats_list));
    }

    Ok(CommandResult::success(format!(
        "Fingerprinted {} tensors",
        stats_list.len()
    )))
}

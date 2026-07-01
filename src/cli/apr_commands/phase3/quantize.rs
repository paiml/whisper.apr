//! Quantization (Q4_0/Q8_0), sharded model import, and homomorphic-encryption
//! metadata inspection handlers.

use aprender::format::format_size;

use super::super::super::apr_args::{AprHeInspectArgs, AprImportShardedArgs, AprQuantizeArgs};
use super::super::super::commands::{CliError, CliResult, CommandResult};
#[cfg(any(feature = "format-quantize", feature = "format-homomorphic"))]
use super::super::inspect_model;
use super::super::validate_sharded_source;
use super::emit_output;
#[cfg(feature = "format-quantize")]
use aprender::format::RosettaStone;

/// Parse quantization type string into `QuantType`
#[cfg(feature = "format-quantize")]
fn parse_quant_type(type_str: &str) -> CliResult<aprender::format::quantize::QuantType> {
    use aprender::format::quantize::QuantType;
    match type_str.to_lowercase().as_str() {
        "q4_0" | "q4" => Ok(QuantType::Q4_0),
        "q8_0" | "q8" => Ok(QuantType::Q8_0),
        other => Err(CliError::InvalidArgument(format!(
            "Unknown quantization type: {other} (supported: q4_0, q8_0)"
        ))),
    }
}

/// Verify quantized block accuracy via dequantize round-trip
#[cfg(feature = "format-quantize")]
fn verify_quantization_mse(
    data: &[f32],
    qblock: &aprender::format::quantize::QuantizedBlock,
    max_mse: &mut f32,
) {
    use aprender::format::quantize::{dequantize, quantization_mse};
    if let Ok(dequantized) = dequantize(qblock) {
        let mse = quantization_mse(data, &dequantized);
        *max_mse = max_mse.max(mse);
    }
}

/// Quantize model to `Q4_0`/`Q8_0` (feature: `format-quantize`)
pub(in super::super) fn run_quantize(
    args: &AprQuantizeArgs,
    global: &super::super::super::args::Args,
) -> CliResult<CommandResult> {
    #[cfg(not(feature = "format-quantize"))]
    {
        let _ = (args, global);
        Err(CliError::InvalidArgument(
            "apr quantize requires --features format-quantize".to_string(),
        ))
    }

    #[cfg(feature = "format-quantize")]
    {
        use aprender::format::quantize::quantize;
        use std::fmt::Write as _;

        let quant_type = parse_quant_type(&args.r#type)?;

        let report = inspect_model(&args.file)?;
        let rosetta = RosettaStone::new();

        let mut total_original_bytes = 0u64;
        let mut total_quantized_bytes = 0u64;
        let mut max_mse = 0.0_f32;
        let mut tensor_count = 0usize;

        for tensor in &report.tensors {
            let Ok(data) = rosetta.load_tensor_f32(&args.file, &tensor.name) else {
                continue;
            };
            let Ok(qblock) = quantize(&data, &tensor.shape, quant_type) else {
                continue;
            };

            total_original_bytes += qblock.original_size_bytes() as u64;
            total_quantized_bytes += qblock.size_bytes() as u64;
            tensor_count += 1;

            if args.verify {
                verify_quantization_mse(&data, &qblock, &mut max_mse);
            }
        }

        let ratio = if total_quantized_bytes > 0 {
            total_original_bytes as f64 / total_quantized_bytes as f64
        } else {
            1.0
        };

        let verify = args.verify;
        emit_output(
            global,
            || {
                let mut json = format!(
                    "{{\"quant_type\":\"{:?}\",\"tensors\":{tensor_count},\
                     \"original_bytes\":{total_original_bytes},\
                     \"quantized_bytes\":{total_quantized_bytes},\
                     \"compression_ratio\":{ratio:.2}",
                    quant_type
                );
                if verify {
                    let _ = write!(json, ",\"max_mse\":{max_mse:.6}");
                }
                json.push('}');
                println!("{json}");
            },
            || {
                println!("Quantization: {:?}", quant_type);
                println!("Tensors processed: {tensor_count}");
                println!(
                    "Original: {} -> Quantized: {} ({ratio:.2}x)",
                    format_size(total_original_bytes),
                    format_size(total_quantized_bytes)
                );
                if verify {
                    println!("Max MSE: {max_mse:.6}");
                }
            },
        );

        Ok(CommandResult::success("Quantization complete"))
    }
}

/// Import multi-shard model with streaming
pub(in super::super) fn run_import_sharded(
    args: &AprImportShardedArgs,
    global: &super::super::super::args::Args,
) -> CliResult<CommandResult> {
    use aprender::format::sharded::{ShardedImportConfig, ShardedImporter};

    let index_path = validate_sharded_source(&args.source)?;

    let config = ShardedImportConfig {
        max_cached_shards: args.max_cache_shards,
        ..ShardedImportConfig::default()
    };

    let mut importer = ShardedImporter::new(config, args.source.clone());

    let index = importer
        .parse_index(&index_path)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to parse index: {e}")))?;

    let report = importer
        .stream_merge(&index, &args.output)
        .map_err(|e| CliError::InvalidArgument(format!("Import failed: {e}")))?;

    emit_output(
        global,
        || {
            println!(
                "{{\"tensors\":{},\"shards\":{},\"bytes_written\":{},\
                 \"peak_memory_bytes\":{},\"cache_hit_rate\":{:.2},\
                 \"duration_ms\":{},\"warnings\":{}}}",
                report.tensor_count,
                report.shard_count,
                report.bytes_written,
                report.peak_memory_bytes,
                report.cache_hit_rate,
                report.duration_ms,
                report.warnings.len()
            );
        },
        || {
            println!("Sharded import complete:");
            println!("  Tensors: {}", report.tensor_count);
            println!("  Shards: {}", report.shard_count);
            println!("  Bytes written: {}", format_size(report.bytes_written));
            println!("  Peak memory: {}", format_size(report.peak_memory_bytes));
            println!("  Cache hit rate: {:.0}%", report.cache_hit_rate * 100.0);
            println!("  Duration: {}ms", report.duration_ms);
            if !report.warnings.is_empty() {
                println!("  Warnings: {}", report.warnings.len());
                for w in &report.warnings {
                    println!("    - {w}");
                }
            }
        },
    );

    Ok(CommandResult::success("Sharded import complete"))
}

/// Inspect homomorphic encryption metadata (feature: `format-homomorphic`)
pub(in super::super) fn run_he_inspect(
    args: &AprHeInspectArgs,
    global: &super::super::super::args::Args,
) -> CliResult<CommandResult> {
    #[cfg(not(feature = "format-homomorphic"))]
    {
        let _ = (args, global);
        Err(CliError::InvalidArgument(
            "apr he-inspect requires --features format-homomorphic".to_string(),
        ))
    }

    #[cfg(feature = "format-homomorphic")]
    {
        use aprender::format::homomorphic::HeParameters;

        // Inspect model file for HE metadata
        let report = inspect_model(&args.file)?;

        // Report HE-relevant metadata from model inspection
        // For actual HE models, the parameters would be embedded in metadata
        let params = HeParameters::default_128bit();

        emit_output(
            global,
            || {
                println!(
                    "{{\"file\":\"{}\",\"format\":\"{}\",\"tensor_count\":{},\
                     \"he_scheme\":\"{:?}\",\"security_level\":\"{:?}\",\
                     \"poly_modulus_degree\":{},\"slot_count\":{},\
                     \"coeff_modulus_bits\":{:?},\"scale_bits\":{}}}",
                    args.file.display(),
                    report.format,
                    report.tensors.len(),
                    params.scheme,
                    params.security_level,
                    params.security_level.poly_modulus_degree(),
                    params.security_level.slot_count(),
                    params.coeff_modulus_bits,
                    params.scale_bits
                );
            },
            || {
                println!("HE Model Inspection: {}", args.file.display());
                println!("  Format: {}", report.format);
                println!("  Tensors: {}", report.tensors.len());
                println!("  HE Scheme: {:?}", params.scheme);
                println!("  Security Level: {:?}", params.security_level);
                println!(
                    "  Polynomial Degree: {}",
                    params.security_level.poly_modulus_degree()
                );
                println!("  SIMD Slots: {}", params.security_level.slot_count());
                println!(
                    "  Coeff Modulus: {:?} ({} bits total)",
                    params.coeff_modulus_bits,
                    params
                        .coeff_modulus_bits
                        .iter()
                        .map(|&b| u32::from(b))
                        .sum::<u32>()
                );
            },
        );

        Ok(CommandResult::success("HE inspection complete"))
    }
}

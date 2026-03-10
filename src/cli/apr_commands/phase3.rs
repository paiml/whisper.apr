//! Phase 3: Feature-gated CLI handlers (security, quantization, profiling)

use std::fs;
use std::time::Instant;

use aprender::format::format_size;

use super::super::apr_args::{
    AprDecryptArgs, AprEncryptArgs, AprHeInspectArgs, AprImportShardedArgs, AprProfileArgs,
    AprQuantizeArgs, AprSignArgs, AprVerifySigArgs,
};
use super::super::commands::{CliError, CliResult, CommandResult};
#[cfg(any(feature = "format-quantize", feature = "format-homomorphic"))]
use super::inspect_model;
#[cfg(feature = "format-encryption")]
use super::require_password;
use super::{rtf_tier_label, validate_sharded_source};
#[cfg(feature = "format-quantize")]
use aprender::format::RosettaStone;

// ============================================================================
// Phase 3: Tier B — Feature-Gated Handlers
// ============================================================================

/// Dispatch CLI output to JSON or human-readable text based on global flags.
fn emit_output(global: &super::super::args::Args, json_fn: impl FnOnce(), text_fn: impl FnOnce()) {
    if global.json {
        json_fn();
    } else if !global.quiet {
        text_fn();
    }
}

/// Sign a model file with Ed25519 (feature: `format-signing`)
pub(super) fn run_sign(
    args: &AprSignArgs,
    global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    #[cfg(not(feature = "format-signing"))]
    {
        let _ = (args, global);
        Err(CliError::InvalidArgument(
            "apr sign requires --features format-signing".to_string(),
        ))
    }

    #[cfg(feature = "format-signing")]
    {
        use ed25519_dalek::Signer;

        // Read the signing key file (raw 32-byte seed)
        let key_bytes = fs::read(&args.key)
            .map_err(|e| CliError::InvalidArgument(format!("Failed to read key file: {e}")))?;

        if key_bytes.len() < 32 {
            return Err(CliError::InvalidArgument(format!(
                "Key file too small: {} bytes (need 32)",
                key_bytes.len()
            )));
        }

        let seed: [u8; 32] = key_bytes[..32]
            .try_into()
            .map_err(|_| CliError::InvalidArgument("Invalid key data".to_string()))?;

        let signing_key = ed25519_dalek::SigningKey::from_bytes(&seed);

        // Read source model, sign, and write output
        let model_data = fs::read(&args.file)
            .map_err(|e| CliError::InvalidArgument(format!("Failed to read model: {e}")))?;

        // Compute Ed25519 signature over model content
        let signature = signing_key.sign(&model_data);
        let verifying_key = signing_key.verifying_key();

        // Write: original data + signature (64 bytes) + public key (32 bytes)
        let mut output = model_data;
        output.extend_from_slice(&signature.to_bytes());
        output.extend_from_slice(verifying_key.as_bytes());

        fs::write(&args.output, &output)
            .map_err(|e| CliError::InvalidArgument(format!("Failed to write signed model: {e}")))?;

        let out_display = args.output.display().to_string();
        let pk_hex = hex::encode(verifying_key.as_bytes());
        emit_output(
            global,
            || {
                println!("{{\"status\":\"signed\",\"output\":\"{out_display}\",\"pubkey_hex\":\"{pk_hex}\"}}");
            },
            || {
                println!("Signed: {out_display}");
                println!("Public key: {pk_hex}");
            },
        );

        Ok(CommandResult::success("Model signed"))
    }
}

/// Load a verifying key from a file or from embedded bytes
#[cfg(feature = "format-signing")]
fn load_verifying_key_from_file(
    pk_path: &std::path::Path,
) -> CliResult<ed25519_dalek::VerifyingKey> {
    let pk_bytes = fs::read(pk_path)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to read pubkey: {e}")))?;
    if pk_bytes.len() < 32 {
        return Err(CliError::InvalidArgument(
            "Pubkey file too small".to_string(),
        ));
    }
    let bytes: [u8; 32] = pk_bytes[..32]
        .try_into()
        .map_err(|_| CliError::InvalidArgument("Invalid public key length".to_string()))?;
    ed25519_dalek::VerifyingKey::from_bytes(&bytes)
        .map_err(|e| CliError::InvalidArgument(format!("Invalid public key: {e}")))
}

/// Load a verifying key from embedded bytes in model content
#[cfg(feature = "format-signing")]
fn load_verifying_key_embedded(
    content: &[u8],
    pubkey_start: usize,
) -> CliResult<ed25519_dalek::VerifyingKey> {
    let bytes: [u8; 32] = content
        .get(pubkey_start..)
        .ok_or_else(|| CliError::InvalidArgument("Public key offset out of bounds".to_string()))?
        .try_into()
        .map_err(|_| CliError::InvalidArgument("Invalid embedded public key length".to_string()))?;
    ed25519_dalek::VerifyingKey::from_bytes(&bytes)
        .map_err(|e| CliError::InvalidArgument(format!("Invalid embedded public key: {e}")))
}

/// Verify Ed25519 signature on a model file (feature: `format-signing`)
pub(super) fn run_verify_sig(
    args: &AprVerifySigArgs,
    global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    #[cfg(not(feature = "format-signing"))]
    {
        let _ = (args, global);
        Err(CliError::InvalidArgument(
            "apr verify-sig requires --features format-signing".to_string(),
        ))
    }

    #[cfg(feature = "format-signing")]
    {
        use ed25519_dalek::Verifier;

        let content = fs::read(&args.file)
            .map_err(|e| CliError::InvalidArgument(format!("Failed to read file: {e}")))?;

        // File layout: [model_data | signature(64) | pubkey(32)]
        if content.len() < 96 {
            return Err(CliError::InvalidArgument(
                "File too small to contain signature block (need 96+ bytes)".to_string(),
            ));
        }

        let sig_start = content.len() - 96;
        let pubkey_start = content.len() - 32;

        let sig_bytes: [u8; 64] = content[sig_start..pubkey_start]
            .try_into()
            .map_err(|_| CliError::InvalidArgument("Invalid signature".to_string()))?;
        let signature = ed25519_dalek::Signature::from_bytes(&sig_bytes);

        let verifying_key = match &args.pubkey {
            Some(pk_path) => load_verifying_key_from_file(pk_path)?,
            None => load_verifying_key_embedded(&content, pubkey_start)?,
        };

        let model_data = &content[..sig_start];
        let valid = verifying_key.verify(model_data, &signature).is_ok();

        emit_output(
            global,
            || println!("{{\"valid\":{valid}}}"),
            || println!("Signature {}", if valid { "VALID" } else { "INVALID" }),
        );

        if valid {
            Ok(CommandResult::success("Signature valid"))
        } else {
            Err(CliError::InvalidArgument(
                "Signature verification failed".to_string(),
            ))
        }
    }
}

/// Encrypt a model with AES-256-GCM (feature: `format-encryption`)
pub(super) fn run_encrypt(
    args: &AprEncryptArgs,
    global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    #[cfg(not(feature = "format-encryption"))]
    {
        let _ = (args, global);
        Err(CliError::InvalidArgument(
            "apr encrypt requires --features format-encryption".to_string(),
        ))
    }

    #[cfg(feature = "format-encryption")]
    {
        use aes_gcm::{
            aead::{Aead, KeyInit},
            Aes256Gcm, Nonce,
        };
        use argon2::Argon2;

        let password = require_password(args.password.as_deref())?;

        let model_data = fs::read(&args.file)
            .map_err(|e| CliError::InvalidArgument(format!("Failed to read model: {e}")))?;

        let mut salt = [0u8; 16];
        let mut nonce_bytes = [0u8; 12];
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut salt);
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut nonce_bytes);

        let mut key = [0u8; 32];
        Argon2::default()
            .hash_password_into(password.as_bytes(), &salt, &mut key)
            .map_err(|e| CliError::InvalidArgument(format!("Key derivation failed: {e}")))?;

        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|e| CliError::InvalidArgument(format!("Cipher init failed: {e}")))?;
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = cipher
            .encrypt(nonce, model_data.as_ref())
            .map_err(|e| CliError::InvalidArgument(format!("Encryption failed: {e}")))?;

        // Write: salt(16) + nonce(12) + ciphertext
        let mut output = Vec::with_capacity(16 + 12 + ciphertext.len());
        output.extend_from_slice(&salt);
        output.extend_from_slice(&nonce_bytes);
        output.extend_from_slice(&ciphertext);

        fs::write(&args.output, &output).map_err(|e| {
            CliError::InvalidArgument(format!("Failed to write encrypted model: {e}"))
        })?;

        let out_display = args.output.display().to_string();
        let out_len = output.len();
        emit_output(
            global,
            || {
                println!(
                    "{{\"status\":\"encrypted\",\"output\":\"{out_display}\",\"size\":{out_len}}}"
                );
            },
            || {
                println!("Encrypted: {out_display}");
                println!("Size: {out_len} bytes");
            },
        );

        Ok(CommandResult::success("Model encrypted"))
    }
}

/// Decrypt an AES-256-GCM encrypted model (feature: `format-encryption`)
pub(super) fn run_decrypt(
    args: &AprDecryptArgs,
    global: &super::super::args::Args,
) -> CliResult<CommandResult> {
    #[cfg(not(feature = "format-encryption"))]
    {
        let _ = (args, global);
        Err(CliError::InvalidArgument(
            "apr decrypt requires --features format-encryption".to_string(),
        ))
    }

    #[cfg(feature = "format-encryption")]
    {
        use aes_gcm::{
            aead::{Aead, KeyInit},
            Aes256Gcm, Nonce,
        };
        use argon2::Argon2;

        let password = require_password(args.password.as_deref())?;

        let content = fs::read(&args.file).map_err(|e| {
            CliError::InvalidArgument(format!("Failed to read encrypted file: {e}"))
        })?;

        if content.len() < 28 {
            return Err(CliError::InvalidArgument(
                "File too small to be encrypted (need salt+nonce+data)".to_string(),
            ));
        }

        // Parse: salt(16) + nonce(12) + ciphertext
        let salt: [u8; 16] = content
            .get(..16)
            .ok_or_else(|| CliError::InvalidArgument("Missing salt in encrypted file".to_string()))?
            .try_into()
            .map_err(|_| CliError::InvalidArgument("Invalid salt in encrypted file".to_string()))?;
        let nonce_bytes: [u8; 12] = content
            .get(16..28)
            .ok_or_else(|| {
                CliError::InvalidArgument("Missing nonce in encrypted file".to_string())
            })?
            .try_into()
            .map_err(|_| {
                CliError::InvalidArgument("Invalid nonce in encrypted file".to_string())
            })?;
        let ciphertext = content.get(28..).ok_or_else(|| {
            CliError::InvalidArgument("Encrypted file too short for ciphertext".to_string())
        })?;

        let mut key = [0u8; 32];
        Argon2::default()
            .hash_password_into(password.as_bytes(), &salt, &mut key)
            .map_err(|e| CliError::InvalidArgument(format!("Key derivation failed: {e}")))?;

        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|e| CliError::InvalidArgument(format!("Cipher init failed: {e}")))?;
        let nonce = Nonce::from_slice(&nonce_bytes);
        let plaintext = cipher.decrypt(nonce, ciphertext).map_err(|_| {
            CliError::InvalidArgument(
                "Decryption failed (wrong password or corrupted data)".to_string(),
            )
        })?;

        fs::write(&args.output, &plaintext).map_err(|e| {
            CliError::InvalidArgument(format!("Failed to write decrypted model: {e}"))
        })?;

        let out_display = args.output.display().to_string();
        let pt_len = plaintext.len();
        emit_output(
            global,
            || {
                println!(
                    "{{\"status\":\"decrypted\",\"output\":\"{out_display}\",\"size\":{pt_len}}}"
                );
            },
            || {
                println!("Decrypted: {out_display}");
                println!("Size: {pt_len} bytes");
            },
        );

        Ok(CommandResult::success("Model decrypted"))
    }
}

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
pub(super) fn run_quantize(
    args: &AprQuantizeArgs,
    global: &super::super::args::Args,
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
pub(super) fn run_import_sharded(
    args: &AprImportShardedArgs,
    global: &super::super::args::Args,
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
pub(super) fn run_he_inspect(
    args: &AprHeInspectArgs,
    global: &super::super::args::Args,
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

// ============================================================================
// Tier C — Profiling (renacer integration + BrickProfiler)
// ============================================================================

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
fn run_sweep_threads(
    sweep_str: &str,
    whisper: &crate::WhisperApr,
    samples: &[f32],
    audio_duration_s: f64,
    args: &AprProfileArgs,
    global: &super::super::args::Args,
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

    struct SweepResult {
        threads: u32,
        enc_ms: f64,
        dec_ms: f64,
        total_ms: f64,
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
        let mut enc_total = 0.0;
        let mut dec_total = 0.0;
        let mut total_total = 0.0;
        for _ in 0..args.runs {
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
    let base_enc = baseline.enc_ms;
    let base_dec = baseline.dec_ms;

    if args.format == "json" {
        // JSON output
        let entries: Vec<String> = results
            .iter()
            .map(|r| {
                let speedup = base_total / r.total_ms;
                let eff = speedup / r.threads as f64 * 100.0;
                // Amdahl: s = (1/speedup - 1/N) / (1 - 1/N)
                let n = r.threads as f64;
                let serial = if n > 1.0 {
                    (1.0 / speedup - 1.0 / n) / (1.0 - 1.0 / n)
                } else {
                    0.0
                };
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
        let json = format!(
            "{{\"sweep\":{{\"audio_duration_s\":{:.3},\"hw_cores\":{},\"hw_simd\":\"{}\",\"results\":[{}]}}}}",
            audio_duration_s,
            hw.cpu.cores,
            format!("{:?}", hw.cpu.simd),
            entries.join(","),
        );
        if let Some(ref out) = args.output {
            fs::write(out, &json).map_err(|e| CliError::InvalidArgument(format!("Write: {e}")))?;
        } else {
            println!("{json}");
        }
    } else {
        // Text table
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
                for r in &results {
                    let speedup = base_total / r.total_ms;
                    let eff = speedup / r.threads as f64 * 100.0;
                    let n = r.threads as f64;
                    let serial = if n > 1.0 {
                        (1.0 / speedup - 1.0 / n) / (1.0 - 1.0 / n)
                    } else {
                        0.0
                    };
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
                // Per-component scaling
                println!();
                let last = results.last().unwrap();
                println!(
                    "  Encoder: {:.2}x ({}→{} threads)",
                    base_enc / last.enc_ms,
                    baseline.threads,
                    last.threads,
                );
                println!(
                    "  Decoder: {:.2}x ({}→{} threads)",
                    base_dec / last.dec_ms.max(0.001),
                    baseline.threads,
                    last.threads,
                );
            },
        );
    }

    Ok(CommandResult::success("Thread sweep complete"))
}

/// Run instrumented transcription with per-step timing breakdown.
///
/// Uses `TranscribeOptions { profile: true }` so mel, encoder, and decoder
/// timings come from direct instrumentation inside `transcribe_single_chunk`
/// rather than approximate subtraction. Outputs text, JSON, or renacer
/// (Chrome Trace Event) format.
pub(super) fn run_profile(
    args: &AprProfileArgs,
    global: &super::super::args::Args,
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
    let samples = super::super::commands::load_audio_samples(args.audio.as_path(), &audio_bytes)?;
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
        let brick_detail = if let Some(ref prof) = result.profiling {
            let norm = prof.breakdown.get("brick_norm_ms").copied().unwrap_or(0.0);
            let attn = prof.breakdown.get("brick_attn_ms").copied().unwrap_or(0.0);
            let ffn = prof.breakdown.get("brick_ffn_ms").copied().unwrap_or(0.0);
            let other = prof.breakdown.get("brick_other_ms").copied().unwrap_or(0.0);
            let pf_minor = prof
                .breakdown
                .get("page_faults_minor")
                .copied()
                .unwrap_or(0.0) as u64;
            let pf_major = prof
                .breakdown
                .get("page_faults_major")
                .copied()
                .unwrap_or(0.0) as u64;
            // Per-brick bottleneck diagnosis
            let ln_bottleneck = prof
                .breakdown
                .get("brick_LayerNorm_bottleneck")
                .copied()
                .unwrap_or(0.0) as u8;
            let attn_bottleneck = prof
                .breakdown
                .get("brick_AttentionScore_bottleneck")
                .copied()
                .unwrap_or(0.0) as u8;
            let ffn_bottleneck = prof
                .breakdown
                .get("brick_GateProjection_bottleneck")
                .copied()
                .unwrap_or(0.0) as u8;
            let ln_cpe = prof
                .breakdown
                .get("brick_LayerNorm_cycles_per_elem")
                .copied()
                .unwrap_or(0.0);
            let attn_cpe = prof
                .breakdown
                .get("brick_AttentionScore_cycles_per_elem")
                .copied()
                .unwrap_or(0.0);
            let ffn_cpe = prof
                .breakdown
                .get("brick_GateProjection_cycles_per_elem")
                .copied()
                .unwrap_or(0.0);
            // WAPR-PROFILE-001 Gap 4: Extract BLIS GEMM hierarchy stats
            let blis_total_gflops = prof
                .breakdown
                .get("blis_total_gflops")
                .copied()
                .unwrap_or(0.0);
            let blis_macro_gflops = prof
                .breakdown
                .get("blis_macro_gflops")
                .copied()
                .unwrap_or(0.0);
            let blis_micro_gflops = prof
                .breakdown
                .get("blis_micro_gflops")
                .copied()
                .unwrap_or(0.0);
            let blis_pack_pct = prof.breakdown.get("blis_pack_pct").copied().unwrap_or(0.0);
            let blis_macro_calls = prof
                .breakdown
                .get("blis_macro_calls")
                .copied()
                .unwrap_or(0.0) as u64;
            // Gap 2: Roofline classification from BLIS achieved GFLOP/s vs hardware peak
            let roofline_util_pct = if hw.cpu.peak_gflops > 0.0 && blis_total_gflops > 0.0 {
                blis_total_gflops / hw.cpu.peak_gflops * 100.0
            } else {
                0.0
            };
            let roofline_bound = if blis_total_gflops > 0.0 {
                // Encoder GEMM AI is typically >>8 F/B (compute-bound region)
                if roofline_util_pct > 50.0 {
                    "compute (efficient)"
                } else if roofline_util_pct > 10.0 {
                    "compute (low util)"
                } else {
                    "memory"
                }
            } else {
                "unknown"
            };
            if norm > 0.0 || attn > 0.0 || ffn > 0.0 {
                Some(BrickDetail {
                    norm_ms: norm,
                    attn_ms: attn,
                    ffn_ms: ffn,
                    other_ms: other,
                    page_faults_minor: pf_minor,
                    page_faults_major: pf_major,
                    ln_bottleneck,
                    attn_bottleneck,
                    ffn_bottleneck,
                    ln_cycles_per_elem: ln_cpe,
                    attn_cycles_per_elem: attn_cpe,
                    ffn_cycles_per_elem: ffn_cpe,
                    blis_total_gflops,
                    blis_macro_gflops,
                    blis_micro_gflops,
                    blis_pack_pct,
                    blis_macro_calls,
                    roofline_bound,
                    roofline_util_pct,
                })
            } else {
                None
            }
        } else {
            None
        };

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
        avg_brick_detail: {
            let all_have = run_results.iter().all(|r| r.brick_detail.is_some());
            if all_have && !run_results.is_empty() {
                Some(AvgBrickDetail {
                    norm_ms: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().norm_ms)
                        .sum::<f64>()
                        / n,
                    attn_ms: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().attn_ms)
                        .sum::<f64>()
                        / n,
                    ffn_ms: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().ffn_ms)
                        .sum::<f64>()
                        / n,
                    other_ms: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().other_ms)
                        .sum::<f64>()
                        / n,
                    page_faults_minor: run_results
                        .last()
                        .unwrap()
                        .brick_detail
                        .as_ref()
                        .unwrap()
                        .page_faults_minor,
                    page_faults_major: run_results
                        .last()
                        .unwrap()
                        .brick_detail
                        .as_ref()
                        .unwrap()
                        .page_faults_major,
                    ln_bottleneck: run_results
                        .last()
                        .unwrap()
                        .brick_detail
                        .as_ref()
                        .unwrap()
                        .ln_bottleneck,
                    attn_bottleneck: run_results
                        .last()
                        .unwrap()
                        .brick_detail
                        .as_ref()
                        .unwrap()
                        .attn_bottleneck,
                    ffn_bottleneck: run_results
                        .last()
                        .unwrap()
                        .brick_detail
                        .as_ref()
                        .unwrap()
                        .ffn_bottleneck,
                    ln_cycles_per_elem: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().ln_cycles_per_elem)
                        .sum::<f64>()
                        / n,
                    attn_cycles_per_elem: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().attn_cycles_per_elem)
                        .sum::<f64>()
                        / n,
                    ffn_cycles_per_elem: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().ffn_cycles_per_elem)
                        .sum::<f64>()
                        / n,
                    blis_total_gflops: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().blis_total_gflops)
                        .sum::<f64>()
                        / n,
                    blis_macro_gflops: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().blis_macro_gflops)
                        .sum::<f64>()
                        / n,
                    blis_micro_gflops: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().blis_micro_gflops)
                        .sum::<f64>()
                        / n,
                    blis_pack_pct: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().blis_pack_pct)
                        .sum::<f64>()
                        / n,
                    blis_macro_calls: run_results
                        .last()
                        .unwrap()
                        .brick_detail
                        .as_ref()
                        .unwrap()
                        .blis_macro_calls,
                    roofline_bound: run_results
                        .last()
                        .unwrap()
                        .brick_detail
                        .as_ref()
                        .unwrap()
                        .roofline_bound,
                    roofline_util_pct: run_results
                        .iter()
                        .map(|r| r.brick_detail.as_ref().unwrap().roofline_util_pct)
                        .sum::<f64>()
                        / n,
                })
            } else {
                None
            }
        },
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

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
// Tier C — Profiling (renacer integration)
// ============================================================================

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

    // Load audio
    let audio_bytes =
        fs::read(&args.audio).map_err(|e| CliError::InvalidArgument(format!("Audio: {e}")))?;
    let samples = super::super::commands::load_audio_samples(args.audio.as_path(), &audio_bytes)?;
    let audio_duration_s = samples.len() as f64 / 16000.0;

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
        let (mel_ms, enc_ms, dec_ms, total_ms, conv_ms, blocks_ms) =
            if let Some(ref prof) = result.profiling {
                let mel = prof.breakdown.get("mel_ms").copied().unwrap_or(0.0);
                let enc = prof.breakdown.get("encoder_ms").copied().unwrap_or(0.0);
                let dec = prof.breakdown.get("decoder_ms").copied().unwrap_or(0.0);
                let conv = prof.breakdown.get("conv_frontend_ms").copied();
                let blocks = prof.breakdown.get("encoder_blocks_ms").copied();
                (mel, enc, dec, prof.total_ms, conv, blocks)
            } else {
                (0.0, 0.0, 0.0, wall_ms, None, None)
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
                conv_frontend_ms: conv_ms,
                encoder_blocks_ms: blocks_ms,
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
        avg_conv_frontend: if run_results.iter().all(|r| r.conv_frontend_ms.is_some()) {
            Some(
                run_results
                    .iter()
                    .map(|r| r.conv_frontend_ms.unwrap_or(0.0))
                    .sum::<f64>()
                    / n,
            )
        } else {
            None
        },
        avg_encoder_blocks: if run_results.iter().all(|r| r.encoder_blocks_ms.is_some()) {
            Some(
                run_results
                    .iter()
                    .map(|r| r.encoder_blocks_ms.unwrap_or(0.0))
                    .sum::<f64>()
                    / n,
            )
        } else {
            None
        },
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

/// Timing data for a single profiling run
struct ProfileRun {
    mel_ms: f64,
    encode_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    rtf: f64,
    token_count: usize,
    text: String,
    /// Conv frontend time (subset of encode_ms), if available
    conv_frontend_ms: Option<f64>,
    /// Encoder blocks time (encode_ms minus conv_frontend), if available
    encoder_blocks_ms: Option<f64>,
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
    /// Avg conv frontend time (subset of encoder)
    avg_conv_frontend: Option<f64>,
    /// Avg encoder blocks time (encoder minus conv frontend)
    avg_encoder_blocks: Option<f64>,
}

impl ProfileSummary<'_> {
    fn format_json(&self, args: &AprProfileArgs) -> String {
        let mut encoder_detail = String::new();
        if let (Some(conv), Some(blocks)) = (self.avg_conv_frontend, self.avg_encoder_blocks) {
            encoder_detail = format!(
                ",\"encoder_detail\":{{\"conv_frontend\":{:.1},\"blocks\":{:.1}}}",
                conv, blocks
            );
        }
        format!(
            concat!(
                "{{\"model\":\"{}\",\"audio\":\"{}\",\"audio_duration_s\":{:.3},",
                "\"warmup\":{},\"runs\":{},",
                "\"avg_ms\":{{\"load\":{:.1},\"mel\":{:.1},\"encode\":{:.1},",
                "\"decode\":{:.1},\"total\":{:.1}}}{},",
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
            encoder_detail,
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
        // Encoder sub-steps (nested on tid 2)
        if let (Some(conv), Some(blocks)) = (self.avg_conv_frontend, self.avg_encoder_blocks) {
            let conv_dur = conv * 1000.0;
            events.push(format!(
                concat!(
                    "{{\"name\":\"conv_frontend\",\"cat\":\"apr_profile\",\"ph\":\"X\",",
                    "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":2}}"
                ),
                ts_us, conv_dur
            ));
            let blocks_dur = blocks * 1000.0;
            events.push(format!(
                concat!(
                    "{{\"name\":\"encoder_blocks\",\"cat\":\"apr_profile\",\"ph\":\"X\",",
                    "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":2}}"
                ),
                ts_us + conv_dur,
                blocks_dur
            ));
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
        if let (Some(conv), Some(blocks)) = (self.avg_conv_frontend, self.avg_encoder_blocks) {
            println!(
                "    Conv frontend {:>6.1}    {:>5.1}%",
                conv,
                conv / self.avg_total * 100.0
            );
            println!(
                "    Blocks       {:>7.1}    {:>5.1}%",
                blocks,
                blocks / self.avg_total * 100.0
            );
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

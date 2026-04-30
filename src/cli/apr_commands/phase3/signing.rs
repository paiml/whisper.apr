// `cli` is now in default features (#55); items below are reachable only under
// the converter / phase3-encryption feature combos and lint as dead code
// when only `cli` is on. This is pre-existing technical debt — file follow-up.
#![allow(dead_code)]

//! Ed25519 signing and verification handlers (feature: `format-signing`)


use super::super::super::apr_args::{AprSignArgs, AprVerifySigArgs};
use super::super::super::commands::{CliError, CliResult, CommandResult};

/// Sign a model file with Ed25519 (feature: `format-signing`)
pub(in super::super) fn run_sign(
    args: &AprSignArgs,
    global: &super::super::super::args::Args,
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
pub(in super::super) fn run_verify_sig(
    args: &AprVerifySigArgs,
    global: &super::super::super::args::Args,
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

// `cli` is now in default features (#55); items below are reachable only under
// the converter / phase3-encryption feature combos and lint as dead code
// when only `cli` is on. This is pre-existing technical debt — file follow-up.
#![allow(dead_code)]

//! AES-256-GCM encrypt / decrypt handlers (feature: `format-encryption`)

use super::super::super::apr_args::{AprDecryptArgs, AprEncryptArgs};
use super::super::super::commands::{CliError, CliResult, CommandResult};
#[cfg(feature = "format-encryption")]
use super::super::require_password;
#[cfg(feature = "format-encryption")]
use super::emit_output;
#[cfg(feature = "format-encryption")]
use std::fs;

/// Encrypt a model with AES-256-GCM (feature: `format-encryption`)
pub(in super::super) fn run_encrypt(
    args: &AprEncryptArgs,
    global: &super::super::super::args::Args,
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
pub(in super::super) fn run_decrypt(
    args: &AprDecryptArgs,
    global: &super::super::super::args::Args,
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

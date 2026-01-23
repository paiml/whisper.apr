//! HuggingFace Publish Script (WAPR-PUB-001)
//!
//! Bashrs source for automated model publishing workflow.
//! Transpile with: `bashrs build scripts/publish.rs -o scripts/publish.sh`
//!
//! # Usage
//!
//! ```bash
//! ./scripts/publish.sh model.apr paiml/whisper-apr-tiny
//! ./scripts/publish.sh model.apr paiml/whisper-apr-tiny --format both
//! ./scripts/publish.sh model.apr paiml/whisper-apr-tiny --dry-run
//! ```

use std::env;
use std::path::Path;
use std::process::{Command, ExitCode};

/// Print usage information
fn print_usage() {
    eprintln!(
        r#"
Usage: publish.sh <model.apr> <repo-id> [OPTIONS]

Arguments:
    <model.apr>     Path to APR model file
    <repo-id>       HuggingFace repository ID (e.g., paiml/whisper-apr-tiny)

Options:
    --format <fmt>  Output format: apr, safetensors, both (default: both)
    --dry-run       Verify without uploading
    --skip-verify   Skip pre-publish verification
    --message <msg> Custom commit message
    --help          Show this help

Environment:
    HF_TOKEN        HuggingFace API token (required for upload)

Examples:
    # Publish to HuggingFace with both formats
    ./scripts/publish.sh whisper-tiny.apr paiml/whisper-apr-tiny

    # Dry run (verify only)
    ./scripts/publish.sh whisper-tiny.apr paiml/whisper-apr-tiny --dry-run

    # SafeTensors only
    ./scripts/publish.sh whisper-tiny.apr paiml/whisper-apr-tiny --format safetensors
"#
    );
}

/// Check if a command exists
fn command_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run a command and check success
fn run_cmd(cmd: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(cmd)
        .args(args)
        .status()
        .map_err(|e| format!("Failed to run {}: {}", cmd, e))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("{} failed with exit code {:?}", cmd, status.code()))
    }
}

/// Main entry point
fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    if args.len() < 3 || args.contains(&"--help".to_string()) {
        print_usage();
        return if args.contains(&"--help".to_string()) {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }

    let model_path = &args[1];
    let repo_id = &args[2];

    // Parse options
    let mut format = "both";
    let mut dry_run = false;
    let mut skip_verify = false;
    let mut message = format!("Upload {} via whisper.apr publish", model_path);

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--format" => {
                i += 1;
                if i < args.len() {
                    format = match args[i].as_str() {
                        "apr" | "safetensors" | "both" => args[i].as_str(),
                        _ => {
                            eprintln!("Error: Invalid format '{}'. Use: apr, safetensors, both", args[i]);
                            return ExitCode::FAILURE;
                        }
                    };
                }
            }
            "--dry-run" => dry_run = true,
            "--skip-verify" => skip_verify = true,
            "--message" => {
                i += 1;
                if i < args.len() {
                    message = args[i].clone();
                }
            }
            _ => {
                eprintln!("Warning: Unknown option '{}'", args[i]);
            }
        }
        i += 1;
    }

    println!("=== whisper.apr Publish Workflow ===\n");
    println!("Model:    {}", model_path);
    println!("Repo:     {}", repo_id);
    println!("Format:   {}", format);
    println!("Dry-run:  {}", dry_run);
    println!();

    // Step 1: Verify model file exists
    println!("[1/6] Checking model file...");
    if !Path::new(model_path).exists() {
        eprintln!("Error: Model file not found: {}", model_path);
        return ExitCode::FAILURE;
    }
    println!("      ✓ Model file exists");

    // Step 2: Check HF_TOKEN
    println!("[2/6] Checking authentication...");
    let has_token = env::var("HF_TOKEN").is_ok();
    if !has_token && !dry_run {
        eprintln!("Error: HF_TOKEN environment variable not set");
        eprintln!("       Set it with: export HF_TOKEN=hf_...");
        return ExitCode::FAILURE;
    }
    if has_token {
        println!("      ✓ HF_TOKEN is set");
    } else {
        println!("      ⚠ HF_TOKEN not set (dry-run mode)");
    }

    // Step 3: Verify APR format
    if !skip_verify {
        println!("[3/6] Verifying APR format...");
        if command_exists("whisper-apr") {
            if let Err(e) = run_cmd("whisper-apr", &["verify", model_path]) {
                eprintln!("Error: APR verification failed: {}", e);
                return ExitCode::FAILURE;
            }
            println!("      ✓ APR format valid");
        } else {
            // Fallback: check magic bytes manually
            match std::fs::read(model_path) {
                Ok(data) if data.len() >= 4 && &data[0..4] == b"APR\0" => {
                    println!("      ✓ APR magic bytes valid");
                }
                Ok(_) => {
                    eprintln!("Error: Invalid APR magic bytes");
                    return ExitCode::FAILURE;
                }
                Err(e) => {
                    eprintln!("Error: Cannot read model file: {}", e);
                    return ExitCode::FAILURE;
                }
            }
        }
    } else {
        println!("[3/6] Skipping verification (--skip-verify)");
    }

    // Step 4: Export to SafeTensors if needed
    let safetensors_path = format!("{}.safetensors", model_path.trim_end_matches(".apr"));
    if format == "safetensors" || format == "both" {
        println!("[4/6] Exporting to SafeTensors...");
        if command_exists("whisper-apr") {
            if let Err(e) = run_cmd(
                "whisper-apr",
                &["export", "--format", "safetensors", model_path, "-o", &safetensors_path],
            ) {
                eprintln!("Error: SafeTensors export failed: {}", e);
                return ExitCode::FAILURE;
            }
            println!("      ✓ Exported to {}", safetensors_path);
        } else {
            println!("      ⚠ whisper-apr CLI not found, skipping export");
            println!("        Install with: cargo install --path .");
        }
    } else {
        println!("[4/6] Skipping SafeTensors export (format=apr)");
    }

    // Step 5: Sign models (if pacha available)
    println!("[5/6] Signing models...");
    if command_exists("batuta") {
        if let Err(e) = run_cmd("batuta", &["pacha", "sign", model_path]) {
            println!("      ⚠ Signing skipped: {}", e);
        } else {
            println!("      ✓ Model signed");
        }
    } else {
        println!("      ⚠ batuta not found, skipping signing");
    }

    // Step 6: Upload to HuggingFace
    println!("[6/6] Uploading to HuggingFace...");
    if dry_run {
        println!("      ⚠ Dry-run mode, skipping upload");
        println!("\n      Would upload:");
        if format == "apr" || format == "both" {
            println!("        - {}", model_path);
        }
        if format == "safetensors" || format == "both" {
            println!("        - {}", safetensors_path);
        }
        println!("      To: https://huggingface.co/{}", repo_id);
    } else {
        // Use batuta hf push if available, otherwise huggingface-cli
        if command_exists("batuta") {
            let mut upload_args = vec!["hf", "push", "model", model_path, "--repo", repo_id];
            if format != "apr" {
                upload_args.extend(&["--formats", format]);
            }
            upload_args.extend(&["--commit-message", &message]);

            if let Err(e) = run_cmd("batuta", &upload_args) {
                eprintln!("Error: Upload failed: {}", e);
                return ExitCode::FAILURE;
            }
        } else if command_exists("huggingface-cli") {
            // Fallback to huggingface-cli (if installed)
            let files_to_upload: Vec<&str> = match format {
                "apr" => vec![model_path],
                "safetensors" => vec![&safetensors_path],
                _ => vec![model_path, &safetensors_path],
            };

            for file in files_to_upload {
                if let Err(e) = run_cmd(
                    "huggingface-cli",
                    &["upload", repo_id, file, "--commit-message", &message],
                ) {
                    eprintln!("Error: Upload failed for {}: {}", file, e);
                    return ExitCode::FAILURE;
                }
            }
        } else {
            eprintln!("Error: No upload tool found");
            eprintln!("       Install batuta or huggingface-cli");
            return ExitCode::FAILURE;
        }
        println!("      ✓ Uploaded to https://huggingface.co/{}", repo_id);
    }

    println!("\n=== Publish Complete ===");
    if !dry_run {
        println!("View at: https://huggingface.co/{}", repo_id);
    }

    ExitCode::SUCCESS
}

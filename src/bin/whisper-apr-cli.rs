//! whisper-apr CLI: Native command-line interface for whisper.apr
//!
//! This is a thin shell that delegates to library functions.
//! All logic lives in `whisper_apr::cli` for testability.
//!
//! # Design (Toyota Way - Jidoka)
//!
//! - Stop on errors with clear, actionable feedback
//! - All logic is testable in the library
//! - Binary is minimal, just parsing and dispatch
//!
//! See docs/specifications/cli-tui-spec.md for full specification.

use clap::Parser;
use whisper_apr::cli::{run, Args};

fn main() {
    // Initialize tracing if enabled
    #[cfg(feature = "tracing")]
    init_tracing();

    // Parse arguments
    let args = Args::parse();

    // Run CLI and handle errors with Jidoka principle
    match run(args) {
        Ok(result) => {
            if !result.success {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "tracing")]
fn init_tracing() {
    use tracing_subscriber::prelude::*;
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();
}

//! CLI module for whisper-apr
//!
//! This module provides the command-line interface implementation.
//! All logic is testable and separate from the binary entry point.
//!
//! # Design Principles (Toyota Way)
//!
//! - **Jidoka**: Stop on errors with clear, actionable feedback
//! - **Genchi Genbutsu**: Provide visibility into the transcription process
//! - **Kaizen**: Incremental, test-driven development
//!
//! # Architecture
//!
//! All logic lives in this module for testability. The binary (`src/bin/whisper-apr.rs`)
//! is a thin shell that only calls `cli::run()`.
//!
//! ```text
//! src/cli/
//! ├── mod.rs       # This file - module exports
//! ├── args.rs      # Argument parsing with clap
//! ├── commands.rs  # Command implementations
//! └── output.rs    # Output formatters (txt, srt, vtt, json)
//! ```

pub mod args;
pub mod commands;
pub mod output;

pub use args::{Args, Command};
pub use commands::{run, CliError, CliResult};
pub use output::OutputFormat;

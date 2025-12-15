//! CLI Usage Example for whisper-apr
//!
//! This example demonstrates how to use the CLI module programmatically
//! for testing or embedding in other applications.
//!
//! # Running the CLI binary
//!
//! ```bash
//! # Build and run CLI
//! cargo run --features cli --bin whisper-apr-cli -- --help
//!
//! # Transcribe audio
//! cargo run --features cli --bin whisper-apr-cli -- transcribe audio.wav
//!
//! # Run backend tests
//! cargo run --features cli --bin whisper-apr-cli -- test --backend all
//!
//! # List models
//! cargo run --features cli --bin whisper-apr-cli -- model list
//!
//! # Benchmark
//! cargo run --features cli --bin whisper-apr-cli -- benchmark tiny --iterations 3
//! ```
//!
//! # Using CLI module in code
//!
//! The CLI module is designed with all logic in the library for testability.
//! You can use it programmatically:
//!
//! ```rust,ignore
//! use whisper_apr::cli::{Args, Command, run};
//! use clap::Parser;
//!
//! let args = Args::try_parse_from(["whisper-apr", "model", "list"])?;
//! let result = run(args)?;
//! println!("Success: {}", result.success);
//! ```

#[cfg(feature = "cli")]
fn main() {
    use clap::Parser;
    use whisper_apr::cli::{run, Args};

    println!("=== whisper-apr CLI Usage Example ===\n");

    // Example 1: List models
    println!("1. Listing available models:");
    let args = Args::try_parse_from(["whisper-apr", "model", "list"])
        .expect("Failed to parse model list command");
    match run(args) {
        Ok(result) => println!("   Result: {}\n", result.message),
        Err(e) => println!("   Error: {e}\n"),
    }

    // Example 2: Test SIMD backend
    println!("2. Testing SIMD backend:");
    let args = Args::try_parse_from(["whisper-apr", "test", "--backend", "simd", "-q"])
        .expect("Failed to parse test command");
    match run(args) {
        Ok(result) => {
            println!("   Success: {}", result.success);
            println!("   Message: {}\n", result.message);
        }
        Err(e) => println!("   Error: {e}\n"),
    }

    // Example 3: Run benchmark (1 iteration for speed)
    println!("3. Running benchmark (1 iteration):");
    let args = Args::try_parse_from([
        "whisper-apr",
        "benchmark",
        "tiny",
        "--iterations",
        "1",
        "-q",
    ])
    .expect("Failed to parse benchmark command");
    match run(args) {
        Ok(result) => {
            println!("   Success: {}", result.success);
            if let Some(rtf) = result.rtf {
                println!("   RTF: {rtf:.2}x");
            }
            println!();
        }
        Err(e) => println!("   Error: {e}\n"),
    }

    // Example 4: Show CLI help
    println!("4. CLI Help:");
    println!("   Run: cargo run --features cli --bin whisper-apr-cli -- --help");
    println!();

    println!("=== Example Complete ===");
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("This example requires the 'cli' feature.");
    eprintln!("Run with: cargo run --features cli --example cli_usage");
}

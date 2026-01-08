//! Generate index.html from Brick definitions
//!
//! This binary ensures zero hand-written HTML by generating
//! the entire page from Rust Brick types.
//!
//! Usage:
//!   cargo run --bin gen-index
//!
//! Output:
//!   Creates index.html in the current directory

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use whisper_apr_demo::bricks::{generate_index_html, HtmlConfig};

fn main() {
    // Determine output path (same directory as Cargo.toml)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    let output_path = PathBuf::from(&manifest_dir).join("index.html");

    // Generate HTML from Brick definitions
    let config = HtmlConfig::default();
    let html = generate_index_html(&config);

    // Write to file
    let mut file = fs::File::create(&output_path)
        .expect("Failed to create index.html");
    file.write_all(html.as_bytes())
        .expect("Failed to write index.html");

    println!("Generated: {}", output_path.display());
    println!("Content length: {} bytes", html.len());
}

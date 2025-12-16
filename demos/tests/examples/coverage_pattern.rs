//! Coverage Pattern Example
//!
//! This example demonstrates the unified coverage pattern used in whisper.apr demos,
//! following the probar/bashrs methodology.
//!
//! ## Quick Start
//!
//! ```bash
//! # Run all coverage
//! make coverage
//!
//! # View HTML report
//! make coverage-open
//! ```
//!
//! ## Coverage Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED COVERAGE SYSTEM                       │
//! ├──────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
//! │  │  Rust Coverage  │  │  GUI Coverage   │  │ Pixel Coverage  │  │
//! │  │  (llvm-cov)     │  │  (UxTracker)    │  │ (SSIM/PSNR)     │  │
//! │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
//! │           │                    │                    │           │
//! │           └──────────┬─────────┴──────────┬─────────┘           │
//! │                      │                    │                     │
//! │                      ▼                    ▼                     │
//! │           ┌──────────────────────────────────────────┐          │
//! │           │         cargo llvm-cov nextest           │          │
//! │           │      (unified instrumentation)           │          │
//! │           └──────────────────────────────────────────┘          │
//! │                                                                  │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## The Pattern (from probar/bashrs)
//!
//! 1. **Single Entry Point**: `make coverage`
//! 2. **Nextest for Parallel Tests**: `cargo llvm-cov --no-report nextest`
//! 3. **Report Generation**: `cargo llvm-cov report --html`
//! 4. **Mold Linker Workaround**: Temporarily disable `~/.cargo/config.toml`
//!
//! ## Coverage Tiers
//!
//! | Tier | When | Time | What |
//! |------|------|------|------|
//! | 1 | On save | <1s | check, fmt, clippy |
//! | 2 | Pre-commit | <5s | unit tests |
//! | 3 | Pre-push | 1-5min | full tests + coverage |
//! | 4 | CI/CD | 5-60min | mutation + quality gates |
//!
//! ## Usage in Tests
//!
//! ```rust,ignore
//! use probar::ux_coverage::UxCoverageTracker;
//! use probar::gui_coverage;
//!
//! fn create_coverage_tracker() -> UxCoverageTracker {
//!     gui_coverage! {
//!         buttons: ["start", "stop", "clear"],
//!         inputs: ["file_input"],
//!         screens: ["main", "recording"]
//!     }
//! }
//!
//! #[test]
//! fn test_button_coverage() {
//!     let mut gui = create_coverage_tracker();
//!     gui.click("start");
//!     gui.click("stop");
//!     gui.click("clear");
//!     assert!(gui.meets(100.0)); // 100% button coverage
//! }
//! ```

use probar::gui_coverage;
use probar::ux_coverage::UxCoverageTracker;

/// Example: Create a coverage tracker for a demo
fn example_coverage_tracker() -> UxCoverageTracker {
    gui_coverage! {
        buttons: [
            "start_recording",
            "stop_recording",
            "clear_transcript"
        ],
        inputs: [],
        screens: [
            "main_view",
            "recording_view"
        ]
    }
}

/// Example: Simulate full GUI coverage
fn example_full_coverage() {
    let mut gui = example_coverage_tracker();

    // Cover all buttons
    gui.click("start_recording");
    gui.click("stop_recording");
    gui.click("clear_transcript");

    // Cover all screens
    gui.visit("main_view");
    gui.visit("recording_view");

    // Verify coverage
    println!("Coverage Summary: {}", gui.summary());
    assert!(gui.is_complete(), "Should have 100% coverage");
}

fn main() {
    println!("whisper.apr Coverage Pattern Example");
    println!("====================================\n");

    println!("1. Running Rust Coverage:");
    println!("   $ make coverage\n");

    println!("2. Coverage Breakdown:");
    println!("   - Rust Code:  via llvm-cov instrumentation");
    println!("   - GUI Tests:  via probar UxCoverageTracker");
    println!("   - Pixel Tests: via SSIM/PSNR/CIEDE2000\n");

    println!("3. Example GUI Coverage Tracker:\n");
    example_full_coverage();

    println!("\n4. Key Commands:");
    println!("   make coverage          # Full coverage report");
    println!("   make coverage-summary  # Quick summary");
    println!("   make coverage-open     # Open HTML report");
    println!("   make tier3             # Pre-push with coverage");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_tracker_creation() {
        let gui = example_coverage_tracker();
        assert!(!gui.is_complete());
    }

    #[test]
    fn test_full_coverage_simulation() {
        let mut gui = example_coverage_tracker();

        gui.click("start_recording");
        gui.click("stop_recording");
        gui.click("clear_transcript");
        gui.visit("main_view");
        gui.visit("recording_view");

        assert!(gui.is_complete());
        assert!(gui.meets(95.0));
    }

    #[test]
    fn test_partial_coverage() {
        let mut gui = example_coverage_tracker();

        // Only cover some buttons
        gui.click("start_recording");

        assert!(!gui.is_complete());
        assert!(!gui.meets(95.0));
    }
}

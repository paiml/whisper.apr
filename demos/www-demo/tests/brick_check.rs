//! Brick Specification for Whisper Demo
//!
//! Validates availability of the new 'Brick' architecture in the probar dependency.

#![allow(unused)]

// Verify jugar_probar exports (the actual crate name)
use jugar_probar::brick;

// Re-export test to verify brick architecture is available
fn verify_brick_types() {
    // These types should be available from jugar_probar::brick
    use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

    // Verify the demo bricks compile
    use whisper_apr_demo::bricks::{
        StatusBrick, TranscriptionBrick, VuMeterBrick, WaveformBrick
    };

    let status = StatusBrick::new();
    let _name = status.brick_name();
    let _assertions = status.assertions();

    println!("Brick architecture verified");
}

fn main() {
    verify_brick_types();
}

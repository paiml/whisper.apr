//! Ground Truth 3-Column Comparison
//!
//! Compares: whisper.apr vs whisper.cpp vs HuggingFace
//!
//! ## Usage
//!
//! ```bash
//! bashrs build scripts/ground_truth_compare.rs -o scripts/ground_truth_compare.sh
//! ./scripts/ground_truth_compare.sh
//! ```

#[bashrs::main]
fn main() {
    echo("======================================================================");
    echo("           GROUND TRUTH 3-COLUMN COMPARISON (WAPR-QA-GT)              ");
    echo("======================================================================");
    echo("");
    echo("Audio: demos/test-audio/test-speech-1.5s.wav");
    echo("");

    echo("Running transcription tests...");
    echo("");

    // Run whisper.apr
    echo("[1/3] whisper.apr:");
    exec("cargo run --release --bin whisper-apr-cli --features cli -- transcribe --model-path models/whisper-tiny.apr -q demos/test-audio/test-speech-1.5s.wav 2>/dev/null");

    echo("");

    // Run whisper.cpp
    echo("[2/3] whisper.cpp:");
    exec("/home/noah/.local/bin/main -m /home/noah/src/whisper.cpp/models/ggml-tiny.bin -f demos/test-audio/test-speech-1.5s.wav 2>/dev/null");

    echo("");

    // Run HuggingFace
    echo("[3/3] HuggingFace (via uv):");
    exec("uv run scripts/hf_transcribe.py demos/test-audio/test-speech-1.5s.wav");

    echo("");
    echo("======================================================================");
    echo("Compare outputs above manually or use ground_truth_validate.py");
    echo("======================================================================");
}

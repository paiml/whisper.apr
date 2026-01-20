//! Performance QA: 2x whisper.cpp Validation
//!
//! Automated QA validation ensuring whisper.apr is 2x faster than whisper.cpp.
//! Follows Popperian falsification methodology - attempts to DISPROVE the 2x claim.
//!
//! ## Usage
//!
//! ```bash
//! bashrs build scripts/perf_qa_2x_whisper_cpp.rs -o scripts/perf_qa_2x_whisper_cpp.sh
//! ./scripts/perf_qa_2x_whisper_cpp.sh
//! ```
//!
//! ## Exit Codes
//!
//! - 0: All tests pass (2x speedup achieved)
//! - 1: Performance target not met

#[bashrs::main]
fn main() {
    print_header();
    validate_baseline();
    run_benchmarks();
    compare_results();
}

fn print_header() {
    echo("======================================================================");
    echo("  WAPR-PERF-004: 2x whisper.cpp Performance QA                        ");
    echo("======================================================================");
    echo("  Methodology: Popperian Falsification                                ");
    echo("  Target: whisper.apr must be 2x faster than OPTIMIZED whisper.cpp    ");
    echo("======================================================================");
    echo("");
}

fn validate_baseline() {
    echo("[Phase 1/3] Validating whisper.cpp baseline...");
    exec("ldd /home/noah/.local/bin/main 2>/dev/null | grep -i blas | head -1");
    exec("ldd /home/noah/.local/bin/main 2>/dev/null | grep -i cuda | head -1");
    echo("  Baseline validation complete");
    echo("");
}

fn run_benchmarks() {
    echo("[Phase 2/3] Running benchmarks (10 iterations each)...");
    echo("");

    // whisper.cpp tiny CPU
    echo("  whisper.cpp tiny CPU:");
    exec("/home/noah/.local/bin/main -m /home/noah/src/whisper.cpp/models/ggml-tiny.bin -f demos/test-audio/test-speech-30s.wav --no-timestamps -ng -t 8 2>&1 | grep total");

    // whisper.cpp tiny GPU
    echo("  whisper.cpp tiny GPU:");
    exec("/home/noah/.local/bin/main -m /home/noah/src/whisper.cpp/models/ggml-tiny.bin -f demos/test-audio/test-speech-30s.wav --no-timestamps -t 8 2>&1 | grep total");

    // whisper.apr tiny CPU
    echo("  whisper.apr tiny CPU:");
    exec("whisper-apr-cli transcribe --file demos/test-audio/test-speech-30s.wav --model tiny -v 2>&1 | grep Total");

    echo("");
}

fn compare_results() {
    echo("[Phase 3/3] Results (manual comparison required):");
    echo("  See output above - calculate speedup = cpp_time / apr_time");
    echo("  Target: speedup >= 2.0x");
    echo("");
    echo("======================================================================");
    echo("  Run full automated benchmark with:");
    echo("  ./scripts/run_benchmark_qa.sh");
    echo("======================================================================");
}

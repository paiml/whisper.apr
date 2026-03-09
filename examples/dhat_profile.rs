//! Heap profiling with dhat-rs
//!
//! Produces a `dhat-heap.json` file analyzable at https://nnethercote.github.io/dh_view/dh_view.html
//!
//! Run with:
//!   cargo run --example dhat_profile --features dhat-profiler --release
//!
//! The output JSON captures every allocation site, count, total bytes, and peak bytes.
//! Use it to find:
//! - Hot allocation sites (high count)
//! - Large transient allocations (high total, low peak)
//! - Memory leaks (monotonically growing peak)

#[cfg(feature = "dhat-profiler")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use whisper_apr::{
    audio::{MelConfig, MelFilterbank},
    model::ModelConfig,
    TranscribeOptions, WhisperApr,
};

fn main() {
    #[cfg(feature = "dhat-profiler")]
    let _profiler = dhat::Profiler::new_heap();

    println!("=== dhat-rs Heap Profile ===\n");

    // Phase 1: Model construction
    println!("[phase 1] Model construction...");
    let whisper = WhisperApr::tiny();
    println!(
        "  model_type={:?}, memory={}KB",
        whisper.model_type(),
        whisper.memory_size() / 1024
    );

    // Phase 2: Audio preprocessing (mel spectrogram)
    println!("[phase 2] Mel spectrogram...");
    let mel = MelFilterbank::new(&MelConfig::whisper());
    let sample_rate = 16000;
    let duration_secs = 3.0;
    let audio: Vec<f32> = (0..((sample_rate as f32 * duration_secs) as usize))
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();
    println!("  audio: {} samples ({:.1}s)", audio.len(), duration_secs);

    match mel.compute(&audio) {
        Ok(mel_spec) => {
            let frames = mel_spec.len() / 80;
            println!(
                "  mel: {} frames x 80 bins = {} floats",
                frames,
                mel_spec.len()
            );
        }
        Err(e) => eprintln!("  mel error: {e}"),
    }

    // Phase 3: Encoder forward pass (synthetic — exercises allocation patterns)
    println!("[phase 3] Encoder...");
    let config = ModelConfig::tiny();
    let n_audio_ctx = 1500; // 30s audio → 1500 frames
    let encoder_output_size = n_audio_ctx * config.n_audio_state as usize;
    let encoder_features = vec![0.0f32; encoder_output_size];
    println!(
        "  encoder output: {} floats ({}KB)",
        encoder_features.len(),
        encoder_features.len() * 4 / 1024
    );

    // Phase 4: Decoder token generation (exercises Vec growth patterns)
    println!("[phase 4] Decoder token generation...");
    let mut tokens: Vec<u32> = Vec::new();
    for i in 0..448 {
        // max_tokens = n_text_ctx
        tokens.push(i);
    }
    println!("  tokens: {} generated", tokens.len());

    // Phase 5: Transcription options (exercises String allocation)
    println!("[phase 5] TranscribeOptions...");
    let _opts = TranscribeOptions {
        language: Some("en".to_string()),
        hotwords: vec![
            "Kubernetes".to_string(),
            "Terraform".to_string(),
            "gRPC".to_string(),
        ],
        ..Default::default()
    };

    // Phase 6: Memory pool exercise
    println!("[phase 6] Memory pool...");
    let pool = whisper_apr::memory::MemoryPool::new();
    for size in [1024, 4096, 16384, 65536, 262144] {
        let buf = pool.get(size);
        pool.return_buffer(buf);
    }
    let buf1 = pool.get(65536);
    let buf2 = pool.get(65536);
    pool.return_buffer(buf1);
    pool.return_buffer(buf2);
    let stats = pool.stats();
    println!(
        "  pool: {} allocs, {:.0}% hit rate, {}KB buffered",
        stats.allocations,
        stats.hit_rate(),
        pool.buffered_bytes() / 1024
    );

    println!("\n=== Profile complete ===");
    println!("Output: dhat-heap.json");
    println!("View at: https://nnethercote.github.io/dh_view/dh_view.html");
}

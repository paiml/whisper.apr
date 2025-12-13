//! Streaming audio processing example
//!
//! Demonstrates the streaming API for real-time audio processing.
//!
//! Run with: `cargo run --example streaming_audio`

use whisper_apr::audio::{RingBuffer, StreamingConfig, StreamingProcessor};

fn main() {
    println!("=== Whisper.apr Streaming Audio Example ===\n");

    // Configure streaming processor
    let config = StreamingConfig::with_sample_rate(16000).without_vad(); // Disable VAD for demo

    println!("Streaming configuration:");
    println!("  Input sample rate: 16000 Hz");
    println!("  Chunk samples: {}", config.chunk_samples());
    println!("  Overlap samples: {}", config.overlap_samples());
    println!();

    // Create streaming processor
    let mut processor = StreamingProcessor::new(config);
    println!("Initial state: {:?}", processor.state());
    println!();

    // Simulate streaming audio input
    let sample_rate: u32 = 16000;
    let chunk_duration_ms = 100; // 100ms chunks
    let chunk_samples = (sample_rate * chunk_duration_ms / 1000) as usize;

    println!(
        "Simulating {} chunks of {}ms audio each...\n",
        5, chunk_duration_ms
    );

    for chunk_idx in 0..5 {
        // Generate synthetic audio chunk (sine wave with varying frequency)
        let frequency = 200.0 + (chunk_idx as f32 * 100.0);
        let audio_chunk: Vec<f32> = (0..chunk_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3
            })
            .collect();

        // Push audio to processor
        processor.push_audio(&audio_chunk);

        let stats = processor.stats();
        println!(
            "Chunk {}: pushed {} samples ({}Hz tone)",
            chunk_idx + 1,
            audio_chunk.len(),
            frequency
        );
        println!("  State: {:?}", processor.state());
        println!("  Buffer fill: {:.1}%", stats.buffer_fill() * 100.0);
        println!(
            "  Duration processed: {:.2}s",
            stats.duration_processed(sample_rate)
        );

        // Check if a chunk is ready for processing
        if let Some(ready_chunk) = processor.get_chunk() {
            println!("  -> Chunk ready! {} samples", ready_chunk.len());
        }
        println!();
    }

    // Demonstrate ring buffer directly
    println!("=== Ring Buffer Demo ===\n");

    let mut ring = RingBuffer::new(1024);
    println!("Ring buffer capacity: {}", ring.capacity());

    // Write some data
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    ring.write(&data);
    println!(
        "Wrote {} samples, available: {}",
        data.len(),
        ring.available_read()
    );

    // Read back
    let mut output = vec![0.0; 50];
    let read = ring.read(&mut output);
    println!(
        "Read {} samples, remaining: {}",
        read,
        ring.available_read()
    );

    // Peek without consuming
    let mut peek_buf = vec![0.0; 10];
    let peeked = ring.peek(&mut peek_buf);
    println!(
        "Peeked {} samples, still available: {}",
        peeked,
        ring.available_read()
    );

    println!("\n=== Example Complete ===");
}

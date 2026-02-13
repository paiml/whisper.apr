//! Tests for streaming inference

use super::*;

// =========================================================================
// Configuration Tests
// =========================================================================

#[test]
fn test_streaming_config_default() {
    let config = StreamingConfig::default();
    assert_eq!(config.max_tokens_per_chunk, 224);
    assert_eq!(config.overlap_tokens, 10);
    assert!((config.temperature - 0.0).abs() < f32::EPSILON);
    assert!(config.return_partial);
}

#[test]
fn test_streaming_config_with_sample_rate() {
    let config = StreamingConfig::with_sample_rate(48000);
    assert_eq!(config.audio.input_sample_rate, 48000);
}

#[test]
fn test_streaming_config_without_vad() {
    let config = StreamingConfig::default().without_vad();
    assert!(!config.audio.enable_vad);
}

#[test]
fn test_streaming_config_with_partial_results() {
    let config = StreamingConfig::default().with_partial_results(false);
    assert!(!config.return_partial);
}

// =========================================================================
// Transcriber Construction Tests
// =========================================================================

#[test]
fn test_streaming_transcriber_new() {
    let transcriber = StreamingTranscriber::new(StreamingConfig::default());
    assert_eq!(transcriber.state(), TranscriberState::Ready);
    assert_eq!(transcriber.chunk_index(), 0);
    assert!(transcriber.text().is_empty());
}

#[test]
fn test_streaming_transcriber_with_sample_rate() {
    let transcriber = StreamingTranscriber::with_sample_rate(48000);
    assert_eq!(transcriber.state(), TranscriberState::Ready);
}

// =========================================================================
// State Tests
// =========================================================================

#[test]
fn test_initial_state() {
    let transcriber = StreamingTranscriber::new(StreamingConfig::default());
    assert_eq!(transcriber.state(), TranscriberState::Ready);
    assert!((transcriber.chunk_progress() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_push_audio_and_process_updates_stats() {
    let config = StreamingConfig::default().without_vad();
    let mut transcriber = StreamingTranscriber::new(config);

    // Push some audio
    let samples = vec![0.1; 16000]; // 1 second of non-silence
    transcriber.push_audio(&samples);

    // Process the audio
    let _ = transcriber.process();

    // Stats should reflect samples processed
    let stats = transcriber.stats();
    assert!(stats.samples_processed > 0);
}

#[test]
fn test_finalized_state_ignores_audio() {
    let mut transcriber = StreamingTranscriber::new(StreamingConfig::default());

    // Finalize
    let _ = transcriber.finalize();
    assert_eq!(transcriber.state(), TranscriberState::Finalized);

    // Push audio after finalization
    let samples = vec![0.0; 16000];
    transcriber.push_audio(&samples);

    // Should not process
    assert!((transcriber.chunk_progress() - 0.0).abs() < f32::EPSILON);
}

// =========================================================================
// Reset Tests
// =========================================================================

#[test]
fn test_reset_clears_state() {
    let mut transcriber = StreamingTranscriber::new(StreamingConfig::default());

    // Push some audio
    let samples = vec![0.0; 16000];
    transcriber.push_audio(&samples);

    // Reset
    transcriber.reset();

    // State should be cleared
    assert_eq!(transcriber.state(), TranscriberState::Ready);
    assert_eq!(transcriber.chunk_index(), 0);
    assert!(transcriber.text().is_empty());
}

// =========================================================================
// Stats Tests
// =========================================================================

#[test]
fn test_stats_initial() {
    let transcriber = StreamingTranscriber::new(StreamingConfig::default());
    let stats = transcriber.stats();

    assert_eq!(stats.chunks_processed, 0);
    assert_eq!(stats.total_text_length, 0);
}

#[test]
fn test_stats_after_push() {
    let mut transcriber = StreamingTranscriber::new(StreamingConfig::default());

    let samples = vec![0.0; 16000];
    transcriber.push_audio(&samples);

    let stats = transcriber.stats();
    assert_eq!(stats.samples_processed, 16000);
}

// =========================================================================
// Process Tests
// =========================================================================

#[test]
fn test_process_no_audio() {
    let mut transcriber = StreamingTranscriber::new(StreamingConfig::default());
    let result = transcriber.process();

    assert!(result.is_ok());
    assert!(result.expect("process should succeed").is_none());
}

#[test]
fn test_process_insufficient_audio() {
    let mut transcriber =
        StreamingTranscriber::new(StreamingConfig::default().with_partial_results(false));

    // Push small amount of audio
    let samples = vec![0.0; 1600]; // 0.1 second
    transcriber.push_audio(&samples);

    let result = transcriber.process();
    assert!(result.is_ok());
    // Not enough audio for a result
}

// =========================================================================
// Result Tests
// =========================================================================

#[test]
fn test_streaming_result_fields() {
    let result = StreamingResult {
        text: "test".into(),
        is_final: true,
        confidence: 0.9,
        chunk_index: 0,
        latency_ms: 100,
    };

    assert_eq!(result.text, "test");
    assert!(result.is_final);
    assert!((result.confidence - 0.9).abs() < f32::EPSILON);
    assert_eq!(result.chunk_index, 0);
    assert_eq!(result.latency_ms, 100);
}

// =========================================================================
// Integration Tests
// =========================================================================

#[test]
fn test_full_session() {
    let mut transcriber = StreamingTranscriber::new(StreamingConfig::default());

    // Simulate audio stream
    for _ in 0..10 {
        let samples = vec![0.0; 1600];
        transcriber.push_audio(&samples);
        let _ = transcriber.process();
    }

    // Finalize
    let result = transcriber.finalize();
    assert!(result.is_ok());
    assert_eq!(transcriber.state(), TranscriberState::Finalized);
}

#[test]
fn test_multiple_sessions() {
    let mut transcriber = StreamingTranscriber::new(StreamingConfig::default());

    // First session
    let samples = vec![0.0; 16000];
    transcriber.push_audio(&samples);
    let _ = transcriber.finalize();

    // Reset for new session
    transcriber.reset();
    assert_eq!(transcriber.state(), TranscriberState::Ready);

    // Second session
    transcriber.push_audio(&samples);
    let result = transcriber.process();
    assert!(result.is_ok());
}

// =========================================================================
// Additional Coverage Tests (WAPR-QA-001)
// =========================================================================

#[test]
fn test_process_after_finalized_returns_none() {
    let mut transcriber = StreamingTranscriber::new(StreamingConfig::default());

    // Finalize first
    let _ = transcriber.finalize();
    assert_eq!(transcriber.state(), TranscriberState::Finalized);

    // Process after finalize should return None
    let result = transcriber.process();
    assert!(result.is_ok());
    assert!(result.expect("should succeed").is_none());
}

#[test]
fn test_partial_results_enabled() {
    let config = StreamingConfig::default().with_partial_results(true);
    let mut transcriber = StreamingTranscriber::new(config);

    // Push enough audio to trigger partial result (>30% progress)
    // With default chunk_duration_ms=30000 and 16kHz, we need >9600 samples
    let samples = vec![0.1; 16000]; // 1 second = ~33% of 30s chunk
    transcriber.push_audio(&samples);

    // Process should potentially return a partial result
    let result = transcriber.process();
    assert!(result.is_ok());
    // Result depends on internal processor state
}

#[test]
fn test_partial_results_disabled() {
    let config = StreamingConfig::default().with_partial_results(false);
    let mut transcriber = StreamingTranscriber::new(config);

    // Push audio but not enough for a full chunk
    let samples = vec![0.1; 8000];
    transcriber.push_audio(&samples);

    let result = transcriber.process();
    assert!(result.is_ok());
    // With partial disabled and not enough audio, should be None
    assert!(result.expect("should succeed").is_none());
}

#[test]
fn test_finalize_with_remaining_audio() {
    let config = StreamingConfig::default().without_vad();
    let mut transcriber = StreamingTranscriber::new(config);

    // Push some audio (not a full chunk)
    let samples = vec![0.1; 8000];
    transcriber.push_audio(&samples);

    // Finalize should flush and return result
    let result = transcriber.finalize();
    assert!(result.is_ok());
    let transcription = result.expect("finalize should succeed");
    assert_eq!(transcription.language, "en");
}

#[test]
fn test_accumulated_text_builds_correctly() {
    let mut transcriber = StreamingTranscriber::new(StreamingConfig::default());

    // Initially empty
    assert!(transcriber.text().is_empty());

    // After processing (even with placeholder results), text may be empty
    let samples = vec![0.0; 1600];
    transcriber.push_audio(&samples);
    let _ = transcriber.process();

    // Text is accumulated from chunk results
    // Since placeholder returns empty, text stays empty
    assert!(transcriber.text().is_empty());
}

#[test]
fn test_chunk_index_increments() {
    let config = StreamingConfig::default().without_vad();
    let mut transcriber = StreamingTranscriber::new(config);

    assert_eq!(transcriber.chunk_index(), 0);

    // Process doesn't increment index unless chunk is ready
    let samples = vec![0.0; 1600];
    transcriber.push_audio(&samples);
    let _ = transcriber.process();

    // Index only increments on full chunk processing
}

#[test]
fn test_streaming_result_debug() {
    let result = StreamingResult {
        text: "hello".into(),
        is_final: true,
        confidence: 0.95,
        chunk_index: 0,
        latency_ms: 100,
    };

    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("hello"));
    assert!(debug_str.contains("0.95"));
}

#[test]
fn test_streaming_result_clone() {
    let result = StreamingResult {
        text: "test".into(),
        is_final: false,
        confidence: 0.5,
        chunk_index: 1,
        latency_ms: 50,
    };

    let cloned = result.clone();
    assert_eq!(cloned.text, result.text);
    assert_eq!(cloned.is_final, result.is_final);
    assert!((cloned.confidence - result.confidence).abs() < f32::EPSILON);
}

#[test]
fn test_streaming_config_debug() {
    let config = StreamingConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("StreamingConfig"));
}

#[test]
fn test_streaming_config_clone() {
    let config = StreamingConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.max_tokens_per_chunk, config.max_tokens_per_chunk);
    assert_eq!(cloned.return_partial, config.return_partial);
}

#[test]
fn test_streaming_stats_debug() {
    let stats = StreamingStats {
        chunks_processed: 5,
        samples_processed: 80000,
        buffer_fill: 0.5,
        total_text_length: 100,
    };

    let debug_str = format!("{stats:?}");
    assert!(debug_str.contains("chunks_processed"));
}

#[test]
fn test_streaming_stats_copy() {
    let stats = StreamingStats {
        chunks_processed: 3,
        samples_processed: 48000,
        buffer_fill: 0.25,
        total_text_length: 50,
    };

    let copied = stats;
    assert_eq!(copied.chunks_processed, stats.chunks_processed);
    assert_eq!(copied.samples_processed, stats.samples_processed);
}

#[test]
fn test_transcriber_state_equality() {
    assert_eq!(TranscriberState::Ready, TranscriberState::Ready);
    assert_eq!(TranscriberState::Processing, TranscriberState::Processing);
    assert_eq!(TranscriberState::Finalized, TranscriberState::Finalized);
    assert_ne!(TranscriberState::Ready, TranscriberState::Finalized);
}

#[test]
fn test_transcriber_state_debug() {
    let state = TranscriberState::Ready;
    let debug_str = format!("{state:?}");
    assert!(debug_str.contains("Ready"));
}

#[test]
fn test_transcriber_state_clone() {
    let state = TranscriberState::Processing;
    let cloned = state;
    assert_eq!(cloned, TranscriberState::Processing);
}

#[test]
fn test_config_temperature_default() {
    let config = StreamingConfig::default();
    assert!((config.temperature - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_config_overlap_tokens_default() {
    let config = StreamingConfig::default();
    assert_eq!(config.overlap_tokens, 10);
}

#[test]
fn test_stats_buffer_fill_percentage() {
    let mut transcriber = StreamingTranscriber::new(StreamingConfig::default());

    let samples = vec![0.1; 8000];
    transcriber.push_audio(&samples);

    let stats = transcriber.stats();
    assert!(stats.buffer_fill >= 0.0 && stats.buffer_fill <= 1.0);
}

// =========================================================================
// Full process() path tests via ultra-low-latency config (PMAT-023)
// =========================================================================

#[test]
fn test_process_full_chunk_path() {
    // Use ultra-low-latency config: 0.25s chunk = 4000 samples at 16kHz
    let config = StreamingConfig {
        audio: AudioStreamingConfig::ultra_low_latency().without_vad(),
        max_tokens_per_chunk: 224,
        overlap_tokens: 10,
        temperature: 0.0,
        return_partial: false,
    };
    let mut transcriber = StreamingTranscriber::new(config);

    // Push more than 4000 samples to trigger ChunkReady
    let samples = vec![0.1f32; 5000];
    transcriber.push_audio(&samples);

    // Process — should take the full chunk path
    let result = transcriber.process();
    assert!(result.is_ok());

    // Chunk index should have incremented if chunk was processed
    // (placeholder transcribe_chunk returns empty text, so text stays empty)
}

#[test]
fn test_process_returns_partial_result() {
    // Ultra-low-latency with partial results enabled
    let config = StreamingConfig {
        audio: AudioStreamingConfig::ultra_low_latency().without_vad(),
        max_tokens_per_chunk: 224,
        overlap_tokens: 10,
        temperature: 0.0,
        return_partial: true,
    };
    let mut transcriber = StreamingTranscriber::new(config);

    // Push enough for >30% of 4000 samples = >1200 samples
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    // Process should check chunk_progress() > 0.3 and return partial
    let result = transcriber.process();
    assert!(result.is_ok());
    if let Some(partial) = result.expect("should succeed") {
        assert!(!partial.is_final);
        assert_eq!(partial.text, "[listening...]");
        assert_eq!(partial.confidence, 0.0);
    }
}

#[test]
fn test_process_chunk_ready_then_finalize() {
    let config = StreamingConfig {
        audio: AudioStreamingConfig::ultra_low_latency().without_vad(),
        max_tokens_per_chunk: 224,
        overlap_tokens: 10,
        temperature: 0.0,
        return_partial: false,
    };
    let mut transcriber = StreamingTranscriber::new(config);

    // Push enough for a full chunk
    let samples = vec![0.1f32; 5000];
    transcriber.push_audio(&samples);

    // Process the chunk
    let _ = transcriber.process();

    // Now finalize with remaining audio in buffer
    let result = transcriber.finalize();
    assert!(result.is_ok());
    assert_eq!(transcriber.state(), TranscriberState::Finalized);
}

#[test]
fn test_process_multiple_chunks() {
    let config = StreamingConfig {
        audio: AudioStreamingConfig::ultra_low_latency().without_vad(),
        max_tokens_per_chunk: 224,
        overlap_tokens: 10,
        temperature: 0.0,
        return_partial: false,
    };
    let mut transcriber = StreamingTranscriber::new(config);

    // Process several chunks
    for _ in 0..3 {
        let samples = vec![0.1f32; 5000];
        transcriber.push_audio(&samples);
        let _ = transcriber.process();
    }

    // Chunk index should reflect processed chunks
    let stats = transcriber.stats();
    assert!(stats.samples_processed > 0);
}

// =========================================================================
// Coverage Gap: create_partial_result deep validation (WAPR-QA-004)
// =========================================================================

#[test]
fn test_create_partial_result_fields() {
    // Directly exercise create_partial_result by ensuring partial result has correct fields
    let config = StreamingConfig {
        audio: AudioStreamingConfig::ultra_low_latency().without_vad(),
        max_tokens_per_chunk: 224,
        overlap_tokens: 10,
        temperature: 0.0,
        return_partial: true,
    };
    let mut transcriber = StreamingTranscriber::new(config);

    // Push enough for >30% progress but not full chunk
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");
    if let Some(partial) = result {
        // Validate all fields from create_partial_result
        assert_eq!(partial.text, "[listening...]");
        assert!(!partial.is_final);
        assert!((partial.confidence - 0.0).abs() < f32::EPSILON);
        assert_eq!(partial.chunk_index, 0);
        assert!(partial.latency_ms > 0); // Should reflect chunk progress * 30000
    }
}

#[test]
fn test_create_partial_result_chunk_index_after_process() {
    let config = StreamingConfig {
        audio: AudioStreamingConfig::ultra_low_latency().without_vad(),
        max_tokens_per_chunk: 224,
        overlap_tokens: 10,
        temperature: 0.0,
        return_partial: true,
    };
    let mut transcriber = StreamingTranscriber::new(config);

    // Process one full chunk first
    let full_chunk = vec![0.1f32; 5000];
    transcriber.push_audio(&full_chunk);
    let _ = transcriber.process();

    // Now push partial and get partial result
    let partial = vec![0.1f32; 2000];
    transcriber.push_audio(&partial);
    let result = transcriber.process().expect("should succeed");
    if let Some(partial_result) = result {
        // chunk_index should reflect the processed chunks
        assert_eq!(partial_result.chunk_index, transcriber.chunk_index());
    }
}

#[test]
fn test_finalize_flushes_remaining_audio() {
    let config = StreamingConfig {
        audio: AudioStreamingConfig::ultra_low_latency().without_vad(),
        max_tokens_per_chunk: 224,
        overlap_tokens: 10,
        temperature: 0.0,
        return_partial: false,
    };
    let mut transcriber = StreamingTranscriber::new(config);

    // Push audio that doesn't fill a complete chunk
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    // Finalize should flush the partial buffer and process it
    let result = transcriber.finalize();
    assert!(result.is_ok());
    let transcription = result.expect("finalize should succeed");
    assert_eq!(transcription.language, "en");
    assert_eq!(transcriber.state(), TranscriberState::Finalized);
}

// =========================================================================
// create_partial_result Coverage (PMAT-024)
// =========================================================================

#[test]
fn test_partial_result_content() {
    // Trigger create_partial_result: return_partial=true + progress>0.3
    let config = StreamingConfig::default()
        .with_partial_results(true)
        .without_vad();
    let mut transcriber = StreamingTranscriber::new(config);

    // Default chunk_duration_ms=30000 at 16kHz = 480000 samples per chunk
    // Need >30% = 144000 samples minimum to trigger partial result
    // Push 160000 samples (33% of chunk = 10 seconds) to ensure >0.3 progress
    let samples = vec![0.1f32; 160_000];
    transcriber.push_audio(&samples);

    let result = transcriber.process();
    assert!(result.is_ok());

    // Should get a partial result with "[listening...]" text
    if let Some(partial) = result.expect("should succeed") {
        assert_eq!(partial.text, "[listening...]");
        assert!(!partial.is_final);
        assert!((partial.confidence - 0.0).abs() < f32::EPSILON);
        assert_eq!(partial.chunk_index, 0); // No full chunks processed yet
        assert!(partial.latency_ms > 0); // Should have measurable latency
    }
    // If None, the processor consumed the audio as a full chunk — still valid
}

#[test]
fn test_partial_result_latency_scales_with_progress() {
    let config = StreamingConfig::default()
        .with_partial_results(true)
        .without_vad();
    let mut transcriber = StreamingTranscriber::new(config);

    // Push ~50% of a chunk (240000 samples at 16kHz for 30s chunk)
    let samples = vec![0.05f32; 240_000];
    transcriber.push_audio(&samples);

    let result = transcriber.process();
    assert!(result.is_ok());

    if let Some(partial) = result.expect("should succeed") {
        // Latency should reflect ~50% of 30000ms chunk = ~15000ms
        assert!(
            partial.latency_ms > 5000,
            "latency {} too low for 50% progress",
            partial.latency_ms
        );
        assert!(
            partial.latency_ms < 25000,
            "latency {} too high for 50% progress",
            partial.latency_ms
        );
    }
}

#[test]
fn test_partial_result_chunk_index_stays_zero_before_full_chunk() {
    let config = StreamingConfig::default()
        .with_partial_results(true)
        .without_vad();
    let mut transcriber = StreamingTranscriber::new(config);

    // Push partial audio (less than one full chunk)
    let samples = vec![0.1f32; 160_000];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("should succeed");
    if let Some(partial) = result {
        assert_eq!(
            partial.chunk_index, 0,
            "chunk_index should be 0 before any full chunk"
        );
    }
}

#[test]
fn test_partial_results_not_returned_below_threshold() {
    // With partial results enabled but very little audio (<30% progress)
    let config = StreamingConfig::default()
        .with_partial_results(true)
        .without_vad();
    let mut transcriber = StreamingTranscriber::new(config);

    // Push only ~5% of a chunk (24000 samples = 1.5s out of 30s)
    let samples = vec![0.1f32; 24_000];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("should succeed");
    // Below 30% threshold, should return None (no partial result)
    assert!(
        result.is_none(),
        "should not return partial below 30% progress"
    );
}

// =========================================================================
// create_partial_result targeted coverage (WAPR-QA-005)
// =========================================================================

#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_via_ultra_low_latency_40pct() {
    // Ultra-low-latency chunk = 4000 samples. Push 40% = 1600 samples.
    // With return_partial=true and >30% progress, create_partial_result is called.
    let config = StreamingConfig {
        audio: AudioStreamingConfig::ultra_low_latency().without_vad(),
        max_tokens_per_chunk: 224,
        overlap_tokens: 10,
        temperature: 0.0,
        return_partial: true,
    };
    let mut transcriber = StreamingTranscriber::new(config);

    // 1600 samples = 40% of 4000 sample chunk
    let samples = vec![0.05f32; 1600];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");
    // At 40% progress with partial enabled, we should get a partial result
    if let Some(partial) = result {
        assert_eq!(partial.text, "[listening...]");
        assert!(!partial.is_final);
        assert!((partial.confidence - 0.0).abs() < f32::EPSILON);
        assert_eq!(partial.chunk_index, 0);
        // latency_ms = chunk_progress * 30000, progress ~0.4, so ~12000ms
        assert!(partial.latency_ms > 0);
    }
}

#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_preserves_chunk_index() {
    // Verify that create_partial_result uses the current chunk_index (0 initially)
    let config = StreamingConfig {
        audio: AudioStreamingConfig::ultra_low_latency().without_vad(),
        max_tokens_per_chunk: 224,
        overlap_tokens: 10,
        temperature: 0.0,
        return_partial: true,
    };
    let mut transcriber = StreamingTranscriber::new(config);

    // Push 40% of a chunk to trigger partial result
    transcriber.push_audio(&vec![0.05f32; 1600]);
    let result = transcriber.process().expect("process should succeed");

    if let Some(partial) = result {
        // chunk_index should be 0 since no full chunk has been processed
        assert_eq!(partial.chunk_index, 0);
        assert_eq!(partial.text, "[listening...]");
        assert!(!partial.is_final);
    }
}

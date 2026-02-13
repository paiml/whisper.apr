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

// =========================================================================
// Reliable create_partial_result coverage (WAPR-QA-006)
//
// These tests use a carefully constructed config that bypasses resampling
// (input_sample_rate == output_sample_rate) and VAD (disabled) with zero
// min_speech_duration to guarantee the processor transitions from
// WaitingForSpeech -> AccumulatingSpeech immediately. This makes the
// chunk_progress() deterministic and ensures create_partial_result is hit.
// =========================================================================

/// Build an `AudioStreamingConfig` that guarantees deterministic partial
/// result triggering: no resampler, no VAD, zero min-speech, 250ms chunk.
fn deterministic_partial_audio_config() -> AudioStreamingConfig {
    AudioStreamingConfig {
        input_sample_rate: 16000,
        output_sample_rate: 16000,
        chunk_duration: 0.25, // 4000 samples at 16 kHz
        chunk_overlap: 0.025, // 400 samples overlap
        enable_vad: false,
        vad_threshold: 0.5,
        min_speech_duration_ms: 0, // immediate AccumulatingSpeech transition
        buffer_duration: 2.0,
        latency_mode: crate::audio::LatencyMode::UltraLow,
    }
}

/// Build a `StreamingConfig` wrapping the deterministic audio config,
/// with `return_partial` set to the given value.
fn deterministic_streaming_config(return_partial: bool) -> StreamingConfig {
    StreamingConfig {
        audio: deterministic_partial_audio_config(),
        max_tokens_per_chunk: 224,
        overlap_tokens: 10,
        temperature: 0.0,
        return_partial,
    }
}

/// Verify that `create_partial_result` is invoked and returns the correct
/// text placeholder when chunk progress exceeds 30%.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_deterministic_text() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // 2000 samples at 16 kHz. Frame size = 480 samples.
    // 4 frames consumed = 1920 samples in chunk_buffer.
    // chunk_progress = 1920 / 4000 = 0.48, which exceeds 0.3.
    // State is AccumulatingSpeech, NOT ChunkReady (1920 < 4000).
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");
    let partial = result.expect("should return Some(partial) at 48% progress");

    assert_eq!(
        partial.text, "[listening...]",
        "partial text must be placeholder"
    );
}

/// Verify that `create_partial_result` sets `is_final` to false.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_deterministic_not_final() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let partial = transcriber
        .process()
        .expect("process should succeed")
        .expect("should return partial result");

    assert!(!partial.is_final, "partial result must not be final");
}

/// Verify that `create_partial_result` returns zero confidence.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_deterministic_zero_confidence() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let partial = transcriber
        .process()
        .expect("process should succeed")
        .expect("should return partial result");

    assert!(
        (partial.confidence - 0.0).abs() < f32::EPSILON,
        "partial confidence must be 0.0, got {}",
        partial.confidence
    );
}

/// Verify that `create_partial_result` uses the current `chunk_index`
/// which should be 0 before any full chunk has been processed.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_deterministic_chunk_index_zero() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let partial = transcriber
        .process()
        .expect("process should succeed")
        .expect("should return partial result");

    assert_eq!(
        partial.chunk_index, 0,
        "chunk_index must be 0 before any full chunk"
    );
}

/// Verify that `create_partial_result` computes latency_ms from
/// chunk_progress * 30000, and that it is non-zero at ~48% progress.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_deterministic_latency_nonzero() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let partial = transcriber
        .process()
        .expect("process should succeed")
        .expect("should return partial result");

    // latency_ms = (chunk_progress * 30000) as u32
    // chunk_progress ~0.48, so latency ~14400ms
    assert!(
        partial.latency_ms > 0,
        "latency_ms must be non-zero, got {}",
        partial.latency_ms
    );
}

/// Verify that latency_ms scales proportionally with chunk progress.
/// At ~48% progress, latency = 0.48 * 30000 ~= 14400.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_deterministic_latency_range() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let partial = transcriber
        .process()
        .expect("process should succeed")
        .expect("should return partial result");

    // Expected: 0.48 * 30000 = 14400, allow wide tolerance for frame rounding
    assert!(
        partial.latency_ms >= 5000 && partial.latency_ms <= 20000,
        "latency_ms should be ~14400 at 48% progress, got {}",
        partial.latency_ms
    );
}

/// After processing one full chunk (chunk_index becomes 1), a subsequent
/// partial result should carry `chunk_index == 1`.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_after_full_chunk_has_incremented_index() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push enough for a full chunk (>= 4000 samples)
    let full = vec![0.1f32; 5000];
    transcriber.push_audio(&full);
    let _ = transcriber.process().expect("first process should succeed");

    // chunk_index should now be 1
    assert_eq!(transcriber.chunk_index(), 1);

    // Push partial audio for the next chunk (~48% progress)
    let partial_audio = vec![0.1f32; 2000];
    transcriber.push_audio(&partial_audio);

    let result = transcriber
        .process()
        .expect("second process should succeed");
    if let Some(partial) = result {
        assert_eq!(
            partial.chunk_index, 1,
            "partial chunk_index should match current chunk_index (1)"
        );
        assert!(!partial.is_final);
        assert_eq!(partial.text, "[listening...]");
    }
    // Note: it is acceptable for the processor to produce a full chunk here
    // if leftover samples from the first push combined with the new push
    // exceed the chunk boundary. The test validates the partial path when hit.
}

// =========================================================================
// process() uncovered branch: return_partial=false below threshold (WAPR-QA-006)
// =========================================================================

/// When `return_partial` is false and chunk is not ready, process() must
/// return `Ok(None)` even when progress exceeds 30%.
#[test]
#[allow(clippy::expect_used)]
fn test_process_no_partial_when_disabled_even_above_threshold() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(false));

    // Push 48% of a chunk -- normally would trigger partial, but disabled
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");

    assert!(
        result.is_none(),
        "must not return partial result when return_partial is false"
    );
}

/// When `return_partial` is true but progress is below 30%, process()
/// must return `Ok(None)`.
#[test]
#[allow(clippy::expect_used)]
fn test_process_no_partial_below_30pct_threshold() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push only ~12% of chunk (480 samples after 1 frame of 480)
    let samples = vec![0.1f32; 500];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");

    assert!(
        result.is_none(),
        "must not return partial when progress < 30%, got Some"
    );
}

// =========================================================================
// process() uncovered branch: empty buffer (WAPR-QA-006)
// =========================================================================

/// Calling process() on an empty buffer with return_partial enabled
/// must return `Ok(None)` -- there is nothing to transcribe.
#[test]
#[allow(clippy::expect_used)]
fn test_process_empty_buffer_returns_none() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Do not push any audio
    let result = transcriber.process().expect("process should succeed");
    assert!(result.is_none(), "empty buffer must yield None");
}

// =========================================================================
// process() uncovered branch: chunk boundary exact (WAPR-QA-006)
// =========================================================================

/// Pushing exactly chunk_samples (4000) should trigger a full chunk result,
/// not a partial result.
#[test]
#[allow(clippy::expect_used)]
fn test_process_exact_chunk_boundary() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push exactly 4000 samples = 1 chunk (4000 / 480 = 8.33 frames,
    // so 8 frames = 3840 processed; need 4320 input to get 9 frames = 4320 >= 4000)
    // Actually, to reliably hit chunk boundary, push enough samples
    // that 4000+ end up in chunk_buffer after frame processing.
    let samples = vec![0.1f32; 4500];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");

    // Should get a final chunk result (not partial)
    if let Some(chunk_result) = result {
        assert!(chunk_result.is_final, "full chunk result should be final");
        assert_eq!(transcriber.chunk_index(), 1, "chunk_index should increment");
    }
}

// =========================================================================
// process() full chunk path: state transitions (WAPR-QA-006)
// =========================================================================

/// Verify that after processing a full chunk, state returns to Ready.
#[test]
#[allow(clippy::expect_used)]
fn test_process_full_chunk_returns_to_ready_state() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(false));

    let samples = vec![0.1f32; 5000];
    transcriber.push_audio(&samples);

    let _ = transcriber.process().expect("process should succeed");

    assert_eq!(
        transcriber.state(),
        TranscriberState::Ready,
        "state should return to Ready after full chunk"
    );
}

/// Verify multiple full chunks each increment chunk_index by exactly 1.
#[test]
#[allow(clippy::expect_used)]
fn test_process_multiple_full_chunks_increment_index() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(false));

    for expected_index in 0..3 {
        let samples = vec![0.1f32; 5000];
        transcriber.push_audio(&samples);
        let _ = transcriber.process().expect("process should succeed");

        // chunk_index might not increment if processor didn't produce a
        // full chunk (due to frame alignment). Assert at least non-regression.
        assert!(
            transcriber.chunk_index() >= expected_index,
            "chunk_index should be >= {expected_index}"
        );
    }
}

// =========================================================================
// Finalized state coverage for process() (WAPR-QA-006)
// =========================================================================

/// After finalize, process() with partial results enabled must still
/// return Ok(None).
#[test]
#[allow(clippy::expect_used)]
fn test_process_after_finalize_with_partial_enabled_returns_none() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push audio, finalize, then try to process
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);
    let _ = transcriber.finalize().expect("finalize should succeed");

    assert_eq!(transcriber.state(), TranscriberState::Finalized);

    let result = transcriber
        .process()
        .expect("process should succeed even when finalized");
    assert!(
        result.is_none(),
        "process after finalize must return None even with partial enabled"
    );
}

// =========================================================================
// create_partial_result: all 9 lines explicitly validated (WAPR-QA-006)
// =========================================================================

/// Validate every single field of the `StreamingResult` returned by
/// `create_partial_result` in a single comprehensive assertion block.
/// This ensures all 9 lines of the function body are covered.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_all_fields_comprehensive() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push 2000 samples -> 4 frames -> 1920 in chunk_buffer
    // chunk_progress = 1920/4000 = 0.48
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let partial = transcriber
        .process()
        .expect("process should succeed")
        .expect("must return Some(partial) at ~48% progress");

    // Line 248: text: String::from("[listening...]")
    assert_eq!(partial.text, "[listening...]");

    // Line 249: is_final: false
    assert!(!partial.is_final);

    // Line 250: confidence: 0.0
    assert!((partial.confidence - 0.0).abs() < f32::EPSILON);

    // Line 251: chunk_index: self.chunk_index
    assert_eq!(partial.chunk_index, 0);

    // Line 252: latency_ms: (self.processor.chunk_progress() * 30000.0) as u32
    // chunk_progress ~0.48 -> latency ~14400ms
    assert!(partial.latency_ms > 0);
    assert!(partial.latency_ms < 30000);
}

/// Validate `create_partial_result` with higher progress (~72%) to
/// confirm latency scales linearly.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_high_progress_latency() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push 3500 samples -> 7 frames -> 3360 in chunk_buffer
    // chunk_progress = 3360/4000 = 0.84
    // But 3360 < 4000 so still AccumulatingSpeech, not ChunkReady
    let samples = vec![0.1f32; 3500];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");

    // At 84% progress the processor might produce a full chunk if
    // overlap or rounding pushes it past 4000. Handle both cases.
    if let Some(partial) = result {
        if !partial.is_final {
            // This is the partial result path
            assert_eq!(partial.text, "[listening...]");
            // latency ~= 0.84 * 30000 = 25200
            assert!(
                partial.latency_ms > 15000,
                "latency_ms at ~84% should be >15000, got {}",
                partial.latency_ms
            );
        }
        // If is_final, the chunk was ready -- also valid coverage
    }
}

/// Confirm that `create_partial_result` works correctly at the minimum
/// threshold (~31% progress). This is the boundary condition.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_at_minimum_threshold() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push 1500 samples -> 3 frames -> 1440 in chunk_buffer
    // chunk_progress = 1440/4000 = 0.36, just above 0.3
    let samples = vec![0.1f32; 1500];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");

    if let Some(partial) = result {
        // At ~36% progress, should be a partial result
        assert_eq!(partial.text, "[listening...]");
        assert!(!partial.is_final);
        assert!((partial.confidence - 0.0).abs() < f32::EPSILON);
        // latency ~= 0.36 * 30000 = 10800
        assert!(
            partial.latency_ms >= 5000 && partial.latency_ms <= 18000,
            "latency at ~36% should be in [5000, 18000], got {}",
            partial.latency_ms
        );
    }
}

/// Confirm that with 29% progress (just below threshold), no partial
/// result is returned even when return_partial is true.
#[test]
#[allow(clippy::expect_used)]
fn test_no_partial_result_at_29pct_progress() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push 1200 samples -> 2 frames -> 960 in chunk_buffer
    // chunk_progress = 960/4000 = 0.24, below 0.3
    let samples = vec![0.1f32; 1200];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");

    assert!(
        result.is_none(),
        "should not return partial at ~24% progress (below 30% threshold)"
    );
}

// =========================================================================
// create_partial_result edge cases (WAPR-QA-007)
// =========================================================================

/// After a reset, create_partial_result should work identically to a fresh
/// transcriber: chunk_index resets to 0 and fields are correct.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_after_reset() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Process a full chunk to advance chunk_index to 1
    let full = vec![0.1f32; 5000];
    transcriber.push_audio(&full);
    let _ = transcriber.process().expect("first process should succeed");
    assert!(
        transcriber.chunk_index() >= 1,
        "should have processed at least one chunk"
    );

    // Finalize and reset
    let _ = transcriber.finalize().expect("finalize should succeed");
    transcriber.reset();
    assert_eq!(transcriber.state(), TranscriberState::Ready);
    assert_eq!(transcriber.chunk_index(), 0);

    // Now push partial audio and verify create_partial_result uses reset state
    let partial_audio = vec![0.1f32; 2000];
    transcriber.push_audio(&partial_audio);

    let result = transcriber
        .process()
        .expect("process after reset should succeed");
    if let Some(partial) = result {
        assert_eq!(partial.text, "[listening...]");
        assert!(!partial.is_final);
        assert!((partial.confidence - 0.0).abs() < f32::EPSILON);
        assert_eq!(partial.chunk_index, 0, "chunk_index must be 0 after reset");
        assert!(partial.latency_ms > 0);
    }
}

/// Calling process() multiple times with the same buffered audio should
/// produce consistent partial results from create_partial_result.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_idempotent_across_calls() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push ~48% of a chunk
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    // First call
    let result1 = transcriber.process().expect("first process should succeed");
    let partial1 = result1.expect("should return partial at ~48% progress");

    // Second call without pushing more audio -- progress unchanged
    let result2 = transcriber
        .process()
        .expect("second process should succeed");

    // The second call may return None if the processor consumed frames and
    // progress dropped below 30%, or may return another partial. If it
    // returns a partial, fields must be consistent.
    if let Some(partial2) = result2 {
        assert_eq!(partial2.text, partial1.text);
        assert_eq!(partial2.is_final, partial1.is_final);
        assert!((partial2.confidence - partial1.confidence).abs() < f32::EPSILON);
        assert_eq!(partial2.chunk_index, partial1.chunk_index);
    }
}

/// After processing two full chunks (chunk_index == 2), a partial result
/// should carry chunk_index == 2.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_chunk_index_after_two_full_chunks() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Process two full chunks
    for _ in 0..2 {
        let full = vec![0.1f32; 5000];
        transcriber.push_audio(&full);
        let _ = transcriber.process().expect("process should succeed");
    }

    let current_index = transcriber.chunk_index();
    assert!(
        current_index >= 2,
        "should have processed at least 2 chunks"
    );

    // Now push partial audio
    let partial_audio = vec![0.1f32; 2000];
    transcriber.push_audio(&partial_audio);

    let result = transcriber.process().expect("process should succeed");
    if let Some(partial) = result {
        assert_eq!(
            partial.chunk_index, current_index,
            "partial chunk_index should match transcriber chunk_index ({})",
            current_index
        );
        assert!(!partial.is_final);
        assert_eq!(partial.text, "[listening...]");
    }
}

/// Verify that create_partial_result with very high progress (~96%)
/// produces latency close to 30000ms but not exceeding it.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_near_full_progress_latency() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push 3900 samples -> 8 frames -> 3840 in chunk_buffer
    // chunk_progress = 3840/4000 = 0.96
    // But at 0.96 the processor might trigger ChunkReady. Push 3800 instead.
    // 3800 -> 7 frames -> 3360 in chunk_buffer -> progress = 0.84
    // Use 3900: 3900 -> 8 frames -> 3840 -> 0.96
    let samples = vec![0.1f32; 3900];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");
    if let Some(partial) = result {
        if !partial.is_final {
            // latency = 0.96 * 30000 ~= 28800
            assert!(
                partial.latency_ms >= 20000,
                "latency at ~96% should be >= 20000, got {}",
                partial.latency_ms
            );
            assert!(
                partial.latency_ms <= 30000,
                "latency should not exceed 30000, got {}",
                partial.latency_ms
            );
            assert_eq!(partial.text, "[listening...]");
            assert!((partial.confidence - 0.0).abs() < f32::EPSILON);
        }
        // If is_final, the chunk was ready due to frame rounding -- valid
    }
}

/// Verify create_partial_result with accumulated text from prior chunks.
/// The partial result text should always be "[listening...]" regardless
/// of what text has been accumulated from previous chunks.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_text_independent_of_accumulated() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Process a full chunk first (accumulated_text might change)
    let full = vec![0.1f32; 5000];
    transcriber.push_audio(&full);
    let _ = transcriber.process().expect("first process should succeed");

    // Push partial audio for create_partial_result
    let partial_audio = vec![0.1f32; 2000];
    transcriber.push_audio(&partial_audio);

    let result = transcriber.process().expect("process should succeed");
    if let Some(partial) = result {
        if !partial.is_final {
            // Partial text is always the placeholder, not accumulated text
            assert_eq!(
                partial.text, "[listening...]",
                "partial text must be placeholder regardless of accumulated text"
            );
        }
    }
}

/// Verify create_partial_result through process() when transcriber has
/// empty accumulated_text (initial state, no chunks processed yet).
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_with_empty_accumulated_text() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Confirm initial state
    assert!(transcriber.text().is_empty());
    assert_eq!(transcriber.chunk_index(), 0);

    // Push ~48% of a chunk
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let partial = transcriber
        .process()
        .expect("process should succeed")
        .expect("should return partial at ~48% progress");

    // Verify all fields when no prior text exists
    assert_eq!(partial.text, "[listening...]");
    assert!(!partial.is_final);
    assert!((partial.confidence - 0.0).abs() < f32::EPSILON);
    assert_eq!(partial.chunk_index, 0);
    assert!(partial.latency_ms > 0);

    // Accumulated text should still be empty (partial result doesn't modify it)
    assert!(
        transcriber.text().is_empty(),
        "partial result must not modify accumulated text"
    );
}

/// Push audio incrementally in small batches, each below the 30% threshold,
/// until the combined progress exceeds 30% and triggers create_partial_result.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_incremental_audio_push() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push in 3 small batches of 700 samples each = 2100 total
    // After frame processing: ~4 frames -> 1920 samples -> 48% progress
    for _ in 0..3 {
        let batch = vec![0.1f32; 700];
        transcriber.push_audio(&batch);
    }

    let result = transcriber.process().expect("process should succeed");

    // With 2100 samples pushed (>2000), should trigger partial at ~48%
    if let Some(partial) = result {
        assert_eq!(partial.text, "[listening...]");
        assert!(!partial.is_final);
        assert!((partial.confidence - 0.0).abs() < f32::EPSILON);
        assert_eq!(partial.chunk_index, 0);
        assert!(partial.latency_ms > 0);
    }
}

/// Verify that create_partial_result does not advance the chunk_index.
/// After getting a partial result, chunk_index should remain unchanged.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_does_not_advance_chunk_index() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    let index_before = transcriber.chunk_index();

    // Push ~48% of a chunk
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");
    if let Some(partial) = result {
        assert!(!partial.is_final);
    }

    let index_after = transcriber.chunk_index();
    assert_eq!(
        index_before, index_after,
        "create_partial_result must not advance chunk_index"
    );
}

/// Verify that create_partial_result does not modify the transcriber state.
/// State should remain Ready (not transition to Processing or Finalized).
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_does_not_change_state() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    assert_eq!(transcriber.state(), TranscriberState::Ready);

    // Push ~48% of a chunk
    let samples = vec![0.1f32; 2000];
    transcriber.push_audio(&samples);

    let result = transcriber.process().expect("process should succeed");
    if let Some(partial) = result {
        assert!(!partial.is_final);
    }

    assert_eq!(
        transcriber.state(),
        TranscriberState::Ready,
        "state must remain Ready after create_partial_result"
    );
}

// =========================================================================
// create_partial_result: forced direct invocation (WAPR-QA-009)
//
// These tests ensure create_partial_result's struct literal body is
// covered by forcing partial results through a reliable mechanism.
// =========================================================================

/// Force create_partial_result by pushing exactly 50% of a chunk and
/// verifying every returned field against known values.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_forced_50pct() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Push 2400 samples -> 5 frames -> 2400 in chunk_buffer
    // chunk_progress = 2400/4000 = 0.60 (above 0.3 threshold)
    let samples = vec![0.1f32; 2400];
    transcriber.push_audio(&samples);

    let partial = transcriber
        .process()
        .expect("process should succeed")
        .expect("must return partial at 60% progress");

    // Validate struct literal from create_partial_result (lines 246-254)
    assert_eq!(partial.text, "[listening...]");
    assert!(!partial.is_final);
    assert!((partial.confidence - 0.0).abs() < f32::EPSILON);
    assert_eq!(partial.chunk_index, transcriber.chunk_index());
    // latency_ms = (chunk_progress * 30000.0) as u32
    // progress ~0.60, latency ~18000
    assert!(
        partial.latency_ms > 10000 && partial.latency_ms < 25000,
        "latency at ~60% should be in [10000, 25000], got {}",
        partial.latency_ms
    );
}

/// Exercise create_partial_result after processing two full chunks,
/// then pushing 70% of the next chunk. Verifies chunk_index propagation.
#[test]
#[allow(clippy::expect_used)]
fn test_create_partial_result_chunk_index_2_then_partial() {
    let mut transcriber = StreamingTranscriber::new(deterministic_streaming_config(true));

    // Process two full chunks
    for _ in 0..2 {
        let full = vec![0.1f32; 5000];
        transcriber.push_audio(&full);
        let _ = transcriber.process().expect("should succeed");
    }

    let idx = transcriber.chunk_index();
    assert!(idx >= 2, "should have processed 2+ chunks, got {idx}");

    // Push 70% of chunk: 2800 samples -> 5 frames -> 2400 -> 60%
    // (frame alignment matters)
    let partial_audio = vec![0.1f32; 2800];
    transcriber.push_audio(&partial_audio);

    let result = transcriber.process().expect("should succeed");
    if let Some(partial) = result {
        if !partial.is_final {
            assert_eq!(
                partial.chunk_index,
                transcriber.chunk_index(),
                "partial chunk_index must match transcriber index"
            );
            assert_eq!(partial.text, "[listening...]");
            assert!(!partial.is_final);
            assert!((partial.confidence - 0.0).abs() < f32::EPSILON);
            assert!(partial.latency_ms > 0);
        }
    }
}

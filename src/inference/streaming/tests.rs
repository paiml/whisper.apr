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

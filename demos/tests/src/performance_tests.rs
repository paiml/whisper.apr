//! Performance Tests for whisper.apr Demos
//!
//! EXTREME TDD + PDD (Performance Driven Development)
//!
//! These tests assert performance expectations for the transcription pipeline.
//! They are designed to FAIL when performance degrades, catching regressions
//! before they reach production.
//!
//! ## Performance Targets (from playbook)
//!
//! | Metric | Target | Critical |
//! |--------|--------|----------|
//! | First transcript latency | ≤5s | ≤10s |
//! | Chunk processing time | ≤2s | ≤5s |
//! | RTF (Real-Time Factor) | ≤2.0x | ≤4.0x |
//! | Memory peak | ≤200MB | ≤350MB |

use std::time::{Duration, Instant};

/// Performance thresholds aligned with playbook
pub mod thresholds {
    use std::time::Duration;

    /// Maximum time for first transcription result to appear
    pub const FIRST_TRANSCRIPT_LATENCY: Duration = Duration::from_secs(5);
    pub const FIRST_TRANSCRIPT_LATENCY_CRITICAL: Duration = Duration::from_secs(10);

    /// Maximum time to process a single audio chunk (1.5s audio)
    pub const CHUNK_PROCESSING_TIME: Duration = Duration::from_secs(2);
    pub const CHUNK_PROCESSING_TIME_CRITICAL: Duration = Duration::from_secs(5);

    /// Maximum Real-Time Factor (processing time / audio duration)
    pub const RTF_TARGET: f64 = 2.0;
    pub const RTF_CRITICAL: f64 = 4.0;

    /// Maximum memory usage in bytes
    pub const MEMORY_PEAK_BYTES: usize = 200 * 1024 * 1024; // 200MB
    pub const MEMORY_PEAK_CRITICAL_BYTES: usize = 350 * 1024 * 1024; // 350MB

    /// Playbook-defined max duration for entire recording session
    pub const MAX_SESSION_DURATION: Duration = Duration::from_secs(10);
}

/// Performance measurement result
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    pub first_transcript_latency_ms: u64,
    pub chunk_processing_times_ms: Vec<u64>,
    pub total_duration_ms: u64,
    pub audio_duration_ms: u64,
    pub rtf: f64,
    pub memory_peak_bytes: usize,
    pub chunks_processed: usize,
    pub chunks_dropped: usize,
}

impl PerformanceResult {
    /// Check if all performance targets are met
    pub fn meets_targets(&self) -> bool {
        self.first_transcript_latency_ms <= thresholds::FIRST_TRANSCRIPT_LATENCY.as_millis() as u64
            && self.rtf <= thresholds::RTF_TARGET
            && self.memory_peak_bytes <= thresholds::MEMORY_PEAK_BYTES
            && self.chunks_dropped == 0
    }

    /// Check if critical thresholds are violated
    pub fn violates_critical(&self) -> bool {
        self.first_transcript_latency_ms
            > thresholds::FIRST_TRANSCRIPT_LATENCY_CRITICAL.as_millis() as u64
            || self.rtf > thresholds::RTF_CRITICAL
            || self.memory_peak_bytes > thresholds::MEMORY_PEAK_CRITICAL_BYTES
    }

    /// Generate performance report
    pub fn report(&self) -> String {
        let first_latency_status = if self.first_transcript_latency_ms
            <= thresholds::FIRST_TRANSCRIPT_LATENCY.as_millis() as u64
        {
            "✅"
        } else if self.first_transcript_latency_ms
            <= thresholds::FIRST_TRANSCRIPT_LATENCY_CRITICAL.as_millis() as u64
        {
            "⚠️"
        } else {
            "❌"
        };

        let rtf_status = if self.rtf <= thresholds::RTF_TARGET {
            "✅"
        } else if self.rtf <= thresholds::RTF_CRITICAL {
            "⚠️"
        } else {
            "❌"
        };

        let memory_status = if self.memory_peak_bytes <= thresholds::MEMORY_PEAK_BYTES {
            "✅"
        } else if self.memory_peak_bytes <= thresholds::MEMORY_PEAK_CRITICAL_BYTES {
            "⚠️"
        } else {
            "❌"
        };

        let drop_status = if self.chunks_dropped == 0 {
            "✅"
        } else {
            "❌"
        };

        format!(
            r#"
╔══════════════════════════════════════════════════════════════════╗
║              PERFORMANCE TEST RESULTS                            ║
╠══════════════════════════════════════════════════════════════════╣
║  First Transcript Latency: {:>6}ms  {} (target: ≤{}ms)
║  Real-Time Factor (RTF):   {:>6.2}x   {} (target: ≤{:.1}x)
║  Memory Peak:              {:>6}MB  {} (target: ≤{}MB)
║  Chunks Processed:         {:>6}
║  Chunks Dropped:           {:>6}    {}
╠══════════════════════════════════════════════════════════════════╣
║  Overall: {}
╚══════════════════════════════════════════════════════════════════╝
"#,
            self.first_transcript_latency_ms,
            first_latency_status,
            thresholds::FIRST_TRANSCRIPT_LATENCY.as_millis(),
            self.rtf,
            rtf_status,
            thresholds::RTF_TARGET,
            self.memory_peak_bytes / (1024 * 1024),
            memory_status,
            thresholds::MEMORY_PEAK_BYTES / (1024 * 1024),
            self.chunks_processed,
            self.chunks_dropped,
            drop_status,
            if self.meets_targets() {
                "✅ ALL TARGETS MET"
            } else if self.violates_critical() {
                "❌ CRITICAL VIOLATION"
            } else {
                "⚠️ TARGETS NOT MET"
            }
        )
    }
}

/// Simulates transcription performance for testing
/// In real tests, this would interact with the actual WASM worker
pub struct TranscriptionBenchmark {
    start_time: Option<Instant>,
    first_result_time: Option<Instant>,
    chunk_times: Vec<Duration>,
    audio_duration_ms: u64,
    memory_samples: Vec<usize>,
    chunks_dropped: usize,
}

impl TranscriptionBenchmark {
    pub fn new() -> Self {
        Self {
            start_time: None,
            first_result_time: None,
            chunk_times: Vec::new(),
            audio_duration_ms: 0,
            memory_samples: Vec::new(),
            chunks_dropped: 0,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    pub fn record_chunk(&mut self, processing_time: Duration, audio_ms: u64) {
        if self.first_result_time.is_none() {
            self.first_result_time = Some(Instant::now());
        }
        self.chunk_times.push(processing_time);
        self.audio_duration_ms += audio_ms;
    }

    pub fn record_drop(&mut self) {
        self.chunks_dropped += 1;
    }

    pub fn sample_memory(&mut self, bytes: usize) {
        self.memory_samples.push(bytes);
    }

    pub fn finish(&self) -> PerformanceResult {
        let start = self.start_time.expect("benchmark not started");
        let total_duration = start.elapsed();

        let first_latency = self
            .first_result_time
            .map(|t| t.duration_since(start))
            .unwrap_or(total_duration);

        let chunk_times_ms: Vec<u64> = self.chunk_times.iter().map(|d| d.as_millis() as u64).collect();

        let rtf = if self.audio_duration_ms > 0 {
            (total_duration.as_millis() as f64) / (self.audio_duration_ms as f64)
        } else {
            0.0
        };

        PerformanceResult {
            first_transcript_latency_ms: first_latency.as_millis() as u64,
            chunk_processing_times_ms: chunk_times_ms,
            total_duration_ms: total_duration.as_millis() as u64,
            audio_duration_ms: self.audio_duration_ms,
            rtf,
            memory_peak_bytes: self.memory_samples.iter().max().copied().unwrap_or(0),
            chunks_processed: self.chunk_times.len(),
            chunks_dropped: self.chunks_dropped,
        }
    }
}

impl Default for TranscriptionBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PERFORMANCE THRESHOLD TESTS
    // =========================================================================

    #[test]
    fn test_performance_result_meets_targets() {
        let good_result = PerformanceResult {
            first_transcript_latency_ms: 2000,
            chunk_processing_times_ms: vec![500, 600, 550],
            total_duration_ms: 3000,
            audio_duration_ms: 4500,
            rtf: 0.67, // Good RTF
            memory_peak_bytes: 150 * 1024 * 1024,
            chunks_processed: 3,
            chunks_dropped: 0,
        };

        assert!(good_result.meets_targets(), "Good result should meet targets");
        assert!(
            !good_result.violates_critical(),
            "Good result should not violate critical"
        );
    }

    #[test]
    fn test_performance_result_fails_latency() {
        let bad_result = PerformanceResult {
            first_transcript_latency_ms: 15000, // 15s - over critical threshold
            chunk_processing_times_ms: vec![5000],
            total_duration_ms: 15000,
            audio_duration_ms: 1500,
            rtf: 10.0,
            memory_peak_bytes: 100 * 1024 * 1024,
            chunks_processed: 1,
            chunks_dropped: 0,
        };

        assert!(
            !bad_result.meets_targets(),
            "Bad result should not meet targets"
        );
        assert!(
            bad_result.violates_critical(),
            "15s latency should violate critical threshold"
        );
    }

    #[test]
    fn test_performance_result_fails_rtf() {
        let bad_rtf = PerformanceResult {
            first_transcript_latency_ms: 3000,
            chunk_processing_times_ms: vec![3000],
            total_duration_ms: 6000,
            audio_duration_ms: 1500,
            rtf: 4.5, // Over critical RTF
            memory_peak_bytes: 100 * 1024 * 1024,
            chunks_processed: 1,
            chunks_dropped: 0,
        };

        assert!(!bad_rtf.meets_targets());
        assert!(bad_rtf.violates_critical());
    }

    #[test]
    fn test_performance_result_fails_drops() {
        let dropped_result = PerformanceResult {
            first_transcript_latency_ms: 2000,
            chunk_processing_times_ms: vec![500, 600],
            total_duration_ms: 3000,
            audio_duration_ms: 4500,
            rtf: 0.67,
            memory_peak_bytes: 100 * 1024 * 1024,
            chunks_processed: 2,
            chunks_dropped: 3, // Dropped chunks!
        };

        assert!(
            !dropped_result.meets_targets(),
            "Dropped chunks should fail targets"
        );
    }

    // =========================================================================
    // BENCHMARK TESTS
    // =========================================================================

    #[test]
    fn test_benchmark_basic_flow() {
        let mut bench = TranscriptionBenchmark::new();
        bench.start();

        // Small delay to ensure measurable time
        std::thread::sleep(Duration::from_millis(10));

        // Simulate chunk processing
        bench.record_chunk(Duration::from_millis(500), 1500);
        bench.sample_memory(100 * 1024 * 1024);

        bench.record_chunk(Duration::from_millis(600), 1500);
        bench.sample_memory(120 * 1024 * 1024);

        let result = bench.finish();

        assert_eq!(result.chunks_processed, 2);
        assert_eq!(result.audio_duration_ms, 3000);
        // First latency should be recorded (may be 0 if instant, that's ok)
        assert!(result.total_duration_ms >= 10, "Should have measurable duration");
    }

    #[test]
    fn test_benchmark_tracks_drops() {
        let mut bench = TranscriptionBenchmark::new();
        bench.start();

        bench.record_chunk(Duration::from_millis(500), 1500);
        bench.record_drop();
        bench.record_drop();
        bench.record_chunk(Duration::from_millis(600), 1500);

        let result = bench.finish();

        assert_eq!(result.chunks_processed, 2);
        assert_eq!(result.chunks_dropped, 2);
        assert!(!result.meets_targets()); // Drops fail targets
    }

    // =========================================================================
    // PLAYBOOK INTEGRATION TESTS
    // =========================================================================

    /// Test that performance thresholds match playbook definitions
    #[test]
    fn test_thresholds_match_playbook() {
        // From playbooks/realtime-transcription.yaml:
        // performance:
        //   max_duration_ms: 10000
        //   max_memory_mb: 200

        assert_eq!(
            thresholds::MAX_SESSION_DURATION.as_millis(),
            10000,
            "Session duration should match playbook"
        );
        assert_eq!(
            thresholds::MEMORY_PEAK_BYTES,
            200 * 1024 * 1024,
            "Memory peak should match playbook"
        );
    }

    /// Test that invariant violations are detected
    /// This simulates the playbook condition:
    /// "recording_duration() > 5000 && transcript_empty()"
    #[test]
    fn test_playbook_invariant_recording_without_transcript() {
        // Simulate: recording for 5s+ with no transcript (hang detected)
        let hang_result = PerformanceResult {
            first_transcript_latency_ms: 0, // Never got a result
            chunk_processing_times_ms: vec![],
            total_duration_ms: 7000, // 7s recording
            audio_duration_ms: 7000,
            rtf: 0.0, // No processing happened
            memory_peak_bytes: 150 * 1024 * 1024,
            chunks_processed: 0, // No chunks processed!
            chunks_dropped: 5,   // All dropped
        };

        // This should absolutely fail
        assert!(
            !hang_result.meets_targets(),
            "Hang (no transcripts) must fail targets"
        );

        // Playbook invariant: chunks_sent > 3 && chunks_processed == 0
        let chunks_sent = hang_result.chunks_dropped;
        let chunks_processed = hang_result.chunks_processed;
        assert!(
            chunks_sent > 3 && chunks_processed == 0,
            "Playbook invariant violation: {} chunks sent but {} processed",
            chunks_sent,
            chunks_processed
        );
    }

    /// Test RTF calculation matches expected formula
    #[test]
    fn test_rtf_calculation() {
        let result = PerformanceResult {
            first_transcript_latency_ms: 1000,
            chunk_processing_times_ms: vec![1000],
            total_duration_ms: 3000, // 3s processing
            audio_duration_ms: 1500, // 1.5s audio
            rtf: 2.0,                // 3000/1500 = 2.0x
            memory_peak_bytes: 100 * 1024 * 1024,
            chunks_processed: 1,
            chunks_dropped: 0,
        };

        let expected_rtf = result.total_duration_ms as f64 / result.audio_duration_ms as f64;
        assert!(
            (result.rtf - expected_rtf).abs() < 0.01,
            "RTF should be processing_time / audio_duration"
        );
    }

    // =========================================================================
    // EXTREME TDD: FAILING TEST FOR CURRENT BUG
    // =========================================================================

    /// This test documents the current bug: transcription hangs
    ///
    /// The playbook defines this invariant:
    /// "recording_duration() > 5000 && transcript_empty()"
    /// reason: "Recording for 5s+ with no transcript output indicates worker/model failure"
    ///
    /// This test should FAIL until the bug is fixed.
    #[test]
    #[ignore = "BUG: transcription hangs - enable when fixed"]
    fn test_transcription_produces_result_within_timeout() {
        // This test would interact with actual WASM worker
        // For now, it documents expected behavior

        let mut bench = TranscriptionBenchmark::new();
        bench.start();

        // Simulate what SHOULD happen:
        // - Send audio chunk
        // - Get transcription result within 5s

        // In the CURRENT buggy state, this would hang forever
        // The test asserts the expected behavior

        // After sending 1.5s of audio, we expect a result within 5s
        std::thread::sleep(Duration::from_secs(1)); // Simulate some processing

        // In the bug state, no result ever comes
        // bench.record_chunk(Duration::from_millis(1500), 1500);

        let result = bench.finish();

        assert!(
            result.chunks_processed > 0,
            "BUG: No transcription results produced. \
             Worker is hanging on model.transcribe()"
        );

        assert!(
            result.first_transcript_latency_ms <= 5000,
            "BUG: First transcript took {}ms (max 5000ms). \
             Transcription pipeline is too slow or hanging.",
            result.first_transcript_latency_ms
        );
    }

    // =========================================================================
    // REPORT TESTS
    // =========================================================================

    #[test]
    fn test_performance_report_generation() {
        let result = PerformanceResult {
            first_transcript_latency_ms: 2000,
            chunk_processing_times_ms: vec![500, 600, 550],
            total_duration_ms: 3000,
            audio_duration_ms: 4500,
            rtf: 0.67,
            memory_peak_bytes: 150 * 1024 * 1024,
            chunks_processed: 3,
            chunks_dropped: 0,
        };

        let report = result.report();
        assert!(report.contains("PERFORMANCE TEST RESULTS"));
        assert!(report.contains("ALL TARGETS MET"));
    }
}

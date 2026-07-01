//! Shared profiling data types used across sweep, run, and summary modules.

/// BrickProfiler category breakdown per run (WAPR-PROFILE-001 Gap 1)
#[derive(Debug, Clone)]
pub(super) struct BrickDetail {
    pub norm_ms: f64,
    pub attn_ms: f64,
    pub ffn_ms: f64,
    pub other_ms: f64,
    pub page_faults_minor: u64,
    pub page_faults_major: u64,
    /// Bottleneck diagnosis (0=insufficient, 1=memory, 2=compute, 3=throttled, 4=balanced)
    pub ln_bottleneck: u8,
    pub attn_bottleneck: u8,
    pub ffn_bottleneck: u8,
    /// Cycles per element (frequency-invariant)
    pub ln_cycles_per_elem: f64,
    pub attn_cycles_per_elem: f64,
    pub ffn_cycles_per_elem: f64,
    /// WAPR-PROFILE-001 Gap 4: BLIS GEMM hierarchy stats
    pub blis_total_gflops: f64,
    pub blis_macro_gflops: f64,
    pub blis_micro_gflops: f64,
    pub blis_pack_pct: f64,
    pub blis_macro_calls: u64,
    /// WAPR-PROFILE-001 Gap 2: Roofline classification
    pub roofline_bound: &'static str,
    pub roofline_util_pct: f64,
}

/// Timing data for a single profiling run
pub(super) struct ProfileRun {
    pub mel_ms: f64,
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub rtf: f64,
    pub token_count: usize,
    pub text: String,
    /// BrickProfiler category breakdown (WAPR-PROFILE-001 Gap 1)
    pub brick_detail: Option<BrickDetail>,
    /// WAPR-PROFILE-001 Gap 5: Structured Chrome Trace JSON from InferenceTracer
    pub trace_json: Option<String>,
}

/// Averaged BrickProfiler category breakdown
#[derive(Debug, Clone)]
pub(super) struct AvgBrickDetail {
    pub norm_ms: f64,
    pub attn_ms: f64,
    pub ffn_ms: f64,
    pub other_ms: f64,
    pub page_faults_minor: u64,
    pub page_faults_major: u64,
    pub ln_bottleneck: u8,
    pub attn_bottleneck: u8,
    pub ffn_bottleneck: u8,
    pub ln_cycles_per_elem: f64,
    pub attn_cycles_per_elem: f64,
    pub ffn_cycles_per_elem: f64,
    /// WAPR-PROFILE-001 Gap 4: BLIS GEMM hierarchy stats
    pub blis_total_gflops: f64,
    pub blis_macro_gflops: f64,
    pub blis_micro_gflops: f64,
    pub blis_pack_pct: f64,
    pub blis_macro_calls: u64,
    /// WAPR-PROFILE-001 Gap 2: Roofline classification
    pub roofline_bound: &'static str,
    pub roofline_util_pct: f64,
}

/// Aggregated profile summary for output formatting
pub(super) struct ProfileSummary<'text> {
    pub load_ms: f64,
    pub avg_mel: f64,
    pub avg_enc: f64,
    pub avg_dec: f64,
    pub avg_total: f64,
    pub avg_rtf: f64,
    pub avg_tokens: usize,
    pub text: &'text str,
    pub audio_duration_s: f64,
    /// BrickProfiler category breakdown averaged across runs (WAPR-PROFILE-001 Gap 1)
    pub avg_brick_detail: Option<AvgBrickDetail>,
    /// WAPR-PROFILE-001 Gap 5: Structured Chrome Trace JSON from last run's InferenceTracer
    pub trace_json: Option<String>,
    /// WAPR-PROFILE-001 Gap 2: Hardware roofline info
    pub hw_cores: usize,
    pub hw_simd: String,
    pub hw_peak_gflops: f64,
    pub hw_bw_gbps: f64,
    pub hw_balance_point: f64,
}

/// Human-readable label for bottleneck diagnosis code
pub(super) fn bottleneck_label(code: u8) -> &'static str {
    match code {
        1 => "memory-bound",
        2 => "compute-bound",
        3 => "throttled",
        4 => "balanced",
        _ => "insufficient data",
    }
}

//! `ProfileSummary` formatting impls: JSON, renacer (Chrome Trace), and
//! human-readable table output.

use super::super::super::super::apr_args::AprProfileArgs;
use super::super::super::rtf_tier_label;
use super::types::{bottleneck_label, AvgBrickDetail, ProfileSummary};

impl ProfileSummary<'_> {
    pub(super) fn format_json(&self, args: &AprProfileArgs) -> String {
        // WAPR-PROFILE-001 Gap 1: BrickProfiler category breakdown in JSON
        let brick_json = self
            .avg_brick_detail
            .as_ref()
            .map(format_brick_json)
            .unwrap_or_default();
        // Gap 2: Hardware info in JSON
        let hw_json = format!(
            concat!(
                ",\"hardware\":{{",
                "\"cores\":{},\"simd\":\"{}\",\"peak_gflops\":{:.1},",
                "\"bw_gbps\":{:.1},\"balance_point\":{:.1}}}"
            ),
            self.hw_cores,
            self.hw_simd,
            self.hw_peak_gflops,
            self.hw_bw_gbps,
            self.hw_balance_point,
        );
        format!(
            concat!(
                "{{\"model\":\"{}\",\"audio\":\"{}\",\"audio_duration_s\":{:.3},",
                "\"warmup\":{},\"runs\":{},",
                "\"avg_ms\":{{\"load\":{:.1},\"mel\":{:.1},\"encode\":{:.1},",
                "\"decode\":{:.1},\"total\":{:.1}}}{}{},",
                "\"rtf\":{:.3},\"tokens\":{},\"text\":\"{}\"}}"
            ),
            args.model.display(),
            args.audio.display(),
            self.audio_duration_s,
            args.warmup,
            args.runs,
            self.load_ms,
            self.avg_mel,
            self.avg_enc,
            self.avg_dec,
            self.avg_total,
            brick_json,
            hw_json,
            self.avg_rtf,
            self.avg_tokens,
            self.text.replace('"', "\\\"")
        )
    }

    /// Format as Chrome Trace Event JSON (renacer-compatible).
    ///
    /// Produces a `traceEvents` array with duration ("X") events for each
    /// pipeline stage. Timestamps are in microseconds. Compatible with
    /// `chrome://tracing`, Perfetto UI, and `renacer --format json`.
    pub(super) fn format_renacer(&self, args: &AprProfileArgs) -> String {
        // WAPR-PROFILE-001 Gap 5: Prefer InferenceTracer's structured trace when available
        if let Some(ref trace) = self.trace_json {
            return trace.clone();
        }

        let mut events = Vec::new();
        let mut ts_us: f64 = 0.0;

        // Model load
        let load_dur = self.load_ms * 1000.0;
        events.push(format!(
            concat!(
                "{{\"name\":\"model_load\",\"cat\":\"apr_profile\",\"ph\":\"X\",",
                "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":1,",
                "\"args\":{{\"model\":\"{}\"}}}}"
            ),
            ts_us,
            load_dur,
            args.model.display()
        ));
        ts_us += load_dur;

        // Mel spectrogram
        let mel_dur = self.avg_mel * 1000.0;
        events.push(format!(
            concat!(
                "{{\"name\":\"mel_spectrogram\",\"cat\":\"apr_profile\",\"ph\":\"X\",",
                "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":1}}"
            ),
            ts_us, mel_dur
        ));
        ts_us += mel_dur;

        // Encoder (parent span) + nested BrickProfiler sub-spans
        let enc_dur = self.avg_enc * 1000.0;
        events.push(format!(
            concat!(
                "{{\"name\":\"encoder\",\"cat\":\"apr_profile\",\"ph\":\"X\",",
                "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":1}}"
            ),
            ts_us, enc_dur
        ));
        if let Some(ref bd) = self.avg_brick_detail {
            push_brick_subspans(&mut events, ts_us, enc_dur, bd);
        }
        ts_us += enc_dur;

        // Decoder
        let dec_dur = self.avg_dec * 1000.0;
        events.push(format!(
            concat!(
                "{{\"name\":\"decoder\",\"cat\":\"apr_profile\",\"ph\":\"X\",",
                "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":1,",
                "\"args\":{{\"tokens\":{},\"ms_per_token\":{:.1}}}}}"
            ),
            ts_us,
            dec_dur,
            self.avg_tokens,
            if self.avg_tokens > 0 {
                self.avg_dec / self.avg_tokens as f64
            } else {
                0.0
            }
        ));

        // Metadata event
        let meta = format!(
            concat!(
                "{{\"name\":\"process_name\",\"cat\":\"__metadata\",\"ph\":\"M\",",
                "\"ts\":0,\"pid\":1,\"tid\":0,",
                "\"args\":{{\"name\":\"apr profile ({} runs)\"}}}}"
            ),
            args.runs
        );

        format!(
            "{{\"traceEvents\":[{},{}],\"metadata\":{{\"audio\":\"{}\",\"audio_duration_s\":{:.3},\"rtf\":{:.3}}}}}",
            meta,
            events.join(","),
            args.audio.display(),
            self.audio_duration_s,
            self.avg_rtf,
        )
    }

    pub(super) fn print_table(&self, args: &AprProfileArgs) {
        print_header(self, args);
        print_pipeline_rows(self);
        if let Some(ref bd) = self.avg_brick_detail {
            print_brick_rows(self, bd);
            print_brick_diagnosis(bd);
            print_blis_rows(self, bd);
        }
        print_decoder_and_totals(self, args);
    }
}

/// Render the BrickProfiler object for JSON output.
fn format_brick_json(bd: &AvgBrickDetail) -> String {
    let total_brick = bd.norm_ms + bd.attn_ms + bd.ffn_ms + bd.other_ms;
    let pct = |ms: f64| {
        if total_brick > 0.0 {
            ms / total_brick * 100.0
        } else {
            0.0
        }
    };
    format!(
        concat!(
            ",\"brick_profile\":{{",
            "\"norm_ms\":{:.2},\"attn_ms\":{:.2},\"ffn_ms\":{:.2},\"other_ms\":{:.2},",
            "\"norm_pct\":{:.1},\"attn_pct\":{:.1},\"ffn_pct\":{:.1},",
            "\"page_faults\":{{\"minor\":{},\"major\":{}}},",
            "\"cycles_per_elem\":{{\"ln\":{:.1},\"attn\":{:.1},\"ffn\":{:.1}}},",
            "\"bottleneck\":{{\"ln\":{},\"attn\":{},\"ffn\":{}}},",
            "\"blis\":{{\"total_gflops\":{:.2},\"macro_gflops\":{:.2},",
            "\"micro_gflops\":{:.2},\"pack_pct\":{:.1},\"macro_calls\":{}}},",
            "\"roofline\":{{\"bound\":\"{}\",\"util_pct\":{:.1}}}}}"
        ),
        bd.norm_ms,
        bd.attn_ms,
        bd.ffn_ms,
        bd.other_ms,
        pct(bd.norm_ms),
        pct(bd.attn_ms),
        pct(bd.ffn_ms),
        bd.page_faults_minor,
        bd.page_faults_major,
        bd.ln_cycles_per_elem,
        bd.attn_cycles_per_elem,
        bd.ffn_cycles_per_elem,
        bd.ln_bottleneck,
        bd.attn_bottleneck,
        bd.ffn_bottleneck,
        bd.blis_total_gflops,
        bd.blis_macro_gflops,
        bd.blis_micro_gflops,
        bd.blis_pack_pct,
        bd.blis_macro_calls,
        bd.roofline_bound,
        bd.roofline_util_pct,
    )
}

/// Append BrickProfiler sub-span trace events to the Chrome Trace event list.
fn push_brick_subspans(events: &mut Vec<String>, ts_us: f64, enc_dur: f64, bd: &AvgBrickDetail) {
    let mut sub_ts = ts_us;
    // Other (conv_frontend) first
    let other_dur = bd.other_ms * 1000.0;
    if other_dur > 0.0 {
        events.push(format!(
            concat!(
                "{{\"name\":\"conv_frontend\",\"cat\":\"brick_profile\",\"ph\":\"X\",",
                "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":2}}"
            ),
            sub_ts, other_dur
        ));
        sub_ts += other_dur;
    }
    // Norm
    let norm_dur = bd.norm_ms * 1000.0;
    events.push(format!(
        concat!(
            "{{\"name\":\"norm\",\"cat\":\"brick_profile\",\"ph\":\"X\",",
            "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":2,",
            "\"args\":{{\"cycles_per_elem\":{:.1},\"bottleneck\":{}}}}}"
        ),
        sub_ts, norm_dur, bd.ln_cycles_per_elem, bd.ln_bottleneck
    ));
    sub_ts += norm_dur;
    // Attention
    let attn_dur = bd.attn_ms * 1000.0;
    events.push(format!(
        concat!(
            "{{\"name\":\"attention\",\"cat\":\"brick_profile\",\"ph\":\"X\",",
            "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":2,",
            "\"args\":{{\"cycles_per_elem\":{:.1},\"bottleneck\":{}}}}}"
        ),
        sub_ts, attn_dur, bd.attn_cycles_per_elem, bd.attn_bottleneck
    ));
    sub_ts += attn_dur;
    // FFN
    let ffn_dur = bd.ffn_ms * 1000.0;
    events.push(format!(
        concat!(
            "{{\"name\":\"ffn\",\"cat\":\"brick_profile\",\"ph\":\"X\",",
            "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":2,",
            "\"args\":{{\"cycles_per_elem\":{:.1},\"bottleneck\":{}}}}}"
        ),
        sub_ts, ffn_dur, bd.ffn_cycles_per_elem, bd.ffn_bottleneck
    ));
    // WAPR-PROFILE-001 Gap 4: BLIS GEMM hierarchy on tid 3
    if bd.blis_macro_calls > 0 {
        events.push(format!(
            concat!(
                "{{\"name\":\"blis_gemm\",\"cat\":\"blis_profile\",\"ph\":\"X\",",
                "\"ts\":{:.0},\"dur\":{:.0},\"pid\":1,\"tid\":3,",
                "\"args\":{{\"total_gflops\":{:.2},\"macro_gflops\":{:.2},",
                "\"micro_gflops\":{:.2},\"pack_pct\":{:.1},\"calls\":{}}}}}"
            ),
            ts_us,
            enc_dur,
            bd.blis_total_gflops,
            bd.blis_macro_gflops,
            bd.blis_micro_gflops,
            bd.blis_pack_pct,
            bd.blis_macro_calls,
        ));
    }
}

fn print_header(s: &ProfileSummary<'_>, args: &AprProfileArgs) {
    println!(
        "Pipeline Profile: {} runs (+ {} warmup)",
        args.runs, args.warmup
    );
    println!("  Model:    {}", args.model.display());
    println!(
        "  Audio:    {} ({:.2}s)",
        args.audio.display(),
        s.audio_duration_s
    );
    // WAPR-PROFILE-001 Gap 2: Hardware roofline info
    println!(
        "  Hardware: {} cores, {}, {:.0} GFLOP/s peak, {:.1} GB/s BW",
        s.hw_cores, s.hw_simd, s.hw_peak_gflops, s.hw_bw_gbps,
    );
    println!();
    println!("  Step          Avg (ms)    % of total");
    println!("  ────────────  ──────────  ──────────");
}

fn print_pipeline_rows(s: &ProfileSummary<'_>) {
    println!("  Model load    {:>8.1}    (excluded)", s.load_ms);
    println!(
        "  Mel spec      {:>8.1}    {:>5.1}%",
        s.avg_mel,
        s.avg_mel / s.avg_total * 100.0
    );
    println!(
        "  Encoder       {:>8.1}    {:>5.1}%",
        s.avg_enc,
        s.avg_enc / s.avg_total * 100.0
    );
}

fn print_brick_rows(s: &ProfileSummary<'_>, bd: &AvgBrickDetail) {
    let total_brick = bd.norm_ms + bd.attn_ms + bd.ffn_ms + bd.other_ms;
    let pct = |ms: f64| {
        if total_brick > 0.0 {
            ms / total_brick * 100.0
        } else {
            0.0
        }
    };
    if bd.other_ms > 0.0 {
        println!(
            "    Conv frontend {:>6.1}    {:>5.1}%",
            bd.other_ms,
            bd.other_ms / s.avg_total * 100.0
        );
    }
    println!(
        "    Norm         {:>7.1}    {:>5.1}%  ({:.0}% of encoder)",
        bd.norm_ms,
        bd.norm_ms / s.avg_total * 100.0,
        pct(bd.norm_ms)
    );
    println!(
        "    Attention    {:>7.1}    {:>5.1}%  ({:.0}% of encoder)",
        bd.attn_ms,
        bd.attn_ms / s.avg_total * 100.0,
        pct(bd.attn_ms)
    );
    println!(
        "    FFN          {:>7.1}    {:>5.1}%  ({:.0}% of encoder)",
        bd.ffn_ms,
        bd.ffn_ms / s.avg_total * 100.0,
        pct(bd.ffn_ms)
    );
}

fn print_brick_diagnosis(bd: &AvgBrickDetail) {
    println!("  ────────────  ──────────  ──────────");
    println!("  BrickProfiler Diagnosis:");
    println!(
        "    Norm:      {:.1} cyc/elem  {}",
        bd.ln_cycles_per_elem,
        bottleneck_label(bd.ln_bottleneck)
    );
    println!(
        "    Attention: {:.1} cyc/elem  {}",
        bd.attn_cycles_per_elem,
        bottleneck_label(bd.attn_bottleneck)
    );
    println!(
        "    FFN:       {:.1} cyc/elem  {}",
        bd.ffn_cycles_per_elem,
        bottleneck_label(bd.ffn_bottleneck)
    );
    if bd.page_faults_minor > 0 || bd.page_faults_major > 0 {
        println!(
            "  Page faults:  {} minor, {} major",
            bd.page_faults_minor, bd.page_faults_major
        );
    }
}

fn print_blis_rows(s: &ProfileSummary<'_>, bd: &AvgBrickDetail) {
    if bd.blis_macro_calls > 0 {
        println!("  ────────────  ──────────  ──────────");
        println!("  BLIS GEMM Hierarchy ({} calls):", bd.blis_macro_calls);
        println!("    Macro:   {:.1} GFLOP/s", bd.blis_macro_gflops);
        println!("    Micro:   {:.1} GFLOP/s", bd.blis_micro_gflops);
        println!("    Pack:    {:.1}% of GEMM time", bd.blis_pack_pct);
        println!("    Total:   {:.1} GFLOP/s", bd.blis_total_gflops);
        // Gap 2: Roofline classification
        println!(
            "    Roofline: {} ({:.1}% of {:.0} GFLOP/s peak)",
            bd.roofline_bound, bd.roofline_util_pct, s.hw_peak_gflops,
        );
    }
}

fn print_decoder_and_totals(s: &ProfileSummary<'_>, args: &AprProfileArgs) {
    println!(
        "  Decoder       {:>8.1}    {:>5.1}%",
        s.avg_dec,
        s.avg_dec / s.avg_total * 100.0
    );
    println!("  ────────────  ──────────  ──────────");
    println!("  Total         {:>8.1}    100.0%", s.avg_total);
    println!();
    println!("  RTF:    {:.2}x", s.avg_rtf);
    println!("  Tokens: {}", s.avg_tokens);
    if args.per_token && s.avg_tokens > 0 {
        println!(
            "  ms/token (decode): {:.1}",
            s.avg_dec / s.avg_tokens as f64
        );
    }
    println!("  Text:   \"{}\"", s.text.trim());
    println!("{}", rtf_tier_label(s.avg_rtf));
}

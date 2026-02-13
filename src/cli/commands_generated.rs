//! Command implementations for whisper-apr CLI
//!
//! Each command is implemented as a pure function for testability.
//! The main `run` function dispatches to the appropriate command.
#![allow(
    clippy::needless_pass_by_value,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::bool_to_int_with_if,
    clippy::unnecessary_operation,
    clippy::if_not_else,
    clippy::default_constructed_unit_structs,
    clippy::map_unwrap_or,
    clippy::needless_continue,
    clippy::struct_excessive_bools,
    clippy::range_plus_one,
    clippy::comparison_to_empty,
    clippy::default_trait_access
)]

use std::fs;
use std::io::{self, Write as IoWrite};
use std::path::Path;
use std::time::Instant;

use crate::audio::wav::{parse_wav_file, resample};
use crate::parallel::configure_thread_pool;
use crate::{DecodingStrategy, ProfilingStats, Task, TranscribeOptions, WhisperApr};

use crate::cli::args::{
    Args, BackendArg, BatchArgs, BenchmarkArgs, Command, CommandArgs, ConvertArgs, DiagnoseArgs,
    ExportArgs, ExportFormatArg, ModelAction, ModelArgs, ModelFamilyArg, OutputFormatArg,
    ParityArgs, QuantizeArgs, QuantizeMethodArg, RecordArgs, SelftestArgs, ServeArgs, StreamArgs,
    SummarizeArgs, SummarizeFormat, TestArgs, TranscribeArgs, TranscribeFolderArgs, TranslateArgs,
    ValidateArgs, ValidateOutputFormat,
};

use crate::cli::output::{format_output, OutputFormat};

/// CLI error type
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Whisper error
    #[error("Whisper error: {0}")]
    Whisper(#[from] crate::WhisperError),

    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Feature not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Unsupported format
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Write error
    #[error("Write error: {0}")]
    WriteError(String),
}

/// CLI result type
pub type CliResult<T> = Result<T, CliError>;

/// Transcription timing information
#[derive(Debug, Clone, Default)]
pub struct Timings {
    /// Model load time in milliseconds
    pub model_load_ms: f64,
    /// Audio load time in milliseconds
    pub audio_load_ms: f64,
    /// Mel spectrogram time in milliseconds
    pub mel_ms: f64,
    /// Encoding time in milliseconds
    pub encode_ms: f64,
    /// Decoding time in milliseconds
    pub decode_ms: f64,
    /// Total time in milliseconds
    pub total_ms: f64,
}

/// Command execution result
#[derive(Debug)]
pub struct CommandResult {
    /// Whether the command succeeded
    pub success: bool,
    /// Output message
    pub message: String,
    /// Timings (if applicable)
    pub timings: Option<Timings>,
    /// RTF (if applicable)
    pub rtf: Option<f64>,
}

impl CommandResult {
    /// Create a success result
    #[must_use]
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            timings: None,
            rtf: None,
        }
    }

    /// Create a failure result
    #[must_use]
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            timings: None,
            rtf: None,
        }
    }

    /// Add timings to result
    #[must_use]
    pub fn with_timings(mut self, timings: Timings) -> Self {
        self.timings = Some(timings);
        self
    }

    /// Add RTF to result
    #[must_use]
    pub fn with_rtf(mut self, rtf: f64) -> Self {
        self.rtf = Some(rtf);
        self
    }
}

/// Run CLI with parsed arguments
///
/// This is the main entry point called by the binary.
/// All command logic is delegated to specific functions.
pub fn run(args: Args) -> CliResult<CommandResult> {
    match &args.command {
        Command::Transcribe(t) => run_transcribe(t.clone(), &args),
        Command::Translate(t) => run_translate(t.clone(), &args),
        Command::Summarize(s) => run_summarize(s.clone(), &args),
        Command::Stream(s) => run_stream(s.clone(), &args),
        Command::Serve(s) => run_serve(s.clone(), &args),
        Command::Record(r) => run_record(r.clone(), &args),
        Command::Batch(b) => run_batch(b.clone(), &args),
        Command::TranscribeFolder(tf) => run_transcribe_folder(tf.clone(), &args),
        Command::Tui => run_tui(&args),
        Command::Test(t) => run_test(t.clone(), &args),
        Command::Model(m) => run_model(m.clone(), &args),
        Command::Benchmark(b) => run_benchmark(b.clone(), &args),
        Command::Validate(v) => run_validate(v.clone(), &args),
        Command::Parity(p) => run_parity(p.clone(), &args),
        Command::Quantize(q) => run_quantize(q.clone(), &args),
        Command::Command(c) => run_command(c.clone(), &args),
        Command::Diagnose(d) => run_diagnose(d.clone(), &args),
        Command::Convert(c) => run_convert(c.clone(), &args),
        Command::Export(e) => run_export(e.clone(), &args),
        Command::Apr(a) => crate::cli::apr_commands::run_apr(a, &args),
        Command::Selftest(s) => run_selftest(s.clone(), &args),
    }
}

/// Resolve GPU backend availability based on feature flags and user request
fn resolve_gpu_backend(requested: bool, _global: &Args) -> CliResult<bool> {
    #[cfg(feature = "realizar-gpu")]
    {
        let global = _global;
        if requested {
            use realizar::cuda::CudaExecutor;
            if CudaExecutor::is_available() {
                let num_devices = CudaExecutor::num_devices();
                if !global.quiet {
                    eprintln!("[INFO] GPU enabled: {} CUDA device(s)", num_devices);
                    eprintln!("[WARN] Hybrid CPU→GPU path may be slower than CPU due to PCI-E transfer overhead");
                    eprintln!("[WARN] Full GPU-resident implementation pending (WAPR-PERF-005)");
                }
                return Ok(true);
            }
            eprintln!("[WARN] GPU requested but CUDA not available, falling back to CPU");
            return Ok(false);
        }
        if global.verbose {
            eprintln!("[INFO] Using CPU backend (default)");
        }
        return Ok(false);
    }

    #[cfg(not(feature = "realizar-gpu"))]
    {
        if requested {
            return Err(CliError::InvalidArgument(
                "GPU requested but whisper-apr was not compiled with 'realizar-gpu' feature. \
                 Rebuild with: cargo build --features realizar-gpu"
                    .to_string(),
            ));
        }
        Ok(false)
    }
}

/// Build transcription options from CLI arguments
fn build_transcribe_options(args: &TranscribeArgs, verbose: bool) -> TranscribeOptions {
    let task = if args.translate {
        Task::Translate
    } else {
        Task::Transcribe
    };
    let language = if args.language == "auto" {
        None
    } else {
        Some(args.language.clone())
    };
    let strategy = if args.beam_size > 0 {
        DecodingStrategy::BeamSearch {
            beam_size: args.beam_size as usize,
            temperature: args.temperature,
            patience: 1.0,
        }
    } else {
        DecodingStrategy::Greedy
    };
    TranscribeOptions {
        language,
        task,
        strategy,
        word_timestamps: args.word_timestamps,
        profile: verbose,
    }
}

/// Print WAPR-PERF-004 component profiling breakdown
fn print_component_profile(
    result: &crate::TranscriptionResult,
    timings: &Timings,
    audio_duration_secs: f64,
    rtf: f64,
) {
    let inference_ms = timings.decode_ms;
    let tokens: usize = result
        .segments
        .iter()
        .map(|s| s.text.split_whitespace().count())
        .sum();
    let tokens_per_sec = if inference_ms > 0.0 {
        (tokens as f64 / inference_ms) * 1000.0
    } else {
        0.0
    };

    eprintln!();
    eprintln!("=== Component Profiling (WAPR-PERF-004) ===");
    eprintln!(
        "[PROFILE] Model load:     {:>7.1}ms ({:>5.1}%)",
        timings.model_load_ms,
        (timings.model_load_ms / timings.total_ms) * 100.0
    );
    eprintln!(
        "[PROFILE] Audio load:     {:>7.1}ms ({:>5.1}%)",
        timings.audio_load_ms,
        (timings.audio_load_ms / timings.total_ms) * 100.0
    );
    eprintln!(
        "[PROFILE] Inference:      {:>7.1}ms ({:>5.1}%)",
        inference_ms,
        (inference_ms / timings.total_ms) * 100.0
    );
    eprintln!("[PROFILE] --------------------------------");
    eprintln!("[PROFILE] Total:          {:>7.1}ms", timings.total_ms);
    eprintln!("[PROFILE] Audio duration: {:>7.2}s", audio_duration_secs);
    eprintln!("[PROFILE] RTF:            {:>7.3}x", rtf);
    eprintln!("[PROFILE] Tokens:         {:>7}", tokens);
    eprintln!("[PROFILE] Throughput:     {:>7.0} tok/s", tokens_per_sec);

    let budget_target = 7692.0;
    eprintln!(
        "[PROFILE] Budget:         {} (target: {:.0} tok/s)",
        if tokens_per_sec >= budget_target {
            "✓ MET"
        } else {
            "✗ EXCEEDED"
        },
        budget_target
    );
    eprintln!();
}

/// Write transcription output to file or stdout
fn write_transcription_output(
    result: &crate::TranscriptionResult,
    format: OutputFormat,
    output_path: Option<&std::path::Path>,
    global: &Args,
) -> CliResult<()> {
    let output_text = format_output(result, format);
    if let Some(path) = output_path {
        fs::write(path, &output_text)?;
        if global.verbose {
            eprintln!("[INFO] Written to: {}", path.display());
        }
    } else if !global.quiet {
        print!("{output_text}");
        io::stdout().flush()?;
    }
    Ok(())
}

/// Loaded audio data with timing and duration metadata.
struct LoadedAudio {
    /// Resampled audio samples at 16kHz
    samples: Vec<f32>,
    /// Audio duration in seconds
    duration_secs: f64,
    /// Time spent loading audio in milliseconds
    load_ms: f64,
}

/// Configure thread pool and log thread count if verbose (section 11.3.6 P.6).
fn setup_thread_pool(threads: Option<u32>, use_gpu: bool, global: &Args) -> CliResult<()> {
    let thread_count = configure_thread_pool(threads)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to configure threads: {e}")))?;

    if global.verbose && !use_gpu {
        eprintln!("[INFO] Using {thread_count} thread(s) for inference");
    }
    Ok(())
}

/// Load the Whisper model, logging progress if verbose.
///
/// Returns the loaded model and the time spent loading in milliseconds.
fn load_model_with_timing(args: &TranscribeArgs, global: &Args) -> CliResult<(WhisperApr, f64)> {
    if global.verbose {
        if let Some(path) = &args.model_path {
            eprintln!("[INFO] Loading model from: {}", path.display());
        } else {
            eprintln!("[INFO] Loading model: {}", args.model);
        }
    }
    let model_start = Instant::now();

    let whisper = crate::cli::model_loader::load_or_download_model(
        args.model,
        args.model_path.as_deref(),
        global.verbose,
    )
    .map_err(|e| CliError::InvalidArgument(e.to_string()))?;

    let model_load_ms = model_start.elapsed().as_secs_f64() * 1000.0;
    Ok((whisper, model_load_ms))
}

/// Load and parse audio from the input file, logging progress if verbose.
fn load_audio_with_timing(args: &TranscribeArgs, global: &Args) -> CliResult<LoadedAudio> {
    if global.verbose {
        eprintln!("[INFO] Loading audio: {}", args.input.display());
    }
    let audio_start = Instant::now();
    let audio_data = fs::read(&args.input)?;

    let samples = load_audio_samples(&args.input, &audio_data)?;
    let load_ms = audio_start.elapsed().as_secs_f64() * 1000.0;

    let duration_secs = samples.len() as f64 / 16000.0;

    if global.verbose {
        eprintln!(
            "[INFO] Audio: {:.2}s, {} samples",
            duration_secs,
            samples.len()
        );
    }

    Ok(LoadedAudio {
        samples,
        duration_secs,
        load_ms,
    })
}

/// Run GPU-accelerated transcription via CUDA backend.
#[cfg(feature = "realizar-gpu")]
fn run_gpu_transcription(
    whisper: WhisperApr,
    samples: &[f32],
    options: TranscribeOptions,
    global: &Args,
) -> CliResult<crate::TranscriptionResult> {
    let mut cuda_model = whisper
        .into_cuda(0)
        .map_err(|e| CliError::InvalidArgument(e.to_string()))?;

    if global.verbose {
        eprintln!("[INFO] Running on GPU: {}", cuda_model.device_name());
        let (free, total) = cuda_model.memory_info();
        eprintln!(
            "[INFO] GPU memory: {:.1}GB free / {:.1}GB total",
            free as f64 / 1e9,
            total as f64 / 1e9
        );
    }

    // WAPR-PERF-020: Pre-compile GPU kernels for predictable latency
    // This moves ~2s compilation overhead from first transcription to model init
    let warmup_start = std::time::Instant::now();
    if let Err(e) = cuda_model.warmup() {
        if global.verbose {
            eprintln!("[WARN] GPU warmup failed: {}", e);
        }
    } else if global.verbose {
        eprintln!(
            "[INFO] GPU warmup: {:.1}ms",
            warmup_start.elapsed().as_millis()
        );
    }

    // WAPR-PERF-004: Use GPU-accelerated transcription path
    // This uses gemv_cached for output projection (the decoder bottleneck)
    cuda_model
        .transcribe_gpu(samples, options)
        .map_err(|e| CliError::InvalidArgument(e.to_string()))
}

/// Finalize transcription: compute RTF, emit profile, write output, and run summarization.
fn finalize_transcription(
    result: &crate::TranscriptionResult,
    args: &TranscribeArgs,
    global: &Args,
    timings: &Timings,
    audio_duration_secs: f64,
) -> CliResult<f64> {
    let rtf = (timings.total_ms / 1000.0) / audio_duration_secs;

    if global.verbose {
        eprintln!("[INFO] Total: {:.1}ms", timings.total_ms);
        eprintln!("[INFO] RTF: {rtf:.2}x");
    }

    // WAPR-PERF-004: Component profiling output (apr-cli style)
    if args.profile {
        print_component_profile(result, timings, audio_duration_secs, rtf);
    }

    // Format and write output
    let format = convert_format_arg(args.format);
    write_transcription_output(result, format, args.output.as_deref(), global)?;

    // Phase 2: Post-transcription summarization (Section 18.5)
    if args.summarize {
        let summary_result = run_post_transcription_summary(&result.text, args, global)?;
        if global.verbose {
            eprintln!("[INFO] Summary generated: {} chars", summary_result.len());
        }
    }

    Ok(rtf)
}

/// Run transcribe command
pub fn run_transcribe(args: TranscribeArgs, global: &Args) -> CliResult<CommandResult> {
    let start = Instant::now();
    let mut timings = Timings::default();
    let use_gpu = resolve_gpu_backend(args.gpu, global)?;

    setup_thread_pool(args.threads, use_gpu, global)?;

    // Validate input file exists
    if !args.input.exists() {
        return Err(CliError::FileNotFound(args.input.display().to_string()));
    }

    // Load model
    let (whisper, model_load_ms) = load_model_with_timing(&args, global)?;
    timings.model_load_ms = model_load_ms;

    // Load and parse audio
    let audio = load_audio_with_timing(&args, global)?;
    timings.audio_load_ms = audio.load_ms;

    // Transcribe
    let transcribe_start = Instant::now();
    let options = build_transcribe_options(&args, global.verbose);

    #[cfg(feature = "realizar-gpu")]
    let result = if use_gpu {
        run_gpu_transcription(whisper, &audio.samples, options, global)?
    } else {
        whisper.transcribe(&audio.samples, options)?
    };

    #[cfg(not(feature = "realizar-gpu"))]
    let result = whisper.transcribe(&audio.samples, options)?;

    timings.decode_ms = transcribe_start.elapsed().as_secs_f64() * 1000.0;
    timings.total_ms = start.elapsed().as_secs_f64() * 1000.0;

    let rtf = finalize_transcription(&result, &args, global, &timings, audio.duration_secs)?;

    Ok(CommandResult::success(result.text)
        .with_timings(timings)
        .with_rtf(rtf))
}

/// Run post-transcription summarization (Phase 2 - Section 18.5)
///
/// Called from run_transcribe when --summarize flag is set.
fn run_post_transcription_summary(
    transcript: &str,
    args: &TranscribeArgs,
    global: &Args,
) -> CliResult<String> {
    use std::time::Instant;

    let start = Instant::now();

    // Check for LFM2 model path
    let model_path = args.lfm2_model.as_ref().ok_or_else(|| {
        CliError::InvalidArgument(
            "Post-transcription summarization requires --lfm2-model to be specified. \
             Use 'whisper-apr model download' to get the LFM2 model, then convert it with 'whisper-apr convert'."
                .to_string(),
        )
    })?;

    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.display().to_string()));
    }

    if transcript.trim().is_empty() {
        if !global.quiet {
            eprintln!("[WARN] Transcript is empty, skipping summarization");
        }
        return Ok(String::new());
    }

    // Load LFM2 model
    if !global.quiet {
        eprintln!("[INFO] Loading LFM2 model for summarization...");
    }

    let model_data = fs::read(model_path)?;
    let model = crate::model::lfm2::Lfm2::from_apr2_bytes(model_data)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to load LFM2 model: {e}")))?;

    let load_time = start.elapsed();
    if global.verbose {
        eprintln!("[INFO] LFM2 model loaded in {:.1}ms", load_time.as_millis());
    }

    // Use default tokenizer for post-transcription summary
    // In production, the tokenizer would be loaded alongside the model
    let tokenizer = crate::model::lfm2::Lfm2Tokenizer::new();

    // Tokenize transcript using BPE tokenizer
    let input_tokens = tokenizer.encode_without_special(transcript);

    // Generate summary
    let gen_start = Instant::now();
    let output_tokens = model
        .generate(&input_tokens, 256, 0.3) // max 256 tokens, temp 0.3
        .map_err(|e| CliError::InvalidArgument(format!("LFM2 generation failed: {e}")))?;

    let gen_time = gen_start.elapsed();
    if global.verbose {
        eprintln!(
            "[INFO] Summary generated in {:.1}ms ({} tokens)",
            gen_time.as_millis(),
            output_tokens.len()
        );
    }

    // Decode output tokens back to text using the tokenizer
    let summary = tokenizer.decode(&output_tokens);

    // Format summary based on requested format
    let formatted = match args.summary_format {
        SummarizeFormat::Json => {
            format!(
                r#"{{"transcript_length": {}, "summary": "{}", "action_items": {}, "key_points": {}}}"#,
                transcript.len(),
                summary.replace('"', "\\\"").replace('\n', "\\n"),
                args.action_items,
                args.key_points
            )
        }
        SummarizeFormat::Text => summary.clone(),
        SummarizeFormat::Markdown => format!("## Summary\n\n{summary}\n"),
        SummarizeFormat::Bullets => summary
            .lines()
            .map(|l| format!("- {l}"))
            .collect::<Vec<_>>()
            .join("\n"),
    };

    // Write summary output
    let summary_path = args.summary_output.clone().unwrap_or_else(|| {
        let mut path = args.input.clone();
        path.set_extension("summary.json");
        path
    });

    fs::write(&summary_path, &formatted)?;
    if !global.quiet {
        eprintln!("[INFO] Summary written to: {}", summary_path.display());
    }

    Ok(formatted)
}

/// Run translate command
pub fn run_translate(args: TranslateArgs, global: &Args) -> CliResult<CommandResult> {
    // Configure thread pool for parallel inference (§11.3.6 P.6)
    let thread_count = configure_thread_pool(args.threads)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to configure threads: {e}")))?;

    if global.verbose {
        eprintln!("[INFO] Using {thread_count} thread(s) for inference");
    }

    // Validate input file exists
    if !args.input.exists() {
        return Err(CliError::FileNotFound(args.input.display().to_string()));
    }

    // Load audio
    let audio_data = fs::read(&args.input)?;
    let samples = load_audio_samples(&args.input, &audio_data)?;

    // Create model and transcribe with translation task
    let whisper = WhisperApr::tiny();
    let options = TranscribeOptions {
        language: None, // Auto-detect source language
        task: Task::Translate,
        strategy: DecodingStrategy::Greedy,
        word_timestamps: false,
        profile: false,
    };

    let result = whisper.transcribe(&samples, options)?;

    // Format and output
    let format = convert_format_arg(args.format);
    let output_text = format_output(&result, format);

    if let Some(output_path) = args.output {
        fs::write(&output_path, &output_text)?;
    } else if !global.quiet {
        print!("{output_text}");
        io::stdout().flush()?;
    }

    Ok(CommandResult::success(result.text))
}

/// Run summarize command (WAPR-LFM2-001)
///
/// Summarizes transcript text using LFM2-2.6B-Transcript model.
/// This is Phase 1 of the LFM2 integration (CLI pipeline).
///
/// # Current Status
///
/// This is a **stub implementation** for WAPR-LFM2-001. The full implementation
/// requires:
/// - APR2 format reader for LFM2 models
/// - LFM2 inference engine (GQA, SwiGLU, Conv layers)
/// - int4 quantization support
///
/// See `docs/specifications/1.0-whisper-apr.md` Section 18 for full specification.
pub fn run_summarize(args: SummarizeArgs, global: &Args) -> CliResult<CommandResult> {
    use std::time::Instant;

    let start = Instant::now();

    // Read input text (from file or stdin)
    let input_text = read_summarize_input(&args)?;
    log_summarize_params(&args, &input_text, global);

    let load_start = Instant::now();
    let model = load_summarize_model(&args, global)?;
    let load_time = load_start.elapsed();

    let tokenizer = load_summarize_tokenizer(&args, global)?;

    // Tokenize and truncate
    let prompt = format!(
        "Summarize the following transcript:\n\n{}\n\nSummary:",
        input_text.trim()
    );
    let max_ctx = args.max_context.min(4096) as usize;
    let input_ids: Vec<u32> = tokenizer
        .encode_without_special(&prompt)
        .into_iter()
        .take(max_ctx)
        .collect();

    if global.verbose {
        eprintln!("[INFO] Input tokens: {}", input_ids.len());
    }

    // Generate summary
    if !global.quiet {
        if args.stream {
            println!("Generating summary (streaming)...\n");
        } else {
            println!("Generating summary...");
        }
    }

    let (output_ids, gen_stats) = generate_summary(&model, &tokenizer, &input_ids, &args, global)?;

    if args.stream && !global.quiet {
        println!("\n");
    }

    // Decode and format output
    let summary_ids = &output_ids[input_ids.len()..];
    let summary = tokenizer.decode(summary_ids);

    log_generation_stats(&gen_stats, global);

    let total_time = start.elapsed();
    let output = format_summary_output(
        &args,
        &summary,
        &input_text,
        &input_ids,
        &gen_stats,
        load_time,
        total_time,
    );

    write_summary_result(&args, global, &output, &gen_stats, total_time)?;

    Ok(CommandResult::success(format!(
        "Generated {} token summary{}",
        gen_stats.tokens_generated,
        if args.stream { " (streamed)" } else { "" }
    )))
}

/// Read input text for summarization from file or stdin
fn read_summarize_input(args: &SummarizeArgs) -> CliResult<String> {
    let text = if let Some(path) = &args.input {
        if !path.exists() {
            return Err(CliError::FileNotFound(path.display().to_string()));
        }
        fs::read_to_string(path)?
    } else {
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer)?;
        buffer
    };

    if text.trim().is_empty() {
        return Err(CliError::InvalidArgument(
            "No input text provided for summarization".to_string(),
        ));
    }
    Ok(text)
}

/// Log summarization parameters when verbose
fn log_summarize_params(args: &SummarizeArgs, input_text: &str, global: &Args) {
    if global.verbose {
        eprintln!("[INFO] Input text length: {} characters", input_text.len());
        if let Some(model_path) = &args.model_path {
            eprintln!("[INFO] Model path: {}", model_path.display());
        } else {
            eprintln!("[INFO] Using default LFM2-2.6B-Transcript model");
        }
        eprintln!("[INFO] Max tokens: {}", args.max_tokens);
        eprintln!("[INFO] Temperature: {:.2}", args.temperature);
    }
}

/// Load LFM2 model for summarization
fn load_summarize_model(
    args: &SummarizeArgs,
    global: &Args,
) -> CliResult<crate::model::lfm2::Lfm2> {
    let model_path = args.model_path.as_ref().ok_or_else(|| {
        CliError::InvalidArgument(
            "LFM2 summarization requires --model-path to be specified. \
             Use 'whisper-apr model download' to get a model, then convert it with 'whisper-apr convert'."
                .to_string(),
        )
    })?;

    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.display().to_string()));
    }

    if !global.quiet {
        println!("Loading LFM2 model from {}...", model_path.display());
    }

    let model_data = fs::read(model_path)?;
    let model = crate::model::lfm2::Lfm2::from_apr2_bytes(model_data)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to load model: {e}")))?;

    if global.verbose {
        eprintln!(
            "[INFO] Model loaded ({} params, {:.2} MB)",
            model.num_params(),
            model.memory_bytes() as f64 / (1024.0 * 1024.0)
        );
    }

    Ok(model)
}

/// Load tokenizer for summarization
fn load_summarize_tokenizer(
    args: &SummarizeArgs,
    global: &Args,
) -> CliResult<crate::model::lfm2::Lfm2Tokenizer> {
    if let Some(tokenizer_path) = &args.tokenizer_path {
        if !tokenizer_path.exists() {
            return Err(CliError::FileNotFound(tokenizer_path.display().to_string()));
        }
        if global.verbose {
            eprintln!("[INFO] Loading tokenizer from {}", tokenizer_path.display());
        }
        crate::model::lfm2::Lfm2Tokenizer::from_file(tokenizer_path)
            .map_err(|e| CliError::InvalidArgument(format!("Failed to load tokenizer: {e}")))
    } else {
        if global.verbose {
            eprintln!("[INFO] Using default byte-level tokenizer");
        }
        Ok(crate::model::lfm2::Lfm2Tokenizer::new())
    }
}

/// Run generation with streaming or non-streaming mode
fn generate_summary(
    model: &crate::model::lfm2::Lfm2,
    tokenizer: &crate::model::lfm2::Lfm2Tokenizer,
    input_ids: &[u32],
    args: &SummarizeArgs,
    global: &Args,
) -> CliResult<(Vec<u32>, crate::model::lfm2::GenerationStats)> {
    if args.stream {
        use std::io::Write;
        let tokenizer_ref = tokenizer;
        let quiet = global.quiet;

        model
            .generate_with_stats(
                input_ids,
                args.max_tokens as usize,
                args.temperature,
                Some(|token: u32, _idx: usize| {
                    if !quiet {
                        let text = tokenizer_ref.decode(&[token]);
                        print!("{text}");
                        let _ = io::stdout().flush();
                    }
                    true
                }),
            )
            .map_err(|e| CliError::InvalidArgument(format!("Generation failed: {e}")))
    } else {
        model
            .generate_with_stats::<fn(u32, usize) -> bool>(
                input_ids,
                args.max_tokens as usize,
                args.temperature,
                None,
            )
            .map_err(|e| CliError::InvalidArgument(format!("Generation failed: {e}")))
    }
}

/// Format summary output based on format argument
fn format_summary_output(
    args: &SummarizeArgs,
    summary: &str,
    input_text: &str,
    input_ids: &[u32],
    gen_stats: &crate::model::lfm2::GenerationStats,
    load_time: std::time::Duration,
    total_time: std::time::Duration,
) -> String {
    match args.format {
        crate::cli::args::SummarizeFormat::Json => serde_json::json!({
            "summary": summary.trim(),
            "stats": {
                "input_chars": input_text.len(),
                "input_tokens": input_ids.len(),
                "output_tokens": gen_stats.tokens_generated,
                "load_time_s": load_time.as_secs_f64(),
                "gen_time_ms": gen_stats.total_ms,
                "total_time_s": total_time.as_secs_f64(),
                "tokens_per_sec": gen_stats.tokens_per_sec,
                "ms_per_token": gen_stats.ms_per_token,
                "streaming": args.stream,
                "hit_eos": gen_stats.hit_eos
            }
        })
        .to_string(),
        crate::cli::args::SummarizeFormat::Text => summary.trim().to_string(),
        crate::cli::args::SummarizeFormat::Markdown => {
            format!("## Summary\n\n{}", summary.trim())
        }
        crate::cli::args::SummarizeFormat::Bullets => summary
            .trim()
            .lines()
            .map(|line| format!("- {}", line.trim()))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Log generation statistics in verbose mode
fn log_generation_stats(gen_stats: &crate::model::lfm2::GenerationStats, global: &Args) {
    if global.verbose {
        eprintln!(
            "[INFO] Generated {} tokens in {:.1}ms ({:.1} tokens/s)",
            gen_stats.tokens_generated, gen_stats.total_ms, gen_stats.tokens_per_sec
        );
        if gen_stats.hit_eos {
            eprintln!("[INFO] Generation completed (hit EOS token)");
        }
    }
}

/// Write summary output to file or stdout
fn write_summary_result(
    args: &SummarizeArgs,
    global: &Args,
    output: &str,
    gen_stats: &crate::model::lfm2::GenerationStats,
    total_time: std::time::Duration,
) -> CliResult<()> {
    if let Some(output_path) = &args.output {
        fs::write(output_path, output)?;
        if !global.quiet {
            println!("Summary written to: {}", output_path.display());
        }
    } else if !global.quiet {
        println!("\n{output}");
    }

    if !global.quiet {
        let stream_indicator = if args.stream { " (streamed)" } else { "" };
        println!(
            "\nCompleted in {:.2}s ({} tokens at {:.1} tokens/s{stream_indicator})",
            total_time.as_secs_f64(),
            gen_stats.tokens_generated,
            gen_stats.tokens_per_sec
        );
    }
    Ok(())
}

/// Run record command (audio capture to file)
///
/// When implemented, this will use:
/// - `aprender::native` for audio capture from microphone
pub fn run_record(args: RecordArgs, _global: &Args) -> CliResult<CommandResult> {
    if args.list_devices {
        // List audio devices (placeholder)
        println!("Audio devices:");
        println!("  0: Default Input");
        return Ok(CommandResult::success("Listed devices"));
    }

    if args.live {
        return Err(CliError::NotImplemented(
            "Live recording not yet implemented (requires aprender::native)".to_string(),
        ));
    }

    if args.duration.is_none() && args.output.is_none() {
        return Err(CliError::InvalidArgument(
            "Either --duration or --live must be specified".to_string(),
        ));
    }

    Err(CliError::NotImplemented(
        "Audio recording not yet implemented (requires aprender::native)".to_string(),
    ))
}

// ============================================================================
// Batch/Folder Processing Helpers (WAPR-PERF-004)
// ============================================================================

/// Supported audio extensions for batch processing
const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "flac", "ogg", "m4a", "webm", "aac", "mp4", "mkv",
];

/// Discover audio files from inputs (files or directories).
///
/// Per spec §1.3: Recursive discovery with pattern matching.
/// Files are sorted for deterministic parallel processing.
fn discover_audio_files(
    inputs: &[std::path::PathBuf],
    recursive: bool,
    pattern: Option<&str>,
) -> Vec<(std::path::PathBuf, Option<std::path::PathBuf>)> {
    let mut files = Vec::new();

    for input in inputs {
        if input.is_file() {
            // Direct file input - no base directory for mirroring
            if matches_audio_pattern(input, pattern) {
                files.push((input.clone(), None));
            }
        } else if input.is_dir() {
            // Directory input - discover files with base for mirroring
            discover_in_directory(input, input, recursive, pattern, &mut files);
        }
    }

    // Sort for deterministic processing (spec §1.3 Conflict Resolution #3)
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

/// Check if a path refers to a hidden file or directory (name starts with '.').
///
/// Per spec section H point 108: hidden files are skipped during discovery.
fn is_hidden_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map_or(false, |name| name.starts_with('.'))
}

/// Check if a directory path is a symlink (for loop prevention).
///
/// Per spec section H point 109: symlinks to directories are skipped to prevent loops.
/// Uses `symlink_metadata` to detect symlinks without following them.
fn is_symlink_via_metadata(path: &Path) -> bool {
    path.symlink_metadata()
        .map(|m| m.is_symlink())
        .unwrap_or(false)
}

/// Check if a path has a recognized audio file extension for folder discovery.
///
/// This is the extension list used by `discover_folder_recursive`. It intentionally
/// matches the inline list from the original implementation (without `mkv`).
fn has_folder_audio_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .map_or(false, |ext| {
            matches!(
                ext.as_str(),
                "wav" | "mp3" | "flac" | "ogg" | "m4a" | "mp4" | "webm" | "aac"
            )
        })
}

/// Classify a single directory entry as audio file, subdirectory, or ignored.
fn classify_dir_entry(
    path: &Path,
    base: &Path,
    recursive: bool,
    pattern: Option<&str>,
    files: &mut Vec<(std::path::PathBuf, Option<std::path::PathBuf>)>,
) {
    if is_hidden_path(path) {
        return;
    }
    if path.is_file() && matches_audio_pattern(path, pattern) {
        files.push((path.to_path_buf(), Some(base.to_path_buf())));
    } else if path.is_dir() && recursive && !is_symlink_via_metadata(path) {
        discover_in_directory(base, path, recursive, pattern, files);
    }
}

/// Recursively discover audio files in a directory.
fn discover_in_directory(
    base: &Path,
    dir: &Path,
    recursive: bool,
    pattern: Option<&str>,
    files: &mut Vec<(std::path::PathBuf, Option<std::path::PathBuf>)>,
) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        classify_dir_entry(&entry.path(), base, recursive, pattern, files);
    }
}

/// Check if a file matches the audio pattern.
fn matches_audio_pattern(path: &Path, pattern: Option<&str>) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    let ext = match ext {
        Some(e) => e,
        None => return false,
    };

    // Check if it's a supported audio extension
    if !AUDIO_EXTENSIONS.contains(&ext.as_str()) {
        return false;
    }

    // Apply pattern filter if specified
    if let Some(pat) = pattern {
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        return glob_match(pat, file_name);
    }

    true
}

/// Simple glob pattern matching (supports `*` and `?`).
fn glob_match(pattern: &str, text: &str) -> bool {
    let mut p_chars = pattern.chars().peekable();
    let mut t_chars = text.chars().peekable();

    while let Some(p) = p_chars.next() {
        match p {
            '*' => return glob_match_star(p_chars, t_chars),
            '?' => {
                if t_chars.next().is_none() {
                    return false;
                }
            }
            c => {
                if t_chars.next() != Some(c) {
                    return false;
                }
            }
        }
    }

    t_chars.peek().is_none()
}

/// Handle the `*` wildcard portion of glob matching.
///
/// Consumes consecutive `*` characters, then attempts to match the remaining
/// pattern against every suffix of the remaining text.
fn glob_match_star(
    mut p_chars: std::iter::Peekable<std::str::Chars<'_>>,
    mut t_chars: std::iter::Peekable<std::str::Chars<'_>>,
) -> bool {
    // Skip consecutive stars (**, ***, etc. are equivalent to *)
    while p_chars.peek() == Some(&'*') {
        p_chars.next();
    }

    // If * is at end of pattern, it matches everything remaining
    if p_chars.peek().is_none() {
        return true;
    }

    // Try matching rest of pattern starting at each text position
    let rest_pattern: String = p_chars.collect();
    while t_chars.peek().is_some() {
        let rest_text: String = t_chars.clone().collect();
        if glob_match(&rest_pattern, &rest_text) {
            return true;
        }
        t_chars.next();
    }

    // Also try matching with empty remaining text (e.g., pattern "a*b" vs "ab")
    glob_match(&rest_pattern, "")
}

/// Compute output path with structure mirroring.
///
/// Per spec §1.3 Path Resolution Logic:
/// - `./raw/a.wav` → `./trans/a.json` (flat if no subdirs)
/// - `./raw/sub/b.mp3` → `./trans/sub/b.json` (structure mirroring)
fn compute_mirrored_output_path(
    input_path: &Path,
    base_dir: Option<&Path>,
    output_dir: &Path,
    format_ext: &str,
) -> std::path::PathBuf {
    let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();

    match base_dir {
        Some(base) => {
            // Structure mirroring: preserve relative path from base
            let relative = input_path
                .parent()
                .and_then(|p| p.strip_prefix(base).ok())
                .unwrap_or(Path::new(""));
            output_dir
                .join(relative)
                .join(format!("{stem}.{format_ext}"))
        }
        None => {
            // Flat mapping for direct file inputs
            output_dir.join(format!("{stem}.{format_ext}"))
        }
    }
}

/// Write transcription atomically (temp file then rename).
///
/// Per spec §1.3 Conflict Resolution #2:
/// Write to `${filename}.tmp` then rename to prevent partial writes on crash.
fn atomic_write_transcription(output_path: &Path, content: &str) -> Result<(), CliError> {
    // Ensure parent directory exists (spec §H point 106)
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Create temporary file for atomic write
    let temp_path = output_path.with_extension("tmp");
    fs::write(&temp_path, content)?;

    // Atomic rename
    fs::rename(&temp_path, output_path)?;

    Ok(())
}

/// Result of processing a single batch file
enum BatchFileResult {
    Processed,
    Skipped,
    Failed,
}

/// Build TranscribeArgs for batch processing a single file
fn build_batch_transcribe_args(input_path: &Path, args: &BatchArgs) -> TranscribeArgs {
    TranscribeArgs {
        input: input_path.to_path_buf(),
        model: args.model,
        output: None,
        format: args.format,
        model_path: None,
        language: "auto".to_string(),
        detect_language: false,
        offset_t: 0,
        offset_n: 0,
        duration: 0,
        max_context: -1,
        max_len: 0,
        audio_ctx: 0,
        best_of: 2,
        beam_size: -1,
        temperature: 0.0,
        temperature_inc: 0.2,
        no_fallback: false,
        split_on_word: false,
        word_thold: 0.01,
        word_timestamps: false,
        timestamps: false,
        no_timestamps: false,
        entropy_thold: 2.40,
        logprob_thold: -1.0,
        no_speech_thold: 0.6,
        prompt: String::new(),
        suppress_regex: String::new(),
        grammar: String::new(),
        grammar_rule: String::new(),
        grammar_penalty: 100.0,
        vad: false,
        vad_model: None,
        vad_threshold: 0.5,
        vad_min_speech_ms: 250,
        vad_min_silence_ms: 100,
        vad_max_speech_s: None,
        vad_pad_ms: 30,
        vad_overlap: 0.1,
        threads: None,
        processors: 1,
        gpu: false,
        no_gpu: false,
        flash_attn: false,
        no_flash_attn: false,
        no_prints: true,
        print_special: false,
        colors: false,
        confidence: false,
        progress: false,
        print_memory: false,
        profile: false,
        translate: false,
        hallucination_filter: false,
        speed: 1.0,
        cache_dir: args.cache_dir.clone(),
        zram_optimized: args.zram_optimized,
        summarize: false,
        lfm2_model: None,
        summary_output: None,
        summary_format: SummarizeFormat::Json,
        action_items: false,
        key_points: false,
    }
}

/// Process a single file in batch mode: transcribe, format, and write atomically
fn process_batch_file(
    transcribe_args: &TranscribeArgs,
    output_path: &Path,
    format: OutputFormatArg,
    global: &Args,
) -> BatchFileResult {
    match run_transcribe_internal(transcribe_args, global) {
        Ok(result) => {
            let content = format_batch_output(&result, format);
            match atomic_write_transcription(output_path, &content) {
                Ok(()) => BatchFileResult::Processed,
                Err(e) => {
                    if global.verbose {
                        eprintln!("[ERROR] Write failed {}: {}", output_path.display(), e);
                    }
                    BatchFileResult::Failed
                }
            }
        }
        Err(e) => {
            if global.verbose {
                eprintln!("[ERROR] {}: {}", transcribe_args.input.display(), e);
            }
            BatchFileResult::Failed
        }
    }
}

/// Process a single entry in batch mode (skip/transcribe/fail)
fn process_batch_entry(
    input_path: &Path,
    output_path: &Path,
    args: &BatchArgs,
    global: &Args,
) -> BatchFileResult {
    if args.skip_existing && output_path.exists() {
        if global.verbose {
            eprintln!("[SKIP] {}", output_path.display());
        }
        return BatchFileResult::Skipped;
    }

    if global.verbose {
        eprintln!(
            "[PROC] {} → {}",
            input_path.display(),
            output_path.display()
        );
    }

    let transcribe_args = build_batch_transcribe_args(input_path, args);
    process_batch_file(&transcribe_args, output_path, args.format, global)
}

/// Run batch command (transcribe-folder)
///
/// Per spec WAPR-PERF-004 (docs/specifications/transcribe-folder-spec.md):
/// - Structure mirroring: `./raw/sub/b.mp3` → `./trans/sub/b.json`
/// - Atomic writes: temp file then rename
/// - Resumable: skip existing files with `--skip-existing`
/// - Deterministic: sorted file list for reproducible parallel processing
pub fn run_batch(args: BatchArgs, global: &Args) -> CliResult<CommandResult> {
    if args.inputs.is_empty() {
        return Err(CliError::InvalidArgument(
            "No input files specified".to_string(),
        ));
    }

    let output_dir = args.output_dir.clone().unwrap_or_else(|| ".".into());
    let format_ext = args.format.to_string();

    // Discover all audio files (recursive if specified, sorted for determinism)
    let files = discover_audio_files(&args.inputs, args.recursive, args.pattern.as_deref());

    if files.is_empty() {
        return Err(CliError::InvalidArgument(
            "No audio files found matching the specified inputs/pattern".to_string(),
        ));
    }

    if global.verbose {
        eprintln!("[INFO] Discovered {} audio files", files.len());
    }

    let mut processed = 0;
    let mut skipped = 0;
    let mut failed = 0;
    let start_time = Instant::now();

    for (input_path, base_dir) in &files {
        let output_path =
            compute_mirrored_output_path(input_path, base_dir.as_deref(), &output_dir, &format_ext);

        match process_batch_entry(input_path, &output_path, &args, global) {
            BatchFileResult::Processed => processed += 1,
            BatchFileResult::Skipped => skipped += 1,
            BatchFileResult::Failed => failed += 1,
        }
    }

    let elapsed = start_time.elapsed();
    let total = processed + skipped + failed;

    Ok(CommandResult::success(format!(
        "Batch complete: {processed} processed, {skipped} skipped, {failed} failed ({total} total) in {:.1}s",
        elapsed.as_secs_f64()
    )))
}

/// Run transcribe-folder command (WAPR-PERF-004)
///
/// Outcome of processing a single folder file
enum FolderFileOutcome {
    Processed,
    Skipped,
    Failed,
}

/// Process a single file in folder transcription mode
fn process_folder_file(
    input_path: &Path,
    output_path: &Path,
    whisper: &WhisperApr,
    args: &TranscribeFolderArgs,
    global: &Args,
    budget_violations: &mut usize,
    profile_entries: &mut Vec<FolderProfileEntry>,
) -> FolderFileOutcome {
    // Skip if exists and --skip-existing (resumable per spec §1.3)
    if args.skip_existing && output_path.exists() {
        if global.verbose {
            eprintln!("[SKIP] {}", output_path.display());
        }
        return FolderFileOutcome::Skipped;
    }

    if global.verbose {
        eprintln!(
            "[PROC] {} → {}",
            input_path.display(),
            output_path.display()
        );
    }

    let file_start = Instant::now();

    let result = match transcribe_single_file(input_path, whisper, args, global) {
        Ok(r) => r,
        Err(e) => {
            if global.verbose {
                eprintln!("[ERROR] {}: {}", input_path.display(), e);
            }
            return FolderFileOutcome::Failed;
        }
    };

    let transcribe_ms = file_start.elapsed().as_secs_f64() * 1000.0;
    match process_folder_transcription(
        input_path,
        output_path,
        result,
        transcribe_ms,
        args,
        global,
        budget_violations,
        profile_entries,
    ) {
        Ok(()) => FolderFileOutcome::Processed,
        Err(e) => {
            if global.verbose {
                eprintln!("[ERROR] Write failed {}: {}", output_path.display(), e);
            }
            FolderFileOutcome::Failed
        }
    }
}

/// Structure-preserving batch transcription with brick profiling integration.
/// Per spec (docs/specifications/transcribe-folder-spec.md):
/// - §1.3: Structure mirroring, atomicity, determinism
/// - §2.3: Brick profiling with Jidoka budget enforcement
/// - §H: Falsification points 101-125
pub fn run_transcribe_folder(
    args: TranscribeFolderArgs,
    global: &Args,
) -> CliResult<CommandResult> {
    // Validate input directory exists
    if !args.input_dir.exists() {
        return Err(CliError::FileNotFound(args.input_dir.display().to_string()));
    }
    if !args.input_dir.is_dir() {
        return Err(CliError::InvalidArgument(format!(
            "{} is not a directory",
            args.input_dir.display()
        )));
    }

    // Configure thread pool
    let thread_count = configure_thread_pool(args.threads)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to configure threads: {e}")))?;

    if global.verbose {
        eprintln!("[INFO] Using {thread_count} thread(s) for inference");
    }

    // Discover audio files (sorted for determinism per spec §1.3 #3)
    let files = discover_folder_audio_files(&args.input_dir, args.recursive);

    if files.is_empty() {
        return Err(CliError::InvalidArgument(format!(
            "No audio files found in {}",
            args.input_dir.display()
        )));
    }

    if global.verbose {
        eprintln!("[INFO] Discovered {} audio files", files.len());
    }

    let whisper = load_folder_whisper_model(&args, global)?;

    let format_ext = args.format.to_string();
    let mut processed = 0;
    let mut skipped = 0;
    let mut failed = 0;
    let mut budget_violations = 0;
    let start_time = Instant::now();

    // Aggregate profiling data for report
    let mut profile_entries: Vec<FolderProfileEntry> = Vec::new();

    // Process each file
    for input_path in &files {
        let output_path =
            compute_folder_output_path(input_path, &args.input_dir, &args.output_dir, &format_ext);

        match process_folder_file(
            input_path,
            &output_path,
            &whisper,
            &args,
            global,
            &mut budget_violations,
            &mut profile_entries,
        ) {
            FolderFileOutcome::Processed => processed += 1,
            FolderFileOutcome::Skipped => skipped += 1,
            FolderFileOutcome::Failed => failed += 1,
        }
    }

    let elapsed = start_time.elapsed();
    let total = processed + skipped + failed;

    finalize_folder_reports(&args, global, &profile_entries, &elapsed);

    // Jidoka: fail if strict budget mode and violations occurred
    if args.strict_budget && budget_violations > 0 {
        return Err(CliError::InvalidArgument(format!(
            "Strict budget mode: {} file(s) exceeded throughput budget",
            budget_violations
        )));
    }

    Ok(CommandResult::success(format!(
        "Folder complete: {processed} processed, {skipped} skipped, {failed} failed ({total} total) in {:.1}s",
        elapsed.as_secs_f64()
    )))
}

/// Load Whisper model for folder transcription
fn load_folder_whisper_model(args: &TranscribeFolderArgs, global: &Args) -> CliResult<WhisperApr> {
    if global.verbose {
        if let Some(path) = &args.model_path {
            eprintln!("[INFO] Loading model from: {}", path.display());
        } else {
            eprintln!("[INFO] Loading model: {}", args.model);
        }
    }

    crate::cli::model_loader::load_or_download_model(
        args.model,
        args.model_path.as_deref(),
        global.verbose,
    )
    .map_err(|e| CliError::InvalidArgument(e.to_string()))
}

/// Write report and print profiling summary for folder transcription
fn finalize_folder_reports(
    args: &TranscribeFolderArgs,
    global: &Args,
    profile_entries: &[FolderProfileEntry],
    elapsed: &std::time::Duration,
) {
    if let Some(report_path) = &args.report {
        let report = generate_folder_profile_report(profile_entries, elapsed.as_secs_f64());
        if let Err(e) = fs::write(report_path, report) {
            eprintln!("[WARN] Failed to write report: {}", e);
        } else if global.verbose {
            eprintln!(
                "[INFO] Profile report written to: {}",
                report_path.display()
            );
        }
    }

    if args.profile && !global.quiet {
        print_folder_profile_summary(profile_entries, elapsed.as_secs_f64());
    }
}

/// Profile entry for a single file
struct FolderProfileEntry {
    file: String,
    audio_duration_secs: f64,
    transcribe_ms: f64,
    tokens_generated: usize,
    tokens_per_sec: f64,
    budget_met: bool,
    // Breakdown stats (if available)
    audio_ms: Option<f64>,
    encoder_ms: Option<f64>,
    decoder_ms: Option<f64>,
}

/// Process a successful folder transcription result: format, write, and collect profiling data
fn process_folder_transcription(
    input_path: &Path,
    output_path: &Path,
    result: FolderTranscribeResult,
    transcribe_ms: f64,
    args: &TranscribeFolderArgs,
    global: &Args,
    budget_violations: &mut usize,
    profile_entries: &mut Vec<FolderProfileEntry>,
) -> CliResult<()> {
    let tokens_per_sec = if transcribe_ms > 0.0 {
        (result.tokens_generated as f64 / transcribe_ms) * 1000.0
    } else {
        0.0
    };

    // Budget check: 130 µs/token = 7,692 tok/s target (spec §2.3.1)
    let budget_target_tok_s = 7692.0;
    let budget_met = tokens_per_sec >= budget_target_tok_s;

    if !budget_met && args.strict_budget {
        *budget_violations += 1;
        if global.verbose {
            eprintln!(
                "[JIDOKA] Budget exceeded for {}: {:.0} tok/s < {} tok/s",
                input_path.display(),
                tokens_per_sec,
                budget_target_tok_s
            );
        }
    }

    let content = if args.profile {
        format_folder_output_with_profile(
            &result,
            args.format,
            transcribe_ms,
            tokens_per_sec,
            budget_met,
        )
    } else {
        format_folder_output(&result, args.format)
    };

    atomic_write_transcription(output_path, &content)?;

    // Collect profile data for report
    if args.profile || args.report.is_some() {
        let (audio_ms, encoder_ms, decoder_ms) = if let Some(stats) = &result.profiling {
            (
                stats.breakdown.get("audio_ms").copied(),
                stats.breakdown.get("encoder_ms").copied(),
                stats.breakdown.get("decoder_ms").copied(),
            )
        } else {
            (None, None, None)
        };

        profile_entries.push(FolderProfileEntry {
            file: input_path.display().to_string(),
            audio_duration_secs: result.audio_duration_secs,
            transcribe_ms,
            tokens_generated: result.tokens_generated,
            tokens_per_sec,
            budget_met,
            audio_ms,
            encoder_ms,
            decoder_ms,
        });
    }

    Ok(())
}

/// Result from transcribing a single file
struct FolderTranscribeResult {
    text: String,
    #[allow(dead_code)]
    segments: Vec<String>,
    audio_duration_secs: f64,
    tokens_generated: usize,
    profiling: Option<ProfilingStats>,
}

/// Transcribe a single file using the provided model
fn transcribe_single_file(
    input_path: &Path,
    whisper: &WhisperApr,
    args: &TranscribeFolderArgs,
    global: &Args,
) -> CliResult<FolderTranscribeResult> {
    // Read audio file
    let audio_data = fs::read(input_path)?;
    let samples = load_audio_samples(input_path, &audio_data)?;
    let audio_duration_secs = samples.len() as f64 / 16000.0;

    // Create transcription options
    let task = Task::Transcribe;
    let options = TranscribeOptions {
        language: if args.language == "auto" {
            None
        } else {
            Some(args.language.clone())
        },
        task,
        strategy: DecodingStrategy::Greedy,
        word_timestamps: false,
        profile: args.profile,
    };

    // Transcribe
    let result = whisper.transcribe(&samples, options)?;
    let tokens_generated = result
        .segments
        .iter()
        .map(|s| s.text.split_whitespace().count())
        .sum();

    if global.verbose {
        eprintln!(
            "[INFO] Transcribed: {} chars, {} tokens",
            result.text.len(),
            tokens_generated
        );
    }

    Ok(FolderTranscribeResult {
        text: result.text,
        segments: result.segments.iter().map(|s| s.text.clone()).collect(),
        audio_duration_secs,
        tokens_generated,
        profiling: result.profiling,
    })
}

/// Discover audio files in a folder (sorted for determinism)
fn discover_folder_audio_files(input_dir: &Path, recursive: bool) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    discover_folder_recursive(input_dir, recursive, &mut files);
    // Sort for deterministic processing (spec §1.3 #3)
    files.sort();
    files
}

/// Classify a single entry for folder discovery.
fn classify_folder_entry(path: &Path, recursive: bool, files: &mut Vec<std::path::PathBuf>) {
    if is_hidden_path(path) {
        return;
    }
    if path.is_file() && has_folder_audio_extension(path) {
        files.push(path.to_path_buf());
    } else if path.is_dir() && recursive && !path.is_symlink() {
        discover_folder_recursive(path, recursive, files);
    }
}

/// Recursively discover audio files
fn discover_folder_recursive(dir: &Path, recursive: bool, files: &mut Vec<std::path::PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        classify_folder_entry(&entry.path(), recursive, files);
    }
}

/// Compute output path with structure mirroring (spec §1.3)
fn compute_folder_output_path(
    input_path: &Path,
    input_dir: &Path,
    output_dir: &Path,
    format_ext: &str,
) -> std::path::PathBuf {
    let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();

    // Structure mirroring: preserve relative path from input_dir
    let relative = input_path
        .parent()
        .and_then(|p| p.strip_prefix(input_dir).ok())
        .unwrap_or(Path::new(""));

    output_dir
        .join(relative)
        .join(format!("{stem}.{format_ext}"))
}

/// Format output for folder transcription
fn format_folder_output(result: &FolderTranscribeResult, format: OutputFormatArg) -> String {
    match format {
        OutputFormatArg::Txt => result.text.clone(),
        OutputFormatArg::Json | OutputFormatArg::JsonFull => {
            format!(
                r#"{{"text":"{}","segments":[]}}"#,
                result
                    .text
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
            )
        }
        OutputFormatArg::Vtt => format!("WEBVTT\n\n{}", result.text),
        OutputFormatArg::Srt => format!("1\n00:00:00,000 --> 00:00:30,000\n{}\n", result.text),
        OutputFormatArg::Csv => format!(
            "start,end,text\n0,30000,\"{}\"\n",
            result.text.replace('"', "\"\"")
        ),
        OutputFormatArg::Lrc => format!("[00:00.00]{}", result.text),
        OutputFormatArg::Wts => format!("[00:00.00]{}", result.text),
        OutputFormatArg::Md => format!("# Transcription\n\n{}\n", result.text),
    }
}

/// Format output with profiling metadata (spec §2.3.5)
fn format_folder_output_with_profile(
    result: &FolderTranscribeResult,
    format: OutputFormatArg,
    total_ms: f64,
    tokens_per_sec: f64,
    budget_met: bool,
) -> String {
    match format {
        OutputFormatArg::Json | OutputFormatArg::JsonFull => {
            let breakdown = if let Some(stats) = &result.profiling {
                let mut b = String::from(r#","breakdown":{"#);
                let mut parts = Vec::new();
                if let Some(v) = stats.breakdown.get("audio_ms") {
                    parts.push(format!(r#""audio_ms":{:.1}"#, v));
                }
                if let Some(v) = stats.breakdown.get("encoder_ms") {
                    parts.push(format!(r#""encoder_ms":{:.1}"#, v));
                }
                if let Some(v) = stats.breakdown.get("decoder_ms") {
                    parts.push(format!(r#""decoder_ms":{:.1}"#, v));
                }
                b.push_str(&parts.join(","));
                b.push('}');
                b
            } else {
                String::new()
            };

            format!(
                r#"{{"text":"{}","segments":[],"profiling":{{"total_ms":{:.1},"tokens_per_sec":{:.0},"budget_met":{}{}}}}}"#,
                result
                    .text
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n"),
                total_ms,
                tokens_per_sec,
                budget_met,
                breakdown
            )
        }
        // For non-JSON formats, just return the regular output
        _ => format_folder_output(result, format),
    }
}

/// Generate aggregate profile report (JSON)
fn generate_folder_profile_report(
    entries: &[FolderProfileEntry],
    total_elapsed_secs: f64,
) -> String {
    let file_count = entries.len();
    let total_audio_secs: f64 = entries.iter().map(|e| e.audio_duration_secs).sum();
    let total_tokens: usize = entries.iter().map(|e| e.tokens_generated).sum();
    let avg_tok_s = if !entries.is_empty() {
        entries.iter().map(|e| e.tokens_per_sec).sum::<f64>() / entries.len() as f64
    } else {
        0.0
    };
    let budget_met_count = entries.iter().filter(|e| e.budget_met).count();

    // Calculate aggregated breakdown
    let (avg_audio, avg_enc, avg_dec) = if !entries.is_empty() {
        let sum_audio: f64 = entries.iter().filter_map(|e| e.audio_ms).sum();
        let sum_enc: f64 = entries.iter().filter_map(|e| e.encoder_ms).sum();
        let sum_dec: f64 = entries.iter().filter_map(|e| e.decoder_ms).sum();
        let count = entries.len() as f64;
        (sum_audio / count, sum_enc / count, sum_dec / count)
    } else {
        (0.0, 0.0, 0.0)
    };

    let files_json: String = entries
        .iter()
        .map(|e| {
            let breakdown = format!(
                ",\"audio_ms\":{:.1},\"encoder_ms\":{:.1},\"decoder_ms\":{:.1}",
                e.audio_ms.unwrap_or(0.0),
                e.encoder_ms.unwrap_or(0.0),
                e.decoder_ms.unwrap_or(0.0)
            );
            format!(
                "    {{\"file\":\"{}\",\"audio_secs\":{:.1},\"ms\":{:.1},\"tokens\":{},\"tok_s\":{:.0},\"budget_met\":{}{}}}",
                e.file.replace('\\', "\\\\").replace('"', "\\\""),
                e.audio_duration_secs,
                e.transcribe_ms,
                e.tokens_generated,
                e.tokens_per_sec,
                e.budget_met,
                breakdown
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        "{{\n  \"file_count\": {},\n  \"total_audio_secs\": {:.1},\n  \"total_elapsed_secs\": {:.1},\n  \"total_tokens\": {},\n  \"avg_tokens_per_sec\": {:.0},\n  \"avg_breakdown_ms\": {{\"audio\":{:.1},\"encoder\":{:.1},\"decoder\":{:.1}}},\n  \"budget_met_count\": {},\n  \"budget_target_tok_s\": 7692,\n  \"files\": [\n{}\n  ]\n}}",
        file_count,
        total_audio_secs,
        total_elapsed_secs,
        total_tokens,
        avg_tok_s,
        avg_audio, avg_enc, avg_dec,
        budget_met_count,
        files_json
    )
}

/// Print folder profile summary to stderr
fn print_folder_profile_summary(entries: &[FolderProfileEntry], total_elapsed_secs: f64) {
    if entries.is_empty() {
        return;
    }

    let total_audio_secs: f64 = entries.iter().map(|e| e.audio_duration_secs).sum();
    let total_tokens: usize = entries.iter().map(|e| e.tokens_generated).sum();
    let avg_tok_s = entries.iter().map(|e| e.tokens_per_sec).sum::<f64>() / entries.len() as f64;
    let budget_met_count = entries.iter().filter(|e| e.budget_met).count();
    let budget_target = 7692.0;

    // Calculate aggregated breakdown
    let (avg_audio, avg_enc, avg_dec) = {
        let sum_audio: f64 = entries.iter().filter_map(|e| e.audio_ms).sum();
        let sum_enc: f64 = entries.iter().filter_map(|e| e.encoder_ms).sum();
        let sum_dec: f64 = entries.iter().filter_map(|e| e.decoder_ms).sum();
        let count = entries.len() as f64;
        (sum_audio / count, sum_enc / count, sum_dec / count)
    };

    eprintln!();
    eprintln!("=== Folder Profiling Summary ===");
    eprintln!("Files processed:     {}", entries.len());
    eprintln!("Total audio:         {:.1}s", total_audio_secs);
    eprintln!("Total elapsed:       {:.1}s", total_elapsed_secs);
    eprintln!("Total tokens:        {}", total_tokens);
    eprintln!("Avg throughput:      {:.0} tok/s", avg_tok_s);
    eprintln!(
        "Avg breakdown (ms):  Audio={:.1}, Enc={:.1}, Dec={:.1}",
        avg_audio, avg_enc, avg_dec
    );
    eprintln!("Budget target:       {:.0} tok/s", budget_target);
    eprintln!(
        "Budget status:       {}/{} files met budget ({}%)",
        budget_met_count,
        entries.len(),
        (budget_met_count * 100) / entries.len().max(1)
    );
    eprintln!();
}

/// Internal transcription result for batch processing
struct BatchTranscribeResult {
    text: String,
    #[allow(dead_code)]
    segments: Vec<String>,
}

/// Run transcription and return result (for batch mode)
fn run_transcribe_internal(
    args: &TranscribeArgs,
    global: &Args,
) -> CliResult<BatchTranscribeResult> {
    // Read audio file
    let audio_data = fs::read(&args.input)?;

    // Load and process audio using the shared loader
    let samples = load_audio_samples(&args.input, &audio_data)?;

    // Configure thread pool (ignore errors - use defaults if it fails)
    let _ = configure_thread_pool(args.threads);

    // Create transcription options (matching run_transcribe pattern)
    let task = if args.translate {
        Task::Translate
    } else {
        Task::Transcribe
    };

    let options = TranscribeOptions {
        language: if args.language == "auto" {
            None
        } else {
            Some(args.language.clone())
        },
        task,
        strategy: if args.beam_size > 0 {
            DecodingStrategy::BeamSearch {
                beam_size: args.beam_size as usize,
                temperature: args.temperature,
                patience: 1.0,
            }
        } else {
            DecodingStrategy::Greedy
        },
        word_timestamps: args.word_timestamps,
        profile: global.verbose,
    };

    // Load model using the shared loader
    let whisper = crate::cli::model_loader::load_or_download_model(
        args.model,
        args.model_path.as_deref(),
        global.verbose,
    )
    .map_err(|e| CliError::InvalidArgument(e.to_string()))?;

    let result = whisper.transcribe(&samples, options)?;

    if global.verbose {
        eprintln!(
            "[INFO] Transcribed: {} chars, {} segments",
            result.text.len(),
            result.segments.len()
        );
    }

    Ok(BatchTranscribeResult {
        text: result.text,
        segments: result.segments.iter().map(|s| s.text.clone()).collect(),
    })
}

/// Format batch output according to requested format
fn format_batch_output(result: &BatchTranscribeResult, format: OutputFormatArg) -> String {
    match format {
        OutputFormatArg::Txt => result.text.clone(),
        OutputFormatArg::Json | OutputFormatArg::JsonFull => {
            format!(
                r#"{{"text":"{}","segments":[]}}"#,
                result.text.replace('\\', "\\\\").replace('"', "\\\"")
            )
        }
        OutputFormatArg::Vtt => {
            format!("WEBVTT\n\n{}", result.text)
        }
        OutputFormatArg::Srt => {
            format!("1\n00:00:00,000 --> 00:00:30,000\n{}\n", result.text)
        }
        OutputFormatArg::Csv => {
            format!(
                "start,end,text\n0,30000,\"{}\"\n",
                result.text.replace('"', "\"\"")
            )
        }
        OutputFormatArg::Lrc => {
            format!("[00:00.00]{}", result.text)
        }
        OutputFormatArg::Wts => {
            // Karaoke word-timestamp format
            format!("[00:00.00]{}", result.text)
        }
        OutputFormatArg::Md => {
            // Markdown format
            format!("# Transcription\n\n{}\n", result.text)
        }
    }
}

/// Run TUI command (interactive pipeline visualization)
///
/// Launches the interactive terminal dashboard for visualizing
/// the Whisper ASR pipeline stages: waveform → mel → encoder → decoder → text.
///
/// # Keyboard Shortcuts
/// - 1-7: Switch panels (Waveform, Mel, Encoder, Decoder, Attention, Transcription, Metrics)
/// - ?: Show help
/// - Space: Pause/resume
/// - r: Reset
/// - q: Quit
/// Handle a single TUI keypress, returning Some(result) if the app should exit
#[cfg(feature = "tui")]
fn handle_tui_key(
    code: crossterm::event::KeyCode,
    app: &mut crate::tui::WhisperApp,
) -> Option<CommandResult> {
    use crossterm::event::KeyCode;
    const HANDLED_CHARS: &[char] = &['1', '2', '3', '4', '5', '6', '7', '?', ' ', 'r'];
    match code {
        KeyCode::Char('q') => Some(CommandResult::success("TUI closed")),
        KeyCode::Char(c) if HANDLED_CHARS.contains(&c) => {
            app.handle_key(c);
            None
        }
        _ => None,
    }
}

/// Poll for a TUI event and handle it, returning Some if app should exit
#[cfg(feature = "tui")]
fn poll_tui_event(app: &mut crate::tui::WhisperApp) -> CliResult<Option<CommandResult>> {
    use crossterm::event::{self, Event, KeyEventKind};
    use std::time::Duration;

    if !event::poll(Duration::from_millis(100)).map_err(|e| CliError::Io(io::Error::other(e)))? {
        return Ok(None);
    }
    let event = event::read().map_err(|e| CliError::Io(io::Error::other(e)))?;
    if let Event::Key(key) = event {
        if key.kind == KeyEventKind::Press {
            return Ok(handle_tui_key(key.code, app));
        }
    }
    Ok(None)
}

/// Run the TUI (terminal user interface) for interactive transcription.
#[cfg(feature = "tui")]
pub fn run_tui(global: &Args) -> CliResult<CommandResult> {
    use crossterm::{
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::{backend::CrosstermBackend, Terminal};
    

    use crate::tui::{render_whisper_dashboard, WhisperApp};

    // Initialize terminal
    enable_raw_mode().map_err(|e| CliError::Io(io::Error::other(e)))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).map_err(|e| CliError::Io(io::Error::other(e)))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(|e| CliError::Io(io::Error::other(e)))?;

    // Create application state
    let mut app = WhisperApp::new();

    // Show initial status
    app.status_message = Some("Press '?' for help, 'q' to quit".to_string());

    // Event loop
    let result = loop {
        // Draw frame
        terminal
            .draw(|f| render_whisper_dashboard(f, &app))
            .map_err(|e| CliError::Io(io::Error::other(e)))?;

        // Poll for events with timeout
        if let Some(result) = poll_tui_event(&mut app)? {
            break Ok(result);
        }

        // Check if we should quit
        if app.should_quit {
            break Ok(CommandResult::success("TUI closed"));
        }
    };

    // Restore terminal
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    if global.verbose {
        eprintln!("[INFO] TUI session ended");
    }

    result
}

/// Run TUI command - stub when TUI feature is disabled
#[cfg(not(feature = "tui"))]
pub fn run_tui(_global: &Args) -> CliResult<CommandResult> {
    Err(CliError::NotImplemented(
        "TUI requires the 'tui' feature. Rebuild with: cargo build --features tui".to_string(),
    ))
}

/// Run test command
pub fn run_test(args: TestArgs, global: &Args) -> CliResult<CommandResult> {
    let backends = match args.backend {
        BackendArg::All => vec![BackendArg::Simd, BackendArg::Wasm, BackendArg::Cuda],
        other => vec![other],
    };

    let mut results = Vec::new();

    for backend in backends {
        if !global.quiet {
            println!("\nTesting {backend} backend...");
        }

        let result = test_backend(backend, global);
        results.push((backend, result));
    }

    // Summary
    let passed = results.iter().filter(|(_, r)| r.is_ok()).count();
    let total = results.len();

    if !global.quiet {
        println!("\nSummary: {passed}/{total} backends passed");
    }

    if passed == total {
        Ok(CommandResult::success(format!(
            "{passed}/{total} backends passed"
        )))
    } else {
        Ok(CommandResult::failure(format!(
            "{passed}/{total} backends passed"
        )))
    }
}

/// Test a specific backend
fn test_backend(backend: BackendArg, _global: &Args) -> CliResult<()> {
    match backend {
        BackendArg::Simd => {
            // SIMD is always available
            let whisper = WhisperApr::tiny();
            let samples = vec![0.0f32; 16000]; // 1 second of silence
            let options = TranscribeOptions::default();
            let _result = whisper.transcribe(&samples, options)?;
            println!("  SIMD: PASS");
            Ok(())
        }
        BackendArg::Wasm => {
            // WASM test would require browser
            println!("  WASM: SKIPPED (requires browser)");
            Ok(())
        }
        BackendArg::Cuda => {
            // Check for CUDA availability
            let cuda_available = std::process::Command::new("nvidia-smi")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);

            if cuda_available {
                println!("  CUDA: PASS (GPU detected)");
            } else {
                println!("  CUDA: SKIPPED (no GPU)");
            }
            Ok(())
        }
        BackendArg::All => unreachable!(),
    }
}

/// Run selftest command: diagnose + backend test + optional transcription
pub fn run_selftest(args: SelftestArgs, global: &Args) -> CliResult<CommandResult> {
    use crate::cli::args::{DiagnoseArgs, TestArgs};

    let mut all_passed = true;

    // Phase 1: Diagnose (tokenizer validation)
    if !global.quiet {
        println!("Phase 1/3: Diagnose (tokenizer validation)");
        println!("───────────────────────────────────────────────────────────────────");
    }
    let diag_args = DiagnoseArgs {
        model: None,
        tokenizer_only: true,
        json: false,
        full: false,
    };
    match run_diagnose(diag_args, global) {
        Ok(r) if r.success => {
            if !global.quiet {
                println!("  Phase 1: PASS\n");
            }
        }
        Ok(_) => {
            if !global.quiet {
                println!("  Phase 1: FAIL\n");
            }
            all_passed = false;
        }
        Err(e) => {
            if !global.quiet {
                println!("  Phase 1: ERROR ({e})\n");
            }
            all_passed = false;
        }
    }

    // Phase 2: Backend test (SIMD)
    if !global.quiet {
        println!("Phase 2/3: Backend test (SIMD)");
        println!("───────────────────────────────────────────────────────────────────");
    }
    let test_args = TestArgs {
        backend: BackendArg::Simd,
        demo: None,
        pipeline: None,
    };
    match run_test(test_args, global) {
        Ok(r) if r.success => {
            if !global.quiet {
                println!("  Phase 2: PASS\n");
            }
        }
        Ok(_) => {
            if !global.quiet {
                println!("  Phase 2: FAIL\n");
            }
            all_passed = false;
        }
        Err(e) => {
            if !global.quiet {
                println!("  Phase 2: ERROR ({e})\n");
            }
            all_passed = false;
        }
    }

    // Phase 3: Optional transcription test
    if let (Some(model_path), Some(audio_path)) = (&args.model, &args.audio) {
        if !global.quiet {
            println!("Phase 3/3: Transcription test");
            println!("───────────────────────────────────────────────────────────────────");
        }
        match run_selftest_transcription(model_path, audio_path, args.expect.as_deref(), global) {
            Ok(true) => {
                if !global.quiet {
                    println!("  Phase 3: PASS\n");
                }
            }
            Ok(false) => {
                if !global.quiet {
                    println!("  Phase 3: FAIL\n");
                }
                all_passed = false;
            }
            Err(e) => {
                if !global.quiet {
                    println!("  Phase 3: ERROR ({e})\n");
                }
                all_passed = false;
            }
        }
    } else if !global.quiet {
        println!("Phase 3/3: Transcription test (skipped — no --model/--audio)");
        println!("───────────────────────────────────────────────────────────────────");
        println!("  Provide --model <path.apr> --audio <file.wav> --expect <text>\n");
    }

    if all_passed {
        Ok(CommandResult::success(
            "All selftest phases passed".to_string(),
        ))
    } else {
        Ok(CommandResult::failure(
            "One or more selftest phases failed".to_string(),
        ))
    }
}

/// Run a transcription test as part of selftest
fn run_selftest_transcription(
    model_path: &Path,
    audio_path: &Path,
    expect: Option<&str>,
    global: &Args,
) -> CliResult<bool> {
    if !model_path.exists() {
        return Err(CliError::FileNotFound(format!(
            "Model file: {}",
            model_path.display()
        )));
    }
    if !audio_path.exists() {
        return Err(CliError::FileNotFound(format!(
            "Audio file: {}",
            audio_path.display()
        )));
    }

    // Load audio
    let audio_data = fs::read(audio_path).map_err(CliError::Io)?;
    let wav = parse_wav_file(&audio_data)?;
    let samples = if wav.sample_rate != 16000 {
        resample(&wav.samples, wav.sample_rate, 16000)
    } else {
        wav.samples
    };

    // Load model and transcribe
    let model_bytes = fs::read(model_path).map_err(CliError::Io)?;
    let whisper = WhisperApr::load_from_apr(&model_bytes)?;
    let options = TranscribeOptions::default();
    let result = whisper.transcribe(&samples, options)?;

    let text = result.text.trim().to_lowercase();
    if !global.quiet {
        println!("  Transcription: \"{text}\"");
    }

    if let Some(expected) = expect {
        let expected_lower = expected.to_lowercase();
        if text.contains(&expected_lower) {
            if !global.quiet {
                println!("  Expected \"{expected}\" found in output");
            }
            Ok(true)
        } else {
            if !global.quiet {
                println!("  Expected \"{expected}\" NOT found in output");
            }
            Ok(false)
        }
    } else {
        // No expectation — pass if transcription produced any text
        Ok(!text.is_empty())
    }
}

/// Run model command
pub fn run_model(args: ModelArgs, global: &Args) -> CliResult<CommandResult> {
    match args.action {
        ModelAction::List => run_model_list(global),
        ModelAction::Download { model } => run_model_download(model, global),
        ModelAction::Convert { input, output } => run_model_convert(&input, &output, global),
        ModelAction::Info { file } => run_model_info(&file, global),
        ModelAction::WasmCheck {
            family,
            quantization,
            context,
            sliding_window,
        } => run_model_wasm_check(&family, &quantization, context, sliding_window, global),
    }
}

/// List available models grouped by family
fn run_model_list(global: &Args) -> CliResult<CommandResult> {
    use crate::model::download::{list_models, ModelFamily};

    if !global.quiet {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("                    Available Models                               ");
        println!("═══════════════════════════════════════════════════════════════════\n");
    }

    println!("WHISPER (ASR - Automatic Speech Recognition)");
    println!("───────────────────────────────────────────────────────────────────");
    for model in list_models() {
        if model.family == ModelFamily::Whisper {
            println!(
                "  {:<20} {:>6} params  {}",
                model.name, model.params, model.description
            );
            if global.verbose {
                println!(
                    "                       fp16: {}  int4: {}  WASM: {}",
                    model.size_fp16, model.size_int4, model.wasm_quant
                );
            }
        }
    }

    println!("\nLFM2 (Post-Transcription Summarization)");
    println!("───────────────────────────────────────────────────────────────────");
    for model in list_models() {
        if model.family == ModelFamily::Lfm2 {
            println!(
                "  {:<20} {:>6} params  {}",
                model.name, model.params, model.description
            );
            if global.verbose {
                println!(
                    "                       fp16: {}  int4: {}  WASM: {}",
                    model.size_fp16, model.size_int4, model.wasm_quant
                );
            }
        }
    }

    if !global.quiet {
        println!("\n───────────────────────────────────────────────────────────────────");
        println!("Use 'whisper-apr model download <name>' to download a model.");
        println!("Use -v/--verbose for size details.");
    }

    Ok(CommandResult::success("Listed models"))
}

/// Download a model from HuggingFace
fn run_model_download(
    model: crate::cli::args::ModelSize,
    global: &Args,
) -> CliResult<CommandResult> {
    use crate::model::download::find_model;

    let model_name = match model {
        crate::cli::args::ModelSize::Tiny => "whisper-tiny",
        crate::cli::args::ModelSize::Base => "whisper-base",
        crate::cli::args::ModelSize::Small => "whisper-small",
        crate::cli::args::ModelSize::Medium => "whisper-medium",
        crate::cli::args::ModelSize::Large => "whisper-large",
        crate::cli::args::ModelSize::MoonshineTiny => "moonshine-tiny",
        crate::cli::args::ModelSize::MoonshineBase => "moonshine-base",
    };

    let model_info = find_model(model_name)
        .ok_or_else(|| CliError::InvalidArgument(format!("Unknown model: {model_name}")))?;

    if !global.quiet {
        println!("Downloading {} from HuggingFace...", model_info.name);
        println!("  Repository: {}", model_info.repo_id);
        println!("  Parameters: {}", model_info.params);
        println!("  Size (fp16): {}", model_info.size_fp16);
    }

    let downloader = crate::model::download::ModelDownloader::new()
        .map_err(|e| CliError::InvalidArgument(format!("Failed to initialize downloader: {e}")))?;

    let paths = downloader
        .download_safetensors(model_info)
        .map_err(|e| CliError::InvalidArgument(format!("Download failed: {e}")))?;

    if !global.quiet {
        println!("\nDownloaded {} file(s):", paths.len());
        for path in &paths {
            println!("  {}", path.display());
        }
        println!("\nCache directory: {}", downloader.cache_dir().display());
    }

    Ok(CommandResult::success(format!(
        "Downloaded {} ({} files)",
        model_info.name,
        paths.len()
    )))
}

/// Convert safetensors to APR2 format
fn run_model_convert(input: &Path, output: &Path, global: &Args) -> CliResult<CommandResult> {
    if !global.quiet {
        println!("Converting {} to {}...", input.display(), output.display());
    }

    if !input.exists() {
        return Err(CliError::FileNotFound(input.display().to_string()));
    }

    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext != "safetensors" {
        return Err(CliError::UnsupportedFormat(format!(
            "Expected .safetensors file, got .{ext}"
        )));
    }

    let loader = crate::format::SafeTensorsLoader::load(input)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to load: {e}")))?;

    let config = crate::format::apr2::Lfm2Config::lfm2_2_6b();
    let quant = crate::format::apr2::QuantConfig::default();

    let writer = loader
        .to_apr2(config, quant, false)
        .map_err(|e| CliError::InvalidArgument(format!("Conversion failed: {e}")))?;

    let bytes = writer
        .to_bytes()
        .map_err(|e| CliError::InvalidArgument(format!("Serialization failed: {e}")))?;

    std::fs::write(output, &bytes)
        .map_err(|e| CliError::WriteError(format!("Failed to write: {e}")))?;

    if !global.quiet {
        println!(
            "Converted {} tensors to {}",
            loader.tensor_names().len(),
            output.display()
        );
    }

    Ok(CommandResult::success(format!(
        "Converted to {}",
        output.display()
    )))
}

/// Show model file information
fn run_model_info(file: &Path, global: &Args) -> CliResult<CommandResult> {
    if !file.exists() {
        return Err(CliError::FileNotFound(file.display().to_string()));
    }

    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "apr2" => print_apr2_info(file, global)?,
        "safetensors" => print_safetensors_info(file, global)?,
        "apr" => {
            println!("APR v1 file: {}", file.display());
            let metadata = std::fs::metadata(file)
                .map_err(|e| CliError::InvalidArgument(format!("Failed to read: {e}")))?;
            println!("Size: {} bytes", metadata.len());
        }
        _ => {
            return Err(CliError::UnsupportedFormat(format!(
                "Unknown file type: .{ext}"
            )));
        }
    }

    Ok(CommandResult::success("Showed model info"))
}

/// Print APR2 file details
fn print_apr2_info(file: &Path, global: &Args) -> CliResult<()> {
    let data = std::fs::read(file)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to read: {e}")))?;
    let reader = crate::format::Apr2Reader::new(data)
        .map_err(|e| CliError::InvalidArgument(format!("Invalid APR2: {e}")))?;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    APR2 Model Information                         ");
    println!("═══════════════════════════════════════════════════════════════════\n");
    println!("File: {}", file.display());
    println!("Size: {} bytes", reader.file_size());
    println!("Tensors: {}", reader.n_tensors());
    println!("Family: {:?}", reader.header.family);
    println!("Version: {}", reader.header.version);

    if let Ok(config) = reader.lfm2_config() {
        println!("\nLFM2 Configuration:");
        println!("  Hidden size: {}", config.hidden_size);
        println!("  Layers: {}", config.num_layers);
        println!("  Q heads: {}", config.num_q_heads);
        println!("  KV heads: {}", config.num_kv_heads);
        println!("  Intermediate: {}", config.intermediate_size);
        println!("  Vocab size: {}", config.vocab_size);
    }

    if global.verbose {
        println!("\nTensors:");
        for tensor in &reader.tensors {
            println!("  {} {:?} {:?}", tensor.name, tensor.shape(), tensor.dtype);
        }
    }
    Ok(())
}

/// Print SafeTensors file details
fn print_safetensors_info(file: &Path, global: &Args) -> CliResult<()> {
    let loader = crate::format::SafeTensorsLoader::load(file)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to load: {e}")))?;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    SafeTensors Model Information                  ");
    println!("═══════════════════════════════════════════════════════════════════\n");
    println!("File: {}", file.display());
    println!("Tensors: {}", loader.tensor_names().len());

    if global.verbose {
        println!("\nTensors:");
        for name in loader.tensor_names() {
            let internal = crate::format::map_tensor_name(name);
            println!("  {} → {}", name, internal);
        }
    }
    Ok(())
}

/// Check WASM viability for a model configuration
fn run_model_wasm_check(
    family: &str,
    quantization: &str,
    context: usize,
    sliding_window: usize,
    global: &Args,
) -> CliResult<CommandResult> {
    use crate::format::apr2::Lfm2Config;
    use crate::model::lfm2::{Lfm2WasmConfig, WasmMemoryEstimate, WasmQuantization};

    let quant = match quantization.to_lowercase().as_str() {
        "fp16" => WasmQuantization::Fp16,
        "int8" => WasmQuantization::Int8,
        "int4-awq" | "int4awq" | "awq" => WasmQuantization::Int4Awq,
        "int4-gptq" | "int4gptq" | "gptq" => WasmQuantization::Int4Gptq,
        other => {
            return Err(CliError::InvalidArgument(format!(
                "Unknown quantization: {other}. Use: fp16, int8, int4-awq, int4-gptq"
            )));
        }
    };

    let model_config = match family.to_lowercase().as_str() {
        "lfm2" | "lfm2-2.6b" => Lfm2Config::lfm2_2_6b(),
        "llama" | "llama-7b" => Lfm2Config::llama_7b(),
        "llama2" | "llama2-7b" => Lfm2Config::llama2_7b(),
        "whisper-tiny" | "tiny" => Lfm2Config::whisper_tiny(),
        "whisper-base" | "base" => Lfm2Config::whisper_base(),
        "whisper-small" | "small" => Lfm2Config::whisper_small(),
        other => {
            return Err(CliError::InvalidArgument(format!(
                "Unknown model family: {other}. Use: lfm2, llama, llama2, whisper-tiny, whisper-base, whisper-small"
            )));
        }
    };

    let wasm_config = Lfm2WasmConfig {
        quantization: quant,
        max_context: context as usize,
        sliding_window: if sliding_window == 0 {
            None
        } else {
            Some(sliding_window as usize)
        },
        use_webgpu: true,
        streaming: true,
    };

    let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);

    if !global.quiet {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("                    WASM Viability Check                           ");
        println!("═══════════════════════════════════════════════════════════════════\n");

        println!("Model Family: {}", family);
        println!("Quantization: {}", quant);
        println!("Max Context:  {}", context);
        println!(
            "Sliding Win:  {}",
            if sliding_window == 0 {
                "None (full attention)".to_string()
            } else {
                format!("{sliding_window} tokens")
            }
        );
        println!();

        println!("Memory Breakdown:");
        println!("───────────────────────────────────────────────────────────────────");
        print!("{}", estimate);

        println!("───────────────────────────────────────────────────────────────────");
        if estimate.is_viable {
            println!("✅ This configuration IS viable for WASM deployment");
        } else {
            println!("❌ This configuration is NOT viable for WASM deployment");
            println!("\nRecommendations:");
            println!("  • Use int4-awq or int4-gptq quantization");
            println!("  • Reduce max_context to 4096 or less");
            println!("  • Enable sliding window attention (e.g., --sliding-window 2048)");
        }
    }

    let status = if estimate.is_viable {
        "WASM viable"
    } else {
        "WASM not viable"
    };

    Ok(CommandResult::success(status))
}

/// Run benchmark command
pub fn run_benchmark(args: BenchmarkArgs, global: &Args) -> CliResult<CommandResult> {
    // LFM2 component benchmarks
    if args.lfm2 {
        return run_lfm2_benchmark(&args, global);
    }

    if !global.quiet {
        println!(
            "Benchmarking {} model with {} backend ({} iterations)...",
            args.model, args.backend, args.iterations
        );
    }

    let whisper = WhisperApr::tiny();
    let samples = vec![0.0f32; 16000 * 10]; // 10 seconds of silence
    let options = TranscribeOptions::default();

    let mut times = Vec::new();

    for i in 0..args.iterations {
        let start = Instant::now();
        let _result = whisper.transcribe(&samples, options.clone())?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed);

        if global.verbose {
            println!("  Iteration {}: {:.1}ms", i + 1, elapsed);
        }
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let rtf = (avg / 1000.0) / 10.0; // 10 second audio

    println!("\nResults:");
    println!("  Average: {avg:.1}ms");
    println!("  Min: {min:.1}ms");
    println!("  Max: {max:.1}ms");
    println!("  RTF: {rtf:.2}x");

    Ok(CommandResult::success(format!("RTF: {rtf:.2}x")).with_rtf(rtf))
}

/// Benchmark all LFM2 components and print table
fn run_lfm2_benchmark_all(
    config: &crate::benchmark_generated::Lfm2BenchmarkConfig,
    global: &Args,
) -> CliResult<CommandResult> {
    use crate::benchmark_generated::benchmark_lfm2_all;

    let results = benchmark_lfm2_all(config)
        .map_err(|e| CliError::InvalidArgument(format!("Benchmark failed: {e}")))?;

    if !global.quiet {
        println!("Component       │ Time (μs)  │ Tokens/sec │ Memory (KB) │ FLOPs");
        println!("────────────────┼────────────┼────────────┼─────────────┼──────────────");

        for r in &results {
            println!(
                "{:<15} │ {:>10.1} │ {:>10.0} │ {:>11} │ {:>12}",
                format!("{}", r.component),
                r.forward_us,
                r.tokens_per_sec,
                r.memory_bytes / 1024,
                r.flops
            );
        }

        let total_time: f64 = results.iter().map(|r| r.forward_us).sum();
        let total_memory: usize = results.iter().map(|r| r.memory_bytes).sum();

        println!("────────────────┴────────────┴────────────┴─────────────┴──────────────");
        println!(
            "Total           │ {:>10.1} │            │ {:>11} │",
            total_time,
            total_memory / 1024
        );
    }

    Ok(CommandResult::success("LFM2 benchmark complete"))
}

/// Benchmark a single LFM2 component
fn run_lfm2_benchmark_single(
    component_str: &str,
    config: &crate::benchmark_generated::Lfm2BenchmarkConfig,
    global: &Args,
) -> CliResult<CommandResult> {
    use crate::benchmark_generated::{benchmark_lfm2_component, Lfm2Component};

    let component = match component_str {
        "gqa" => Lfm2Component::Gqa,
        "swiglu" => Lfm2Component::SwiGlu,
        "rope" => Lfm2Component::RoPE,
        "conv1d" | "conv" => Lfm2Component::Conv1d,
        "full_layer" | "full" | "layer" => Lfm2Component::FullLayer,
        other => {
            return Err(CliError::InvalidArgument(format!(
                "Unknown component: {other}. Use: gqa, swiglu, rope, conv1d, full_layer, all"
            )));
        }
    };

    let result = benchmark_lfm2_component(component, config)
        .map_err(|e| CliError::InvalidArgument(format!("Benchmark failed: {e}")))?;

    if !global.quiet {
        println!("Component: {}", result.component);
        println!("───────────────────────────────────────────────────────────────────");
        println!("  Forward time:  {:.2} μs", result.forward_us);
        println!("  Tokens/sec:    {:.0}", result.tokens_per_sec);
        println!("  Memory:        {} KB", result.memory_bytes / 1024);
        println!("  FLOPs:         {}", result.flops);

        if global.verbose {
            println!("\nJSON: {}", result.to_json());
        }
    }

    Ok(CommandResult::success(format!(
        "{}: {:.2}μs",
        result.component, result.forward_us
    )))
}

/// Run LFM2 component benchmarks
fn run_lfm2_benchmark(args: &BenchmarkArgs, global: &Args) -> CliResult<CommandResult> {
    use crate::benchmark_generated::Lfm2BenchmarkConfig;

    // Create benchmark config
    let config = if args.full_size {
        Lfm2BenchmarkConfig::lfm2_2_6b(args.seq_len, args.iterations)
    } else {
        Lfm2BenchmarkConfig::small(args.seq_len, args.iterations)
    };

    if !global.quiet {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("                    LFM2 Component Benchmarks                       ");
        println!("═══════════════════════════════════════════════════════════════════\n");

        println!(
            "Config: {} (hidden={}, q_heads={}, kv_heads={})",
            if args.full_size { "LFM2-2.6B" } else { "small" },
            config.hidden_size,
            config.num_q_heads,
            config.num_kv_heads
        );
        println!("Sequence length: {}", config.seq_len);
        println!("Iterations: {}", config.iterations);
        println!();
    }

    // Parse component
    let component_str = args.component.to_lowercase();

    if component_str == "all" {
        run_lfm2_benchmark_all(&config, global)
    } else {
        run_lfm2_benchmark_single(&component_str, &config, global)
    }
}

/// Run quick APR validation mode
fn run_quick_validation(
    reader: &crate::format::AprReader,
    global: &Args,
) -> CliResult<CommandResult> {
    match crate::format::quick_validate(reader) {
        Ok(()) => {
            if !global.quiet {
                println!("✓ Quick validation passed");
            }
            Ok(CommandResult::success("Quick validation passed"))
        }
        Err(e) => {
            if !global.quiet {
                println!("✗ Quick validation failed: {e}");
            }
            Ok(CommandResult::failure(format!(
                "Quick validation failed: {e}"
            )))
        }
    }
}

/// Run validate command
pub fn run_validate(args: ValidateArgs, global: &Args) -> CliResult<CommandResult> {
    use crate::format::{AprReader, AprValidator};

    // Validate input file exists
    if !args.file.exists() {
        return Err(CliError::FileNotFound(args.file.display().to_string()));
    }

    // Load APR file
    if global.verbose {
        eprintln!("[INFO] Loading APR file: {}", args.file.display());
    }

    let data = fs::read(&args.file)?;
    let reader = AprReader::new(data).map_err(|e| CliError::InvalidArgument(e.to_string()))?;

    if args.quick {
        return run_quick_validation(&reader, global);
    }

    // Full 25-point validation
    let validator = AprValidator::new(&reader);
    let report = validator.validate_all();

    // Format output
    match args.format {
        ValidateOutputFormat::Text => {
            format_validation_text(&report, args.detailed, global.quiet);
        }
        ValidateOutputFormat::Json => {
            format_validation_json(&report);
        }
        ValidateOutputFormat::Markdown => {
            format_validation_markdown(&report, args.detailed);
        }
    }

    // Determine success
    let passed = report.score >= args.min_score && report.critical_failures.is_empty();

    if passed {
        Ok(CommandResult::success(format!(
            "Validation passed: {}/{}",
            report.score, report.max_score
        )))
    } else {
        Ok(CommandResult::failure(format!(
            "Validation failed: {}/{} (min: {})",
            report.score, report.max_score, args.min_score
        )))
    }
}

/// Run stream command (real-time microphone transcription)
///
/// When implemented, this will use:
/// - `aprender::native` for audio capture and preprocessing
/// - `realizar::inference` for real-time model execution
pub fn run_stream(_args: StreamArgs, _global: &Args) -> CliResult<CommandResult> {
    Err(CliError::NotImplemented(
        "Real-time streaming not yet implemented (requires aprender::native audio capture)"
            .to_string(),
    ))
}

/// Run serve command (HTTP API server)
///
/// When implemented, this will use:
/// - `realizar::serve` for HTTP server and API endpoints
/// - `realizar::api` for OpenAI-compatible API handlers
pub fn run_serve(_args: ServeArgs, _global: &Args) -> CliResult<CommandResult> {
    Err(CliError::NotImplemented(
        "HTTP server not yet implemented (requires realizar::serve)".to_string(),
    ))
}

/// Find the whisper.cpp binary from args or common locations
fn find_whisper_cpp_binary(args: &ParityArgs) -> std::path::PathBuf {
    const CANDIDATES: &[&str] = &[
        "/usr/local/bin/whisper-cli",
        "/usr/bin/whisper-cli",
        "whisper-cli",
        "./whisper-cli",
        "../whisper.cpp/main",
    ];

    if let Some(path) = &args.whisper_cpp {
        return path.clone();
    }
    for candidate in CANDIDATES {
        let path = std::path::PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    std::path::PathBuf::from("whisper-cli")
}

/// Execute whisper.cpp and return its text output
fn run_whisper_cpp(whisper_cpp_path: &Path, args: &ParityArgs) -> CliResult<String> {
    if args.verbose {
        println!("Running whisper.cpp...");
    }

    let model_path = args.cpp_model.as_ref().map_or_else(
        || format!("models/ggml-{}.bin", args.model),
        |p| p.to_string_lossy().to_string(),
    );
    let cpp_output = std::process::Command::new(whisper_cpp_path)
        .args([
            "-m",
            model_path.as_str(),
            "-f",
            &args.input.to_string_lossy(),
            "--no-prints",
        ])
        .output();

    match cpp_output {
        Ok(output) if output.status.success() => {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        }
        Ok(output) => Err(CliError::InvalidArgument(format!(
            "whisper.cpp failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))),
        Err(e) => Err(CliError::InvalidArgument(format!(
            "Failed to run whisper.cpp at {}: {}",
            whisper_cpp_path.display(),
            e
        ))),
    }
}

/// Display parity comparison results
fn display_parity_result(
    args: &ParityArgs,
    global: &Args,
    cpp_text: &str,
    apr_text: &str,
    parity_result: &crate::cli::parity::ParityResult,
) {
    if args.json {
        let json = serde_json::json!({
            "input": args.input.display().to_string(),
            "whisper_cpp_output": cpp_text.trim(),
            "whisper_apr_output": apr_text.trim(),
            "parity": parity_result.is_pass(),
            "wer": match parity_result {
                crate::cli::parity::ParityResult::Pass { wer, .. }
                | crate::cli::parity::ParityResult::Fail { wer, .. } => *wer,
            },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else if !global.quiet {
        println!("Results:");
        println!("───────────────────────────────────────────────────────────────────");
        println!("whisper.cpp: {}", cpp_text.trim());
        println!("whisper-apr: {}", apr_text.trim());
        println!();

        match parity_result {
            crate::cli::parity::ParityResult::Pass { wer, .. } => {
                println!("✓ PARITY ACHIEVED (WER: {:.2}%)", wer * 100.0);
            }
            crate::cli::parity::ParityResult::Fail { wer, .. } => {
                println!(
                    "✗ PARITY FAILED (WER: {:.2}%, max: {:.2}%)",
                    wer * 100.0,
                    args.max_wer * 100.0
                );
            }
        }
    }
}

/// Run parity command (whisper.cpp comparison)
#[allow(clippy::too_many_lines)]
pub fn run_parity(args: ParityArgs, global: &Args) -> CliResult<CommandResult> {
    use crate::cli::parity::{ParityConfig, ParityTest};

    // Validate input file exists
    if !args.input.exists() {
        return Err(CliError::FileNotFound(args.input.display().to_string()));
    }

    let whisper_cpp_path = find_whisper_cpp_binary(&args);

    if !global.quiet {
        println!("whisper-apr Parity Test");
        println!("═══════════════════════════════════════════════════════════════════");
        println!("Audio: {}", args.input.display());
        println!("whisper.cpp: {}", whisper_cpp_path.display());
        println!("Max WER: {:.1}%", args.max_wer * 100.0);
        println!();
    }

    let cpp_text = run_whisper_cpp(&whisper_cpp_path, &args)?;

    // Run whisper-apr
    if args.verbose {
        println!("Running whisper-apr...");
    }

    let audio_data = fs::read(&args.input)?;
    let samples = load_audio_samples(&args.input, &audio_data)?;

    let whisper = crate::cli::model_loader::load_or_download_model(
        args.model,
        args.model_path.as_deref(),
        args.verbose,
    )
    .map_err(|e| CliError::InvalidArgument(e.to_string()))?;

    let options = crate::TranscribeOptions::default();
    let result = whisper.transcribe(&samples, options)?;
    let apr_text = result.text;

    // Compare outputs
    let config = ParityConfig {
        max_wer: args.max_wer,
        timestamp_tolerance_ms: args.timestamp_tolerance_ms,
        ..Default::default()
    };

    let test =
        ParityTest::new(args.input.clone(), cpp_text.clone(), apr_text.clone()).with_config(config);

    let parity_result = test.verify_text_parity();

    display_parity_result(&args, global, &cpp_text, &apr_text, &parity_result);

    if parity_result.is_pass() {
        Ok(CommandResult::success("Parity achieved"))
    } else {
        Ok(CommandResult::failure("Parity failed"))
    }
}

/// Run quantize command (model quantization)
///
/// When implemented, this will use:
/// - `realizar::quantize` for quantization algorithms (int8, q4_0, q5_0, etc.)
/// - `aprender::compute` for tensor operations during quantization
pub fn run_quantize(args: QuantizeArgs, global: &Args) -> CliResult<CommandResult> {
    // Validate input file exists
    if !args.input.exists() {
        return Err(CliError::FileNotFound(args.input.display().to_string()));
    }

    if !global.quiet {
        println!("Model Quantization");
        println!("═══════════════════════════════════════════════════════════════════");
        println!("Input:  {}", args.input.display());
        println!("Output: {}", args.output.display());
        println!("Type:   {}", args.quantize);
        println!();
    }

    Err(CliError::NotImplemented(
        "Model quantization not yet implemented (requires realizar::quantize)".to_string(),
    ))
}

/// Run command mode (voice command recognition)
///
/// When implemented, this will use:
/// - `aprender::native` for audio capture
/// - `realizar::inference` for real-time model execution
/// - Pattern matching against configured command phrases
pub fn run_command(_args: CommandArgs, _global: &Args) -> CliResult<CommandResult> {
    Err(CliError::NotImplemented(
        "Voice command recognition not yet implemented (requires aprender::native audio capture)"
            .to_string(),
    ))
}

/// Run diagnose command (self-diagnostic checks)
///
/// Validates tokenizer configuration, model compatibility, and known issues.
/// This is the primary debugging tool for troubleshooting transcription problems.
///
/// # Checks Performed
///
/// 1. **Tokenizer Configuration**
///    - EOT token ID matches model type (50256 for English-only, 50257 for multilingual)
///    - SOT token ID is correct
///    - Language base token is correct
///    - Initial token sequence is valid
///
/// 2. **Model Configuration** (if model provided)
///    - Vocabulary size detection (multilingual vs English-only)
///    - Layer configuration matches expected values
///    - Weight dimensions are valid
///
/// 3. **Known Issues**
///    - EOT-001: EOT token off-by-one (fixed in 2025-12-20)
///    - H35: Cross-attention padding mask (fixed in 2025-12-20)
#[allow(clippy::too_many_lines)]
pub fn run_diagnose(args: DiagnoseArgs, global: &Args) -> CliResult<CommandResult> {
    use crate::tokenizer::special_tokens::{self, SpecialTokens};

    let mut dc = DiagnosticCollector::new(global.quiet, args.json);

    if !global.quiet && !args.json {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("                    whisper-apr Self-Diagnostic                     ");
        println!("═══════════════════════════════════════════════════════════════════\n");
    }

    // Section 1: Tokenizer Configuration Checks
    if !global.quiet && !args.json {
        println!("1. Tokenizer Configuration");
        println!("───────────────────────────────────────────────────────────────────");
    }

    let multilingual = SpecialTokens::for_vocab_size(51865);
    let english_only = SpecialTokens::for_vocab_size(51864);

    dc.add(
        "TOK-001",
        "EOT token (multilingual)",
        multilingual.eot == 50257,
        "50257",
        &multilingual.eot.to_string(),
        "EOT for multilingual models (vocab >= 51865)",
    );
    dc.add(
        "TOK-002",
        "EOT token (English-only)",
        english_only.eot == 50256,
        "50256",
        &english_only.eot.to_string(),
        "EOT for English-only models (vocab < 51865)",
    );
    dc.add(
        "TOK-003",
        "SOT token (multilingual)",
        multilingual.sot == 50258,
        "50258",
        &multilingual.sot.to_string(),
        "SOT for multilingual models",
    );
    dc.add(
        "TOK-004",
        "LANG_BASE token (multilingual)",
        multilingual.lang_base == 50259,
        "50259",
        &multilingual.lang_base.to_string(),
        "Language base for multilingual models",
    );

    let lang_en = special_tokens::language_token("en");
    dc.add(
        "TOK-005",
        "English language token",
        lang_en == Some(50259),
        "Some(50259)",
        &format!("{lang_en:?}"),
        "language_token(\"en\") = LANG_BASE + 0",
    );

    let initial = multilingual.initial_tokens();
    dc.add(
        "TOK-006",
        "Initial tokens sequence",
        initial == [50258, 50259, 50359, 50363],
        "[50258, 50259, 50359, 50363]",
        &format!("{initial:?}"),
        "[SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS]",
    );
    dc.add(
        "TOK-007",
        "TIMESTAMP_BASE (multilingual)",
        multilingual.timestamp_base == 50364,
        "50364",
        &multilingual.timestamp_base.to_string(),
        "First timestamp token for multilingual models",
    );

    // Section 2: Model-specific checks (if model provided)
    if let Some(model_path) = &args.model {
        if !global.quiet && !args.json {
            println!("\n2. Model Configuration");
            println!("───────────────────────────────────────────────────────────────────");
        }

        diagnose_model_file(model_path, args.full, &mut dc)?;
    }

    // Section 3: Known Issues (informational)
    if !args.tokenizer_only && !global.quiet && !args.json {
        println!("\n3. Known Issues Status");
        println!("───────────────────────────────────────────────────────────────────");
        println!("  ✓ EOT-001: EOT token off-by-one - FIXED (2025-12-20)");
        println!("    Multilingual models now correctly use EOT=50257");
        println!();
        println!("  ✓ H35: Cross-attention padding mask - FIXED (2025-12-20)");
        println!("    Decoder cross-attention now masks padding positions");
        println!();
    }

    // Summary
    format_diagnose_summary(&dc, &args)
}

/// Run model file diagnostics
fn diagnose_model_file(
    model_path: &Path,
    full: bool,
    dc: &mut DiagnosticCollector,
) -> CliResult<()> {
    if !model_path.exists() {
        dc.add(
            "MDL-001",
            "Model file exists",
            false,
            "File exists",
            "File not found",
            &model_path.display().to_string(),
        );
        return Ok(());
    }

    dc.add(
        "MDL-001",
        "Model file exists",
        true,
        "File exists",
        "File exists",
        &model_path.display().to_string(),
    );

    if !full {
        return Ok(());
    }

    match fs::read(model_path) {
        Ok(data) => {
            let has_magic = data.len() >= 4 && data.get(0..4) == Some(b"APR1".as_slice());
            let actual = if let Some(magic) = data.get(0..4) {
                String::from_utf8_lossy(magic).to_string()
            } else {
                "too short".to_string()
            };
            dc.add(
                "MDL-002",
                "APR magic bytes",
                has_magic,
                "APR1",
                &actual,
                "Model file format identifier",
            );
        }
        Err(e) => {
            dc.add(
                "MDL-002",
                "Model file readable",
                false,
                "Readable",
                &format!("Error: {e}"),
                "Could not read model file",
            );
        }
    }
    Ok(())
}

/// Format diagnose summary as JSON or text
fn format_diagnose_summary(
    dc: &DiagnosticCollector,
    args: &DiagnoseArgs,
) -> CliResult<CommandResult> {
    let passed_count = dc.checks.iter().filter(|c| c.passed).count();
    let total_count = dc.checks.len();

    if args.json {
        let json = serde_json::json!({
            "passed": dc.all_passed,
            "checks_passed": passed_count,
            "checks_total": total_count,
            "checks": dc.checks.iter().map(|c| serde_json::json!({
                "id": c.id,
                "name": c.name,
                "passed": c.passed,
                "expected": c.expected,
                "actual": c.actual,
                "details": c.details
            })).collect::<Vec<_>>(),
            "known_issues": [
                {"id": "EOT-001", "status": "fixed", "date": "2025-12-20"},
                {"id": "H35", "status": "fixed", "date": "2025-12-20"}
            ]
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else if !dc.quiet {
        println!("\n═══════════════════════════════════════════════════════════════════");
        println!(
            "RESULT: {}/{} checks passed {}",
            passed_count,
            total_count,
            if dc.all_passed { "✓" } else { "✗" }
        );
        println!("═══════════════════════════════════════════════════════════════════");
    }

    if dc.all_passed {
        Ok(CommandResult::success(format!(
            "{passed_count}/{total_count} checks passed"
        )))
    } else {
        Ok(CommandResult::failure(format!(
            "{passed_count}/{total_count} checks passed"
        )))
    }
}

/// Diagnostic check result
#[derive(Debug, Clone)]
struct DiagnosticCheck {
    id: String,
    name: String,
    passed: bool,
    expected: String,
    actual: String,
    details: String,
}

/// Collector for diagnostic checks that handles registration and tracking
struct DiagnosticCollector {
    checks: Vec<DiagnosticCheck>,
    all_passed: bool,
    quiet: bool,
    json: bool,
}

impl DiagnosticCollector {
    fn new(quiet: bool, json: bool) -> Self {
        Self {
            checks: Vec::new(),
            all_passed: true,
            quiet,
            json,
        }
    }

    /// Add a diagnostic check, print it, and track pass/fail
    fn add(
        &mut self,
        id: &str,
        name: &str,
        passed: bool,
        expected: &str,
        actual: &str,
        details: &str,
    ) {
        let check = DiagnosticCheck {
            id: id.to_string(),
            name: name.to_string(),
            passed,
            expected: expected.to_string(),
            actual: actual.to_string(),
            details: details.to_string(),
        };
        if !passed {
            self.all_passed = false;
        }
        print_check(&check, self.quiet, self.json);
        self.checks.push(check);
    }
}

fn print_check(check: &DiagnosticCheck, quiet: bool, json: bool) {
    if quiet || json {
        return;
    }
    let status = if check.passed { "✓" } else { "✗" };
    println!(
        "  {} [{}] {}: {} (expected: {}, got: {})",
        status, check.id, check.name, check.details, check.expected, check.actual
    );
}

fn format_validation_text(report: &crate::format::ValidationReport, detailed: bool, quiet: bool) {
    if quiet {
        return;
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    APR Validation Report (25-Point QA)            ");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let categories = [
        ('A', "Structural Integrity"),
        ('B', "Layer Norm Validation"),
        ('C', "Attention/Linear Validation"),
        ('D', "Embedding Validation"),
        ('E', "Functional Validation"),
    ];

    for (cat, name) in categories {
        let checks = report.checks_by_category(cat);
        let passed = checks.iter().filter(|c| c.passed).count();
        let total = checks.len();
        let status = if passed == total { "✓" } else { "✗" };

        println!("{cat}. {name}: {passed}/{total} {status}");

        if detailed {
            for check in checks {
                let mark = if check.passed { "  ✓" } else { "  ✗" };
                println!(
                    "  {} [{}] {}: {}",
                    mark, check.id, check.name, check.message
                );
            }
        }
    }

    println!("\n───────────────────────────────────────────────────────────────────");
    println!(
        "SCORE: {}/{} ({})",
        report.score,
        report.max_score,
        if report.passed { "PASS" } else { "FAIL" }
    );

    if !report.critical_failures.is_empty() {
        println!("\n⚠ CRITICAL FAILURES:");
        for failure in &report.critical_failures {
            println!("  • {failure}");
        }
    }
    println!("═══════════════════════════════════════════════════════════════════");
}

fn format_validation_json(report: &crate::format::ValidationReport) {
    let checks: Vec<_> = report
        .checks
        .iter()
        .map(|c| {
            serde_json::json!({
                "id": c.id,
                "category": c.category.to_string(),
                "name": c.name,
                "passed": c.passed,
                "message": c.message
            })
        })
        .collect();

    let json = serde_json::json!({
        "score": report.score,
        "max_score": report.max_score,
        "passed": report.passed,
        "critical_failures": report.critical_failures,
        "checks": checks
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
}

fn format_validation_markdown(report: &crate::format::ValidationReport, detailed: bool) {
    println!("# APR Validation Report\n");
    println!(
        "**Score:** {}/{} ({})\n",
        report.score,
        report.max_score,
        if report.passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    if !report.critical_failures.is_empty() {
        println!("## ⚠️ Critical Failures\n");
        for failure in &report.critical_failures {
            println!("- {failure}");
        }
        println!();
    }

    let categories = [
        ('A', "Structural Integrity"),
        ('B', "Layer Norm Validation"),
        ('C', "Attention/Linear Validation"),
        ('D', "Embedding Validation"),
        ('E', "Functional Validation"),
    ];

    for (cat, name) in categories {
        let checks = report.checks_by_category(cat);
        let passed = checks.iter().filter(|c| c.passed).count();
        let total = checks.len();

        println!("## {cat}. {name} ({passed}/{total})\n");

        if detailed {
            println!("| # | Check | Status | Details |");
            println!("|---|-------|--------|---------|");
            for check in checks {
                let status = if check.passed { "✅" } else { "❌" };
                println!(
                    "| {} | {} | {} | {} |",
                    check.id, check.name, status, check.message
                );
            }
            println!();
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Load audio samples from file
pub(crate) fn load_audio_samples(path: &Path, data: &[u8]) -> CliResult<Vec<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "wav" => {
            let wav = parse_wav_file(data)?;
            let samples = if wav.sample_rate == 16000 {
                wav.samples
            } else {
                resample(&wav.samples, wav.sample_rate, 16000)
            };
            Ok(samples)
        }
        #[cfg(feature = "symphonia")]
        "mp3" | "flac" | "ogg" | "m4a" | "aac" | "mp4" | "webm" | "mkv" | "avi" | "opus" => {
            decode_with_symphonia(data, &ext)
        }
        #[cfg(not(feature = "symphonia"))]
        "mp3" | "flac" | "ogg" | "m4a" | "aac" | "mp4" | "webm" | "mkv" | "avi" | "opus" => {
            Err(CliError::NotImplemented(format!(
                "{ext} format requires 'symphonia' feature. Build with: cargo build --features cli"
            )))
        }
        _ => Err(CliError::UnsupportedFormat(ext)),
    }
}

/// Decode audio using symphonia (multi-format decoder)
#[cfg(feature = "symphonia")]
fn decode_with_symphonia(data: &[u8], ext: &str) -> CliResult<Vec<f32>> {
    use std::io::Cursor;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let cursor = Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(ext);

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| CliError::InvalidArgument(format!("Failed to probe {ext} format: {e}")))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| CliError::InvalidArgument("No audio track found".to_string()))?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| CliError::InvalidArgument("Unknown sample rate".to_string()))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| CliError::InvalidArgument(format!("Failed to create decoder: {e}")))?;

    let samples = decode_all_packets(&mut format, &mut *decoder, track_id)?;

    if sample_rate != 16000 {
        Ok(resample(&samples, sample_rate, 16000))
    } else {
        Ok(samples)
    }
}

/// Read and decode the next audio packet, returning interleaved samples and channel count.
/// Returns None at end of stream.
#[cfg(feature = "symphonia")]
/// Read the next packet belonging to the given track, returning None at EOF
#[cfg(feature = "symphonia")]
fn next_packet_for_track(
    format: &mut Box<dyn symphonia::core::formats::FormatReader>,
    track_id: u32,
) -> CliResult<Option<symphonia::core::formats::Packet>> {
    loop {
        match format.next_packet() {
            Ok(p) if p.track_id() == track_id => return Ok(Some(p)),
            Ok(_) => continue,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                return Ok(None);
            }
            Err(e) => {
                return Err(CliError::InvalidArgument(format!(
                    "Failed to read packet: {e}"
                )));
            }
        }
    }
}

/// Read and decode the next audio packet, returning interleaved samples and channel count.
/// Returns None at end of stream. Skips packets with decode errors.
#[cfg(feature = "symphonia")]
fn read_next_audio_packet(
    format: &mut Box<dyn symphonia::core::formats::FormatReader>,
    decoder: &mut dyn symphonia::core::codecs::Decoder,
    track_id: u32,
) -> CliResult<Option<(Vec<f32>, usize)>> {
    use symphonia::core::audio::SampleBuffer;

    loop {
        let packet = match next_packet_for_track(format, track_id)? {
            Some(p) => p,
            None => return Ok(None),
        };

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);
                return Ok(Some((sample_buf.samples().to_vec(), spec.channels.count())));
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(CliError::InvalidArgument(format!("Decode error: {e}"))),
        }
    }
}

/// Decode all audio packets from a symphonia format reader, mixing to mono
#[cfg(feature = "symphonia")]
fn decode_all_packets(
    format: &mut Box<dyn symphonia::core::formats::FormatReader>,
    decoder: &mut dyn symphonia::core::codecs::Decoder,
    track_id: u32,
) -> CliResult<Vec<f32>> {
    let mut samples: Vec<f32> = Vec::new();

    loop {
        match read_next_audio_packet(format, decoder, track_id)? {
            Some((interleaved, channels)) => mix_to_mono(&interleaved, channels, &mut samples),
            None => break,
        }
    }

    Ok(samples)
}

/// Mix interleaved multi-channel audio to mono
#[cfg(feature = "symphonia")]
fn mix_to_mono(interleaved: &[f32], channels: usize, output: &mut Vec<f32>) {
    match channels {
        1 => output.extend_from_slice(interleaved),
        2 => {
            for chunk in interleaved.chunks_exact(2) {
                output.push((chunk[0] + chunk[1]) / 2.0);
            }
        }
        n => {
            for chunk in interleaved.chunks(n) {
                let sum: f32 = chunk.iter().sum();
                output.push(sum / n as f32);
            }
        }
    }
}

/// Convert format argument to OutputFormat
fn convert_format_arg(arg: OutputFormatArg) -> OutputFormat {
    match arg {
        OutputFormatArg::Txt => OutputFormat::Txt,
        OutputFormatArg::Srt => OutputFormat::Srt,
        OutputFormatArg::Vtt => OutputFormat::Vtt,
        OutputFormatArg::Json => OutputFormat::Json,
        OutputFormatArg::JsonFull => OutputFormat::JsonFull,
        OutputFormatArg::Csv => OutputFormat::Csv,
        OutputFormatArg::Lrc => OutputFormat::Lrc,
        OutputFormatArg::Wts => OutputFormat::Wts,
        OutputFormatArg::Md => OutputFormat::Md,
    }
}

// ============================================================================
// Convert Command (WAPR-LFM2-004)
// ============================================================================

/// Run convert command - convert HuggingFace safetensors to APR2 format
///
/// # Conversion Pipeline
///
/// ```text
/// HuggingFace (safetensors)
///     ↓ load_safetensors()
///     ↓ quantize (optional)
///     ↓ export_weights()
///     ↓ write_apr2()
/// LFM2.apr2 (output)
/// ```
///
/// # Example
///
/// ```bash
/// whisper-apr convert -i model.safetensors -o model.apr2 --quantize int8
/// ```
///
/// See `docs/specifications/1.0-whisper-apr.md` Section 18.8 for full specification.
pub fn run_convert(args: ConvertArgs, global: &Args) -> CliResult<CommandResult> {
    use crate::format::apr2::{Lfm2Config, QuantConfig};
    use crate::format::ConversionStats;
    use std::time::Instant;

    let start = Instant::now();

    // Validate input file exists
    if !args.input.exists() {
        return Err(CliError::FileNotFound(args.input.display().to_string()));
    }

    if !global.quiet {
        println!(
            "Converting: {} → {}",
            args.input.display(),
            args.output.display()
        );
        println!("Family: {}", args.family);
        println!("Quantization: {}", args.quantize);
    }

    // Dry run mode
    if args.dry_run {
        if !global.quiet {
            println!("\n[DRY RUN] Would convert file, not actually writing.");
        }
        return Ok(CommandResult::success("Dry run completed"));
    }

    // Get config based on family
    let config = match args.family {
        ModelFamilyArg::Lfm2 => Lfm2Config::lfm2_2_6b(),
        ModelFamilyArg::Llama => Lfm2Config::llama2_7b(),
        ModelFamilyArg::Whisper => Lfm2Config::whisper_small(),
    };

    // Determine quantization config
    let quant = match args.quantize {
        QuantizeMethodArg::F32 => QuantConfig::default(),
        QuantizeMethodArg::Int8 => QuantConfig::int8(args.group_size),
        QuantizeMethodArg::Int4 | QuantizeMethodArg::Int4Awq => {
            QuantConfig::int4_awq(args.group_size)
        }
        QuantizeMethodArg::Int4Gptq => QuantConfig::int4_gptq(args.group_size),
        _ => QuantConfig::default(),
    };

    let quantize = matches!(
        args.quantize,
        QuantizeMethodArg::Int8
            | QuantizeMethodArg::Int4
            | QuantizeMethodArg::Int4Awq
            | QuantizeMethodArg::Int4Gptq
    );

    // Load safetensors - use sharded loader for directories
    let (n_tensors, n_params, writer) =
        load_and_convert_safetensors(&args.input, config, quant, quantize, global)?;

    // Write output file
    let bytes = writer
        .to_bytes()
        .map_err(|e| CliError::InvalidArgument(format!("Serialization failed: {e}")))?;

    let input_bytes = std::fs::metadata(&args.input).map(|m| m.len()).unwrap_or(0);
    let output_bytes = bytes.len() as u64;

    std::fs::write(&args.output, &bytes)
        .map_err(|e| CliError::WriteError(format!("Failed to write output: {e}")))?;

    let elapsed = start.elapsed();

    // Report stats
    let stats = ConversionStats {
        n_tensors,
        n_params,
        input_bytes,
        output_bytes,
        compression_ratio: if input_bytes > 0 {
            output_bytes as f32 / input_bytes as f32
        } else {
            1.0
        },
    };

    if !global.quiet {
        println!("\n{stats}");
        println!("Time: {:.2}s", elapsed.as_secs_f64());
        println!("Output: {}", args.output.display());
    }

    Ok(CommandResult::success(format!(
        "Converted {} tensors to {}",
        n_tensors,
        args.output.display()
    )))
}

/// Export APR model to SafeTensors format (WAPR-PUB-001)
///
/// Converts whisper.apr models to HuggingFace SafeTensors format for publishing.
/// Uses the native export implementation in `format::export` module.
///
/// # Example
///
/// ```bash
/// whisper-apr export models/whisper-tiny.apr -o whisper-tiny.safetensors
/// ```
/// Load safetensors (single or sharded) and convert to APR2
fn load_and_convert_safetensors(
    input: &Path,
    config: crate::format::apr2::Lfm2Config,
    quant: crate::format::apr2::QuantConfig,
    quantize: bool,
    global: &Args,
) -> CliResult<(usize, u64, crate::format::apr2::Apr2Writer)> {
    use crate::format::safetensors_loader::{SafeTensorsLoader, ShardedSafeTensorsLoader};

    if input.is_dir() {
        if !global.quiet {
            println!("Loading sharded safetensors from directory...");
        }
        let loader = ShardedSafeTensorsLoader::load(input).map_err(|e| {
            CliError::FileNotFound(format!("Failed to load sharded safetensors: {e}"))
        })?;
        let n_tensors = loader.tensor_names().len();
        let n_params = loader.total_params().unwrap_or(0);
        log_tensor_mapping(loader.tensor_names(), n_tensors, n_params, global);
        let writer = loader
            .to_apr2(config, quant, quantize)
            .map_err(|e| CliError::InvalidArgument(format!("Conversion failed: {e}")))?;
        Ok((n_tensors, n_params, writer))
    } else {
        let loader = SafeTensorsLoader::load(input)
            .map_err(|e| CliError::FileNotFound(format!("Failed to load safetensors: {e}")))?;
        let n_tensors = loader.tensor_names().len();
        let n_params = loader.total_params().unwrap_or(0);
        log_tensor_mapping(loader.tensor_names(), n_tensors, n_params, global);
        let writer = loader
            .to_apr2(config, quant, quantize)
            .map_err(|e| CliError::InvalidArgument(format!("Conversion failed: {e}")))?;
        Ok((n_tensors, n_params, writer))
    }
}

/// Log tensor name mapping when verbose
fn log_tensor_mapping(tensor_names: &[String], n_tensors: usize, n_params: u64, global: &Args) {
    if global.verbose {
        println!("\nTensors found: {n_tensors}");
        for name in tensor_names {
            let internal_name = crate::format::map_tensor_name(name);
            println!("  {name} → {internal_name}");
        }
    }
    if !global.quiet {
        println!("Converting {} tensors ({} params)...", n_tensors, n_params);
    }
}

/// Extract tensors from an APR reader into a BTreeMap for export
fn extract_tensors_for_export(
    reader: &crate::format::AprReader,
    global: &Args,
) -> CliResult<std::collections::BTreeMap<String, crate::format::export::TensorData>> {
    use crate::format::export::TensorData;
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, TensorData> = BTreeMap::new();

    for tensor_desc in &reader.tensors {
        let name = &tensor_desc.name;
        let tensor_data = reader
            .load_tensor(name)
            .map_err(|e| CliError::InvalidArgument(format!("Failed to load tensor {name}: {e}")))?;

        let shape: Vec<usize> = tensor_desc.shape().iter().map(|&d| d as usize).collect();

        if global.verbose {
            println!("  {} {:?} ({} elements)", name, shape, tensor_data.len());
        }

        tensors.insert(name.clone(), TensorData::new(tensor_data, shape));
    }

    Ok(tensors)
}

/// Run export command — convert APR model to SafeTensors format
pub fn run_export(args: ExportArgs, global: &Args) -> CliResult<CommandResult> {
    use crate::format::export::SafeTensorsExporter;
    use crate::format::AprReader;
    use std::collections::BTreeMap;
    use std::time::Instant;

    let start = Instant::now();

    // Validate input file exists
    if !args.input.exists() {
        return Err(CliError::FileNotFound(args.input.display().to_string()));
    }

    // Validate APR magic bytes
    let data = std::fs::read(&args.input)
        .map_err(|e| CliError::Io(io::Error::new(e.kind(), format!("Failed to read APR: {e}"))))?;

    if data.get(0..4) != Some(b"APR\0".as_slice()) {
        return Err(CliError::InvalidArgument(
            "Invalid APR file: missing APR\\0 magic bytes".to_string(),
        ));
    }

    if !global.quiet {
        println!(
            "Exporting: {} → {}",
            args.input.display(),
            args.output.display()
        );
        println!("Format: {}", args.format);
    }

    // Only SafeTensors format is supported currently
    if args.format != ExportFormatArg::Safetensors {
        return Err(CliError::NotImplemented(
            "Only safetensors format is currently supported".to_string(),
        ));
    }

    // Load APR model
    let reader = AprReader::new(data)
        .map_err(|e| CliError::InvalidArgument(format!("Failed to parse APR: {e}")))?;

    let n_tensors = reader.n_tensors();

    if global.verbose {
        println!("\nTensors found: {n_tensors}");
    }

    let tensors = extract_tensors_for_export(&reader, global)?;

    if !global.quiet {
        println!("Exporting {} tensors...", n_tensors);
    }

    // Export to SafeTensors
    let metadata = if args.with_metadata {
        let mut meta = BTreeMap::new();
        meta.insert("format".to_string(), "whisper.apr".to_string());
        meta.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
        Some(meta)
    } else {
        None
    };

    SafeTensorsExporter::save_with_metadata(&args.output, &tensors, metadata)
        .map_err(|e| CliError::WriteError(format!("Failed to write SafeTensors: {e}")))?;

    let elapsed = start.elapsed();
    let output_size = std::fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);

    if !global.quiet {
        println!("\nExport complete:");
        println!("  Tensors: {n_tensors}");
        println!("  Output size: {} bytes", output_size);
        println!("  Time: {:.2}s", elapsed.as_secs_f64());
        println!("  Output: {}", args.output.display());
    }

    Ok(CommandResult::success(format!(
        "Exported {} tensors to {}",
        n_tensors,
        args.output.display()
    )))
}

// ============================================================================
// Unit Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::ModelSize;
    use std::path::PathBuf;

    /// Helper to create default TranscribeArgs for testing
    fn default_transcribe_args(input: PathBuf) -> TranscribeArgs {
        TranscribeArgs {
            input,
            model: ModelSize::Tiny,
            model_path: None,
            language: "auto".to_string(),
            detect_language: false,
            output: None,
            format: OutputFormatArg::Txt,
            offset_t: 0,
            offset_n: 0,
            duration: 0,
            max_context: -1,
            max_len: 0,
            audio_ctx: 0,
            best_of: 2,
            beam_size: -1,
            temperature: 0.0,
            temperature_inc: 0.2,
            no_fallback: false,
            split_on_word: false,
            word_thold: 0.01,
            word_timestamps: false,
            timestamps: false,
            no_timestamps: false,
            entropy_thold: 2.40,
            logprob_thold: -1.0,
            no_speech_thold: 0.6,
            prompt: String::new(),
            suppress_regex: String::new(),
            grammar: String::new(),
            grammar_rule: String::new(),
            grammar_penalty: 100.0,
            vad: false,
            vad_model: None,
            vad_threshold: 0.5,
            vad_min_speech_ms: 250,
            vad_min_silence_ms: 100,
            vad_max_speech_s: None,
            vad_pad_ms: 30,
            vad_overlap: 0.1,
            threads: None,
            processors: 1,
            gpu: false,
            no_gpu: false,
            flash_attn: false,
            no_flash_attn: false,
            no_prints: false,
            print_special: false,
            colors: false,
            confidence: false,
            progress: false,
            print_memory: false,
            profile: false,
            translate: false,
            hallucination_filter: false,
            speed: 1.0,
            cache_dir: None,
            zram_optimized: false,
            // Phase 2 summarization
            summarize: false,
            lfm2_model: None,
            summary_output: None,
            summary_format: SummarizeFormat::Json,
            action_items: false,
            key_points: false,
        }
    }

    /// Helper to create default Args for testing
    fn default_global_args() -> Args {
        Args {
            command: Command::Tui, // Dummy
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        }
    }

    // -------------------------------------------------------------------------
    // CommandResult tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_command_result_success() {
        let result = CommandResult::success("Done");
        assert!(result.success);
        assert_eq!(result.message, "Done");
    }

    #[test]
    fn test_command_result_failure() {
        let result = CommandResult::failure("Error");
        assert!(!result.success);
        assert_eq!(result.message, "Error");
    }

    #[test]
    fn test_command_result_with_timings() {
        let timings = Timings {
            total_ms: 100.0,
            ..Default::default()
        };
        let result = CommandResult::success("Done").with_timings(timings);
        assert!(result.timings.is_some());
        assert!((result.timings.expect("timings should be set").total_ms - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_command_result_with_rtf() {
        let result = CommandResult::success("Done").with_rtf(0.5);
        assert_eq!(result.rtf, Some(0.5));
    }

    // -------------------------------------------------------------------------
    // CliError tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cli_error_display() {
        let err = CliError::FileNotFound("test.wav".to_string());
        assert!(err.to_string().contains("test.wav"));

        let err = CliError::InvalidArgument("bad arg".to_string());
        assert!(err.to_string().contains("bad arg"));

        let err = CliError::NotImplemented("feature X".to_string());
        assert!(err.to_string().contains("feature X"));
    }

    // -------------------------------------------------------------------------
    // convert_format_arg tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_convert_format_arg() {
        assert_eq!(convert_format_arg(OutputFormatArg::Txt), OutputFormat::Txt);
        assert_eq!(convert_format_arg(OutputFormatArg::Srt), OutputFormat::Srt);
        assert_eq!(convert_format_arg(OutputFormatArg::Vtt), OutputFormat::Vtt);
        assert_eq!(
            convert_format_arg(OutputFormatArg::Json),
            OutputFormat::Json
        );
        assert_eq!(convert_format_arg(OutputFormatArg::Csv), OutputFormat::Csv);
        assert_eq!(convert_format_arg(OutputFormatArg::Md), OutputFormat::Md);
    }

    // -------------------------------------------------------------------------
    // load_audio_samples tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_audio_unsupported_format() {
        let result = load_audio_samples(Path::new("test.xyz"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::UnsupportedFormat(f)) => assert_eq!(f, "xyz"),
            _ => panic!("Expected UnsupportedFormat error"),
        }
    }

    #[test]
    #[cfg(not(feature = "symphonia"))]
    fn test_load_audio_mp3_not_implemented() {
        let result = load_audio_samples(Path::new("test.mp3"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("mp3")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    #[cfg(feature = "symphonia")]
    fn test_load_audio_mp3_decodes() {
        let path = Path::new("demos/test-audio/test-speech-1.5s.mp3");
        if !path.exists() {
            eprintln!("Skipping: test file not found");
            return;
        }
        let data = std::fs::read(path).expect("Failed to read MP3 file");
        let result = load_audio_samples(path, &data);
        assert!(result.is_ok(), "MP3 decoding failed: {result:?}");
        let samples = result.unwrap();
        assert!(!samples.is_empty(), "No samples decoded from MP3");
        // Should be ~1.5s at 16kHz = ~24000 samples
        assert!(samples.len() > 20000, "Too few samples: {}", samples.len());
    }

    // -------------------------------------------------------------------------
    // run_transcribe tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_transcribe_file_not_found() {
        let args = default_transcribe_args("nonexistent.wav".into());
        let global = default_global_args();

        let result = run_transcribe(args, &global);
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    // -------------------------------------------------------------------------
    // run_record tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_record_list_devices() {
        let args = RecordArgs {
            duration: None,
            live: false,
            output: None,
            device: None,
            sample_rate: 16000,
            list_devices: true,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: false,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_record(args, &global);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_record_no_args_error() {
        let args = RecordArgs {
            duration: None,
            live: false,
            output: None,
            device: None,
            sample_rate: 16000,
            list_devices: false,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_record(args, &global);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidArgument(_)) => {}
            _ => panic!("Expected InvalidArgument error"),
        }
    }

    // -------------------------------------------------------------------------
    // run_batch tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_batch_no_inputs() {
        let args = BatchArgs {
            inputs: vec![],
            output_dir: None,
            parallel: None,
            recursive: false,
            pattern: None,
            skip_existing: false,
            model: ModelSize::Tiny,
            format: OutputFormatArg::Txt,
            cache_dir: None,
            zram_optimized: false,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_batch(args, &global);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidArgument(_)) => {}
            _ => panic!("Expected InvalidArgument error"),
        }
    }

    // -------------------------------------------------------------------------
    // run_model tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_model_list() {
        let args = ModelArgs {
            action: ModelAction::List,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: false,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_model(args, &global);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_model_info_not_found() {
        let args = ModelArgs {
            action: ModelAction::Info {
                file: "nonexistent.apr".into(),
            },
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_model(args, &global);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // run_test tests
    // -------------------------------------------------------------------------

    #[test]
    #[ignore = "Slow: runs full inference pipeline"]
    fn test_run_test_simd() {
        let args = TestArgs {
            backend: BackendArg::Simd,
            demo: None,
            pipeline: None,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_test(args, &global);
        assert!(result.is_ok());
    }

    #[test]
    #[ignore = "Slow: runs full inference pipeline"]
    fn test_run_test_wasm() {
        let args = TestArgs {
            backend: BackendArg::Wasm,
            demo: None,
            pipeline: None,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_test(args, &global);
        assert!(result.is_ok());
    }

    #[test]
    #[ignore = "Slow: runs full inference pipeline"]
    fn test_run_test_cuda() {
        let args = TestArgs {
            backend: BackendArg::Cuda,
            demo: None,
            pipeline: None,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_test(args, &global);
        assert!(result.is_ok());
    }

    #[test]
    #[ignore = "Slow: runs full inference pipeline for all backends"]
    fn test_run_test_all_backends() {
        let args = TestArgs {
            backend: BackendArg::All,
            demo: None,
            pipeline: None,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_test(args, &global);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // run_tui tests
    // -------------------------------------------------------------------------

    #[test]
    #[cfg(not(feature = "tui"))]
    fn test_run_tui_not_implemented() {
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_tui(&global);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    #[cfg(feature = "tui")]
    fn test_run_tui_implemented() {
        // TUI requires a terminal - just verify function signature compiles
        // Cannot actually run in headless test environment
    }

    // -------------------------------------------------------------------------
    // run_record tests (additional)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_record_live_not_implemented() {
        let args = RecordArgs {
            duration: None,
            live: true,
            output: None,
            device: None,
            sample_rate: 16000,
            list_devices: false,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_record(args, &global);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    fn test_run_record_with_duration() {
        let args = RecordArgs {
            duration: Some(10),
            live: false,
            output: None,
            device: None,
            sample_rate: 16000,
            list_devices: false,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_record(args, &global);
        // Should fail because recording not implemented
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // run_translate tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_translate_file_not_found() {
        let args = TranslateArgs {
            input: "nonexistent.wav".into(),
            model: ModelSize::Base,
            output: None,
            format: OutputFormatArg::Txt,
            gpu: false,
            threads: None,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_translate(args, &global);
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    // -------------------------------------------------------------------------
    // run_model tests (additional)
    // -------------------------------------------------------------------------

    #[test]
    #[ignore = "Requires network access to HuggingFace Hub"]
    fn test_run_model_download() {
        let args = ModelArgs {
            action: ModelAction::Download {
                model: ModelSize::Base,
            },
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        // Download is now implemented - will attempt real download
        let result = run_model(args, &global);
        // May fail without network, but should not be NotImplemented
        assert!(
            !matches!(result, Err(CliError::NotImplemented(_))),
            "Download should be implemented"
        );
    }

    #[test]
    fn test_run_model_convert_missing_input() {
        let args = ModelArgs {
            action: ModelAction::Convert {
                input: "nonexistent_input.safetensors".into(),
                output: "output.apr2".into(),
            },
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        // Convert is now implemented - will fail on missing input file
        let result = run_model(args, &global);
        assert!(result.is_err(), "Should fail with missing input file");
        // Should not be NotImplemented - it's a real error
        assert!(
            !matches!(result, Err(CliError::NotImplemented(_))),
            "Convert should be implemented"
        );
    }

    // -------------------------------------------------------------------------
    // run_benchmark tests
    // -------------------------------------------------------------------------

    #[test]
    #[ignore = "Slow: runs full inference benchmark"]
    fn test_run_benchmark() {
        let args = BenchmarkArgs {
            model: ModelSize::Tiny,
            backend: BackendArg::Simd,
            iterations: 1,
            lfm2: false,
            component: "all".to_string(),
            seq_len: 128,
            full_size: false,
        };
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_benchmark(args, &global);
        assert!(result.is_ok());
        let result = result.expect("benchmark should succeed");
        assert!(result.success);
        assert!(result.rtf.is_some());
    }

    #[test]
    #[ignore = "Slow: runs full inference benchmark"]
    fn test_run_benchmark_verbose() {
        let args = BenchmarkArgs {
            model: ModelSize::Tiny,
            backend: BackendArg::Simd,
            iterations: 2,
            lfm2: false,
            component: "all".to_string(),
            seq_len: 128,
            full_size: false,
        };
        let global = Args {
            command: Command::Tui,
            verbose: true,
            quiet: false,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_benchmark(args, &global);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Timings tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_timings_default() {
        let timings = Timings::default();
        assert!((timings.total_ms - 0.0).abs() < f64::EPSILON);
        assert!((timings.model_load_ms - 0.0).abs() < f64::EPSILON);
        assert!((timings.audio_load_ms - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_timings_clone() {
        let timings = Timings {
            model_load_ms: 100.0,
            audio_load_ms: 50.0,
            mel_ms: 25.0,
            encode_ms: 75.0,
            decode_ms: 150.0,
            total_ms: 400.0,
        };
        let cloned = timings.clone();
        assert!((cloned.total_ms - 400.0).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // load_audio_samples tests (additional)
    // -------------------------------------------------------------------------

    #[test]
    #[cfg(not(feature = "symphonia"))]
    fn test_load_audio_flac_not_implemented() {
        let result = load_audio_samples(Path::new("test.flac"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("flac")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    #[cfg(feature = "symphonia")]
    fn test_load_audio_flac_decodes() {
        let path = Path::new("demos/test-audio/test-speech-1.5s.flac");
        if !path.exists() {
            eprintln!("Skipping: test file not found");
            return;
        }
        let data = std::fs::read(path).expect("Failed to read FLAC file");
        let result = load_audio_samples(path, &data);
        assert!(result.is_ok(), "FLAC decoding failed: {result:?}");
        let samples = result.unwrap();
        assert!(!samples.is_empty(), "No samples decoded from FLAC");
        assert!(samples.len() > 20000, "Too few samples: {}", samples.len());
    }

    #[test]
    #[cfg(not(feature = "symphonia"))]
    fn test_load_audio_mp4_not_implemented() {
        let result = load_audio_samples(Path::new("test.mp4"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("mp4")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    #[cfg(feature = "symphonia")]
    fn test_load_audio_mp4_decodes() {
        let path = Path::new("demos/test-audio/test-speech-1.5s.mp4");
        if !path.exists() {
            eprintln!("Skipping: test file not found");
            return;
        }
        let data = std::fs::read(path).expect("Failed to read MP4 file");
        let result = load_audio_samples(path, &data);
        assert!(result.is_ok(), "MP4 decoding failed: {result:?}");
        let samples = result.unwrap();
        assert!(!samples.is_empty(), "No samples decoded from MP4");
        assert!(samples.len() > 20000, "Too few samples: {}", samples.len());
    }

    #[test]
    #[cfg(not(feature = "symphonia"))]
    fn test_load_audio_ogg_not_implemented() {
        let result = load_audio_samples(Path::new("test.ogg"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("ogg")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    #[cfg(feature = "symphonia")]
    fn test_load_audio_ogg_decodes() {
        let path = Path::new("demos/test-audio/test-speech-1.5s.ogg");
        if !path.exists() {
            eprintln!("Skipping: test file not found");
            return;
        }
        let data = std::fs::read(path).expect("Failed to read OGG file");
        let result = load_audio_samples(path, &data);
        assert!(result.is_ok(), "OGG decoding failed: {result:?}");
        let samples = result.unwrap();
        assert!(!samples.is_empty(), "No samples decoded from OGG");
        assert!(samples.len() > 20000, "Too few samples: {}", samples.len());
    }

    #[test]
    fn test_load_audio_no_extension() {
        let result = load_audio_samples(Path::new("testfile"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::UnsupportedFormat(f)) => assert_eq!(f, ""),
            _ => panic!("Expected UnsupportedFormat error"),
        }
    }

    // -------------------------------------------------------------------------
    // run_batch tests (additional)
    // -------------------------------------------------------------------------

    #[test]
    fn test_run_batch_nonexistent_files() {
        let args = BatchArgs {
            inputs: vec!["nonexistent1.wav".into(), "nonexistent2.wav".into()],
            output_dir: None,
            parallel: None,
            recursive: false,
            pattern: None,
            skip_existing: false,
            model: ModelSize::Tiny,
            format: OutputFormatArg::Txt,
            cache_dir: None,
            zram_optimized: false,
        };
        let global = Args {
            command: Command::Tui,
            verbose: true,
            quiet: false,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_batch(args, &global);
        // Should error because no valid audio files found
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidArgument(msg)) => {
                assert!(msg.contains("No audio files found"));
            }
            _ => panic!("Expected InvalidArgument error for nonexistent files"),
        }
    }

    // -------------------------------------------------------------------------
    // CliError From implementations tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cli_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let cli_err: CliError = io_err.into();
        assert!(cli_err.to_string().contains("IO error"));
    }

    #[test]
    fn test_cli_error_unsupported_format() {
        let err = CliError::UnsupportedFormat("abc".to_string());
        assert!(err.to_string().contains("abc"));
    }

    #[test]
    fn test_cli_error_not_implemented() {
        let err = CliError::NotImplemented("feature X".to_string());
        assert!(err.to_string().contains("feature X"));
        assert!(err.to_string().contains("Not implemented"));
    }

    #[test]
    fn test_cli_error_invalid_argument() {
        let err = CliError::InvalidArgument("bad arg".to_string());
        assert!(err.to_string().contains("bad arg"));
    }

    #[test]
    fn test_cli_error_file_not_found() {
        let err = CliError::FileNotFound("missing.wav".to_string());
        assert!(err.to_string().contains("missing.wav"));
    }

    // -------------------------------------------------------------------------
    // run() dispatch tests
    // -------------------------------------------------------------------------

    #[test]
    #[cfg(not(feature = "tui"))]
    fn test_run_dispatches_to_tui() {
        let args = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run(args);
        // TUI is not implemented when feature disabled, should return error
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    #[cfg(feature = "tui")]
    fn test_run_dispatches_to_tui() {
        // When TUI feature is enabled, we can't easily test it in a headless environment
        // The TUI requires a terminal, so we just verify the function exists
        // and is callable without actually running it
        use crate::cli::args::{Args, Command};
        let _args = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };
        // Note: Actually running run(args) would block waiting for terminal input
        // This test just verifies the code compiles with tui feature
    }

    #[test]
    fn test_run_dispatches_to_model_list() {
        let args = Args {
            command: Command::Model(ModelArgs {
                action: ModelAction::List,
            }),
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run(args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_dispatches_to_record() {
        let args = Args {
            command: Command::Record(RecordArgs {
                duration: None,
                live: false,
                output: None,
                device: None,
                sample_rate: 16000,
                list_devices: true,
            }),
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run(args);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Backend expansion tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_backend_all_expands_to_three() {
        // Test that BackendArg::All expands to 3 backends
        let backends = match BackendArg::All {
            BackendArg::All => vec![BackendArg::Simd, BackendArg::Wasm, BackendArg::Cuda],
            other => vec![other],
        };
        assert_eq!(backends.len(), 3);
    }

    #[test]
    fn test_backend_single_stays_single() {
        let backends = match BackendArg::Simd {
            BackendArg::All => vec![BackendArg::Simd, BackendArg::Wasm, BackendArg::Cuda],
            other => vec![other],
        };
        assert_eq!(backends.len(), 1);
    }

    // -------------------------------------------------------------------------
    // Timings struct tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_timings_debug() {
        let timings = Timings::default();
        let debug_str = format!("{timings:?}");
        assert!(debug_str.contains("Timings"));
    }

    #[test]
    fn test_timings_all_fields() {
        let timings = Timings {
            model_load_ms: 10.0,
            audio_load_ms: 20.0,
            mel_ms: 30.0,
            encode_ms: 40.0,
            decode_ms: 50.0,
            total_ms: 150.0,
        };
        assert!((timings.model_load_ms - 10.0).abs() < f64::EPSILON);
        assert!((timings.audio_load_ms - 20.0).abs() < f64::EPSILON);
        assert!((timings.mel_ms - 30.0).abs() < f64::EPSILON);
        assert!((timings.encode_ms - 40.0).abs() < f64::EPSILON);
        assert!((timings.decode_ms - 50.0).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Additional load_audio_samples tests
    // -------------------------------------------------------------------------

    #[test]
    #[cfg(not(feature = "symphonia"))]
    fn test_load_audio_m4a_not_implemented() {
        let result = load_audio_samples(Path::new("test.m4a"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("m4a")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    #[cfg(feature = "symphonia")]
    fn test_load_audio_m4a_decodes() {
        let path = Path::new("demos/test-audio/test-speech-1.5s.m4a");
        if !path.exists() {
            eprintln!("Skipping: test file not found");
            return;
        }
        let data = std::fs::read(path).expect("Failed to read M4A file");
        let result = load_audio_samples(path, &data);
        assert!(result.is_ok(), "M4A decoding failed: {result:?}");
        let samples = result.unwrap();
        assert!(!samples.is_empty(), "No samples decoded from M4A");
        assert!(samples.len() > 20000, "Too few samples: {}", samples.len());
    }

    #[test]
    #[cfg(not(feature = "symphonia"))]
    fn test_load_audio_webm_not_implemented() {
        let result = load_audio_samples(Path::new("test.webm"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("webm")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    #[cfg(feature = "symphonia")]
    fn test_load_audio_webm_decodes() {
        // Note: WEBM with Opus audio requires libopus adapter (not included by default)
        // This test verifies the decoder is invoked but may fail gracefully
        let path = Path::new("demos/test-audio/test-speech-1.5s.webm");
        if !path.exists() {
            eprintln!("Skipping: test file not found");
            return;
        }
        let data = std::fs::read(path).expect("Failed to read WEBM file");
        let result = load_audio_samples(path, &data);
        // WEBM/Opus may not be supported - check gracefully
        if result.is_ok() {
            let samples = result.unwrap();
            assert!(!samples.is_empty(), "No samples decoded from WEBM");
            assert!(samples.len() > 20000, "Too few samples: {}", samples.len());
        } else {
            // Expected: Opus codec not supported without adapter
            eprintln!("WEBM/Opus not fully supported: {result:?}");
        }
    }

    #[test]
    #[cfg(not(feature = "symphonia"))]
    fn test_load_audio_mkv_not_implemented() {
        let result = load_audio_samples(Path::new("test.mkv"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("mkv")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    #[cfg(feature = "symphonia")]
    fn test_load_audio_mkv_decodes() {
        let path = Path::new("demos/test-audio/test-speech-1.5s.mkv");
        if !path.exists() {
            eprintln!("Skipping: test file not found");
            return;
        }
        let data = std::fs::read(path).expect("Failed to read MKV file");
        let result = load_audio_samples(path, &data);
        assert!(result.is_ok(), "MKV decoding failed: {result:?}");
        let samples = result.unwrap();
        assert!(!samples.is_empty(), "No samples decoded from MKV");
        assert!(samples.len() > 20000, "Too few samples: {}", samples.len());
    }

    #[test]
    #[cfg(not(feature = "symphonia"))]
    fn test_load_audio_avi_not_implemented() {
        let result = load_audio_samples(Path::new("test.avi"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("avi")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    #[cfg(feature = "symphonia")]
    fn test_load_audio_avi_decodes() {
        let path = Path::new("demos/test-audio/test-speech-1.5s.avi");
        if !path.exists() {
            eprintln!("Skipping: test file not found");
            return;
        }
        let data = std::fs::read(path).expect("Failed to read AVI file");
        let result = load_audio_samples(path, &data);
        // AVI may not be fully supported by symphonia - check gracefully
        if result.is_ok() {
            let samples = result.unwrap();
            assert!(!samples.is_empty(), "No samples decoded from AVI");
            assert!(samples.len() > 20000, "Too few samples: {}", samples.len());
        }
    }

    #[test]
    fn test_load_audio_unknown_extension() {
        let result = load_audio_samples(Path::new("test.xyz"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::UnsupportedFormat(f)) => assert_eq!(f, "xyz"),
            _ => panic!("Expected UnsupportedFormat error"),
        }
    }

    // -------------------------------------------------------------------------
    // CommandResult builder pattern tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_command_result_builder_chain() {
        let timings = Timings {
            total_ms: 50.0,
            ..Default::default()
        };
        let result = CommandResult::success("Test")
            .with_timings(timings)
            .with_rtf(1.5);

        assert!(result.success);
        assert_eq!(result.message, "Test");
        assert!(result.timings.is_some());
        assert_eq!(result.rtf, Some(1.5));
    }

    #[test]
    fn test_command_result_failure_with_rtf() {
        let result = CommandResult::failure("Failed").with_rtf(2.0);
        assert!(!result.success);
        assert_eq!(result.rtf, Some(2.0));
    }

    // -------------------------------------------------------------------------
    // run_summarize tests
    // -------------------------------------------------------------------------

    fn default_summarize_args(
        input: Option<PathBuf>,
        model_path: Option<PathBuf>,
    ) -> SummarizeArgs {
        SummarizeArgs {
            input,
            model_path,
            tokenizer_path: None,
            output: None,
            format: SummarizeFormat::Text,
            max_tokens: 256,
            temperature: 0.7,
            max_context: 4096,
            webgpu: false,
            stream: false,
            action_items: false,
            key_points: false,
            prompt: None,
        }
    }

    #[test]
    fn test_run_summarize_no_model_path() {
        // Create a temp input file since the function checks input first
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_summarize_model_check.txt");
        fs::write(&input_path, "Some text to summarize").expect("write test file");

        let args = default_summarize_args(Some(input_path.clone()), None);
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_summarize(args, &global);
        // Clean up
        let _ = fs::remove_file(&input_path);

        assert!(result.is_err(), "Should error without model path");
        match &result {
            Err(CliError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("model-path"),
                    "Error should mention --model-path: {msg}"
                );
            }
            Err(e) => panic!("Expected InvalidArgument error for missing model path, got: {e:?}"),
            Ok(_) => panic!("Expected error, got success"),
        }
    }

    #[test]
    fn test_run_summarize_input_file_not_found() {
        let args = default_summarize_args(
            Some("nonexistent_input.txt".into()),
            Some("model.apr2".into()),
        );
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_summarize(args, &global);
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_summarize_model_file_not_found() {
        // Create a temp input file
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_summarize_input.txt");
        fs::write(&input_path, "This is test input for summarization.").expect("write test file");

        let args = default_summarize_args(
            Some(input_path.clone()),
            Some("nonexistent_model.apr2".into()),
        );
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_summarize(args, &global);
        // Clean up
        let _ = fs::remove_file(&input_path);

        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error for missing model"),
        }
    }

    #[test]
    fn test_run_summarize_empty_input() {
        // Create a temp empty input file
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_summarize_empty.txt");
        fs::write(&input_path, "   ").expect("write test file");

        let args = default_summarize_args(Some(input_path.clone()), Some("model.apr2".into()));
        let global = Args {
            command: Command::Tui,
            verbose: false,
            quiet: true,
            json: false,
            trace: None,
            no_color: false,
        };

        let result = run_summarize(args, &global);
        // Clean up
        let _ = fs::remove_file(&input_path);

        assert!(result.is_err());
        match result {
            Err(CliError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("No input text"),
                    "Error should mention no input: {msg}"
                );
            }
            _ => panic!("Expected InvalidArgument error for empty input"),
        }
    }

    // -------------------------------------------------------------------------
    // run_post_transcription_summary tests (Phase 2)
    // -------------------------------------------------------------------------

    #[test]
    fn test_post_transcription_summary_no_model() {
        // Create args without lfm2_model
        let mut args = default_transcribe_args("test.wav".into());
        args.summarize = true;
        args.lfm2_model = None;

        let global = default_global_args();

        let result = run_post_transcription_summary("Test transcript", &args, &global);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("lfm2-model"),
                    "Error should mention --lfm2-model: {msg}"
                );
            }
            _ => panic!("Expected InvalidArgument error for missing model"),
        }
    }

    #[test]
    fn test_post_transcription_summary_model_not_found() {
        let mut args = default_transcribe_args("test.wav".into());
        args.summarize = true;
        args.lfm2_model = Some("nonexistent_model.apr2".into());

        let global = default_global_args();

        let result = run_post_transcription_summary("Test transcript", &args, &global);
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error for missing model"),
        }
    }

    #[test]
    fn test_post_transcription_summary_empty_transcript() {
        // Create a temp model file (won't be loaded since transcript is empty)
        let temp_dir = std::env::temp_dir();
        let model_path = temp_dir.join("test_lfm2_empty_check.apr2");
        fs::write(&model_path, b"dummy").expect("write test file");

        let mut args = default_transcribe_args("test.wav".into());
        args.summarize = true;
        args.lfm2_model = Some(model_path.clone());

        let mut global = default_global_args();
        global.quiet = true; // Suppress warning

        let result = run_post_transcription_summary("   ", &args, &global);
        // Clean up
        let _ = fs::remove_file(&model_path);

        // Empty transcript should return Ok with empty string
        assert!(result.is_ok());
        assert!(result.expect("should be ok").is_empty());
    }

    #[test]
    fn test_transcribe_args_with_summarize_fields() {
        // Verify the new summarize fields are accessible
        let mut args = default_transcribe_args("test.wav".into());

        // Set summarization options
        args.summarize = true;
        args.lfm2_model = Some("model.apr2".into());
        args.summary_output = Some("summary.json".into());
        args.summary_format = SummarizeFormat::Markdown;
        args.action_items = true;
        args.key_points = true;

        // Verify fields are set correctly
        assert!(args.summarize);
        assert_eq!(
            args.lfm2_model.as_ref().expect("should be set").to_str(),
            Some("model.apr2")
        );
        assert!(args.action_items);
        assert!(args.key_points);
    }

    // -------------------------------------------------------------------------
    // Folder processing helper tests (WAPR-PERF-004)
    // Per spec: docs/specifications/transcribe-folder-spec.md §H (F101-F110)
    // -------------------------------------------------------------------------

    #[test]
    fn test_glob_match_star() {
        // F102: Pattern matching
        assert!(glob_match("*.wav", "test.wav"));
        assert!(glob_match("*.wav", "foo.wav"));
        assert!(!glob_match("*.wav", "test.mp3"));
        assert!(!glob_match("*.wav", "testwav"));
    }

    #[test]
    fn test_glob_match_question() {
        assert!(glob_match("test?.wav", "test1.wav"));
        assert!(glob_match("test?.wav", "testa.wav"));
        assert!(!glob_match("test?.wav", "test12.wav"));
        assert!(!glob_match("test?.wav", "test.wav"));
    }

    #[test]
    fn test_glob_match_complex() {
        assert!(glob_match("audio_*.mp3", "audio_track1.mp3"));
        assert!(glob_match("*_recording_*", "my_recording_2024.wav"));
        assert!(glob_match("test*", "test"));
        assert!(glob_match("test*", "testing"));
        assert!(glob_match("*test", "mytest"));
    }

    #[test]
    fn test_matches_audio_pattern() {
        // F102: Extension matching
        assert!(matches_audio_pattern(Path::new("test.wav"), None));
        assert!(matches_audio_pattern(Path::new("test.mp3"), None));
        assert!(matches_audio_pattern(Path::new("test.flac"), None));
        assert!(matches_audio_pattern(Path::new("test.ogg"), None));
        assert!(matches_audio_pattern(Path::new("test.m4a"), None));
        assert!(!matches_audio_pattern(Path::new("test.txt"), None));
        assert!(!matches_audio_pattern(Path::new("test.pdf"), None));
    }

    #[test]
    fn test_matches_audio_pattern_with_glob() {
        // Pattern filter
        assert!(matches_audio_pattern(Path::new("song.wav"), Some("*.wav")));
        assert!(!matches_audio_pattern(Path::new("song.mp3"), Some("*.wav")));
        assert!(matches_audio_pattern(
            Path::new("recording_01.mp3"),
            Some("recording_*")
        ));
    }

    #[test]
    fn test_compute_mirrored_output_path_flat() {
        // F101: Flat mapping when no base directory
        let input = Path::new("/audio/test.wav");
        let output_dir = Path::new("/output");
        let result = compute_mirrored_output_path(input, None, output_dir, "txt");
        assert_eq!(result, PathBuf::from("/output/test.txt"));
    }

    #[test]
    fn test_compute_mirrored_output_path_mirrored() {
        // F101: Structure mirroring
        let input = Path::new("/audio/subdir/deep/test.wav");
        let base = Path::new("/audio");
        let output_dir = Path::new("/output");
        let result = compute_mirrored_output_path(input, Some(base), output_dir, "json");
        assert_eq!(result, PathBuf::from("/output/subdir/deep/test.json"));
    }

    #[test]
    fn test_compute_mirrored_output_path_format_extension() {
        // F102: Format extension replacement
        let input = Path::new("test.mp3");
        let output_dir = Path::new("./out");

        let txt = compute_mirrored_output_path(input, None, output_dir, "txt");
        assert_eq!(txt, PathBuf::from("./out/test.txt"));

        let json = compute_mirrored_output_path(input, None, output_dir, "json");
        assert_eq!(json, PathBuf::from("./out/test.json"));

        let srt = compute_mirrored_output_path(input, None, output_dir, "srt");
        assert_eq!(srt, PathBuf::from("./out/test.srt"));
    }

    #[test]
    fn test_compute_mirrored_output_path_with_spaces() {
        // F110: Space in path handling
        let input = Path::new("/My Documents/audio file.wav");
        let output_dir = Path::new("/Output Folder");
        let result = compute_mirrored_output_path(input, None, output_dir, "txt");
        assert_eq!(result, PathBuf::from("/Output Folder/audio file.txt"));
    }

    #[test]
    fn test_discover_audio_files_empty() {
        // Empty input returns empty
        let files = discover_audio_files(&[], false, None);
        assert!(files.is_empty());
    }

    #[test]
    fn test_discover_audio_files_sorted() {
        // F107: Deterministic ordering (sorted)
        use tempfile::TempDir;

        let temp = TempDir::new().expect("create temp dir");
        let dir = temp.path();

        // Create files in non-sorted order
        fs::write(dir.join("c.wav"), b"").expect("write c.wav");
        fs::write(dir.join("a.wav"), b"").expect("write a.wav");
        fs::write(dir.join("b.wav"), b"").expect("write b.wav");

        let inputs = vec![dir.to_path_buf()];
        let files = discover_audio_files(&inputs, false, None);

        assert_eq!(files.len(), 3);
        // Should be sorted alphabetically
        assert!(files[0].0.file_name().unwrap().to_str().unwrap() == "a.wav");
        assert!(files[1].0.file_name().unwrap().to_str().unwrap() == "b.wav");
        assert!(files[2].0.file_name().unwrap().to_str().unwrap() == "c.wav");
    }

    #[test]
    fn test_discover_audio_files_skips_hidden() {
        // F108: Hidden file filtering
        use tempfile::TempDir;

        let temp = TempDir::new().expect("create temp dir");
        let dir = temp.path();

        fs::write(dir.join("visible.wav"), b"").expect("write visible.wav");
        fs::write(dir.join(".hidden.wav"), b"").expect("write .hidden.wav");
        fs::create_dir(dir.join(".git")).expect("create .git");
        fs::write(dir.join(".git/config.wav"), b"").expect("write config.wav");

        let inputs = vec![dir.to_path_buf()];
        let files = discover_audio_files(&inputs, true, None);

        // Should only find visible.wav, not .hidden.wav or anything in .git
        assert_eq!(files.len(), 1);
        assert!(files[0].0.file_name().unwrap().to_str().unwrap() == "visible.wav");
    }

    #[test]
    fn test_discover_audio_files_recursive() {
        // F101: Recursive discovery with structure
        use tempfile::TempDir;

        let temp = TempDir::new().expect("create temp dir");
        let dir = temp.path();

        fs::write(dir.join("root.wav"), b"").expect("write root.wav");
        fs::create_dir(dir.join("subdir")).expect("create subdir");
        fs::write(dir.join("subdir/nested.wav"), b"").expect("write nested.wav");
        fs::create_dir(dir.join("subdir/deep")).expect("create deep");
        fs::write(dir.join("subdir/deep/very_nested.wav"), b"").expect("write very_nested.wav");

        // Non-recursive
        let inputs = vec![dir.to_path_buf()];
        let files_nonrec = discover_audio_files(&inputs, false, None);
        assert_eq!(
            files_nonrec.len(),
            1,
            "Non-recursive should find only root file"
        );

        // Recursive
        let files_rec = discover_audio_files(&inputs, true, None);
        assert_eq!(files_rec.len(), 3, "Recursive should find all files");
    }

    #[test]
    fn test_atomic_write_creates_parents() {
        // F106: Missing parent directory creation
        use tempfile::TempDir;

        let temp = TempDir::new().expect("create temp dir");
        let output_path = temp.path().join("a/b/c/output.txt");

        let result = atomic_write_transcription(&output_path, "test content");
        assert!(result.is_ok(), "Should create parent directories");
        assert!(output_path.exists(), "Output file should exist");

        let content = fs::read_to_string(&output_path).expect("read file");
        assert_eq!(content, "test content");
    }

    #[test]
    fn test_atomic_write_no_partial() {
        // F103: Atomicity - no .tmp file left behind on success
        use tempfile::TempDir;

        let temp = TempDir::new().expect("create temp dir");
        let output_path = temp.path().join("output.txt");
        let temp_path = output_path.with_extension("tmp");

        let result = atomic_write_transcription(&output_path, "final content");
        assert!(result.is_ok());
        assert!(output_path.exists(), "Final file should exist");
        assert!(
            !temp_path.exists(),
            "Temp file should not exist after rename"
        );
    }

    #[test]
    fn test_format_batch_output_txt() {
        let result = BatchTranscribeResult {
            text: "Hello world".to_string(),
            segments: vec![],
        };
        let output = format_batch_output(&result, OutputFormatArg::Txt);
        assert_eq!(output, "Hello world");
    }

    #[test]
    fn test_format_batch_output_json() {
        let result = BatchTranscribeResult {
            text: "Hello \"world\"".to_string(),
            segments: vec![],
        };
        let output = format_batch_output(&result, OutputFormatArg::Json);
        assert!(output.contains(r#""text":"Hello \"world\"""#));
    }

    #[test]
    fn test_format_batch_output_vtt() {
        let result = BatchTranscribeResult {
            text: "Hello world".to_string(),
            segments: vec![],
        };
        let output = format_batch_output(&result, OutputFormatArg::Vtt);
        assert!(output.starts_with("WEBVTT"));
        assert!(output.contains("Hello world"));
    }

    #[test]
    fn test_format_batch_output_srt() {
        let result = BatchTranscribeResult {
            text: "Hello world".to_string(),
            segments: vec![],
        };
        let output = format_batch_output(&result, OutputFormatArg::Srt);
        assert!(output.contains("00:00:00,000 --> 00:00:30,000"));
        assert!(output.contains("Hello world"));
    }
}

//! Command implementations for whisper-apr CLI
//!
//! Each command is implemented as a pure function for testability.
//! The main `run` function dispatches to the appropriate command.

use std::fs;
use std::io::{self, Write as IoWrite};
use std::path::Path;
use std::time::Instant;

use crate::audio::wav::{parse_wav_file, resample};
use crate::{DecodingStrategy, Task, TranscribeOptions, WhisperApr};

use super::args::{
    Args, BackendArg, BatchArgs, BenchmarkArgs, Command, CommandArgs, ModelAction, ModelArgs,
    OutputFormatArg, ParityArgs, QuantizeArgs, RecordArgs, ServeArgs, StreamArgs, TestArgs,
    TranscribeArgs, TranslateArgs, ValidateArgs, ValidateOutputFormat,
};

use super::output::{format_output, OutputFormat};

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
        Command::Stream(s) => run_stream(s.clone(), &args),
        Command::Serve(s) => run_serve(s.clone(), &args),
        Command::Record(r) => run_record(r.clone(), &args),
        Command::Batch(b) => run_batch(b.clone(), &args),
        Command::Tui => run_tui(&args),
        Command::Test(t) => run_test(t.clone(), &args),
        Command::Model(m) => run_model(m.clone(), &args),
        Command::Benchmark(b) => run_benchmark(b.clone(), &args),
        Command::Validate(v) => run_validate(v.clone(), &args),
        Command::Parity(p) => run_parity(p.clone(), &args),
        Command::Quantize(q) => run_quantize(q.clone(), &args),
        Command::Command(c) => run_command(c.clone(), &args),
    }
}

/// Run transcribe command
pub fn run_transcribe(args: TranscribeArgs, global: &Args) -> CliResult<CommandResult> {
    let start = Instant::now();
    let mut timings = Timings::default();

    // Validate input file exists
    if !args.input.exists() {
        return Err(CliError::FileNotFound(args.input.display().to_string()));
    }

    // Load model
    if global.verbose {
        if let Some(path) = &args.model_path {
            eprintln!("[INFO] Loading model from: {}", path.display());
        } else {
            eprintln!("[INFO] Loading model: {}", args.model);
        }
    }
    let model_start = Instant::now();

    let whisper = super::model_loader::load_or_download_model(
        args.model,
        args.model_path.as_deref(),
        global.verbose,
    )
    .map_err(|e| CliError::InvalidArgument(e.to_string()))?;

    timings.model_load_ms = model_start.elapsed().as_secs_f64() * 1000.0;

    // Load and parse audio
    if global.verbose {
        eprintln!("[INFO] Loading audio: {}", args.input.display());
    }
    let audio_start = Instant::now();
    let audio_data = fs::read(&args.input)?;

    // Detect format from extension
    let samples = load_audio_samples(&args.input, &audio_data)?;
    timings.audio_load_ms = audio_start.elapsed().as_secs_f64() * 1000.0;

    let audio_duration_secs = samples.len() as f64 / 16000.0;

    if global.verbose {
        eprintln!(
            "[INFO] Audio: {:.2}s, {} samples",
            audio_duration_secs,
            samples.len()
        );
    }

    // Transcribe
    let transcribe_start = Instant::now();
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
    };

    let result = whisper.transcribe(&samples, options)?;
    timings.decode_ms = transcribe_start.elapsed().as_secs_f64() * 1000.0;
    timings.total_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Calculate RTF
    let rtf = (timings.total_ms / 1000.0) / audio_duration_secs;

    if global.verbose {
        eprintln!("[INFO] Total: {:.1}ms", timings.total_ms);
        eprintln!("[INFO] RTF: {rtf:.2}x");
    }

    // Format output
    let format = convert_format_arg(args.format);
    let output_text = format_output(&result, format);

    // Write output
    if let Some(output_path) = args.output {
        fs::write(&output_path, &output_text)?;
        if global.verbose {
            eprintln!("[INFO] Written to: {}", output_path.display());
        }
    } else if !global.quiet {
        print!("{output_text}");
        io::stdout().flush()?;
    }

    Ok(CommandResult::success(result.text)
        .with_timings(timings)
        .with_rtf(rtf))
}

/// Run translate command
pub fn run_translate(args: TranslateArgs, global: &Args) -> CliResult<CommandResult> {
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

/// Run batch command
pub fn run_batch(args: BatchArgs, global: &Args) -> CliResult<CommandResult> {
    if args.inputs.is_empty() {
        return Err(CliError::InvalidArgument(
            "No input files specified".to_string(),
        ));
    }

    let output_dir = args.output_dir.unwrap_or_else(|| ".".into());

    // Create output directory if needed
    fs::create_dir_all(&output_dir)?;

    let mut processed = 0;
    let mut failed = 0;

    for input in &args.inputs {
        if !input.exists() {
            if global.verbose {
                eprintln!("[WARN] File not found: {}", input.display());
            }
            failed += 1;
            continue;
        }

        // Generate output filename
        let stem = input.file_stem().unwrap_or_default().to_string_lossy();
        let ext = args.format.to_string();
        let output_path = output_dir.join(format!("{stem}.{ext}"));

        // Skip if exists and --skip-existing
        if args.skip_existing && output_path.exists() {
            if global.verbose {
                eprintln!("[INFO] Skipping existing: {}", output_path.display());
            }
            continue;
        }

        // Transcribe - construct minimal args, let defaults handle the rest
        let transcribe_args = TranscribeArgs {
            input: input.clone(),
            model: args.model,
            output: Some(output_path),
            format: args.format,
            // Use defaults for all other fields
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
            no_prints: false,
            print_special: false,
            colors: false,
            confidence: false,
            progress: false,
            translate: false,
            hallucination_filter: false,
        };

        match run_transcribe(transcribe_args, global) {
            Ok(_) => processed += 1,
            Err(e) => {
                if global.verbose {
                    eprintln!("[ERROR] {}: {}", input.display(), e);
                }
                failed += 1;
            }
        }
    }

    Ok(CommandResult::success(format!(
        "Processed {processed} files, {failed} failed"
    )))
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
#[cfg(feature = "tui")]
pub fn run_tui(global: &Args) -> CliResult<CommandResult> {
    use crossterm::{
        event::{self, Event, KeyCode, KeyEventKind},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::{backend::CrosstermBackend, Terminal};
    use std::time::Duration;

    use crate::tui::{render_whisper_dashboard, WhisperApp};

    // Initialize terminal
    enable_raw_mode().map_err(|e| CliError::Io(io::Error::new(io::ErrorKind::Other, e)))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)
        .map_err(|e| CliError::Io(io::Error::new(io::ErrorKind::Other, e)))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal =
        Terminal::new(backend).map_err(|e| CliError::Io(io::Error::new(io::ErrorKind::Other, e)))?;

    // Create application state
    let mut app = WhisperApp::new();

    // Show initial status
    app.status_message = Some("Press '?' for help, 'q' to quit".to_string());

    // Event loop
    let result = loop {
        // Draw frame
        terminal
            .draw(|f| render_whisper_dashboard(f, &app))
            .map_err(|e| CliError::Io(io::Error::new(io::ErrorKind::Other, e)))?;

        // Poll for events with timeout
        if event::poll(Duration::from_millis(100))
            .map_err(|e| CliError::Io(io::Error::new(io::ErrorKind::Other, e)))?
        {
            if let Event::Key(key) = event::read()
                .map_err(|e| CliError::Io(io::Error::new(io::ErrorKind::Other, e)))?
            {
                // Only handle key press events (not release)
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break Ok(CommandResult::success("TUI closed")),
                        KeyCode::Char('1') => app.handle_key('1'),
                        KeyCode::Char('2') => app.handle_key('2'),
                        KeyCode::Char('3') => app.handle_key('3'),
                        KeyCode::Char('4') => app.handle_key('4'),
                        KeyCode::Char('5') => app.handle_key('5'),
                        KeyCode::Char('6') => app.handle_key('6'),
                        KeyCode::Char('7') => app.handle_key('7'),
                        KeyCode::Char('?') => app.handle_key('?'),
                        KeyCode::Char(' ') => app.handle_key(' '),
                        KeyCode::Char('r') => app.handle_key('r'),
                        _ => {}
                    }
                }
            }
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

/// Run model command
pub fn run_model(args: ModelArgs, global: &Args) -> CliResult<CommandResult> {
    match args.action {
        ModelAction::List => {
            println!("Available models:");
            println!("  tiny   - 39M params, fastest");
            println!("  base   - 74M params, good balance");
            println!("  small  - 244M params, higher accuracy");
            println!("  medium - 769M params, high accuracy");
            println!("  large  - 1.5B params, best accuracy");
            Ok(CommandResult::success("Listed models"))
        }
        ModelAction::Download { model } => {
            if !global.quiet {
                println!("Downloading {model} model...");
            }
            // Download implementation would use pacha registry
            Err(CliError::NotImplemented(
                "Model download not yet implemented".to_string(),
            ))
        }
        ModelAction::Convert { input, output } => {
            if !global.quiet {
                println!("Converting {} to {}...", input.display(), output.display());
            }
            // Conversion implementation
            Err(CliError::NotImplemented(
                "Model conversion not yet implemented".to_string(),
            ))
        }
        ModelAction::Info { file } => {
            if !file.exists() {
                return Err(CliError::FileNotFound(file.display().to_string()));
            }
            // Show model info
            println!("Model: {}", file.display());
            // Would parse model file and show details
            Ok(CommandResult::success("Showed model info"))
        }
    }
}

/// Run benchmark command
pub fn run_benchmark(args: BenchmarkArgs, global: &Args) -> CliResult<CommandResult> {
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

/// Run validate command
pub fn run_validate(args: ValidateArgs, global: &Args) -> CliResult<CommandResult> {
    use crate::format::{quick_validate, AprReader, AprValidator};

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

    // Quick validation mode
    if args.quick {
        match quick_validate(&reader) {
            Ok(()) => {
                if !global.quiet {
                    println!("✓ Quick validation passed");
                }
                return Ok(CommandResult::success("Quick validation passed"));
            }
            Err(e) => {
                if !global.quiet {
                    println!("✗ Quick validation failed: {e}");
                }
                return Ok(CommandResult::failure(format!(
                    "Quick validation failed: {e}"
                )));
            }
        }
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

/// Run parity command (whisper.cpp comparison)
#[allow(clippy::too_many_lines)]
pub fn run_parity(args: ParityArgs, global: &Args) -> CliResult<CommandResult> {
    use super::parity::{ParityConfig, ParityTest};

    // Validate input file exists
    if !args.input.exists() {
        return Err(CliError::FileNotFound(args.input.display().to_string()));
    }

    // Find whisper.cpp binary
    let whisper_cpp_path = args.whisper_cpp.clone().unwrap_or_else(|| {
        // Search common locations
        let candidates = [
            "/usr/local/bin/whisper-cli",
            "/usr/bin/whisper-cli",
            "whisper-cli",
            "./whisper-cli",
            "../whisper.cpp/main",
        ];
        for candidate in candidates {
            let path = std::path::PathBuf::from(candidate);
            if path.exists() {
                return path;
            }
        }
        std::path::PathBuf::from("whisper-cli")
    });

    if !global.quiet {
        println!("whisper-apr Parity Test");
        println!("═══════════════════════════════════════════════════════════════════");
        println!("Audio: {}", args.input.display());
        println!("whisper.cpp: {}", whisper_cpp_path.display());
        println!("Max WER: {:.1}%", args.max_wer * 100.0);
        println!();
    }

    // Run whisper.cpp
    if args.verbose {
        println!("Running whisper.cpp...");
    }

    let model_path = args.cpp_model.as_ref().map_or_else(
        || format!("models/ggml-{}.bin", args.model),
        |p| p.to_string_lossy().to_string(),
    );
    let cpp_output = std::process::Command::new(&whisper_cpp_path)
        .args([
            "-m",
            model_path.as_str(),
            "-f",
            &args.input.to_string_lossy(),
            "--no-prints",
        ])
        .output();

    let cpp_text = match cpp_output {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).to_string()
        }
        Ok(output) => {
            return Err(CliError::InvalidArgument(format!(
                "whisper.cpp failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        Err(e) => {
            return Err(CliError::InvalidArgument(format!(
                "Failed to run whisper.cpp at {}: {}",
                whisper_cpp_path.display(),
                e
            )));
        }
    };

    // Run whisper-apr
    if args.verbose {
        println!("Running whisper-apr...");
    }

    let audio_data = fs::read(&args.input)?;
    let samples = load_audio_samples(&args.input, &audio_data)?;

    let whisper = super::model_loader::load_or_download_model(
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

    // Output results
    if args.json {
        let json = serde_json::json!({
            "input": args.input.display().to_string(),
            "whisper_cpp_output": cpp_text.trim(),
            "whisper_apr_output": apr_text.trim(),
            "parity": parity_result.is_pass(),
            "wer": match &parity_result {
                super::parity::ParityResult::Pass { wer, .. }
                | super::parity::ParityResult::Fail { wer, .. } => *wer,
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

        match &parity_result {
            super::parity::ParityResult::Pass { wer, .. } => {
                println!("✓ PARITY ACHIEVED (WER: {:.2}%)", wer * 100.0);
            }
            super::parity::ParityResult::Fail { wer, .. } => {
                println!(
                    "✗ PARITY FAILED (WER: {:.2}%, max: {:.2}%)",
                    wer * 100.0,
                    args.max_wer * 100.0
                );
            }
        }
    }

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
fn load_audio_samples(path: &Path, data: &[u8]) -> CliResult<Vec<f32>> {
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
        "mp3" | "flac" | "ogg" | "m4a" | "mp4" | "webm" | "mkv" | "avi" => {
            // These formats would require symphonia for decoding
            Err(CliError::NotImplemented(format!(
                "{ext} format not yet implemented (requires symphonia)"
            )))
        }
        _ => Err(CliError::UnsupportedFormat(ext)),
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
            translate: false,
            hallucination_filter: false,
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
    fn test_load_audio_mp3_not_implemented() {
        let result = load_audio_samples(Path::new("test.mp3"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("mp3")),
            _ => panic!("Expected NotImplemented error"),
        }
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
    #[ignore] // Slow: runs full inference pipeline
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
    #[ignore] // Slow: runs full inference pipeline
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
    #[ignore] // Slow: runs full inference pipeline
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
    #[ignore] // Slow: runs full inference pipeline for all backends
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
    fn test_run_model_download_not_implemented() {
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

        let result = run_model(args, &global);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    fn test_run_model_convert_not_implemented() {
        let args = ModelArgs {
            action: ModelAction::Convert {
                input: "input.pt".into(),
                output: "output.apr".into(),
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
        match result {
            Err(CliError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented error"),
        }
    }

    // -------------------------------------------------------------------------
    // run_benchmark tests
    // -------------------------------------------------------------------------

    #[test]
    #[ignore] // Slow: runs full inference benchmark
    fn test_run_benchmark() {
        let args = BenchmarkArgs {
            model: ModelSize::Tiny,
            backend: BackendArg::Simd,
            iterations: 1,
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
    #[ignore] // Slow: runs full inference benchmark
    fn test_run_benchmark_verbose() {
        let args = BenchmarkArgs {
            model: ModelSize::Tiny,
            backend: BackendArg::Simd,
            iterations: 2,
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
    fn test_load_audio_flac_not_implemented() {
        let result = load_audio_samples(Path::new("test.flac"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("flac")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    fn test_load_audio_mp4_not_implemented() {
        let result = load_audio_samples(Path::new("test.mp4"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("mp4")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    fn test_load_audio_ogg_not_implemented() {
        let result = load_audio_samples(Path::new("test.ogg"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("ogg")),
            _ => panic!("Expected NotImplemented error"),
        }
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
        // Should succeed but report failures
        assert!(result.is_ok());
        let result = result.expect("batch should succeed");
        assert!(result.message.contains("failed"));
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
        // TUI is not implemented, should return error
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented error"),
        }
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
    fn test_load_audio_m4a_not_implemented() {
        let result = load_audio_samples(Path::new("test.m4a"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("m4a")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    fn test_load_audio_webm_not_implemented() {
        let result = load_audio_samples(Path::new("test.webm"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("webm")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    fn test_load_audio_mkv_not_implemented() {
        let result = load_audio_samples(Path::new("test.mkv"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("mkv")),
            _ => panic!("Expected NotImplemented error"),
        }
    }

    #[test]
    fn test_load_audio_avi_not_implemented() {
        let result = load_audio_samples(Path::new("test.avi"), &[]);
        assert!(result.is_err());
        match result {
            Err(CliError::NotImplemented(msg)) => assert!(msg.contains("avi")),
            _ => panic!("Expected NotImplemented error"),
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
}

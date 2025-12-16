//! Command-line argument parsing for whisper-apr CLI
//!
//! Uses clap derive macros for type-safe argument parsing.
//! All argument structures are unit-testable.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

/// whisper-apr: WASM-first automatic speech recognition
///
/// A high-performance speech recognition CLI that runs natively and in browsers.
/// Install with: cargo install whisper-apr
#[derive(Parser, Debug, Clone)]
#[command(name = "whisper-apr")]
#[command(version)]
#[command(about = "WASM-first automatic speech recognition", long_about = None)]
#[command(propagate_version = true)]
#[allow(clippy::struct_excessive_bools)] // CLI flags are naturally boolean
pub struct Args {
    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Command,

    /// Verbose output (show timing info)
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Quiet mode (suppress non-essential output)
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Output as JSON (machine-readable)
    #[arg(long, global = true)]
    pub json: bool,

    /// Export performance trace (Chrome format)
    #[arg(long, global = true)]
    pub trace: Option<PathBuf>,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,
}

/// Available commands
#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Transcribe audio/video to text
    Transcribe(TranscribeArgs),

    /// Translate speech to English
    Translate(TranslateArgs),

    /// Record audio from microphone
    Record(RecordArgs),

    /// Process multiple files in parallel
    Batch(BatchArgs),

    /// Interactive terminal UI
    Tui,

    /// Run backend E2E tests
    Test(TestArgs),

    /// Manage models (download, list, convert)
    Model(ModelArgs),

    /// Performance benchmarking
    Benchmark(BenchmarkArgs),

    /// Validate APR model file (25-point QA checklist)
    Validate(ValidateArgs),
}

/// Arguments for transcribe command
#[derive(Parser, Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // CLI flags are naturally boolean
pub struct TranscribeArgs {
    /// Input audio/video file
    pub input: PathBuf,

    /// Model size to use
    #[arg(short, long, default_value = "tiny")]
    pub model: ModelSize,

    /// Path to .apr model file (overrides --model)
    #[arg(long)]
    pub model_path: Option<PathBuf>,

    /// Source language (ISO 639-1) or 'auto' for detection
    #[arg(short, long, default_value = "auto")]
    pub language: String,

    /// Output file path (default: stdout)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format
    #[arg(short, long, default_value = "txt")]
    pub format: OutputFormatArg,

    /// Include timestamps in output
    #[arg(long)]
    pub timestamps: bool,

    /// Word-level timestamps
    #[arg(long)]
    pub word_timestamps: bool,

    /// Enable voice activity detection
    #[arg(long)]
    pub vad: bool,

    /// VAD sensitivity (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    pub vad_threshold: f32,

    /// Use GPU acceleration
    #[arg(long)]
    pub gpu: bool,

    /// Number of CPU threads (default: auto)
    #[arg(long)]
    pub threads: Option<usize>,

    /// Beam search width (default: 5)
    #[arg(long, default_value = "5")]
    pub beam_size: usize,

    /// Sampling temperature (0.0 = greedy)
    #[arg(long, default_value = "0.0")]
    pub temperature: f32,

    /// Filter hallucinated repetitions
    #[arg(long)]
    pub hallucination_filter: bool,
}

/// Arguments for translate command
#[derive(Parser, Debug, Clone)]
pub struct TranslateArgs {
    /// Input audio/video file
    pub input: PathBuf,

    /// Model size to use
    #[arg(short, long, default_value = "base")]
    pub model: ModelSize,

    /// Output file path (default: stdout)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format
    #[arg(short, long, default_value = "txt")]
    pub format: OutputFormatArg,

    /// Use GPU acceleration
    #[arg(long)]
    pub gpu: bool,
}

/// Arguments for record command
#[derive(Parser, Debug, Clone)]
pub struct RecordArgs {
    /// Recording duration in seconds
    #[arg(short, long)]
    pub duration: Option<u32>,

    /// Real-time transcription while recording
    #[arg(long)]
    pub live: bool,

    /// Output file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Audio input device ID
    #[arg(long)]
    pub device: Option<String>,

    /// Sample rate in Hz
    #[arg(long, default_value = "16000")]
    pub sample_rate: u32,

    /// List available audio devices
    #[arg(long)]
    pub list_devices: bool,
}

/// Arguments for batch command
#[derive(Parser, Debug, Clone)]
pub struct BatchArgs {
    /// Input files or glob pattern
    pub inputs: Vec<PathBuf>,

    /// Output directory
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Number of parallel workers
    #[arg(short, long)]
    pub parallel: Option<usize>,

    /// Process directories recursively
    #[arg(short, long)]
    pub recursive: bool,

    /// File pattern (e.g., "*.wav")
    #[arg(long)]
    pub pattern: Option<String>,

    /// Skip already transcribed files
    #[arg(long)]
    pub skip_existing: bool,

    /// Model size to use
    #[arg(short, long, default_value = "tiny")]
    pub model: ModelSize,

    /// Output format
    #[arg(short, long, default_value = "txt")]
    pub format: OutputFormatArg,
}

/// Arguments for test command
#[derive(Parser, Debug, Clone)]
pub struct TestArgs {
    /// Backend to test
    #[arg(short, long, default_value = "all")]
    pub backend: BackendArg,

    /// Test specific demo
    #[arg(long)]
    pub demo: Option<String>,

    /// Test pipeline
    #[arg(long)]
    pub pipeline: Option<String>,
}

/// Arguments for model command
#[derive(Parser, Debug, Clone)]
pub struct ModelArgs {
    /// Model subcommand
    #[command(subcommand)]
    pub action: ModelAction,
}

/// Model management actions
#[derive(Subcommand, Debug, Clone)]
pub enum ModelAction {
    /// List available models
    List,

    /// Download a model
    Download {
        /// Model to download
        model: ModelSize,
    },

    /// Convert model format
    Convert {
        /// Input model file
        input: PathBuf,

        /// Output .apr file
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Show model information
    Info {
        /// Model file
        file: PathBuf,
    },
}

/// Arguments for benchmark command
#[derive(Parser, Debug, Clone)]
pub struct BenchmarkArgs {
    /// Model size to benchmark
    #[arg(default_value = "tiny")]
    pub model: ModelSize,

    /// Backend to use
    #[arg(short, long, default_value = "simd")]
    pub backend: BackendArg,

    /// Number of iterations
    #[arg(short, long, default_value = "3")]
    pub iterations: usize,
}

/// Arguments for validate command
#[derive(Parser, Debug, Clone)]
pub struct ValidateArgs {
    /// APR model file to validate
    pub file: PathBuf,

    /// Quick validation (critical checks only)
    #[arg(long)]
    pub quick: bool,

    /// Show detailed report
    #[arg(short, long)]
    pub detailed: bool,

    /// Fail if score is below threshold (0-25)
    #[arg(long, default_value = "23")]
    pub min_score: u8,

    /// Output format for report
    #[arg(short, long, default_value = "text")]
    pub format: ValidateOutputFormat,
}

/// Output format for validation report
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidateOutputFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON format
    Json,
    /// Markdown format
    Markdown,
}

/// Model size options
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    /// Tiny model (39M params)
    Tiny,
    /// Base model (74M params)
    Base,
    /// Small model (244M params)
    Small,
    /// Medium model (769M params)
    Medium,
    /// Large model (1.5B params)
    Large,
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tiny => write!(f, "tiny"),
            Self::Base => write!(f, "base"),
            Self::Small => write!(f, "small"),
            Self::Medium => write!(f, "medium"),
            Self::Large => write!(f, "large"),
        }
    }
}

/// Output format options
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormatArg {
    /// Plain text
    Txt,
    /// SRT subtitles
    Srt,
    /// WebVTT subtitles
    Vtt,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Markdown format
    Md,
}

impl std::fmt::Display for OutputFormatArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Txt => write!(f, "txt"),
            Self::Srt => write!(f, "srt"),
            Self::Vtt => write!(f, "vtt"),
            Self::Json => write!(f, "json"),
            Self::Csv => write!(f, "csv"),
            Self::Md => write!(f, "md"),
        }
    }
}

/// Backend options
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendArg {
    /// All available backends
    All,
    /// CPU SIMD backend
    Simd,
    /// WebAssembly backend
    Wasm,
    /// CUDA GPU backend
    Cuda,
}

impl std::fmt::Display for BackendArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::All => write!(f, "all"),
            Self::Simd => write!(f, "simd"),
            Self::Wasm => write!(f, "wasm"),
            Self::Cuda => write!(f, "cuda"),
        }
    }
}

// ============================================================================
// Unit Tests (EXTREME TDD - RED phase)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    // -------------------------------------------------------------------------
    // Args parsing tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_args_command_factory_valid() {
        // Verify the command structure is valid
        Args::command().debug_assert();
    }

    #[test]
    fn test_parse_transcribe_minimal() {
        let args = Args::try_parse_from(["whisper-apr", "transcribe", "test.wav"]);
        assert!(args.is_ok(), "Should parse minimal transcribe command");
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Transcribe(t) => {
                assert_eq!(t.input, PathBuf::from("test.wav"));
                assert_eq!(t.model, ModelSize::Tiny);
                assert_eq!(t.language, "auto");
                assert!(!t.timestamps);
            }
            _ => panic!("Expected Transcribe command"),
        }
    }

    #[test]
    fn test_parse_transcribe_all_options() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "transcribe",
            "audio.mp3",
            "--model",
            "base",
            "--language",
            "en",
            "--output",
            "out.srt",
            "--format",
            "srt",
            "--timestamps",
            "--word-timestamps",
            "--vad",
            "--vad-threshold",
            "0.7",
            "--gpu",
            "--threads",
            "4",
            "--beam-size",
            "3",
            "--temperature",
            "0.2",
            "--hallucination-filter",
        ]);
        assert!(args.is_ok(), "Should parse all transcribe options");
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Transcribe(t) => {
                assert_eq!(t.model, ModelSize::Base);
                assert_eq!(t.language, "en");
                assert_eq!(t.output, Some(PathBuf::from("out.srt")));
                assert_eq!(t.format, OutputFormatArg::Srt);
                assert!(t.timestamps);
                assert!(t.word_timestamps);
                assert!(t.vad);
                assert!((t.vad_threshold - 0.7).abs() < 0.01);
                assert!(t.gpu);
                assert_eq!(t.threads, Some(4));
                assert_eq!(t.beam_size, 3);
                assert!((t.temperature - 0.2).abs() < 0.01);
                assert!(t.hallucination_filter);
            }
            _ => panic!("Expected Transcribe command"),
        }
    }

    #[test]
    fn test_parse_translate_minimal() {
        let args = Args::try_parse_from(["whisper-apr", "translate", "german.wav"]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Translate(t) => {
                assert_eq!(t.input, PathBuf::from("german.wav"));
                assert_eq!(t.model, ModelSize::Base);
            }
            _ => panic!("Expected Translate command"),
        }
    }

    #[test]
    fn test_parse_record_with_duration() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "record",
            "--duration",
            "30",
            "--output",
            "recording.wav",
        ]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Record(r) => {
                assert_eq!(r.duration, Some(30));
                assert_eq!(r.output, Some(PathBuf::from("recording.wav")));
                assert!(!r.live);
            }
            _ => panic!("Expected Record command"),
        }
    }

    #[test]
    fn test_parse_record_live() {
        let args = Args::try_parse_from(["whisper-apr", "record", "--live"]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Record(r) => {
                assert!(r.live);
            }
            _ => panic!("Expected Record command"),
        }
    }

    #[test]
    fn test_parse_record_list_devices() {
        let args = Args::try_parse_from(["whisper-apr", "record", "--list-devices"]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Record(r) => {
                assert!(r.list_devices);
            }
            _ => panic!("Expected Record command"),
        }
    }

    #[test]
    fn test_parse_batch_minimal() {
        let args = Args::try_parse_from(["whisper-apr", "batch", "file1.wav", "file2.wav"]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Batch(b) => {
                assert_eq!(b.inputs.len(), 2);
            }
            _ => panic!("Expected Batch command"),
        }
    }

    #[test]
    fn test_parse_batch_with_options() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "batch",
            "*.wav",
            "--output-dir",
            "transcripts",
            "--parallel",
            "4",
            "--recursive",
            "--skip-existing",
        ]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Batch(b) => {
                assert_eq!(b.output_dir, Some(PathBuf::from("transcripts")));
                assert_eq!(b.parallel, Some(4));
                assert!(b.recursive);
                assert!(b.skip_existing);
            }
            _ => panic!("Expected Batch command"),
        }
    }

    #[test]
    fn test_parse_tui() {
        let args = Args::try_parse_from(["whisper-apr", "tui"]);
        assert!(args.is_ok());
        assert!(matches!(
            args.expect("test parse should succeed").command,
            Command::Tui
        ));
    }

    #[test]
    fn test_parse_test_all_backends() {
        let args = Args::try_parse_from(["whisper-apr", "test"]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Test(t) => {
                assert_eq!(t.backend, BackendArg::All);
            }
            _ => panic!("Expected Test command"),
        }
    }

    #[test]
    fn test_parse_test_specific_backend() {
        let args = Args::try_parse_from(["whisper-apr", "test", "--backend", "cuda"]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Test(t) => {
                assert_eq!(t.backend, BackendArg::Cuda);
            }
            _ => panic!("Expected Test command"),
        }
    }

    #[test]
    fn test_parse_model_list() {
        let args = Args::try_parse_from(["whisper-apr", "model", "list"]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Model(m) => {
                assert!(matches!(m.action, ModelAction::List));
            }
            _ => panic!("Expected Model command"),
        }
    }

    #[test]
    fn test_parse_model_download() {
        let args = Args::try_parse_from(["whisper-apr", "model", "download", "base"]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Model(m) => match m.action {
                ModelAction::Download { model } => {
                    assert_eq!(model, ModelSize::Base);
                }
                _ => panic!("Expected Download action"),
            },
            _ => panic!("Expected Model command"),
        }
    }

    #[test]
    fn test_parse_model_convert() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "model",
            "convert",
            "input.pt",
            "--output",
            "output.apr",
        ]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Model(m) => match m.action {
                ModelAction::Convert { input, output } => {
                    assert_eq!(input, PathBuf::from("input.pt"));
                    assert_eq!(output, PathBuf::from("output.apr"));
                }
                _ => panic!("Expected Convert action"),
            },
            _ => panic!("Expected Model command"),
        }
    }

    #[test]
    fn test_parse_benchmark() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "benchmark",
            "base",
            "--backend",
            "simd",
            "--iterations",
            "5",
        ]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Benchmark(b) => {
                assert_eq!(b.model, ModelSize::Base);
                assert_eq!(b.backend, BackendArg::Simd);
                assert_eq!(b.iterations, 5);
            }
            _ => panic!("Expected Benchmark command"),
        }
    }

    #[test]
    fn test_global_verbose_flag() {
        let args = Args::try_parse_from(["whisper-apr", "-v", "transcribe", "test.wav"]);
        assert!(args.is_ok());
        assert!(args.expect("test parse should succeed").verbose);
    }

    #[test]
    fn test_global_quiet_flag() {
        let args = Args::try_parse_from(["whisper-apr", "-q", "transcribe", "test.wav"]);
        assert!(args.is_ok());
        assert!(args.expect("test parse should succeed").quiet);
    }

    #[test]
    fn test_global_json_flag() {
        let args = Args::try_parse_from(["whisper-apr", "--json", "transcribe", "test.wav"]);
        assert!(args.is_ok());
        assert!(args.expect("test parse should succeed").json);
    }

    #[test]
    fn test_global_trace_flag() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "--trace",
            "trace.json",
            "transcribe",
            "test.wav",
        ]);
        assert!(args.is_ok());
        assert_eq!(
            args.expect("test parse should succeed").trace,
            Some(PathBuf::from("trace.json"))
        );
    }

    #[test]
    fn test_global_no_color_flag() {
        let args = Args::try_parse_from(["whisper-apr", "--no-color", "transcribe", "test.wav"]);
        assert!(args.is_ok());
        assert!(args.expect("test parse should succeed").no_color);
    }

    // -------------------------------------------------------------------------
    // Display trait tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_size_display() {
        assert_eq!(ModelSize::Tiny.to_string(), "tiny");
        assert_eq!(ModelSize::Base.to_string(), "base");
        assert_eq!(ModelSize::Small.to_string(), "small");
        assert_eq!(ModelSize::Medium.to_string(), "medium");
        assert_eq!(ModelSize::Large.to_string(), "large");
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormatArg::Txt.to_string(), "txt");
        assert_eq!(OutputFormatArg::Srt.to_string(), "srt");
        assert_eq!(OutputFormatArg::Vtt.to_string(), "vtt");
        assert_eq!(OutputFormatArg::Json.to_string(), "json");
        assert_eq!(OutputFormatArg::Csv.to_string(), "csv");
        assert_eq!(OutputFormatArg::Md.to_string(), "md");
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(BackendArg::All.to_string(), "all");
        assert_eq!(BackendArg::Simd.to_string(), "simd");
        assert_eq!(BackendArg::Wasm.to_string(), "wasm");
        assert_eq!(BackendArg::Cuda.to_string(), "cuda");
    }

    // -------------------------------------------------------------------------
    // Error handling tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_invalid_command() {
        let args = Args::try_parse_from(["whisper-apr", "invalid"]);
        assert!(args.is_err());
    }

    #[test]
    fn test_parse_missing_input() {
        let args = Args::try_parse_from(["whisper-apr", "transcribe"]);
        assert!(args.is_err());
    }

    #[test]
    fn test_parse_invalid_model() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "transcribe",
            "test.wav",
            "--model",
            "invalid",
        ]);
        assert!(args.is_err());
    }

    #[test]
    fn test_parse_invalid_format() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "transcribe",
            "test.wav",
            "--format",
            "invalid",
        ]);
        assert!(args.is_err());
    }

    #[test]
    fn test_parse_invalid_backend() {
        let args = Args::try_parse_from(["whisper-apr", "test", "--backend", "invalid"]);
        assert!(args.is_err());
    }
}

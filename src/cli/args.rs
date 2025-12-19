//! Command-line argument parsing for whisper-apr CLI
//!
//! Uses clap derive macros for type-safe argument parsing.
//! All argument structures are unit-testable.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

/// Parse temperature value with range validation (0.0 to 1.0)
fn parse_temperature(s: &str) -> Result<f32, String> {
    let temp: f32 = s.parse().map_err(|_| format!("'{}' is not a valid number", s))?;
    if !(0.0..=1.0).contains(&temp) {
        return Err(format!(
            "temperature {} is out of range (must be between 0.0 and 1.0)",
            temp
        ));
    }
    Ok(temp)
}

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
    #[arg(short, long, global = true, conflicts_with = "verbose")]
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

    /// Real-time streaming transcription from microphone (whisper.cpp: whisper-stream)
    Stream(StreamArgs),

    /// HTTP API server (whisper.cpp: whisper-server)
    #[command(alias = "server")]
    Serve(ServeArgs),

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
    #[command(alias = "bench")]
    Benchmark(BenchmarkArgs),

    /// Validate APR model file (25-point QA checklist)
    Validate(ValidateArgs),

    /// Compare output against whisper.cpp (parity testing)
    Parity(ParityArgs),

    /// Quantize model to smaller size (whisper.cpp: whisper-quantize)
    Quantize(QuantizeArgs),

    /// Voice command recognition (whisper.cpp: whisper-command)
    Command(CommandArgs),
}

/// Arguments for transcribe command
///
/// Designed for parity with whisper.cpp CLI (§6.2 of whisper-cli-parity.md)
#[derive(Parser, Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // CLI flags are naturally boolean
pub struct TranscribeArgs {
    /// Input audio/video file
    #[arg(short = 'f', long = "file")]
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

    /// Detect language and exit (whisper.cpp: -dl)
    #[arg(long)]
    pub detect_language: bool,

    /// Output file path (default: stdout)
    #[arg(long = "output-file")]
    pub output: Option<PathBuf>,

    /// Output format
    #[arg(short = 'o', long, default_value = "txt")]
    pub format: OutputFormatArg,

    // -------------------------------------------------------------------------
    // Timing/offset arguments (whisper.cpp parity §6.2)
    // -------------------------------------------------------------------------
    /// Time offset in milliseconds (whisper.cpp: -ot)
    #[arg(long = "offset-t", default_value = "0")]
    pub offset_t: u32,

    /// Segment offset (whisper.cpp: -on)
    #[arg(long = "offset-n", default_value = "0")]
    pub offset_n: u32,

    /// Duration to process in milliseconds (whisper.cpp: -d)
    #[arg(short = 'd', long, default_value = "0")]
    pub duration: u32,

    // -------------------------------------------------------------------------
    // Context/length arguments
    // -------------------------------------------------------------------------
    /// Max context tokens (-1 = use default) (whisper.cpp: -mc)
    #[arg(long = "max-context", default_value = "-1")]
    pub max_context: i32,

    /// Max segment length (0 = no limit) (whisper.cpp: -ml)
    #[arg(long = "max-len", default_value = "0")]
    pub max_len: u32,

    /// Audio context size (whisper.cpp: -ac)
    #[arg(long = "audio-ctx", default_value = "0")]
    pub audio_ctx: u32,

    // -------------------------------------------------------------------------
    // Decoding strategy arguments
    // -------------------------------------------------------------------------
    /// Best-of candidates for sampling (whisper.cpp: -bo)
    #[arg(long = "best-of", default_value = "2")]
    pub best_of: u32,

    /// Beam search size (-1 = greedy) (whisper.cpp: -bs)
    #[arg(long = "beam-size", default_value = "-1")]
    pub beam_size: i32,

    /// Sampling temperature (0.0 = greedy, max 1.0) (whisper.cpp: -tp)
    #[arg(long = "temperature", default_value = "0.0", value_parser = parse_temperature)]
    pub temperature: f32,

    /// Temperature increment on fallback (whisper.cpp: -tpi)
    #[arg(long = "temperature-inc", default_value = "0.2")]
    pub temperature_inc: f32,

    /// Disable temperature fallback (whisper.cpp: -nf)
    #[arg(long = "no-fallback")]
    pub no_fallback: bool,

    // -------------------------------------------------------------------------
    // Word/segment splitting
    // -------------------------------------------------------------------------
    /// Split on word boundaries (whisper.cpp: -sow)
    #[arg(long = "split-on-word")]
    pub split_on_word: bool,

    /// Word timestamp threshold (whisper.cpp: -wt)
    #[arg(long = "word-thold", default_value = "0.01")]
    pub word_thold: f32,

    /// Word-level timestamps
    #[arg(long)]
    pub word_timestamps: bool,

    /// Include timestamps in output
    #[arg(long)]
    pub timestamps: bool,

    /// Omit timestamps from output (whisper.cpp: -nt)
    #[arg(long = "no-timestamps")]
    pub no_timestamps: bool,

    // -------------------------------------------------------------------------
    // Threshold arguments
    // -------------------------------------------------------------------------
    /// Entropy threshold for decoder (whisper.cpp: -et)
    #[arg(long = "entropy-thold", default_value = "2.40")]
    pub entropy_thold: f32,

    /// Log probability threshold (whisper.cpp: -lpt)
    #[arg(long = "logprob-thold", default_value = "-1.0")]
    pub logprob_thold: f32,

    /// No-speech probability threshold (whisper.cpp: -nth)
    #[arg(long = "no-speech-thold", default_value = "0.6")]
    pub no_speech_thold: f32,

    // -------------------------------------------------------------------------
    // Prompt/grammar arguments
    // -------------------------------------------------------------------------
    /// Initial prompt for decoder (whisper.cpp: --prompt)
    #[arg(long, default_value = "")]
    pub prompt: String,

    /// Regex pattern to suppress tokens (whisper.cpp: --suppress-regex)
    #[arg(long = "suppress-regex", default_value = "")]
    pub suppress_regex: String,

    /// GBNF grammar for constrained decoding (whisper.cpp: --grammar)
    #[arg(long, default_value = "")]
    pub grammar: String,

    /// Grammar rule name (whisper.cpp: --grammar-rule)
    #[arg(long = "grammar-rule", default_value = "")]
    pub grammar_rule: String,

    /// Grammar penalty (whisper.cpp: --grammar-penalty)
    #[arg(long = "grammar-penalty", default_value = "100.0")]
    pub grammar_penalty: f32,

    // -------------------------------------------------------------------------
    // VAD arguments (§6.5)
    // -------------------------------------------------------------------------
    /// Enable voice activity detection (whisper.cpp: --vad)
    #[arg(long)]
    pub vad: bool,

    /// Path to VAD model file (whisper.cpp: -vm)
    #[arg(long = "vad-model")]
    pub vad_model: Option<PathBuf>,

    /// VAD threshold (whisper.cpp: -vt)
    #[arg(long = "vad-threshold", default_value = "0.5")]
    pub vad_threshold: f32,

    /// Min speech duration in ms (whisper.cpp: -vspd)
    #[arg(long = "vad-min-speech-ms", default_value = "250")]
    pub vad_min_speech_ms: u32,

    /// Min silence duration in ms (whisper.cpp: -vsd)
    #[arg(long = "vad-min-silence-ms", default_value = "100")]
    pub vad_min_silence_ms: u32,

    /// Max speech duration in seconds (whisper.cpp: -vmsd)
    #[arg(long = "vad-max-speech-s")]
    pub vad_max_speech_s: Option<f32>,

    /// Speech padding in ms (whisper.cpp: -vp)
    #[arg(long = "vad-pad-ms", default_value = "30")]
    pub vad_pad_ms: u32,

    /// VAD samples overlap (whisper.cpp: -vo)
    #[arg(long = "vad-overlap", default_value = "0.1")]
    pub vad_overlap: f32,

    // -------------------------------------------------------------------------
    // Hardware/performance arguments
    // -------------------------------------------------------------------------
    /// Number of CPU threads (default: auto) (whisper.cpp: -t)
    #[arg(short = 't', long)]
    pub threads: Option<u32>,

    /// Number of processors (whisper.cpp: -p)
    #[arg(short = 'p', long, default_value = "1")]
    pub processors: u32,

    /// Use GPU acceleration
    #[arg(long)]
    pub gpu: bool,

    /// Disable GPU (whisper.cpp: -ng)
    #[arg(long = "no-gpu")]
    pub no_gpu: bool,

    /// Enable flash attention (whisper.cpp: -fa)
    #[arg(long = "flash-attn")]
    pub flash_attn: bool,

    /// Disable flash attention (whisper.cpp: -nfa)
    #[arg(long = "no-flash-attn")]
    pub no_flash_attn: bool,

    // -------------------------------------------------------------------------
    // Display arguments (§6.4)
    // -------------------------------------------------------------------------
    /// Suppress non-essential output (whisper.cpp: -np)
    #[arg(long = "no-prints")]
    pub no_prints: bool,

    /// Print special tokens (whisper.cpp: -ps)
    #[arg(long = "print-special")]
    pub print_special: bool,

    /// Color-coded confidence output (whisper.cpp: -pc)
    #[arg(long = "colors")]
    pub colors: bool,

    /// Show confidence scores (whisper.cpp: --print-confidence)
    #[arg(long = "confidence")]
    pub confidence: bool,

    /// Show progress percentage (whisper.cpp: -pp)
    /// Note: Incompatible with --quiet (use only one)
    #[arg(long = "progress")]
    pub progress: bool,

    // -------------------------------------------------------------------------
    // Other
    // -------------------------------------------------------------------------
    /// Translate to English (whisper.cpp: -tr)
    #[arg(long = "translate")]
    pub translate: bool,

    /// Filter hallucinated repetitions
    #[arg(long)]
    pub hallucination_filter: bool,
}

/// Arguments for translate command
#[derive(Parser, Debug, Clone)]
pub struct TranslateArgs {
    /// Input audio/video file
    #[arg(short = 'f', long = "file")]
    pub input: PathBuf,

    /// Model size to use
    #[arg(short, long, default_value = "base")]
    pub model: ModelSize,

    /// Output file path (default: stdout)
    #[arg(long = "output-file")]
    pub output: Option<PathBuf>,

    /// Output format
    #[arg(short = 'o', long, default_value = "txt")]
    pub format: OutputFormatArg,

    /// Use GPU acceleration
    #[arg(long)]
    pub gpu: bool,
}

/// Arguments for stream command (§6.7 - whisper.cpp: whisper-stream)
#[derive(Parser, Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct StreamArgs {
    /// Model size to use
    #[arg(short, long, default_value = "tiny")]
    pub model: ModelSize,

    /// Path to .apr model file (overrides --model)
    #[arg(long)]
    pub model_path: Option<PathBuf>,

    /// Source language (ISO 639-1) or 'auto' for detection
    #[arg(short, long, default_value = "auto")]
    pub language: String,

    /// Step size in milliseconds (whisper.cpp: --step)
    #[arg(long, default_value = "3000")]
    pub step: u32,

    /// Audio length in milliseconds (whisper.cpp: --length)
    #[arg(long, default_value = "10000")]
    pub length: u32,

    /// Audio to keep from previous step (whisper.cpp: --keep)
    #[arg(long, default_value = "200")]
    pub keep: u32,

    /// Audio capture device ID (whisper.cpp: -c)
    #[arg(short = 'c', long = "capture", default_value = "-1")]
    pub capture: i32,

    /// Max tokens per audio chunk (whisper.cpp: -mt)
    #[arg(long = "max-tokens", default_value = "32")]
    pub max_tokens: u32,

    /// VAD threshold (whisper.cpp: -vth)
    #[arg(long = "vad-thold", default_value = "0.6")]
    pub vad_thold: f32,

    /// High-pass frequency threshold (whisper.cpp: -fth)
    #[arg(long = "freq-thold", default_value = "100.0")]
    pub freq_thold: f32,

    /// Keep context between audio chunks (whisper.cpp: -kc)
    #[arg(long = "keep-context")]
    pub keep_context: bool,

    /// Save audio to file (whisper.cpp: -sa)
    #[arg(long = "save-audio")]
    pub save_audio: bool,

    /// Number of CPU threads
    #[arg(short = 't', long)]
    pub threads: Option<u32>,

    /// Translate to English
    #[arg(long)]
    pub translate: bool,
}

/// Arguments for serve command (§6.6 - whisper.cpp: whisper-server)
#[derive(Parser, Debug, Clone)]
pub struct ServeArgs {
    /// Model size to use
    #[arg(short, long, default_value = "tiny")]
    pub model: ModelSize,

    /// Path to .apr model file (overrides --model)
    #[arg(long)]
    pub model_path: Option<PathBuf>,

    /// Host address to bind (whisper.cpp: --host)
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to listen on (whisper.cpp: --port)
    #[arg(long, default_value = "8080")]
    pub port: u16,

    /// Path to public directory for static files (whisper.cpp: --public)
    #[arg(long)]
    pub public: Option<PathBuf>,

    /// Request path prefix (whisper.cpp: --request-path)
    #[arg(long = "request-path", default_value = "")]
    pub request_path: String,

    /// Inference endpoint path (whisper.cpp: --inference-path)
    #[arg(long = "inference-path", default_value = "/inference")]
    pub inference_path: String,

    /// Auto-convert uploaded audio to WAV (whisper.cpp: --convert)
    #[arg(long)]
    pub convert: bool,

    /// Temporary directory for conversions (whisper.cpp: --tmp-dir)
    #[arg(long = "tmp-dir", default_value = ".")]
    pub tmp_dir: PathBuf,

    /// Number of CPU threads
    #[arg(short = 't', long)]
    pub threads: Option<u32>,
}

/// Arguments for parity command (whisper.cpp comparison)
#[derive(Parser, Debug, Clone)]
pub struct ParityArgs {
    /// Input audio file
    #[arg(short = 'f', long = "file")]
    pub input: PathBuf,

    /// Path to whisper.cpp binary (default: search PATH)
    #[arg(long = "whisper-cpp")]
    pub whisper_cpp: Option<PathBuf>,

    /// Path to whisper.cpp model file (ggml format)
    #[arg(long = "cpp-model")]
    pub cpp_model: Option<PathBuf>,

    /// Model size to use for whisper-apr
    #[arg(short, long, default_value = "tiny")]
    pub model: ModelSize,

    /// Path to .apr model file (overrides --model)
    #[arg(long)]
    pub model_path: Option<PathBuf>,

    /// Maximum allowed Word Error Rate (0.0-1.0)
    #[arg(long = "max-wer", default_value = "0.01")]
    pub max_wer: f64,

    /// Timestamp tolerance in milliseconds
    #[arg(long = "timestamp-tolerance", default_value = "50")]
    pub timestamp_tolerance_ms: u32,

    /// Output comparison report as JSON
    #[arg(long)]
    pub json: bool,

    /// Include HuggingFace Transformers comparison
    #[arg(long = "include-hf")]
    pub include_hf: bool,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

/// Arguments for quantize command (whisper.cpp: whisper-quantize)
#[derive(Parser, Debug, Clone)]
pub struct QuantizeArgs {
    /// Input model file (ggml or apr format)
    pub input: PathBuf,

    /// Output model file
    pub output: PathBuf,

    /// Quantization type
    #[arg(short = 'Q', long, default_value = "q5-0")]
    pub quantize: QuantizeType,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

/// Quantization types (whisper.cpp parity)
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeType {
    /// 32-bit floating point (no quantization)
    #[value(name = "f32")]
    F32,
    /// 16-bit floating point
    #[value(name = "f16")]
    F16,
    /// 8-bit integer (fastest, lowest quality)
    #[value(name = "q8-0")]
    Q8_0,
    /// 5-bit quantization (default)
    #[value(name = "q5-0")]
    Q5_0,
    /// 5-bit quantization variant 1
    #[value(name = "q5-1")]
    Q5_1,
    /// 4-bit quantization (smallest)
    #[value(name = "q4-0")]
    Q4_0,
    /// 4-bit quantization variant 1
    #[value(name = "q4-1")]
    Q4_1,
}

impl std::fmt::Display for QuantizeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::Q8_0 => write!(f, "q8-0"),
            Self::Q5_0 => write!(f, "q5-0"),
            Self::Q5_1 => write!(f, "q5-1"),
            Self::Q4_0 => write!(f, "q4-0"),
            Self::Q4_1 => write!(f, "q4-1"),
        }
    }
}

/// Arguments for command mode (voice command recognition)
#[derive(Parser, Debug, Clone)]
pub struct CommandArgs {
    /// Model size to use
    #[arg(short, long, default_value = "tiny")]
    pub model: ModelSize,

    /// Path to .apr model file (overrides --model)
    #[arg(long)]
    pub model_path: Option<PathBuf>,

    /// Commands file (one command per line)
    #[arg(short = 'c', long)]
    pub commands: Option<PathBuf>,

    /// Prompt with available commands
    #[arg(long)]
    pub prompt: Option<String>,

    /// Grammar file for constrained recognition
    #[arg(long)]
    pub grammar: Option<PathBuf>,

    /// Audio capture device ID
    #[arg(long, default_value = "-1")]
    pub capture: i32,

    /// VAD threshold
    #[arg(long = "vad-thold", default_value = "0.6")]
    pub vad_thold: f32,

    /// Continuous mode (loop listening)
    #[arg(long)]
    pub continuous: bool,

    /// Number of CPU threads
    #[arg(short = 't', long)]
    pub threads: Option<u32>,
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
    /// Extended JSON with token-level details
    JsonFull,
    /// CSV format
    Csv,
    /// LRC lyrics format (whisper.cpp: -olrc)
    Lrc,
    /// Karaoke script with word timestamps (whisper.cpp: -owts)
    Wts,
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
            Self::JsonFull => write!(f, "json-full"),
            Self::Csv => write!(f, "csv"),
            Self::Lrc => write!(f, "lrc"),
            Self::Wts => write!(f, "wts"),
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
        let args = Args::try_parse_from(["whisper-apr", "transcribe", "-f", "test.wav"]);
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
            "-f",
            "audio.mp3",
            "--model",
            "base",
            "--language",
            "en",
            "--output-file",
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
        let args = Args::try_parse_from(["whisper-apr", "translate", "-f", "german.wav"]);
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
        let args = Args::try_parse_from(["whisper-apr", "-v", "transcribe", "-f", "test.wav"]);
        assert!(args.is_ok());
        assert!(args.expect("test parse should succeed").verbose);
    }

    #[test]
    fn test_global_quiet_flag() {
        let args = Args::try_parse_from(["whisper-apr", "-q", "transcribe", "-f", "test.wav"]);
        assert!(args.is_ok());
        assert!(args.expect("test parse should succeed").quiet);
    }

    #[test]
    fn test_global_json_flag() {
        let args = Args::try_parse_from(["whisper-apr", "--json", "transcribe", "-f", "test.wav"]);
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
            "-f",
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
        let args =
            Args::try_parse_from(["whisper-apr", "--no-color", "transcribe", "-f", "test.wav"]);
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

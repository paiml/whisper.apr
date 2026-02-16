//! Command-line argument parsing for whisper-apr CLI
//!
//! Uses clap derive macros for type-safe argument parsing.
//! All argument structures are unit-testable.
#![allow(clippy::struct_excessive_bools)]

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

/// Parse temperature value with range validation (0.0 to 1.0)
fn parse_temperature(s: &str) -> Result<f32, String> {
    let temp: f32 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    if !(0.0..=1.0).contains(&temp) {
        return Err(format!(
            "temperature {temp} is out of range (must be between 0.0 and 1.0)"
        ));
    }
    Ok(temp)
}

/// Expand response files in command-line arguments.
///
/// Arguments prefixed with `@` are treated as response files containing
/// additional arguments, one per line. This allows passing many arguments
/// via a file instead of the command line.
///
/// # Arguments
/// * `args` - Command-line arguments to expand
///
/// # Returns
/// Expanded arguments with response file contents inlined
///
/// # Errors
/// Returns error if a response file cannot be read
///
/// # Example
/// ```ignore
/// // args.txt contains:
/// // -f
/// // input.wav
/// // --language
/// // en
///
/// let args = vec!["whisper-apr".into(), "transcribe".into(), "@args.txt".into()];
/// let expanded = expand_response_files(args)?;
/// // expanded = ["whisper-apr", "transcribe", "-f", "input.wav", "--language", "en"]
/// ```
pub fn expand_response_files(args: Vec<String>) -> Result<Vec<String>, std::io::Error> {
    let mut result = Vec::with_capacity(args.len());

    for arg in args {
        if let Some(file_path) = arg.strip_prefix('@') {
            // Read response file and add each line as an argument
            let contents = std::fs::read_to_string(file_path)?;
            for line in contents.lines() {
                let trimmed = line.trim();
                // Skip empty lines and comments
                if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    result.push(trimmed.to_string());
                }
            }
        } else {
            result.push(arg);
        }
    }

    Ok(result)
}

/// whisper-apr: WASM-first automatic speech recognition
///
/// A high-performance speech recognition CLI that runs natively and in browsers.
/// Install with: cargo install whisper-apr --features cli
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
#[allow(clippy::large_enum_variant)]
pub enum Command {
    /// Transcribe audio/video to text
    Transcribe(TranscribeArgs),

    /// Translate speech to English
    Translate(TranslateArgs),

    /// Summarize transcript using LFM2 model (WAPR-LFM2-001)
    Summarize(SummarizeArgs),

    /// Real-time streaming transcription from microphone (whisper.cpp: whisper-stream)
    Stream(StreamArgs),

    /// HTTP API server (whisper.cpp: whisper-server)
    #[command(alias = "server")]
    Serve(ServeArgs),

    /// Record audio from microphone
    Record(RecordArgs),

    /// Process multiple files in parallel
    Batch(BatchArgs),

    /// Transcribe all audio files in a folder (WAPR-PERF-004)
    ///
    /// Structure-preserving batch transcription with brick profiling.
    /// Mirrors input directory structure to output directory.
    #[command(alias = "folder")]
    TranscribeFolder(TranscribeFolderArgs),

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

    /// Self-diagnostic checks (tokenizer, model config, known issues)
    #[command(alias = "doctor")]
    Diagnose(DiagnoseArgs),

    /// Convert HuggingFace safetensors to APR2 format (WAPR-LFM2-004)
    Convert(ConvertArgs),

    /// Export APR model to SafeTensors format (WAPR-PUB-001)
    Export(ExportArgs),

    /// APR model format tools (inspect, lint, convert, diff)
    Apr(crate::cli::apr_args::AprArgs),

    /// Run self-test (diagnose + backend test + optional transcription)
    Selftest(SelftestArgs),
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

    /// Print memory usage stats (whisper.cpp: -pm)
    #[arg(long = "print-memory")]
    pub print_memory: bool,

    // -------------------------------------------------------------------------
    // Profiling (WAPR-PERF-004)
    // -------------------------------------------------------------------------
    /// Enable component timing breakdown (mel, encoder, decoder)
    ///
    /// Outputs timing breakdown like apr-cli:
    /// [PROFILE] Mel spectrogram:   15ms (2.5%)
    /// [PROFILE] Encoder:          120ms (20.1%)
    /// [PROFILE] Decoder:          450ms (75.5%)
    #[arg(long)]
    pub profile: bool,

    // -------------------------------------------------------------------------
    // Other
    // -------------------------------------------------------------------------
    /// Translate to English (whisper.cpp: -tr)
    #[arg(long = "translate")]
    pub translate: bool,

    /// Filter hallucinated repetitions
    #[arg(long)]
    pub hallucination_filter: bool,

    /// Audio playback speed multiplier (whisper.cpp: --speed)
    #[arg(long = "speed", default_value = "1.0")]
    pub speed: f32,

    // -------------------------------------------------------------------------
    // ZRAM optimization arguments (GitHub #8)
    // -------------------------------------------------------------------------
    /// Cache directory for models and intermediate data
    #[arg(long = "cache-dir")]
    pub cache_dir: Option<PathBuf>,

    /// Enable ZRAM-aware allocation for reduced memory usage
    /// When enabled, uses optimized buffer sizes for trueno-ublk ZRAM
    #[arg(long = "zram-optimized")]
    pub zram_optimized: bool,

    // -------------------------------------------------------------------------
    // Post-transcription summarization (Phase 2 - Section 18.5)
    // -------------------------------------------------------------------------
    /// Enable post-transcription summarization with LFM2
    #[arg(long)]
    pub summarize: bool,

    /// Path to LFM2 model file for summarization (.apr2 format)
    #[arg(long = "lfm2-model")]
    pub lfm2_model: Option<PathBuf>,

    /// Output file for summary (default: <input>.summary.json)
    #[arg(long = "summary-output")]
    pub summary_output: Option<PathBuf>,

    /// Summary format (json, text, markdown, bullets)
    #[arg(long = "summary-format", default_value = "json")]
    pub summary_format: SummarizeFormat,

    /// Include action items in summary
    #[arg(long = "action-items")]
    pub action_items: bool,

    /// Include key points in summary
    #[arg(long = "key-points")]
    pub key_points: bool,
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

    /// Number of CPU threads (default: auto) (whisper.cpp: -t)
    #[arg(short = 't', long)]
    pub threads: Option<u32>,
}

/// Arguments for summarize command (WAPR-LFM2-001)
///
/// Summarizes transcript text using LFM2-2.6B-Transcript model.
/// This is a post-transcription step that converts raw transcripts
/// into structured summaries with bullet points and action items.
///
/// # Example
///
/// ```bash
/// # Summarize a transcript file
/// whisper-apr summarize -f transcript.txt -o summary.json
///
/// # End-to-end: transcribe + summarize
/// whisper-apr transcribe -f meeting.wav | whisper-apr summarize
///
/// # With custom model path
/// whisper-apr summarize -f transcript.txt --model-path ./lfm2-2.6b.apr2
/// ```
#[derive(Parser, Debug, Clone)]
pub struct SummarizeArgs {
    /// Input transcript file (or stdin if not provided)
    #[arg(short = 'f', long = "file")]
    pub input: Option<PathBuf>,

    /// Path to LFM2 model file (.apr2 format)
    #[arg(long)]
    pub model_path: Option<PathBuf>,

    /// Path to tokenizer.json file (HuggingFace format)
    #[arg(long)]
    pub tokenizer_path: Option<PathBuf>,

    /// Output file path (default: stdout)
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    /// Output format for summary
    #[arg(long, default_value = "json")]
    pub format: SummarizeFormat,

    /// Maximum tokens to generate
    #[arg(long, default_value = "1024")]
    pub max_tokens: u32,

    /// Sampling temperature (0.0 = deterministic)
    #[arg(long, default_value = "0.3")]
    pub temperature: f32,

    /// Maximum context length for input
    #[arg(long, default_value = "4096")]
    pub max_context: u32,

    /// Use WebGPU acceleration
    #[arg(long)]
    pub webgpu: bool,

    /// Stream output token by token
    #[arg(long)]
    pub stream: bool,

    /// Include action items extraction
    #[arg(long)]
    pub action_items: bool,

    /// Include key points extraction
    #[arg(long)]
    pub key_points: bool,

    /// Custom prompt template (overrides defaults)
    #[arg(long)]
    pub prompt: Option<String>,
}

/// Output format for summarization
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SummarizeFormat {
    /// JSON with structured fields
    #[default]
    Json,
    /// Plain text summary
    Text,
    /// Markdown with sections
    Markdown,
    /// Bullet points only
    Bullets,
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

    // -------------------------------------------------------------------------
    // ZRAM optimization arguments (GitHub #8)
    // -------------------------------------------------------------------------
    /// Cache directory for models and intermediate data
    #[arg(long = "cache-dir")]
    pub cache_dir: Option<PathBuf>,

    /// Enable ZRAM-aware allocation for reduced memory usage
    /// Provides ~48% RAM reduction for batch transcription (515 MB → 267 MB)
    #[arg(long = "zram-optimized")]
    pub zram_optimized: bool,
}

/// Arguments for transcribe-folder command (WAPR-PERF-004)
///
/// Structure-preserving batch transcription with brick profiling integration.
/// Per spec §1.3 (docs/specifications/transcribe-folder-spec.md):
/// - Structure Mirroring: `./raw/sub/b.mp3` → `./trans/sub/b.json`
/// - Atomic writes: Write to `${filename}.tmp` then rename
/// - Resumable: Skip existing files
/// - Deterministic: Sorted file list for reproducible parallel processing
///
/// # Example
///
/// ```bash
/// whisper-apr-cli transcribe-folder \
///     --input-dir ./raw_audio \
///     --output-dir ./transcripts \
///     --format json \
///     --recursive \
///     --workers 4 \
///     --profile
/// ```
#[derive(Parser, Debug, Clone)]
pub struct TranscribeFolderArgs {
    /// Input directory containing audio files
    #[arg(long = "input-dir", short = 'i')]
    pub input_dir: PathBuf,

    /// Output directory for transcriptions (structure mirrors input)
    #[arg(long = "output-dir", short = 'o')]
    pub output_dir: PathBuf,

    /// Output format for transcription files
    #[arg(long, short = 'f', default_value = "json")]
    pub format: OutputFormatArg,

    /// Process subdirectories recursively
    #[arg(long, short = 'r')]
    pub recursive: bool,

    /// Number of parallel workers (default: number of CPU cores)
    #[arg(long, short = 'w')]
    pub workers: Option<usize>,

    /// Model size to use
    #[arg(long, short = 'm', default_value = "tiny")]
    pub model: ModelSize,

    /// Path to .apr model file (overrides --model)
    #[arg(long)]
    pub model_path: Option<PathBuf>,

    /// Source language (ISO 639-1) or 'auto' for detection
    #[arg(long, short = 'l', default_value = "auto")]
    pub language: String,

    /// Skip files that already have transcriptions
    #[arg(long)]
    pub skip_existing: bool,

    // -------------------------------------------------------------------------
    // Brick Profiling Integration (§2.3 of spec)
    // -------------------------------------------------------------------------
    /// Enable brick profiling (per-stage timing breakdown)
    ///
    /// When enabled, each output file includes profiling metadata:
    /// audio_ms, encoder_ms, decoder_ms, tokens_per_sec, budget_met
    #[arg(long)]
    pub profile: bool,

    /// Strict budget mode: exit with error if any file exceeds budget
    ///
    /// Jidoka (自働化) principle: Stop the line on defect.
    /// Budget is 130 µs/token = 7,692 tok/s.
    #[arg(long)]
    pub strict_budget: bool,

    /// Enable anomaly detection during inference
    ///
    /// Checks for NaN, explosion (>1e10), vanishing gradients (<1e-10)
    /// in layer activations. Logs warnings on detection.
    #[arg(long)]
    pub trace_anomalies: bool,

    /// Output file for aggregate profiling report (JSON)
    #[arg(long)]
    pub report: Option<PathBuf>,

    // -------------------------------------------------------------------------
    // GPU/Hardware options
    // -------------------------------------------------------------------------
    /// Use GPU acceleration
    #[arg(long)]
    pub gpu: bool,

    /// Number of CPU threads (default: auto)
    #[arg(long, short = 't')]
    pub threads: Option<u32>,

    // -------------------------------------------------------------------------
    // ZRAM optimization (GitHub #8)
    // -------------------------------------------------------------------------
    /// Cache directory for models and intermediate data
    #[arg(long = "cache-dir")]
    pub cache_dir: Option<PathBuf>,

    /// Enable ZRAM-aware allocation for reduced memory usage
    #[arg(long = "zram-optimized")]
    pub zram_optimized: bool,
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

    /// Check WASM viability for LFM2 deployment
    WasmCheck {
        /// Model family (lfm2, llama, whisper)
        #[arg(short = 'm', long, default_value = "lfm2")]
        family: String,

        /// Quantization type (fp16, int8, int4-awq, int4-gptq)
        #[arg(short = 'Q', long, default_value = "int4-awq")]
        quantization: String,

        /// Maximum context length
        #[arg(short, long, default_value = "4096")]
        context: usize,

        /// Sliding window size (0 for full attention)
        #[arg(short = 'w', long, default_value = "2048")]
        sliding_window: usize,
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

    /// Benchmark LFM2 components instead of Whisper
    #[arg(long)]
    pub lfm2: bool,

    /// LFM2 component to benchmark (gqa, swiglu, rope, conv1d, full_layer, all)
    #[arg(long, default_value = "all")]
    pub component: String,

    /// Sequence length for LFM2 benchmarks
    #[arg(long, default_value = "128")]
    pub seq_len: usize,

    /// Use LFM2-2.6B config (larger, slower) vs small test config
    #[arg(long)]
    pub full_size: bool,
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

/// Arguments for diagnose command
///
/// Self-diagnostic checks for whisper.apr configuration and known issues.
/// Validates tokenizer settings, model compatibility, and common pitfalls.
#[derive(Parser, Debug, Clone)]
pub struct DiagnoseArgs {
    /// Optional APR model file to check
    #[arg(short, long)]
    pub model: Option<PathBuf>,

    /// Check only tokenizer configuration
    #[arg(long)]
    pub tokenizer_only: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Run all checks including slow ones
    #[arg(long)]
    pub full: bool,
}

/// Arguments for selftest command
///
/// Runs a multi-phase self-test: diagnose, backend test, and optional transcription.
/// Use after `cargo install whisper-apr --features cli` to verify the installation.
#[derive(Parser, Debug, Clone)]
pub struct SelftestArgs {
    /// Path to .apr model file for transcription test
    #[arg(long)]
    pub model: Option<PathBuf>,

    /// Path to audio file for transcription test
    #[arg(long)]
    pub audio: Option<PathBuf>,

    /// Expected substring in transcription output
    #[arg(long)]
    pub expect: Option<String>,
}

/// Arguments for convert command (WAPR-LFM2-004)
///
/// Converts HuggingFace safetensors models to APR2 format.
#[derive(Parser, Debug, Clone)]
pub struct ConvertArgs {
    /// Input safetensors file or HuggingFace model ID
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output APR2 file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Model family (lfm2, llama, whisper)
    #[arg(long, default_value = "lfm2")]
    pub family: ModelFamilyArg,

    /// Quantization method
    #[arg(short = 'Q', long, default_value = "f32")]
    pub quantize: QuantizeMethodArg,

    /// Group size for quantization
    #[arg(long, default_value = "128")]
    pub group_size: u32,

    /// Verbose output (show tensor names)
    #[arg(short, long)]
    pub verbose: bool,

    /// Dry run (show what would be converted)
    #[arg(long)]
    pub dry_run: bool,
}

/// Arguments for export command (WAPR-PUB-001)
///
/// Exports APR models to SafeTensors format for HuggingFace Hub publishing.
#[derive(Parser, Debug, Clone)]
pub struct ExportArgs {
    /// Input APR model file
    pub input: PathBuf,

    /// Output SafeTensors file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Output format
    #[arg(long, default_value = "safetensors")]
    pub format: ExportFormatArg,

    /// Include metadata in output
    #[arg(long)]
    pub with_metadata: bool,

    /// Verbose output (show tensor names)
    #[arg(short, long)]
    pub verbose: bool,
}

/// Export format options
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportFormatArg {
    /// SafeTensors format (HuggingFace standard)
    #[default]
    Safetensors,
    /// GGML format (whisper.cpp compatible)
    Ggml,
}

impl std::fmt::Display for ExportFormatArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Safetensors => write!(f, "safetensors"),
            Self::Ggml => write!(f, "ggml"),
        }
    }
}

/// Arguments for publish command (WAPR-PUB-004)
///
/// Complete publishing workflow to HuggingFace Hub.
/// Generates model card, exports SafeTensors, uploads all files.
///
/// # Example
///
/// ```bash
/// # Publish whisper-tiny to HuggingFace
/// whisper-apr-cli publish \
///     --input models/whisper-tiny.apr \
///     --repo paiml/whisper-apr-tiny \
///     --model-name "Whisper APR Tiny" \
///     --model-size tiny
///
/// # Dry run (preview without uploading)
/// whisper-apr-cli publish \
///     --input models/whisper-tiny.apr \
///     --repo paiml/whisper-apr-tiny \
///     --dry-run
/// ```
#[derive(Parser, Debug, Clone)]
pub struct PublishArgs {
    /// Input APR model file
    #[arg(short, long)]
    pub input: PathBuf,

    /// HuggingFace repository ID (e.g., paiml/whisper-apr-tiny)
    #[arg(short, long)]
    pub repo: String,

    /// Model display name for the model card
    #[arg(long, default_value = "Whisper APR")]
    pub model_name: String,

    /// Model size (tiny, base, small, medium, large)
    #[arg(long, default_value = "tiny")]
    pub model_size: ModelSize,

    /// Publishing format
    #[arg(long, default_value = "both")]
    pub format: PublishFormatArg,

    /// Commit message
    #[arg(long, default_value = "Upload via whisper-apr publish")]
    pub message: String,

    /// Dry run (preview what would be uploaded)
    #[arg(long)]
    pub dry_run: bool,

    /// Skip model verification
    #[arg(long)]
    pub skip_verify: bool,

    /// Custom license (default: mit)
    #[arg(long, default_value = "mit")]
    pub license: String,

    /// Custom model card content (README.md path)
    #[arg(long)]
    pub model_card: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

/// Publishing format options
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PublishFormatArg {
    /// APR format only
    Apr,
    /// SafeTensors format only
    Safetensors,
    /// Both formats (default)
    #[default]
    Both,
}

impl std::fmt::Display for PublishFormatArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Apr => write!(f, "apr"),
            Self::Safetensors => write!(f, "safetensors"),
            Self::Both => write!(f, "both"),
        }
    }
}

/// Model family for conversion
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamilyArg {
    /// LFM2 (LiquidAI transcript summarization)
    Lfm2,
    /// LLaMA-style models
    Llama,
    /// Whisper ASR models
    Whisper,
}

impl std::fmt::Display for ModelFamilyArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lfm2 => write!(f, "lfm2"),
            Self::Llama => write!(f, "llama"),
            Self::Whisper => write!(f, "whisper"),
        }
    }
}

/// Quantization method for conversion
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeMethodArg {
    /// Full precision (4 bytes per weight)
    F32,
    /// Half precision (2 bytes per weight)
    F16,
    /// BFloat16 (2 bytes per weight)
    Bf16,
    /// 8-bit quantization (1 byte per weight)
    Int8,
    /// 4-bit quantization (0.5 bytes per weight)
    Int4,
    /// 4-bit AWQ quantization
    Int4Awq,
    /// 4-bit GPTQ quantization
    Int4Gptq,
}

impl std::fmt::Display for QuantizeMethodArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::Bf16 => write!(f, "bf16"),
            Self::Int8 => write!(f, "int8"),
            Self::Int4 => write!(f, "int4"),
            Self::Int4Awq => write!(f, "int4-awq"),
            Self::Int4Gptq => write!(f, "int4-gptq"),
        }
    }
}

/// Model size options
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    /// Whisper tiny model (39M params)
    Tiny,
    /// Whisper base model (74M params)
    Base,
    /// Whisper small model (244M params)
    Small,
    /// Whisper medium model (769M params)
    Medium,
    /// Whisper large model (1.5B params)
    Large,
    /// Whisper large v3 turbo model (809M params, 32 enc + 4 dec layers)
    #[value(name = "large-v3-turbo")]
    LargeV3Turbo,
    /// Moonshine tiny model (27M params, faster for short audio)
    MoonshineTiny,
    /// Moonshine base model (62M params, faster for short audio)
    MoonshineBase,
}

impl ModelSize {
    /// Returns true if this is a Moonshine model
    #[must_use]
    pub fn is_moonshine(&self) -> bool {
        matches!(self, Self::MoonshineTiny | Self::MoonshineBase)
    }
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tiny => write!(f, "tiny"),
            Self::Base => write!(f, "base"),
            Self::Small => write!(f, "small"),
            Self::Medium => write!(f, "medium"),
            Self::Large => write!(f, "large"),
            Self::LargeV3Turbo => write!(f, "large-v3-turbo"),
            Self::MoonshineTiny => write!(f, "moonshine-tiny"),
            Self::MoonshineBase => write!(f, "moonshine-base"),
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
        assert_eq!(ModelSize::LargeV3Turbo.to_string(), "large-v3-turbo");
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

    // -------------------------------------------------------------------------
    // Summarize command tests (WAPR-LFM2-001)
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_summarize_minimal() {
        let args = Args::try_parse_from(["whisper-apr", "summarize", "-f", "transcript.txt"]);
        assert!(args.is_ok(), "Should parse minimal summarize command");
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Summarize(s) => {
                assert_eq!(s.input, Some(PathBuf::from("transcript.txt")));
                assert_eq!(s.format, SummarizeFormat::Json);
                assert_eq!(s.max_tokens, 1024);
                assert!(!s.stream);
            }
            _ => panic!("Expected Summarize command"),
        }
    }

    #[test]
    fn test_parse_summarize_all_options() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "summarize",
            "-f",
            "meeting.txt",
            "--model-path",
            "./lfm2.apr2",
            "-o",
            "summary.json",
            "--format",
            "markdown",
            "--max-tokens",
            "2048",
            "--temperature",
            "0.5",
            "--max-context",
            "8192",
            "--webgpu",
            "--stream",
            "--action-items",
            "--key-points",
        ]);
        assert!(args.is_ok(), "Should parse all summarize options");
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Summarize(s) => {
                assert_eq!(s.model_path, Some(PathBuf::from("./lfm2.apr2")));
                assert_eq!(s.output, Some(PathBuf::from("summary.json")));
                assert_eq!(s.format, SummarizeFormat::Markdown);
                assert_eq!(s.max_tokens, 2048);
                assert!((s.temperature - 0.5).abs() < 0.01);
                assert_eq!(s.max_context, 8192);
                assert!(s.webgpu);
                assert!(s.stream);
                assert!(s.action_items);
                assert!(s.key_points);
            }
            _ => panic!("Expected Summarize command"),
        }
    }

    #[test]
    fn test_parse_summarize_stdin() {
        // Summarize without input file (reads from stdin)
        let args = Args::try_parse_from(["whisper-apr", "summarize"]);
        assert!(args.is_ok(), "Should parse summarize without input (stdin)");
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Summarize(s) => {
                assert!(s.input.is_none());
            }
            _ => panic!("Expected Summarize command"),
        }
    }

    #[test]
    fn test_parse_summarize_with_prompt() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "summarize",
            "-f",
            "transcript.txt",
            "--prompt",
            "Summarize this meeting in 3 bullet points:",
        ]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Summarize(s) => {
                assert_eq!(
                    s.prompt,
                    Some("Summarize this meeting in 3 bullet points:".to_string())
                );
            }
            _ => panic!("Expected Summarize command"),
        }
    }

    // -------------------------------------------------------------------------
    // ZRAM optimization tests (GitHub #8)
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_transcribe_with_zram() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "transcribe",
            "-f",
            "test.wav",
            "--cache-dir",
            "/mnt/whisper-cache",
            "--zram-optimized",
        ]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Transcribe(t) => {
                assert_eq!(t.cache_dir, Some(PathBuf::from("/mnt/whisper-cache")));
                assert!(t.zram_optimized);
            }
            _ => panic!("Expected Transcribe command"),
        }
    }

    #[test]
    fn test_parse_batch_with_zram() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "batch",
            "*.wav",
            "--cache-dir",
            "/mnt/whisper-cache",
            "--zram-optimized",
        ]);
        assert!(args.is_ok());
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Batch(b) => {
                assert_eq!(b.cache_dir, Some(PathBuf::from("/mnt/whisper-cache")));
                assert!(b.zram_optimized);
            }
            _ => panic!("Expected Batch command"),
        }
    }

    // -------------------------------------------------------------------------
    // TranscribeFolder tests (WAPR-PERF-004)
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_transcribe_folder_minimal() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "transcribe-folder",
            "--input-dir",
            "./audio",
            "--output-dir",
            "./transcripts",
        ]);
        assert!(args.is_ok(), "Should parse minimal transcribe-folder");
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::TranscribeFolder(tf) => {
                assert_eq!(tf.input_dir, PathBuf::from("./audio"));
                assert_eq!(tf.output_dir, PathBuf::from("./transcripts"));
                assert_eq!(tf.format, OutputFormatArg::Json);
                assert_eq!(tf.model, ModelSize::Tiny);
                assert!(!tf.recursive);
                assert!(!tf.profile);
            }
            _ => panic!("Expected TranscribeFolder command"),
        }
    }

    #[test]
    fn test_parse_transcribe_folder_alias() {
        // Test the 'folder' alias
        let args = Args::try_parse_from([
            "whisper-apr",
            "folder",
            "-i",
            "./audio",
            "-o",
            "./transcripts",
        ]);
        assert!(args.is_ok(), "Should parse 'folder' alias");
        let args = args.expect("test parse should succeed");
        assert!(matches!(args.command, Command::TranscribeFolder(_)));
    }

    #[test]
    fn test_parse_transcribe_folder_all_options() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "transcribe-folder",
            "--input-dir",
            "./raw_audio",
            "--output-dir",
            "./trans",
            "--format",
            "json",
            "--recursive",
            "--workers",
            "4",
            "--model",
            "base",
            "--language",
            "en",
            "--skip-existing",
            "--profile",
            "--strict-budget",
            "--trace-anomalies",
            "--report",
            "profile-report.json",
            "--gpu",
            "--threads",
            "8",
            "--zram-optimized",
        ]);
        assert!(args.is_ok(), "Should parse all transcribe-folder options");
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::TranscribeFolder(tf) => {
                assert_eq!(tf.input_dir, PathBuf::from("./raw_audio"));
                assert_eq!(tf.output_dir, PathBuf::from("./trans"));
                assert_eq!(tf.format, OutputFormatArg::Json);
                assert!(tf.recursive);
                assert_eq!(tf.workers, Some(4));
                assert_eq!(tf.model, ModelSize::Base);
                assert_eq!(tf.language, "en");
                assert!(tf.skip_existing);
                assert!(tf.profile);
                assert!(tf.strict_budget);
                assert!(tf.trace_anomalies);
                assert_eq!(tf.report, Some(PathBuf::from("profile-report.json")));
                assert!(tf.gpu);
                assert_eq!(tf.threads, Some(8));
                assert!(tf.zram_optimized);
            }
            _ => panic!("Expected TranscribeFolder command"),
        }
    }

    #[test]
    fn test_parse_transcribe_folder_short_flags() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "transcribe-folder",
            "-i",
            "./audio",
            "-o",
            "./out",
            "-f",
            "txt",
            "-r",
            "-w",
            "2",
            "-m",
            "small",
            "-l",
            "fr",
            "-t",
            "4",
        ]);
        assert!(args.is_ok(), "Should parse short flags");
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::TranscribeFolder(tf) => {
                assert_eq!(tf.input_dir, PathBuf::from("./audio"));
                assert_eq!(tf.output_dir, PathBuf::from("./out"));
                assert_eq!(tf.format, OutputFormatArg::Txt);
                assert!(tf.recursive);
                assert_eq!(tf.workers, Some(2));
                assert_eq!(tf.model, ModelSize::Small);
                assert_eq!(tf.language, "fr");
                assert_eq!(tf.threads, Some(4));
            }
            _ => panic!("Expected TranscribeFolder command"),
        }
    }

    #[test]
    fn test_parse_transcribe_folder_missing_input_dir() {
        let args =
            Args::try_parse_from(["whisper-apr", "transcribe-folder", "--output-dir", "./out"]);
        assert!(args.is_err(), "Should fail without --input-dir");
    }

    #[test]
    fn test_parse_transcribe_folder_missing_output_dir() {
        let args =
            Args::try_parse_from(["whisper-apr", "transcribe-folder", "--input-dir", "./audio"]);
        assert!(args.is_err(), "Should fail without --output-dir");
    }

    // -------------------------------------------------------------------------
    // Selftest command tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_selftest_minimal() {
        let args = Args::try_parse_from(["whisper-apr", "selftest"]);
        assert!(args.is_ok(), "Should parse minimal selftest command");
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Selftest(s) => {
                assert!(s.model.is_none());
                assert!(s.audio.is_none());
                assert!(s.expect.is_none());
            }
            _ => panic!("Expected Selftest command"),
        }
    }

    #[test]
    fn test_parse_selftest_full_args() {
        let args = Args::try_parse_from([
            "whisper-apr",
            "selftest",
            "--model",
            "tiny.apr",
            "--audio",
            "test.wav",
            "--expect",
            "birds",
        ]);
        assert!(args.is_ok(), "Should parse selftest with all args");
        let args = args.expect("test parse should succeed");
        match args.command {
            Command::Selftest(s) => {
                assert_eq!(s.model, Some(PathBuf::from("tiny.apr")));
                assert_eq!(s.audio, Some(PathBuf::from("test.wav")));
                assert_eq!(s.expect, Some("birds".to_string()));
            }
            _ => panic!("Expected Selftest command"),
        }
    }
}

//! APR model inspection and format conversion CLI arguments
//!
//! Integrates aprender's format library into whisper-apr-cli under
//! the `apr` subcommand group.

use std::path::PathBuf;

use clap::{Args, Subcommand};

/// APR model format tools (inspect, lint, convert, diff)
#[derive(Args, Debug, Clone)]
pub struct AprArgs {
    /// APR subcommand
    #[command(subcommand)]
    pub action: AprAction,
}

/// APR subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum AprAction {
    /// Inspect model metadata and structure
    Inspect(AprInspectArgs),

    /// List tensor names, shapes, and statistics
    Tensors(AprTensorsArgs),

    /// Hex dump of model binary data
    Hex(AprHexArgs),

    /// Model architecture tree view
    Tree(AprTreeArgs),

    /// Data flow visualization
    Flow(AprFlowArgs),

    /// Lint model for best practices
    Lint(AprLintArgs),

    /// Compare two model files
    Diff(AprDiffArgs),

    /// Import GGUF/`SafeTensors`/`HuggingFace` to APR format
    Import(AprImportArgs),

    /// Merge multiple models
    Merge(AprMergeArgs),

    /// Rosetta Stone format-aware operations
    Rosetta(RosettaArgs),

    /// Create regression canary baseline
    Canary(AprCanaryArgs),

    /// Verify golden trace logit fingerprints
    Golden(AprGoldenArgs),

    /// Validate tensor contracts (Poka-Yoke)
    Validate(AprValidateArgs),

    /// Verify layout contracts (GGUF→APR transpose)
    Contract(AprContractArgs),

    /// Model family detection and checking
    Family(AprFamilyArgs),

    /// Statistical weight comparison between models
    Compare(AprCompareArgs),

    /// Export APR model to GGUF/SafeTensors
    Export(AprExportArgs),

    /// Audit F16 scale factors for NaN/Inf/subnormal
    F16Audit(AprF16AuditArgs),

    // ── Phase 3: Tier B — Feature-Gated ──
    /// Sign model with Ed25519 key (feature: `format-signing`)
    Sign(AprSignArgs),

    /// Verify Ed25519 signature (feature: `format-signing`)
    VerifySig(AprVerifySigArgs),

    /// Encrypt model with AES-256-GCM (feature: `format-encryption`)
    Encrypt(AprEncryptArgs),

    /// Decrypt AES-256-GCM encrypted model (feature: `format-encryption`)
    Decrypt(AprDecryptArgs),

    /// Quantize model to `Q4_0`/`Q8_0` (feature: `format-quantize`)
    Quantize(AprQuantizeArgs),

    /// Import multi-shard model with streaming
    ImportSharded(AprImportShardedArgs),

    /// Inspect homomorphic encryption metadata (feature: `format-homomorphic`)
    HeInspect(AprHeInspectArgs),

    /// Profile transcription with per-step timing breakdown (renacer integration)
    Profile(AprProfileArgs),

    /// Probe activation tensors through the forward pass for parity debugging
    Probe(AprProbeArgs),

    /// Compare probed activations against a reference for numerical parity
    Parity(AprParityArgs),

    /// Check model configuration against known reference configs
    ConfigCheck(AprConfigCheckArgs),

    /// Pull a model from HuggingFace (proxied from aprender's `apr` CLI)
    Pull(AprPullArgs),

    /// List cached models (proxied from aprender's `apr` CLI)
    #[command(name = "ls")]
    PullList(AprPullListArgs),
}

// ============================================================================
// Tier 1 — Inspection
// ============================================================================

/// Arguments for `apr inspect`
#[derive(Args, Debug, Clone)]
pub struct AprInspectArgs {
    /// Model file (APR, GGUF, or `SafeTensors`)
    pub file: PathBuf,
}

/// Arguments for `apr tensors`
#[derive(Args, Debug, Clone)]
pub struct AprTensorsArgs {
    /// Model file (APR, GGUF, or `SafeTensors`)
    pub file: PathBuf,

    /// Compute per-tensor statistics (mean, std, min, max)
    #[arg(long)]
    pub stats: bool,

    /// Filter tensors by name pattern
    #[arg(long)]
    pub filter: Option<String>,

    /// Maximum number of tensors to show
    #[arg(long, default_value = "0")]
    pub limit: usize,
}

/// Arguments for `apr hex`
#[derive(Args, Debug, Clone)]
pub struct AprHexArgs {
    /// Model file to hex dump
    pub file: PathBuf,

    /// Maximum bytes to dump
    #[arg(long, default_value = "256")]
    pub limit: usize,

    /// Specific tensor name to dump
    #[arg(long)]
    pub tensor: Option<String>,
}

/// Arguments for `apr tree`
#[derive(Args, Debug, Clone)]
pub struct AprTreeArgs {
    /// Model file
    pub file: PathBuf,

    /// Show tensor sizes
    #[arg(long)]
    pub sizes: bool,

    /// Maximum tree depth
    #[arg(long)]
    pub depth: Option<usize>,
}

/// Arguments for `apr flow`
#[derive(Args, Debug, Clone)]
pub struct AprFlowArgs {
    /// Model file
    pub file: PathBuf,

    /// Layer number to visualize (default: all)
    #[arg(long)]
    pub layer: Option<usize>,
}

/// Arguments for `apr lint`
#[derive(Args, Debug, Clone)]
pub struct AprLintArgs {
    /// Model file to lint
    pub file: PathBuf,
}

/// Arguments for `apr diff`
#[derive(Args, Debug, Clone)]
pub struct AprDiffArgs {
    /// First model file
    pub file1: PathBuf,

    /// Second model file
    pub file2: PathBuf,

    /// Filter tensors by name pattern
    #[arg(long)]
    pub filter: Option<String>,
}

// ============================================================================
// Tier 2 — Format Conversion
// ============================================================================

/// Arguments for `apr import`
#[derive(Args, Debug, Clone)]
pub struct AprImportArgs {
    /// Source (file path or `HuggingFace` repo: `org/repo`)
    pub source: String,

    /// Output .apr file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Model architecture (llama, phi, qwen, etc.)
    #[arg(long)]
    pub arch: Option<String>,

    /// Quantization type (`q4_0`, `q8_0`, etc.)
    #[arg(long)]
    pub quantize: Option<String>,
}

/// Arguments for `apr merge`
#[derive(Args, Debug, Clone)]
pub struct AprMergeArgs {
    /// Model files to merge (2 or more)
    pub files: Vec<PathBuf>,

    /// Output merged model
    #[arg(short, long)]
    pub output: PathBuf,

    /// Merge strategy (average, weighted, ties, dare, slerp)
    #[arg(long, default_value = "average")]
    pub strategy: String,

    /// Weights for weighted merge (comma-separated)
    #[arg(long)]
    pub weights: Option<String>,
}

// ============================================================================
// Tier 3 — Rosetta (universal format operations)
// ============================================================================

/// Rosetta Stone format-aware operations
#[derive(Args, Debug, Clone)]
pub struct RosettaArgs {
    /// Rosetta subcommand
    #[command(subcommand)]
    pub action: RosettaAction,
}

/// Rosetta subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum RosettaAction {
    /// Format-aware model inspection
    Inspect(RosettaInspectArgs),

    /// Convert between model formats
    Convert(RosettaConvertArgs),

    /// Round-trip verification
    Verify(RosettaVerifyArgs),

    /// Format-aware tensor diff (detects layout bugs)
    Diff(RosettaDiffArgs),

    /// Per-tensor statistical fingerprint
    Fingerprint(RosettaFingerprintArgs),
}

/// Arguments for `apr rosetta inspect`
#[derive(Args, Debug, Clone)]
pub struct RosettaInspectArgs {
    /// Model file
    pub file: PathBuf,
}

/// Arguments for `apr rosetta convert`
#[derive(Args, Debug, Clone)]
pub struct RosettaConvertArgs {
    /// Source model file
    pub source: PathBuf,

    /// Destination model file
    pub dest: PathBuf,

    /// Apply quantization during conversion
    #[arg(long)]
    pub quantize: bool,

    /// Verify conversion with round-trip check
    #[arg(long)]
    pub verify: bool,
}

/// Arguments for `apr rosetta verify`
#[derive(Args, Debug, Clone)]
pub struct RosettaVerifyArgs {
    /// Model file to verify
    pub file: PathBuf,

    /// Tolerance for floating-point comparison
    #[arg(long, default_value = "1e-5")]
    pub tolerance: f32,
}

/// Arguments for `apr rosetta diff`
#[derive(Args, Debug, Clone)]
pub struct RosettaDiffArgs {
    /// First model file
    pub file1: PathBuf,

    /// Second model file
    pub file2: PathBuf,
}

/// Arguments for `apr rosetta fingerprint`
#[derive(Args, Debug, Clone)]
pub struct RosettaFingerprintArgs {
    /// Model file
    pub file: PathBuf,
}

// ============================================================================
// Tier 4 — Canary
// ============================================================================

/// Arguments for `apr canary`
#[derive(Args, Debug, Clone)]
pub struct AprCanaryArgs {
    /// Model file to create canary from
    pub file: PathBuf,

    /// Output canary file
    #[arg(short, long)]
    pub output: PathBuf,
}

// ============================================================================
// Tier A — Phase 2 Commands
// ============================================================================

/// Arguments for `apr golden`
#[derive(Args, Debug, Clone)]
pub struct AprGoldenArgs {
    /// Golden trace JSON file to verify against
    pub trace_file: PathBuf,

    /// Actual logits file (binary f32 or JSON)
    #[arg(long)]
    pub logits: Option<PathBuf>,

    /// Override tolerance (default: 1e-4)
    #[arg(long)]
    pub tolerance: Option<f32>,
}

/// Arguments for `apr validate`
#[derive(Args, Debug, Clone)]
pub struct AprValidateArgs {
    /// Model file to validate
    pub file: PathBuf,

    /// Vocabulary size (for embedding validation)
    #[arg(long)]
    pub vocab_size: Option<usize>,

    /// Hidden dimension (for embedding validation)
    #[arg(long)]
    pub hidden_dim: Option<usize>,
}

/// Arguments for `apr contract`
#[derive(Args, Debug, Clone)]
pub struct AprContractArgs {
    /// Model file to check contracts against
    pub file: PathBuf,

    /// Specific tensor name to verify
    #[arg(long)]
    pub tensor: Option<String>,
}

/// Arguments for `apr family`
#[derive(Args, Debug, Clone)]
pub struct AprFamilyArgs {
    /// Family subcommand
    #[command(subcommand)]
    pub action: FamilyAction,
}

/// Family subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum FamilyAction {
    /// Identify model family from tensor names
    Identify(AprFamilyIdentifyArgs),

    /// Check model against a specific family contract
    Check(AprFamilyCheckArgs),
}

/// Arguments for `apr family identify`
#[derive(Args, Debug, Clone)]
pub struct AprFamilyIdentifyArgs {
    /// Model file to identify
    pub file: PathBuf,
}

/// Arguments for `apr family check`
#[derive(Args, Debug, Clone)]
pub struct AprFamilyCheckArgs {
    /// Model file to check
    pub file: PathBuf,

    /// Expected family name (llama, qwen2, whisper, etc.)
    pub family: String,

    /// Expected size variant (0.5b, 7b, etc.)
    #[arg(long)]
    pub size: Option<String>,
}

/// Arguments for `apr compare`
#[derive(Args, Debug, Clone)]
pub struct AprCompareArgs {
    /// Source model file
    pub source: PathBuf,

    /// Target model file
    pub target: PathBuf,

    /// L2 tolerance (default: 1e-5)
    #[arg(long, default_value = "1e-5")]
    pub l2_tolerance: f64,

    /// Max element-wise tolerance (default: 1e-5)
    #[arg(long, default_value = "1e-5")]
    pub max_tolerance: f64,
}

/// Arguments for `apr export`
#[derive(Args, Debug, Clone)]
pub struct AprExportArgs {
    /// Input model file (APR or `SafeTensors`)
    pub input: PathBuf,

    /// Output file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Target format (safetensors, gguf)
    #[arg(long, default_value = "safetensors")]
    pub format: String,
}

/// Arguments for `apr f16-audit`
#[derive(Args, Debug, Clone)]
pub struct AprF16AuditArgs {
    /// Model file to audit
    pub file: PathBuf,

    /// Show per-tensor details
    #[arg(long)]
    pub verbose: bool,
}

// ============================================================================
// Tier B — Phase 3: Feature-Gated Commands
// ============================================================================

/// Arguments for `apr sign`
#[derive(Args, Debug, Clone)]
pub struct AprSignArgs {
    /// Model file to sign
    pub file: PathBuf,

    /// Ed25519 private key file (PEM or raw 32 bytes)
    #[arg(long)]
    pub key: PathBuf,

    /// Output signed model file
    #[arg(short, long)]
    pub output: PathBuf,
}

/// Arguments for `apr verify-sig`
#[derive(Args, Debug, Clone)]
pub struct AprVerifySigArgs {
    /// Signed model file to verify
    pub file: PathBuf,

    /// Ed25519 public key file (PEM or raw 32 bytes)
    #[arg(long)]
    pub pubkey: Option<PathBuf>,
}

/// Arguments for `apr encrypt`
#[derive(Args, Debug, Clone)]
pub struct AprEncryptArgs {
    /// Model file to encrypt
    pub file: PathBuf,

    /// Output encrypted model file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Encryption password (read from stdin if not provided)
    #[arg(long)]
    pub password: Option<String>,
}

/// Arguments for `apr decrypt`
#[derive(Args, Debug, Clone)]
pub struct AprDecryptArgs {
    /// Encrypted model file
    pub file: PathBuf,

    /// Output decrypted model file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Decryption password (read from stdin if not provided)
    #[arg(long)]
    pub password: Option<String>,
}

/// Arguments for `apr quantize`
#[derive(Args, Debug, Clone)]
pub struct AprQuantizeArgs {
    /// Model file to quantize
    pub file: PathBuf,

    /// Output quantized model file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Quantization type (`q4_0` or `q8_0`)
    #[arg(long, default_value = "q8_0")]
    pub r#type: String,

    /// Verify roundtrip accuracy after quantization
    #[arg(long)]
    pub verify: bool,
}

/// Arguments for `apr import-sharded`
#[derive(Args, Debug, Clone)]
pub struct AprImportShardedArgs {
    /// Directory containing sharded model files
    pub source: PathBuf,

    /// Output .apr file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Maximum shards to cache in memory (default: 2)
    #[arg(long, default_value = "2")]
    pub max_cache_shards: usize,
}

/// Arguments for `apr he-inspect`
#[derive(Args, Debug, Clone)]
pub struct AprHeInspectArgs {
    /// HE-encrypted model file to inspect
    pub file: PathBuf,
}

// ============================================================================
// Tier C — Profiling (renacer integration)
// ============================================================================

/// Arguments for `apr profile`
///
/// Runs renacer-instrumented transcription with per-step timing breakdown:
/// mel spectrogram, encoder, decoder (per-token), detokenize.
/// Outputs a structured report compatible with `renacer` trace format.
#[derive(Args, Debug, Clone)]
pub struct AprProfileArgs {
    /// Model file (.apr format)
    pub model: PathBuf,

    /// Audio file to transcribe (WAV, MP3, FLAC, etc.)
    pub audio: PathBuf,

    /// Number of warmup runs before measurement
    #[arg(long, default_value = "1")]
    pub warmup: usize,

    /// Number of measurement runs (results averaged)
    #[arg(long, default_value = "3")]
    pub runs: usize,

    /// Output format: text, json, or renacer (trace JSON)
    #[arg(long, default_value = "text")]
    pub format: String,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Show per-token decoder timing
    #[arg(long)]
    pub per_token: bool,

    /// Compare against whisper.cpp timing (if available)
    #[arg(long)]
    pub compare_cpp: bool,
}

// ============================================================================
// Tier D — Forward-Pass Debugging (WAPR-MOONSHINE-013)
// ============================================================================

/// Arguments for `apr probe`
///
/// Runs a probed forward pass, recording activation statistics at each
/// checkpoint in the pipeline (ConvStem → Encoder → Decoder).
#[derive(Args, Debug, Clone)]
pub struct AprProbeArgs {
    /// Model file (.apr format)
    pub model: PathBuf,

    /// Audio file to process (WAV, MP3, FLAC, etc.)
    pub audio: PathBuf,

    /// Output JSON file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Filter to specific pipeline stage (e.g. "conv_stem", "encoder", "decoder")
    #[arg(long)]
    pub stage: Option<String>,

    /// Filter to specific layer (e.g. "encoder.block_0")
    #[arg(long)]
    pub layer: Option<String>,

    /// Capture full tensor data (large output)
    #[arg(long)]
    pub full_tensor: bool,

    /// Number of leading values to show per checkpoint
    #[arg(long, default_value = "8")]
    pub first_n: usize,

    /// Decoder input tokens (comma-separated IDs; default: SOT for Whisper, 1 for Moonshine)
    #[arg(long)]
    pub tokens: Option<String>,
}

/// Arguments for `apr parity`
///
/// Compares two probe JSON outputs checkpoint-by-checkpoint, reporting
/// where activations first diverge beyond tolerance.
#[derive(Args, Debug, Clone)]
pub struct AprParityArgs {
    /// Our probe JSON file
    pub ours: PathBuf,

    /// Reference probe JSON file (e.g. from HuggingFace)
    pub reference: PathBuf,

    /// Relative L2 tolerance (default: 1% = 0.01)
    #[arg(long, default_value = "0.01")]
    pub tolerance: f64,

    /// Absolute tolerance for near-zero L2 values
    #[arg(long, default_value = "1e-5")]
    pub abs_tolerance: f64,

    /// Stop and mark remaining as propagated after first failure
    #[arg(long)]
    pub stop_first: bool,
}

/// Arguments for `apr config-check`
///
/// Validates that a model's configuration parameters match a known reference
/// configuration for its model family and size.
#[derive(Args, Debug, Clone)]
pub struct AprConfigCheckArgs {
    /// Model file (.apr format)
    pub model: PathBuf,

    /// Reference model name (e.g. "moonshine-tiny", "whisper-tiny") or JSON config file
    #[arg(long)]
    pub reference: Option<String>,

    /// Show all checked parameters (not just mismatches)
    #[arg(long)]
    pub verbose: bool,
}

// ============================================================================
// Proxy Commands (delegated to aprender's `apr` CLI)
// ============================================================================

/// Arguments for `apr pull` (proxied from aprender's `apr` CLI)
#[derive(Args, Debug, Clone)]
pub struct AprPullArgs {
    /// Model reference (e.g. `openai/whisper-base`, `hf://org/repo/file.safetensors`)
    pub model_ref: String,

    /// Force re-download even if cached
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `apr ls` (proxied from aprender's `apr` CLI)
#[derive(Args, Debug, Clone)]
pub struct AprPullListArgs {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

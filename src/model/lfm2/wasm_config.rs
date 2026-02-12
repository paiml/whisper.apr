//! WASM Configuration for LFM2 (Section 18.7)
//!
//! Provides configuration types and memory estimation for running
//! LFM2 models in WebAssembly environments.

use crate::format::apr2::Lfm2Config;

/// Quantization type for WASM deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmQuantization {
    /// 16-bit floating point (5.2GB for LFM2-2.6B) - NOT viable for WASM
    Fp16,
    /// 8-bit integer (2.6GB for LFM2-2.6B) - Marginal for WASM
    Int8,
    /// 4-bit integer with AWQ (1.3GB for LFM2-2.6B) - Viable for WASM
    Int4Awq,
    /// 4-bit integer with GPTQ (1.3GB for LFM2-2.6B) - Viable for WASM
    Int4Gptq,
}

impl WasmQuantization {
    /// Bytes per parameter for this quantization type
    #[must_use]
    pub const fn bytes_per_param(&self) -> f32 {
        match self {
            Self::Fp16 => 2.0,
            Self::Int8 => 1.0,
            Self::Int4Awq | Self::Int4Gptq => 0.5,
        }
    }

    /// Whether this quantization is viable for WASM (< 2GB)
    #[must_use]
    pub fn is_wasm_viable(&self, num_params: u64) -> bool {
        let model_bytes = (num_params as f64) * (self.bytes_per_param() as f64);
        model_bytes < 2_000_000_000.0 // 2GB limit
    }
}

impl std::fmt::Display for WasmQuantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp16 => write!(f, "fp16"),
            Self::Int8 => write!(f, "int8"),
            Self::Int4Awq => write!(f, "int4-awq"),
            Self::Int4Gptq => write!(f, "int4-gptq"),
        }
    }
}

/// WASM deployment configuration for LFM2
///
/// Based on Section 18.7 of the specification, this struct defines
/// the recommended configuration for running LFM2 in WebAssembly.
///
/// # Memory Budget
///
/// ```text
/// WASM 32-bit address space limit: 4 GB
/// Browser practical limit:         ~2 GB
///
/// Total (int4 + 4K context):
///   Model:     1.3 GB
///   KV Cache:  1.0 GB
///   Overhead:  0.2 GB
///   ─────────────────
///   Total:     2.5 GB  ⚠️ Tight but possible
/// ```
#[derive(Debug, Clone)]
pub struct Lfm2WasmConfig {
    /// Quantization type (int4 recommended for WASM)
    pub quantization: WasmQuantization,
    /// Maximum context length (limited for memory)
    pub max_context: usize,
    /// Sliding window size for bounded KV cache (None = full attention)
    pub sliding_window: Option<usize>,
    /// Whether to use WebGPU for acceleration
    pub use_webgpu: bool,
    /// Whether to enable token-by-token streaming
    pub streaming: bool,
}

impl Default for Lfm2WasmConfig {
    fn default() -> Self {
        Self {
            quantization: WasmQuantization::Int4Awq,
            max_context: 4096,
            sliding_window: Some(2048),
            use_webgpu: true,
            streaming: true,
        }
    }
}

impl Lfm2WasmConfig {
    /// Create recommended WASM config for LFM2-2.6B
    #[must_use]
    pub fn lfm2_2_6b() -> Self {
        Self::default()
    }

    /// Create config with no sliding window (full attention)
    #[must_use]
    pub fn full_attention() -> Self {
        Self {
            sliding_window: None,
            ..Self::default()
        }
    }

    /// Create conservative config for low-memory devices
    #[must_use]
    pub fn low_memory() -> Self {
        Self {
            quantization: WasmQuantization::Int4Awq,
            max_context: 2048,
            sliding_window: Some(1024),
            use_webgpu: true,
            streaming: true,
        }
    }
}

/// Memory estimation for WASM deployment
///
/// Provides detailed memory breakdown for planning LFM2 deployment
/// in WebAssembly environments.
#[derive(Debug, Clone)]
pub struct WasmMemoryEstimate {
    /// Model weights in bytes
    pub model_bytes: u64,
    /// KV cache in bytes (for max context)
    pub kv_cache_bytes: u64,
    /// Runtime overhead estimate in bytes
    pub overhead_bytes: u64,
    /// Total memory in bytes
    pub total_bytes: u64,
    /// Whether this configuration is viable for WASM
    pub is_viable: bool,
    /// Warning messages (if any)
    pub warnings: Vec<String>,
}

impl WasmMemoryEstimate {
    /// Calculate memory estimate for given config
    #[must_use]
    pub fn calculate(config: &Lfm2Config, wasm_config: &Lfm2WasmConfig) -> Self {
        // LFM2-2.6B has approximately 2.6 billion parameters
        let num_params: u64 = 2_600_000_000;
        let model_bytes =
            (num_params as f64 * wasm_config.quantization.bytes_per_param() as f64) as u64;

        // KV cache calculation
        // Per-token: 2 * num_layers * num_kv_heads * head_dim * 2 bytes (fp16)
        let num_layers = config.num_layers as u64;
        let num_kv_heads = config.num_kv_heads as u64;
        let head_dim = (config.hidden_size / config.num_q_heads) as u64;
        let bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2; // K and V, fp16

        let effective_context = wasm_config
            .sliding_window
            .unwrap_or(wasm_config.max_context) as u64;
        let kv_cache_bytes = bytes_per_token * effective_context;

        // Runtime overhead (WASM runtime, JS heap, etc.)
        let overhead_bytes: u64 = 200_000_000; // ~200MB

        let total_bytes = model_bytes + kv_cache_bytes + overhead_bytes;

        // Check viability
        let browser_limit: u64 = 2_000_000_000; // ~2GB practical limit
        let is_viable = total_bytes < browser_limit;

        let mut warnings = Vec::new();

        if !is_viable {
            warnings.push(format!(
                "Total memory ({:.2} GB) exceeds browser limit (~2 GB)",
                total_bytes as f64 / 1_000_000_000.0
            ));
        }

        if matches!(wasm_config.quantization, WasmQuantization::Fp16) {
            warnings.push("fp16 quantization exceeds WASM memory limits".to_string());
        }

        if wasm_config.max_context > 8192 {
            warnings.push(format!(
                "Large context ({}) may cause OOM in browser",
                wasm_config.max_context
            ));
        }

        if wasm_config.sliding_window.is_none() && wasm_config.max_context > 4096 {
            warnings.push("Full attention with >4K context may exceed memory".to_string());
        }

        Self {
            model_bytes,
            kv_cache_bytes,
            overhead_bytes,
            total_bytes,
            is_viable,
            warnings,
        }
    }

    /// Format as human-readable summary
    #[must_use]
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "Model:    {:>7.2} GB",
            self.model_bytes as f64 / 1_000_000_000.0
        );
        let _ = writeln!(
            s,
            "KV Cache: {:>7.2} GB",
            self.kv_cache_bytes as f64 / 1_000_000_000.0
        );
        let _ = writeln!(
            s,
            "Overhead: {:>7.2} GB",
            self.overhead_bytes as f64 / 1_000_000_000.0
        );
        s.push_str("─────────────────\n");
        let _ = writeln!(
            s,
            "Total:    {:>7.2} GB  {}",
            self.total_bytes as f64 / 1_000_000_000.0,
            if self.is_viable { "✅" } else { "❌" }
        );

        if !self.warnings.is_empty() {
            s.push_str("\nWarnings:\n");
            for w in &self.warnings {
                let _ = writeln!(s, "  ⚠️ {w}");
            }
        }

        s
    }
}

impl std::fmt::Display for WasmMemoryEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

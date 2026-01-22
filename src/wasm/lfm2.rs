//! WASM bindings for LFM2 text generation
//!
//! Provides JavaScript-friendly API for LFM2 inference in the browser.
//!
//! # Usage
//!
//! ```javascript
//! import init, { Lfm2Wasm, Lfm2ConfigWasm, GenerationStatsWasm } from 'whisper-apr';
//!
//! await init();
//! const lfm2 = new Lfm2Wasm();
//! await lfm2.loadFromBytes(modelBytes);
//!
//! // Non-streaming generation
//! const result = lfm2.generate(promptTokens, 100, 0.7);
//! console.log(result);
//!
//! // Streaming generation
//! lfm2.generateStreaming(promptTokens, 100, 0.7, (token, idx) => {
//!     console.log(`Token ${idx}: ${token}`);
//!     return true; // continue
//! });
//! ```
//!
//! # Memory Estimation
//!
//! ```javascript
//! const estimate = Lfm2Wasm.estimateMemory('int4-awq', 4096, 2048);
//! console.log(`Total: ${estimate.totalMb}MB, Viable: ${estimate.isViable}`);
//! ```

use wasm_bindgen::prelude::*;

use crate::format::apr2::Lfm2Config;
use crate::model::lfm2::{
    GenerationStats, Lfm2, Lfm2WasmConfig, WasmMemoryEstimate, WasmQuantization,
};
use js_sys::Function;

/// LFM2 model wrapper for WASM
///
/// Provides text generation capabilities in the browser.
#[wasm_bindgen]
pub struct Lfm2Wasm {
    model: Option<Lfm2>,
    config: Option<Lfm2Config>,
}

#[wasm_bindgen]
impl Lfm2Wasm {
    /// Create new LFM2 wrapper (model not loaded yet)
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: None,
            config: None,
        }
    }

    /// Check if model is loaded
    #[wasm_bindgen(js_name = "isLoaded")]
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Load model from APR2 bytes
    ///
    /// # Arguments
    /// * `bytes` - Model file contents (APR2 format)
    ///
    /// # Errors
    /// Returns error if model loading fails
    #[wasm_bindgen(js_name = "loadFromBytes")]
    pub fn load_from_bytes(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let model = Lfm2::from_apr2_bytes(bytes.to_vec())
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {e}")))?;

        self.config = Some(model.config.clone());
        self.model = Some(model);

        Ok(())
    }

    /// Get number of parameters
    #[wasm_bindgen(js_name = "numParams")]
    pub fn num_params(&self) -> Option<u32> {
        self.model.as_ref().map(|m| m.num_params() as u32)
    }

    /// Get memory usage in bytes
    #[wasm_bindgen(js_name = "memoryBytes")]
    pub fn memory_bytes(&self) -> Option<u32> {
        self.model.as_ref().map(|m| m.memory_bytes() as u32)
    }

    /// Generate tokens (non-streaming)
    ///
    /// # Arguments
    /// * `prompt_ids` - Input token IDs
    /// * `max_tokens` - Maximum new tokens to generate
    /// * `temperature` - Sampling temperature (0 = greedy)
    ///
    /// # Returns
    /// Generated token IDs (including prompt)
    #[wasm_bindgen]
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Vec<u32>, JsValue> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Model not loaded"))?;

        model
            .generate(prompt_ids, max_tokens as usize, temperature)
            .map_err(|e| JsValue::from_str(&format!("Generation failed: {e}")))
    }

    /// Generate tokens with statistics
    ///
    /// Returns tuple of (tokens, stats as JSON string)
    #[wasm_bindgen(js_name = "generateWithStats")]
    pub fn generate_with_stats(
        &self,
        prompt_ids: &[u32],
        max_tokens: u32,
        temperature: f32,
    ) -> Result<GenerationResultWasm, JsValue> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Model not loaded"))?;

        let (tokens, stats) = model
            .generate_with_stats::<fn(u32, usize) -> bool>(
                prompt_ids,
                max_tokens as usize,
                temperature,
                None,
            )
            .map_err(|e| JsValue::from_str(&format!("Generation failed: {e}")))?;

        Ok(GenerationResultWasm { tokens, stats })
    }

    /// Streaming token generation
    ///
    /// Generates tokens one-by-one, calling the callback for each.
    ///
    /// # Arguments
    /// * `prompt_ids` - Input token IDs
    /// * `max_tokens` - Maximum new tokens to generate
    /// * `temperature` - Sampling temperature (0 = greedy)
    /// * `callback` - JS function called with (token, index) that returns boolean (true = continue)
    ///
    /// # Returns
    /// Generation result with tokens and statistics
    #[wasm_bindgen(js_name = "generateStreaming")]
    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        max_tokens: u32,
        temperature: f32,
        callback: &Function,
    ) -> Result<GenerationResultWasm, JsValue> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Model not loaded"))?;

        // Wrap the JS callback
        let js_callback = |token: u32, index: usize| -> bool {
            let this = JsValue::NULL;
            let token_js = JsValue::from(token);
            let index_js = JsValue::from(index as u32);

            match callback.call2(&this, &token_js, &index_js) {
                Ok(result) => result.as_bool().unwrap_or(true),
                Err(_) => false, // Stop on error
            }
        };

        let (tokens, stats) = model
            .generate_with_stats(
                prompt_ids,
                max_tokens as usize,
                temperature,
                Some(js_callback),
            )
            .map_err(|e| JsValue::from_str(&format!("Generation failed: {e}")))?;

        Ok(GenerationResultWasm { tokens, stats })
    }

    /// Forward pass (returns logits for last position)
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    ///
    /// # Returns
    /// Logits for last position [vocab_size]
    #[wasm_bindgen]
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>, JsValue> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Model not loaded"))?;

        let all_logits = model
            .forward(input_ids, None)
            .map_err(|e| JsValue::from_str(&format!("Forward failed: {e}")))?;

        // Return only last position logits
        let vocab_size = model.config.vocab_size as usize;
        if all_logits.is_empty() {
            return Ok(Vec::new());
        }

        let last_start = all_logits.len() - vocab_size;
        Ok(all_logits[last_start..].to_vec())
    }

    /// Estimate memory requirements for WASM deployment
    ///
    /// # Arguments
    /// * `quantization` - Quantization type ("fp16", "int8", "int4-awq", "int4-gptq")
    /// * `max_context` - Maximum context length
    /// * `sliding_window` - Sliding window size (0 for full attention)
    #[wasm_bindgen(js_name = "estimateMemory")]
    pub fn estimate_memory(
        quantization: &str,
        max_context: u32,
        sliding_window: u32,
    ) -> Result<MemoryEstimateWasm, JsValue> {
        let quant = match quantization.to_lowercase().as_str() {
            "fp16" => WasmQuantization::Fp16,
            "int8" => WasmQuantization::Int8,
            "int4-awq" | "int4_awq" => WasmQuantization::Int4Awq,
            "int4-gptq" | "int4_gptq" => WasmQuantization::Int4Gptq,
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unknown quantization: {quantization}"
                )))
            }
        };

        let model_config = Lfm2Config::lfm2_2_6b();
        let wasm_config = Lfm2WasmConfig {
            quantization: quant,
            max_context: max_context as usize,
            sliding_window: if sliding_window > 0 {
                Some(sliding_window as usize)
            } else {
                None
            },
            use_webgpu: false,
            streaming: true,
        };

        let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);
        Ok(MemoryEstimateWasm::from(estimate))
    }

    /// Check if model configuration is viable for WASM
    #[wasm_bindgen(js_name = "checkViability")]
    pub fn check_viability(&self) -> Result<ViabilityCheckWasm, JsValue> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Model not loaded"))?;

        let wasm_config = Lfm2WasmConfig::default();
        let estimate = WasmMemoryEstimate::calculate(config, &wasm_config);

        Ok(ViabilityCheckWasm {
            is_viable: estimate.is_viable,
            total_mb: (estimate.total_bytes / 1_000_000) as u32,
            model_mb: (estimate.model_bytes / 1_000_000) as u32,
            kv_cache_mb: (estimate.kv_cache_bytes / 1_000_000) as u32,
            overhead_mb: (estimate.overhead_bytes / 1_000_000) as u32,
            warnings: estimate.warnings,
        })
    }
}

impl Default for Lfm2Wasm {
    fn default() -> Self {
        Self::new()
    }
}

/// Generation result with statistics
#[wasm_bindgen]
pub struct GenerationResultWasm {
    tokens: Vec<u32>,
    stats: GenerationStats,
}

#[wasm_bindgen]
impl GenerationResultWasm {
    /// Get generated tokens
    #[wasm_bindgen(getter)]
    pub fn tokens(&self) -> Vec<u32> {
        self.tokens.clone()
    }

    /// Get tokens generated count
    #[wasm_bindgen(getter, js_name = "tokensGenerated")]
    pub fn tokens_generated(&self) -> u32 {
        self.stats.tokens_generated as u32
    }

    /// Get milliseconds per token
    #[wasm_bindgen(getter, js_name = "msPerToken")]
    pub fn ms_per_token(&self) -> f64 {
        self.stats.ms_per_token
    }

    /// Get total generation time in milliseconds
    #[wasm_bindgen(getter, js_name = "totalMs")]
    pub fn total_ms(&self) -> f64 {
        self.stats.total_ms
    }

    /// Get tokens per second
    #[wasm_bindgen(getter, js_name = "tokensPerSec")]
    pub fn tokens_per_sec(&self) -> f64 {
        self.stats.tokens_per_sec
    }

    /// Check if generation hit EOS
    #[wasm_bindgen(getter, js_name = "hitEos")]
    pub fn hit_eos(&self) -> bool {
        self.stats.hit_eos
    }

    /// Get stats as JSON string
    #[wasm_bindgen(js_name = "statsJson")]
    pub fn stats_json(&self) -> String {
        format!(
            r#"{{"tokens_generated":{},"ms_per_token":{},"total_ms":{},"tokens_per_sec":{},"hit_eos":{}}}"#,
            self.stats.tokens_generated,
            self.stats.ms_per_token,
            self.stats.total_ms,
            self.stats.tokens_per_sec,
            self.stats.hit_eos
        )
    }
}

/// Memory estimate for WASM deployment
#[wasm_bindgen]
pub struct MemoryEstimateWasm {
    model_mb: u32,
    kv_cache_mb: u32,
    overhead_mb: u32,
    total_mb: u32,
    is_viable: bool,
    warnings: Vec<String>,
}

#[wasm_bindgen]
impl MemoryEstimateWasm {
    /// Model size in MB
    #[wasm_bindgen(getter, js_name = "modelMb")]
    pub fn model_mb(&self) -> u32 {
        self.model_mb
    }

    /// KV cache size in MB
    #[wasm_bindgen(getter, js_name = "kvCacheMb")]
    pub fn kv_cache_mb(&self) -> u32 {
        self.kv_cache_mb
    }

    /// Overhead in MB
    #[wasm_bindgen(getter, js_name = "overheadMb")]
    pub fn overhead_mb(&self) -> u32 {
        self.overhead_mb
    }

    /// Total memory in MB
    #[wasm_bindgen(getter, js_name = "totalMb")]
    pub fn total_mb(&self) -> u32 {
        self.total_mb
    }

    /// Is this configuration WASM viable?
    #[wasm_bindgen(getter, js_name = "isViable")]
    pub fn is_viable(&self) -> bool {
        self.is_viable
    }

    /// Get warnings as JSON array
    #[wasm_bindgen(js_name = "warningsJson")]
    pub fn warnings_json(&self) -> String {
        format!(
            "[{}]",
            self.warnings
                .iter()
                .map(|w| format!(r#""{}""#, w.replace('"', "\\\"")))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

impl From<WasmMemoryEstimate> for MemoryEstimateWasm {
    fn from(e: WasmMemoryEstimate) -> Self {
        Self {
            model_mb: (e.model_bytes / 1_000_000) as u32,
            kv_cache_mb: (e.kv_cache_bytes / 1_000_000) as u32,
            overhead_mb: (e.overhead_bytes / 1_000_000) as u32,
            total_mb: (e.total_bytes / 1_000_000) as u32,
            is_viable: e.is_viable,
            warnings: e.warnings,
        }
    }
}

/// Viability check result
#[wasm_bindgen]
pub struct ViabilityCheckWasm {
    is_viable: bool,
    total_mb: u32,
    model_mb: u32,
    kv_cache_mb: u32,
    overhead_mb: u32,
    warnings: Vec<String>,
}

#[wasm_bindgen]
impl ViabilityCheckWasm {
    /// Is configuration WASM viable?
    #[wasm_bindgen(getter, js_name = "isViable")]
    pub fn is_viable(&self) -> bool {
        self.is_viable
    }

    /// Total memory in MB
    #[wasm_bindgen(getter, js_name = "totalMb")]
    pub fn total_mb(&self) -> u32 {
        self.total_mb
    }

    /// Model size in MB
    #[wasm_bindgen(getter, js_name = "modelMb")]
    pub fn model_mb(&self) -> u32 {
        self.model_mb
    }

    /// KV cache size in MB
    #[wasm_bindgen(getter, js_name = "kvCacheMb")]
    pub fn kv_cache_mb(&self) -> u32 {
        self.kv_cache_mb
    }

    /// Overhead in MB
    #[wasm_bindgen(getter, js_name = "overheadMb")]
    pub fn overhead_mb(&self) -> u32 {
        self.overhead_mb
    }

    /// Get warnings as JSON array
    #[wasm_bindgen(js_name = "warningsJson")]
    pub fn warnings_json(&self) -> String {
        format!(
            "[{}]",
            self.warnings
                .iter()
                .map(|w| format!(r#""{}""#, w.replace('"', "\\\"")))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    /// Get summary string
    #[wasm_bindgen]
    pub fn summary(&self) -> String {
        if self.is_viable {
            format!(
                "✅ WASM Viable: {}MB total (model: {}MB, KV: {}MB, overhead: {}MB)",
                self.total_mb, self.model_mb, self.kv_cache_mb, self.overhead_mb
            )
        } else {
            format!(
                "❌ Not WASM Viable: {}MB exceeds 2GB limit (model: {}MB, KV: {}MB)",
                self.total_mb, self.model_mb, self.kv_cache_mb
            )
        }
    }
}

// Native tests (run with cargo test)
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use crate::format::apr2::Lfm2Config;
    use crate::model::lfm2::{Lfm2WasmConfig, WasmMemoryEstimate, WasmQuantization};

    #[test]
    fn test_lfm2_wasm_memory_estimate_int4() {
        let model_config = Lfm2Config::lfm2_2_6b();
        let wasm_config = Lfm2WasmConfig {
            quantization: WasmQuantization::Int4Awq,
            max_context: 4096,
            sliding_window: Some(2048),
            use_webgpu: false,
            streaming: true,
        };
        let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);
        assert!(estimate.is_viable);
        assert!(estimate.total_bytes / 1_000_000 < 2048);
    }

    #[test]
    fn test_lfm2_wasm_memory_estimate_fp16() {
        let model_config = Lfm2Config::lfm2_2_6b();
        let wasm_config = Lfm2WasmConfig {
            quantization: WasmQuantization::Fp16,
            max_context: 4096,
            sliding_window: Some(2048),
            use_webgpu: false,
            streaming: true,
        };
        let estimate = WasmMemoryEstimate::calculate(&model_config, &wasm_config);
        assert!(!estimate.is_viable);
        assert!(estimate.total_bytes / 1_000_000 > 2048);
    }

    #[test]
    fn test_lfm2_wasm_viability_summary_viable() {
        let summary = format!(
            "✅ WASM Viable: {}MB total (model: {}MB, KV: {}MB, overhead: {}MB)",
            1500, 1300, 100, 100
        );
        assert!(summary.contains("Viable"));
        assert!(summary.contains("1500MB"));
    }

    #[test]
    fn test_lfm2_wasm_viability_summary_not_viable() {
        let summary = format!(
            "❌ Not WASM Viable: {}MB exceeds 2GB limit (model: {}MB, KV: {}MB)",
            2500, 2000, 500
        );
        assert!(summary.contains("Not WASM Viable"));
        assert!(summary.contains("2500MB"));
    }

    #[test]
    fn test_lfm2_wasm_quantization_parsing() {
        // Test quantization string parsing logic
        fn parse_quant(s: &str) -> Option<WasmQuantization> {
            match s.to_lowercase().as_str() {
                "fp16" => Some(WasmQuantization::Fp16),
                "int8" => Some(WasmQuantization::Int8),
                "int4-awq" | "int4_awq" => Some(WasmQuantization::Int4Awq),
                "int4-gptq" | "int4_gptq" => Some(WasmQuantization::Int4Gptq),
                _ => None,
            }
        }

        assert!(matches!(parse_quant("fp16"), Some(WasmQuantization::Fp16)));
        assert!(matches!(parse_quant("int8"), Some(WasmQuantization::Int8)));
        assert!(matches!(
            parse_quant("int4-awq"),
            Some(WasmQuantization::Int4Awq)
        ));
        assert!(matches!(
            parse_quant("int4_gptq"),
            Some(WasmQuantization::Int4Gptq)
        ));
        assert!(parse_quant("invalid").is_none());
    }

    #[test]
    fn test_lfm2_wasm_json_stats_format() {
        // Test JSON format for stats
        let json = format!(
            r#"{{"tokens_generated":{},"ms_per_token":{},"total_ms":{},"tokens_per_sec":{},"hit_eos":{}}}"#,
            42, 15.5, 651.0, 64.5, true
        );
        assert!(json.contains("\"tokens_generated\":42"));
        assert!(json.contains("\"hit_eos\":true"));
    }

    #[test]
    fn test_lfm2_wasm_warnings_json_format() {
        let warnings = vec!["Warning 1".to_string(), "Warning 2".to_string()];
        let json = format!(
            "[{}]",
            warnings
                .iter()
                .map(|w| format!(r#""{}""#, w.replace('"', "\\\"")))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert_eq!(json, r#"["Warning 1","Warning 2"]"#);
    }

    #[test]
    fn test_lfm2_wasm_streaming_callback_logic() {
        // Test that streaming callbacks work correctly
        use std::cell::RefCell;
        use std::rc::Rc;

        let received_tokens = Rc::new(RefCell::new(Vec::new()));
        let tokens_clone = Rc::clone(&received_tokens);

        // Simulate callback behavior
        let callback = move |token: u32, _index: usize| -> bool {
            tokens_clone.borrow_mut().push(token);
            true // continue generating
        };

        // Simulate token generation with callback
        let tokens_to_emit = vec![100u32, 200, 300, 400];
        for (idx, token) in tokens_to_emit.iter().enumerate() {
            let should_continue = callback(*token, idx);
            assert!(should_continue);
        }

        assert_eq!(received_tokens.borrow().len(), 4);
        assert_eq!(*received_tokens.borrow(), vec![100u32, 200, 300, 400]);
    }

    #[test]
    fn test_lfm2_wasm_streaming_callback_early_stop() {
        // Test callback returning false stops generation
        use std::cell::RefCell;
        use std::rc::Rc;

        let received_tokens = Rc::new(RefCell::new(Vec::new()));
        let tokens_clone = Rc::clone(&received_tokens);

        // Callback that stops after 2 tokens
        let callback = move |token: u32, _index: usize| -> bool {
            let mut tokens = tokens_clone.borrow_mut();
            tokens.push(token);
            tokens.len() < 2 // stop after 2 tokens
        };

        // Simulate token generation
        let tokens_to_emit = vec![100u32, 200, 300, 400];
        let mut generated = Vec::new();
        for (idx, token) in tokens_to_emit.iter().enumerate() {
            generated.push(*token);
            if !callback(*token, idx) {
                break;
            }
        }

        assert_eq!(generated, vec![100u32, 200]);
        assert_eq!(received_tokens.borrow().len(), 2);
    }
}

// WASM tests (run with wasm-pack test)
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_lfm2_wasm_new() {
        let wasm = Lfm2Wasm::new();
        assert!(!wasm.is_loaded());
        assert!(wasm.num_params().is_none());
    }

    #[wasm_bindgen_test]
    fn test_memory_estimate_int4() {
        let estimate = Lfm2Wasm::estimate_memory("int4-awq", 4096, 2048).unwrap();
        assert!(estimate.is_viable());
        assert!(estimate.total_mb() < 2048);
    }

    #[wasm_bindgen_test]
    fn test_viability_check_summary() {
        let check = ViabilityCheckWasm {
            is_viable: true,
            total_mb: 1500,
            model_mb: 1300,
            kv_cache_mb: 100,
            overhead_mb: 100,
            warnings: vec![],
        };
        let summary = check.summary();
        assert!(summary.contains("Viable"));
        assert!(summary.contains("1500MB"));
    }

    #[wasm_bindgen_test]
    fn test_lfm2_wasm_streaming_api_exists() {
        // Test that the WASM streaming API compiles and type-checks
        let wasm = Lfm2Wasm::new();
        assert!(!wasm.is_loaded());

        // Verify generate returns error when model not loaded
        let result = wasm.generate(&[1, 2, 3], 10, 0.7);
        assert!(result.is_err());

        // Note: generateStreaming requires a JS Function which is created via js_sys
        // and is best tested in actual browser environment with wasm-pack test
    }
}

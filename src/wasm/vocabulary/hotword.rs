//! WASM hotword bindings

use wasm_bindgen::prelude::*;

use crate::vocabulary::{HotwordBooster, HotwordConfig};

/// Configuration for hotword boosting (WASM)
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct HotwordConfigWasm {
    pub(crate) inner: HotwordConfig,
}

#[wasm_bindgen]
impl HotwordConfigWasm {
    /// Create new configuration with defaults
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: HotwordConfig::new(),
        }
    }

    /// Set default bias value
    #[wasm_bindgen(js_name = setDefaultBias)]
    pub fn set_default_bias(&mut self, bias: f32) {
        self.inner.default_bias = bias;
    }

    /// Get default bias
    #[wasm_bindgen(js_name = getDefaultBias)]
    pub fn get_default_bias(&self) -> f32 {
        self.inner.default_bias
    }

    /// Set maximum bias
    #[wasm_bindgen(js_name = setMaxBias)]
    pub fn set_max_bias(&mut self, max: f32) {
        self.inner.max_bias = max;
    }

    /// Get maximum bias
    #[wasm_bindgen(js_name = getMaxBias)]
    pub fn get_max_bias(&self) -> f32 {
        self.inner.max_bias
    }

    /// Set case sensitivity
    #[wasm_bindgen(js_name = setCaseSensitive)]
    pub fn set_case_sensitive(&mut self, case_sensitive: bool) {
        self.inner.case_sensitive = case_sensitive;
    }

    /// Get case sensitivity
    #[wasm_bindgen(js_name = isCaseSensitive)]
    pub fn is_case_sensitive(&self) -> bool {
        self.inner.case_sensitive
    }

    /// Set partial match decay
    #[wasm_bindgen(js_name = setPartialMatchDecay)]
    pub fn set_partial_match_decay(&mut self, decay: f32) {
        self.inner.partial_match_decay = decay;
    }

    /// Get partial match decay
    #[wasm_bindgen(js_name = getPartialMatchDecay)]
    pub fn get_partial_match_decay(&self) -> f32 {
        self.inner.partial_match_decay
    }
}

impl Default for HotwordConfigWasm {
    fn default() -> Self {
        Self::new()
    }
}

/// A hotword entry (WASM)
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct HotwordWasm {
    /// Text representation
    text: String,
    /// Token sequence
    tokens: Vec<u32>,
    /// Boost value
    bias: f32,
    /// Priority
    priority: u32,
}

#[wasm_bindgen]
impl HotwordWasm {
    /// Create a new hotword
    #[wasm_bindgen(constructor)]
    pub fn new(text: &str, tokens: &[u32], bias: f32) -> Self {
        Self {
            text: text.to_string(),
            tokens: tokens.to_vec(),
            bias,
            priority: 0,
        }
    }

    /// Get text
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    /// Get tokens
    #[wasm_bindgen(getter)]
    pub fn tokens(&self) -> Vec<u32> {
        self.tokens.clone()
    }

    /// Get bias
    #[wasm_bindgen(getter)]
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Set priority
    #[wasm_bindgen(js_name = setPriority)]
    pub fn set_priority(&mut self, priority: u32) {
        self.priority = priority;
    }

    /// Get priority
    #[wasm_bindgen(getter)]
    pub fn priority(&self) -> u32 {
        self.priority
    }

    /// Get token count
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.tokens.len()
    }
}

/// Hotword booster for logit biasing (WASM)
#[wasm_bindgen]
pub struct HotwordBoosterWasm {
    inner: HotwordBooster,
}

#[wasm_bindgen]
impl HotwordBoosterWasm {
    /// Create new booster with default config
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: HotwordBooster::new(),
        }
    }

    /// Create with custom config
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(config: &HotwordConfigWasm) -> Self {
        Self {
            inner: HotwordBooster::with_config(config.inner.clone()),
        }
    }

    /// Add a hotword with tokens and bias
    #[wasm_bindgen(js_name = addHotword)]
    pub fn add_hotword(&mut self, text: &str, tokens: &[u32], bias: f32) {
        self.inner
            .add_hotword_with_tokens(text, tokens.to_vec(), bias);
    }

    /// Add hotword with default bias
    #[wasm_bindgen(js_name = addHotwordDefault)]
    pub fn add_hotword_default(&mut self, text: &str, tokens: &[u32]) {
        self.inner
            .add_hotword_with_tokens_default(text, tokens.to_vec());
    }

    /// Apply bias to logits in place
    #[wasm_bindgen(js_name = applyBias)]
    pub fn apply_bias(&self, logits: &mut [f32], context: &[u32]) {
        self.inner.apply_bias(logits, context);
    }

    /// Get completion tokens with biases
    #[wasm_bindgen(js_name = getCompletionTokens)]
    pub fn get_completion_tokens(&self, context: &[u32]) -> Vec<u32> {
        self.inner
            .get_completion_tokens(context)
            .into_iter()
            .map(|(token, _)| token)
            .collect()
    }

    /// Get completion biases
    #[wasm_bindgen(js_name = getCompletionBiases)]
    pub fn get_completion_biases(&self, context: &[u32]) -> Vec<f32> {
        self.inner
            .get_completion_tokens(context)
            .into_iter()
            .map(|(_, bias)| bias)
            .collect()
    }

    /// Clear all hotwords
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get hotword count
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for HotwordBoosterWasm {
    fn default() -> Self {
        Self::new()
    }
}

impl HotwordBoosterWasm {
    /// Convert to inner type (consumes self)
    pub(super) fn into_inner(self) -> HotwordBooster {
        self.inner
    }
}

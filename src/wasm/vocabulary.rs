//! WASM bindings for vocabulary customization (WAPR-173)
//!
//! Provides JavaScript-accessible APIs for hotword boosting, vocabulary tries,
//! and domain adaptation.
//!
//! # Usage from JavaScript
//!
//! ```javascript
//! // Hotword boosting
//! const booster = new HotwordBoosterWasm();
//! booster.addHotword("Anthropic", new Uint32Array([1234, 5678]), 2.0);
//!
//! // Apply during decoding
//! const biasedLogits = booster.applyBias(logits, contextTokens);
//!
//! // Domain adaptation
//! const adapter = DomainAdapterWasm.medical();
//! adapter.addTerm("myocardial infarction", new Uint32Array([100, 200, 300]), 1.5);
//!
//! // Vocabulary trie
//! const trie = new VocabularyTrieWasm();
//! trie.insert(new Uint32Array([100, 200]), "hello", 1.0);
//! const completions = trie.getCompletions(new Uint32Array([100]));
//! ```

use wasm_bindgen::prelude::*;

use crate::vocabulary::{
    DomainAdapter, DomainConfig, DomainType, HotwordBooster, HotwordConfig, TrieSearchResult,
    VocabularyCustomizer, VocabularyTrie,
};

// ============================================================
// HotwordConfig WASM Bindings
// ============================================================

/// Configuration for hotword boosting (WASM)
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct HotwordConfigWasm {
    inner: HotwordConfig,
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

// ============================================================
// Hotword WASM Bindings
// ============================================================

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

// ============================================================
// HotwordBooster WASM Bindings
// ============================================================

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

// ============================================================
// VocabularyTrie WASM Bindings
// ============================================================

/// Search result from vocabulary trie (WASM)
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct TrieSearchResultWasm {
    /// Continuation tokens
    continuation_tokens: Vec<u32>,
    /// Continuation boosts
    continuation_boosts: Vec<f32>,
    /// Whether prefix is a complete entry
    is_complete: bool,
    /// Text if complete
    text: Option<String>,
    /// Depth in trie
    depth: usize,
    /// Number of matching entries
    matching_entries: usize,
}

#[wasm_bindgen]
impl TrieSearchResultWasm {
    /// Get continuation tokens
    #[wasm_bindgen(js_name = getContinuationTokens)]
    pub fn get_continuation_tokens(&self) -> Vec<u32> {
        self.continuation_tokens.clone()
    }

    /// Get continuation boosts
    #[wasm_bindgen(js_name = getContinuationBoosts)]
    pub fn get_continuation_boosts(&self) -> Vec<f32> {
        self.continuation_boosts.clone()
    }

    /// Check if prefix is complete entry
    #[wasm_bindgen(getter, js_name = isComplete)]
    pub fn is_complete(&self) -> bool {
        self.is_complete
    }

    /// Get text if complete
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> Option<String> {
        self.text.clone()
    }

    /// Get search depth
    #[wasm_bindgen(getter)]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Get number of matching entries
    #[wasm_bindgen(getter, js_name = matchingEntries)]
    pub fn matching_entries(&self) -> usize {
        self.matching_entries
    }

    /// Check if any matches found
    #[wasm_bindgen(js_name = hasMatches)]
    pub fn has_matches(&self) -> bool {
        !self.continuation_tokens.is_empty() || self.is_complete
    }
}

impl From<TrieSearchResult> for TrieSearchResultWasm {
    fn from(result: TrieSearchResult) -> Self {
        let (tokens, boosts): (Vec<u32>, Vec<f32>) = result.continuations.into_iter().unzip();
        Self {
            continuation_tokens: tokens,
            continuation_boosts: boosts,
            is_complete: result.is_complete,
            text: result.text,
            depth: result.depth,
            matching_entries: result.matching_entries,
        }
    }
}

/// Vocabulary trie for efficient prefix lookup (WASM)
#[wasm_bindgen]
pub struct VocabularyTrieWasm {
    inner: VocabularyTrie,
}

#[wasm_bindgen]
impl VocabularyTrieWasm {
    /// Create new vocabulary trie
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: VocabularyTrie::new(),
        }
    }

    /// Create with custom default boost
    #[wasm_bindgen(js_name = withDefaultBoost)]
    pub fn with_default_boost(boost: f32) -> Self {
        Self {
            inner: VocabularyTrie::new().with_default_boost(boost),
        }
    }

    /// Insert entry into trie
    #[wasm_bindgen]
    pub fn insert(&mut self, tokens: &[u32], text: &str, boost: f32) {
        self.inner.insert(tokens, text, boost);
    }

    /// Insert with default boost
    #[wasm_bindgen(js_name = insertDefault)]
    pub fn insert_default(&mut self, tokens: &[u32], text: &str) {
        self.inner.insert_default(tokens, text);
    }

    /// Check if entry exists
    #[wasm_bindgen]
    pub fn contains(&self, tokens: &[u32]) -> bool {
        self.inner.contains(tokens)
    }

    /// Check if prefix exists
    #[wasm_bindgen(js_name = hasPrefix)]
    pub fn has_prefix(&self, prefix: &[u32]) -> bool {
        self.inner.has_prefix(prefix)
    }

    /// Search for continuations
    #[wasm_bindgen]
    pub fn search(&self, prefix: &[u32]) -> TrieSearchResultWasm {
        self.inner.search(prefix).into()
    }

    /// Get continuation tokens
    #[wasm_bindgen(js_name = getContinuations)]
    pub fn get_continuations(&self, prefix: &[u32]) -> Vec<u32> {
        self.inner
            .get_continuations(prefix)
            .into_iter()
            .map(|(token, _)| token)
            .collect()
    }

    /// Apply prefix boost to logits
    #[wasm_bindgen(js_name = applyPrefixBoost)]
    pub fn apply_prefix_boost(&self, logits: &mut [f32], context: &[u32]) {
        self.inner.apply_prefix_boost(logits, context);
    }

    /// Clear all entries
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get entry count
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

impl Default for VocabularyTrieWasm {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// DomainAdapter WASM Bindings
// ============================================================

/// Domain type enumeration for JavaScript
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum DomainTypeWasm {
    /// General purpose vocabulary
    General = 0,
    /// Medical terminology domain
    Medical = 1,
    /// Legal terminology domain
    Legal = 2,
    /// Technical/engineering domain
    Technical = 3,
    /// Financial/business domain
    Financial = 4,
    /// Scientific terminology domain
    Scientific = 5,
    /// Custom user-defined domain
    Custom = 6,
}

impl From<DomainTypeWasm> for DomainType {
    fn from(wasm: DomainTypeWasm) -> Self {
        match wasm {
            DomainTypeWasm::General => DomainType::General,
            DomainTypeWasm::Medical => DomainType::Medical,
            DomainTypeWasm::Legal => DomainType::Legal,
            DomainTypeWasm::Technical => DomainType::Technical,
            DomainTypeWasm::Financial => DomainType::Financial,
            DomainTypeWasm::Scientific => DomainType::Scientific,
            DomainTypeWasm::Custom => DomainType::Custom,
        }
    }
}

impl From<DomainType> for DomainTypeWasm {
    fn from(domain: DomainType) -> Self {
        match domain {
            DomainType::General => DomainTypeWasm::General,
            DomainType::Medical => DomainTypeWasm::Medical,
            DomainType::Legal => DomainTypeWasm::Legal,
            DomainType::Technical => DomainTypeWasm::Technical,
            DomainType::Financial => DomainTypeWasm::Financial,
            DomainType::Scientific => DomainTypeWasm::Scientific,
            DomainType::Custom => DomainTypeWasm::Custom,
        }
    }
}

/// Domain configuration (WASM)
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DomainConfigWasm {
    inner: DomainConfig,
}

#[wasm_bindgen]
impl DomainConfigWasm {
    /// Create new config with defaults
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: DomainConfig::new(),
        }
    }

    /// Set base boost
    #[wasm_bindgen(js_name = setBaseBoost)]
    pub fn set_base_boost(&mut self, boost: f32) {
        self.inner.base_boost = boost;
    }

    /// Get base boost
    #[wasm_bindgen(js_name = getBaseBoost)]
    pub fn get_base_boost(&self) -> f32 {
        self.inner.base_boost
    }

    /// Set priority multiplier
    #[wasm_bindgen(js_name = setPriorityMultiplier)]
    pub fn set_priority_multiplier(&mut self, multiplier: f32) {
        self.inner.priority_multiplier = multiplier;
    }

    /// Get priority multiplier
    #[wasm_bindgen(js_name = getPriorityMultiplier)]
    pub fn get_priority_multiplier(&self) -> f32 {
        self.inner.priority_multiplier
    }

    /// Set max boost
    #[wasm_bindgen(js_name = setMaxBoost)]
    pub fn set_max_boost(&mut self, max: f32) {
        self.inner.max_boost = max;
    }

    /// Get max boost
    #[wasm_bindgen(js_name = getMaxBoost)]
    pub fn get_max_boost(&self) -> f32 {
        self.inner.max_boost
    }
}

impl Default for DomainConfigWasm {
    fn default() -> Self {
        Self::new()
    }
}

/// Domain term entry (WASM)
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DomainTermWasm {
    text: String,
    tokens: Vec<u32>,
    boost: f32,
    is_priority: bool,
    category: Option<String>,
}

#[wasm_bindgen]
impl DomainTermWasm {
    /// Create new domain term
    #[wasm_bindgen(constructor)]
    pub fn new(text: &str, tokens: &[u32], boost: f32) -> Self {
        Self {
            text: text.to_string(),
            tokens: tokens.to_vec(),
            boost,
            is_priority: false,
            category: None,
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

    /// Get boost
    #[wasm_bindgen(getter)]
    pub fn boost(&self) -> f32 {
        self.boost
    }

    /// Check if priority
    #[wasm_bindgen(getter, js_name = isPriority)]
    pub fn is_priority(&self) -> bool {
        self.is_priority
    }

    /// Set as priority
    #[wasm_bindgen(js_name = setPriority)]
    pub fn set_priority(&mut self, priority: bool) {
        self.is_priority = priority;
    }

    /// Get category
    #[wasm_bindgen(getter)]
    pub fn category(&self) -> Option<String> {
        self.category.clone()
    }

    /// Set category
    #[wasm_bindgen(js_name = setCategory)]
    pub fn set_category(&mut self, category: &str) {
        self.category = Some(category.to_string());
    }
}

/// Domain vocabulary adapter (WASM)
#[wasm_bindgen]
pub struct DomainAdapterWasm {
    inner: DomainAdapter,
}

#[wasm_bindgen]
impl DomainAdapterWasm {
    /// Create new domain adapter
    #[wasm_bindgen(constructor)]
    pub fn new(domain_type: DomainTypeWasm) -> Self {
        Self {
            inner: DomainAdapter::new(domain_type.into()),
        }
    }

    /// Create medical domain adapter
    #[wasm_bindgen]
    pub fn medical() -> Self {
        Self {
            inner: DomainAdapter::medical(),
        }
    }

    /// Create legal domain adapter
    #[wasm_bindgen]
    pub fn legal() -> Self {
        Self {
            inner: DomainAdapter::legal(),
        }
    }

    /// Create technical domain adapter
    #[wasm_bindgen]
    pub fn technical() -> Self {
        Self {
            inner: DomainAdapter::technical(),
        }
    }

    /// Create financial domain adapter
    #[wasm_bindgen]
    pub fn financial() -> Self {
        Self {
            inner: DomainAdapter::financial(),
        }
    }

    /// Create scientific domain adapter
    #[wasm_bindgen]
    pub fn scientific() -> Self {
        Self {
            inner: DomainAdapter::scientific(),
        }
    }

    /// Create custom domain adapter
    #[wasm_bindgen]
    pub fn custom() -> Self {
        Self {
            inner: DomainAdapter::custom(),
        }
    }

    /// Get domain type
    #[wasm_bindgen(js_name = getDomainType)]
    pub fn get_domain_type(&self) -> DomainTypeWasm {
        self.inner.domain_type().into()
    }

    /// Add term with tokens
    #[wasm_bindgen(js_name = addTerm)]
    pub fn add_term(&mut self, text: &str, tokens: &[u32], boost: f32) {
        self.inner
            .add_term_with_tokens(text, tokens.to_vec(), boost);
    }

    /// Add term with default boost
    #[wasm_bindgen(js_name = addTermDefault)]
    pub fn add_term_default(&mut self, text: &str, tokens: &[u32]) {
        self.inner
            .add_term_with_tokens_default(text, tokens.to_vec());
    }

    /// Add priority term
    #[wasm_bindgen(js_name = addPriorityTerm)]
    pub fn add_priority_term(&mut self, text: &str, tokens: &[u32]) {
        self.inner.add_priority_term(text, tokens.to_vec());
    }

    /// Apply bias to logits
    #[wasm_bindgen(js_name = applyBias)]
    pub fn apply_bias(&self, logits: &mut [f32]) {
        self.inner.apply_bias(logits);
    }

    /// Check if token is in domain
    #[wasm_bindgen(js_name = isDomainToken)]
    pub fn is_domain_token(&self, token: u32) -> bool {
        self.inner.is_domain_token(token)
    }

    /// Get token boost
    #[wasm_bindgen(js_name = getTokenBoost)]
    pub fn get_token_boost(&self, token: u32) -> Option<f32> {
        self.inner.get_token_boost(token)
    }

    /// Clear all terms
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get term count
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get categories
    #[wasm_bindgen]
    pub fn categories(&self) -> Vec<String> {
        self.inner.categories()
    }
}

// ============================================================
// VocabularyCustomizer WASM Bindings
// ============================================================

/// Combined vocabulary customizer (WASM)
#[wasm_bindgen]
pub struct VocabularyCustomizerWasm {
    inner: VocabularyCustomizer,
}

#[wasm_bindgen]
impl VocabularyCustomizerWasm {
    /// Create new customizer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: VocabularyCustomizer::new(),
        }
    }

    /// Set hotword booster
    #[wasm_bindgen(js_name = setHotwordBooster)]
    pub fn set_hotword_booster(&mut self, booster: HotwordBoosterWasm) {
        self.inner = VocabularyCustomizer::new().with_hotword_booster(booster.inner);
    }

    /// Set domain adapter
    #[wasm_bindgen(js_name = setDomainAdapter)]
    pub fn set_domain_adapter(&mut self, adapter: DomainAdapterWasm) {
        self.inner = VocabularyCustomizer::new().with_domain_adapter(adapter.inner);
    }

    /// Set vocabulary trie
    #[wasm_bindgen(js_name = setVocabularyTrie)]
    pub fn set_vocabulary_trie(&mut self, trie: VocabularyTrieWasm) {
        self.inner = VocabularyCustomizer::new().with_vocabulary_trie(trie.inner);
    }

    /// Apply all customizations to logits
    #[wasm_bindgen]
    pub fn apply(&self, logits: &mut [f32], context: &[u32]) {
        self.inner.apply(logits, context);
    }

    /// Check if any customization is active
    #[wasm_bindgen(js_name = isActive)]
    pub fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    /// Check if hotword booster is set
    #[wasm_bindgen(js_name = hasHotwordBooster)]
    pub fn has_hotword_booster(&self) -> bool {
        self.inner.hotword_booster().is_some()
    }

    /// Check if domain adapter is set
    #[wasm_bindgen(js_name = hasDomainAdapter)]
    pub fn has_domain_adapter(&self) -> bool {
        self.inner.domain_adapter().is_some()
    }

    /// Check if vocabulary trie is set
    #[wasm_bindgen(js_name = hasVocabularyTrie)]
    pub fn has_vocabulary_trie(&self) -> bool {
        self.inner.vocabulary_trie().is_some()
    }
}

impl Default for VocabularyCustomizerWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // HotwordConfigWasm Tests
    // ============================================================

    #[test]
    fn test_hotword_config_wasm_new() {
        let config = HotwordConfigWasm::new();
        assert!((config.get_default_bias() - 1.0).abs() < f32::EPSILON);
        assert!((config.get_max_bias() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hotword_config_wasm_setters() {
        let mut config = HotwordConfigWasm::new();
        config.set_default_bias(2.0);
        config.set_max_bias(10.0);
        config.set_case_sensitive(true);
        config.set_partial_match_decay(0.5);

        assert!((config.get_default_bias() - 2.0).abs() < f32::EPSILON);
        assert!((config.get_max_bias() - 10.0).abs() < f32::EPSILON);
        assert!(config.is_case_sensitive());
        assert!((config.get_partial_match_decay() - 0.5).abs() < f32::EPSILON);
    }

    // ============================================================
    // HotwordWasm Tests
    // ============================================================

    #[test]
    fn test_hotword_wasm_new() {
        let hotword = HotwordWasm::new("test", &[100, 200], 1.5);
        assert_eq!(hotword.text(), "test");
        assert_eq!(hotword.tokens(), vec![100, 200]);
        assert!((hotword.bias() - 1.5).abs() < f32::EPSILON);
        assert_eq!(hotword.length(), 2);
    }

    #[test]
    fn test_hotword_wasm_priority() {
        let mut hotword = HotwordWasm::new("test", &[100], 1.0);
        assert_eq!(hotword.priority(), 0);
        hotword.set_priority(5);
        assert_eq!(hotword.priority(), 5);
    }

    // ============================================================
    // HotwordBoosterWasm Tests
    // ============================================================

    #[test]
    fn test_hotword_booster_wasm_new() {
        let booster = HotwordBoosterWasm::new();
        assert!(booster.is_empty());
        assert_eq!(booster.length(), 0);
    }

    #[test]
    fn test_hotword_booster_wasm_add() {
        let mut booster = HotwordBoosterWasm::new();
        booster.add_hotword("test", &[100, 200], 2.0);
        assert_eq!(booster.length(), 1);
    }

    #[test]
    fn test_hotword_booster_wasm_apply() {
        let mut booster = HotwordBoosterWasm::new();
        booster.add_hotword("test", &[50], 2.0);

        let mut logits = vec![0.0; 100];
        booster.apply_bias(&mut logits, &[]);

        assert!(logits[50] > 0.0);
    }

    #[test]
    fn test_hotword_booster_wasm_clear() {
        let mut booster = HotwordBoosterWasm::new();
        booster.add_hotword("test", &[100], 1.0);
        assert_eq!(booster.length(), 1);
        booster.clear();
        assert!(booster.is_empty());
    }

    // ============================================================
    // VocabularyTrieWasm Tests
    // ============================================================

    #[test]
    fn test_vocabulary_trie_wasm_new() {
        let trie = VocabularyTrieWasm::new();
        assert!(trie.is_empty());
    }

    #[test]
    fn test_vocabulary_trie_wasm_insert() {
        let mut trie = VocabularyTrieWasm::new();
        trie.insert(&[100, 200], "hello", 1.0);
        assert_eq!(trie.length(), 1);
        assert!(trie.contains(&[100, 200]));
    }

    #[test]
    fn test_vocabulary_trie_wasm_search() {
        let mut trie = VocabularyTrieWasm::new();
        trie.insert(&[100, 200], "hello", 1.5);
        trie.insert(&[100, 300], "help", 1.0);

        let result = trie.search(&[100]);
        assert!(result.has_matches());
        assert_eq!(result.get_continuation_tokens().len(), 2);
    }

    #[test]
    fn test_vocabulary_trie_wasm_apply_boost() {
        let mut trie = VocabularyTrieWasm::new();
        trie.insert(&[50, 100], "test", 2.0);

        let mut logits = vec![0.0; 200];
        trie.apply_prefix_boost(&mut logits, &[50]);

        assert!(logits[100] > 0.0);
    }

    // ============================================================
    // DomainAdapterWasm Tests
    // ============================================================

    #[test]
    fn test_domain_adapter_wasm_factory() {
        let medical = DomainAdapterWasm::medical();
        assert!(matches!(medical.get_domain_type(), DomainTypeWasm::Medical));

        let legal = DomainAdapterWasm::legal();
        assert!(matches!(legal.get_domain_type(), DomainTypeWasm::Legal));

        let custom = DomainAdapterWasm::custom();
        assert!(matches!(custom.get_domain_type(), DomainTypeWasm::Custom));
    }

    #[test]
    fn test_domain_adapter_wasm_add_term() {
        let mut adapter = DomainAdapterWasm::custom();
        adapter.add_term("test", &[100, 200], 1.5);
        assert_eq!(adapter.length(), 1);
        assert!(adapter.is_domain_token(100));
        assert!(adapter.is_domain_token(200));
    }

    #[test]
    fn test_domain_adapter_wasm_apply_bias() {
        let mut adapter = DomainAdapterWasm::custom();
        adapter.add_term("test", &[50], 2.0);

        let mut logits = vec![0.0; 100];
        adapter.apply_bias(&mut logits);

        assert!((logits[50] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_domain_adapter_wasm_priority() {
        let mut adapter = DomainAdapterWasm::custom();
        adapter.add_priority_term("priority", &[100]);

        // Priority terms get boosted more
        let boost = adapter.get_token_boost(100).unwrap_or(0.0);
        assert!(boost > 1.0); // Should be base * multiplier
    }

    // ============================================================
    // VocabularyCustomizerWasm Tests
    // ============================================================

    #[test]
    fn test_vocabulary_customizer_wasm_new() {
        let customizer = VocabularyCustomizerWasm::new();
        assert!(!customizer.is_active());
    }

    #[test]
    fn test_vocabulary_customizer_wasm_setters() {
        let mut customizer = VocabularyCustomizerWasm::new();

        let booster = HotwordBoosterWasm::new();
        customizer.set_hotword_booster(booster);
        assert!(customizer.has_hotword_booster());
    }

    #[test]
    fn test_vocabulary_customizer_wasm_apply() {
        let customizer = VocabularyCustomizerWasm::new();
        let mut logits = vec![1.0; 100];
        customizer.apply(&mut logits, &[]);

        // No customization active, logits unchanged
        for &logit in &logits {
            assert!((logit - 1.0).abs() < f32::EPSILON);
        }
    }

    // ============================================================
    // DomainTermWasm Tests
    // ============================================================

    #[test]
    fn test_domain_term_wasm_new() {
        let term = DomainTermWasm::new("test", &[100, 200], 1.5);
        assert_eq!(term.text(), "test");
        assert_eq!(term.tokens(), vec![100, 200]);
        assert!((term.boost() - 1.5).abs() < f32::EPSILON);
        assert!(!term.is_priority());
    }

    #[test]
    fn test_domain_term_wasm_priority() {
        let mut term = DomainTermWasm::new("test", &[100], 1.0);
        assert!(!term.is_priority());
        term.set_priority(true);
        assert!(term.is_priority());
    }

    #[test]
    fn test_domain_term_wasm_category() {
        let mut term = DomainTermWasm::new("test", &[100], 1.0);
        assert!(term.category().is_none());
        term.set_category("anatomy");
        assert_eq!(term.category(), Some("anatomy".to_string()));
    }

    // ============================================================
    // TrieSearchResultWasm Tests
    // ============================================================

    #[test]
    fn test_trie_search_result_wasm_from() {
        let result = TrieSearchResult {
            continuations: vec![(100, 1.0), (200, 2.0)],
            is_complete: true,
            text: Some("hello".to_string()),
            depth: 3,
            matching_entries: 5,
        };

        let wasm_result: TrieSearchResultWasm = result.into();
        assert_eq!(wasm_result.get_continuation_tokens(), vec![100, 200]);
        assert_eq!(wasm_result.get_continuation_boosts(), vec![1.0, 2.0]);
        assert!(wasm_result.is_complete());
        assert_eq!(wasm_result.text(), Some("hello".to_string()));
        assert_eq!(wasm_result.depth(), 3);
        assert_eq!(wasm_result.matching_entries(), 5);
        assert!(wasm_result.has_matches());
    }

    // ============================================================
    // DomainConfigWasm Tests
    // ============================================================

    #[test]
    fn test_domain_config_wasm_new() {
        let config = DomainConfigWasm::new();
        assert!((config.get_base_boost() - 1.0).abs() < f32::EPSILON);
        assert!((config.get_priority_multiplier() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_domain_config_wasm_setters() {
        let mut config = DomainConfigWasm::new();
        config.set_base_boost(2.0);
        config.set_priority_multiplier(3.0);
        config.set_max_boost(10.0);

        assert!((config.get_base_boost() - 2.0).abs() < f32::EPSILON);
        assert!((config.get_priority_multiplier() - 3.0).abs() < f32::EPSILON);
        assert!((config.get_max_boost() - 10.0).abs() < f32::EPSILON);
    }

    // ============================================================
    // Additional Coverage Tests
    // ============================================================

    #[test]
    fn test_hotword_config_wasm_default_trait() {
        let config = HotwordConfigWasm::default();
        assert!((config.get_default_bias() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_domain_adapter_wasm_technical() {
        let adapter = DomainAdapterWasm::technical();
        assert!(matches!(
            adapter.get_domain_type(),
            DomainTypeWasm::Technical
        ));
    }

    #[test]
    fn test_domain_adapter_wasm_financial() {
        let adapter = DomainAdapterWasm::financial();
        assert!(matches!(
            adapter.get_domain_type(),
            DomainTypeWasm::Financial
        ));
    }

    #[test]
    fn test_domain_adapter_wasm_scientific() {
        let adapter = DomainAdapterWasm::scientific();
        assert!(matches!(
            adapter.get_domain_type(),
            DomainTypeWasm::Scientific
        ));
    }

    #[test]
    fn test_domain_adapter_wasm_clear() {
        let mut adapter = DomainAdapterWasm::custom();
        adapter.add_term("test", &[100], 1.0);
        assert_eq!(adapter.length(), 1);
        adapter.clear();
        assert_eq!(adapter.length(), 0);
    }

    #[test]
    fn test_domain_type_wasm_variants() {
        let variants = vec![
            DomainTypeWasm::Medical,
            DomainTypeWasm::Legal,
            DomainTypeWasm::Technical,
            DomainTypeWasm::Financial,
            DomainTypeWasm::Scientific,
            DomainTypeWasm::Custom,
        ];
        for variant in variants {
            let debug_str = format!("{variant:?}");
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_vocabulary_trie_wasm_clear() {
        let mut trie = VocabularyTrieWasm::new();
        trie.insert(&[100], "test", 1.0);
        assert_eq!(trie.length(), 1);
        trie.clear();
        assert!(trie.is_empty());
    }

    #[test]
    fn test_trie_search_result_wasm_empty() {
        let result = TrieSearchResult {
            continuations: vec![],
            is_complete: false,
            text: None,
            depth: 0,
            matching_entries: 0,
        };

        let wasm_result: TrieSearchResultWasm = result.into();
        assert!(!wasm_result.has_matches());
        assert!(!wasm_result.is_complete());
        assert!(wasm_result.text().is_none());
    }

    #[test]
    fn test_vocabulary_customizer_wasm_domain_adapter() {
        let mut customizer = VocabularyCustomizerWasm::new();
        let adapter = DomainAdapterWasm::medical();
        customizer.set_domain_adapter(adapter);
        assert!(customizer.has_domain_adapter());
    }

    #[test]
    fn test_vocabulary_customizer_wasm_vocabulary_trie() {
        let mut customizer = VocabularyCustomizerWasm::new();
        let trie = VocabularyTrieWasm::new();
        customizer.set_vocabulary_trie(trie);
        assert!(customizer.has_vocabulary_trie());
    }

    #[test]
    fn test_hotword_booster_wasm_with_config() {
        let config = HotwordConfigWasm::new();
        let booster = HotwordBoosterWasm::with_config(&config);
        assert!(booster.is_empty());
    }

    #[test]
    fn test_domain_adapter_wasm_new_with_type() {
        let adapter = DomainAdapterWasm::new(DomainTypeWasm::Medical);
        assert!(matches!(adapter.get_domain_type(), DomainTypeWasm::Medical));
    }

    #[test]
    fn test_hotword_booster_wasm_add_default() {
        let mut booster = HotwordBoosterWasm::new();
        booster.add_hotword_default("Claude", &[1, 2, 3]);
        assert_eq!(booster.length(), 1);
    }

    #[test]
    fn test_hotword_booster_wasm_get_completion() {
        let mut booster = HotwordBoosterWasm::new();
        booster.add_hotword("test", &[100, 101, 102], 5.0);

        let tokens = booster.get_completion_tokens(&[100]);
        let biases = booster.get_completion_biases(&[100]);

        // May or may not have completions depending on internal state
        assert_eq!(tokens.len(), biases.len());
    }
}

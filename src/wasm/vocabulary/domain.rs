//! WASM domain adapter bindings

use wasm_bindgen::prelude::*;

use crate::vocabulary::{DomainAdapter, DomainConfig, DomainType};

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
            DomainTypeWasm::General => Self::General,
            DomainTypeWasm::Medical => Self::Medical,
            DomainTypeWasm::Legal => Self::Legal,
            DomainTypeWasm::Technical => Self::Technical,
            DomainTypeWasm::Financial => Self::Financial,
            DomainTypeWasm::Scientific => Self::Scientific,
            DomainTypeWasm::Custom => Self::Custom,
        }
    }
}

impl From<DomainType> for DomainTypeWasm {
    fn from(domain: DomainType) -> Self {
        match domain {
            DomainType::General => Self::General,
            DomainType::Medical => Self::Medical,
            DomainType::Legal => Self::Legal,
            DomainType::Technical => Self::Technical,
            DomainType::Financial => Self::Financial,
            DomainType::Scientific => Self::Scientific,
            DomainType::Custom => Self::Custom,
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

impl DomainAdapterWasm {
    /// Convert to inner type (consumes self)
    pub(super) fn into_inner(self) -> DomainAdapter {
        self.inner
    }
}

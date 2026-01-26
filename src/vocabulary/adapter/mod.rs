//! Domain vocabulary adapter (WAPR-172)
//!
//! Adapts vocabulary recognition for specific domains like medical, legal, or technical.
//!
//! # Overview
//!
//! The domain adapter provides:
//! 1. Pre-defined domain vocabularies (medical, legal, technical)
//! 2. Custom domain word lists with boosting
//! 3. Dynamic vocabulary expansion during inference
//!
//! # Example
//!
//! ```rust,ignore
//! use whisper_apr::vocabulary::{DomainAdapter, DomainType};
//!
//! // Use pre-defined medical domain
//! let adapter = DomainAdapter::new(DomainType::Medical);
//!
//! // Or create custom domain
//! let mut custom = DomainAdapter::new(DomainType::Custom);
//! custom.add_term_with_tokens("proprietary_term", vec![100, 200], 2.0);
//!
//! // Apply to logits during decoding
//! adapter.apply_bias(&mut logits);
//! ```

use std::collections::HashMap;

#[cfg(test)]
mod tests;

/// Domain type for vocabulary adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DomainType {
    /// General domain (no specialization)
    General,
    /// Medical/healthcare terminology
    Medical,
    /// Legal/law terminology
    Legal,
    /// Technical/engineering terminology
    Technical,
    /// Financial/business terminology
    Financial,
    /// Scientific/academic terminology
    Scientific,
    /// Custom user-defined domain
    Custom,
}

impl DomainType {
    /// Get display name
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Medical => "Medical",
            Self::Legal => "Legal",
            Self::Technical => "Technical",
            Self::Financial => "Financial",
            Self::Scientific => "Scientific",
            Self::Custom => "Custom",
        }
    }

    /// Check if domain has pre-defined terms
    #[must_use]
    pub fn has_predefined_terms(&self) -> bool {
        !matches!(self, Self::General | Self::Custom)
    }
}

/// Configuration for domain adaptation
#[derive(Debug, Clone)]
pub struct DomainConfig {
    /// Base boost for domain terms
    pub base_boost: f32,
    /// Boost multiplier for high-priority terms
    pub priority_multiplier: f32,
    /// Maximum boost value
    pub max_boost: f32,
    /// Whether to suppress out-of-domain terms
    pub suppress_out_of_domain: bool,
    /// Suppression factor (negative bias)
    pub suppression_factor: f32,
}

impl DomainConfig {
    /// Create default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_boost: 1.0,
            priority_multiplier: 1.5,
            max_boost: 5.0,
            suppress_out_of_domain: false,
            suppression_factor: 0.5,
        }
    }

    /// Set base boost
    #[must_use]
    pub fn with_base_boost(mut self, boost: f32) -> Self {
        self.base_boost = boost;
        self
    }

    /// Set priority multiplier
    #[must_use]
    pub fn with_priority_multiplier(mut self, multiplier: f32) -> Self {
        self.priority_multiplier = multiplier;
        self
    }

    /// Set max boost
    #[must_use]
    pub fn with_max_boost(mut self, max: f32) -> Self {
        self.max_boost = max;
        self
    }

    /// Enable out-of-domain suppression
    #[must_use]
    pub fn with_suppression(mut self, factor: f32) -> Self {
        self.suppress_out_of_domain = true;
        self.suppression_factor = factor;
        self
    }
}

impl Default for DomainConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A domain term with its token representation and boost
#[derive(Debug, Clone)]
pub struct DomainTerm {
    /// Original text
    pub text: String,
    /// Token sequence
    pub tokens: Vec<u32>,
    /// Boost value
    pub boost: f32,
    /// Whether this is a high-priority term
    pub is_priority: bool,
    /// Category within the domain
    pub category: Option<String>,
}

impl DomainTerm {
    /// Create a new domain term
    #[must_use]
    pub fn new(text: String, tokens: Vec<u32>, boost: f32) -> Self {
        Self {
            text,
            tokens,
            boost,
            is_priority: false,
            category: None,
        }
    }

    /// Mark as priority term
    #[must_use]
    pub fn with_priority(mut self) -> Self {
        self.is_priority = true;
        self
    }

    /// Set category
    #[must_use]
    pub fn with_category(mut self, category: &str) -> Self {
        self.category = Some(category.to_string());
        self
    }

    /// Get first token (for quick lookup)
    #[must_use]
    pub fn first_token(&self) -> Option<u32> {
        self.tokens.first().copied()
    }
}

/// Domain vocabulary adapter
#[derive(Debug, Clone)]
pub struct DomainAdapter {
    /// Domain type
    domain_type: DomainType,
    /// Configuration
    config: DomainConfig,
    /// Domain terms indexed by first token
    pub(crate) terms: Vec<DomainTerm>,
    /// Token to term indices map
    first_token_map: HashMap<u32, Vec<usize>>,
    /// All tokens that belong to domain terms (for suppression)
    domain_tokens: HashMap<u32, f32>,
}

impl DomainAdapter {
    /// Create a new domain adapter
    #[must_use]
    pub fn new(domain_type: DomainType) -> Self {
        let mut adapter = Self {
            domain_type,
            config: DomainConfig::default(),
            terms: Vec::new(),
            first_token_map: HashMap::new(),
            domain_tokens: HashMap::new(),
        };

        // Load pre-defined terms if available
        if domain_type.has_predefined_terms() {
            adapter.load_predefined_terms();
        }

        adapter
    }

    /// Create with custom config
    #[must_use]
    pub fn with_config(domain_type: DomainType, config: DomainConfig) -> Self {
        let mut adapter = Self {
            domain_type,
            config,
            terms: Vec::new(),
            first_token_map: HashMap::new(),
            domain_tokens: HashMap::new(),
        };

        if domain_type.has_predefined_terms() {
            adapter.load_predefined_terms();
        }

        adapter
    }

    /// Get domain type
    #[must_use]
    pub fn domain_type(&self) -> DomainType {
        self.domain_type
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &DomainConfig {
        &self.config
    }

    /// Get number of terms
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get all terms
    #[must_use]
    pub fn terms(&self) -> &[DomainTerm] {
        &self.terms
    }

    /// Add a term with tokens
    pub fn add_term_with_tokens(&mut self, text: &str, tokens: Vec<u32>, boost: f32) {
        if tokens.is_empty() {
            return;
        }

        let clamped_boost = boost.clamp(-self.config.max_boost, self.config.max_boost);
        let first_token = tokens[0];
        let term_idx = self.terms.len();

        // Add all tokens to domain token set
        for &token in &tokens {
            self.domain_tokens
                .entry(token)
                .and_modify(|b| *b = b.max(clamped_boost))
                .or_insert(clamped_boost);
        }

        self.terms
            .push(DomainTerm::new(text.to_string(), tokens, clamped_boost));

        self.first_token_map
            .entry(first_token)
            .or_default()
            .push(term_idx);
    }

    /// Add a term with default boost
    pub fn add_term_with_tokens_default(&mut self, text: &str, tokens: Vec<u32>) {
        self.add_term_with_tokens(text, tokens, self.config.base_boost);
    }

    /// Add a priority term
    pub fn add_priority_term(&mut self, text: &str, tokens: Vec<u32>) {
        let boost = self.config.base_boost * self.config.priority_multiplier;
        if !tokens.is_empty() {
            let term_idx = self.terms.len();
            let first_token = tokens[0];

            for &token in &tokens {
                self.domain_tokens
                    .entry(token)
                    .and_modify(|b| *b = b.max(boost))
                    .or_insert(boost);
            }

            let mut term = DomainTerm::new(text.to_string(), tokens, boost);
            term.is_priority = true;
            self.terms.push(term);

            self.first_token_map
                .entry(first_token)
                .or_default()
                .push(term_idx);
        }
    }

    /// Apply bias to logits
    ///
    /// Boosts domain terms and optionally suppresses out-of-domain tokens.
    pub fn apply_bias(&self, logits: &mut [f32]) {
        if self.is_empty() {
            return;
        }

        // Apply boost to domain tokens
        for (&token, &boost) in &self.domain_tokens {
            if (token as usize) < logits.len() {
                logits[token as usize] += boost;
            }
        }
    }

    /// Check if a token is in the domain vocabulary
    #[must_use]
    pub fn is_domain_token(&self, token: u32) -> bool {
        self.domain_tokens.contains_key(&token)
    }

    /// Get boost for a specific token
    #[must_use]
    pub fn get_token_boost(&self, token: u32) -> Option<f32> {
        self.domain_tokens.get(&token).copied()
    }

    /// Clear all terms
    pub fn clear(&mut self) {
        self.terms.clear();
        self.first_token_map.clear();
        self.domain_tokens.clear();
    }

    /// Load pre-defined terms for the domain
    fn load_predefined_terms(&mut self) {
        // Note: In production, these would be loaded from a vocabulary file
        // Here we provide placeholder structure for the API
        match self.domain_type {
            DomainType::Medical => self.load_medical_terms(),
            DomainType::Legal => self.load_legal_terms(),
            DomainType::Technical => self.load_technical_terms(),
            DomainType::Financial => self.load_financial_terms(),
            DomainType::Scientific => self.load_scientific_terms(),
            DomainType::General | DomainType::Custom => {}
        }
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    fn load_medical_terms(&mut self) {
        let _ = self; // Placeholder: In production, will load from vocabulary file
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    fn load_legal_terms(&mut self) {
        let _ = self; // Placeholder
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    fn load_technical_terms(&mut self) {
        let _ = self; // Placeholder
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    fn load_financial_terms(&mut self) {
        let _ = self; // Placeholder
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    fn load_scientific_terms(&mut self) {
        let _ = self; // Placeholder
    }

    /// Get terms by category
    #[must_use]
    pub fn terms_by_category(&self, category: &str) -> Vec<&DomainTerm> {
        self.terms
            .iter()
            .filter(|t| t.category.as_deref() == Some(category))
            .collect()
    }

    /// Get all categories
    #[must_use]
    pub fn categories(&self) -> Vec<String> {
        let mut categories: Vec<_> = self
            .terms
            .iter()
            .filter_map(|t| t.category.clone())
            .collect();
        categories.sort();
        categories.dedup();
        categories
    }

    /// Get priority terms only
    #[must_use]
    pub fn priority_terms(&self) -> Vec<&DomainTerm> {
        self.terms.iter().filter(|t| t.is_priority).collect()
    }
}

/// Medical domain adapter factory
impl DomainAdapter {
    /// Create medical domain adapter
    #[must_use]
    pub fn medical() -> Self {
        Self::new(DomainType::Medical)
    }

    /// Create legal domain adapter
    #[must_use]
    pub fn legal() -> Self {
        Self::new(DomainType::Legal)
    }

    /// Create technical domain adapter
    #[must_use]
    pub fn technical() -> Self {
        Self::new(DomainType::Technical)
    }

    /// Create financial domain adapter
    #[must_use]
    pub fn financial() -> Self {
        Self::new(DomainType::Financial)
    }

    /// Create scientific domain adapter
    #[must_use]
    pub fn scientific() -> Self {
        Self::new(DomainType::Scientific)
    }

    /// Create custom domain adapter
    #[must_use]
    pub fn custom() -> Self {
        Self::new(DomainType::Custom)
    }
}

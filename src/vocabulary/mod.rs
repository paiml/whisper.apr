//! Custom vocabulary fine-tuning module (WAPR-170 to WAPR-173)
//!
//! Provides hotword boosting, custom vocabulary tries, and domain adaptation.
//!
//! # Overview
//!
//! This module implements:
//! 1. Hotword boosting via logit biasing
//! 2. Custom vocabulary trie for efficient lookup
//! 3. Domain vocabulary adapter for specialized recognition
//!
//! # Usage
//!
//! ```rust,ignore
//! use whisper_apr::vocabulary::{HotwordBooster, VocabularyTrie, DomainAdapter};
//!
//! // Hotword boosting
//! let mut booster = HotwordBooster::new();
//! booster.add_hotword("Anthropic", 2.0);
//! let biased_logits = booster.apply(&logits, &tokenizer);
//!
//! // Domain vocabulary
//! let adapter = DomainAdapter::medical();
//! let adapted_logits = adapter.apply(&logits);
//! ```

pub mod adapter;
pub mod hotwords;
pub mod trie;

pub use adapter::{DomainAdapter, DomainConfig, DomainTerm, DomainType};
pub use hotwords::{Hotword, HotwordBooster, HotwordConfig};
pub use trie::{TrieNode, VocabularyTrie, TrieSearchResult};


/// Combined vocabulary customization for inference
#[derive(Debug, Clone)]
pub struct VocabularyCustomizer {
    /// Hotword booster for specific terms
    hotword_booster: Option<HotwordBooster>,
    /// Domain adapter for specialized vocabulary
    domain_adapter: Option<DomainAdapter>,
    /// Custom vocabulary trie
    vocabulary_trie: Option<VocabularyTrie>,
}

impl VocabularyCustomizer {
    /// Create a new vocabulary customizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            hotword_booster: None,
            domain_adapter: None,
            vocabulary_trie: None,
        }
    }

    /// Add hotword booster
    #[must_use]
    pub fn with_hotword_booster(mut self, booster: HotwordBooster) -> Self {
        self.hotword_booster = Some(booster);
        self
    }

    /// Add domain adapter
    #[must_use]
    pub fn with_domain_adapter(mut self, adapter: DomainAdapter) -> Self {
        self.domain_adapter = Some(adapter);
        self
    }

    /// Add vocabulary trie
    #[must_use]
    pub fn with_vocabulary_trie(mut self, trie: VocabularyTrie) -> Self {
        self.vocabulary_trie = Some(trie);
        self
    }

    /// Apply all vocabulary customizations to logits
    ///
    /// # Arguments
    /// * `logits` - Raw logits from decoder
    /// * `context_tokens` - Previous tokens for context-aware boosting
    ///
    /// # Returns
    /// Modified logits with all customizations applied
    pub fn apply(&self, logits: &mut [f32], context_tokens: &[u32]) {
        // Apply hotword boosting first (token-level biasing)
        if let Some(ref booster) = self.hotword_booster {
            booster.apply_bias(logits, context_tokens);
        }

        // Apply domain adaptation (vocabulary-level adjustments)
        if let Some(ref adapter) = self.domain_adapter {
            adapter.apply_bias(logits);
        }

        // Apply trie-based constraints if vocabulary trie is set
        if let Some(ref trie) = self.vocabulary_trie {
            trie.apply_prefix_boost(logits, context_tokens);
        }
    }

    /// Check if any customization is active
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.hotword_booster.is_some()
            || self.domain_adapter.is_some()
            || self.vocabulary_trie.is_some()
    }

    /// Get hotword booster reference
    #[must_use]
    pub fn hotword_booster(&self) -> Option<&HotwordBooster> {
        self.hotword_booster.as_ref()
    }

    /// Get domain adapter reference
    #[must_use]
    pub fn domain_adapter(&self) -> Option<&DomainAdapter> {
        self.domain_adapter.as_ref()
    }

    /// Get vocabulary trie reference
    #[must_use]
    pub fn vocabulary_trie(&self) -> Option<&VocabularyTrie> {
        self.vocabulary_trie.as_ref()
    }
}

impl Default for VocabularyCustomizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_customizer_new() {
        let customizer = VocabularyCustomizer::new();
        assert!(!customizer.is_active());
        assert!(customizer.hotword_booster().is_none());
        assert!(customizer.domain_adapter().is_none());
        assert!(customizer.vocabulary_trie().is_none());
    }

    #[test]
    fn test_vocabulary_customizer_default() {
        let customizer = VocabularyCustomizer::default();
        assert!(!customizer.is_active());
    }

    #[test]
    fn test_vocabulary_customizer_with_hotword_booster() {
        let booster = HotwordBooster::new();
        let customizer = VocabularyCustomizer::new().with_hotword_booster(booster);
        assert!(customizer.is_active());
        assert!(customizer.hotword_booster().is_some());
    }

    #[test]
    fn test_vocabulary_customizer_with_domain_adapter() {
        let adapter = DomainAdapter::new(DomainType::General);
        let customizer = VocabularyCustomizer::new().with_domain_adapter(adapter);
        assert!(customizer.is_active());
        assert!(customizer.domain_adapter().is_some());
    }

    #[test]
    fn test_vocabulary_customizer_with_vocabulary_trie() {
        let trie = VocabularyTrie::new();
        let customizer = VocabularyCustomizer::new().with_vocabulary_trie(trie);
        assert!(customizer.is_active());
        assert!(customizer.vocabulary_trie().is_some());
    }

    #[test]
    fn test_vocabulary_customizer_apply_empty() {
        let customizer = VocabularyCustomizer::new();
        let mut logits = vec![0.0, 1.0, 2.0, 3.0];
        let context = vec![];

        customizer.apply(&mut logits, &context);

        // Logits should be unchanged when no customization is active
        assert!((logits[0] - 0.0).abs() < f32::EPSILON);
        assert!((logits[1] - 1.0).abs() < f32::EPSILON);
        assert!((logits[2] - 2.0).abs() < f32::EPSILON);
        assert!((logits[3] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vocabulary_customizer_chained_builders() {
        let booster = HotwordBooster::new();
        let adapter = DomainAdapter::new(DomainType::General);
        let trie = VocabularyTrie::new();

        let customizer = VocabularyCustomizer::new()
            .with_hotword_booster(booster)
            .with_domain_adapter(adapter)
            .with_vocabulary_trie(trie);

        assert!(customizer.is_active());
        assert!(customizer.hotword_booster().is_some());
        assert!(customizer.domain_adapter().is_some());
        assert!(customizer.vocabulary_trie().is_some());
    }
}

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

mod customizer;
mod domain;
mod hotword;
mod trie;

pub use customizer::VocabularyCustomizerWasm;
pub use domain::{DomainAdapterWasm, DomainConfigWasm, DomainTermWasm, DomainTypeWasm};
pub use hotword::{HotwordBoosterWasm, HotwordConfigWasm, HotwordWasm};
pub use trie::{TrieSearchResultWasm, VocabularyTrieWasm};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocabulary::TrieSearchResult;

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

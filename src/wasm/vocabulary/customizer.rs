//! WASM vocabulary customizer bindings

use wasm_bindgen::prelude::*;

use crate::vocabulary::VocabularyCustomizer;

use super::{DomainAdapterWasm, HotwordBoosterWasm, VocabularyTrieWasm};

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
        self.inner = VocabularyCustomizer::new().with_hotword_booster(booster.into_inner());
    }

    /// Set domain adapter
    #[wasm_bindgen(js_name = setDomainAdapter)]
    pub fn set_domain_adapter(&mut self, adapter: DomainAdapterWasm) {
        self.inner = VocabularyCustomizer::new().with_domain_adapter(adapter.into_inner());
    }

    /// Set vocabulary trie
    #[wasm_bindgen(js_name = setVocabularyTrie)]
    pub fn set_vocabulary_trie(&mut self, trie: VocabularyTrieWasm) {
        self.inner = VocabularyCustomizer::new().with_vocabulary_trie(trie.into_inner());
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

//! WASM vocabulary trie bindings

use wasm_bindgen::prelude::*;

use crate::vocabulary::{TrieSearchResult, VocabularyTrie};

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

impl VocabularyTrieWasm {
    /// Convert to inner type (consumes self)
    pub(super) fn into_inner(self) -> VocabularyTrie {
        self.inner
    }
}

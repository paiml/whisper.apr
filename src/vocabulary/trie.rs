//! Custom vocabulary trie structure (WAPR-171)
//!
//! Efficient prefix-based vocabulary lookup using a trie data structure.
//!
//! # Overview
//!
//! The vocabulary trie provides:
//! 1. O(k) prefix lookups where k is prefix length
//! 2. Memory-efficient storage of token sequences
//! 3. Fast completion suggestions during decoding
//!
//! # Algorithm
//!
//! The trie stores token sequences, where each path from root to a terminal
//! node represents a valid vocabulary entry. During decoding, we traverse
//! the trie based on context tokens to find valid continuations.
//!
//! # Example
//!
//! ```rust,ignore
//! use whisper_apr::vocabulary::VocabularyTrie;
//!
//! let mut trie = VocabularyTrie::new();
//! trie.insert(&[100, 200, 300], "hello", 1.0);
//! trie.insert(&[100, 250], "help", 1.0);
//!
//! // Find valid continuations after [100]
//! let next_tokens = trie.get_continuations(&[100]);
//! // Returns [200, 250] with their boost values
//! ```

use std::collections::HashMap;

/// A node in the vocabulary trie
#[derive(Debug, Clone)]
pub struct TrieNode {
    /// Children nodes indexed by token ID
    children: HashMap<u32, Self>,
    /// Whether this node represents end of a vocabulary entry
    is_terminal: bool,
    /// Boost value if terminal (affects logit biasing)
    boost: f32,
    /// Original text if terminal
    text: Option<String>,
    /// Depth of this node in the trie
    depth: usize,
}

impl TrieNode {
    /// Create a new non-terminal trie node
    #[must_use]
    pub fn new(depth: usize) -> Self {
        Self {
            children: HashMap::new(),
            is_terminal: false,
            boost: 0.0,
            text: None,
            depth,
        }
    }

    /// Check if this is a terminal node
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    /// Get boost value
    #[must_use]
    pub fn boost(&self) -> f32 {
        self.boost
    }

    /// Get associated text
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// Get depth in trie
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Check if node has children
    #[must_use]
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Get number of children
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Get child for token
    #[must_use]
    pub fn get_child(&self, token: u32) -> Option<&Self> {
        self.children.get(&token)
    }

    /// Get mutable child for token
    pub fn get_child_mut(&mut self, token: u32) -> Option<&mut Self> {
        self.children.get_mut(&token)
    }

    /// Get all child tokens
    #[must_use]
    pub fn child_tokens(&self) -> Vec<u32> {
        self.children.keys().copied().collect()
    }

    /// Insert or get child node
    pub fn get_or_create_child(&mut self, token: u32) -> &mut Self {
        let next_depth = self.depth + 1;
        self.children
            .entry(token)
            .or_insert_with(|| Self::new(next_depth))
    }

    /// Mark as terminal with boost value
    pub fn set_terminal(&mut self, text: String, boost: f32) {
        self.is_terminal = true;
        self.text = Some(text);
        self.boost = boost;
    }
}

impl Default for TrieNode {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Search result from trie lookup
#[derive(Debug, Clone)]
pub struct TrieSearchResult {
    /// Valid continuation tokens with their boost values
    pub continuations: Vec<(u32, f32)>,
    /// Whether current position is a complete entry
    pub is_complete: bool,
    /// Text if complete
    pub text: Option<String>,
    /// Depth reached in trie
    pub depth: usize,
    /// Total entries that start with this prefix
    pub matching_entries: usize,
}

impl TrieSearchResult {
    /// Create an empty result
    #[must_use]
    pub fn empty() -> Self {
        Self {
            continuations: Vec::new(),
            is_complete: false,
            text: None,
            depth: 0,
            matching_entries: 0,
        }
    }

    /// Check if search found any matches
    #[must_use]
    pub fn has_matches(&self) -> bool {
        !self.continuations.is_empty() || self.is_complete
    }
}

/// Vocabulary trie for efficient prefix-based lookup
#[derive(Debug, Clone)]
pub struct VocabularyTrie {
    /// Root node of the trie
    root: TrieNode,
    /// Total number of entries
    entry_count: usize,
    /// Default boost for entries without explicit boost
    default_boost: f32,
    /// Prefix boost factor (how much to boost partial matches)
    prefix_boost_factor: f32,
}

impl VocabularyTrie {
    /// Create a new vocabulary trie
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(0),
            entry_count: 0,
            default_boost: 0.5,
            prefix_boost_factor: 0.8,
        }
    }

    /// Create trie with custom default boost
    #[must_use]
    pub fn with_default_boost(mut self, boost: f32) -> Self {
        self.default_boost = boost;
        self
    }

    /// Set prefix boost factor
    #[must_use]
    pub fn with_prefix_boost_factor(mut self, factor: f32) -> Self {
        self.prefix_boost_factor = factor;
        self
    }

    /// Get number of entries in trie
    #[must_use]
    pub fn len(&self) -> usize {
        self.entry_count
    }

    /// Check if trie is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Insert a token sequence into the trie
    ///
    /// # Arguments
    /// * `tokens` - Token sequence representing the vocabulary entry
    /// * `text` - Original text
    /// * `boost` - Boost value for this entry
    pub fn insert(&mut self, tokens: &[u32], text: &str, boost: f32) {
        if tokens.is_empty() {
            return;
        }

        let mut node = &mut self.root;
        for &token in tokens {
            node = node.get_or_create_child(token);
        }

        // Mark terminal
        if !node.is_terminal() {
            self.entry_count += 1;
        }
        node.set_terminal(text.to_string(), boost);
    }

    /// Insert with default boost
    pub fn insert_default(&mut self, tokens: &[u32], text: &str) {
        self.insert(tokens, text, self.default_boost);
    }

    /// Check if a token sequence exists in the trie
    #[must_use]
    pub fn contains(&self, tokens: &[u32]) -> bool {
        self.get_node(tokens).is_some_and(|n| n.is_terminal())
    }

    /// Check if any entry starts with the given prefix
    #[must_use]
    pub fn has_prefix(&self, prefix: &[u32]) -> bool {
        self.get_node(prefix).is_some()
    }

    /// Get node at given path
    fn get_node(&self, tokens: &[u32]) -> Option<&TrieNode> {
        let mut node = &self.root;
        for &token in tokens {
            node = node.get_child(token)?;
        }
        Some(node)
    }

    /// Search for continuations from a given prefix
    #[must_use]
    #[allow(clippy::option_if_let_else)]
    pub fn search(&self, prefix: &[u32]) -> TrieSearchResult {
        match self.get_node(prefix) {
            Some(node) => {
                let continuations: Vec<(u32, f32)> = node
                    .children
                    .iter()
                    .map(|(&token, child)| {
                        // Use child's boost if terminal, otherwise use prefix factor
                        let boost = if child.is_terminal() {
                            child.boost()
                        } else {
                            self.default_boost * self.prefix_boost_factor
                        };
                        (token, boost)
                    })
                    .collect();

                let matching_entries = Self::count_entries_under(node);

                TrieSearchResult {
                    continuations,
                    is_complete: node.is_terminal(),
                    text: node.text().map(String::from),
                    depth: node.depth(),
                    matching_entries,
                }
            }
            None => TrieSearchResult::empty(),
        }
    }

    /// Count total entries under a node
    fn count_entries_under(node: &TrieNode) -> usize {
        let mut count = usize::from(node.is_terminal());
        for child in node.children.values() {
            count += Self::count_entries_under(child);
        }
        count
    }

    /// Get all valid continuation tokens from prefix
    #[must_use]
    pub fn get_continuations(&self, prefix: &[u32]) -> Vec<(u32, f32)> {
        self.search(prefix).continuations
    }

    /// Apply prefix boost to logits based on context
    ///
    /// This method boosts tokens that would continue valid vocabulary entries.
    pub fn apply_prefix_boost(&self, logits: &mut [f32], context: &[u32]) {
        if self.is_empty() {
            return;
        }

        let result = self.search(context);
        for (token, boost) in result.continuations {
            if (token as usize) < logits.len() {
                logits[token as usize] += boost;
            }
        }
    }

    /// Get all entries in the trie (for debugging)
    #[must_use]
    pub fn all_entries(&self) -> Vec<(Vec<u32>, String, f32)> {
        let mut entries = Vec::new();
        Self::collect_entries(&self.root, Vec::new(), &mut entries);
        entries
    }

    fn collect_entries(
        node: &TrieNode,
        path: Vec<u32>,
        entries: &mut Vec<(Vec<u32>, String, f32)>,
    ) {
        if node.is_terminal() {
            if let Some(text) = node.text() {
                entries.push((path.clone(), text.to_string(), node.boost()));
            }
        }

        for (&token, child) in &node.children {
            let mut new_path = path.clone();
            new_path.push(token);
            Self::collect_entries(child, new_path, entries);
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.root = TrieNode::new(0);
        self.entry_count = 0;
    }
}

impl Default for VocabularyTrie {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // TrieNode Tests
    // ============================================================

    #[test]
    fn test_trie_node_new() {
        let node = TrieNode::new(3);
        assert_eq!(node.depth(), 3);
        assert!(!node.is_terminal());
        assert!(!node.has_children());
        assert!((node.boost() - 0.0).abs() < f32::EPSILON);
        assert!(node.text().is_none());
    }

    #[test]
    fn test_trie_node_default() {
        let node = TrieNode::default();
        assert_eq!(node.depth(), 0);
    }

    #[test]
    fn test_trie_node_set_terminal() {
        let mut node = TrieNode::new(0);
        node.set_terminal("hello".to_string(), 1.5);

        assert!(node.is_terminal());
        assert!((node.boost() - 1.5).abs() < f32::EPSILON);
        assert_eq!(node.text(), Some("hello"));
    }

    #[test]
    fn test_trie_node_get_or_create_child() {
        let mut node = TrieNode::new(0);
        assert!(!node.has_children());

        let child = node.get_or_create_child(100);
        assert_eq!(child.depth(), 1);
        assert!(node.has_children());
        assert_eq!(node.child_count(), 1);
    }

    #[test]
    fn test_trie_node_get_child() {
        let mut node = TrieNode::new(0);
        node.get_or_create_child(100);

        assert!(node.get_child(100).is_some());
        assert!(node.get_child(200).is_none());
    }

    #[test]
    fn test_trie_node_child_tokens() {
        let mut node = TrieNode::new(0);
        node.get_or_create_child(100);
        node.get_or_create_child(200);

        let tokens = node.child_tokens();
        assert_eq!(tokens.len(), 2);
        assert!(tokens.contains(&100));
        assert!(tokens.contains(&200));
    }

    // ============================================================
    // TrieSearchResult Tests
    // ============================================================

    #[test]
    fn test_trie_search_result_empty() {
        let result = TrieSearchResult::empty();
        assert!(!result.has_matches());
        assert!(!result.is_complete);
        assert!(result.continuations.is_empty());
        assert_eq!(result.depth, 0);
    }

    #[test]
    fn test_trie_search_result_has_matches() {
        let mut result = TrieSearchResult::empty();
        assert!(!result.has_matches());

        result.continuations.push((100, 1.0));
        assert!(result.has_matches());
    }

    #[test]
    fn test_trie_search_result_complete() {
        let mut result = TrieSearchResult::empty();
        result.is_complete = true;
        assert!(result.has_matches());
    }

    // ============================================================
    // VocabularyTrie Tests
    // ============================================================

    #[test]
    fn test_vocabulary_trie_new() {
        let trie = VocabularyTrie::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn test_vocabulary_trie_default() {
        let trie = VocabularyTrie::default();
        assert!(trie.is_empty());
    }

    #[test]
    fn test_vocabulary_trie_with_default_boost() {
        let trie = VocabularyTrie::new().with_default_boost(2.0);
        assert!((trie.default_boost - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vocabulary_trie_with_prefix_boost_factor() {
        let trie = VocabularyTrie::new().with_prefix_boost_factor(0.5);
        assert!((trie.prefix_boost_factor - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vocabulary_trie_insert() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100, 200, 300], "hello", 1.5);

        assert!(!trie.is_empty());
        assert_eq!(trie.len(), 1);
    }

    #[test]
    fn test_vocabulary_trie_insert_empty() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[], "empty", 1.0);

        assert!(trie.is_empty());
    }

    #[test]
    fn test_vocabulary_trie_insert_default() {
        let mut trie = VocabularyTrie::new().with_default_boost(2.0);
        trie.insert_default(&[100], "test");

        let entries = trie.all_entries();
        assert_eq!(entries.len(), 1);
        assert!((entries[0].2 - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vocabulary_trie_insert_multiple() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100, 200], "hello", 1.0);
        trie.insert(&[100, 250], "help", 1.0);
        trie.insert(&[300], "world", 1.0);

        assert_eq!(trie.len(), 3);
    }

    #[test]
    fn test_vocabulary_trie_insert_duplicate() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100], "first", 1.0);
        trie.insert(&[100], "second", 2.0);

        // Count should not increase for duplicate
        assert_eq!(trie.len(), 1);

        // But value should be updated
        let entries = trie.all_entries();
        assert_eq!(entries[0].1, "second");
    }

    #[test]
    fn test_vocabulary_trie_contains() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100, 200], "hello", 1.0);

        assert!(trie.contains(&[100, 200]));
        assert!(!trie.contains(&[100])); // Prefix only, not terminal
        assert!(!trie.contains(&[100, 200, 300])); // Extends beyond entry
        assert!(!trie.contains(&[999])); // Doesn't exist
    }

    #[test]
    fn test_vocabulary_trie_has_prefix() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100, 200, 300], "hello", 1.0);

        assert!(trie.has_prefix(&[100]));
        assert!(trie.has_prefix(&[100, 200]));
        assert!(trie.has_prefix(&[100, 200, 300]));
        assert!(!trie.has_prefix(&[100, 200, 300, 400]));
        assert!(!trie.has_prefix(&[999]));
    }

    #[test]
    fn test_vocabulary_trie_search_empty_prefix() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100], "a", 1.0);
        trie.insert(&[200], "b", 2.0);

        let result = trie.search(&[]);
        assert_eq!(result.continuations.len(), 2);
        assert!(!result.is_complete);
    }

    #[test]
    fn test_vocabulary_trie_search_partial() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100, 200], "hello", 1.0);
        trie.insert(&[100, 250], "help", 1.5);

        let result = trie.search(&[100]);
        assert_eq!(result.continuations.len(), 2);
        assert!(!result.is_complete);

        // Check that both continuations are present
        let tokens: Vec<u32> = result.continuations.iter().map(|(t, _)| *t).collect();
        assert!(tokens.contains(&200));
        assert!(tokens.contains(&250));
    }

    #[test]
    fn test_vocabulary_trie_search_complete() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100, 200], "hello", 1.0);

        let result = trie.search(&[100, 200]);
        assert!(result.is_complete);
        assert_eq!(result.text, Some("hello".to_string()));
        assert_eq!(result.depth, 2);
    }

    #[test]
    fn test_vocabulary_trie_search_no_match() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100], "test", 1.0);

        let result = trie.search(&[999]);
        assert!(!result.has_matches());
        assert_eq!(result.matching_entries, 0);
    }

    #[test]
    fn test_vocabulary_trie_get_continuations() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100, 200], "a", 1.0);
        trie.insert(&[100, 300], "b", 2.0);

        let continuations = trie.get_continuations(&[100]);
        assert_eq!(continuations.len(), 2);
    }

    #[test]
    fn test_vocabulary_trie_apply_prefix_boost() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[50, 100], "test", 2.0);

        let mut logits = vec![0.0; 200];
        trie.apply_prefix_boost(&mut logits, &[50]);

        // Token 100 should be boosted
        assert!(logits[100] > 0.0);
    }

    #[test]
    fn test_vocabulary_trie_apply_prefix_boost_empty() {
        let trie = VocabularyTrie::new();
        let mut logits = vec![1.0; 100];

        trie.apply_prefix_boost(&mut logits, &[50]);

        // Logits should be unchanged
        for &logit in &logits {
            assert!((logit - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_vocabulary_trie_apply_prefix_boost_out_of_bounds() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[50, 1000], "test", 2.0); // Token 1000 is out of bounds

        let mut logits = vec![0.0; 100];
        trie.apply_prefix_boost(&mut logits, &[50]);

        // Should not panic, logits unchanged for out of bounds
        assert!((logits[50] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vocabulary_trie_all_entries() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100], "a", 1.0);
        trie.insert(&[200, 300], "b", 2.0);

        let entries = trie.all_entries();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_vocabulary_trie_clear() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100], "a", 1.0);
        trie.insert(&[200], "b", 2.0);

        assert_eq!(trie.len(), 2);
        trie.clear();
        assert!(trie.is_empty());
    }

    #[test]
    fn test_vocabulary_trie_matching_entries_count() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[100, 200], "a", 1.0);
        trie.insert(&[100, 300], "b", 1.0);
        trie.insert(&[100, 200, 400], "c", 1.0);

        // All three start with [100]
        let result = trie.search(&[100]);
        assert_eq!(result.matching_entries, 3);

        // Only two start with [100, 200]
        let result = trie.search(&[100, 200]);
        // [100, 200] is terminal, and [100, 200, 400] is under it
        assert_eq!(result.matching_entries, 2);
    }

    #[test]
    fn test_vocabulary_trie_deep_nesting() {
        let mut trie = VocabularyTrie::new();
        let tokens: Vec<u32> = (0..100).collect();
        trie.insert(&tokens, "deep", 1.0);

        assert!(trie.contains(&tokens));
        assert!(trie.has_prefix(&tokens[..50]));
    }

    #[test]
    fn test_vocabulary_trie_shared_prefix() {
        let mut trie = VocabularyTrie::new();
        trie.insert(&[1, 2, 3], "abc", 1.0);
        trie.insert(&[1, 2, 4], "abd", 1.0);
        trie.insert(&[1, 2, 5], "abe", 1.0);

        // All share prefix [1, 2]
        let result = trie.search(&[1, 2]);
        assert_eq!(result.continuations.len(), 3);
        assert_eq!(result.matching_entries, 3);
    }
}

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

#[cfg(test)]
mod tests;

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

    /// Insert a token sequence into the trie.
    ///
    /// Walks from root to leaf, creating intermediate nodes as needed.
    /// If the token sequence already exists, its text and boost are updated.
    /// Empty token sequences are silently ignored (no-op).
    ///
    /// # Arguments
    /// * `tokens` - Token sequence representing the vocabulary entry
    /// * `text` - Original text associated with this entry
    /// * `boost` - Log-probability boost applied during constrained decoding
    pub fn insert(&mut self, tokens: &[u32], text: &str, boost: f32) {
        if tokens.is_empty() {
            return;
        }

        let mut node = &mut self.root;
        for &token in tokens {
            node = node.get_or_create_child(token);
        }

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
        Self::collect_entries(&self.root, &[], &mut entries);
        entries
    }

    fn collect_entries(node: &TrieNode, path: &[u32], entries: &mut Vec<(Vec<u32>, String, f32)>) {
        if node.is_terminal() {
            if let Some(text) = node.text() {
                entries.push((path.to_vec(), text.to_string(), node.boost()));
            }
        }

        for (&token, child) in &node.children {
            let mut new_path = path.to_vec();
            new_path.push(token);
            Self::collect_entries(child, &new_path, entries);
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

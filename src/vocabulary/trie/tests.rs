//! Tests for vocabulary trie

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

#[test]
fn test_trie_node_get_child_mut() {
    let mut node = TrieNode::new(0);
    node.get_or_create_child(42);
    let child = node.get_child_mut(42);
    assert!(child.is_some());
    child.unwrap().set_terminal("hello".to_string(), 1.5);
    assert!(node.get_child(42).unwrap().is_terminal());
    assert!((node.get_child(42).unwrap().boost() - 1.5).abs() < f32::EPSILON);

    assert!(node.get_child_mut(99).is_none());
}

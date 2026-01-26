//! Tests for domain vocabulary adapter

#![allow(clippy::unwrap_used)]

use super::*;

// ============================================================
// DomainType Tests
// ============================================================

#[test]
fn test_domain_type_name() {
    assert_eq!(DomainType::General.name(), "General");
    assert_eq!(DomainType::Medical.name(), "Medical");
    assert_eq!(DomainType::Legal.name(), "Legal");
    assert_eq!(DomainType::Technical.name(), "Technical");
    assert_eq!(DomainType::Financial.name(), "Financial");
    assert_eq!(DomainType::Scientific.name(), "Scientific");
    assert_eq!(DomainType::Custom.name(), "Custom");
}

#[test]
fn test_domain_type_has_predefined_terms() {
    assert!(!DomainType::General.has_predefined_terms());
    assert!(DomainType::Medical.has_predefined_terms());
    assert!(DomainType::Legal.has_predefined_terms());
    assert!(DomainType::Technical.has_predefined_terms());
    assert!(DomainType::Financial.has_predefined_terms());
    assert!(DomainType::Scientific.has_predefined_terms());
    assert!(!DomainType::Custom.has_predefined_terms());
}

// ============================================================
// DomainConfig Tests
// ============================================================

#[test]
fn test_domain_config_new() {
    let config = DomainConfig::new();
    assert!((config.base_boost - 1.0).abs() < f32::EPSILON);
    assert!((config.priority_multiplier - 1.5).abs() < f32::EPSILON);
    assert!((config.max_boost - 5.0).abs() < f32::EPSILON);
    assert!(!config.suppress_out_of_domain);
}

#[test]
fn test_domain_config_default() {
    let config = DomainConfig::default();
    assert!((config.base_boost - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_domain_config_with_base_boost() {
    let config = DomainConfig::new().with_base_boost(2.0);
    assert!((config.base_boost - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_domain_config_with_priority_multiplier() {
    let config = DomainConfig::new().with_priority_multiplier(2.0);
    assert!((config.priority_multiplier - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_domain_config_with_max_boost() {
    let config = DomainConfig::new().with_max_boost(10.0);
    assert!((config.max_boost - 10.0).abs() < f32::EPSILON);
}

#[test]
fn test_domain_config_with_suppression() {
    let config = DomainConfig::new().with_suppression(0.3);
    assert!(config.suppress_out_of_domain);
    assert!((config.suppression_factor - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_domain_config_builder_chain() {
    let config = DomainConfig::new()
        .with_base_boost(2.0)
        .with_priority_multiplier(3.0)
        .with_max_boost(15.0)
        .with_suppression(0.2);

    assert!((config.base_boost - 2.0).abs() < f32::EPSILON);
    assert!((config.priority_multiplier - 3.0).abs() < f32::EPSILON);
    assert!((config.max_boost - 15.0).abs() < f32::EPSILON);
    assert!(config.suppress_out_of_domain);
    assert!((config.suppression_factor - 0.2).abs() < f32::EPSILON);
}

// ============================================================
// DomainTerm Tests
// ============================================================

#[test]
fn test_domain_term_new() {
    let term = DomainTerm::new("test".to_string(), vec![100, 200], 1.5);
    assert_eq!(term.text, "test");
    assert_eq!(term.tokens, vec![100, 200]);
    assert!((term.boost - 1.5).abs() < f32::EPSILON);
    assert!(!term.is_priority);
    assert!(term.category.is_none());
}

#[test]
fn test_domain_term_with_priority() {
    let term = DomainTerm::new("test".to_string(), vec![100], 1.0).with_priority();
    assert!(term.is_priority);
}

#[test]
fn test_domain_term_with_category() {
    let term = DomainTerm::new("test".to_string(), vec![100], 1.0).with_category("anatomy");
    assert_eq!(term.category, Some("anatomy".to_string()));
}

#[test]
fn test_domain_term_first_token() {
    let term = DomainTerm::new("test".to_string(), vec![100, 200, 300], 1.0);
    assert_eq!(term.first_token(), Some(100));

    let empty = DomainTerm::new("empty".to_string(), vec![], 1.0);
    assert_eq!(empty.first_token(), None);
}

// ============================================================
// DomainAdapter Tests
// ============================================================

#[test]
fn test_domain_adapter_new() {
    let adapter = DomainAdapter::new(DomainType::General);
    assert_eq!(adapter.domain_type(), DomainType::General);
    assert!(adapter.is_empty());
}

#[test]
fn test_domain_adapter_with_config() {
    let config = DomainConfig::new().with_base_boost(2.0);
    let adapter = DomainAdapter::with_config(DomainType::Custom, config);
    assert!((adapter.config().base_boost - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_domain_adapter_factory_methods() {
    assert_eq!(DomainAdapter::medical().domain_type(), DomainType::Medical);
    assert_eq!(DomainAdapter::legal().domain_type(), DomainType::Legal);
    assert_eq!(
        DomainAdapter::technical().domain_type(),
        DomainType::Technical
    );
    assert_eq!(
        DomainAdapter::financial().domain_type(),
        DomainType::Financial
    );
    assert_eq!(
        DomainAdapter::scientific().domain_type(),
        DomainType::Scientific
    );
    assert_eq!(DomainAdapter::custom().domain_type(), DomainType::Custom);
}

#[test]
fn test_domain_adapter_add_term() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);
    adapter.add_term_with_tokens("test", vec![100, 200], 1.5);

    assert_eq!(adapter.len(), 1);
    assert!(!adapter.is_empty());
    assert!(adapter.is_domain_token(100));
    assert!(adapter.is_domain_token(200));
}

#[test]
fn test_domain_adapter_add_term_empty() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);
    adapter.add_term_with_tokens("empty", vec![], 1.0);

    assert!(adapter.is_empty());
}

#[test]
fn test_domain_adapter_add_term_clamps() {
    let config = DomainConfig::new().with_max_boost(2.0);
    let mut adapter = DomainAdapter::with_config(DomainType::Custom, config);
    adapter.add_term_with_tokens("test", vec![100], 10.0);

    // Boost should be clamped
    let boost = adapter.get_token_boost(100).unwrap_or(0.0);
    assert!((boost - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_domain_adapter_add_term_default() {
    let config = DomainConfig::new().with_base_boost(2.5);
    let mut adapter = DomainAdapter::with_config(DomainType::Custom, config);
    adapter.add_term_with_tokens_default("test", vec![100]);

    let boost = adapter.get_token_boost(100).unwrap_or(0.0);
    assert!((boost - 2.5).abs() < f32::EPSILON);
}

#[test]
fn test_domain_adapter_add_priority_term() {
    let config = DomainConfig::new()
        .with_base_boost(1.0)
        .with_priority_multiplier(2.0);
    let mut adapter = DomainAdapter::with_config(DomainType::Custom, config);
    adapter.add_priority_term("priority", vec![100]);

    // Boost should be base * multiplier = 2.0
    let boost = adapter.get_token_boost(100).unwrap_or(0.0);
    assert!((boost - 2.0).abs() < f32::EPSILON);

    let priority_terms = adapter.priority_terms();
    assert_eq!(priority_terms.len(), 1);
    assert!(priority_terms[0].is_priority);
}

#[test]
fn test_domain_adapter_apply_bias() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);
    adapter.add_term_with_tokens("test", vec![50], 2.0);

    let mut logits = vec![0.0; 100];
    adapter.apply_bias(&mut logits);

    // Token 50 should be boosted
    assert!((logits[50] - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_domain_adapter_apply_bias_empty() {
    let adapter = DomainAdapter::new(DomainType::Custom);
    let mut logits = vec![1.0; 100];

    adapter.apply_bias(&mut logits);

    // Logits should be unchanged
    for &logit in &logits {
        assert!((logit - 1.0).abs() < f32::EPSILON);
    }
}

#[test]
fn test_domain_adapter_apply_bias_out_of_bounds() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);
    adapter.add_term_with_tokens("test", vec![1000], 2.0);

    let mut logits = vec![0.0; 100];
    adapter.apply_bias(&mut logits);

    // Should not panic
    for &logit in &logits {
        assert!((logit - 0.0).abs() < f32::EPSILON);
    }
}

#[test]
fn test_domain_adapter_is_domain_token() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);
    adapter.add_term_with_tokens("test", vec![100, 200], 1.0);

    assert!(adapter.is_domain_token(100));
    assert!(adapter.is_domain_token(200));
    assert!(!adapter.is_domain_token(300));
}

#[test]
fn test_domain_adapter_get_token_boost() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);
    adapter.add_term_with_tokens("test", vec![100], 1.5);

    assert_eq!(adapter.get_token_boost(100), Some(1.5));
    assert_eq!(adapter.get_token_boost(999), None);
}

#[test]
fn test_domain_adapter_clear() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);
    adapter.add_term_with_tokens("test", vec![100], 1.0);

    assert!(!adapter.is_empty());
    adapter.clear();
    assert!(adapter.is_empty());
    assert!(!adapter.is_domain_token(100));
}

#[test]
fn test_domain_adapter_terms_by_category() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);

    // Add terms with categories
    adapter.add_term_with_tokens("term1", vec![100], 1.0);
    adapter.terms.last_mut().unwrap().category = Some("cat_a".to_string());

    adapter.add_term_with_tokens("term2", vec![200], 1.0);
    adapter.terms.last_mut().unwrap().category = Some("cat_a".to_string());

    adapter.add_term_with_tokens("term3", vec![300], 1.0);
    adapter.terms.last_mut().unwrap().category = Some("cat_b".to_string());

    let cat_a_terms = adapter.terms_by_category("cat_a");
    assert_eq!(cat_a_terms.len(), 2);

    let cat_b_terms = adapter.terms_by_category("cat_b");
    assert_eq!(cat_b_terms.len(), 1);
}

#[test]
fn test_domain_adapter_categories() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);

    adapter.add_term_with_tokens("term1", vec![100], 1.0);
    adapter.terms.last_mut().unwrap().category = Some("cat_b".to_string());

    adapter.add_term_with_tokens("term2", vec![200], 1.0);
    adapter.terms.last_mut().unwrap().category = Some("cat_a".to_string());

    adapter.add_term_with_tokens("term3", vec![300], 1.0);
    adapter.terms.last_mut().unwrap().category = Some("cat_a".to_string());

    let categories = adapter.categories();
    assert_eq!(categories.len(), 2);
    assert_eq!(categories[0], "cat_a");
    assert_eq!(categories[1], "cat_b");
}

#[test]
fn test_domain_adapter_priority_terms() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);
    adapter.add_term_with_tokens("normal", vec![100], 1.0);
    adapter.add_priority_term("priority1", vec![200]);
    adapter.add_priority_term("priority2", vec![300]);

    let priority = adapter.priority_terms();
    assert_eq!(priority.len(), 2);
}

#[test]
fn test_domain_adapter_overlapping_tokens() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);
    adapter.add_term_with_tokens("term1", vec![100, 200], 1.0);
    adapter.add_term_with_tokens("term2", vec![100, 300], 2.0);

    // Token 100 should have the max boost from both terms
    let boost = adapter.get_token_boost(100).unwrap_or(0.0);
    assert!((boost - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_domain_adapter_terms_accessor() {
    let mut adapter = DomainAdapter::new(DomainType::Custom);
    adapter.add_term_with_tokens("test1", vec![100], 1.0);
    adapter.add_term_with_tokens("test2", vec![200], 2.0);

    let terms = adapter.terms();
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0].text, "test1");
    assert_eq!(terms[1].text, "test2");
}

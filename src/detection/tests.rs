//! Tests for language detection

use super::*;
use crate::error::WhisperResult;
use crate::tokenizer::special_tokens;

// =========================================================================
// LanguageProbs Tests
// =========================================================================

#[test]
fn test_language_probs_default() {
    let probs = LanguageProbs::default();
    assert_eq!(probs.top_language(), Some("en"));
    assert!((probs.confidence() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_language_probs_from_logits() {
    // Create fake logits with English having highest probability
    let mut logits = vec![0.0_f32; 60000];

    // Set English (LANG_BASE + 0) higher than others
    logits[special_tokens::LANG_BASE as usize] = 10.0;
    logits[special_tokens::LANG_BASE as usize + 1] = 5.0; // Chinese
    logits[special_tokens::LANG_BASE as usize + 2] = 3.0; // German

    let probs = LanguageProbs::from_logits(&logits);

    assert_eq!(probs.top_language(), Some("en"));
    assert!(probs.confidence() > 0.5);
}

#[test]
fn test_language_probs_from_logits_spanish_top() {
    let mut logits = vec![0.0_f32; 60000];

    // Set Spanish (LANG_BASE + 3) highest
    logits[special_tokens::LANG_BASE as usize + 3] = 10.0; // Spanish
    logits[special_tokens::LANG_BASE as usize] = 2.0; // English

    let probs = LanguageProbs::from_logits(&logits);

    assert_eq!(probs.top_language(), Some("es"));
}

#[test]
fn test_language_probs_top_n() {
    let mut logits = vec![0.0_f32; 60000];

    logits[special_tokens::LANG_BASE as usize] = 10.0; // English
    logits[special_tokens::LANG_BASE as usize + 1] = 8.0; // Chinese
    logits[special_tokens::LANG_BASE as usize + 2] = 6.0; // German

    let probs = LanguageProbs::from_logits(&logits);
    let top3 = probs.top_n(3);

    assert_eq!(top3.len(), 3);
    assert_eq!(top3[0].0, "en");
    assert_eq!(top3[1].0, "zh");
    assert_eq!(top3[2].0, "de");
}

#[test]
fn test_language_probs_probability_for() {
    let mut logits = vec![0.0_f32; 60000];

    logits[special_tokens::LANG_BASE as usize] = 5.0; // English
    logits[special_tokens::LANG_BASE as usize + 3] = 5.0; // Spanish

    let probs = LanguageProbs::from_logits(&logits);

    // Both should have similar probability (roughly 0.5 each among these two)
    let en_prob = probs.probability_for("en").unwrap_or(0.0);
    let es_prob = probs.probability_for("es").unwrap_or(0.0);

    assert!(en_prob > 0.0);
    assert!(es_prob > 0.0);
}

#[test]
fn test_language_probs_is_confident() {
    let probs = LanguageProbs {
        languages: vec!["en".to_string(), "es".to_string()],
        probabilities: vec![0.7, 0.3],
    };

    assert!(probs.is_confident(0.5));
    assert!(probs.is_confident(0.7));
    assert!(!probs.is_confident(0.8));
}

// =========================================================================
// LanguageDetector Tests
// =========================================================================

#[test]
fn test_language_detector_new() {
    let detector = LanguageDetector::new();
    assert!((detector.confidence_threshold() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_language_detector_with_threshold() {
    let detector = LanguageDetector::with_threshold(0.8);
    assert!((detector.confidence_threshold() - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_language_detector_default() {
    let detector = LanguageDetector::default();
    assert!((detector.confidence_threshold() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_language_detector_detect_from_logits() {
    let detector = LanguageDetector::new();
    let mut logits = vec![0.0_f32; 60000];

    logits[special_tokens::LANG_BASE as usize + 6] = 10.0; // French

    let probs = detector.detect_from_logits(&logits);
    assert_eq!(probs.top_language(), Some("fr"));
}

#[test]
fn test_language_detector_detect_with_fn() {
    let detector = LanguageDetector::new();

    let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
        let mut logits = vec![0.0_f32; 60000];
        logits[special_tokens::LANG_BASE as usize + 7] = 10.0; // Japanese
        Ok(logits)
    };

    let probs = detector
        .detect(logits_fn)
        .expect("detection should succeed");
    assert_eq!(probs.top_language(), Some("ja"));
}

#[test]
fn test_language_detector_is_confident() {
    let detector = LanguageDetector::with_threshold(0.6);

    let confident_probs = LanguageProbs {
        languages: vec!["en".to_string()],
        probabilities: vec![0.8],
    };
    assert!(detector.is_confident(&confident_probs));

    let unconfident_probs = LanguageProbs {
        languages: vec!["en".to_string()],
        probabilities: vec![0.5],
    };
    assert!(!detector.is_confident(&unconfident_probs));
}

// =========================================================================
// Utility Function Tests
// =========================================================================

#[test]
fn test_supported_languages_count() {
    assert_eq!(SUPPORTED_LANGUAGES.len(), 99);
}

#[test]
fn test_language_name() {
    assert_eq!(language_name("en"), Some("English"));
    assert_eq!(language_name("es"), Some("Spanish"));
    assert_eq!(language_name("ja"), Some("Japanese"));
    assert_eq!(language_name("invalid"), None);
}

#[test]
fn test_is_supported() {
    assert!(is_supported("en"));
    assert!(is_supported("zh"));
    assert!(is_supported("ja"));
    assert!(!is_supported("invalid"));
    assert!(!is_supported(""));
}

#[test]
fn test_language_index() {
    assert_eq!(language_index("en"), Some(0));
    assert_eq!(language_index("zh"), Some(1));
    assert_eq!(language_index("es"), Some(3));
    assert_eq!(language_index("ja"), Some(7));
    assert_eq!(language_index("invalid"), None);
}

// =========================================================================
// Softmax Tests
// =========================================================================

#[test]
fn test_softmax_sums_to_one() {
    let mut logits = vec![0.0_f32; 60000];

    // Set some language logits
    for i in 0..50 {
        logits[special_tokens::LANG_BASE as usize + i] = (i as f32) * 0.1;
    }

    let probs = LanguageProbs::from_logits(&logits);

    let sum: f32 = probs.probabilities.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Probabilities should sum to 1, got {}",
        sum
    );
}

#[test]
fn test_softmax_preserves_order() {
    let mut logits = vec![0.0_f32; 60000];

    // English > Spanish > German
    logits[special_tokens::LANG_BASE as usize] = 10.0; // English
    logits[special_tokens::LANG_BASE as usize + 3] = 5.0; // Spanish
    logits[special_tokens::LANG_BASE as usize + 2] = 3.0; // German

    let probs = LanguageProbs::from_logits(&logits);

    let en_prob = probs.probability_for("en").unwrap_or(0.0);
    let es_prob = probs.probability_for("es").unwrap_or(0.0);
    let de_prob = probs.probability_for("de").unwrap_or(0.0);

    assert!(en_prob > es_prob);
    assert!(es_prob > de_prob);
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_empty_logits() {
    let probs = LanguageProbs::from_logits(&[]);
    assert!(probs.languages.is_empty());
    assert!(probs.top_language().is_none());
}

#[test]
fn test_logits_too_short() {
    // Logits shorter than language token indices
    let logits = vec![0.0_f32; 1000];
    let probs = LanguageProbs::from_logits(&logits);
    assert!(probs.languages.is_empty());
}

#[test]
fn test_all_equal_logits() {
    let mut logits = vec![0.0_f32; 60000];

    // All language logits equal
    for i in 0..99 {
        logits[special_tokens::LANG_BASE as usize + i] = 1.0;
    }

    let probs = LanguageProbs::from_logits(&logits);

    // All should have roughly equal probability
    let first_prob = probs.probabilities.first().copied().unwrap_or(0.0);
    let last_prob = probs.probabilities.last().copied().unwrap_or(0.0);

    // Should be approximately equal (1/99 each)
    assert!((first_prob - last_prob).abs() < 0.01);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_language_name_extended_coverage() {
    // Test all supported language names
    assert_eq!(language_name("zh"), Some("Chinese"));
    assert_eq!(language_name("de"), Some("German"));
    assert_eq!(language_name("ru"), Some("Russian"));
    assert_eq!(language_name("ko"), Some("Korean"));
    assert_eq!(language_name("fr"), Some("French"));
    assert_eq!(language_name("pt"), Some("Portuguese"));
    assert_eq!(language_name("tr"), Some("Turkish"));
    assert_eq!(language_name("pl"), Some("Polish"));
    assert_eq!(language_name("ca"), Some("Catalan"));
    assert_eq!(language_name("nl"), Some("Dutch"));
    assert_eq!(language_name("ar"), Some("Arabic"));
    assert_eq!(language_name("sv"), Some("Swedish"));
    assert_eq!(language_name("it"), Some("Italian"));
    assert_eq!(language_name("id"), Some("Indonesian"));
    assert_eq!(language_name("hi"), Some("Hindi"));
    assert_eq!(language_name("fi"), Some("Finnish"));
    assert_eq!(language_name("vi"), Some("Vietnamese"));
    assert_eq!(language_name("he"), Some("Hebrew"));
    assert_eq!(language_name("uk"), Some("Ukrainian"));
    assert_eq!(language_name("el"), Some("Greek"));
    assert_eq!(language_name("ms"), Some("Malay"));
    assert_eq!(language_name("cs"), Some("Czech"));
    assert_eq!(language_name("ro"), Some("Romanian"));
    assert_eq!(language_name("da"), Some("Danish"));
    assert_eq!(language_name("hu"), Some("Hungarian"));
    assert_eq!(language_name("ta"), Some("Tamil"));
    assert_eq!(language_name("no"), Some("Norwegian"));
    assert_eq!(language_name("th"), Some("Thai"));
    assert_eq!(language_name("ur"), Some("Urdu"));
    assert_eq!(language_name("hr"), Some("Croatian"));
    assert_eq!(language_name("bg"), Some("Bulgarian"));
    assert_eq!(language_name("lt"), Some("Lithuanian"));
    assert_eq!(language_name("la"), Some("Latin"));
    assert_eq!(language_name("mi"), Some("Maori"));
    assert_eq!(language_name("ml"), Some("Malayalam"));
    assert_eq!(language_name("cy"), Some("Welsh"));
    assert_eq!(language_name("sk"), Some("Slovak"));
    assert_eq!(language_name("te"), Some("Telugu"));
    assert_eq!(language_name("fa"), Some("Persian"));
    assert_eq!(language_name("lv"), Some("Latvian"));
    assert_eq!(language_name("bn"), Some("Bengali"));
    assert_eq!(language_name("sr"), Some("Serbian"));
    assert_eq!(language_name("az"), Some("Azerbaijani"));
    assert_eq!(language_name("sl"), Some("Slovenian"));
    assert_eq!(language_name("kn"), Some("Kannada"));
    assert_eq!(language_name("et"), Some("Estonian"));
    assert_eq!(language_name("mk"), Some("Macedonian"));
}

#[test]
fn test_language_probs_top_probability() {
    let probs = LanguageProbs {
        languages: vec!["en".to_string(), "es".to_string()],
        probabilities: vec![0.8, 0.2],
    };
    assert_eq!(probs.top_probability(), Some(0.8));
}

#[test]
fn test_language_probs_top_probability_empty() {
    let probs = LanguageProbs {
        languages: vec![],
        probabilities: vec![],
    };
    assert_eq!(probs.top_probability(), None);
    assert_eq!(probs.top_language(), None);
}

#[test]
fn test_language_probs_confidence_empty() {
    let probs = LanguageProbs {
        languages: vec![],
        probabilities: vec![],
    };
    assert!((probs.confidence() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_language_probs_probability_for_not_found() {
    let probs = LanguageProbs {
        languages: vec!["en".to_string()],
        probabilities: vec![1.0],
    };
    assert_eq!(probs.probability_for("fr"), None);
}

#[test]
fn test_language_index_various() {
    assert_eq!(language_index("de"), Some(2));
    assert_eq!(language_index("ru"), Some(4));
    assert_eq!(language_index("fr"), Some(6));
    assert_eq!(language_index("pt"), Some(8));
    assert_eq!(language_index("su"), Some(98)); // Last language
}

#[test]
fn test_is_supported_various_languages() {
    // Test various supported languages
    assert!(is_supported("de"));
    assert!(is_supported("fr"));
    assert!(is_supported("it"));
    assert!(is_supported("su")); // Sundanese (last)
    assert!(is_supported("haw")); // Hawaiian

    // Test unsupported
    assert!(!is_supported("zz"));
    assert!(!is_supported("xxx"));
}

#[test]
fn test_language_probs_top_n_more_than_available() {
    let probs = LanguageProbs {
        languages: vec!["en".to_string(), "es".to_string()],
        probabilities: vec![0.6, 0.4],
    };
    let top5 = probs.top_n(5);
    assert_eq!(top5.len(), 2); // Only 2 available
}

#[test]
fn test_language_probs_from_logits_with_negative() {
    let mut logits = vec![-1.0_f32; 60000];

    // Set one high
    logits[special_tokens::LANG_BASE as usize] = 5.0;

    let probs = LanguageProbs::from_logits(&logits);
    assert_eq!(probs.top_language(), Some("en"));
    assert!(probs.confidence() > 0.5);
}

#[test]
fn test_language_detector_detect_error() {
    let detector = LanguageDetector::new();

    let logits_fn = |_tokens: &[u32]| -> WhisperResult<Vec<f32>> {
        Err(crate::error::WhisperError::Model("test error".into()))
    };

    let result = detector.detect(logits_fn);
    assert!(result.is_err());
}

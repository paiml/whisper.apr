//! Language detection module
//!
//! Implements automatic language detection for Whisper ASR.
//!
//! # Algorithm
//!
//! Whisper detects language by:
//! 1. Encoding audio to get encoder features
//! 2. Running decoder with SOT token
//! 3. Examining language token probabilities
//! 4. Selecting highest probability language
//!
//! # Example
//!
//! ```rust,ignore
//! use whisper_apr::detection::LanguageDetector;
//!
//! let detector = LanguageDetector::new();
//! let probs = detector.detect_language_probs(&audio_features)?;
//! println!("Detected: {}", probs.top_language());
//! ```

use crate::error::WhisperResult;
use crate::tokenizer::special_tokens;

/// Language detection result with probabilities
#[derive(Debug, Clone)]
pub struct LanguageProbs {
    /// Language codes in order of probability (highest first)
    pub languages: Vec<String>,
    /// Probabilities for each language (same order)
    pub probabilities: Vec<f32>,
}

impl LanguageProbs {
    /// Create new language probabilities from logits
    ///
    /// # Arguments
    /// * `logits` - Logits over vocabulary from decoder
    #[must_use]
    pub fn from_logits(logits: &[f32]) -> Self {
        // Get language token indices and their logits
        let lang_logits: Vec<(String, f32)> = SUPPORTED_LANGUAGES
            .iter()
            .enumerate()
            .filter_map(|(offset, &lang)| {
                let token_id = special_tokens::LANG_BASE + offset as u32;
                logits
                    .get(token_id as usize)
                    .map(|&logit| (lang.to_string(), logit))
            })
            .collect();

        // Compute softmax probabilities over language tokens only
        let max_logit = lang_logits
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = lang_logits.iter().map(|(_, l)| (l - max_logit).exp()).sum();

        let mut probs: Vec<(String, f32)> = lang_logits
            .iter()
            .map(|(lang, logit)| (lang.clone(), (logit - max_logit).exp() / exp_sum))
            .collect();

        // Sort by probability (descending)
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Self {
            languages: probs.iter().map(|(l, _)| l.clone()).collect(),
            probabilities: probs.iter().map(|(_, p)| *p).collect(),
        }
    }

    /// Get the top detected language
    #[must_use]
    pub fn top_language(&self) -> Option<&str> {
        self.languages.first().map(String::as_str)
    }

    /// Get probability for the top language
    #[must_use]
    pub fn top_probability(&self) -> Option<f32> {
        self.probabilities.first().copied()
    }

    /// Get the confidence score (top probability)
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.top_probability().unwrap_or(0.0)
    }

    /// Check if detection is confident (above threshold)
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence() >= threshold
    }

    /// Get top N languages with their probabilities
    #[must_use]
    pub fn top_n(&self, n: usize) -> Vec<(&str, f32)> {
        self.languages
            .iter()
            .zip(self.probabilities.iter())
            .take(n)
            .map(|(l, &p)| (l.as_str(), p))
            .collect()
    }

    /// Get probability for a specific language
    #[must_use]
    pub fn probability_for(&self, language: &str) -> Option<f32> {
        self.languages
            .iter()
            .position(|l| l == language)
            .and_then(|idx| self.probabilities.get(idx).copied())
    }
}

impl Default for LanguageProbs {
    fn default() -> Self {
        Self {
            languages: vec!["en".to_string()],
            probabilities: vec![1.0],
        }
    }
}

/// Language detector for automatic language identification
#[derive(Debug, Clone, Copy)]
pub struct LanguageDetector {
    /// Confidence threshold for "confident" detection
    confidence_threshold: f32,
}

impl LanguageDetector {
    /// Create a new language detector with default settings
    #[must_use]
    pub const fn new() -> Self {
        Self {
            confidence_threshold: 0.5,
        }
    }

    /// Create detector with custom confidence threshold
    #[must_use]
    pub const fn with_threshold(threshold: f32) -> Self {
        Self {
            confidence_threshold: threshold,
        }
    }

    /// Get the confidence threshold
    #[must_use]
    pub const fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    /// Detect language from decoder logits
    ///
    /// # Arguments
    /// * `logits` - Logits from decoder after processing SOT token
    ///
    /// # Returns
    /// Language probabilities
    #[must_use]
    pub fn detect_from_logits(&self, logits: &[f32]) -> LanguageProbs {
        LanguageProbs::from_logits(logits)
    }

    /// Detect language using a logits function
    ///
    /// # Arguments
    /// * `logits_fn` - Function that takes tokens and returns logits
    ///
    /// # Returns
    /// Language probabilities or error
    pub fn detect<F>(&self, mut logits_fn: F) -> WhisperResult<LanguageProbs>
    where
        F: FnMut(&[u32]) -> WhisperResult<Vec<f32>>,
    {
        // Get logits for just the SOT token
        let logits = logits_fn(&[special_tokens::SOT])?;
        Ok(self.detect_from_logits(&logits))
    }

    /// Check if detection result is confident
    #[must_use]
    pub fn is_confident(&self, probs: &LanguageProbs) -> bool {
        probs.is_confident(self.confidence_threshold)
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported languages in Whisper (99 languages)
///
/// Order matches Whisper's language token indices.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "en",  // English
    "zh",  // Chinese
    "de",  // German
    "es",  // Spanish
    "ru",  // Russian
    "ko",  // Korean
    "fr",  // French
    "ja",  // Japanese
    "pt",  // Portuguese
    "tr",  // Turkish
    "pl",  // Polish
    "ca",  // Catalan
    "nl",  // Dutch
    "ar",  // Arabic
    "sv",  // Swedish
    "it",  // Italian
    "id",  // Indonesian
    "hi",  // Hindi
    "fi",  // Finnish
    "vi",  // Vietnamese
    "he",  // Hebrew
    "uk",  // Ukrainian
    "el",  // Greek
    "ms",  // Malay
    "cs",  // Czech
    "ro",  // Romanian
    "da",  // Danish
    "hu",  // Hungarian
    "ta",  // Tamil
    "no",  // Norwegian
    "th",  // Thai
    "ur",  // Urdu
    "hr",  // Croatian
    "bg",  // Bulgarian
    "lt",  // Lithuanian
    "la",  // Latin
    "mi",  // Maori
    "ml",  // Malayalam
    "cy",  // Welsh
    "sk",  // Slovak
    "te",  // Telugu
    "fa",  // Persian
    "lv",  // Latvian
    "bn",  // Bengali
    "sr",  // Serbian
    "az",  // Azerbaijani
    "sl",  // Slovenian
    "kn",  // Kannada
    "et",  // Estonian
    "mk",  // Macedonian
    "br",  // Breton
    "eu",  // Basque
    "is",  // Icelandic
    "hy",  // Armenian
    "ne",  // Nepali
    "mn",  // Mongolian
    "bs",  // Bosnian
    "kk",  // Kazakh
    "sq",  // Albanian
    "sw",  // Swahili
    "gl",  // Galician
    "mr",  // Marathi
    "pa",  // Punjabi
    "si",  // Sinhala
    "km",  // Khmer
    "sn",  // Shona
    "yo",  // Yoruba
    "so",  // Somali
    "af",  // Afrikaans
    "oc",  // Occitan
    "ka",  // Georgian
    "be",  // Belarusian
    "tg",  // Tajik
    "sd",  // Sindhi
    "gu",  // Gujarati
    "am",  // Amharic
    "yi",  // Yiddish
    "lo",  // Lao
    "uz",  // Uzbek
    "fo",  // Faroese
    "ht",  // Haitian Creole
    "ps",  // Pashto
    "tk",  // Turkmen
    "nn",  // Norwegian Nynorsk
    "mt",  // Maltese
    "sa",  // Sanskrit
    "lb",  // Luxembourgish
    "my",  // Myanmar
    "bo",  // Tibetan
    "tl",  // Tagalog
    "mg",  // Malagasy
    "as",  // Assamese
    "tt",  // Tatar
    "haw", // Hawaiian
    "ln",  // Lingala
    "ha",  // Hausa
    "ba",  // Bashkir
    "jw",  // Javanese
    "su",  // Sundanese
];

/// Get language name from code
#[must_use]
pub fn language_name(code: &str) -> Option<&'static str> {
    match code {
        "en" => Some("English"),
        "zh" => Some("Chinese"),
        "de" => Some("German"),
        "es" => Some("Spanish"),
        "ru" => Some("Russian"),
        "ko" => Some("Korean"),
        "fr" => Some("French"),
        "ja" => Some("Japanese"),
        "pt" => Some("Portuguese"),
        "tr" => Some("Turkish"),
        "pl" => Some("Polish"),
        "ca" => Some("Catalan"),
        "nl" => Some("Dutch"),
        "ar" => Some("Arabic"),
        "sv" => Some("Swedish"),
        "it" => Some("Italian"),
        "id" => Some("Indonesian"),
        "hi" => Some("Hindi"),
        "fi" => Some("Finnish"),
        "vi" => Some("Vietnamese"),
        "he" => Some("Hebrew"),
        "uk" => Some("Ukrainian"),
        "el" => Some("Greek"),
        "ms" => Some("Malay"),
        "cs" => Some("Czech"),
        "ro" => Some("Romanian"),
        "da" => Some("Danish"),
        "hu" => Some("Hungarian"),
        "ta" => Some("Tamil"),
        "no" => Some("Norwegian"),
        "th" => Some("Thai"),
        "ur" => Some("Urdu"),
        "hr" => Some("Croatian"),
        "bg" => Some("Bulgarian"),
        "lt" => Some("Lithuanian"),
        "la" => Some("Latin"),
        "mi" => Some("Maori"),
        "ml" => Some("Malayalam"),
        "cy" => Some("Welsh"),
        "sk" => Some("Slovak"),
        "te" => Some("Telugu"),
        "fa" => Some("Persian"),
        "lv" => Some("Latvian"),
        "bn" => Some("Bengali"),
        "sr" => Some("Serbian"),
        "az" => Some("Azerbaijani"),
        "sl" => Some("Slovenian"),
        "kn" => Some("Kannada"),
        "et" => Some("Estonian"),
        "mk" => Some("Macedonian"),
        _ => None,
    }
}

/// Check if a language code is supported
#[must_use]
pub fn is_supported(code: &str) -> bool {
    SUPPORTED_LANGUAGES.contains(&code)
}

/// Get language index (for token computation)
#[must_use]
pub fn language_index(code: &str) -> Option<usize> {
    SUPPORTED_LANGUAGES.iter().position(|&l| l == code)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

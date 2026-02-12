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

#[cfg(test)]
mod tests;

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

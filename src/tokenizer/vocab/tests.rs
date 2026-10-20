//! Tests for vocabulary module

use super::*;

// =========================================================================
// Vocabulary Construction Tests
// =========================================================================

#[test]
fn test_vocabulary_new() {
    let vocab = Vocabulary::new();
    assert!(vocab.is_empty());
    assert_eq!(vocab.len(), 0);
}

#[test]
fn test_vocabulary_with_base_tokens() {
    let vocab = Vocabulary::with_base_tokens();
    assert_eq!(vocab.len(), 256);

    // Check some byte tokens
    assert_eq!(vocab.get_bytes(0), Some(&[0u8][..]));
    assert_eq!(vocab.get_bytes(65), Some(&[65u8][..])); // 'A'
    assert_eq!(vocab.get_bytes(255), Some(&[255u8][..]));
}

#[test]
fn test_vocabulary_add_token() {
    let mut vocab = Vocabulary::new();
    let id1 = vocab.add_token(vec![104, 101, 108, 108, 111]); // "hello"
    let id2 = vocab.add_token(vec![119, 111, 114, 108, 100]); // "world"

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(vocab.len(), 2);
}

#[test]
fn test_vocabulary_get_bytes() {
    let mut vocab = Vocabulary::new();
    vocab.add_token(vec![104, 105]); // "hi"

    assert_eq!(vocab.get_bytes(0), Some(&[104u8, 105][..]));
    assert_eq!(vocab.get_bytes(1), None);
}

#[test]
fn test_vocabulary_get_id() {
    let mut vocab = Vocabulary::new();
    vocab.add_token(vec![104, 105]); // "hi"

    assert_eq!(vocab.get_id(&[104, 105]), Some(0));
    assert_eq!(vocab.get_id(&[98, 121, 101]), None); // "bye"
}

// =========================================================================
// Merge Rule Tests
// =========================================================================

#[test]
fn test_merge_rule_new() {
    let rule = MergeRule::new(vec![104], vec![105]); // 'h' + 'i'
    assert_eq!(rule.first, vec![104]);
    assert_eq!(rule.second, vec![105]);
}

#[test]
fn test_merge_rule_merged() {
    let rule = MergeRule::new(vec![104], vec![105]);
    assert_eq!(rule.merged(), vec![104, 105]); // "hi"
}

#[test]
fn test_vocabulary_add_merge() {
    let mut vocab = Vocabulary::with_base_tokens();
    let merged_id = vocab.add_merge(vec![104], vec![105]); // 'h' + 'i' -> "hi"

    assert_eq!(merged_id, 256); // First merged token
    assert_eq!(vocab.len(), 257);
    assert_eq!(vocab.get_merge(&[104], &[105]), Some(256));
}

#[test]
fn test_vocabulary_merge_priority() {
    let mut vocab = Vocabulary::with_base_tokens();
    vocab.add_merge(vec![104], vec![105]); // 'h' + 'i' (priority 0)
    vocab.add_merge(vec![116], vec![104]); // 't' + 'h' (priority 1)

    assert_eq!(vocab.merge_priority(&[104], &[105]), Some(0));
    assert_eq!(vocab.merge_priority(&[116], &[104]), Some(1));
    assert_eq!(vocab.merge_priority(&[97], &[98]), None);
}

// =========================================================================
// Decode Tests
// =========================================================================

#[test]
fn test_decode_empty() {
    let vocab = Vocabulary::new();
    let result = vocab.decode(&[]);
    assert_eq!(result, Some(String::new()));
}

#[test]
fn test_decode_single_byte_tokens() {
    let vocab = Vocabulary::with_base_tokens();
    // "hi" = [104, 105]
    let result = vocab.decode(&[104, 105]);
    assert_eq!(result, Some("hi".to_string()));
}

#[test]
fn test_decode_ascii() {
    let vocab = Vocabulary::with_base_tokens();
    // "Hello" = [72, 101, 108, 108, 111]
    let result = vocab.decode(&[72, 101, 108, 108, 111]);
    assert_eq!(result, Some("Hello".to_string()));
}

#[test]
fn test_decode_invalid_token() {
    let vocab = Vocabulary::new();
    let result = vocab.decode(&[0]); // No tokens in empty vocab
    assert_eq!(result, None);
}

#[test]
fn test_decode_skips_special_tokens() {
    let vocab = Vocabulary::with_base_tokens();
    // Include EOT token - should be skipped
    let result = vocab.decode(&[72, 105, special_tokens::EOT]); // "Hi" + EOT
    assert_eq!(result, Some("Hi".to_string()));
}

// =========================================================================
// Special Token Tests
// =========================================================================

#[test]
fn test_special_tokens_values_multilingual() {
    // Default constants are for multilingual models (whisper-tiny, whisper-base, etc.)
    // These match whisper.cpp's multilingual token IDs
    assert_eq!(special_tokens::EOT, 50257); // Multilingual EOT
    assert_eq!(special_tokens::SOT, 50258); // Multilingual SOT
    assert_eq!(special_tokens::LANG_BASE, 50259); // Multilingual lang base
    assert_eq!(special_tokens::TRANSLATE, 50358);
    assert_eq!(special_tokens::TRANSCRIBE, 50359); // Multilingual transcribe
    assert_eq!(special_tokens::SPEAKER_TURN, 50360);
    assert_eq!(special_tokens::PREV, 50361);
    assert_eq!(special_tokens::NO_SPEECH, 50362);
    assert_eq!(special_tokens::NO_TIMESTAMPS, 50363); // Multilingual no_timestamps
    assert_eq!(special_tokens::TIMESTAMP_BASE, 50364); // Multilingual timestamp base
}

#[test]
fn test_special_tokens_english_only() {
    // English-only models (whisper-tiny.en, whisper-base.en) use GPT-2 tokenizer
    assert_eq!(special_tokens::EOT_ENGLISH, 50256);
    assert_eq!(special_tokens::SOT_ENGLISH, 50257);
}

#[test]
fn test_special_tokens_for_vocab_size() {
    use special_tokens::SpecialTokens;

    // Multilingual model (vocab >= 51865)
    let multi = SpecialTokens::for_vocab_size(51865);
    assert!(multi.is_multilingual);
    assert_eq!(multi.eot, 50257);
    assert_eq!(multi.sot, 50258);

    // English-only model (vocab < 51865)
    let english = SpecialTokens::for_vocab_size(51864);
    assert!(!english.is_multilingual);
    assert_eq!(english.eot, 50256);
    assert_eq!(english.sot, 50257);
}

#[test]
fn test_language_token() {
    // English is at LANG_BASE_MULTILINGUAL + 0 = 50259
    assert_eq!(special_tokens::language_token("en"), Some(50259));
    assert_eq!(special_tokens::language_token("zh"), Some(50260));
    assert_eq!(special_tokens::language_token("es"), Some(50262)); // es is index 3
    assert_eq!(special_tokens::language_token("invalid"), None);
}

#[test]
fn test_is_timestamp() {
    assert!(!special_tokens::is_timestamp(50362)); // NO_SPEECH
    assert!(!special_tokens::is_timestamp(50363)); // NO_TIMESTAMPS_MULTILINGUAL
    assert!(special_tokens::is_timestamp(50364)); // TIMESTAMP_BASE (multilingual)
    assert!(special_tokens::is_timestamp(50365)); // First timestamp after base
}

#[test]
fn test_timestamp_to_seconds() {
    // Timestamp 0 = 0.0 seconds
    assert_eq!(
        special_tokens::timestamp_to_seconds(special_tokens::TIMESTAMP_BASE),
        Some(0.0)
    );
    // Timestamp 50 = 1.0 second (50 * 0.02 = 1.0)
    assert_eq!(
        special_tokens::timestamp_to_seconds(special_tokens::TIMESTAMP_BASE + 50),
        Some(1.0)
    );
    // Non-timestamp token
    assert_eq!(special_tokens::timestamp_to_seconds(100), None);
}

// =========================================================================
// Property Tests
// =========================================================================

#[test]
fn test_vocabulary_roundtrip() {
    let mut vocab = Vocabulary::with_base_tokens();

    // Add some merges
    vocab.add_merge(vec![116], vec![104]); // "th"
    vocab.add_merge(vec![116, 104], vec![101]); // "the"

    // Check we can retrieve them
    assert!(vocab.get_merge(&[116], &[104]).is_some());
    assert!(vocab.get_merge(&[116, 104], &[101]).is_some());
}

#[test]
fn test_num_merges() {
    let mut vocab = Vocabulary::with_base_tokens();
    assert_eq!(vocab.num_merges(), 0);

    vocab.add_merge(vec![104], vec![105]);
    assert_eq!(vocab.num_merges(), 1);

    vocab.add_merge(vec![116], vec![104]);
    assert_eq!(vocab.num_merges(), 2);
}

// =========================================================================
// Serialization Tests
// =========================================================================

#[test]
fn test_vocabulary_to_bytes_empty() {
    let vocab = Vocabulary::new();
    let bytes = vocab.to_bytes();

    // 4 bytes for n_tokens (0) + 4 bytes for n_merges (0)
    assert_eq!(bytes.len(), 8);
    assert_eq!(&bytes[0..4], &0u32.to_le_bytes());
    assert_eq!(&bytes[4..8], &0u32.to_le_bytes());
}

#[test]
fn test_vocabulary_to_bytes_base_tokens() {
    let vocab = Vocabulary::with_base_tokens();
    let bytes = vocab.to_bytes();

    // Check header
    let n_tokens = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let n_merges = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

    assert_eq!(n_tokens, 256);
    assert_eq!(n_merges, 0);

    // Total size: 8 header + 256 * (2 len + 1 byte) = 8 + 768 = 776 bytes
    assert_eq!(bytes.len(), 776);
}

#[test]
fn test_vocabulary_roundtrip_empty() {
    let original = Vocabulary::new();
    let bytes = original.to_bytes();
    let restored = Vocabulary::from_bytes(&bytes).expect("should parse");

    assert_eq!(restored.len(), 0);
    assert_eq!(restored.num_merges(), 0);
}

#[test]
fn test_vocabulary_roundtrip_base_tokens() {
    let original = Vocabulary::with_base_tokens();
    let bytes = original.to_bytes();
    let restored = Vocabulary::from_bytes(&bytes).expect("should parse");

    assert_eq!(restored.len(), 256);
    assert_eq!(restored.num_merges(), 0);

    // Verify all base tokens
    for i in 0..256u32 {
        assert_eq!(restored.get_bytes(i), Some(&[i as u8][..]));
    }
}

#[test]
fn test_vocabulary_roundtrip_with_merges() {
    let mut original = Vocabulary::with_base_tokens();
    original.add_merge(vec![104], vec![105]); // "hi"
    original.add_merge(vec![116], vec![104]); // "th"
    original.add_merge(vec![116, 104], vec![101]); // "the"

    let bytes = original.to_bytes();
    let restored = Vocabulary::from_bytes(&bytes).expect("should parse");

    assert_eq!(restored.len(), original.len());
    assert_eq!(restored.num_merges(), 3);

    // Verify merge lookup works
    assert!(restored.merge_priority(&[104], &[105]).is_some());
    assert!(restored.merge_priority(&[116], &[104]).is_some());
    assert!(restored.merge_priority(&[116, 104], &[101]).is_some());
}

#[test]
fn test_vocabulary_from_bytes_too_short() {
    let bytes = vec![0u8; 4]; // Too short
    assert!(Vocabulary::from_bytes(&bytes).is_none());
}

#[test]
fn test_vocabulary_from_bytes_truncated_tokens() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u32.to_le_bytes()); // n_tokens = 1
    bytes.extend_from_slice(&0u32.to_le_bytes()); // n_merges = 0
    bytes.extend_from_slice(&10u16.to_le_bytes()); // token len = 10
                                                   // But no token data

    assert!(Vocabulary::from_bytes(&bytes).is_none());
}

#[test]
fn test_merge_rules_accessor() {
    let mut vocab = Vocabulary::with_base_tokens();
    vocab.add_merge(vec![104], vec![105]);

    let rules = vocab.merge_rules();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].first, vec![104]);
    assert_eq!(rules[0].second, vec![105]);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_special_tokens_initial_tokens() {
    let multi = special_tokens::SpecialTokens::multilingual();
    let tokens = multi.initial_tokens();
    assert_eq!(tokens.len(), 4);
    assert_eq!(tokens[0], multi.sot);
    assert_eq!(tokens[1], multi.lang_base);
    assert_eq!(tokens[2], multi.transcribe);
    assert_eq!(tokens[3], multi.no_timestamps);

    let english = special_tokens::SpecialTokens::english_only();
    let eng_tokens = english.initial_tokens();
    assert_eq!(eng_tokens[0], english.sot);
}

#[test]
fn test_special_tokens_default() {
    let default = special_tokens::SpecialTokens::default();
    let multi = special_tokens::SpecialTokens::multilingual();

    assert_eq!(default.eot, multi.eot);
    assert_eq!(default.sot, multi.sot);
    assert!(default.is_multilingual);
}

#[test]
fn test_language_offset_coverage() {
    use special_tokens::language_offset;

    // Test more languages to increase coverage
    assert_eq!(language_offset("en"), Some(0));
    assert_eq!(language_offset("zh"), Some(1));
    assert_eq!(language_offset("de"), Some(2));
    assert_eq!(language_offset("es"), Some(3));
    assert_eq!(language_offset("ru"), Some(4));
    assert_eq!(language_offset("ko"), Some(5));
    assert_eq!(language_offset("fr"), Some(6));
    assert_eq!(language_offset("ja"), Some(7));
    assert_eq!(language_offset("pt"), Some(8));
    assert_eq!(language_offset("tr"), Some(9));
    assert_eq!(language_offset("pl"), Some(10));
    assert_eq!(language_offset("ca"), Some(11));
    assert_eq!(language_offset("nl"), Some(12));
    assert_eq!(language_offset("ar"), Some(13));
    assert_eq!(language_offset("sv"), Some(14));
    assert_eq!(language_offset("it"), Some(15));
    assert_eq!(language_offset("id"), Some(16));
    assert_eq!(language_offset("hi"), Some(17));
    assert_eq!(language_offset("fi"), Some(18));
    assert_eq!(language_offset("vi"), Some(19));
    assert_eq!(language_offset("he"), Some(20));
    assert_eq!(language_offset("uk"), Some(21));
    assert_eq!(language_offset("el"), Some(22));
    assert_eq!(language_offset("ms"), Some(23));
    assert_eq!(language_offset("cs"), Some(24));
    assert_eq!(language_offset("ro"), Some(25));
    assert_eq!(language_offset("da"), Some(26));
    assert_eq!(language_offset("hu"), Some(27));
    assert_eq!(language_offset("ta"), Some(28));
    assert_eq!(language_offset("no"), Some(29));
    assert_eq!(language_offset("th"), Some(30));
    assert_eq!(language_offset("ur"), Some(31));
    assert_eq!(language_offset("hr"), Some(32));
    assert_eq!(language_offset("bg"), Some(33));
    assert_eq!(language_offset("lt"), Some(34));
    assert_eq!(language_offset("la"), Some(35));
    assert_eq!(language_offset("mi"), Some(36));
    assert_eq!(language_offset("ml"), Some(37));
    assert_eq!(language_offset("unknown"), None);
}

#[test]
fn test_language_token_extended_coverage() {
    use special_tokens::language_token;

    // Test more languages for coverage
    assert_eq!(language_token("de"), Some(50261));
    assert_eq!(language_token("fr"), Some(50265));
    assert_eq!(language_token("ja"), Some(50266));
    assert_eq!(language_token("pt"), Some(50267));
    assert_eq!(language_token("tr"), Some(50268));
    assert_eq!(language_token("pl"), Some(50269));
    assert_eq!(language_token("ca"), Some(50270));
    assert_eq!(language_token("nl"), Some(50271));
    assert_eq!(language_token("ar"), Some(50272));
    assert_eq!(language_token("sv"), Some(50273));
    assert_eq!(language_token("it"), Some(50274));
    assert_eq!(language_token("id"), Some(50275));
    assert_eq!(language_token("hi"), Some(50276));
    assert_eq!(language_token("fi"), Some(50277));
    assert_eq!(language_token("vi"), Some(50278));
    assert_eq!(language_token("he"), Some(50279));
    assert_eq!(language_token("uk"), Some(50280));
    assert_eq!(language_token("el"), Some(50281));
    assert_eq!(language_token("ms"), Some(50282));
    assert_eq!(language_token("cs"), Some(50283));
    assert_eq!(language_token("ro"), Some(50284));
    assert_eq!(language_token("da"), Some(50285));
    assert_eq!(language_token("hu"), Some(50286));
    assert_eq!(language_token("ta"), Some(50287));
    assert_eq!(language_token("no"), Some(50288));
    assert_eq!(language_token("th"), Some(50289));
    assert_eq!(language_token("ur"), Some(50290));
    assert_eq!(language_token("hr"), Some(50291));
    assert_eq!(language_token("bg"), Some(50292));
    assert_eq!(language_token("lt"), Some(50293));
    assert_eq!(language_token("la"), Some(50294));
    assert_eq!(language_token("mi"), Some(50295));
    assert_eq!(language_token("ml"), Some(50296));
    assert_eq!(language_token("cy"), Some(50297));
    assert_eq!(language_token("sk"), Some(50298));
    assert_eq!(language_token("te"), Some(50299));
    assert_eq!(language_token("fa"), Some(50300));
    assert_eq!(language_token("lv"), Some(50301));
    assert_eq!(language_token("bn"), Some(50302));
    assert_eq!(language_token("sr"), Some(50303));
    assert_eq!(language_token("az"), Some(50304));
    assert_eq!(language_token("sl"), Some(50305));
    assert_eq!(language_token("kn"), Some(50306));
    assert_eq!(language_token("et"), Some(50307));
    assert_eq!(language_token("mk"), Some(50308));
}

#[test]
fn test_add_merge_existing_token() {
    let mut vocab = Vocabulary::with_base_tokens();

    // First add "hi" as a merged token
    let id1 = vocab.add_merge(vec![104], vec![105]); // h + i -> hi

    // Now add a new merge that results in same bytes
    // This should reuse the existing token
    let id2 = vocab.add_token(vec![104, 105]); // directly add "hi"

    // Both should reference the same underlying bytes
    // Note: add_token will add a new entry even if bytes exist
    // This is expected behavior - the bytes_to_id might be updated
    assert!(vocab.get_bytes(id1).is_some());
    assert!(vocab.get_bytes(id2).is_some());
}

#[test]
fn test_special_tokens_english_only_struct() {
    let eng = special_tokens::SpecialTokens::english_only();
    assert!(!eng.is_multilingual);
    assert_eq!(eng.eot, special_tokens::EOT_ENGLISH);
    assert_eq!(eng.sot, special_tokens::SOT_ENGLISH);
    assert_eq!(eng.timestamp_base, 50363);
}

#[test]
fn test_decode_merged_tokens() {
    let mut vocab = Vocabulary::with_base_tokens();
    let hi_id = vocab.add_merge(vec![104], vec![105]); // "hi"

    // Decode using the merged token ID
    let result = vocab.decode(&[hi_id]);
    assert_eq!(result, Some("hi".to_string()));
}

#[test]
fn test_decode_utf8_invalid() {
    let mut vocab = Vocabulary::new();
    vocab.add_token(vec![0xFF, 0xFE]); // Invalid UTF-8 sequence

    let result = vocab.decode(&[0]);
    // Should fail gracefully (return None or replacement char)
    assert!(result.is_none() || result.as_ref().is_some_and(|s| s.contains('\u{FFFD}')));
}

// =========================================================================
// Additional Edge Case Coverage Tests
// =========================================================================

#[test]
fn test_vocabulary_default() {
    let vocab = Vocabulary::default();
    assert!(vocab.is_empty());
    assert_eq!(vocab.len(), 0);
    assert_eq!(vocab.num_merges(), 0);
}

#[test]
fn test_vocabulary_is_not_empty() {
    let vocab = Vocabulary::with_base_tokens();
    assert!(!vocab.is_empty());
}

#[test]
fn test_from_bytes_truncated_token_len() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u32.to_le_bytes()); // n_tokens = 1
    bytes.extend_from_slice(&0u32.to_le_bytes()); // n_merges = 0
                                                  // Only 1 byte for token len instead of 2
    bytes.push(5u8);

    assert!(Vocabulary::from_bytes(&bytes).is_none());
}

#[test]
fn test_from_bytes_truncated_merge_first_len() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0u32.to_le_bytes()); // n_tokens = 0
    bytes.extend_from_slice(&1u32.to_le_bytes()); // n_merges = 1
                                                  // Only 1 byte for first_len instead of 2
    bytes.push(3u8);

    assert!(Vocabulary::from_bytes(&bytes).is_none());
}

#[test]
fn test_from_bytes_truncated_merge_first_data() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0u32.to_le_bytes()); // n_tokens = 0
    bytes.extend_from_slice(&1u32.to_le_bytes()); // n_merges = 1
    bytes.extend_from_slice(&5u16.to_le_bytes()); // first_len = 5
    bytes.extend_from_slice(&[1, 2, 3]); // Only 3 bytes, need 5

    assert!(Vocabulary::from_bytes(&bytes).is_none());
}

#[test]
fn test_from_bytes_truncated_merge_second_len() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0u32.to_le_bytes()); // n_tokens = 0
    bytes.extend_from_slice(&1u32.to_le_bytes()); // n_merges = 1
    bytes.extend_from_slice(&2u16.to_le_bytes()); // first_len = 2
    bytes.extend_from_slice(&[104, 105]); // first data "hi"
                                          // Only 1 byte for second_len instead of 2
    bytes.push(3u8);

    assert!(Vocabulary::from_bytes(&bytes).is_none());
}

#[test]
fn test_from_bytes_truncated_merge_second_data() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0u32.to_le_bytes()); // n_tokens = 0
    bytes.extend_from_slice(&1u32.to_le_bytes()); // n_merges = 1
    bytes.extend_from_slice(&2u16.to_le_bytes()); // first_len = 2
    bytes.extend_from_slice(&[104, 105]); // first data "hi"
    bytes.extend_from_slice(&5u16.to_le_bytes()); // second_len = 5
    bytes.extend_from_slice(&[1, 2, 3]); // Only 3 bytes, need 5

    assert!(Vocabulary::from_bytes(&bytes).is_none());
}

#[test]
fn test_get_merge_nonexistent() {
    let vocab = Vocabulary::with_base_tokens();
    assert!(vocab.get_merge(&[200], &[201]).is_none());
}

#[test]
fn test_add_merge_reuses_existing_merged_token() {
    let mut vocab = Vocabulary::with_base_tokens();
    // First manually add a token "ab"
    let ab_id = vocab.add_token(vec![97, 98]); // "ab"

    // Now add a merge a + b -> should reuse the existing "ab" token
    let merge_id = vocab.add_merge(vec![97], vec![98]);

    // The merge should reference the existing token
    assert_eq!(merge_id, ab_id);
    assert_eq!(vocab.get_merge(&[97], &[98]), Some(ab_id));
}

#[test]
fn test_decode_multiple_special_tokens() {
    let vocab = Vocabulary::with_base_tokens();
    // Test that multiple special tokens are all skipped
    let result = vocab.decode(&[
        72, // 'H'
        special_tokens::EOT,
        105, // 'i'
        special_tokens::SOT,
        special_tokens::TRANSCRIBE,
    ]);
    assert_eq!(result, Some("Hi".to_string()));
}

#[test]
fn test_decode_only_special_tokens() {
    let vocab = Vocabulary::with_base_tokens();
    let result = vocab.decode(&[
        special_tokens::EOT,
        special_tokens::SOT,
        special_tokens::TRANSCRIBE,
    ]);
    assert_eq!(result, Some(String::new()));
}

#[test]
fn test_timestamp_boundary_values() {
    // Test boundary at TIMESTAMP_BASE - 1
    assert!(!special_tokens::is_timestamp(
        special_tokens::TIMESTAMP_BASE - 1
    ));
    assert!(special_tokens::is_timestamp(special_tokens::TIMESTAMP_BASE));

    // Test timestamp conversion for boundary
    assert_eq!(
        special_tokens::timestamp_to_seconds(special_tokens::TIMESTAMP_BASE - 1),
        None
    );
    assert_eq!(
        special_tokens::timestamp_to_seconds(special_tokens::TIMESTAMP_BASE),
        Some(0.0)
    );
}

#[test]
fn test_special_tokens_equality() {
    let multi1 = special_tokens::SpecialTokens::multilingual();
    let multi2 = special_tokens::SpecialTokens::multilingual();
    assert_eq!(multi1, multi2);

    let english = special_tokens::SpecialTokens::english_only();
    assert_ne!(multi1, english);
}

// =========================================================================
// Special-token layouts per vocabulary size (issue #108,
// contracts/special-tokens-v1.yaml)
// =========================================================================

/// OpenAI's language codes in tokenizer order; large-v3 appends "yue".
const OPENAI_LANGUAGES: [&str; 100] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// The special-token strings of a Whisper vocabulary exactly as OpenAI's
/// tokenizer appends them after the BPE ranks, built independently of
/// `SpecialTokens`: `names[id - eot]` is the token `id` decodes to.
fn openai_special_names(eot: u32, n_languages: usize) -> (u32, Vec<String>) {
    let mut names = vec![
        "<|endoftext|>".to_string(),
        "<|startoftranscript|>".to_string(),
    ];
    names.extend(
        OPENAI_LANGUAGES[..n_languages]
            .iter()
            .map(|l| format!("<|{l}|>")),
    );
    for name in [
        "translate",
        "transcribe",
        "startoflm",
        "startofprev",
        "nospeech",
        "notimestamps",
    ] {
        names.push(format!("<|{name}|>"));
    }
    names.extend((0..=1500).map(|i| format!("<|{:.2}|>", i as f32 * 0.02)));
    (eot, names)
}

fn decode_special(reference: &(u32, Vec<String>), id: u32) -> &str {
    let (eot, names) = reference;
    &names[(id - eot) as usize]
}

/// The independent reference fills each vocabulary exactly.
#[test]
fn the_reference_special_names_fill_each_vocabulary() {
    for (n_vocab, eot, langs) in [
        (51864_usize, 50256, 99),
        (51865, 50257, 99),
        (51866, 50257, 100),
    ] {
        let (eot, names) = openai_special_names(eot, langs);
        assert_eq!(eot as usize + names.len(), n_vocab, "n_vocab {n_vocab}");
    }
}

/// FALSIFY-SPECIAL-001: on large-v3/turbo (51866) the transcribe task token
/// decodes to `<|transcribe|>` (it decoded to `<|translate|>` before #108).
#[test]
fn on_v3_the_task_token_decodes_to_transcribe() {
    let v3 = special_tokens::SpecialTokens::from_vocab_size(51866).expect("51866 is known");
    let reference = openai_special_names(50257, 100);
    assert_eq!(v3.transcribe, 50360);
    assert_eq!(decode_special(&reference, v3.transcribe), "<|transcribe|>");
    assert_eq!(decode_special(&reference, v3.translate), "<|translate|>");
    assert_eq!(decode_special(&reference, v3.sot), "<|startoftranscript|>");
    assert_eq!(decode_special(&reference, v3.no_speech), "<|nospeech|>");
    assert_eq!(
        decode_special(&reference, v3.no_timestamps),
        "<|notimestamps|>"
    );
    assert_eq!(decode_special(&reference, v3.lang_base + 99), "<|yue|>");
    assert_eq!(v3.name_of(v3.transcribe), Some("<|transcribe|>"));
    // The v2 id 50359 is the translate token on v3.
    assert_eq!(decode_special(&reference, 50359), "<|translate|>");
}

/// FALSIFY-SPECIAL-002: on 51866 `<|0.00|>` is 50365 and 50364 is
/// `<|notimestamps|>`, not a timestamp.
#[test]
fn on_v3_the_first_timestamp_is_50365() {
    let v3 = special_tokens::SpecialTokens::from_vocab_size(51866).expect("51866 is known");
    let reference = openai_special_names(50257, 100);
    assert_eq!(v3.timestamp_base, 50365);
    assert_eq!(decode_special(&reference, 50365), "<|0.00|>");
    assert_eq!(v3.timestamp_to_seconds(50365), Some(0.0));
    assert!(!v3.is_timestamp(50364));
    assert!(v3.is_control_token(50364));
    let last = 51865;
    assert_eq!(decode_special(&reference, last), "<|30.00|>");
    assert!((v3.timestamp_to_seconds(last).expect("timestamp") - 30.0).abs() < 1e-4);
}

/// FALSIFY-SPECIAL-003: on 51866 `<|startofprev|>` is 50362.
#[test]
fn on_v3_prev_is_50362() {
    let v3 = special_tokens::SpecialTokens::from_vocab_size(51866).expect("51866 is known");
    let reference = openai_special_names(50257, 100);
    assert_eq!(v3.prev, 50362);
    assert_eq!(decode_special(&reference, v3.prev), "<|startofprev|>");
    assert_eq!(decode_special(&reference, v3.start_of_lm), "<|startoflm|>");
}

/// FALSIFY-SPECIAL-004: on 51865 every id is the one the legacy constants
/// have always had, and on 51864 the English-only layout.
#[test]
fn on_v2_the_old_ids_are_unchanged() {
    let v2 = special_tokens::SpecialTokens::from_vocab_size(51865).expect("51865 is known");
    assert_eq!(v2.eot, special_tokens::EOT);
    assert_eq!(v2.sot, special_tokens::SOT);
    assert_eq!(v2.lang_base, special_tokens::LANG_BASE);
    assert_eq!(v2.translate, special_tokens::TRANSLATE);
    assert_eq!(v2.transcribe, special_tokens::TRANSCRIBE);
    assert_eq!(v2.start_of_lm, special_tokens::SPEAKER_TURN);
    assert_eq!(v2.prev, special_tokens::PREV);
    assert_eq!(v2.no_speech, special_tokens::NO_SPEECH);
    assert_eq!(v2.no_timestamps, special_tokens::NO_TIMESTAMPS);
    assert_eq!(v2.timestamp_base, special_tokens::TIMESTAMP_BASE);
    assert_eq!(
        [
            v2.eot,
            v2.sot,
            v2.translate,
            v2.transcribe,
            v2.prev,
            v2.no_speech
        ],
        [50257, 50258, 50358, 50359, 50361, 50362]
    );
    assert_eq!([v2.no_timestamps, v2.timestamp_base], [50363, 50364]);
    let reference = openai_special_names(50257, 99);
    for id in [
        v2.eot,
        v2.sot,
        v2.translate,
        v2.transcribe,
        v2.prev,
        v2.no_timestamps,
    ] {
        assert_eq!(Some(decode_special(&reference, id)), v2.name_of(id));
    }
    assert_eq!(decode_special(&reference, v2.timestamp_base), "<|0.00|>");

    let en = special_tokens::SpecialTokens::from_vocab_size(51864).expect("51864 is known");
    let reference = openai_special_names(50256, 99);
    assert_eq!(
        [en.eot, en.sot, en.transcribe, en.timestamp_base],
        [50256, 50257, 50358, 50363]
    );
    for id in [
        en.eot,
        en.sot,
        en.translate,
        en.transcribe,
        en.prev,
        en.no_timestamps,
    ] {
        assert_eq!(Some(decode_special(&reference, id)), en.name_of(id));
    }
}

/// FALSIFY-SPECIAL-005: any other vocabulary size is refused.
#[test]
fn an_unknown_vocab_size_is_refused() {
    for n_vocab in [0, 32768, 50257, 51863, 51867, 51872, 100_000] {
        assert_eq!(
            special_tokens::SpecialTokens::from_vocab_size(n_vocab),
            None
        );
        let err = special_tokens::SpecialTokens::try_for_vocab_size(n_vocab)
            .expect_err("unknown size must be refused");
        assert!(
            err.to_string()
                .contains(&format!("unsupported Whisper vocabulary size {n_vocab}")),
            "{err}"
        );
    }
    for n_vocab in [51864, 51865, 51866] {
        assert!(special_tokens::SpecialTokens::try_for_vocab_size(n_vocab).is_ok());
    }
}

/// Every known layout satisfies the id-table equations of the contract.
#[test]
fn every_layout_satisfies_the_id_table() {
    for (n_vocab, eot, langs) in [
        (51864_usize, 50256, 99),
        (51865, 50257, 99),
        (51866, 50257, 100),
    ] {
        let s = special_tokens::SpecialTokens::from_vocab_size(n_vocab).expect("known size");
        assert_eq!(s.eot, eot);
        assert_eq!(s.n_languages, langs);
        assert_eq!(s.sot, s.eot + 1);
        assert_eq!(s.lang_base, s.eot + 2);
        assert_eq!(s.translate, s.lang_base + s.n_languages);
        assert_eq!(
            [
                s.transcribe,
                s.start_of_lm,
                s.prev,
                s.no_speech,
                s.no_timestamps,
                s.timestamp_base
            ],
            [1, 2, 3, 4, 5, 6].map(|k| s.translate + k)
        );
        assert_eq!(s.timestamp_base as usize + 1501, n_vocab);
    }
}

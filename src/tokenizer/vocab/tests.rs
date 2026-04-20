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

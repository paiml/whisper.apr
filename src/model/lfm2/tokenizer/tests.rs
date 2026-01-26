//! Tests for LFM2 tokenizer implementation

use super::*;

#[test]
fn test_byte_level_tokenizer_encode() {
    let tokenizer = ByteLevelTokenizer::new();
    let tokens = tokenizer.encode("Hi");

    // BOS + 'H' + 'i' + EOS
    assert_eq!(tokens.len(), 4);
    assert_eq!(tokens[0], tokenizer.special_tokens.bos);
    assert_eq!(tokens[1], b'H' as u32 + tokenizer.byte_offset);
    assert_eq!(tokens[2], b'i' as u32 + tokenizer.byte_offset);
    assert_eq!(tokens[3], tokenizer.special_tokens.eos);
}

#[test]
fn test_byte_level_tokenizer_decode() {
    let tokenizer = ByteLevelTokenizer::new();
    let tokens = tokenizer.encode("Hello");
    let decoded = tokenizer.decode(&tokens);
    assert_eq!(decoded, "Hello");
}

#[test]
fn test_byte_level_tokenizer_roundtrip() {
    let tokenizer = ByteLevelTokenizer::new();
    let text = "The quick brown fox jumps over the lazy dog.";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens);
    assert_eq!(decoded, text);
}

#[test]
fn test_byte_level_tokenizer_unicode() {
    let tokenizer = ByteLevelTokenizer::new();
    // Note: byte-level encoding works with UTF-8 bytes
    let text = "café";
    let tokens = tokenizer.encode_without_special(text);
    // 'c', 'a', 'f', 'é' (2 bytes in UTF-8)
    assert!(tokens.len() >= 4);
}

#[test]
fn test_lfm2_tokenizer_new() {
    let tokenizer = Lfm2Tokenizer::new();
    assert!(tokenizer.vocab_size() > 256); // Special + byte tokens
}

#[test]
fn test_lfm2_tokenizer_encode() {
    let tokenizer = Lfm2Tokenizer::new();
    let tokens = tokenizer.encode("test");

    assert!(tokens.len() >= 2); // At least BOS and EOS
    assert_eq!(tokens[0], tokenizer.bos_token_id());
    assert_eq!(*tokens.last().expect("has last"), tokenizer.eos_token_id());
}

#[test]
fn test_lfm2_tokenizer_decode() {
    let tokenizer = Lfm2Tokenizer::new();
    let text = "Hello world";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens);
    assert_eq!(decoded, text);
}

#[test]
fn test_lfm2_tokenizer_special_tokens() {
    let tokenizer = Lfm2Tokenizer::new();
    assert!(tokenizer.is_special_token(tokenizer.bos_token_id()));
    assert!(tokenizer.is_special_token(tokenizer.eos_token_id()));
    assert!(tokenizer.is_special_token(tokenizer.pad_token_id()));
    assert!(!tokenizer.is_special_token(1000)); // Random ID
}

#[test]
fn test_lfm2_tokenizer_from_vocab() {
    let vocab = vec![
        "<pad>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "<unk>".to_string(),
        "hello".to_string(),
        "world".to_string(),
    ];

    let tokenizer = Lfm2Tokenizer::from_vocab(&vocab).expect("should create tokenizer");
    assert_eq!(tokenizer.vocab_size(), vocab.len());

    // Check that special tokens are set correctly
    assert_eq!(tokenizer.pad_token_id(), 0);
    assert_eq!(tokenizer.bos_token_id(), 1);
    assert_eq!(tokenizer.eos_token_id(), 2);
}

#[test]
fn test_byte_level_vocab_size() {
    let tokenizer = ByteLevelTokenizer::new();
    assert_eq!(tokenizer.vocab_size(), 512);
}

#[test]
fn test_encode_without_special() {
    let byte_tok = ByteLevelTokenizer::new();
    let tokens = byte_tok.encode_without_special("AB");
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0], b'A' as u32 + byte_tok.byte_offset);
    assert_eq!(tokens[1], b'B' as u32 + byte_tok.byte_offset);

    let lfm2_tok = Lfm2Tokenizer::new();
    let tokens = lfm2_tok.encode_without_special("AB");
    // Without BPE merges, encodes as characters
    assert!(!tokens.is_empty());
}

#[test]
fn test_from_huggingface_json_basic() {
    let json = r#"{
        "model": {
            "vocab": {
                "hello": 100,
                "world": 101,
                "test": 102
            },
            "merges": ["h e", "he l", "hel lo"]
        },
        "added_tokens": [
            {"id": 0, "content": "<pad>", "special": true},
            {"id": 1, "content": "<s>", "special": true},
            {"id": 2, "content": "</s>", "special": true}
        ]
    }"#;

    let tokenizer = Lfm2Tokenizer::from_huggingface_json(json).expect("should parse");

    // Check vocab
    assert!(tokenizer.vocab_size() >= 3);

    // Check special tokens
    assert_eq!(tokenizer.pad_token_id(), 0);
    assert_eq!(tokenizer.bos_token_id(), 1);
    assert_eq!(tokenizer.eos_token_id(), 2);
}

#[test]
fn test_from_huggingface_json_merges() {
    let json = r#"{
        "model": {
            "vocab": {
                "a": 10,
                "b": 11,
                "ab": 12
            },
            "merges": ["a b"]
        }
    }"#;

    let tokenizer = Lfm2Tokenizer::from_huggingface_json(json).expect("should parse");

    // With merges, "ab" should encode using the merged token
    let tokens = tokenizer.encode_without_special("ab");
    assert!(!tokens.is_empty());
}

#[test]
fn test_from_huggingface_json_empty() {
    let json = r#"{}"#;
    let tokenizer = Lfm2Tokenizer::from_huggingface_json(json).expect("should parse empty");
    assert_eq!(tokenizer.vocab_size(), 0);
}

#[test]
fn test_from_huggingface_json_special_token_variants() {
    // Test different special token naming conventions
    let json = r#"{
        "added_tokens": [
            {"id": 10, "content": "[PAD]", "special": true},
            {"id": 11, "content": "[CLS]", "special": true},
            {"id": 12, "content": "[SEP]", "special": true},
            {"id": 13, "content": "[UNK]", "special": true}
        ]
    }"#;

    let tokenizer = Lfm2Tokenizer::from_huggingface_json(json).expect("should parse");

    assert_eq!(tokenizer.pad_token_id(), 10);
    assert_eq!(tokenizer.bos_token_id(), 11);
    assert_eq!(tokenizer.eos_token_id(), 12);
}

#[test]
fn test_unescape_json_string() {
    assert_eq!(unescape_json_string("hello"), "hello");
    assert_eq!(unescape_json_string(r"hello\nworld"), "hello\nworld");
    assert_eq!(unescape_json_string(r"tab\there"), "tab\there");
    assert_eq!(unescape_json_string(r#"quote\"here"#), "quote\"here");
    assert_eq!(unescape_json_string(r"\u0041"), "A");
}

#[test]
fn test_parse_vocab_entries() {
    let json = r#"{"a": 1, "b": 2, "hello": 100}"#;
    let mut vocab = HashMap::new();
    let mut id_to_token = HashMap::new();

    parse_vocab_entries(json, &mut vocab, &mut id_to_token);

    assert_eq!(vocab.get("a"), Some(&1));
    assert_eq!(vocab.get("b"), Some(&2));
    assert_eq!(vocab.get("hello"), Some(&100));
    assert_eq!(id_to_token.get(&1), Some(&"a".to_string()));
}

#[test]
fn test_parse_merge_entries() {
    let json = r#"["a b", "c d", "ab cd"]"#;
    let mut merges = Vec::new();

    parse_merge_entries(json, &mut merges);

    assert_eq!(merges.len(), 3);
    assert_eq!(merges[0], ("a".to_string(), "b".to_string()));
    assert_eq!(merges[1], ("c".to_string(), "d".to_string()));
    assert_eq!(merges[2], ("ab".to_string(), "cd".to_string()));
}

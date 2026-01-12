//! LFM2 Tokenizer Implementation (WAPR-LFM2-007)
//!
//! This module provides tokenization for LFM2 text input/output.
//!
//! # Tokenizer Types
//!
//! - `ByteLevelTokenizer`: Simple byte-level encoding (fallback)
//! - `Lfm2Tokenizer`: Full BPE tokenizer with vocab
//!
//! # Spec Reference
//!
//! See `docs/specifications/1.0-whisper-apr.md` Section 18.2:
//! - Vocab Size: 100,288 tokens
//! - Tokenizer: BPE (Byte-Pair Encoding)

use crate::error::WhisperResult;
use std::collections::HashMap;

// =============================================================================
// Special Tokens
// =============================================================================

/// Special token IDs for LFM2
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokens {
    /// Beginning of sequence
    pub bos: u32,
    /// End of sequence
    pub eos: u32,
    /// Padding token
    pub pad: u32,
    /// Unknown token
    pub unk: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos: 1,
            eos: 2,
            pad: 0,
            unk: 3,
        }
    }
}

// =============================================================================
// Byte-Level Tokenizer (Fallback)
// =============================================================================

/// Simple byte-level tokenizer for fallback
///
/// This tokenizer encodes each byte as a token ID (offset by 256 to leave
/// room for special tokens). Used when no vocabulary is available.
#[derive(Debug, Clone)]
pub struct ByteLevelTokenizer {
    /// Special token configuration
    pub special_tokens: SpecialTokens,
    /// Offset for byte tokens (to avoid special token IDs)
    pub byte_offset: u32,
}

impl Default for ByteLevelTokenizer {
    fn default() -> Self {
        Self {
            special_tokens: SpecialTokens::default(),
            byte_offset: 256,
        }
    }
}

impl ByteLevelTokenizer {
    /// Create new byte-level tokenizer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Encode text to tokens
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        tokens.push(self.special_tokens.bos);
        for byte in text.bytes() {
            tokens.push(u32::from(byte) + self.byte_offset);
        }
        tokens.push(self.special_tokens.eos);
        tokens
    }

    /// Encode text without special tokens
    #[must_use]
    pub fn encode_without_special(&self, text: &str) -> Vec<u32> {
        text.bytes()
            .map(|b| u32::from(b) + self.byte_offset)
            .collect()
    }

    /// Decode tokens to text
    #[must_use]
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&t| {
                if t == self.special_tokens.bos
                    || t == self.special_tokens.eos
                    || t == self.special_tokens.pad
                {
                    None
                } else if t >= self.byte_offset && t < self.byte_offset + 256 {
                    Some((t - self.byte_offset) as u8 as char)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get vocabulary size
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        256 + 256 // Special tokens + byte tokens
    }
}

// =============================================================================
// BPE Tokenizer
// =============================================================================

/// BPE (Byte-Pair Encoding) tokenizer for LFM2
///
/// Supports loading vocabulary from HuggingFace tokenizer.json format.
#[derive(Debug, Clone)]
pub struct Lfm2Tokenizer {
    /// Token to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// Merge rules for BPE
    merges: Vec<(String, String)>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// Fallback byte tokenizer (reserved for future use)
    #[allow(dead_code)]
    byte_fallback: ByteLevelTokenizer,
}

impl Default for Lfm2Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lfm2Tokenizer {
    /// Create new tokenizer with empty vocabulary
    #[must_use]
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Add special tokens
        let special = SpecialTokens::default();
        vocab.insert("<pad>".to_string(), special.pad);
        vocab.insert("<s>".to_string(), special.bos);
        vocab.insert("</s>".to_string(), special.eos);
        vocab.insert("<unk>".to_string(), special.unk);

        id_to_token.insert(special.pad, "<pad>".to_string());
        id_to_token.insert(special.bos, "<s>".to_string());
        id_to_token.insert(special.eos, "</s>".to_string());
        id_to_token.insert(special.unk, "<unk>".to_string());

        // Add byte tokens (for fallback)
        for byte in 0u8..=255 {
            let token = format!("<0x{byte:02X}>");
            let id = u32::from(byte) + 256;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        Self {
            vocab,
            id_to_token,
            merges: Vec::new(),
            special_tokens: special,
            byte_fallback: ByteLevelTokenizer::default(),
        }
    }

    /// Load vocabulary from a list of tokens
    ///
    /// # Arguments
    /// * `tokens` - List of token strings in vocabulary order
    ///
    /// # Errors
    /// Returns error if vocabulary is invalid
    pub fn from_vocab(tokens: &[String]) -> WhisperResult<Self> {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in tokens.iter().enumerate() {
            let id = id as u32;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
        }

        // Use default special tokens (assumes standard ordering)
        let special = SpecialTokens::default();

        Ok(Self {
            vocab,
            id_to_token,
            merges: Vec::new(),
            special_tokens: special,
            byte_fallback: ByteLevelTokenizer::default(),
        })
    }

    /// Add BPE merge rules
    pub fn add_merges(&mut self, merges: Vec<(String, String)>) {
        self.merges = merges;
    }

    /// Encode text to token IDs
    ///
    /// Uses BPE if merges are available, otherwise falls back to byte encoding.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        tokens.push(self.special_tokens.bos);

        if self.merges.is_empty() {
            // No BPE merges, use byte-level encoding
            for ch in text.chars() {
                if let Some(&id) = self.vocab.get(&ch.to_string()) {
                    tokens.push(id);
                } else {
                    // Encode as bytes
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf);
                    for byte in encoded.bytes() {
                        tokens.push(u32::from(byte) + 256);
                    }
                }
            }
        } else {
            // BPE encoding
            let encoded = self.bpe_encode(text);
            tokens.extend(encoded);
        }

        tokens.push(self.special_tokens.eos);
        tokens
    }

    /// Encode without adding special tokens
    #[must_use]
    pub fn encode_without_special(&self, text: &str) -> Vec<u32> {
        if self.merges.is_empty() {
            // Byte-level fallback
            text.chars()
                .flat_map(|ch| {
                    if let Some(&id) = self.vocab.get(&ch.to_string()) {
                        vec![id]
                    } else {
                        ch.to_string().bytes().map(|b| u32::from(b) + 256).collect()
                    }
                })
                .collect()
        } else {
            self.bpe_encode(text)
        }
    }

    /// BPE encoding algorithm
    fn bpe_encode(&self, text: &str) -> Vec<u32> {
        // Start with characters
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply merges
        for (a, b) in &self.merges {
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if &tokens[i] == a && &tokens[i + 1] == b {
                    tokens[i] = format!("{a}{b}");
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Convert to IDs
        tokens
            .iter()
            .map(|t| {
                self.vocab
                    .get(t)
                    .copied()
                    .unwrap_or(self.special_tokens.unk)
            })
            .collect()
    }

    /// Decode token IDs to text
    #[must_use]
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();

        for &id in tokens {
            if id == self.special_tokens.bos
                || id == self.special_tokens.eos
                || id == self.special_tokens.pad
            {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&id) {
                // Handle byte tokens
                if token.starts_with("<0x") && token.ends_with('>') {
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        result.push(byte as char);
                        continue;
                    }
                }
                result.push_str(token);
            } else if (256..512).contains(&id) {
                // Byte fallback
                result.push((id - 256) as u8 as char);
            }
        }

        result
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Check if token ID is a special token
    #[must_use]
    pub fn is_special_token(&self, id: u32) -> bool {
        id == self.special_tokens.bos
            || id == self.special_tokens.eos
            || id == self.special_tokens.pad
            || id == self.special_tokens.unk
    }

    /// Get BOS token ID
    #[must_use]
    pub const fn bos_token_id(&self) -> u32 {
        self.special_tokens.bos
    }

    /// Get EOS token ID
    #[must_use]
    pub const fn eos_token_id(&self) -> u32 {
        self.special_tokens.eos
    }

    /// Get PAD token ID
    #[must_use]
    pub const fn pad_token_id(&self) -> u32 {
        self.special_tokens.pad
    }

    /// Load tokenizer from HuggingFace tokenizer.json format
    ///
    /// Parses the JSON structure to extract vocabulary and merge rules.
    ///
    /// # Arguments
    /// * `json_str` - Contents of tokenizer.json file
    ///
    /// # Errors
    /// Returns error if JSON is invalid or missing required fields
    pub fn from_huggingface_json(json_str: &str) -> WhisperResult<Self> {
        // Minimal JSON parsing without serde dependency
        // HuggingFace tokenizer.json structure:
        // {
        //   "model": {
        //     "vocab": { "token": id, ... },
        //     "merges": [ "a b", "c d", ... ]
        //   },
        //   "added_tokens": [ { "id": N, "content": "...", "special": true }, ... ]
        // }

        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut merges = Vec::new();
        let mut special = SpecialTokens::default();

        // Parse vocab section
        if let Some(vocab_start) = json_str.find("\"vocab\"") {
            if let Some(brace_start) = json_str[vocab_start..].find('{') {
                let vocab_section_start = vocab_start + brace_start;
                let mut depth = 0;
                let mut vocab_end = vocab_section_start;

                for (i, c) in json_str[vocab_section_start..].char_indices() {
                    match c {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                vocab_end = vocab_section_start + i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }

                let vocab_json = &json_str[vocab_section_start..vocab_end];
                parse_vocab_entries(vocab_json, &mut vocab, &mut id_to_token);
            }
        }

        // Parse merges section
        if let Some(merges_start) = json_str.find("\"merges\"") {
            if let Some(bracket_start) = json_str[merges_start..].find('[') {
                let merges_section_start = merges_start + bracket_start;
                if let Some(bracket_end) = json_str[merges_section_start..].find(']') {
                    let merges_json =
                        &json_str[merges_section_start..merges_section_start + bracket_end + 1];
                    parse_merge_entries(merges_json, &mut merges);
                }
            }
        }

        // Parse added_tokens for special token IDs
        if let Some(added_start) = json_str.find("\"added_tokens\"") {
            if let Some(bracket_start) = json_str[added_start..].find('[') {
                let added_section_start = added_start + bracket_start;
                let mut depth = 0;
                let mut added_end = added_section_start;

                for (i, c) in json_str[added_section_start..].char_indices() {
                    match c {
                        '[' => depth += 1,
                        ']' => {
                            depth -= 1;
                            if depth == 0 {
                                added_end = added_section_start + i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }

                let added_json = &json_str[added_section_start..added_end];
                parse_special_tokens(added_json, &mut special, &mut vocab, &mut id_to_token);
            }
        }

        Ok(Self {
            vocab,
            id_to_token,
            merges,
            special_tokens: special,
            byte_fallback: ByteLevelTokenizer::default(),
        })
    }

    /// Load tokenizer from a file path
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer.json file
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> WhisperResult<Self> {
        let json_str = std::fs::read_to_string(path)?;
        Self::from_huggingface_json(&json_str)
    }
}

// =============================================================================
// JSON Parsing Helpers (no serde dependency)
// =============================================================================

/// Parse vocab entries from JSON object string
fn parse_vocab_entries(
    json: &str,
    vocab: &mut HashMap<String, u32>,
    id_to_token: &mut HashMap<u32, String>,
) {
    // Simple parser for "token": id pairs
    let mut i = 0;
    let chars: Vec<char> = json.chars().collect();

    while i < chars.len() {
        // Find quoted token
        if chars[i] == '"' {
            let token_start = i + 1;
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' {
                    i += 1; // Skip escaped char
                }
                i += 1;
            }
            let token_end = i;
            i += 1; // Skip closing quote

            // Skip to colon
            while i < chars.len() && chars[i] != ':' {
                i += 1;
            }
            i += 1; // Skip colon

            // Skip whitespace
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }

            // Parse number
            let num_start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '-') {
                i += 1;
            }

            if num_start < i && token_start < token_end {
                let token: String = chars[token_start..token_end].iter().collect();
                let token = unescape_json_string(&token);
                let num_str: String = chars[num_start..i].iter().collect();
                if let Ok(id) = num_str.parse::<u32>() {
                    vocab.insert(token.clone(), id);
                    id_to_token.insert(id, token);
                }
            }
        } else {
            i += 1;
        }
    }
}

/// Parse merge entries from JSON array string
fn parse_merge_entries(json: &str, merges: &mut Vec<(String, String)>) {
    // Simple parser for ["a b", "c d", ...] format
    let mut in_string = false;
    let mut current = String::new();
    let mut escape_next = false;

    for c in json.chars() {
        if escape_next {
            current.push(c);
            escape_next = false;
            continue;
        }

        match c {
            '\\' if in_string => escape_next = true,
            '"' => {
                if in_string {
                    // End of merge string, parse "a b" into ("a", "b")
                    let parts: Vec<&str> = current.split(' ').collect();
                    if parts.len() == 2 {
                        merges.push((parts[0].to_string(), parts[1].to_string()));
                    }
                    current.clear();
                }
                in_string = !in_string;
            }
            _ if in_string => current.push(c),
            _ => {}
        }
    }
}

/// State for JSON key-value parsing
#[derive(Default)]
struct JsonParseState {
    current_id: Option<u32>,
    current_content: Option<String>,
    in_string: bool,
    escape_next: bool,
    current_key: String,
    current_value: String,
    in_key: bool,
    in_value: bool,
}

impl JsonParseState {
    /// Handle escaped character
    fn handle_escape(&mut self, c: char) {
        if self.in_string {
            self.current_value.push(c);
        }
        self.escape_next = false;
    }

    /// Handle quote character - toggle string state
    fn handle_quote(&mut self) {
        if self.in_string {
            self.in_key = false;
            self.in_value = false;
        } else if self.current_key.is_empty() && self.current_value.is_empty() {
            self.in_key = true;
        } else if !self.current_key.is_empty() {
            self.in_value = true;
        }
        self.in_string = !self.in_string;
    }

    /// Handle end of key-value pair (comma or close brace)
    fn handle_pair_end(&mut self) {
        if self.current_key == "id" {
            self.current_id = self.current_value.parse().ok();
        } else if self.current_key == "content" {
            self.current_content = Some(unescape_json_string(&self.current_value));
        }
        self.current_key.clear();
        self.current_value.clear();
    }

    /// Handle character inside a string
    fn handle_string_char(&mut self, c: char) {
        if self.in_key {
            self.current_key.push(c);
        } else if self.in_value {
            self.current_value.push(c);
        }
    }

    /// Finalize object and return (id, content) if valid
    fn finalize_object(&mut self) -> Option<(u32, String)> {
        let result = match (self.current_id, self.current_content.take()) {
            (Some(id), Some(content)) => Some((id, content)),
            _ => None,
        };
        self.current_id = None;
        result
    }
}

/// Map token content to special token type
fn update_special_token(special: &mut SpecialTokens, content: &str, id: u32) {
    match content {
        "<s>" | "<bos>" | "[CLS]" => special.bos = id,
        "</s>" | "<eos>" | "[SEP]" => special.eos = id,
        "<pad>" | "[PAD]" => special.pad = id,
        "<unk>" | "[UNK]" => special.unk = id,
        _ => {}
    }
}

/// Parse special tokens from added_tokens array
fn parse_special_tokens(
    json: &str,
    special: &mut SpecialTokens,
    vocab: &mut HashMap<String, u32>,
    id_to_token: &mut HashMap<u32, String>,
) {
    let mut state = JsonParseState::default();

    for c in json.chars() {
        if state.escape_next {
            state.handle_escape(c);
            continue;
        }

        match c {
            '\\' if state.in_string => state.escape_next = true,
            '"' => state.handle_quote(),
            ':' if !state.in_string => {} // Key complete, ready for value
            ',' if !state.in_string => state.handle_pair_end(),
            '}' if !state.in_string => {
                state.handle_pair_end();
                if let Some((id, content)) = state.finalize_object() {
                    update_special_token(special, &content, id);
                    vocab.insert(content.clone(), id);
                    id_to_token.insert(id, content);
                }
            }
            _ if state.in_string => state.handle_string_char(c),
            _ if !state.in_string && c.is_ascii_digit() && !state.current_key.is_empty() => {
                state.current_value.push(c);
            }
            _ => {}
        }
    }
}

/// Unescape JSON string (basic escape sequences)
fn unescape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('/') => result.push('/'),
                Some('u') => {
                    // Unicode escape \uXXXX
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = char::from_u32(cp) {
                            result.push(ch);
                        }
                    }
                }
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }

    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
}

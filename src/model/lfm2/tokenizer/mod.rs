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

#[cfg(test)]
mod tests;

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
                    if let Some(hex) = token.get(3..5) {
                        if let Ok(byte) = u8::from_str_radix(hex, 16) {
                            result.push(byte as char);
                            continue;
                        }
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
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut merges = Vec::new();
        let mut special = SpecialTokens::default();

        // Parse vocab section
        if let Some(section) = find_json_section(json_str, "\"vocab\"", '{', '}') {
            parse_vocab_entries(section, &mut vocab, &mut id_to_token);
        }

        // Parse merges section (simple bracket find — no nesting)
        if let Some(merges_start) = json_str.find("\"merges\"") {
            if let Some(bracket_start) = json_str[merges_start..].find('[') {
                let start = merges_start + bracket_start;
                if let Some(bracket_end) = json_str[start..].find(']') {
                    let section = &json_str[start..=start + bracket_end];
                    parse_merge_entries(section, &mut merges);
                }
            }
        }

        // Parse added_tokens for special token IDs
        if let Some(section) = find_json_section(json_str, "\"added_tokens\"", '[', ']') {
            parse_special_tokens(section, &mut special, &mut vocab, &mut id_to_token);
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

/// Find a balanced JSON section by key, matching open/close delimiters.
/// Returns the substring from the opening delimiter to the matching close.
fn find_json_section<'a>(json_str: &'a str, key: &str, open: char, close: char) -> Option<&'a str> {
    let key_pos = json_str.find(key)?;
    let after_key = &json_str[key_pos..];
    let delim_offset = after_key.find(open)?;
    let section_start = key_pos + delim_offset;

    let mut depth = 0;
    for (i, c) in json_str[section_start..].char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(&json_str[section_start..section_start + i + 1]);
            }
        }
    }
    None
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn parse_vocab_entries(
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
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn parse_merge_entries(json: &str, merges: &mut Vec<(String, String)>) {
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
#[allow(clippy::struct_excessive_bools)]
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
pub(crate) fn unescape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
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
                None | Some('\\') => result.push('\\'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

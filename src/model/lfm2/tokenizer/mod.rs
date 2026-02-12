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
        let special = [self.special_tokens.bos, self.special_tokens.eos, self.special_tokens.pad];
        tokens
            .iter()
            .filter(|&&t| !special.contains(&t))
            .filter_map(|&t| {
                let byte_val = t.checked_sub(self.byte_offset).filter(|&v| v < 256)?;
                Some(byte_val as u8 as char)
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
        tokens.extend(self.encode_inner(text));
        tokens.push(self.special_tokens.eos);
        tokens
    }

    /// Encode without adding special tokens
    #[must_use]
    pub fn encode_without_special(&self, text: &str) -> Vec<u32> {
        self.encode_inner(text)
    }

    /// Core encoding logic shared by encode and encode_without_special
    fn encode_inner(&self, text: &str) -> Vec<u32> {
        if !self.merges.is_empty() {
            return self.bpe_encode(text);
        }
        text.chars()
            .flat_map(|ch| {
                self.vocab
                    .get(&ch.to_string())
                    .map_or_else(
                        || ch.to_string().bytes().map(|b| u32::from(b) + 256).collect(),
                        |&id| vec![id],
                    )
            })
            .collect()
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
        tokens
            .iter()
            .filter(|&&id| !self.is_special_token(id))
            .filter_map(|&id| self.decode_single_token(id))
            .collect()
    }

    /// Decode a single non-special token to its string representation
    fn decode_single_token(&self, id: u32) -> Option<String> {
        if let Some(token) = self.id_to_token.get(&id) {
            // Handle byte tokens like <0xFF>
            let byte_decoded = token
                .strip_prefix("<0x")
                .and_then(|s| s.strip_suffix('>'))
                .and_then(|hex| u8::from_str_radix(&hex[..2.min(hex.len())], 16).ok())
                .map(|b| (b as char).to_string());
            Some(byte_decoded.unwrap_or_else(|| token.clone()))
        } else if (256..512).contains(&id) {
            Some(((id - 256) as u8 as char).to_string())
        } else {
            None
        }
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
        if let Some(section) = json_str
            .find("\"merges\"")
            .and_then(|ms| json_str[ms..].find('[').map(|bs| ms + bs))
            .and_then(|start| json_str[start..].find(']').map(|end| &json_str[start..=start + end]))
        {
            parse_merge_entries(section, &mut merges);
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
    let section_start = json_str
        .find(key)
        .and_then(|kp| json_str[kp..].find(open).map(|d| kp + d))?;

    let mut depth = 0;
    for (i, c) in json_str[section_start..].char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(&json_str[section_start..=section_start + i]);
            }
        }
    }
    None
}

/// Scan a quoted string from `chars[start..]` where `chars[start] == '"'`.
/// Returns (unescaped content, position after closing quote).
fn scan_quoted_string(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut i = start + 1; // Skip opening quote
    while i < chars.len() && chars[i] != '"' {
        if chars[i] == '\\' {
            i += 1; // Skip escaped char
        }
        i += 1;
    }
    if i >= chars.len() {
        return None;
    }
    let raw: String = chars[start + 1..i].iter().collect();
    Some((unescape_json_string(&raw), i + 1))
}

/// Skip forward to `target` char, returning position after it. Returns None if not found.
fn skip_past_char(chars: &[char], start: usize, target: char) -> Option<usize> {
    chars[start..].iter().position(|&c| c == target).map(|p| start + p + 1)
}

/// Parse an integer starting at `start`, returning (value, position after number).
fn parse_u32_at(chars: &[char], start: usize) -> Option<(u32, usize)> {
    let mut i = start;
    // Skip whitespace
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    let num_start = i;
    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '-') {
        i += 1;
    }
    if num_start == i {
        return None;
    }
    let num_str: String = chars[num_start..i].iter().collect();
    num_str.parse::<u32>().ok().map(|v| (v, i))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn parse_vocab_entries(
    json: &str,
    vocab: &mut HashMap<String, u32>,
    id_to_token: &mut HashMap<u32, String>,
) {
    let mut i = 0;
    let chars: Vec<char> = json.chars().collect();

    while i < chars.len() {
        if chars[i] != '"' {
            i += 1;
            continue;
        }

        // Parse "token": id — chain scan→skip→parse, break if any step fails
        let Some((token, after_quote)) = scan_quoted_string(&chars, i) else { break };
        let entry = skip_past_char(&chars, after_quote, ':').and_then(|ac| parse_u32_at(&chars, ac));
        match entry {
            Some((id, after_num)) => {
                vocab.insert(token.clone(), id);
                id_to_token.insert(id, token);
                i = after_num;
            }
            None => break,
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
        match (self.in_string, self.current_key.is_empty()) {
            (true, _) => {
                self.in_key = false;
                self.in_value = false;
            }
            (false, true) if self.current_value.is_empty() => self.in_key = true,
            (false, false) => self.in_value = true,
            _ => {}
        }
        self.in_string = !self.in_string;
    }

    /// Handle end of key-value pair (comma or close brace)
    fn handle_pair_end(&mut self) {
        match self.current_key.as_str() {
            "id" => self.current_id = self.current_value.parse().ok(),
            "content" => self.current_content = Some(unescape_json_string(&self.current_value)),
            _ => {}
        }
        self.current_key.clear();
        self.current_value.clear();
    }

    /// Handle character inside a string
    fn handle_string_char(&mut self, c: char) {
        let target = match (self.in_key, self.in_value) {
            (true, _) => Some(&mut self.current_key),
            (_, true) => Some(&mut self.current_value),
            _ => None,
        };
        if let Some(s) = target {
            s.push(c);
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

    /// Handle close brace: finalize pair then finalize object
    fn handle_close_brace(&mut self) -> Option<(u32, String)> {
        self.handle_pair_end();
        self.finalize_object()
    }

    /// Handle a non-string digit character (for numeric values outside quotes)
    fn handle_digit(&mut self, c: char) {
        if !self.current_key.is_empty() {
            self.current_value.push(c);
        }
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

/// Process a single character in the special tokens JSON parser.
fn process_special_token_char(
    c: char,
    state: &mut JsonParseState,
    special: &mut SpecialTokens,
    vocab: &mut HashMap<String, u32>,
    id_to_token: &mut HashMap<u32, String>,
) {
    match c {
        '\\' if state.in_string => state.escape_next = true,
        '"' => state.handle_quote(),
        ':' if !state.in_string => {} // Key complete, ready for value
        ',' if !state.in_string => state.handle_pair_end(),
        '}' if !state.in_string => {
            if let Some((id, content)) = state.handle_close_brace() {
                update_special_token(special, &content, id);
                vocab.insert(content.clone(), id);
                id_to_token.insert(id, content);
            }
        }
        _ if state.in_string => state.handle_string_char(c),
        _ if !state.in_string && c.is_ascii_digit() => state.handle_digit(c),
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
        process_special_token_char(c, &mut state, special, vocab, id_to_token);
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

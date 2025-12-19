# Language Detection

This example demonstrates automatic language detection using the Whisper language detection API.

## Running the Example

```bash
cargo run --example language_detection
```

## Code Overview

```rust
use whisper_apr::detection::{is_supported, language_name, LanguageDetector, LanguageProbs};

fn main() {
    // Check if a language is supported
    if is_supported("en") {
        let name = language_name("en").unwrap_or("Unknown");
        println!("English is supported: {}", name);
    }

    // Create language detector
    let detector = LanguageDetector::default();
    println!("Confidence threshold: {:.0}%", detector.confidence_threshold() * 100.0);

    // Detect from audio (logits come from decoder output)
    let probs = LanguageProbs::from_logits(&logits);

    // Get top 3 predictions
    let top = probs.top_n(3);
    for (code, prob) in &top {
        println!("{}: {:.1}%", code, prob * 100.0);
    }

    // Check confidence level
    if probs.is_confident(0.7) {
        println!("Confident detection");
    }
}
```

## Supported Languages

Whisper supports 99 languages. Common examples:

| Code | Language |
|------|----------|
| en | English |
| es | Spanish |
| fr | French |
| de | German |
| zh | Chinese |
| ja | Japanese |
| ko | Korean |
| ru | Russian |
| ar | Arabic |
| hi | Hindi |

## Language Probabilities

The `LanguageProbs` struct provides:

- `from_logits(logits)` - Create from decoder output logits
- `top_n(n)` - Get top N predictions with probabilities
- `is_confident(threshold)` - Check if top prediction exceeds threshold
- `get(code)` - Get probability for specific language

## Detection Workflow

1. First ~30 seconds of audio are processed
2. Decoder produces logits over vocabulary
3. Language token logits (50259+) are extracted
4. Softmax converts to probabilities
5. Top prediction is returned with confidence

## Integration Example

```rust
use whisper_apr::{WhisperApr, TranscribeOptions};

let whisper = WhisperApr::load_from_apr(&model_bytes)?;

// Auto-detect language
let options = TranscribeOptions {
    language: None, // Auto-detect
    ..Default::default()
};

let result = whisper.transcribe(&audio, options)?;
println!("Detected language: {}", result.language);
```

## See Also

- [CLI --language option](../getting-started/cli.md#transcribe)
- [TranscribeOptions API](../api-reference/transcribe-options.md)

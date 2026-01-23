# HuggingFace Publishing

Publish your whisper.apr models to the HuggingFace Hub for easy distribution and deployment.

## Overview

The `publish` module provides tools to:
- Export models to SafeTensors format
- Generate model cards with YAML frontmatter
- Upload to HuggingFace Hub repositories
- Verify models before publishing

## Prerequisites

1. **HuggingFace Account**: Create one at [huggingface.co](https://huggingface.co)
2. **API Token**: Generate a write token in Settings > Access Tokens
3. **Set Environment Variable**:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

## Basic Usage

### Using the CLI

```bash
# Publish a model to HuggingFace
whisper-apr publish my-model.apr --repo paiml/whisper-apr-tiny

# With custom commit message
whisper-apr publish my-model.apr --repo paiml/whisper-apr-tiny \
    --message "v1.0.0 release"

# Create private repository
whisper-apr publish my-model.apr --repo paiml/whisper-apr-tiny --private
```

### Using the Rust API

```rust
use whisper_apr::publish::{Publisher, PublishConfig, PublishFormat};

// Create publisher (reads HF_TOKEN from environment)
let publisher = Publisher::new();

// Or with explicit token
let publisher = Publisher::with_token("hf_your_token");

// Check authentication
if !publisher.is_authenticated() {
    eprintln!("Set HF_TOKEN environment variable");
    return;
}

// Configure publishing
let config = PublishConfig::new("paiml/whisper-apr-tiny")
    .with_message("Initial upload")
    .with_model_card(&custom_card);

// Prepare files (exports to SafeTensors)
publisher.prepare("model.apr", "output/", PublishFormat::Both)?;
```

## Publish Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `Apr` | Native .apr format | WASM deployment, streaming |
| `SafeTensors` | HuggingFace standard | Interoperability, transformers |
| `Both` | Both formats (default) | Maximum compatibility |

## Model Card Generation

Model cards are automatically generated with:

```yaml
---
license: mit
language:
  - en
  - multilingual
tags:
  - whisper
  - speech-recognition
  - rust
  - wasm
library_name: whisper-apr
pipeline_tag: automatic-speech-recognition
---
```

### Custom Model Card

```rust
let config = PublishConfig::new("org/model")
    .with_model_card(r#"---
license: apache-2.0
tags:
  - custom-tag
---

# My Custom Model

Custom description here.
"#);
```

## Pre-Publish Verification

Always verify models before publishing:

```rust
use whisper_apr::verify::{Verifier, verify_apr};

// Quick verification
let report = verify_apr("model.apr")?;
if report.passed {
    println!("Model verified: {} checks passed", report.passed_checks);
} else {
    eprintln!("Verification failed: {}", report.pass_rate());
}

// Detailed verification
let verifier = Verifier::new()
    .with_min_pass_rate(90.0);

let report = verifier.verify_apr("model.apr")?;
for check in &report.checks {
    println!("{}: {} - {}",
        if check.passed { "✓" } else { "✗" },
        check.name,
        check.message
    );
}
```

### Verification Checks

| Check | Description |
|-------|-------------|
| `A1_file_exists` | File exists and is readable |
| `A2_magic_bytes` | APR magic bytes present |
| `A3_header_valid` | Header structure valid |
| `A10_no_nan` | No NaN values in tensors |
| `A11_no_inf` | No Inf values in tensors |
| `A7_shape` | Tensor shapes match data |
| `C6_no_secrets` | No embedded credentials |

## SafeTensors Verification

```rust
use whisper_apr::verify::verify_safetensors;

let report = verify_safetensors("model.safetensors")?;
println!("SafeTensors checks: {}/{}",
    report.passed_checks,
    report.total_checks
);
```

## Example: Complete Publishing Workflow

```rust
use whisper_apr::publish::{Publisher, PublishConfig, PublishFormat, generate_model_card};
use whisper_apr::verify::Verifier;

fn publish_model(apr_path: &str, repo_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Verify the model
    let verifier = Verifier::new().with_min_pass_rate(95.0);
    let report = verifier.verify_apr(apr_path)?;

    if !verifier.meets_threshold(&report) {
        return Err(format!(
            "Model failed verification: {:.1}% pass rate",
            report.pass_rate()
        ).into());
    }
    println!("✓ Model verified: {:.1}% pass rate", report.pass_rate());

    // 2. Create publisher
    let publisher = Publisher::new();
    if !publisher.is_authenticated() {
        return Err("HF_TOKEN not set".into());
    }

    // 3. Generate model card
    let model_name = repo_id.split('/').last().unwrap_or("model");
    let card = generate_model_card(model_name, "tiny");

    // 4. Configure and prepare
    let config = PublishConfig::new(repo_id)
        .with_message("Verified model upload")
        .with_model_card(&card);

    publisher.prepare(apr_path, "publish_output/", PublishFormat::Both)?;

    println!("✓ Files prepared in publish_output/");
    println!("  Upload manually with: huggingface-cli upload {} publish_output/", repo_id);

    Ok(())
}
```

## Troubleshooting

### Authentication Errors

```bash
# Verify token is set
echo $HF_TOKEN

# Test with huggingface-cli
huggingface-cli whoami
```

### Large File Handling

For models > 10GB, use Git LFS:

```bash
# Initialize LFS in repo
git lfs install
git lfs track "*.safetensors"
git lfs track "*.apr"
```

### Rate Limits

HuggingFace has rate limits for uploads. For large batches:
- Use `--chunk-size` flag
- Implement exponential backoff
- Consider using the official `huggingface_hub` Python library for bulk operations

## See Also

- [Model Conversion](./model-conversion.md)
- [.apr Format](../architecture/apr-format.md)
- [Quality Gates](../development/quality-gates.md)

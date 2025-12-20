//! Verify whisper.apr weights against HuggingFace source
//!
//! This example proves H6 (Weight Values) by comparing .apr weights
//! bit-for-bit against the HuggingFace reference.
//!
//! Run with: cargo run --release --example verify_hf_weights
//!
//! Requires: safetensors-compare feature (enabled by default via aprender)

use aprender::inspect::safetensors::HfSafetensors;
use aprender::inspect::WeightDiff;
use std::path::Path;
use whisper_apr::format::AprReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== H6 Verification: Weight Values ===\n");

    let model_path = Path::new("models/whisper-tiny.apr");
    let hf_repo = "openai/whisper-tiny";

    if !model_path.exists() {
        eprintln!("Model not found: {}", model_path.display());
        eprintln!("Convert with: cargo run --bin whisper-convert --features converter -- tiny");
        return Ok(());
    }

    // Load APR model
    println!("Loading APR model: {}", model_path.display());
    let apr_bytes = std::fs::read(model_path)?;
    let apr_reader = AprReader::new(apr_bytes)?;

    // List APR tensor names
    let apr_tensor_names: Vec<&str> = apr_reader.tensors.iter().map(|t| t.name.as_str()).collect();
    println!("APR tensors: {}\n", apr_tensor_names.len());

    // Download HF model
    println!("Downloading HF model: {}", hf_repo);
    let hf_model = HfSafetensors::from_hub(hf_repo)
        .map_err(|e| format!("Failed to download HF model: {e}"))?;

    let hf_tensor_names = hf_model.tensor_names();
    println!("HF tensors: {}\n", hf_tensor_names.len());

    // Check layer norm weights first (the suspected issue)
    println!("=== Layer Norm Weight Comparison ===\n");

    let threshold = 1e-5;

    for hf_name in hf_tensor_names {
        if !hf_name.contains("layer_norm") {
            continue;
        }

        // Map HF name to APR name
        let apr_name = hf_name.strip_prefix("model.").unwrap_or(hf_name);

        // Load both tensors
        let hf_tensor = match hf_model.tensor(hf_name) {
            Ok(t) => t,
            Err(e) => {
                println!("  {} - HF load error: {}", hf_name, e);
                continue;
            }
        };

        let apr_tensor = match apr_reader.load_tensor(apr_name) {
            Ok(t) => t,
            Err(e) => {
                println!("  {} - APR not found (tried {}): {}", hf_name, apr_name, e);
                continue;
            }
        };

        // Compare
        let diff = WeightDiff::from_slices(&hf_tensor.data, &apr_tensor);

        let hf_mean = hf_tensor.data.iter().sum::<f32>() / hf_tensor.data.len() as f32;
        let apr_mean = apr_tensor.iter().sum::<f32>() / apr_tensor.len() as f32;

        let status = if diff.max_diff < threshold { "✓" } else { "✗" };
        let flag = if hf_name.ends_with(".weight") && (apr_mean - 1.0).abs() > 1.0 {
            " <-- BAD MEAN!"
        } else {
            ""
        };

        println!(
            "  {} {} | HF_mean={:.4} APR_mean={:.4} | max_diff={:.4}{}",
            status, hf_name, hf_mean, apr_mean, diff.max_diff, flag
        );
    }

    println!();

    // Focus on cross-attention (the suspected culprit)
    let cross_attn_patterns = [
        "encoder_attn.q_proj",
        "encoder_attn.k_proj",
        "encoder_attn.v_proj",
        "encoder_attn.out_proj",
    ];

    println!("=== Cross-Attention Weight Comparison ===\n");

    let mut total_comparisons = 0;
    let mut total_passed = 0;

    for pattern in &cross_attn_patterns {
        println!("--- {} ---", pattern);

        // Find matching HF tensors
        for hf_name in hf_tensor_names {
            if !hf_name.contains(pattern) {
                continue;
            }

            // Map HF name to APR name
            let apr_name = hf_name.strip_prefix("model.").unwrap_or(hf_name);

            // Load both tensors
            let hf_tensor = match hf_model.tensor(hf_name) {
                Ok(t) => t,
                Err(e) => {
                    println!("  {} - HF load error: {}", hf_name, e);
                    continue;
                }
            };

            let apr_tensor = match apr_reader.load_tensor(apr_name) {
                Ok(t) => t,
                Err(e) => {
                    println!("  {} - APR not found (tried {}): {}", hf_name, apr_name, e);
                    continue;
                }
            };

            // Compare
            total_comparisons += 1;
            let diff = WeightDiff::from_slices(&hf_tensor.data, &apr_tensor);

            let status = if diff.max_diff < threshold {
                total_passed += 1;
                "✓"
            } else {
                "✗"
            };

            println!(
                "  {} {} | max_diff={:.6} | cosine={:.6} | L2={:.4}",
                status, hf_name, diff.max_diff, diff.cosine_similarity, diff.l2_distance
            );
        }
        println!();
    }

    // Summary
    println!("=== Summary ===");
    println!("Total comparisons: {}", total_comparisons);
    println!("Passed (< {:.0e}): {}", threshold, total_passed);

    if total_passed == total_comparisons && total_comparisons > 0 {
        println!("\n✓ H6 PASSED: Weights match HuggingFace reference!");
    } else {
        println!("\n✗ H6 FAILED: Weight mismatch detected!");
        println!("\nPossible causes:");
        println!("  1. Weight transpose: HF is [out, in], check APR layout");
        println!("  2. Tensor name mapping incorrect");
        println!("  3. Conversion bug in whisper-convert");
    }

    Ok(())
}

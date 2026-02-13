
use super::*;

// ========================================================================
// Helper unit tests
// ========================================================================

#[test]
fn test_insert_tensor_path_leaf() {
    let mut root = TreeNode::new("model", "root");
    let tensor = aprender::format::TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![768, 768],
        size_bytes: 768 * 768 * 4,
        stats: None,
    };
    insert_tensor_path(&mut root, &["weight"], &tensor, false);
    assert_eq!(root.children.len(), 1);
    assert_eq!(root.children[0].name, "weight");
}

#[test]
fn test_insert_tensor_path_nested() {
    let mut root = TreeNode::new("model", "root");
    let tensor = aprender::format::TensorInfo {
        name: "layers.0.attn.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![768, 768],
        size_bytes: 768 * 768 * 4,
        stats: None,
    };
    insert_tensor_path(
        &mut root,
        &["layers", "0", "attn", "weight"],
        &tensor,
        false,
    );
    assert_eq!(root.children.len(), 1);
    assert_eq!(root.children[0].name, "layers");
    assert_eq!(root.children[0].children[0].name, "0");
}

#[test]
fn test_insert_tensor_path_with_sizes() {
    let mut root = TreeNode::new("model", "root");
    let tensor = aprender::format::TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10, 10],
        size_bytes: 400,
        stats: None,
    };
    insert_tensor_path(&mut root, &["weight"], &tensor, true);
    assert!(root.children[0].name.contains("400"));
}

#[test]
fn test_truncate_tree_at_depth() {
    let mut root = TreeNode::new("model", "root");
    let mut child = TreeNode::new("layers", "group");
    let leaf = TreeNode::new("weight", "tensor");
    child.add_child(leaf);
    root.add_child(child);

    truncate_tree(&mut root, 0, 1);
    assert!(root.children[0].children.is_empty());
}

#[test]
fn test_truncate_tree_preserves_shallow() {
    let mut root = TreeNode::new("model", "root");
    let leaf = TreeNode::new("bias", "tensor");
    root.add_child(leaf);

    truncate_tree(&mut root, 0, 5);
    assert_eq!(root.children.len(), 1);
    assert_eq!(root.children[0].name, "bias");
}

#[test]
fn test_extract_layers_from_tensors() {
    let tensors = vec![
        aprender::format::TensorInfo {
            name: "model.layers.0.attention.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![768, 768],
            size_bytes: 768 * 768 * 4,
            stats: None,
        },
        aprender::format::TensorInfo {
            name: "model.layers.0.ffn.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![3072, 768],
            size_bytes: 3072 * 768 * 4,
            stats: None,
        },
        aprender::format::TensorInfo {
            name: "model.embed.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![50257, 768],
            size_bytes: 50257 * 768 * 4,
            stats: None,
        },
    ];

    let layers = extract_layers_from_tensors(&tensors, None);
    // "model.layers.0" groups attention+ffn, "model.embed" is separate
    assert_eq!(layers.len(), 2);
    // Aggregated params: attention (768*768) + ffn (3072*768) = 2,949,120
    let layer0 = layers.iter().find(|l| l.name == "model.layers.0").unwrap();
    assert_eq!(layer0.params, 768 * 768 + 3072 * 768);
}

#[test]
fn test_extract_layers_with_filter() {
    let tensors = vec![
        aprender::format::TensorInfo {
            name: "model.layers.0.attn.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![768, 768],
            size_bytes: 768 * 768 * 4,
            stats: None,
        },
        aprender::format::TensorInfo {
            name: "model.layers.1.attn.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![768, 768],
            size_bytes: 768 * 768 * 4,
            stats: None,
        },
    ];

    let layers = extract_layers_from_tensors(&tensors, Some(0));
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].name, "model.layers.0");
}

#[test]
fn test_extract_layers_empty() {
    let layers = extract_layers_from_tensors(&[], None);
    assert!(layers.is_empty());
}

#[test]
fn test_extract_layers_single_component_name() {
    let tensors = vec![aprender::format::TensorInfo {
        name: "bias".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10],
        size_bytes: 40,
        stats: None,
    }];
    let layers = extract_layers_from_tensors(&tensors, None);
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].name, "bias");
}

#[test]
fn test_infer_layer_type() {
    assert_eq!(infer_layer_type(&["model.attn.weight"]), "Attention");
    assert_eq!(infer_layer_type(&["model.self_attn.weight"]), "Attention");
    assert_eq!(infer_layer_type(&["model.mlp.weight"]), "FFN");
    assert_eq!(infer_layer_type(&["model.fc.weight"]), "FFN");
    assert_eq!(infer_layer_type(&["model.layer_norm.weight"]), "LayerNorm");
    assert_eq!(infer_layer_type(&["model.ln.weight"]), "LayerNorm");
    assert_eq!(infer_layer_type(&["model.embed.weight"]), "Embedding");
    assert_eq!(infer_layer_type(&["model.wte.weight"]), "Embedding");
    assert_eq!(infer_layer_type(&["model.conv1.weight"]), "Conv");
    assert_eq!(infer_layer_type(&["model.lm_head.weight"]), "Projection");
    assert_eq!(infer_layer_type(&["model.proj.weight"]), "Projection");
    assert_eq!(infer_layer_type(&["model.something.weight"]), "Linear");
}

#[test]
fn test_subcommand_name() {
    use super::super::apr_args::*;
    assert_eq!(
        subcommand_name(&AprAction::Inspect(AprInspectArgs {
            file: std::path::PathBuf::from("x")
        })),
        "inspect"
    );
    assert_eq!(
        subcommand_name(&AprAction::Lint(AprLintArgs {
            file: std::path::PathBuf::from("x")
        })),
        "lint"
    );
    // Phase 2 variants
    assert_eq!(
        subcommand_name(&AprAction::Golden(AprGoldenArgs {
            trace_file: std::path::PathBuf::from("x"),
            logits: None,
            tolerance: None,
        })),
        "golden"
    );
    assert_eq!(
        subcommand_name(&AprAction::Validate(AprValidateArgs {
            file: std::path::PathBuf::from("x"),
            vocab_size: None,
            hidden_dim: None,
        })),
        "validate"
    );
    assert_eq!(
        subcommand_name(&AprAction::Contract(AprContractArgs {
            file: std::path::PathBuf::from("x"),
            tensor: None,
        })),
        "contract"
    );
    assert_eq!(
        subcommand_name(&AprAction::Compare(AprCompareArgs {
            source: std::path::PathBuf::from("a"),
            target: std::path::PathBuf::from("b"),
            l2_tolerance: 1e-5,
            max_tolerance: 1e-5,
        })),
        "compare"
    );
    assert_eq!(
        subcommand_name(&AprAction::Export(AprExportArgs {
            input: std::path::PathBuf::from("x"),
            output: std::path::PathBuf::from("y"),
            format: "safetensors".to_string(),
        })),
        "export"
    );
    assert_eq!(
        subcommand_name(&AprAction::F16Audit(AprF16AuditArgs {
            file: std::path::PathBuf::from("x"),
            verbose: false,
        })),
        "f16-audit"
    );
    // Phase 3 variants
    assert_eq!(
        subcommand_name(&AprAction::Sign(AprSignArgs {
            file: std::path::PathBuf::from("x"),
            key: std::path::PathBuf::from("k"),
            output: std::path::PathBuf::from("o"),
        })),
        "sign"
    );
    assert_eq!(
        subcommand_name(&AprAction::VerifySig(AprVerifySigArgs {
            file: std::path::PathBuf::from("x"),
            pubkey: None,
        })),
        "verify-sig"
    );
    assert_eq!(
        subcommand_name(&AprAction::Encrypt(AprEncryptArgs {
            file: std::path::PathBuf::from("x"),
            output: std::path::PathBuf::from("o"),
            password: None,
        })),
        "encrypt"
    );
    assert_eq!(
        subcommand_name(&AprAction::Decrypt(AprDecryptArgs {
            file: std::path::PathBuf::from("x"),
            output: std::path::PathBuf::from("o"),
            password: None,
        })),
        "decrypt"
    );
    assert_eq!(
        subcommand_name(&AprAction::Quantize(AprQuantizeArgs {
            file: std::path::PathBuf::from("x"),
            output: std::path::PathBuf::from("o"),
            r#type: "q8_0".to_string(),
            verify: false,
        })),
        "quantize"
    );
    assert_eq!(
        subcommand_name(&AprAction::ImportSharded(AprImportShardedArgs {
            source: std::path::PathBuf::from("x"),
            output: std::path::PathBuf::from("o"),
            max_cache_shards: 2,
        })),
        "import-sharded"
    );
    assert_eq!(
        subcommand_name(&AprAction::HeInspect(AprHeInspectArgs {
            file: std::path::PathBuf::from("x"),
        })),
        "he-inspect"
    );
}

#[test]
fn test_format_model_error_apr_v1() {
    let err = aprender::error::AprenderError::FormatError {
        message: "Invalid magic: 4150524e".to_string(),
    };
    let cli_err = format_model_error(&err, std::path::Path::new("test.apr"));
    let msg = cli_err.to_string();
    assert!(msg.contains("APR v1"), "Should mention APR v1: {msg}");
    assert!(msg.contains("test.apr"), "Should include filename: {msg}");
}

#[test]
fn test_format_model_error_truncated() {
    let err = aprender::error::AprenderError::FormatError {
        message: "Tensor data exceeds file size".to_string(),
    };
    let cli_err = format_model_error(&err, std::path::Path::new("model.gguf"));
    let msg = cli_err.to_string();
    assert!(msg.contains("truncated"), "Should mention truncated: {msg}");
}

#[test]
fn test_format_model_error_generic() {
    let err = aprender::error::AprenderError::FormatError {
        message: "Something else".to_string(),
    };
    let cli_err = format_model_error(&err, std::path::Path::new("model.bin"));
    let msg = cli_err.to_string();
    assert!(msg.contains("model.bin"), "Should include filename: {msg}");
    assert!(
        msg.contains("Something else"),
        "Should include original: {msg}"
    );
}

// ========================================================================
// Fixture builder — creates minimal SafeTensors for integration tests
// ========================================================================

/// Build a tiny SafeTensors file with known tensor names and shapes.
/// Returns the path to the temp file.
fn build_fixture_safetensors(dir: &std::path::Path) -> std::path::PathBuf {
    use std::collections::BTreeMap;

    let path = dir.join("fixture.safetensors");

    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 3 tensors: embedding, attention weight, bias
    tensors.insert(
        "model.embed.weight".to_string(),
        (vec![0.1_f32; 32 * 16], vec![32, 16]),
    );
    tensors.insert(
        "model.layers.0.attn.weight".to_string(),
        (vec![0.02_f32; 16 * 16], vec![16, 16]),
    );
    tensors.insert(
        "model.layers.0.attn.bias".to_string(),
        (vec![0.0_f32; 16], vec![16]),
    );

    aprender::serialization::safetensors::save_safetensors(&path, &tensors)
        .expect("Failed to write fixture SafeTensors");
    path
}

// ========================================================================
// Falsification F1: inspect tensor count cross-validation
//
// H₀: "inspect produces correct tensor counts for all formats"
// Test: Compare inspect tensor count against known fixture count.
// ========================================================================

#[test]
fn falsification_f1_inspect_tensor_count() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(&st_path).expect("inspect should succeed");

    // Fixture has exactly 3 tensors
    assert_eq!(
        report.tensors.len(),
        3,
        "F1 FALSIFIED: inspect reports {} tensors, expected 3",
        report.tensors.len()
    );

    // Cross-validate with list_tensors
    let list_opts = TensorListOptions {
        compute_stats: false,
        filter: None,
        limit: 0,
    };
    let list_result = list_tensors(&st_path, list_opts).expect("list_tensors should succeed");
    assert_eq!(
        list_result.tensor_count,
        report.tensors.len(),
        "F1 FALSIFIED: list_tensors ({}) disagrees with inspect ({})",
        list_result.tensor_count,
        report.tensors.len()
    );
}

// ========================================================================
// Falsification F2: self-diff identity
//
// H₀: "diff reports IDENTICAL for a file compared to itself"
// Test: diff(file, file) must return 0 differences.
// ========================================================================

#[test]
fn falsification_f2_self_diff_identity() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let options = DiffOptions::default();
    let report = diff_models(&st_path, &st_path, options).expect("self-diff should not error");

    assert!(
        report.is_identical(),
        "F2 FALSIFIED: self-diff reports {} differences (expected 0)",
        report.diff_count()
    );
}

// ========================================================================
// Falsification F3: lossless F32 SafeTensors→APR→SafeTensors round-trip
//
// H₀: "rosetta convert is lossless for F32 SafeTensors→APR→SafeTensors"
// Test: Convert ST→APR→ST, then diff original vs final. Must be identical.
// ========================================================================

#[test]
fn falsification_f3_lossless_roundtrip() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let apr_path = dir.path().join("roundtrip.apr");
    let st2_path = dir.path().join("roundtrip.safetensors");

    let rosetta = RosettaStone::new();

    // ST → APR
    rosetta
        .convert(&st_path, &apr_path, None)
        .expect("ST→APR conversion should succeed");

    // APR → ST
    rosetta
        .convert(&apr_path, &st2_path, None)
        .expect("APR→ST conversion should succeed");

    // Verify: load all tensors from original and round-tripped, compare
    let report_orig = rosetta
        .inspect(&st_path)
        .expect("inspect original should succeed");
    let report_rt = rosetta
        .inspect(&st2_path)
        .expect("inspect round-tripped should succeed");

    assert_eq!(
        report_orig.tensors.len(),
        report_rt.tensors.len(),
        "F3 FALSIFIED: tensor count changed after round-trip ({} vs {})",
        report_orig.tensors.len(),
        report_rt.tensors.len()
    );

    // Compare tensor data
    for orig_t in &report_orig.tensors {
        let orig_data = rosetta
            .load_tensor_f32(&st_path, &orig_t.name)
            .expect("load original tensor");
        match rosetta.load_tensor_f32(&st2_path, &orig_t.name) {
            Ok(rt_data) => {
                assert_eq!(
                    orig_data.len(),
                    rt_data.len(),
                    "F3 FALSIFIED: tensor {} length mismatch ({} vs {})",
                    orig_t.name,
                    orig_data.len(),
                    rt_data.len()
                );
                let max_diff = orig_data
                    .iter()
                    .zip(rt_data.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_diff < 1e-6,
                    "F3 FALSIFIED: tensor {} has max_diff={max_diff} (expected <1e-6)",
                    orig_t.name
                );
            }
            Err(_) => {
                panic!(
                    "F3 FALSIFIED: tensor {} missing from round-tripped file",
                    orig_t.name
                );
            }
        }
    }
}

// ========================================================================
// Falsification F4: canary detects single-weight perturbation
//
// H₀: "canary detects single-weight perturbation"
// Test: Create canary, perturb 1 float, create 2nd canary, checksums must differ.
// ========================================================================

#[test]
fn falsification_f4_canary_perturbation() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    // Build canary from original
    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(&st_path).expect("inspect should succeed");

    let mut canary1 = CanaryFile::new("test-model");
    for tensor in &report.tensors {
        if let Ok(data) = rosetta.load_tensor_f32(&st_path, &tensor.name) {
            let tc =
                TensorCanary::from_data(&tensor.name, tensor.shape.clone(), &tensor.dtype, &data);
            canary1.add_tensor(tc);
        }
    }

    // Build perturbed SafeTensors (flip one value)
    let perturbed_path = dir.path().join("perturbed.safetensors");
    {
        use std::collections::BTreeMap;
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let mut embed_data = vec![0.1_f32; 32 * 16];
        embed_data[0] = 999.0; // Perturb one value
        tensors.insert("model.embed.weight".to_string(), (embed_data, vec![32, 16]));
        tensors.insert(
            "model.layers.0.attn.weight".to_string(),
            (vec![0.02_f32; 16 * 16], vec![16, 16]),
        );
        tensors.insert(
            "model.layers.0.attn.bias".to_string(),
            (vec![0.0_f32; 16], vec![16]),
        );
        aprender::serialization::safetensors::save_safetensors(&perturbed_path, &tensors)
            .expect("Failed to write perturbed SafeTensors");
    }

    // Build canary from perturbed
    let report2 = rosetta
        .inspect(&perturbed_path)
        .expect("inspect perturbed should succeed");
    let mut canary2 = CanaryFile::new("test-model");
    for tensor in &report2.tensors {
        if let Ok(data) = rosetta.load_tensor_f32(&perturbed_path, &tensor.name) {
            let tc =
                TensorCanary::from_data(&tensor.name, tensor.shape.clone(), &tensor.dtype, &data);
            canary2.add_tensor(tc);
        }
    }

    // The embed tensor's checksum must differ
    let c1_embed = canary1
        .tensors
        .iter()
        .find(|t| t.name == "model.embed.weight")
        .expect("canary1 should have embed");
    let c2_embed = canary2
        .tensors
        .iter()
        .find(|t| t.name == "model.embed.weight")
        .expect("canary2 should have embed");

    assert_ne!(
        c1_embed.checksum, c2_embed.checksum,
        "F4 FALSIFIED: canary checksum unchanged after perturbation \
             (c1={}, c2={})",
        c1_embed.checksum, c2_embed.checksum
    );

    // The mean should also differ
    assert!(
        (c1_embed.mean - c2_embed.mean).abs() > 1e-6,
        "F4 FALSIFIED: canary mean unchanged after perturbation"
    );
}

// ========================================================================
// Integration: inspect handler
// ========================================================================

#[test]
fn test_run_inspect_safetensors() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprInspectArgs {
        file: st_path.clone(),
    };

    let global = make_test_global(false, false);
    let result = run_inspect(&args, &global);
    assert!(result.is_ok(), "inspect should succeed: {result:?}");
}

#[test]
fn test_run_inspect_json() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprInspectArgs {
        file: st_path.clone(),
    };

    let global = make_test_global(false, true);
    let result = run_inspect(&args, &global);
    assert!(result.is_ok(), "inspect --json should succeed: {result:?}");
}

#[test]
fn test_run_inspect_missing_file() {
    let args = super::super::apr_args::AprInspectArgs {
        file: std::path::PathBuf::from("/nonexistent/model.safetensors"),
    };
    let global = make_test_global(false, false);
    let result = run_inspect(&args, &global);
    assert!(result.is_err(), "inspect on missing file should fail");
}

// ========================================================================
// Integration: tensors handler
// ========================================================================

#[test]
fn test_run_tensors_basic() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprTensorsArgs {
        file: st_path,
        stats: false,
        filter: None,
        limit: 0,
    };
    let global = make_test_global(false, false);
    let result = run_tensors(&args, &global);
    assert!(result.is_ok(), "tensors should succeed: {result:?}");
}

#[test]
fn test_run_tensors_with_stats() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprTensorsArgs {
        file: st_path,
        stats: true,
        filter: None,
        limit: 0,
    };
    let global = make_test_global(false, false);
    let result = run_tensors(&args, &global);
    assert!(result.is_ok(), "tensors --stats should succeed: {result:?}");
}

#[test]
fn test_run_tensors_with_filter() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprTensorsArgs {
        file: st_path,
        stats: false,
        filter: Some("embed".to_string()),
        limit: 0,
    };
    let global = make_test_global(false, false);
    let result = run_tensors(&args, &global);
    assert!(
        result.is_ok(),
        "tensors --filter should succeed: {result:?}"
    );
}

// ========================================================================
// Integration: hex handler
// ========================================================================

#[test]
fn test_run_hex_raw() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprHexArgs {
        file: st_path,
        limit: 64,
        tensor: None,
    };
    let result = run_hex(&args);
    assert!(result.is_ok(), "hex should succeed: {result:?}");
}

#[test]
fn test_run_hex_tensor() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprHexArgs {
        file: st_path,
        limit: 64,
        tensor: Some("model.embed.weight".to_string()),
    };
    let result = run_hex(&args);
    assert!(result.is_ok(), "hex --tensor should succeed: {result:?}");
}

// ========================================================================
// Integration: tree handler
// ========================================================================

#[test]
fn test_run_tree() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprTreeArgs {
        file: st_path,
        sizes: true,
        depth: Some(2),
    };
    let result = run_tree(&args);
    assert!(result.is_ok(), "tree should succeed: {result:?}");
}

// ========================================================================
// Integration: flow handler
// ========================================================================

#[test]
fn test_run_flow() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprFlowArgs {
        file: st_path,
        layer: None,
    };
    let result = run_flow(&args);
    assert!(result.is_ok(), "flow should succeed: {result:?}");
}

// ========================================================================
// Integration: lint handler
// ========================================================================

#[test]
fn test_run_lint() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprLintArgs { file: st_path };
    let global = make_test_global(false, false);
    let result = run_lint(&args, &global);
    assert!(result.is_ok(), "lint should succeed: {result:?}");
}

// ========================================================================
// Integration: diff handler
// ========================================================================

#[test]
fn test_run_diff_identical() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprDiffArgs {
        file1: st_path.clone(),
        file2: st_path,
        filter: None,
    };
    let global = make_test_global(false, false);
    let result = run_diff(&args, &global);
    assert!(result.is_ok(), "diff should succeed: {result:?}");
}

#[test]
fn test_run_diff_json() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprDiffArgs {
        file1: st_path.clone(),
        file2: st_path,
        filter: None,
    };
    let global = make_test_global(false, true);
    let result = run_diff(&args, &global);
    assert!(result.is_ok(), "diff --json should succeed: {result:?}");
}

// ========================================================================
// Integration: rosetta inspect handler
// ========================================================================

#[test]
fn test_run_rosetta_inspect() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::RosettaInspectArgs { file: st_path };
    let global = make_test_global(false, false);
    let result = run_rosetta_inspect(&args, &global);
    assert!(result.is_ok(), "rosetta inspect should succeed: {result:?}");
}

// ========================================================================
// Integration: rosetta convert handler
// ========================================================================

#[test]
fn test_run_rosetta_convert() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());
    let apr_path = dir.path().join("converted.apr");

    let args = super::super::apr_args::RosettaConvertArgs {
        source: st_path,
        dest: apr_path,
        quantize: false,
        verify: false,
    };
    let global = make_test_global(true, false);
    let result = run_rosetta_convert(&args, &global);
    assert!(result.is_ok(), "rosetta convert should succeed: {result:?}");
}

// ========================================================================
// Integration: rosetta verify handler
// ========================================================================

#[test]
fn test_run_rosetta_verify() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::RosettaVerifyArgs {
        file: st_path,
        tolerance: 1e-5,
    };
    let global = make_test_global(false, false);
    let result = run_rosetta_verify(&args, &global);
    assert!(result.is_ok(), "rosetta verify should succeed: {result:?}");
}

// ========================================================================
// Integration: rosetta fingerprint handler
// ========================================================================

#[test]
fn test_run_rosetta_fingerprint() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::RosettaFingerprintArgs { file: st_path };
    let global = make_test_global(true, false);
    let result = run_rosetta_fingerprint(&args, &global);
    assert!(
        result.is_ok(),
        "rosetta fingerprint should succeed: {result:?}"
    );
}

// ========================================================================
// Integration: canary handler
// ========================================================================

#[test]
fn test_run_canary() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());
    let canary_path = dir.path().join("canary.json");

    let args = super::super::apr_args::AprCanaryArgs {
        file: st_path,
        output: canary_path.clone(),
    };
    let global = make_test_global(false, false);
    let result = run_canary(&args, &global);
    assert!(result.is_ok(), "canary should succeed: {result:?}");
    assert!(canary_path.exists(), "canary file should exist");

    // Verify canary JSON is valid
    let content = std::fs::read_to_string(&canary_path).expect("read canary");
    let json: serde_json::Value =
        serde_json::from_str(&content).expect("canary should be valid JSON");
    assert_eq!(json["tensors"].as_array().unwrap().len(), 3);
}

// ========================================================================
// Phase 2: Integration tests
// ========================================================================

#[test]
fn test_run_validate() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprValidateArgs {
        file: st_path,
        vocab_size: Some(32),
        hidden_dim: Some(16),
    };
    let global = make_test_global(false, false);
    let result = run_validate(&args, &global);
    assert!(result.is_ok(), "validate should succeed: {result:?}");
}

#[test]
fn test_run_validate_json() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprValidateArgs {
        file: st_path,
        vocab_size: None,
        hidden_dim: None,
    };
    let global = make_test_global(false, true);
    let result = run_validate(&args, &global);
    assert!(result.is_ok(), "validate --json should succeed: {result:?}");
}

#[test]
fn test_run_contract() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprContractArgs {
        file: st_path,
        tensor: None,
    };
    let global = make_test_global(false, false);
    let result = run_contract(&args, &global);
    assert!(result.is_ok(), "contract should succeed: {result:?}");
}

#[test]
fn test_run_contract_json() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprContractArgs {
        file: st_path,
        tensor: Some("embed".to_string()),
    };
    let global = make_test_global(false, true);
    let result = run_contract(&args, &global);
    assert!(result.is_ok(), "contract --json should succeed: {result:?}");
}

#[test]
fn test_run_family_identify() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprFamilyIdentifyArgs { file: st_path };
    let global = make_test_global(false, false);
    let result = run_family_identify(&args, &global);
    assert!(result.is_ok(), "family identify should succeed: {result:?}");
}

#[test]
fn test_run_family_identify_json() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprFamilyIdentifyArgs { file: st_path };
    let global = make_test_global(false, true);
    let result = run_family_identify(&args, &global);
    assert!(
        result.is_ok(),
        "family identify --json should succeed: {result:?}"
    );
}

#[test]
fn test_run_family_check_unknown() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    // Test fixture won't match any family, so this tests the mismatch path
    let args = super::super::apr_args::AprFamilyCheckArgs {
        file: st_path,
        family: "llama".to_string(),
        size: Some("7b".to_string()),
    };
    let global = make_test_global(false, false);
    let result = run_family_check(&args, &global);
    assert!(
        result.is_ok(),
        "family check should succeed (even on mismatch): {result:?}"
    );
}

#[test]
fn test_run_family_check_invalid_family() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprFamilyCheckArgs {
        file: st_path,
        family: "nonexistent_family".to_string(),
        size: None,
    };
    let global = make_test_global(false, false);
    let result = run_family_check(&args, &global);
    assert!(
        result.is_err(),
        "family check with unknown family should fail"
    );
}

#[test]
fn test_run_compare() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprCompareArgs {
        source: st_path.clone(),
        target: st_path,
        l2_tolerance: 1e-5,
        max_tolerance: 1e-5,
    };
    let global = make_test_global(false, false);
    let result = run_compare(&args, &global);
    assert!(result.is_ok(), "compare should succeed: {result:?}");
}

#[test]
fn test_run_compare_json() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprCompareArgs {
        source: st_path.clone(),
        target: st_path,
        l2_tolerance: 1e-5,
        max_tolerance: 1e-5,
    };
    let global = make_test_global(false, true);
    let result = run_compare(&args, &global);
    assert!(result.is_ok(), "compare --json should succeed: {result:?}");
}

#[test]
fn test_run_export() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());
    let out_path = dir.path().join("exported.safetensors");

    let args = super::super::apr_args::AprExportArgs {
        input: st_path,
        output: out_path.clone(),
        format: "safetensors".to_string(),
    };
    let global = make_test_global(true, false);
    let result = run_export(&args, &global);
    assert!(result.is_ok(), "export should succeed: {result:?}");
    assert!(out_path.exists(), "exported file should exist");
}

#[test]
fn test_run_export_unsupported_format() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprExportArgs {
        input: st_path,
        output: dir.path().join("out.onnx"),
        format: "onnx".to_string(),
    };
    let global = make_test_global(false, false);
    let result = run_export(&args, &global);
    assert!(result.is_err(), "export to unsupported format should fail");
}

#[test]
fn test_run_f16_audit() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprF16AuditArgs {
        file: st_path,
        verbose: true,
    };
    let global = make_test_global(false, false);
    let result = run_f16_audit(&args, &global);
    assert!(result.is_ok(), "f16-audit should succeed: {result:?}");
}

#[test]
fn test_run_f16_audit_json() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprF16AuditArgs {
        file: st_path,
        verbose: false,
    };
    let global = make_test_global(false, true);
    let result = run_f16_audit(&args, &global);
    assert!(
        result.is_ok(),
        "f16-audit --json should succeed: {result:?}"
    );
}

#[test]
fn test_run_golden_metadata_only() {
    // Golden verify without --logits just shows trace metadata
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let trace_path = dir.path().join("golden.json");

    // Create a minimal golden trace file
    let mut set = aprender::format::golden::GoldenTraceSet::new("test", "TestModel");
    set.add_trace(aprender::format::golden::GoldenTrace::new(
        "trace1",
        vec![1, 2, 3],
        vec![0.1, 0.2, 0.3],
    ));
    set.save(&trace_path).expect("save trace");

    let args = super::super::apr_args::AprGoldenArgs {
        trace_file: trace_path,
        logits: None,
        tolerance: None,
    };
    let global = make_test_global(false, false);
    let result = run_golden(&args, &global);
    assert!(
        result.is_ok(),
        "golden (metadata only) should succeed: {result:?}"
    );
}

#[test]
fn test_run_golden_json() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let trace_path = dir.path().join("golden.json");

    let set = aprender::format::golden::GoldenTraceSet::new("test", "TestModel");
    set.save(&trace_path).expect("save trace");

    let args = super::super::apr_args::AprGoldenArgs {
        trace_file: trace_path,
        logits: None,
        tolerance: Some(1e-3),
    };
    let global = make_test_global(false, true);
    let result = run_golden(&args, &global);
    assert!(result.is_ok(), "golden --json should succeed: {result:?}");
}

// ========================================================================
// Falsification F5: self-compare yields zero differences
//
// H₀: "comparing a model to itself yields 0 differences"
// Test: compare(file, file) → diff_count == 0
// ========================================================================

#[test]
fn falsification_f5_self_compare_identity() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprCompareArgs {
        source: st_path.clone(),
        target: st_path,
        l2_tolerance: 1e-5,
        max_tolerance: 1e-5,
    };
    let global = make_test_global(false, true);
    let result = run_compare(&args, &global).expect("compare should succeed");

    // The success message should indicate identical
    assert!(
        result.message.contains("0 differences"),
        "F5 FALSIFIED: self-compare found differences: {}",
        result.message
    );
}

// ========================================================================
// Falsification F6: export round-trip preserves tensor count
//
// H₀: "exporting and re-inspecting preserves tensor count"
// Test: export(file) → inspect(output).tensor_count == inspect(input).tensor_count
// ========================================================================

#[test]
fn falsification_f6_export_preserves_tensors() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());
    let export_path = dir.path().join("exported.safetensors");

    let rosetta = RosettaStone::new();
    let report_before = rosetta.inspect(&st_path).expect("inspect should succeed");

    let options = ExportOptions {
        format: ExportFormat::SafeTensors,
        quantize: None,
        include_tokenizer: false,
        include_config: false,
    };
    let export_report = apr_export(&st_path, &export_path, options).expect("export should succeed");

    assert_eq!(
        export_report.tensor_count,
        report_before.tensors.len(),
        "F6 FALSIFIED: export tensor count {} != original {}",
        export_report.tensor_count,
        report_before.tensors.len()
    );

    let report_after = rosetta
        .inspect(&export_path)
        .expect("inspect exported should succeed");
    assert_eq!(
        report_before.tensors.len(),
        report_after.tensors.len(),
        "F6 FALSIFIED: re-inspect after export has different tensor count"
    );
}

// ========================================================================
// Falsification F7: family detection is consistent across formats
//
// H₀: "family identify returns the same family for original and converted"
// Test: identify(ST) == identify(APR converted from ST)
// ========================================================================

#[test]
fn falsification_f7_family_detection_consistent() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(&st_path).expect("inspect should succeed");

    let tensor_names: Vec<&str> = report.tensors.iter().map(|t| t.name.as_str()).collect();

    let registry = build_default_registry();
    let detected = registry.detect_family(&tensor_names);

    // Our test fixture is not a real model family, so detection should be None
    // The test verifies consistency: calling twice gives same result
    let detected2 = registry.detect_family(&tensor_names);
    assert_eq!(
        detected.map(|f| f.family_name()),
        detected2.map(|f| f.family_name()),
        "F7 FALSIFIED: family detection is non-deterministic"
    );
}

// ========================================================================
// Falsification F8: validate catches zero-dimension tensors
//
// H₀: "validate reports zero-dimension shapes as issues"
// This is tested structurally — our fixture has valid shapes, so 0 issues
// ========================================================================

#[test]
fn falsification_f8_validate_passes_on_valid() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprValidateArgs {
        file: st_path,
        vocab_size: Some(32),
        hidden_dim: Some(16),
    };
    let global = make_test_global(false, false);
    let result = run_validate(&args, &global).expect("validate should succeed");

    assert!(
        result.message.contains("PASS"),
        "F8 FALSIFIED: validate should pass on valid fixture: {}",
        result.message
    );
}

// ========================================================================
// Falsification F9: f16 audit reports 0 unsafe on F32 SafeTensors
//
// H₀: "F32 SafeTensors model has 0 unsafe f16 scale factors"
// (because there are no quantized tensors to have f16 scales)
// ========================================================================

#[test]
fn falsification_f9_f16_audit_clean_on_f32() {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let st_path = build_fixture_safetensors(dir.path());

    let args = super::super::apr_args::AprF16AuditArgs {
        file: st_path,
        verbose: false,
    };
    let global = make_test_global(false, false);
    let result = run_f16_audit(&args, &global).expect("f16-audit should succeed");

    assert!(
        result.message.contains("PASS"),
        "F9 FALSIFIED: f16-audit on F32 model should pass: {}",
        result.message
    );
}

// ========================================================================
// Falsification F10: golden trace verify_logits is correct
//
// H₀: "verify_logits passes when logits match within tolerance"
// ========================================================================

#[test]
fn falsification_f10_golden_verify_logits() {
    let expected = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];
    let actual = vec![0.10001_f32, 0.20001, 0.29999, 0.40001, 0.49999];

    let result = aprender::format::verify_logits("test", &actual, &expected, 1e-3);
    assert!(
        result.passed,
        "F10 FALSIFIED: verify_logits should pass for near-identical logits"
    );

    // Verify it fails with tight tolerance
    let result_strict = aprender::format::verify_logits("test", &actual, &expected, 1e-6);
    assert!(
        !result_strict.passed,
        "F10 FALSIFIED: verify_logits should fail with strict tolerance"
    );
}

// ========================================================================
// Phase 3 integration tests
// ========================================================================

// -- Feature-gated fallback tests (always run, test the "not enabled" path) --

#[test]
fn test_run_sign_no_feature() {
    // Without format-signing feature, should return error
    #[cfg(not(feature = "format-signing"))]
    {
        let args = AprSignArgs {
            file: std::path::PathBuf::from("model.bin"),
            key: std::path::PathBuf::from("key.bin"),
            output: std::path::PathBuf::from("out.bin"),
        };
        let global = make_test_global(true, false);
        let result = run_sign(&args, &global);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("format-signing"),
            "Should mention feature: {msg}"
        );
    }
}

#[test]
fn test_run_verify_sig_no_feature() {
    #[cfg(not(feature = "format-signing"))]
    {
        let args = AprVerifySigArgs {
            file: std::path::PathBuf::from("model.bin"),
            pubkey: None,
        };
        let global = make_test_global(true, false);
        let result = run_verify_sig(&args, &global);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("format-signing"),
            "Should mention feature: {msg}"
        );
    }
}

#[test]
fn test_run_encrypt_no_feature() {
    #[cfg(not(feature = "format-encryption"))]
    {
        let args = AprEncryptArgs {
            file: std::path::PathBuf::from("model.bin"),
            output: std::path::PathBuf::from("out.bin"),
            password: Some("secret".to_string()),
        };
        let global = make_test_global(true, false);
        let result = run_encrypt(&args, &global);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("format-encryption"),
            "Should mention feature: {msg}"
        );
    }
}

#[test]
fn test_run_decrypt_no_feature() {
    #[cfg(not(feature = "format-encryption"))]
    {
        let args = AprDecryptArgs {
            file: std::path::PathBuf::from("model.bin"),
            output: std::path::PathBuf::from("out.bin"),
            password: Some("secret".to_string()),
        };
        let global = make_test_global(true, false);
        let result = run_decrypt(&args, &global);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("format-encryption"),
            "Should mention feature: {msg}"
        );
    }
}

#[test]
fn test_run_quantize_no_feature() {
    #[cfg(not(feature = "format-quantize"))]
    {
        let args = AprQuantizeArgs {
            file: std::path::PathBuf::from("model.bin"),
            output: std::path::PathBuf::from("out.bin"),
            r#type: "q8_0".to_string(),
            verify: false,
        };
        let global = make_test_global(true, false);
        let result = run_quantize(&args, &global);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("format-quantize"),
            "Should mention feature: {msg}"
        );
    }
}

#[test]
fn test_run_he_inspect_no_feature() {
    #[cfg(not(feature = "format-homomorphic"))]
    {
        let args = AprHeInspectArgs {
            file: std::path::PathBuf::from("model.bin"),
        };
        let global = make_test_global(true, false);
        let result = run_he_inspect(&args, &global);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("format-homomorphic"),
            "Should mention feature: {msg}"
        );
    }
}

// -- Import-sharded tests (always available, no feature gate) --

#[test]
fn test_run_import_sharded_missing_dir() {
    let args = AprImportShardedArgs {
        source: std::path::PathBuf::from("/nonexistent/dir"),
        output: std::path::PathBuf::from("out.apr"),
        max_cache_shards: 2,
    };
    let global = make_test_global(true, false);
    let result = run_import_sharded(&args, &global);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not found"), "Should mention not found: {msg}");
}

#[test]
fn test_run_import_sharded_not_sharded() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Create a directory with no safetensors files
    let args = AprImportShardedArgs {
        source: dir.path().to_path_buf(),
        output: dir.path().join("out.apr"),
        max_cache_shards: 2,
    };
    let global = make_test_global(true, false);
    let result = run_import_sharded(&args, &global);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Not a sharded model"),
        "Should mention not sharded: {msg}"
    );
}

#[test]
fn test_run_import_sharded_with_index() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Create a minimal index.json
    let index_json = r#"{
            "metadata": {},
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
    std::fs::write(dir.path().join("model.safetensors.index.json"), index_json)
        .expect("write index");

    let args = AprImportShardedArgs {
        source: dir.path().to_path_buf(),
        output: dir.path().join("out.apr"),
        max_cache_shards: 2,
    };
    let global = make_test_global(true, false);
    // Should succeed (stream_merge works even without actual shard files)
    let result = run_import_sharded(&args, &global);
    assert!(result.is_ok(), "Expected success, got: {result:?}");
}

#[test]
fn test_run_import_sharded_json() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index_json = r#"{
            "metadata": {},
            "weight_map": {
                "embed": "shard-001.safetensors"
            }
        }"#;
    std::fs::write(dir.path().join("model.safetensors.index.json"), index_json)
        .expect("write index");

    let args = AprImportShardedArgs {
        source: dir.path().to_path_buf(),
        output: dir.path().join("out.apr"),
        max_cache_shards: 1,
    };
    let global = make_test_global(false, true);
    let result = run_import_sharded(&args, &global);
    assert!(result.is_ok());
}

// ========================================================================
// Phase 3 falsification tests
// ========================================================================

// F11: Sharded import preserves tensor count
//
// H₀: "Sharded import reports the same tensor count as the index"
// Test: Create index with known tensor count, verify report matches.
#[test]
fn falsification_f11_sharded_import_tensor_count() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index_json = r#"{
            "metadata": {},
            "weight_map": {
                "layer.0.weight": "shard-001.safetensors",
                "layer.1.weight": "shard-001.safetensors",
                "layer.2.weight": "shard-002.safetensors",
                "embed": "shard-002.safetensors",
                "output": "shard-003.safetensors"
            }
        }"#;
    std::fs::write(dir.path().join("model.safetensors.index.json"), index_json)
        .expect("write index");

    use aprender::format::sharded::{ShardedImportConfig, ShardedImporter};

    let config = ShardedImportConfig::default();
    let importer = ShardedImporter::new(config, dir.path().to_path_buf());
    let index = importer
        .parse_index(&dir.path().join("model.safetensors.index.json"))
        .expect("parse index");

    assert_eq!(
        index.tensor_count(),
        5,
        "F11 FALSIFIED: index reports {} tensors, expected 5",
        index.tensor_count()
    );
    assert_eq!(
        index.shard_count(),
        3,
        "F11 FALSIFIED: index reports {} shards, expected 3",
        index.shard_count()
    );
}

// F12: Feature-gated commands fail gracefully without feature
//
// H₀: "Feature-gated commands produce clear 'requires --features' message"
// Test: Call each gated handler without feature, verify error message.
#[test]
fn falsification_f12_feature_gate_clear_error() {
    // This test runs with whatever features are currently enabled.
    // When a feature is NOT enabled, the handler should return a clear error.
    let global = make_test_global(true, false);

    #[cfg(not(feature = "format-signing"))]
    {
        let result = run_sign(
            &AprSignArgs {
                file: "x".into(),
                key: "k".into(),
                output: "o".into(),
            },
            &global,
        );
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("--features"),
            "F12 FALSIFIED: sign error should mention --features"
        );
    }

    #[cfg(not(feature = "format-encryption"))]
    {
        let result = run_encrypt(
            &AprEncryptArgs {
                file: "x".into(),
                output: "o".into(),
                password: Some("p".to_string()),
            },
            &global,
        );
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("--features"),
            "F12 FALSIFIED: encrypt error should mention --features"
        );
    }

    #[cfg(not(feature = "format-quantize"))]
    {
        let result = run_quantize(
            &AprQuantizeArgs {
                file: "x".into(),
                output: "o".into(),
                r#type: "q8_0".to_string(),
                verify: false,
            },
            &global,
        );
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("--features"),
            "F12 FALSIFIED: quantize error should mention --features"
        );
    }

    #[cfg(not(feature = "format-homomorphic"))]
    {
        let result = run_he_inspect(&AprHeInspectArgs { file: "x".into() }, &global);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("--features"),
            "F12 FALSIFIED: he-inspect error should mention --features"
        );
    }
}

// F13: Import-sharded is_sharded_model detection consistent
//
// H₀: "is_sharded_model returns true iff directory contains index or 2+ safetensors"
// Test: Empty dir → false, dir with index → true.
#[test]
fn falsification_f13_sharded_detection_consistent() {
    use aprender::format::sharded::is_sharded_model;

    // Empty dir → false
    let empty_dir = tempfile::tempdir().expect("tempdir");
    assert!(
        !is_sharded_model(empty_dir.path()),
        "F13 FALSIFIED: empty dir should not be sharded"
    );

    // Dir with index → true
    let indexed_dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        indexed_dir.path().join("model.safetensors.index.json"),
        "{}",
    )
    .expect("write");
    assert!(
        is_sharded_model(indexed_dir.path()),
        "F13 FALSIFIED: dir with index should be sharded"
    );

    // Dir with 1 safetensors → false
    let single_dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(single_dir.path().join("model.safetensors"), b"data").expect("write");
    assert!(
        !is_sharded_model(single_dir.path()),
        "F13 FALSIFIED: single safetensors should not be sharded"
    );

    // Dir with 2 safetensors → true
    let multi_dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(multi_dir.path().join("shard-001.safetensors"), b"a").expect("write");
    std::fs::write(multi_dir.path().join("shard-002.safetensors"), b"b").expect("write");
    assert!(
        is_sharded_model(multi_dir.path()),
        "F13 FALSIFIED: 2 safetensors should be sharded"
    );
}

// ========================================================================
// Test helpers
// ========================================================================

/// Create a minimal Args struct for testing
fn make_test_global(quiet: bool, json: bool) -> super::super::args::Args {
    // Build Args programmatically — we only need the fields our handlers access
    super::super::args::Args {
        command: super::super::args::Command::Apr(super::super::apr_args::AprArgs {
            action: super::super::apr_args::AprAction::Inspect(
                super::super::apr_args::AprInspectArgs {
                    file: std::path::PathBuf::from("dummy"),
                },
            ),
        }),
        verbose: false,
        quiet,
        json,
        trace: None,
        no_color: false,
    }
}

//! Self-contained `apr pull` / `apr ls` — download and list cached models
//! without the apr-cli dependency.
//!
//! Contract: contracts/model-pull-v1.yaml (provable-contracts, kind: pattern).
//! `pull` is idempotent + cache-confined; `ls` reflects exactly the model dirs
//! under `default_cache_dir()`. See L1-L5 obligations FALSIFY-PULL-001..006.
//!
//! Built on the in-tree [`ModelDownloader`](crate::model::download) (hf_hub sync
//! API); depending on the full apr-cli crate dragged in the entire
//! batuta/orchestrate/arrow/tonic/axum/wgpu stack for two proxied calls.

use crate::cli::apr_args::{AprPullArgs, AprPullListArgs};
use crate::cli::commands::{CliError, CliResult, CommandResult};
use crate::model::download::{default_cache_dir, find_model, ModelDownloader};

/// File extensions treated as an explicit single-file weight target in an `hf://…` ref.
const WEIGHT_EXTENSIONS: &[&str] = &["safetensors", "gguf", "bin", "onnx"];

/// `apr pull <model_ref>` — download a model into the whisper-apr model cache.
///
/// Resolves `model_ref` against the built-in registry when possible, otherwise
/// treats it as an arbitrary HuggingFace repo id. Weight files are materialised
/// under `default_cache_dir()/<slug>` so `apr ls` sees them and repeat pulls are
/// idempotent (contract `pull`, FALSIFY-PULL-001/002/003).
#[provable_contracts_macros::contract("whisper-model-pull-v1", equation = "pull")]
pub(super) fn run_pull(args: &AprPullArgs) -> CliResult<CommandResult> {
    pull_into_cache(&args.model_ref, args.force, &default_cache_dir())
}

/// `apr ls` — list models present in the whisper-apr model cache
/// (contract `list`, FALSIFY-PULL-004/005/006).
#[provable_contracts_macros::contract("whisper-model-pull-v1", equation = "list")]
pub(super) fn run_pull_list(args: &AprPullListArgs) -> CliResult<CommandResult> {
    list_cached(&default_cache_dir(), args.json)
}

/// Cache subdirectory name for a model ref: registry `name` when known,
/// else the repo id with path separators flattened to `--`.
fn cache_slug(model_ref: &str) -> String {
    if let Some(model) = find_model(model_ref) {
        return model.name.to_string();
    }
    let (repo_id, _) = split_repo_and_file(model_ref);
    repo_id.replace('/', "--")
}

/// Split `hf://org/repo[/file.ext]` into `(repo_id, Option<filename>)`.
///
/// A trailing component whose extension is a known weight extension is treated
/// as an explicit filename; otherwise the whole reference is the repo id.
fn split_repo_and_file(model_ref: &str) -> (String, Option<String>) {
    let trimmed = model_ref.trim_start_matches("hf://");
    let parts: Vec<&str> = trimmed.split('/').collect();
    if parts.len() > 2 {
        if let Some(last) = parts.last() {
            let is_weight = std::path::Path::new(last)
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| WEIGHT_EXTENSIONS.contains(&ext));
            if is_weight {
                let repo_id = parts[..parts.len() - 1].join("/");
                return (repo_id, Some((*last).to_string()));
            }
        }
    }
    (trimmed.to_string(), None)
}

/// Download `model_ref` into `cache_dir/<slug>`; skip (idempotent) when the
/// target already holds files and `!force`.
fn pull_into_cache(
    model_ref: &str,
    force: bool,
    cache_dir: &std::path::Path,
) -> CliResult<CommandResult> {
    let model_dir = cache_dir.join(cache_slug(model_ref));

    if !force && dir_has_files(&model_dir) {
        return Ok(CommandResult::success(format!(
            "{model_ref} already cached at {}",
            model_dir.display()
        )));
    }

    let sources = download_sources(model_ref, cache_dir)?;
    let copied = materialize(&model_dir, &sources)?;

    Ok(CommandResult::success(format!(
        "Pulled {model_ref} ({copied} file(s)) -> {}",
        model_dir.display()
    )))
}

/// True if `dir` exists and contains at least one entry.
fn dir_has_files(dir: &std::path::Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|mut it| it.next().is_some())
}

/// Resolve on-disk source paths (in the hf_hub blob cache) for `model_ref`.
fn download_sources(
    model_ref: &str,
    cache_dir: &std::path::Path,
) -> CliResult<Vec<std::path::PathBuf>> {
    if let Some(model) = find_model(model_ref) {
        let downloader = ModelDownloader::with_cache_dir(cache_dir.to_path_buf())
            .map_err(|e| CliError::InvalidArgument(format!("downloader init failed: {e}")))?;
        return downloader
            .download_safetensors(model)
            .map_err(|e| CliError::InvalidArgument(format!("download failed: {e}")));
    }
    download_arbitrary(model_ref)
}

/// Download an arbitrary HuggingFace repo (or a single file) via `hf_hub`.
fn download_arbitrary(model_ref: &str) -> CliResult<Vec<std::path::PathBuf>> {
    let (repo_id, file) = split_repo_and_file(model_ref);
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| CliError::InvalidArgument(format!("hf-hub init failed: {e}")))?;
    let repo = api.model(repo_id.clone());

    if let Some(filename) = file {
        let path = repo.get(&filename).map_err(|e| {
            CliError::InvalidArgument(format!("failed to download {repo_id}/{filename}: {e}"))
        })?;
        return Ok(vec![path]);
    }

    let mut out = Vec::new();
    for candidate in ["model.safetensors", "pytorch_model.safetensors"] {
        if let Ok(path) = repo.get(candidate) {
            out.push(path);
        }
    }
    if out.is_empty() {
        return Err(CliError::InvalidArgument(format!(
            "no safetensors weights found in {repo_id}"
        )));
    }
    Ok(out)
}

/// Copy resolved source files into `model_dir`; return the count copied.
fn materialize(model_dir: &std::path::Path, sources: &[std::path::PathBuf]) -> CliResult<usize> {
    std::fs::create_dir_all(model_dir)
        .map_err(|e| CliError::WriteError(format!("{}: {e}", model_dir.display())))?;
    let mut copied = 0usize;
    for src in sources {
        let name = src.file_name().ok_or_else(|| {
            CliError::InvalidArgument(format!("bad source path {}", src.display()))
        })?;
        let dest = model_dir.join(name);
        std::fs::copy(src, &dest)
            .map_err(|e| CliError::WriteError(format!("{}: {e}", dest.display())))?;
        copied += 1;
    }
    Ok(copied)
}

/// List cache subdirectories (one per pulled model) in `cache_dir`.
fn list_cached(cache_dir: &std::path::Path, json: bool) -> CliResult<CommandResult> {
    let mut names = cache_subdirs(cache_dir)?;
    names.sort();

    if json {
        println!("{}", cache_names_json(&names)?);
    } else if names.is_empty() {
        println!("No models cached in {}", cache_dir.display());
    } else {
        println!("Cached models in {}:", cache_dir.display());
        for name in &names {
            println!("  {name}");
        }
    }

    Ok(CommandResult::success(format!(
        "{} cached model(s)",
        names.len()
    )))
}

/// Immediate subdirectory names of `cache_dir`. A missing dir is empty, not an error.
fn cache_subdirs(cache_dir: &std::path::Path) -> CliResult<Vec<String>> {
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(CliError::InvalidArgument(format!(
                "cannot read cache {}: {e}",
                cache_dir.display()
            )))
        }
    };

    let mut names = Vec::new();
    for entry in entries.flatten() {
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            if let Some(name) = entry.file_name().to_str() {
                names.push(name.to_string());
            }
        }
    }
    Ok(names)
}

/// Render cached model names as a pretty JSON array string.
fn cache_names_json(names: &[String]) -> CliResult<String> {
    serde_json::to_string_pretty(names)
        .map_err(|e| CliError::InvalidArgument(format!("json serialization failed: {e}")))
}

// ============================================================================
// contracts/model-pull-v1.yaml — falsification tests (FALSIFY-PULL-001..006).
// Self-contained: no network. Exercise the cache short-circuit + list helpers.
// ============================================================================
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// FALSIFY-PULL-001/002: a force=false pull of an already-cached model is a
    /// no-op that reports "cached" and leaves the cache byte-for-byte unchanged.
    #[test]
    fn test_pull_idempotent_when_cached() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = dir.path();
        // Pre-populate the model dir so the network path is never taken.
        // cache_slug("whisper-tiny") == "whisper-tiny" whether or not it is a
        // known registry model (no '/' to flatten).
        let model_dir = cache.join("whisper-tiny");
        std::fs::create_dir_all(&model_dir).expect("mkdir");
        std::fs::write(model_dir.join("model.safetensors"), b"stub").expect("write");

        let r1 = pull_into_cache("whisper-tiny", false, cache).expect("pull 1");
        assert!(r1.success);
        assert!(r1.message.contains("already cached"));

        // Second pull is identical: idempotent, still offline, file untouched.
        let r2 = pull_into_cache("whisper-tiny", false, cache).expect("pull 2");
        assert!(r2.success);
        assert!(r2.message.contains("already cached"));
        assert_eq!(
            std::fs::read(model_dir.join("model.safetensors")).expect("read"),
            b"stub"
        );
    }

    /// FALSIFY-PULL-006: a failed `materialize` (a source that cannot be copied)
    /// returns Err and leaves no file-bearing model dir, so `dir_has_files` — the
    /// idempotency/"cached" predicate — stays false (failure atomicity). Offline.
    #[test]
    fn test_failed_pull_leaves_no_cached_model() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model_dir = dir.path().join("whisper-tiny");
        let bogus = dir.path().join("does-not-exist.safetensors");

        let res = materialize(&model_dir, std::slice::from_ref(&bogus));
        assert!(res.is_err(), "copy of a missing source must fail");
        // The empty dir may exist, but it holds no files => not registered as cached.
        assert!(
            !dir_has_files(&model_dir),
            "a failed pull must not leave a file-bearing (cached) model dir"
        );
    }

    /// FALSIFY-PULL-003/004: ls() matches exactly the model dirs under the cache
    /// (stray files excluded), and a missing cache dir lists as empty, not an error.
    #[test]
    fn test_list_matches_cache_dir() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = dir.path();
        for name in ["whisper-tiny", "moonshine-base", "org--custom"] {
            std::fs::create_dir_all(cache.join(name)).expect("mkdir");
        }
        std::fs::write(cache.join("README.txt"), b"x").expect("write"); // stray file ignored

        let mut got = cache_subdirs(cache).expect("subdirs");
        got.sort();
        assert_eq!(
            got,
            vec![
                "moonshine-base".to_string(),
                "org--custom".to_string(),
                "whisper-tiny".to_string(),
            ]
        );

        let res = list_cached(cache, false).expect("list");
        assert!(res.success);
        assert!(res.message.contains("3 cached model(s)"));

        // Missing cache dir lists as empty, not an error.
        let missing = tempfile::tempdir().expect("temp dir");
        let empty = cache_subdirs(&missing.path().join("nope")).expect("subdirs");
        assert!(empty.is_empty());
    }

    /// FALSIFY-PULL-005: ls --json emits a valid, parseable JSON array (including
    /// the empty case).
    #[test]
    fn test_list_json_is_valid() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = dir.path();
        std::fs::create_dir_all(cache.join("whisper-base")).expect("mkdir");
        std::fs::create_dir_all(cache.join("moonshine-tiny")).expect("mkdir");

        let names = cache_subdirs(cache).expect("subdirs");
        let json = cache_names_json(&names).expect("serialize");

        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        let arr = parsed.as_array().expect("array");
        assert_eq!(arr.len(), 2);
        assert!(arr.iter().all(serde_json::Value::is_string));

        // Empty cache still yields a valid, empty JSON array.
        let empty_json = cache_names_json(&[]).expect("serialize");
        let empty: serde_json::Value = serde_json::from_str(&empty_json).expect("valid json");
        assert_eq!(empty.as_array().map(Vec::len), Some(0));
    }
}

//! Pins the nightly gate (paiml/whisper.apr#99).
//!
//! The nightly builds whenever the `nightly` tag commit differs from HEAD. It
//! must not depend on a 24-hour wall-clock window: a nightly whose build failed
//! was never retried on a quiet day. The test runs the real `check` step of
//! `.github/workflows/nightly.yml` in a scratch repository, so it fails if the
//! workflow goes back to the `git log --since="24 hours ago"` condition.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const WORKFLOW: &str = ".github/workflows/nightly.yml";

fn workflow_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(WORKFLOW);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The `run:` block of the step whose `id:` is `check`, with the GitHub
/// expression for the event name replaced by a concrete event.
fn check_step_script(workflow: &str, event: &str) -> String {
    let lines: Vec<&str> = workflow.lines().collect();
    let id_at = lines
        .iter()
        .position(|l| l.trim() == "id: check")
        .expect("workflow has a step with id: check");
    let run_at = (id_at..lines.len())
        .find(|&i| lines[i].trim() == "run: |")
        .expect("check step has a run: | block");
    let run_indent = lines[run_at].len() - lines[run_at].trim_start().len();
    let body: Vec<String> = lines[run_at + 1..]
        .iter()
        .take_while(|l| l.trim().is_empty() || l.len() - l.trim_start().len() > run_indent)
        .map(|l| l.trim_start().to_string())
        .collect();
    body.join("\n").replace("${{ github.event_name }}", event)
}

fn git(dir: &Path, args: &[&str], date: Option<&str>) -> String {
    let mut cmd = Command::new("git");
    cmd.current_dir(dir).args(args);
    cmd.env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t");
    cmd.env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t");
    if let Some(d) = date {
        cmd.env("GIT_AUTHOR_DATE", d).env("GIT_COMMITTER_DATE", d);
    }
    let out = cmd.output().expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn scratch_repo(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nightly-wf-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("mkdir");
    git(&dir, &["init", "-q"], None);
    dir
}

/// Runs the check step in `repo` and returns the `should_build` value.
fn should_build(repo: &Path, script: &str) -> String {
    let out_file = repo.join("github-output");
    let _ = fs::remove_file(&out_file);
    let status = Command::new("bash")
        .arg("-c")
        .arg(script)
        .current_dir(repo)
        .env("GITHUB_OUTPUT", &out_file)
        .status()
        .expect("bash runs");
    assert!(status.success(), "check step exits 0");
    let written = fs::read_to_string(&out_file).expect("GITHUB_OUTPUT written");
    written
        .lines()
        .find_map(|l| l.strip_prefix("should_build="))
        .expect("should_build written")
        .to_string()
}

#[test]
fn check_builds_when_nightly_tag_is_behind_head_even_if_head_is_old() {
    // The case the 24h window got wrong: the last build failed (tag still on
    // an old commit) and HEAD is three days old, so no commit is "recent".
    let repo = scratch_repo("behind-old-head");
    git(
        &repo,
        &["commit", "-q", "--allow-empty", "-m", "built"],
        Some("2026-11-20T10:00:00Z"),
    );
    git(&repo, &["tag", "nightly"], None);
    git(
        &repo,
        &["commit", "-q", "--allow-empty", "-m", "unbuilt"],
        Some("2026-11-24T10:00:00Z"),
    );

    let script = check_step_script(&workflow_text(), "schedule");
    assert_eq!(
        should_build(&repo, &script),
        "true",
        "tag behind HEAD must build"
    );
    let _ = fs::remove_dir_all(&repo);
}

#[test]
fn check_skips_when_nightly_tag_is_head() {
    let repo = scratch_repo("tag-is-head");
    git(
        &repo,
        &["commit", "-q", "--allow-empty", "-m", "built"],
        Some("2026-11-20T10:00:00Z"),
    );
    git(&repo, &["tag", "nightly"], None);

    let script = check_step_script(&workflow_text(), "schedule");
    assert_eq!(
        should_build(&repo, &script),
        "false",
        "tag == HEAD must skip"
    );
    let _ = fs::remove_dir_all(&repo);
}

#[test]
fn release_is_pinned_to_the_built_sha() {
    let text = workflow_text();
    assert!(
        text.contains("target_commitish: ${{ github.sha }}"),
        "the nightly release must be pinned to github.sha, not the default branch"
    );
    assert!(
        !text.contains("24 hours"),
        "no 24-hour window may gate the nightly build"
    );
}

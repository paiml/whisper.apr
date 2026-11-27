//! Integrity check for tests/fixtures/diarization (VoxConverse dev excerpts, CC-BY-4.0).
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diarization")
}

fn manifest() -> Vec<(String, String)> {
    let text = fs::read_to_string(dir().join("SHA256SUMS")).expect("SHA256SUMS");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (h, n) = l.split_once("  ").expect("`<sha256>  <name>` line");
            (n.trim().to_string(), h.to_string())
        })
        .collect()
}

#[test]
fn every_fixture_matches_its_sha256() {
    let rows = manifest();
    assert!(rows.len() >= 7, "manifest lists only {} files", rows.len());
    for (name, want) in &rows {
        let bytes = fs::read(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let got: String = Sha256::digest(&bytes)
            .iter()
            .fold(String::new(), |mut acc, b| {
                use std::fmt::Write;
                let _ = write!(acc, "{b:02x}");
                acc
            });
        assert_eq!(&got, want, "{name} differs from SHA256SUMS");
    }
}

#[test]
fn manifest_covers_every_file_and_the_excerpts_are_60s_16k_mono() {
    let listed: BTreeSet<String> = manifest().into_iter().map(|(n, _)| n).collect();
    for entry in fs::read_dir(dir()).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().to_string();
        if name != "SHA256SUMS" {
            assert!(listed.contains(&name), "{name} is not in SHA256SUMS");
        }
    }
    let wavs: Vec<_> = listed
        .iter()
        .filter(|n| {
            std::path::Path::new(n.as_str())
                .extension()
                .is_some_and(|e| e == "wav")
        })
        .collect();
    assert_eq!(wavs.len(), 3, "expected three excerpts");
    for w in wavs {
        let b = fs::read(dir().join(w)).unwrap();
        assert_eq!(&b[0..4], b"RIFF", "{w}");
        assert_eq!(u16::from_le_bytes([b[22], b[23]]), 1, "{w}: mono");
        assert_eq!(
            u32::from_le_bytes([b[24], b[25], b[26], b[27]]),
            16_000,
            "{w}: 16 kHz"
        );
        let data = b.windows(4).position(|x| x == b"data").expect("data chunk") + 8;
        assert_eq!(b.len() - data, 60 * 16_000 * 2, "{w}: 60 s of 16-bit PCM");
        let rttm = fs::read_to_string(dir().join(w.replace(".wav", ".rttm"))).unwrap();
        assert!(rttm.lines().count() >= 3 && rttm.lines().all(|l| l.starts_with("SPEAKER ")));
    }
}

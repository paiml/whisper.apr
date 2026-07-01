//! Tier C — Profiling (renacer integration + BrickProfiler).
//!
//! Split into four submodules by concern:
//! - [`types`]   — shared `BrickDetail`/`ProfileRun`/`AvgBrickDetail`/`ProfileSummary` structs
//! - [`sweep`]   — thread-scaling sweep (Amdahl metrics, WAPR-PROFILE-001 Gap 3)
//! - [`run`]     — `run_profile` entry point + roofline/brick-detail extraction
//! - [`summary`] — `ProfileSummary` formatters (JSON, renacer, print_table)

mod run;
mod summary;
mod sweep;
mod types;

pub(in super::super) use run::run_profile;

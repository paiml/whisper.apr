//! Phase 3: Feature-gated CLI handlers (security, quantization, profiling)
//!
//! This module is split into four submodules by concern:
//! - [`signing`] — Ed25519 sign / verify-sig
//! - [`encryption`] — AES-256-GCM encrypt / decrypt
//! - [`quantize`] — Q4_0 / Q8_0 quantization, sharded import, HE inspection
//! - [`profile`] — profiling / sweeps / amdahl / roofline

mod encryption;
mod profile;
mod quantize;
mod signing;

pub(super) use encryption::{run_decrypt, run_encrypt};
pub(super) use profile::run_profile;
pub(super) use quantize::{run_he_inspect, run_import_sharded, run_quantize};
pub(super) use signing::{run_sign, run_verify_sig};

/// Dispatch CLI output to JSON or human-readable text based on global flags.
///
/// Shared helper used by all phase3 submodules. `json_fn` runs when
/// `global.json` is set; otherwise `text_fn` runs unless `global.quiet`
/// suppresses all output.
pub(super) fn emit_output(
    global: &super::super::args::Args,
    json_fn: impl FnOnce(),
    text_fn: impl FnOnce(),
) {
    if global.json {
        json_fn();
    } else if !global.quiet {
        text_fn();
    }
}

//! Up-front memory check for `transcribe --gpu` (whisper.apr#145).
//!
//! On a shared-memory device (Jetson Orin Nano class, 8 GB) the CUDA upload of
//! large-v3-turbo ran the device out of memory after the host copy of the weights
//! had already taken its share. The check here refuses before the upload starts and
//! names the memory the upload needs, instead of letting the OOM killer end the run.
//!
//! The estimate is deliberately simple and stated as one: FP32 weights, counted
//! twice. The device copy is one W; the upload also builds a transposed host copy
//! of the same size while the original is still resident, so the upload needs about
//! 2 W of free memory on top of what the process already holds.

#![cfg_attr(not(feature = "realizar-gpu"), allow(dead_code))]

/// Bytes per FP32 weight.
const F32_BYTES: u64 = 4;

/// Free memory the GPU upload needs, in bytes: the FP32 weights, twice.
pub fn gpu_upload_bytes_needed(parameter_count: u64) -> u64 {
    parameter_count * F32_BYTES * 2
}

/// Parse `MemAvailable` (kB in `/proc/meminfo`) into bytes.
pub fn parse_mem_available(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Free memory now, in bytes. `None` where `/proc/meminfo` is not readable,
/// in which case the check does not refuse.
pub fn available_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_mem_available(&meminfo)
}

/// The refusal message when `need` bytes will not fit in `available`, or `None`
/// when the upload may proceed. Unknown availability never refuses.
pub fn gpu_fit_refusal(need: u64, available: Option<u64>) -> Option<String> {
    let available = available?;
    if need <= available {
        return None;
    }
    Some(format!(
        "GPU upload needs about {:.1} GB of free memory, but only {:.1} GB is available \
         on this device. Run without --gpu, or use a device with more memory.",
        need as f64 / 1e9,
        available as f64 / 1e9,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn parses_mem_available_in_bytes() {
        let meminfo = "MemTotal:        7800000 kB\nMemFree:          100000 kB\nMemAvailable:    2000000 kB\n";
        assert_eq!(parse_mem_available(meminfo), Some(2_000_000 * 1024));
    }

    #[test]
    fn missing_mem_available_is_none() {
        assert_eq!(parse_mem_available("MemTotal: 1 kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: x kB\n"), None);
    }

    #[test]
    fn fits_when_need_is_within_available() {
        assert!(gpu_fit_refusal(3 * GB, Some(3 * GB)).is_none());
        assert!(gpu_fit_refusal(GB, Some(6 * GB)).is_none());
    }

    #[test]
    fn refuses_and_names_the_need_when_it_does_not_fit() {
        let msg = gpu_fit_refusal(6_500_000_000, Some(3_900_000_000)).expect("must refuse");
        assert!(msg.contains("6.5 GB"), "names the need: {msg}");
        assert!(msg.contains("3.9 GB"), "names the available: {msg}");
        assert!(msg.contains("without --gpu"), "says what to do: {msg}");
    }

    #[test]
    fn unknown_availability_does_not_refuse() {
        assert!(gpu_fit_refusal(10 * GB, None).is_none());
    }

    #[test]
    fn need_is_two_fp32_copies_of_the_parameters() {
        assert_eq!(gpu_upload_bytes_needed(1_000), 8_000);
    }
}

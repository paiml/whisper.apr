//! Check filterbank sum/scaling matches reference

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FILTERBANK SCALING CHECK ===\n");

    // Load reference filterbank from binary file
    let ref_bytes = std::fs::read("test_data/ref_b_filterbank.bin")?;
    let ref_fb: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    println!("Reference filterbank: {} values", ref_fb.len());
    let ref_sum: f32 = ref_fb.iter().sum();
    let ref_max: f32 = ref_fb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Sum: {:.6}", ref_sum);
    println!("  Max: {:.6}", ref_max);
    println!("  First 5 values: {:?}", &ref_fb[..5]);

    // Load our filterbank
    use whisper_apr::audio::mel_filterbank_data::MEL_80_FILTERBANK;
    let our_fb = &MEL_80_FILTERBANK[..];
    println!("\nOur filterbank: {} values", our_fb.len());
    let our_sum: f32 = our_fb.iter().sum();
    let our_max: f32 = our_fb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Sum: {:.6}", our_sum);
    println!("  Max: {:.6}", our_max);
    println!("  First 5 values: {:?}", &our_fb[..5]);

    // Per-row sums (each mel filter should sum to ~0.025 for Slaney normalization)
    let n_mels = 80;
    let n_freqs = our_fb.len() / n_mels;
    println!("\nPer-mel-band sums (first 5):");
    for mel in 0..5 {
        let row_start = mel * n_freqs;
        let row_sum: f32 = our_fb[row_start..row_start + n_freqs].iter().sum();
        let ref_row_sum: f32 = if mel * 201 < ref_fb.len() {
            ref_fb[mel * 201..(mel + 1) * 201].iter().sum()
        } else {
            0.0
        };
        println!("  Mel {}: ours={:.6}, ref={:.6}", mel, row_sum, ref_row_sum);
    }

    // Check if filterbanks match
    let min_len = our_fb.len().min(ref_fb.len());
    let mut max_diff: f32 = 0.0;
    for i in 0..min_len {
        let diff = (our_fb[i] - ref_fb[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("\nMax filterbank difference: {:.10}", max_diff);

    if max_diff < 1e-6 {
        println!("✓ Filterbanks match exactly!");
    } else {
        println!("✗ Filterbanks differ!");
    }

    Ok(())
}

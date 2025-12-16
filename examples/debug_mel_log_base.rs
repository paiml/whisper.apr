//! H22: Verify mel spectrogram log base
//!
//! OpenAI Whisper uses natural log (ln), not log10
//! This is a ~2.3x scale error that could cause Q·K misalignment

fn main() {
    println!("=== H22: MEL LOG BASE VERIFICATION ===\n");

    // OpenAI Whisper mel normalization (from whisper/audio.py):
    // log_spec = torch.clamp(mel.log10(), min=mel.log10().max() - 8.0)
    // log_spec = (log_spec + 4.0) / 4.0
    //
    // Wait - OpenAI actually DOES use log10! Let me check again...

    // From https://github.com/openai/whisper/blob/main/whisper/audio.py:
    // magnitudes = stft[..., :-1].abs() ** 2
    // mel_spec = filters @ magnitudes
    // log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    // log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    // log_spec = (log_spec + 4.0) / 4.0

    println!("Checking OpenAI Whisper source code reference...\n");

    println!("From whisper/audio.py:");
    println!("  log_spec = torch.clamp(mel_spec, min=1e-10).log10()");
    println!("  log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)");
    println!("  log_spec = (log_spec + 4.0) / 4.0");

    println!("\nOur implementation (mel.rs line 296):");
    println!("  let log_mel = (mel_energy.max(1e-10)).log10();");
    println!("  *x = (*x).max(max_val - 8.0);");
    println!("  *x = (*x + 4.0) / 4.0;");

    println!("\n✅ Log base matches! Both use log10.");

    // So the log base is NOT the issue. Let me check other potential differences...

    println!("\n=== CHECKING OTHER POTENTIAL DIFFERENCES ===\n");

    // 1. FFT normalization
    println!("1. FFT Normalization:");
    println!("   OpenAI: torch.stft with return_complex=True (no normalization)");
    println!("   Ours: rustfft with default normalization (need to verify)");

    // 2. Power spectrum
    println!("\n2. Power Spectrum:");
    println!("   OpenAI: magnitudes = stft[..., :-1].abs() ** 2");
    println!("   Ours: c.norm_sqr() (complex magnitude squared)");
    println!("   ✅ These should be equivalent");

    // 3. Filterbank
    println!("\n3. Filterbank:");
    println!("   OpenAI: Slaney-normalized mel filterbank from librosa");
    println!("   Ours: Uses pre-computed filterbank from .apr file OR computes own");
    println!("   ⚠️  Need to verify filterbank matches exactly!");

    // 4. Frame layout
    println!("\n4. Output layout:");
    println!("   OpenAI: [n_mels, n_frames] (channel-first)");
    println!("   Ours: [n_frames * n_mels] flattened as [frame][mel]");
    println!("   ⚠️  This could cause transposition issues!");

    // Let me compute an example to see the actual values
    println!("\n=== NUMERICAL EXAMPLE ===\n");

    // Test value: a typical mel energy
    let mel_energy = 0.001f32;

    let log10_val = (mel_energy.max(1e-10)).log10();
    let ln_val = (mel_energy.max(1e-10)).ln();

    println!("For mel_energy = {}:", mel_energy);
    println!("  log10 = {:.6}", log10_val);
    println!("  ln    = {:.6}", ln_val);
    println!("  ratio = {:.6}", ln_val / log10_val);

    // After normalization
    let max_val = -2.0f32; // typical max log10 value
    let clamped_log10 = log10_val.max(max_val - 8.0);
    let normalized_log10 = (clamped_log10 + 4.0) / 4.0;

    println!("\nAfter Whisper normalization (with max=-2.0):");
    println!("  clamped = {:.6}", clamped_log10);
    println!("  normalized = {:.6}", normalized_log10);

    // Check expected range
    println!("\n=== EXPECTED VS OBSERVED RANGES ===");
    println!("\nTypical Whisper mel values (normalized):");
    println!("  Range: approximately -1.5 to 1.0");
    println!("  Mean: approximately -0.3 to -0.5 for speech");

    println!("\nOur observed mel values (from H17):");
    println!("  Audio region (0-75): mean=0.066, std=0.365");
    println!("  Padding region: mean=-0.331");

    println!("\n⚠️  Our values seem to be in a DIFFERENT range than expected!");
    println!("   Expected padding ~ -0.25 (after normalization)");
    println!("   Got padding ~ -0.33");
}

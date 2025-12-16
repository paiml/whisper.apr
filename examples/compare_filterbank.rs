//! Compare our filterbank with whisper.cpp's filterbank
//!
//! This performs a t-test to determine if the filterbanks are statistically different.

use whisper_apr::audio::MelFilterbank;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   FILTERBANK COMPARISON (FALSIFICATION TEST)               ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Load whisper.cpp filterbank
    let whisper_cpp_path = "/tmp/whisper_cpp_filterbank.bin";
    let whisper_cpp_bytes = match std::fs::read(whisper_cpp_path) {
        Ok(b) => b,
        Err(_) => {
            println!("Error: Run 'python3 tools/extract_filterbank.py ../whisper.cpp/models/ggml-tiny.bin' first");
            return Ok(());
        }
    };

    let n_mels = 80usize;
    let n_freqs = 201usize;
    let expected_size = n_mels * n_freqs * 4;

    if whisper_cpp_bytes.len() != expected_size {
        println!(
            "Error: Expected {} bytes, got {}",
            expected_size,
            whisper_cpp_bytes.len()
        );
        return Ok(());
    }

    let whisper_cpp_filterbank: Vec<f32> = whisper_cpp_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Compute our filterbank
    let mel_filterbank = MelFilterbank::new(n_mels, 400, 16000);
    let our_filterbank = mel_filterbank.filters();

    println!("Filterbank Dimensions:");
    println!("  whisper.cpp: {} x {} = {}", n_mels, n_freqs, whisper_cpp_filterbank.len());
    println!("  ours:        {} x {} = {}", n_mels, mel_filterbank.n_freqs(), our_filterbank.len());

    // Basic statistics
    let wcpp_sum: f64 = whisper_cpp_filterbank.iter().map(|&x| x as f64).sum();
    let wcpp_nonzero = whisper_cpp_filterbank.iter().filter(|&&x| x > 1e-10).count();

    let our_sum: f64 = our_filterbank.iter().map(|&x| x as f64).sum();
    let our_nonzero = our_filterbank.iter().filter(|&&x| x > 1e-10).count();

    println!("\nBasic Statistics:");
    println!("  whisper.cpp sum: {:.6}, non-zero: {}", wcpp_sum, wcpp_nonzero);
    println!("  our sum:         {:.6}, non-zero: {}", our_sum, our_nonzero);

    // Compare row by row (each mel filter)
    println!("\nPer-Filter Comparison (first 10 mel bands):");
    println!("  {:>5} {:>12} {:>12} {:>12} {:>12}", "Mel", "wcpp_sum", "our_sum", "diff", "pct_diff");

    let mut total_diff = 0.0f64;
    let mut max_diff = 0.0f64;

    for mel_idx in 0..n_mels {
        let wcpp_row_start = mel_idx * n_freqs;
        let our_row_start = mel_idx * mel_filterbank.n_freqs();

        let wcpp_row_sum: f64 = whisper_cpp_filterbank[wcpp_row_start..wcpp_row_start + n_freqs]
            .iter()
            .map(|&x| x as f64)
            .sum();
        let our_row_sum: f64 = our_filterbank[our_row_start..our_row_start + mel_filterbank.n_freqs()]
            .iter()
            .map(|&x| x as f64)
            .sum();

        let diff = (wcpp_row_sum - our_row_sum).abs();
        let pct_diff = if wcpp_row_sum > 1e-10 {
            diff / wcpp_row_sum * 100.0
        } else {
            0.0
        };

        total_diff += diff;
        max_diff = max_diff.max(diff);

        if mel_idx < 10 || diff > 0.01 {
            println!(
                "  {:>5} {:>12.6} {:>12.6} {:>12.6} {:>11.2}%",
                mel_idx, wcpp_row_sum, our_row_sum, diff, pct_diff
            );
        }
    }

    println!("\nAggregate Differences:");
    println!("  Total absolute diff: {:.6}", total_diff);
    println!("  Max row diff:        {:.6}", max_diff);
    println!("  Mean row diff:       {:.6}", total_diff / n_mels as f64);

    // Compare specific filter shapes
    println!("\nFilter Shape Comparison (Mel band 0, first 20 freq bins):");
    println!("  {:>4} {:>12} {:>12} {:>12}", "Bin", "wcpp", "ours", "diff");

    for freq_idx in 0..20.min(n_freqs) {
        let wcpp_val = whisper_cpp_filterbank[freq_idx];
        let our_val = our_filterbank[freq_idx];
        let diff = (wcpp_val - our_val).abs();

        if wcpp_val > 1e-10 || our_val > 1e-10 {
            println!("  {:>4} {:>12.8} {:>12.8} {:>12.8}", freq_idx, wcpp_val, our_val, diff);
        }
    }

    // T-test
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STATISTICAL COMPARISON (Welch's t-test)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Calculate means
    let wcpp_mean = wcpp_sum / whisper_cpp_filterbank.len() as f64;
    let our_mean = our_sum / our_filterbank.len() as f64;

    // Calculate variances
    let wcpp_var: f64 = whisper_cpp_filterbank
        .iter()
        .map(|&x| ((x as f64) - wcpp_mean).powi(2))
        .sum::<f64>()
        / (whisper_cpp_filterbank.len() - 1) as f64;

    let our_var: f64 = our_filterbank
        .iter()
        .map(|&x| ((x as f64) - our_mean).powi(2))
        .sum::<f64>()
        / (our_filterbank.len() - 1) as f64;

    let n1 = whisper_cpp_filterbank.len() as f64;
    let n2 = our_filterbank.len() as f64;

    // Welch's t-statistic
    let t_stat = (wcpp_mean - our_mean) / (wcpp_var / n1 + our_var / n2).sqrt();

    println!("H0: The filterbanks have the same mean");
    println!("H1: The filterbanks have different means\n");
    println!("  whisper.cpp mean: {:.10}", wcpp_mean);
    println!("  our mean:         {:.10}", our_mean);
    println!("  whisper.cpp var:  {:.10}", wcpp_var);
    println!("  our var:          {:.10}", our_var);
    println!("  t-statistic:      {:.6}", t_stat);

    // Approximate p-value (two-tailed)
    // For large samples, t ~ N(0,1), so we can use normal approximation
    let p_value_approx = 2.0 * (1.0 - normal_cdf(t_stat.abs()));
    println!("  p-value (approx): {:.6}", p_value_approx);

    let alpha = 0.05;
    let reject_null = p_value_approx < alpha;
    println!(
        "\n✓ Conclusion (α={}): {} - {}",
        alpha,
        if reject_null { "REJECT H0" } else { "FAIL TO REJECT H0" },
        if reject_null {
            "Filterbanks are SIGNIFICANTLY DIFFERENT"
        } else {
            "Filterbanks are statistically similar"
        }
    );

    // Element-wise comparison (cosine similarity for non-zero elements)
    let mut dot_product = 0.0f64;
    let mut wcpp_norm_sq = 0.0f64;
    let mut our_norm_sq = 0.0f64;

    for (i, (&wcpp_val, &our_val)) in whisper_cpp_filterbank.iter().zip(our_filterbank.iter()).enumerate() {
        if i < our_filterbank.len() {
            dot_product += (wcpp_val as f64) * (our_val as f64);
            wcpp_norm_sq += (wcpp_val as f64).powi(2);
            our_norm_sq += (our_val as f64).powi(2);
        }
    }

    let cosine_sim = dot_product / (wcpp_norm_sq.sqrt() * our_norm_sq.sqrt());
    println!("\nCosine Similarity: {:.6} (1.0 = identical)", cosine_sim);

    // Conclusion
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║   FILTERBANK FALSIFICATION RESULT                         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    if cosine_sim < 0.99 || reject_null {
        println!("⚠️  FILTERBANKS ARE DIFFERENT!");
        println!("   This is likely the root cause of the 'rererer' hallucination.");
        println!("   The model was trained with a specific filterbank that we're not matching.");
        println!("\n   RECOMMENDED FIX:");
        println!("   Load filterbank from model file instead of computing it.");
    } else {
        println!("✓ Filterbanks appear similar. Look elsewhere for the bug.");
    }

    Ok(())
}

/// Standard normal CDF approximation (Hastings)
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if x > 0.0 {
        1.0 - p
    } else {
        p
    }
}

use std::fs;
use std::time::Instant;

fn main() {
    let data = fs::read("models/whisper-tiny-int8.apr").expect("read");
    let mut model = whisper_apr::WhisperApr::load_from_apr(&data).expect("load");

    let audio: Vec<f32> = (0..16000)
        .map(|i| ((i as f32) * 0.01).sin() * 0.3)
        .collect();
    let mel = model.compute_mel(&audio).expect("mel");
    let enc = model.encoder_mut().forward_mel(&mel).expect("enc");

    println!("Token count | Batch forward | Per-token");
    println!("------------|---------------|----------");

    for n in [1, 2, 4, 8, 16] {
        let tokens: Vec<u32> = (0..n).map(|i| 50258 + i as u32).collect();

        // Warmup
        for _ in 0..3 {
            let _ = model.decoder_mut().forward(&tokens, &enc);
        }

        let start = Instant::now();
        for _ in 0..5 {
            let _ = model.decoder_mut().forward(&tokens, &enc).expect("dec");
        }
        let total = start.elapsed().as_secs_f64() * 1000.0 / 5.0;

        println!("{:11} | {:>11.1}ms | {:>7.1}ms", n, total, total / n as f64);
    }
}

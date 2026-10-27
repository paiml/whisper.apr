use super::*;
use crate::inference::GreedyDecoder;

const EOT: u32 = 9;
const N_VOCAB: usize = 10;

/// A planted model: at every step token `next(context)` has logit 4 and
/// every other token 0, EOT included.
fn planted_logits(context: &[u32], next: impl Fn(&[u32]) -> u32) -> Vec<f32> {
    let mut logits = vec![0.0f32; N_VOCAB];
    logits[next(context) as usize] = 4.0;
    logits
}

/// Words for the planted vocabulary, so a decode has a text to compress.
fn words(tokens: &[u32]) -> String {
    const WORDS: [&str; 9] = [
        "thank", "you", "so", "much", "for", "joining", "us", "today", "music",
    ];
    tokens
        .iter()
        .filter(|&&t| t != EOT)
        .map(|&t| WORDS[t as usize % WORDS.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

/// One planted decode at `temperature`: greedy at 0, seeded sampling above,
/// with stats from the same ledger and compression ratio the model uses.
fn planted_attempt(
    temperature: f32,
    seed: u64,
    next: impl Fn(&[u32]) -> u32,
    no_speech_prob: f32,
) -> (Vec<u32>, DecodeStats) {
    let ledger = std::cell::RefCell::new(LogprobLedger::new(5));
    let logits_fn = |context: &[u32]| {
        let logits = planted_logits(context, &next);
        ledger.borrow_mut().observe(context, &logits);
        Ok(logits)
    };
    let decoder = GreedyDecoder::new(40)
        .with_temperature(temperature)
        .with_seed(seed);
    let tokens = decoder.decode(logits_fn, &[], EOT).unwrap_or_default();
    let stats = DecodeStats {
        compression_ratio: compression_ratio(&words(&tokens)),
        avg_logprob: ledger.borrow().avg_logprob(&tokens, 0, EOT),
        no_speech_prob,
    };
    (tokens, stats)
}

/// The loop: "thank you thank you ..." until the context is full.
fn looping(context: &[u32]) -> u32 {
    (context.len() % 2) as u32
}

/// A normal utterance: eight distinct words, then EOT.
fn normal(context: &[u32]) -> u32 {
    if context.len() < 8 {
        context.len() as u32
    } else {
        EOT
    }
}

#[test]
fn a_planted_repetition_loop_triggers_the_fallback() {
    let fallback = DecodeFallback::default();
    let (_, greedy) = planted_attempt(0.0, 0, looping, 0.0);
    assert!(
        greedy.compression_ratio > fallback.compression_ratio_threshold,
        "a loop compresses well: {greedy:?}"
    );
    assert!(fallback.needs_fallback(&greedy));

    let mut seen = Vec::new();
    let outcome = fallback
        .run(3, |temperature, seed| {
            seen.push(temperature);
            Ok(planted_attempt(temperature, seed, looping, 0.0))
        })
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(outcome.attempts > 1, "the loop was retried: {outcome:?}");
    assert_eq!(seen[0], 0.0);
    assert!(seen[1..].iter().all(|&t| t > 0.0));
}

#[test]
fn the_fallback_stops_at_the_first_temperature_that_passes() {
    let fallback = DecodeFallback::default();
    let outcome = fallback
        .run(0, |temperature, seed| {
            let next: fn(&[u32]) -> u32 = if temperature == 0.0 { looping } else { normal };
            Ok(planted_attempt(temperature, seed, next, 0.0))
        })
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(outcome.attempts, 2);
    assert!((outcome.temperature - 0.2).abs() < 1e-6);
    assert!(!fallback.needs_fallback(&outcome.stats));
}

#[test]
fn a_normal_window_is_not_escalated() {
    let fallback = DecodeFallback::default();
    let mut calls = 0;
    let outcome = fallback
        .run(0, |temperature, seed| {
            calls += 1;
            Ok(planted_attempt(temperature, seed, normal, 0.01))
        })
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(calls, 1);
    assert_eq!(outcome.attempts, 1);
    assert_eq!(outcome.temperature, 0.0);
    assert!(!outcome.silent);
    assert_eq!(outcome.output, vec![0, 1, 2, 3, 4, 5, 6, 7, EOT]);
    assert!(outcome.stats.compression_ratio <= fallback.compression_ratio_threshold);
    assert!(outcome.stats.avg_logprob >= fallback.logprob_threshold);
}

#[test]
fn a_planted_silent_window_emits_nothing() {
    let fallback = DecodeFallback::default();
    // SOT logits of music: <|nospeech|> (id 7 here) dominates.
    let mut sot_logits = vec![0.0f32; N_VOCAB];
    sot_logits[7] = 5.0;
    let nsp = no_speech_prob(&sot_logits, 7);
    assert!(nsp > fallback.no_speech_threshold, "{nsp}");

    // The invented closing line, with unlikely tokens (flat logits).
    let mut calls = 0;
    let outcome = fallback
        .run(0, |_, _| {
            calls += 1;
            let tokens = vec![0, 1, 2, 3, EOT];
            let stats = DecodeStats {
                compression_ratio: compression_ratio(&words(&tokens)),
                avg_logprob: -1.8,
                no_speech_prob: nsp,
            };
            Ok((tokens, stats))
        })
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(calls, 1, "silence is never retried");
    assert!(outcome.silent);
    assert_eq!(outcome.clone().into_kept_output(), Vec::<u32>::new());

    // Likely tokens over the same SOT keep the window: speech wins.
    let speech = DecodeStats {
        avg_logprob: -0.2,
        ..outcome.stats
    };
    assert!(!fallback.is_silent(&speech));
    // A low no-speech probability keeps an unlikely window too.
    let unsure = DecodeStats {
        no_speech_prob: 0.3,
        ..outcome.stats
    };
    assert!(!fallback.is_silent(&unsure));
}

#[test]
fn the_same_input_decodes_to_identical_bytes_twice() {
    let fallback = DecodeFallback {
        temperatures: vec![1.0],
        ..DecodeFallback::default()
    };
    // A flat-ish model, so sampling really draws different tokens.
    let wobbly = |context: &[u32]| (context.len() as u32 * 7 + 3) % 9;
    let run = |window| {
        fallback
            .run(window, |t, seed| Ok(planted_attempt(t, seed, wobbly, 0.0)))
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let first = run(5);
    let second = run(5);
    let bytes = |o: &FallbackOutcome<Vec<u32>>| {
        o.output
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .chain(o.stats.avg_logprob.to_le_bytes())
            .chain(o.stats.compression_ratio.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    assert_eq!(bytes(&first), bytes(&second));
    // The seed is a function of the window: another window samples anew.
    assert_ne!(
        fallback.attempt_seed(5, 0),
        fallback.attempt_seed(6, 0),
        "windows share a seed"
    );
    assert_ne!(fallback.attempt_seed(5, 0), fallback.attempt_seed(5, 1));
}

#[test]
fn seeded_sampling_draws_by_probability_and_never_a_banned_token() {
    let mut rng = SplitMix64::new(42);
    let logits = [0.0f32, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
    let mut counts = [0usize; 4];
    for _ in 0..2000 {
        counts[sample_token(&logits, 1.0, &mut rng) as usize] += 1;
    }
    assert_eq!(counts[1] + counts[3], 0);
    assert!(counts[0] > 800 && counts[2] > 800, "{counts:?}");
}

#[test]
fn compression_ratio_matches_zlib_on_known_text() {
    assert_eq!(compression_ratio(""), 0.0);
    let looped = "Thank you. ".repeat(20);
    assert!(compression_ratio(&looped) > 2.4);
    let normal = "The quick brown fox jumps over the lazy dog.";
    assert!(compression_ratio(normal) < 2.4);
}

#[test]
fn avg_logprob_is_openai_sum_over_n_plus_one() {
    let mut ledger = LogprobLedger::new(2);
    // Two steps, each over two equally likely tokens: log(0.5) each.
    ledger.observe(&[100], &[0.0, 0.0]);
    ledger.observe(&[100, 0], &[0.0, 0.0]);
    let eot = 1;
    let avg = ledger.avg_logprob(&[100, 0, eot], 1, eot);
    // One text token (n = 1): (log .5 + log .5) / 2.
    assert!((avg - 0.5f32.ln()).abs() < 1e-6, "{avg}");
}

#[test]
fn the_ledger_scores_beam_contexts_from_their_top_k() {
    let mut ledger = LogprobLedger::new(2);
    ledger.observe(&[7], &[2.0, 1.0, -5.0]);
    ledger.observe(&[7, 1], &[0.0, 3.0, 0.0]);
    ledger.observe(&[7, 0], &[0.0, 0.0, 0.0]);
    // The beam kept [7, 1, 1]: its contexts were not the last one seen.
    let lp0 = log_softmax(&[2.0, 1.0, -5.0])[1];
    let lp1 = log_softmax(&[0.0, 3.0, 0.0])[1];
    let avg = ledger.avg_logprob(&[7, 1, 1], 1, 99);
    assert!((avg - (lp0 + lp1) / 3.0).abs() < 1e-5, "{avg}");
}

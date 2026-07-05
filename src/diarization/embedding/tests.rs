#![allow(clippy::expect_used)]
//! Tests for speaker embedding extraction

use super::*;

// =========================================================================
// EmbeddingConfig Tests
// =========================================================================

#[test]
fn test_embedding_config_default() {
    let config = EmbeddingConfig::default();
    assert_eq!(config.embedding_dim, 256);
    assert!(config.normalize);
    assert!(config.use_mean_pooling);
}

#[test]
fn test_embedding_config_for_realtime() {
    let config = EmbeddingConfig::for_realtime();
    assert!((config.window_size - 1.0).abs() < f32::EPSILON);
    assert!((config.hop_size - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_embedding_config_for_accuracy() {
    let config = EmbeddingConfig::for_accuracy();
    assert!((config.window_size - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_embedding_config_with_window_size() {
    let config = EmbeddingConfig::default().with_window_size(3.0);
    assert!((config.window_size - 3.0).abs() < f32::EPSILON);
}

// =========================================================================
// SpeakerEmbedding Tests
// =========================================================================

#[test]
fn test_speaker_embedding_new() {
    let vec = vec![0.1; 256];
    let emb = SpeakerEmbedding::new(vec.clone(), 0);

    assert_eq!(emb.speaker_id(), 0);
    assert_eq!(emb.dim(), 256);
    assert!((emb.confidence() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_speaker_embedding_with_confidence() {
    let emb = SpeakerEmbedding::new(vec![0.1; 256], 0).with_confidence(0.8);
    assert!((emb.confidence() - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_speaker_embedding_cosine_similarity_identical() {
    let emb1 = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0);
    let emb2 = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0);

    let sim = emb1.cosine_similarity(&emb2);
    assert!((sim - 1.0).abs() < 0.001);
}

#[test]
fn test_speaker_embedding_cosine_similarity_orthogonal() {
    let emb1 = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0);
    let emb2 = SpeakerEmbedding::new(vec![0.0, 1.0, 0.0], 0);

    let sim = emb1.cosine_similarity(&emb2);
    assert!(sim.abs() < 0.001);
}

#[test]
fn test_speaker_embedding_cosine_similarity_opposite() {
    let emb1 = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0);
    let emb2 = SpeakerEmbedding::new(vec![-1.0, 0.0, 0.0], 0);

    let sim = emb1.cosine_similarity(&emb2);
    assert!((sim + 1.0).abs() < 0.001);
}

#[test]
fn test_speaker_embedding_euclidean_distance() {
    let emb1 = SpeakerEmbedding::new(vec![0.0, 0.0, 0.0], 0);
    let emb2 = SpeakerEmbedding::new(vec![3.0, 4.0, 0.0], 0);

    let dist = emb1.euclidean_distance(&emb2);
    assert!((dist - 5.0).abs() < 0.001);
}

#[test]
fn test_speaker_embedding_normalized() {
    let emb = SpeakerEmbedding::new(vec![3.0, 4.0], 0);
    let normalized = emb.normalized();

    let norm: f32 = normalized
        .vector()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    assert!((norm - 1.0).abs() < 0.001);
}

#[test]
fn test_speaker_embedding_mean() {
    let embeddings = vec![
        SpeakerEmbedding::new(vec![1.0, 2.0, 3.0], 0),
        SpeakerEmbedding::new(vec![3.0, 4.0, 5.0], 0),
    ];

    let mean = SpeakerEmbedding::mean(&embeddings).expect("should compute mean");
    assert!((mean.vector()[0] - 2.0).abs() < 0.001);
    assert!((mean.vector()[1] - 3.0).abs() < 0.001);
    assert!((mean.vector()[2] - 4.0).abs() < 0.001);
}

#[test]
fn test_speaker_embedding_mean_empty() {
    let embeddings: Vec<SpeakerEmbedding> = vec![];
    let mean = SpeakerEmbedding::mean(&embeddings);
    assert!(mean.is_none());
}

// =========================================================================
// SpeakerEmbeddingModel Tests
// =========================================================================

#[test]
fn test_speaker_embedding_model_default() {
    let model = SpeakerEmbeddingModel::default();
    assert_eq!(model, SpeakerEmbeddingModel::MfccSimple);
}

// =========================================================================
// EmbeddingExtractor Tests
// =========================================================================

#[test]
fn test_embedding_extractor_new() {
    let config = EmbeddingConfig::default();
    let extractor = EmbeddingExtractor::new(config);
    assert_eq!(extractor.config().embedding_dim, 256);
}

#[test]
fn test_embedding_extractor_with_model() {
    let extractor = EmbeddingExtractor::new(EmbeddingConfig::default())
        .with_model(SpeakerEmbeddingModel::XVector);
    // Model is set (internal state)
    assert!(extractor.config().normalize);
}

#[test]
fn test_embedding_extractor_extract_empty() {
    let extractor = EmbeddingExtractor::new(EmbeddingConfig::default());
    let result = extractor.extract(&[], 16000);
    assert!(result.is_err());
}

#[test]
fn test_embedding_extractor_extract_short_audio() {
    let extractor = EmbeddingExtractor::new(EmbeddingConfig::default());
    let audio = vec![0.1; 100]; // Very short audio
    let result = extractor.extract(&audio, 16000);
    assert!(result.is_err());
}

#[test]
fn test_embedding_extractor_extract_valid_audio() {
    let extractor = EmbeddingExtractor::new(EmbeddingConfig::default());
    // Generate 1 second of audio at 16kHz
    let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

    let result = extractor.extract(&audio, 16000);
    assert!(result.is_ok());

    let embedding = result.expect("should succeed");
    assert_eq!(embedding.dim(), 256);
}

#[test]
fn test_embedding_extractor_extract_normalized() {
    let config = EmbeddingConfig::default();
    let extractor = EmbeddingExtractor::new(config);

    let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

    let embedding = extractor.extract(&audio, 16000).expect("should succeed");
    let norm: f32 = embedding.vector().iter().map(|x| x * x).sum::<f32>().sqrt();

    // Should be approximately unit norm
    assert!((norm - 1.0).abs() < 0.1 || norm < f32::EPSILON);
}

#[test]
fn test_embedding_extractor_resample() {
    let extractor = EmbeddingExtractor::new(EmbeddingConfig::default());
    let audio: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0];

    let resampled = extractor.resample(&audio, 8000, 16000);
    assert!(resampled.len() > audio.len());
}

#[test]
fn test_hz_to_mel_to_hz() {
    let hz: f32 = 1000.0;
    let mel = EmbeddingExtractor::hz_to_mel(hz);
    let hz_back = EmbeddingExtractor::mel_to_hz(mel);
    assert!((hz - hz_back).abs() < 0.1);
}

#[test]
fn test_dct_matrix_dimensions() {
    let matrix = EmbeddingExtractor::compute_dct_matrix(40, 80);
    assert_eq!(matrix.len(), 40);
    assert_eq!(matrix[0].len(), 80);
}

#[test]
fn test_mel_filterbank_dimensions() {
    let filterbank = EmbeddingExtractor::compute_mel_filterbank(80, 512, 16000);
    assert_eq!(filterbank.len(), 80);
    assert_eq!(filterbank[0].len(), 257); // 512/2 + 1
}

// =========================================================================
// Builder Coverage Tests (PMAT-024)
// =========================================================================

#[test]
fn test_embedding_config_with_hop_size() {
    let config = EmbeddingConfig::default().with_hop_size(0.25);
    assert!((config.hop_size - 0.25).abs() < f32::EPSILON);
}

#[test]
fn test_embedding_config_with_hop_size_small() {
    let config = EmbeddingConfig::default().with_hop_size(0.1);
    assert!((config.hop_size - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_embedding_config_with_hop_and_window() {
    let config = EmbeddingConfig::default()
        .with_window_size(3.0)
        .with_hop_size(1.5);
    assert!((config.window_size - 3.0).abs() < f32::EPSILON);
    assert!((config.hop_size - 1.5).abs() < f32::EPSILON);
}

// =========================================================================
// with_hop_size additional coverage (WAPR-QA-005)
// =========================================================================

#[test]
fn test_embedding_config_with_hop_size_large() {
    let config = EmbeddingConfig::default().with_hop_size(2.0);
    assert!((config.hop_size - 2.0).abs() < f32::EPSILON);
    // Other fields should be unchanged
    assert_eq!(config.embedding_dim, EMBEDDING_DIM);
    assert!(config.normalize);
}

#[test]
fn test_embedding_config_with_hop_size_preserves_other_fields() {
    let config = EmbeddingConfig::for_accuracy().with_hop_size(0.3);
    // hop_size updated
    assert!((config.hop_size - 0.3).abs() < f32::EPSILON);
    // window_size from for_accuracy() preserved
    assert!((config.window_size - 2.0).abs() < f32::EPSILON);
    assert!(config.use_mean_pooling);
}

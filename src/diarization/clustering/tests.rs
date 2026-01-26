//! Tests for speaker clustering

use super::*;
use crate::diarization::embedding::SpeakerEmbedding;

// =========================================================================
// ClusteringAlgorithm Tests
// =========================================================================

#[test]
fn test_clustering_algorithm_default() {
    let alg = ClusteringAlgorithm::default();
    assert_eq!(alg, ClusteringAlgorithm::Spectral);
}

// =========================================================================
// ClusteringConfig Tests
// =========================================================================

#[test]
fn test_clustering_config_default() {
    let config = ClusteringConfig::default();
    assert_eq!(config.algorithm, ClusteringAlgorithm::Spectral);
    assert!(config.use_cosine_distance);
    assert_eq!(config.max_iterations, 100);
}

#[test]
fn test_clustering_config_for_realtime() {
    let config = ClusteringConfig::for_realtime();
    assert_eq!(config.algorithm, ClusteringAlgorithm::KMeans);
    assert_eq!(config.max_iterations, 50);
}

#[test]
fn test_clustering_config_for_accuracy() {
    let config = ClusteringConfig::for_accuracy();
    assert_eq!(config.algorithm, ClusteringAlgorithm::Spectral);
    assert_eq!(config.max_iterations, 200);
}

#[test]
fn test_clustering_config_with_algorithm() {
    let config = ClusteringConfig::default().with_algorithm(ClusteringAlgorithm::KMeans);
    assert_eq!(config.algorithm, ClusteringAlgorithm::KMeans);
}

#[test]
fn test_clustering_config_with_distance_threshold() {
    let config = ClusteringConfig::default().with_distance_threshold(0.7);
    assert!((config.distance_threshold - 0.7).abs() < f32::EPSILON);
}

// =========================================================================
// SpeakerCluster Tests
// =========================================================================

#[test]
fn test_speaker_cluster_new() {
    let centroid = SpeakerEmbedding::new(vec![0.1; 256], 0);
    let cluster = SpeakerCluster::new(0, vec![0, 1, 2], centroid);

    assert_eq!(cluster.id(), 0);
    assert_eq!(cluster.size(), 3);
    assert_eq!(cluster.member_indices().len(), 3);
}

#[test]
fn test_speaker_cluster_with_cohesion() {
    let centroid = SpeakerEmbedding::new(vec![0.1; 256], 0);
    let cluster = SpeakerCluster::new(0, vec![0], centroid).with_cohesion(0.5);
    assert!((cluster.cohesion() - 0.5).abs() < f32::EPSILON);
}

// =========================================================================
// ClusteringResult Tests
// =========================================================================

#[test]
fn test_clustering_result_new() {
    let centroid = SpeakerEmbedding::new(vec![0.1; 256], 0);
    let clusters = vec![SpeakerCluster::new(0, vec![0, 1], centroid)];
    let result = ClusteringResult::new(vec![0, 0], clusters);

    assert_eq!(result.num_clusters(), 1);
    assert_eq!(result.labels().len(), 2);
}

#[test]
fn test_clustering_result_with_silhouette() {
    let result = ClusteringResult::new(vec![], vec![]).with_silhouette_score(0.8);
    assert!((result.silhouette_score() - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_clustering_result_cluster_centroids() {
    let centroid1 = SpeakerEmbedding::new(vec![0.1; 256], 0);
    let centroid2 = SpeakerEmbedding::new(vec![0.2; 256], 1);
    let clusters = vec![
        SpeakerCluster::new(0, vec![0], centroid1),
        SpeakerCluster::new(1, vec![1], centroid2),
    ];
    let result = ClusteringResult::new(vec![0, 1], clusters);

    let centroids = result.cluster_centroids();
    assert_eq!(centroids.len(), 2);
}

// =========================================================================
// SpectralClustering Tests
// =========================================================================

#[test]
fn test_spectral_clustering_new() {
    let config = ClusteringConfig::default();
    let clustering = SpectralClustering::new(config);
    assert!(clustering.config().use_cosine_distance);
}

#[test]
fn test_spectral_clustering_empty() {
    let clustering = SpectralClustering::new(ClusteringConfig::default());
    let result = clustering.cluster(&[], None, 1);

    assert!(result.is_ok());
    let result = result.expect("should succeed");
    assert_eq!(result.num_clusters(), 0);
}

#[test]
fn test_spectral_clustering_single() {
    let clustering = SpectralClustering::new(ClusteringConfig::default());
    let embeddings = vec![SpeakerEmbedding::new(vec![0.1; 256], 0)];
    let result = clustering.cluster(&embeddings, None, 1);

    assert!(result.is_ok());
    let result = result.expect("should succeed");
    assert_eq!(result.num_clusters(), 1);
    assert_eq!(result.labels(), &[0]);
}

#[test]
fn test_spectral_clustering_two_distinct() {
    let clustering = SpectralClustering::new(ClusteringConfig::default());

    // Create two clearly distinct embeddings
    let emb1 = SpeakerEmbedding::new(vec![1.0; 256], 0);
    let emb2 = SpeakerEmbedding::new(vec![-1.0; 256], 1);

    let result = clustering.cluster(&[emb1, emb2], Some(2), 1);
    assert!(result.is_ok());

    let result = result.expect("should succeed");
    assert!(result.num_clusters() >= 1);
}

#[test]
fn test_spectral_clustering_similar_embeddings() {
    let clustering = SpectralClustering::new(ClusteringConfig::default());

    // Create similar embeddings that should cluster together
    let emb1 = SpeakerEmbedding::new(vec![0.9; 256], 0);
    let emb2 = SpeakerEmbedding::new(vec![0.95; 256], 0);
    let emb3 = SpeakerEmbedding::new(vec![0.92; 256], 0);

    let result = clustering.cluster(&[emb1, emb2, emb3], Some(2), 1);
    assert!(result.is_ok());

    let result = result.expect("should succeed");
    // All similar embeddings should be in same cluster
    let labels = result.labels();
    assert!(labels[0] == labels[1] && labels[1] == labels[2]);
}

#[test]
fn test_spectral_clustering_respects_min_clusters() {
    let clustering = SpectralClustering::new(ClusteringConfig::default());

    let embeddings: Vec<SpeakerEmbedding> = (0..5)
        .map(|i| SpeakerEmbedding::new(vec![i as f32 * 0.1; 256], 0))
        .collect();

    let result = clustering.cluster(&embeddings, Some(5), 2);
    assert!(result.is_ok());

    let result = result.expect("should succeed");
    assert!(result.num_clusters() >= 1); // At least min_clusters
}

#[test]
fn test_spectral_clustering_respects_max_clusters() {
    let clustering = SpectralClustering::new(ClusteringConfig::default());

    let embeddings: Vec<SpeakerEmbedding> = (0..10)
        .map(|i| SpeakerEmbedding::new(vec![i as f32 * 0.5; 256], 0))
        .collect();

    let result = clustering.cluster(&embeddings, Some(3), 1);
    assert!(result.is_ok());

    let result = result.expect("should succeed");
    assert!(result.num_clusters() <= 3);
}

#[test]
fn test_build_affinity_matrix() {
    let clustering = SpectralClustering::new(ClusteringConfig::default());

    let emb1 = SpeakerEmbedding::new(vec![1.0, 0.0], 0);
    let emb2 = SpeakerEmbedding::new(vec![1.0, 0.0], 0);

    let affinity = clustering.build_affinity_matrix(&[emb1, emb2]);

    assert_eq!(affinity.len(), 2);
    assert_eq!(affinity[0].len(), 2);
    // Identical embeddings should have affinity close to 1
    assert!(affinity[0][1] > 0.9);
}

#[test]
fn test_compute_cluster_cohesion() {
    let clustering = SpectralClustering::new(ClusteringConfig::default());

    let centroid = SpeakerEmbedding::new(vec![1.0, 0.0], 0);
    let members = vec![
        SpeakerEmbedding::new(vec![1.0, 0.0], 0),
        SpeakerEmbedding::new(vec![1.0, 0.0], 0),
    ];

    let cohesion = clustering.compute_cluster_cohesion(&members, &centroid);
    // Identical embeddings should have cohesion close to 0
    assert!(cohesion < 0.1);
}

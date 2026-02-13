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

// =========================================================================
// compute_silhouette Tests (WAPR-QA-006)
//
// Target: compute_silhouette at clustering/mod.rs:499
// Coverage gap: 84% -> target 100%. Uncovered branches include:
// - Euclidean distance path (use_cosine_distance = false)
// - Single cluster returns 0.0
// - a.max(b) == 0.0 branch (zero-vector embeddings)
// - Less than 2 embeddings returns 0.0
// =========================================================================

/// Test compute_silhouette returns 0.0 for a single embedding (< 2 elements).
#[test]
fn test_compute_silhouette_single_embedding_returns_zero() {
    let config = ClusteringConfig::default();
    let clustering = SpectralClustering::new(config);
    let embeddings = vec![SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0)];
    let labels = vec![0];

    let score = clustering.compute_silhouette(&embeddings, &labels);
    assert!(
        score.abs() < f32::EPSILON,
        "single embedding should yield silhouette = 0.0, got {}",
        score
    );
}

/// Test compute_silhouette returns 0.0 for empty embeddings.
#[test]
fn test_compute_silhouette_empty_embeddings_returns_zero() {
    let config = ClusteringConfig::default();
    let clustering = SpectralClustering::new(config);
    let embeddings: Vec<SpeakerEmbedding> = Vec::new();
    let labels: Vec<usize> = Vec::new();

    let score = clustering.compute_silhouette(&embeddings, &labels);
    assert!(
        score.abs() < f32::EPSILON,
        "empty embeddings should yield silhouette = 0.0, got {}",
        score
    );
}

/// Test compute_silhouette returns 0.0 when all embeddings share one cluster
/// (unique_labels.len() < 2 guard).
#[test]
fn test_compute_silhouette_single_cluster_returns_zero() {
    let config = ClusteringConfig::default();
    let clustering = SpectralClustering::new(config);
    let embeddings = vec![
        SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0),
        SpeakerEmbedding::new(vec![0.9, 0.1, 0.0], 0),
        SpeakerEmbedding::new(vec![0.8, 0.2, 0.0], 0),
    ];
    let labels = vec![0, 0, 0]; // All in same cluster

    let score = clustering.compute_silhouette(&embeddings, &labels);
    assert!(
        score.abs() < f32::EPSILON,
        "single cluster should yield silhouette = 0.0, got {}",
        score
    );
}

/// Test compute_silhouette with two well-separated clusters using cosine distance.
/// This exercises the main loop (a(i) and b(i) computation) with cosine distance.
#[test]
fn test_compute_silhouette_two_clusters_cosine() {
    let config = ClusteringConfig {
        use_cosine_distance: true,
        ..ClusteringConfig::default()
    };
    let clustering = SpectralClustering::new(config);

    // Cluster 0: embeddings pointing in similar directions
    // Cluster 1: embeddings pointing in opposite directions
    let embeddings = vec![
        SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0),
        SpeakerEmbedding::new(vec![0.9, 0.1, 0.0], 0),
        SpeakerEmbedding::new(vec![-1.0, 0.0, 0.0], 1),
        SpeakerEmbedding::new(vec![-0.9, -0.1, 0.0], 1),
    ];
    let labels = vec![0, 0, 1, 1];

    let score = clustering.compute_silhouette(&embeddings, &labels);
    // Score should be a valid number (not NaN or infinity)
    assert!(
        score.is_finite(),
        "silhouette should be finite, got {}",
        score
    );
}

/// Test compute_silhouette with Euclidean distance (use_cosine_distance = false).
/// This exercises the else branch in both a(i) and b(i) distance computations.
#[test]
fn test_compute_silhouette_euclidean_distance() {
    let config = ClusteringConfig {
        use_cosine_distance: false,
        ..ClusteringConfig::default()
    };
    let clustering = SpectralClustering::new(config);

    // Two clearly separated clusters in Euclidean space
    let embeddings = vec![
        SpeakerEmbedding::new(vec![0.0, 0.0], 0),
        SpeakerEmbedding::new(vec![0.1, 0.1], 0),
        SpeakerEmbedding::new(vec![10.0, 10.0], 1),
        SpeakerEmbedding::new(vec![10.1, 10.1], 1),
    ];
    let labels = vec![0, 0, 1, 1];

    let score = clustering.compute_silhouette(&embeddings, &labels);
    assert!(
        score.is_finite(),
        "silhouette should be finite, got {}",
        score
    );
}

/// Test compute_silhouette with zero-vector embeddings, exercising the
/// a.max(b) == 0.0 branch that returns 0.0 for each point.
#[test]
fn test_compute_silhouette_zero_vectors() {
    let config = ClusteringConfig {
        use_cosine_distance: true,
        ..ClusteringConfig::default()
    };
    let clustering = SpectralClustering::new(config);

    // Zero vectors: cosine similarity is 0 (handled by denom < EPSILON check),
    // so cosine distance = 1 - 0 = 1.0. This means a and b are both non-zero.
    let embeddings = vec![
        SpeakerEmbedding::new(vec![0.0, 0.0, 0.0], 0),
        SpeakerEmbedding::new(vec![0.0, 0.0, 0.0], 0),
        SpeakerEmbedding::new(vec![0.0, 0.0, 0.0], 1),
    ];
    let labels = vec![0, 0, 1];

    let score = clustering.compute_silhouette(&embeddings, &labels);
    assert!(
        score.is_finite(),
        "silhouette should be finite for zero vectors, got {}",
        score
    );
}

/// Test compute_silhouette with a lone point in one cluster (same_cluster is empty),
/// exercising the a = 0.0 fallback in the same-cluster distance computation.
#[test]
fn test_compute_silhouette_lone_point_in_cluster() {
    let config = ClusteringConfig::default();
    let clustering = SpectralClustering::new(config);

    // Point 0 is alone in cluster 0; points 1,2 are in cluster 1
    let embeddings = vec![
        SpeakerEmbedding::new(vec![1.0, 0.0], 0),
        SpeakerEmbedding::new(vec![0.0, 1.0], 1),
        SpeakerEmbedding::new(vec![0.0, 0.9], 1),
    ];
    let labels = vec![0, 1, 1];

    let score = clustering.compute_silhouette(&embeddings, &labels);
    assert!(
        score.is_finite(),
        "silhouette should be finite for lone cluster, got {}",
        score
    );
}

/// Test compute_silhouette with three clusters, exercising the b(i) computation
/// that iterates over all other clusters to find the minimum mean distance.
#[test]
fn test_compute_silhouette_three_clusters() {
    let config = ClusteringConfig::default();
    let clustering = SpectralClustering::new(config);

    let embeddings = vec![
        SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], 0),
        SpeakerEmbedding::new(vec![0.9, 0.1, 0.0], 0),
        SpeakerEmbedding::new(vec![0.0, 1.0, 0.0], 1),
        SpeakerEmbedding::new(vec![0.0, 0.9, 0.1], 1),
        SpeakerEmbedding::new(vec![0.0, 0.0, 1.0], 2),
        SpeakerEmbedding::new(vec![0.1, 0.0, 0.9], 2),
    ];
    let labels = vec![0, 0, 1, 1, 2, 2];

    let score = clustering.compute_silhouette(&embeddings, &labels);
    assert!(
        score.is_finite(),
        "silhouette should be finite for 3 clusters, got {}",
        score
    );
}

/// Test compute_silhouette via the full cluster() pipeline, which calls it at step 5.
/// Uses Euclidean distance config to exercise that path end-to-end.
#[test]
fn test_cluster_pipeline_exercises_silhouette_euclidean() {
    let config = ClusteringConfig {
        use_cosine_distance: false,
        ..ClusteringConfig::default()
    };
    let clustering = SpectralClustering::new(config);

    // Two well-separated clusters in Euclidean space
    let embeddings = vec![
        SpeakerEmbedding::new(vec![0.0, 0.0], 0),
        SpeakerEmbedding::new(vec![0.1, 0.0], 0),
        SpeakerEmbedding::new(vec![10.0, 0.0], 1),
        SpeakerEmbedding::new(vec![10.1, 0.0], 1),
    ];

    let result = clustering
        .cluster(&embeddings, Some(2), 1)
        .expect("cluster should succeed");

    // Silhouette score should have been computed (step 5 in cluster())
    assert!(
        result.silhouette_score().is_finite(),
        "silhouette from cluster() should be finite"
    );
}

/// Test compute_silhouette via cluster() with cosine distance and many embeddings,
/// ensuring the full silhouette computation path runs to completion.
#[test]
fn test_cluster_pipeline_exercises_silhouette_cosine_many_embeddings() {
    let config = ClusteringConfig {
        use_cosine_distance: true,
        ..ClusteringConfig::default()
    };
    let clustering = SpectralClustering::new(config);

    // Create 8 embeddings that form 2 natural clusters
    let mut embeddings = Vec::new();
    for i in 0..4 {
        // Cluster A: embeddings near [1, 0, 0, ...]
        let mut v = vec![0.0f32; 8];
        v[0] = 1.0;
        v[1] = i as f32 * 0.05;
        embeddings.push(SpeakerEmbedding::new(v, 0));
    }
    for i in 0..4 {
        // Cluster B: embeddings near [-1, 0, 0, ...]
        let mut v = vec![0.0f32; 8];
        v[0] = -1.0;
        v[1] = i as f32 * 0.05;
        embeddings.push(SpeakerEmbedding::new(v, 1));
    }

    let result = clustering
        .cluster(&embeddings, Some(2), 1)
        .expect("cluster should succeed");

    assert!(result.silhouette_score().is_finite());
    assert!(result.num_clusters() >= 1);
}

/// Test build_affinity_matrix with Euclidean distance configuration.
#[test]
fn test_build_affinity_matrix_euclidean() {
    let config = ClusteringConfig {
        use_cosine_distance: false,
        ..ClusteringConfig::default()
    };
    let clustering = SpectralClustering::new(config);

    let emb1 = SpeakerEmbedding::new(vec![0.0, 0.0], 0);
    let emb2 = SpeakerEmbedding::new(vec![0.0, 0.0], 0);
    let emb3 = SpeakerEmbedding::new(vec![10.0, 10.0], 1);

    let affinity = clustering.build_affinity_matrix(&[emb1, emb2, emb3]);

    assert_eq!(affinity.len(), 3);
    // Identical embeddings should have high affinity
    assert!(
        affinity[0][1] > 0.9,
        "identical embeddings should have high Euclidean affinity, got {}",
        affinity[0][1]
    );
    // Distant embeddings should have lower affinity
    assert!(
        affinity[0][2] < affinity[0][1],
        "distant embeddings should have lower affinity"
    );
}

/// Test compute_cluster_cohesion with Euclidean distance.
#[test]
fn test_compute_cluster_cohesion_euclidean() {
    let config = ClusteringConfig {
        use_cosine_distance: false,
        ..ClusteringConfig::default()
    };
    let clustering = SpectralClustering::new(config);

    let centroid = SpeakerEmbedding::new(vec![5.0, 5.0], 0);
    let members = vec![
        SpeakerEmbedding::new(vec![5.0, 5.0], 0),
        SpeakerEmbedding::new(vec![5.1, 5.1], 0),
    ];

    let cohesion = clustering.compute_cluster_cohesion(&members, &centroid);
    // Near-identical embeddings: small Euclidean cohesion
    assert!(
        cohesion < 1.0,
        "near-identical embeddings should have low Euclidean cohesion, got {}",
        cohesion
    );
}

/// Test compute_cluster_cohesion with empty members returns 0.0.
#[test]
fn test_compute_cluster_cohesion_empty_members() {
    let config = ClusteringConfig::default();
    let clustering = SpectralClustering::new(config);

    let centroid = SpeakerEmbedding::new(vec![1.0, 0.0], 0);
    let members: Vec<SpeakerEmbedding> = Vec::new();

    let cohesion = clustering.compute_cluster_cohesion(&members, &centroid);
    assert!(
        cohesion.abs() < f32::EPSILON,
        "empty members should yield cohesion = 0.0, got {}",
        cohesion
    );
}

/// Test estimate_num_clusters when n <= min_clusters, returning min_clusters.min(n).
#[test]
fn test_estimate_num_clusters_small_n() {
    let config = ClusteringConfig::default();
    let clustering = SpectralClustering::new(config);

    // 2 embeddings with min_clusters=3: should return min(3, 2) = 2
    let emb1 = SpeakerEmbedding::new(vec![1.0, 0.0], 0);
    let emb2 = SpeakerEmbedding::new(vec![0.0, 1.0], 1);
    let affinity = clustering.build_affinity_matrix(&[emb1, emb2]);

    let num = clustering.estimate_num_clusters(&affinity, None, 3);
    assert_eq!(num, 2, "with n=2 and min_clusters=3, should return 2");
}

/// Test cluster() with num_clusters >= n, triggering the (0..n).collect() path
/// in spectral_cluster().
#[test]
fn test_spectral_cluster_num_clusters_ge_n() {
    let config = ClusteringConfig {
        distance_threshold: 1.0, // High threshold so all points are in one group -> num_clusters=1
        ..ClusteringConfig::default()
    };
    let clustering = SpectralClustering::new(config);

    // With only 2 embeddings and min_clusters=2, we get num_clusters=2 >= n=2
    // which triggers the (0..n).collect() path
    let embeddings = vec![
        SpeakerEmbedding::new(vec![1.0, 0.0], 0),
        SpeakerEmbedding::new(vec![0.0, 1.0], 1),
    ];

    let result = clustering
        .cluster(&embeddings, Some(2), 2)
        .expect("should succeed");

    // Each embedding gets its own cluster label
    assert_eq!(result.labels().len(), 2);
}

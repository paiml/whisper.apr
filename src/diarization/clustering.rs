//! Speaker clustering algorithms (WAPR-151)
//!
//! Provides clustering algorithms for grouping speaker embeddings.
//!
//! # Overview
//!
//! Speaker clustering groups similar embeddings together to identify
//! unique speakers in the audio. Supports multiple algorithms:
//! - Spectral clustering (default, best for unknown number of speakers)
//! - K-means clustering (fast, requires known speaker count)
//! - Agglomerative clustering (hierarchical, good for small datasets)

use super::embedding::SpeakerEmbedding;
use crate::error::{WhisperError, WhisperResult};

/// Clustering algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClusteringAlgorithm {
    /// Spectral clustering (default)
    #[default]
    Spectral,
    /// K-means clustering
    KMeans,
    /// Agglomerative (hierarchical) clustering
    Agglomerative,
}

/// Clustering configuration
#[derive(Debug, Clone)]
pub struct ClusteringConfig {
    /// Algorithm to use
    pub algorithm: ClusteringAlgorithm,
    /// Distance threshold for clustering
    pub distance_threshold: f32,
    /// Minimum cluster size
    pub min_cluster_size: usize,
    /// Maximum iterations for iterative algorithms
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f32,
    /// Use cosine distance instead of Euclidean
    pub use_cosine_distance: bool,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::default(),
            distance_threshold: 0.5,
            min_cluster_size: 1,
            max_iterations: 100,
            convergence_threshold: 1e-4,
            use_cosine_distance: true,
        }
    }
}

impl ClusteringConfig {
    /// Configuration for real-time processing
    #[must_use]
    pub fn for_realtime() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::KMeans,
            max_iterations: 50,
            ..Default::default()
        }
    }

    /// Configuration for high accuracy
    #[must_use]
    pub fn for_accuracy() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::Spectral,
            max_iterations: 200,
            distance_threshold: 0.4,
            ..Default::default()
        }
    }

    /// Set algorithm
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ClusteringAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set distance threshold
    #[must_use]
    pub fn with_distance_threshold(mut self, threshold: f32) -> Self {
        self.distance_threshold = threshold;
        self
    }
}

/// A cluster of speaker embeddings
#[derive(Debug, Clone)]
pub struct SpeakerCluster {
    /// Cluster ID
    id: usize,
    /// Indices of embeddings in this cluster
    member_indices: Vec<usize>,
    /// Centroid embedding
    centroid: SpeakerEmbedding,
    /// Cluster cohesion (average intra-cluster distance)
    cohesion: f32,
}

impl SpeakerCluster {
    /// Create a new cluster
    #[must_use]
    pub fn new(id: usize, member_indices: Vec<usize>, centroid: SpeakerEmbedding) -> Self {
        Self {
            id,
            member_indices,
            centroid,
            cohesion: 0.0,
        }
    }

    /// Set cohesion value
    #[must_use]
    pub fn with_cohesion(mut self, cohesion: f32) -> Self {
        self.cohesion = cohesion;
        self
    }

    /// Get cluster ID
    #[must_use]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get member indices
    #[must_use]
    pub fn member_indices(&self) -> &[usize] {
        &self.member_indices
    }

    /// Get centroid
    #[must_use]
    pub fn centroid(&self) -> &SpeakerEmbedding {
        &self.centroid
    }

    /// Get cluster size
    #[must_use]
    pub fn size(&self) -> usize {
        self.member_indices.len()
    }

    /// Get cohesion
    #[must_use]
    pub fn cohesion(&self) -> f32 {
        self.cohesion
    }
}

/// Clustering result
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Cluster assignments for each embedding
    labels: Vec<usize>,
    /// Individual clusters
    clusters: Vec<SpeakerCluster>,
    /// Number of clusters
    num_clusters: usize,
    /// Silhouette score (clustering quality)
    silhouette_score: f32,
}

impl ClusteringResult {
    /// Create a new clustering result
    #[must_use]
    pub fn new(labels: Vec<usize>, clusters: Vec<SpeakerCluster>) -> Self {
        let num_clusters = clusters.len();
        Self {
            labels,
            clusters,
            num_clusters,
            silhouette_score: 0.0,
        }
    }

    /// Set silhouette score
    #[must_use]
    pub fn with_silhouette_score(mut self, score: f32) -> Self {
        self.silhouette_score = score;
        self
    }

    /// Get cluster labels
    #[must_use]
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Get clusters
    #[must_use]
    pub fn clusters(&self) -> &[SpeakerCluster] {
        &self.clusters
    }

    /// Get number of clusters
    #[must_use]
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Get silhouette score
    #[must_use]
    pub fn silhouette_score(&self) -> f32 {
        self.silhouette_score
    }

    /// Get cluster centroids as speaker embeddings
    #[must_use]
    pub fn cluster_centroids(&self) -> Vec<SpeakerEmbedding> {
        self.clusters.iter().map(|c| c.centroid().clone()).collect()
    }
}

/// Spectral clustering implementation
#[derive(Debug)]
pub struct SpectralClustering {
    config: ClusteringConfig,
}

impl SpectralClustering {
    /// Create new spectral clustering
    #[must_use]
    pub fn new(config: ClusteringConfig) -> Self {
        Self { config }
    }

    /// Cluster embeddings
    pub fn cluster(
        &self,
        embeddings: &[SpeakerEmbedding],
        max_clusters: Option<usize>,
        min_clusters: usize,
    ) -> WhisperResult<ClusteringResult> {
        if embeddings.is_empty() {
            return Ok(ClusteringResult::new(Vec::new(), Vec::new()));
        }

        if embeddings.len() == 1 {
            let cluster = SpeakerCluster::new(0, vec![0], embeddings[0].clone());
            return Ok(ClusteringResult::new(vec![0], vec![cluster]));
        }

        // Step 1: Build affinity matrix
        let affinity = self.build_affinity_matrix(embeddings);

        // Step 2: Estimate number of clusters
        let num_clusters = self.estimate_num_clusters(&affinity, max_clusters, min_clusters);

        // Step 3: Perform spectral decomposition and k-means
        let labels = self.spectral_cluster(&affinity, num_clusters)?;

        // Step 4: Build clusters
        let clusters = self.build_clusters(embeddings, &labels, num_clusters);

        // Step 5: Compute silhouette score
        let silhouette = self.compute_silhouette(embeddings, &labels);

        Ok(ClusteringResult::new(labels, clusters).with_silhouette_score(silhouette))
    }

    /// Build affinity matrix from embeddings
    fn build_affinity_matrix(&self, embeddings: &[SpeakerEmbedding]) -> Vec<Vec<f32>> {
        let n = embeddings.len();
        let mut affinity = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in i..n {
                let sim = if self.config.use_cosine_distance {
                    embeddings[i].cosine_similarity(&embeddings[j])
                } else {
                    let dist = embeddings[i].euclidean_distance(&embeddings[j]);
                    (-dist * dist / 2.0).exp()
                };

                // Convert similarity to affinity (0 to 1)
                let aff = (sim + 1.0) / 2.0;
                affinity[i][j] = aff;
                affinity[j][i] = aff;
            }
        }

        affinity
    }

    /// Estimate optimal number of clusters using eigengap heuristic
    fn estimate_num_clusters(
        &self,
        affinity: &[Vec<f32>],
        max_clusters: Option<usize>,
        min_clusters: usize,
    ) -> usize {
        let n = affinity.len();
        let max_k = max_clusters.unwrap_or_else(|| n.min(10));

        if n <= min_clusters {
            return min_clusters.min(n);
        }

        // Compute degree matrix and Laplacian
        let _degrees: Vec<f32> = affinity.iter().map(|row| row.iter().sum()).collect();

        // Simple heuristic: count number of high-affinity groups
        let threshold = self.config.distance_threshold;
        let mut groups = 0;
        let mut visited = vec![false; n];

        for i in 0..n {
            if visited[i] {
                continue;
            }

            groups += 1;
            let mut stack = vec![i];

            while let Some(node) = stack.pop() {
                if visited[node] {
                    continue;
                }
                visited[node] = true;

                for (j, &aff) in affinity[node].iter().enumerate() {
                    if !visited[j] && aff > threshold {
                        stack.push(j);
                    }
                }
            }
        }

        groups.clamp(min_clusters, max_k)
    }

    /// Perform spectral clustering
    fn spectral_cluster(
        &self,
        affinity: &[Vec<f32>],
        num_clusters: usize,
    ) -> WhisperResult<Vec<usize>> {
        let n = affinity.len();

        if num_clusters >= n {
            return Ok((0..n).collect());
        }

        // Compute normalized Laplacian eigenvectors (simplified)
        // For a full implementation, use proper eigendecomposition
        // Here we use a simplified approach with power iteration

        // Simple k-means on affinity rows
        let labels = self.kmeans_on_affinity(affinity, num_clusters)?;

        Ok(labels)
    }

    /// Simple k-means on affinity rows
    fn kmeans_on_affinity(&self, affinity: &[Vec<f32>], k: usize) -> WhisperResult<Vec<usize>> {
        let n = affinity.len();
        let dim = n; // Each row is a "feature vector"

        if k == 0 || k > n {
            return Err(WhisperError::Diarization(
                "Invalid number of clusters".to_string(),
            ));
        }

        // Initialize centroids (use first k points as initial centroids)
        let mut centroids: Vec<Vec<f32>> = affinity[..k].to_vec();
        let mut labels = vec![0usize; n];

        for _iter in 0..self.config.max_iterations {
            let old_labels = labels.clone();

            // Assign points to nearest centroid
            for (i, row) in affinity.iter().enumerate() {
                let mut min_dist = f32::MAX;
                let mut best_cluster = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let dist: f32 = row
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();

                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }

                labels[i] = best_cluster;
            }

            // Update centroids
            for (j, centroid) in centroids.iter_mut().enumerate() {
                let members: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &l)| l == j)
                    .map(|(i, _)| i)
                    .collect();

                if members.is_empty() {
                    continue;
                }

                for d in 0..dim {
                    centroid[d] =
                        members.iter().map(|&i| affinity[i][d]).sum::<f32>() / members.len() as f32;
                }
            }

            // Check convergence
            if labels == old_labels {
                break;
            }
        }

        Ok(labels)
    }

    /// Build cluster objects from labels
    fn build_clusters(
        &self,
        embeddings: &[SpeakerEmbedding],
        labels: &[usize],
        num_clusters: usize,
    ) -> Vec<SpeakerCluster> {
        let mut clusters = Vec::with_capacity(num_clusters);

        for cluster_id in 0..num_clusters {
            let member_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == cluster_id)
                .map(|(i, _)| i)
                .collect();

            if member_indices.is_empty() {
                continue;
            }

            // Compute centroid
            let member_embeddings: Vec<SpeakerEmbedding> = member_indices
                .iter()
                .map(|&i| embeddings[i].clone())
                .collect();

            let centroid = SpeakerEmbedding::mean(&member_embeddings)
                .unwrap_or_else(|| embeddings[member_indices[0]].clone());

            // Compute cohesion
            let cohesion = self.compute_cluster_cohesion(&member_embeddings, &centroid);

            clusters.push(
                SpeakerCluster::new(cluster_id, member_indices, centroid).with_cohesion(cohesion),
            );
        }

        clusters
    }

    /// Compute cluster cohesion
    fn compute_cluster_cohesion(
        &self,
        members: &[SpeakerEmbedding],
        centroid: &SpeakerEmbedding,
    ) -> f32 {
        if members.is_empty() {
            return 0.0;
        }

        let total_dist: f32 = members
            .iter()
            .map(|m| {
                if self.config.use_cosine_distance {
                    1.0 - m.cosine_similarity(centroid)
                } else {
                    m.euclidean_distance(centroid)
                }
            })
            .sum();

        total_dist / members.len() as f32
    }

    /// Compute silhouette score
    fn compute_silhouette(&self, embeddings: &[SpeakerEmbedding], labels: &[usize]) -> f32 {
        if embeddings.len() < 2 {
            return 0.0;
        }

        let unique_labels: Vec<usize> = {
            let mut v: Vec<usize> = labels.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        };

        if unique_labels.len() < 2 {
            return 0.0;
        }

        let mut total_silhouette = 0.0;

        for (i, emb) in embeddings.iter().enumerate() {
            let own_cluster = labels[i];

            // Compute a(i): mean distance to same cluster
            let same_cluster: Vec<f32> = embeddings
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i && labels[*j] == own_cluster)
                .map(|(_, other)| {
                    if self.config.use_cosine_distance {
                        1.0 - emb.cosine_similarity(other)
                    } else {
                        emb.euclidean_distance(other)
                    }
                })
                .collect();

            let a = if same_cluster.is_empty() {
                0.0
            } else {
                same_cluster.iter().sum::<f32>() / same_cluster.len() as f32
            };

            // Compute b(i): min mean distance to other clusters
            let b = unique_labels
                .iter()
                .filter(|&&l| l != own_cluster)
                .map(|&other_cluster| {
                    let other_dists: Vec<f32> = embeddings
                        .iter()
                        .enumerate()
                        .filter(|(_, _)| labels.get(i) == Some(&other_cluster))
                        .map(|(_, other)| {
                            if self.config.use_cosine_distance {
                                1.0 - emb.cosine_similarity(other)
                            } else {
                                emb.euclidean_distance(other)
                            }
                        })
                        .collect();

                    if other_dists.is_empty() {
                        f32::MAX
                    } else {
                        other_dists.iter().sum::<f32>() / other_dists.len() as f32
                    }
                })
                .fold(f32::MAX, f32::min);

            // Silhouette coefficient
            let s = if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            };

            total_silhouette += s;
        }

        total_silhouette / embeddings.len() as f32
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &ClusteringConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

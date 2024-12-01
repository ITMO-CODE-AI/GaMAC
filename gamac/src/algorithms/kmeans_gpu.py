from utils.utils import gpu_distance
from numba import cuda, types
import numpy as np
import sys

sys.path.append('../')

K = 3
MAX_ITERATIONS = 500

@cuda.jit(device=True)
def _init_random_centroids(X):
    """Initialize the centroids as K random samples of X"""
    n_samples, n_features = X.shape
    centroids = np.zeros((K, n_features))
    for i in range(K):
        centroid = X[np.random.choice(range(n_samples))]
        centroids[i] = centroid
    return centroids
@cuda.jit(device=True)
def _closest_centroid(sample, centroids):
    """Return the index of the closest centroid to the sample"""
    closest_i = 0
    closest_dist = float("inf")
    for i, centroid in enumerate(centroids):
        distance = gpu_distance(sample, centroid)
        if distance < closest_dist:
            closest_i = i
            closest_dist = distance
    return closest_i
@cuda.jit(device=True)
def _create_clusters(centroids, X):
    """Assign the samples to the closest centroids to create clusters"""
    # n_samples = np.shape(X)[0]
    clusters = [[] for _ in range(K)]
    for sample_i, sample in enumerate(X):
        centroid_i = _closest_centroid(sample, centroids)
        clusters[centroid_i].append(sample_i)
    return clusters
@cuda.jit(device=True)
def _calculate_centroids(clusters, X):
    """Calculate new centroids as the means of the samples in each cluster"""
    n_features = np.shape(X)[1]
    centroids = np.zeros((K, n_features))
    for i, cluster in enumerate(clusters):
        centroid = np.mean(X[cluster], axis=0)
        centroids[i] = centroid
    return centroids
@cuda.jit(device=True)
def _get_cluster_labels(clusters, X):
    """Classify samples as the index of their clusters"""
    # One prediction for each sample
    y_pred = np.zeros(np.shape(X)[0])
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred

class KMeansGpu:
    """A simple clustering method that forms K clusters by iteratively reassigning
    samples to the closest centroids and after that moves the centroids to the center
    of the new formed clusters.
    """
    @staticmethod
    @cuda.jit()
    def predict(X):
        """Do K-Means clustering and return cluster indices"""

        # Initialize centroids as K random samples from X
        centroids = _init_random_centroids(X)

        # Iterate until convergence or for max iterations
        for _ in range(MAX_ITERATIONS):
            # Assign samples to closest centroids (create clusters)
            clusters = _create_clusters(centroids, X)
            # Save current centroids for convergence check
            prev_centroids = centroids
            # Calculate new centroids from the clusters
            centroids = _calculate_centroids(clusters, X)
            # If no centroids have changed => convergence
            diff = centroids - prev_centroids
            if not diff.any():
                break

        return _get_cluster_labels(clusters, X)

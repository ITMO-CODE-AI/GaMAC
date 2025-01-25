import cupy as cp
import numpy as np
import sys

sys.path.append('../')
from utils.utils import gpu_distance

class KMeansGPU:
    """A simple clustering method that forms k clusters by iteratively reassigning
    samples to the closest centroids and after that moves the centroids to the center
    of the new formed clusters using GPU.


    Parameters:
    -----------
    k: int
        The number of clusters the algorithm will form.
    max_iterations: int
        The number of iterations the algorithm will run for if it does
        not converge before that.
    """

    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations

    def _init_random_centroids(self, X):
        """Initialize the centroids as k random samples of X"""
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def _closest_centroid(self, sample, centroids):
        """Return the index of the closest centroid to the sample"""
        closest_i = 0
        closest_dist = float("inf")
        for i, centroid in enumerate(centroids):
            distance = gpu_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def _create_clusters(self, centroids, X):
        """Assign the samples to the closest centroids to create clusters"""
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def _calculate_centroids(self, clusters, X):
        """Calculate new centroids as the means of the samples in each cluster using GPU"""
        n_features = cp.shape(X)[1]
        centroids = cp.zeros((self.k, n_features))
        get_new_centroids = cp.RawKernel(r'''
        extern "C" __global__
        void new_centroids(float* centroids, float* X) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            }''', 'new_centroids')
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids
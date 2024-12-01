from numba import cuda, types
import numpy as np
import sys

sys.path.append('../')
from utils.utils import gpu_distance

@cuda.jit(device=True)
def _get_neighbors(sample_i, X, eps):
    neighbors = []
    idxs = np.arange(len(X))
    for i, _sample in enumerate(X[idxs != sample_i]):
        distance = gpu_distance(X[sample_i], _sample)
        if distance < eps:
            neighbors.append(i)
    return np.array(neighbors)


@cuda.jit(device=True)
def _expand_cluster(sample_i, neighbors, visited_samples, min_samples, X, eps):
    """Recursive method which expands the cluster until we have reached the border
    of the dense area (density determined by eps and min_samples)"""
    cluster = [sample_i]
    # Iterate through neighbors
    for neighbor_i in neighbors:
        if neighbor_i not in visited_samples:
            visited_samples.append(neighbor_i)
            # Fetch the sample's distant neighbors (neighbors of neighbor)
            neighbors[neighbor_i] = _get_neighbors(neighbor_i, X, eps)
            # Make sure the neighbor's neighbors are more than min_samples
            # (If this is true the neighbor is a core point)
            if len(neighbors[neighbor_i]) >= min_samples:
                # Expand the cluster from the neighbor
                expanded_cluster = _expand_cluster(
                    neighbor_i, neighbors[neighbor_i], visited_samples, min_samples, X, eps
                )
                # Add expanded cluster to this cluster
                cluster = cluster + expanded_cluster
            else:
                # If the neighbor is not a core point we only add the neighbor point
                cluster.append(neighbor_i)
    return cluster


@cuda.jit(device=True)
def _get_cluster_labels(X, clusters):
    """Return the samples labels as the index of the cluster in which they are
    contained"""
    # Set default value to number of clusters
    # Will make sure all outliers have same cluster label
    labels = np.full(shape=X.shape[0], fill_value=len(clusters))
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            labels[sample_i] = cluster_i
    return labels


class DBSCANonGPU:
    def __init__(self, eps=1, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
    # DBSCAN

    @staticmethod
    @cuda.jit()
    def predict(self, X):
        self.X = X
        self.clusters = []
        self.visited_samples = []
        self.neighbors = {}
        n_samples = np.shape(self.X)[0]
        # Iterate through samples and expand clusters from them
        # if they have more neighbors than self.min_samples
        for sample_i in range(n_samples):
            if sample_i in self.visited_samples:
                continue
            self.neighbors[sample_i] = _get_neighbors(sample_i, X, self.eps)
            if len(self.neighbors[sample_i]) >= self.min_samples:
                # If core point => mark as visited
                self.visited_samples.append(sample_i)
                # Sample has more neighbors than self.min_samples => expand
                # cluster from sample
                new_cluster = _expand_cluster(sample_i, self.neighbors[sample_i], self.visited_samples, self.min_samples, self.X, self.eps)
                # Add cluster to list of clusters
                self.clusters.append(new_cluster)

        # Get the resulting cluster labels
        cluster_labels = _get_cluster_labels(self.X, self.clusters)
        return cluster_labels

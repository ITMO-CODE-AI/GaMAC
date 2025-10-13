import cupy as cp
import numpy as np
import pylibraft.config
from sklearn.neighbors import NearestNeighbors

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig

pylibraft.config.set_output_as("cupy")


class HDBSCANModel(ClusteringModel):
    """A clustering model trained by HDBSCAN algorithm.
    
    Attributes:
        labels_ (cupy.ndarray): Cluster labels for each point in the training set.
        X_ (cupy.ndarray): The training data used to fit the model.
    """
    
    def __init__(self, labels_, X_):
        """Initialize HDBSCANModel with cluster labels and training data.
        
        Args:
            labels_ (cupy.ndarray): Cluster labels for each point.
            X_ (cupy.ndarray): The training data.
        """
        super().__init__(labels_)
        self.X_ = X_

    def predict(self, X):
        """Predict cluster labels for new data points using nearest core points.
        
        Args:
            X (cupy.ndarray): New data points to predict clusters for.
            
        Returns:
            cupy.ndarray: Predicted cluster labels for each point in X.
            
        Raises:
            Exception: If model hasn't been fitted yet.
        """
        if not hasattr(self, 'labels_'):
            raise Exception("Model not fitted yet")

        nbrs = NearestNeighbors().fit(self.X_)
        _, indices = nbrs.kneighbors(X)
        labels_final = np.array([self.labels_[idx[0]] for idx in indices])
        return cp.array(labels_final, dtype=cp.int32)


class HDBSCAN(ClusteringAlgo):
    """HDBSCAN clustering algorithm implementation.
    
    Args:
        min_cluster_size (int, optional): Minimum size of clusters. Defaults to 5.
        min_samples (int, optional): Number of samples in neighborhood for core point.
            If None, set to min_cluster_size. Defaults to None.
    """
    
    def __init__(self, min_cluster_size=5, min_samples=None):
        super().__init__()
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples else min_cluster_size

    def fit(self, X):
        """Fit HDBSCAN clustering model to the data.
        
        Args:
            X (cupy.ndarray): Training data to cluster.
            
        Returns:
            HDBSCANModel: Fitted clustering model.
        """
        # Шаг 1: вычисление ближайших соседей
        nbrs = NearestNeighbors(n_neighbors=self.min_samples).fit(X.get())
        distances, indices = nbrs.kneighbors(X.get())

        # Преобразование в cupy массивы
        distances = cp.array(distances)
        indices = cp.array(indices)

        # Шаг 2: вычисление достижимости и формирования кластера
        labels_ = self._cluster(distances, indices)

        return HDBSCANModel(
            labels_=labels_, X_=X
        )

    def _cluster(self, distances, indices):
        """Perform clustering based on reachability distances.
        
        Args:
            distances (cupy.ndarray): Distances to nearest neighbors.
            indices (cupy.ndarray): Indices of nearest neighbors.
            
        Returns:
            cupy.ndarray: Cluster labels for each point.
        """
        labels = cp.full(distances.shape[0], -1, dtype=cp.int32)
        cluster_id = 0

        for i in range(distances.shape[0]):
            if labels[i] != -1:
                continue

            # Проверка, является ли точка ядром кластера
            if cp.sum(distances[i, :] <= distances[i, -1]) >= self.min_cluster_size:
                labels[i] = cluster_id
                self._expand_cluster(i, indices, labels, cluster_id)
                cluster_id += 1

        return labels  # Преобразование обратно в numpy массив

    def _expand_cluster(self, point_index, indices, labels, cluster_id):
        """Expand cluster from core point to its neighbors.
        
        Args:
            point_index (int): Index of core point to expand from.
            indices (cupy.ndarray): Indices of nearest neighbors for all points.
            labels (cupy.ndarray): Cluster labels array to update.
            cluster_id (int): Current cluster ID to assign.
        """
        stack = [point_index]
        while stack:
            current_point = stack.pop()
            for neighbor_index in indices[current_point]:
                if labels[neighbor_index] == -1:
                    labels[neighbor_index] = cluster_id
                    stack.append(neighbor_index)


class HDBSCANConfig(AlgoConfig):
    """Configuration for HDBSCAN algorithm hyperparameter optimization.
    
    Args:
        min_cluster_size (tuple, optional): Range for min_cluster_size parameter. Defaults to (5, 15).
        min_samples (tuple, optional): Range for min_samples parameter. Defaults to (5, 15).
    """
    
    def __init__(
            self, *,
            min_cluster_size=(5, 15),
            min_samples=(5, 15),
    ):
        super().__init__(
            HDBSCAN,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
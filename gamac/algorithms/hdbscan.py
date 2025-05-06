import cupy as cp
import numpy as np
import pylibraft.config
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class HDBSCANModel(ClusteringModel):
    def __init__(self, labels_):
        super().__init__(labels_)
        self.labels_ = labels_

    def predict(self, X):
        """Predict using nearest core points"""
        X = cp.asarray(X, dtype=cp.float32)
        sum_X = cp.sum(X**2, axis=1)
        dists = cp.sqrt(cp.abs(sum_X[:, None] + sum_X[None, :] - 2 * cp.dot(X, X.T)))
        nearest = cp.argmin(dists, axis=1)
        return self.labels_[nearest.get()]

class HDBSCAN(ClusteringAlgo):
    def __init__(self, min_cluster_size=5, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples else min_cluster_size

    def fit(self, X):
        # Шаг 1: вычисление ближайших соседей
        nbrs = NearestNeighbors(n_neighbors=self.min_samples).fit(X.get())
        distances, indices = nbrs.kneighbors(X.get())

        # Преобразование в cupy массивы
        distances = cp.array(distances)
        indices = cp.array(indices)

        # Шаг 2: вычисление достижимости и формирования кластера
        self.labels_ = self._cluster(distances, indices)
        self.labels_ = cp.array(self.labels_, dtype=cp.int32)

        return HDBSCANModel(
            labels_=self.labels_
        )

    def _cluster(self, distances, indices):
        labels = cp.full(distances.shape[0], -1)
        cluster_id = 0

        for i in range(distances.shape[0]):
            if labels[i] != -1:
                continue

            # Проверка, является ли точка ядром кластера
            if cp.sum(distances[i, :] <= distances[i, -1]) >= self.min_cluster_size:
                labels[i] = cluster_id
                self._expand_cluster(i, indices, labels, cluster_id)
                cluster_id += 1

        return cp.asnumpy(labels)  # Преобразование обратно в numpy массив

    def _expand_cluster(self, point_index, indices, labels, cluster_id):
        stack = [point_index]
        while stack:
            current_point = stack.pop()
            for neighbor_index in indices[current_point]:
                if labels[neighbor_index] == -1:
                    labels[neighbor_index] = cluster_id
                    stack.append(neighbor_index)

    def predict(self, X):
        if not hasattr(self, 'labels_'):
            raise Exception("Model not fitted yet")

        nbrs = NearestNeighbors().fit(self.X)
        _, indices = nbrs.kneighbors(X)
        return np.array([self.labels_[idx[0]] for idx in indices])


class HDBSCANConfig(AlgoConfig):
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
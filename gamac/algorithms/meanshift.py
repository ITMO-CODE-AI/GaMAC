import cupy as cp
import pylibraft.config

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class MeanShiftModel(ClusteringModel):
    def __init__(self, labels_, centroids_):
        super().__init__(labels_)
        self.centroids_ = centroids_

    def predict(self, X: DataFrameType) -> LabelsType:
        if self.centroids_ is None:
            raise ValueError("Модель еще не обучена!")

        # Оптимизированное вычисление квадратов расстояний
        labels = cp.zeros(X.shape[0], dtype=cp.int32)
        for i, x in enumerate(X):
            distances = cp.linalg.norm(self.centroids_ - x, axis=1)
            labels[i] = cp.argmin(distances)
        return labels


class MeanShift(ClusteringAlgo):
    def __init__(self, bandwidth=1.0, max_iter=300, tol=1e-3):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        centroids = X.copy()

        for _ in range(self.max_iter):
            max_shift = 0.0
            for i in range(len(centroids)):
                centroid = centroids[i]
                distances = cp.linalg.norm(X - centroid, axis=1)
                in_window = distances <= self.bandwidth
                if not cp.any(in_window):
                    continue
                new_centroid = cp.mean(X[in_window], axis=0)
                shift = cp.linalg.norm(new_centroid - centroid)
                centroids[i] = new_centroid
                max_shift = max(max_shift, shift)
            if max_shift < self.tol:
                break

        # Объединение центроидов
        unique_centroids = []
        for centroid in centroids:
            if not unique_centroids:
                unique_centroids.append(centroid)
                continue
            distances = cp.linalg.norm(cp.array(unique_centroids) - centroid, axis=1)
            if cp.min(distances) > self.bandwidth:
                unique_centroids.append(centroid)
        self.centroids = cp.array(unique_centroids, dtype=cp.float32)

        # Назначение меток
        labels = self._assign_labels(X)
        return MeanShiftModel(labels_=labels, centroids_=self.centroids)

    def _assign_labels(self, X):
        labels = cp.empty(X.shape[0], dtype=cp.int32)
        for i, x in enumerate(X):
            distances = cp.linalg.norm(self.centroids - x, axis=1)
            labels[i] = cp.argmin(distances, dtype=cp.int32)
        return labels


class MeanShiftConfig(AlgoConfig):
    def __init__(
            self, *,
            bandwidth=(1e-4, 1.0),
            max_iter=(50, 300),
            tol=(1e-5, 1e-4)
    ):
        super().__init__(
            MeanShift,
            bandwidth=bandwidth,
            max_iter=max_iter,
            tol=tol
        )

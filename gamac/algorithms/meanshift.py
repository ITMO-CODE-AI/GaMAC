import cupy as cp
import pylibraft.config

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class MeanShiftModel(ClusteringModel):
    def __init__(self, labels_, centroids_):
        super().__init__(labels_)
        self.centroids_ = centroids_

    def predict(self, data):
        data = cp.asarray(data, dtype=cp.float32)
        if self.centroids_ is None:
            raise ValueError("Модель еще не обучена!")

        # Оптимизированное вычисление квадратов расстояний
        data_sq = cp.sum(data**2, axis=1, keepdims=True)
        centroids_sq = cp.sum(self.centroids_**2, axis=1)
        dot_product = cp.dot(data, self.centroids_.T)
        distances_sq = data_sq + centroids_sq - 2 * dot_product

        return cp.argmin(distances_sq, axis=1)


class MeanShift(ClusteringAlgo):
    def __init__(self, radius=4, max_iter=100):
        super().__init__()
        self.radius = radius
        self.radius_sq = radius**2  # Предвычисляем квадрат радиуса
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, data):
        data_cp = cp.asarray(data, dtype=cp.float32)
        centroids = data_cp.copy()
        optimized = False
        iteration = 0

        while not optimized and iteration < self.max_iter:
            iteration += 1

            # Оптимизированное вычисление масок через квадраты расстояний
            centroids_sq = cp.sum(centroids**2, axis=1)[:, cp.newaxis]
            data_sq = cp.sum(data_cp**2, axis=1)[cp.newaxis, :]
            dot_product = cp.dot(centroids, data_cp.T)
            distances_sq = centroids_sq + data_sq - 2 * dot_product

            masks = distances_sq < self.radius_sq

            # Векторизованное вычисление новых центроидов
            sum_data = masks.astype(cp.float32) @ data_cp
            counts = cp.sum(masks, axis=1, keepdims=True)
            valid = counts > 0
            new_centroids = cp.where(valid, sum_data / counts, centroids)

            # Оптимизированное определение уникальности
            rounded = cp.round(new_centroids, decimals=4)
            unique_centroids = cp.unique(rounded, axis=0)

            if unique_centroids.shape == centroids.shape:
                optimized = True
            centroids = unique_centroids

        self.centroids = centroids
        labels = self.predict(data)
        labels = cp.array(labels, dtype=cp.int32)
        return MeanShiftModel(labels_=labels, centroids_=self.centroids)

    def predict(self, data):
        data = cp.asarray(data, dtype=cp.float32)
        if self.centroids is None:
            raise ValueError("Модель еще не обучена!")

        # Повторно используем оптимизированный расчет расстояний
        data_sq = cp.sum(data**2, axis=1, keepdims=True)
        centroids_sq = cp.sum(self.centroids**2, axis=1)
        dot_product = cp.dot(data, self.centroids.T)
        distances_sq = data_sq + centroids_sq - 2 * dot_product

        return cp.argmin(distances_sq, axis=1).get()


class MeanShiftConfig(AlgoConfig):
    def __init__(
            self, *,
            radius=(2, 8),
            max_iter=100
    ):
        super().__init__(
            MeanShift,
            radius=radius,
            max_iter=max_iter
        )

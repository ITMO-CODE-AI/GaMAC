import cupy as cp
import numpy as np
import pylibraft.config

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class BisectingKMeansModel(ClusteringModel):
    def __init__(self, labels_, centroids_):
        super().__init__(labels_)
        self.centroids_ = centroids_

    def predict(self, df: DataFrameType) -> LabelsType:
        diff = df[:, None] - self.centroids_
        distances = cp.linalg.norm(diff, axis=2)
        return cp.argmin(distances, axis=1)


class BisectingKMeans(ClusteringAlgo):
    def __init__(
            self,
            n_clusters=2,
            max_iter=100,
            init='k-means++',
            tol=1e-4
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.tol = tol

    def _kmeans_pp_init(self, data, k):
        """Инициализация центров методом K-Means++"""
        n, d = data.shape
        centers = cp.empty(shape=(k, d), dtype=cp.float32)

        # Первый центр выбирается случайно
        first_idx = cp.random.randint(n)
        centers[0] = data[first_idx]

        for i in range(1, k):
            # Вычисление расстояний до ближайшего центра
            diff = data[:, None] - centers[:i]
            distances = cp.linalg.norm(diff, axis=2)
            min_dists = cp.min(distances, axis=1)
            # Вероятности пропорциональны квадрату расстояний
            probs = min_dists ** 2
            probs /= cp.sum(probs)
            # Выбор следующего центра
            next_idx = cp.where(cp.random.rand() < cp.cumsum(probs))[0][0]
            centers[i] = data[next_idx]
        return centers

    def _random_init(self, data, k):
        n = data.shape[0]
        idx = cp.random.choice(n, k, replace=False)
        return data[idx]

    def _init_centroids(self, data, k):
        match self.init:
            case 'k-means++':
                return self._kmeans_pp_init(data, k)
            case 'random':
                return self._random_init(data, k)
            case _:
                raise ValueError

    def _kmeans(self, data, k):
        """Оптимизированный K-Means с использованием CuPy"""
        centers = self._init_centroids(data, k)

        for _ in range(self.max_iter):
            # Векторизованное вычисление расстояний
            diff = data[:, None] - centers
            distances = cp.linalg.norm(diff, axis=2)
            labels = cp.argmin(distances, axis=1)

            # Векторизованное обновление центров
            new_centers = cp.zeros_like(centers)
            for i in range(k):
                mask = (labels == i)
                if cp.any(mask):
                    new_centers[i] = data[mask].mean(axis=0)
                else:
                    new_centers[i] = centers[i]

            # Проверка схождения
            if cp.linalg.norm(centers - new_centers) < self.tol:
                break
            centers = new_centers

        return labels, centers

    def _sse(self, data, labels):
        """Векторизованный расчет SSE"""
        unique_labels = cp.unique(labels)
        sse_accumulator = 0.0
        for lbl in unique_labels:
            cluster = data[labels == lbl]
            if len(cluster) > 0:
                centroid = cluster.mean(axis=0)
                diff = cluster - centroid
                sse_accumulator += cp.sum(diff * diff).item()
        return sse_accumulator

    def fit(self, df: DataFrameType) -> BisectingKMeansModel:
        N, D = df.shape
        clusters = [cp.arange(N)]

        while len(clusters) < self.n_clusters:
            sse_values = []
            kmeans_results = []

            for cluster in clusters:
                cluster_data = df[cluster]
                if len(cluster_data) < 2:
                    sse_values.append(0.0)
                    kmeans_results.append(None)
                    continue

                labels, centers = self._kmeans(cluster_data, 2)
                kmeans_results.append((labels, centers))
                sse_values.append(self._sse(cluster_data, labels))

            # Выбор кластера для разделения
            max_sse_idx = np.argmax(sse_values)
            if sse_values[max_sse_idx] <= 0:
                break  # Не осталось кластеров для разделения

            # Обновление кластеров
            cluster_to_split = clusters.pop(max_sse_idx)
            labels = kmeans_results[max_sse_idx][0]
            clusters.extend([
                cluster_to_split[labels == 0],
                cluster_to_split[labels == 1]
            ])

        # Формирование финальных меток
        labels_ = cp.full(shape=N, fill_value=-1, dtype=cp.int32)
        for i, cluster in enumerate(clusters):
            labels_[cluster] = i

        # Вычисление центров
        centroids_ = cp.stack([
            df[labels_ == i].mean(axis=0)
            for i in range(self.n_clusters)
        ])

        return BisectingKMeansModel(
            labels_=labels_,
            centroids_=centroids_
        )


class BisectingKMeansConfig(AlgoConfig):
    def __init__(
            self, *,
            n_clusters=(2, 15),
            init=frozenset(['random', 'k-means++']),
            max_iter=100,
            tol=1e-4
    ):
        super().__init__(
            BisectingKMeans,
            n_clusters=n_clusters,
            max_iter=max_iter,
            init=init,
            tol=tol
        )

"""Реализация алгоритма Bisecting K-Means для GPU.

Модуль содержит:
- Класс BisectingKMeansModel: модель кластеризации Bisecting K-Means
- Класс BisectingKMeans: основной алгоритм Bisecting K-Means
- Класс BisectingKMeansConfig: конфигурация алгоритма
"""

import cupy as cp
import numpy as np
import pylibraft.config

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class BisectingKMeansModel(ClusteringModel):
    """Модель кластеризации Bisecting K-Means.
    
    Хранит метки кластеров и центроиды для предсказания.
    
    Attributes:
        centroids_ (cp.ndarray): Центроиды кластеров.
    """

    def __init__(self, labels_, centroids_):
        """Инициализирует модель Bisecting K-Means.
        
        Args:
            labels_: Метки кластеров для обучающих данных.
            centroids_: Центроиды кластеров.
        """
        super().__init__(labels_)
        self.centroids_ = centroids_

    def predict(self, df: DataFrameType) -> LabelsType:
        """Предсказывает метки кластеров для новых данных.
        
        Args:
            df: Данные для предсказания.
            
        Returns:
            Метки кластеров для входных данных.
        """
        diff = df[:, None] - self.centroids_
        distances = cp.linalg.norm(diff, axis=2)
        return cp.argmin(distances, axis=1)


class BisectingKMeans(ClusteringAlgo):
    """Алгоритм Bisecting K-Means для GPU.
    
    Иерархический алгоритм кластеризации, который рекурсивно делит данные 
    на две части, используя K-Means с K=2 на каждом шаге.
    
    Attributes:
        n_clusters (int): Количество кластеров.
        max_iter (int): Максимальное количество итераций.
        init (str): Метод инициализации центроидов ('k-means++' или 'random').
        tol (float): Порог сходимости.
    """

    def __init__(
            self,
            n_clusters=2,
            max_iter=20,
            init='k-means++',
            tol=1e-4
    ):
        """Инициализирует алгоритм Bisecting K-Means.
        
        Args:
            n_clusters: Количество кластеров.
            max_iter: Максимальное количество итераций.
            init: Метод инициализации ('k-means++' или 'random').
            tol: Порог сходимости.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.tol = tol

    def _kmeans_pp_init(self, data, k):
        """Инициализация центроидов методом K-Means++.
        
        Args:
            data: Входные данные.
            k: Количество центроидов.
            
        Returns:
            Инициализированные центроиды.
        """
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
        """Случайная инициализация центроидов.
        
        Args:
            data: Входные данные.
            k: Количество центроидов.
            
        Returns:
            Случайно выбранные центроиды.
        """
        n = data.shape[0]
        idx = cp.random.choice(n, k, replace=False)
        return data[idx]

    def _init_centroids(self, data, k):
        """Выбор метода инициализации центроидов.
        
        Args:
            data: Входные данные.
            k: Количество центроидов.
            
        Returns:
            Инициализированные центроиды.
            
        Raises:
            ValueError: Если указан неизвестный метод инициализации.
        """
        match self.init:
            case 'k-means++':
                return self._kmeans_pp_init(data, k)
            case 'random':
                return self._random_init(data, k)
            case _:
                raise ValueError("Unknown initialization method")

    def _kmeans(self, data, k):
        """Оптимизированный K-Means с использованием CuPy.
        
        Args:
            data: Входные данные.
            k: Количество кластеров.
            
        Returns:
            Кортеж (метки, центроиды).
        """
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
        """Векторизованный расчет суммы квадратов ошибок (SSE).
        
        Args:
            data: Входные данные.
            labels: Метки кластеров.
            
        Returns:
            Сумма квадратов ошибок.
        """
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
        """Обучает модель Bisecting K-Means на данных.
        
        Args:
            df: Данные для обучения.
            
        Returns:
            Обученная модель BisectingKMeansModel.
        """
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

        # Вычисление центроидов
        centroids_ = cp.stack([
            df[labels_ == i].mean(axis=0)
            for i in range(self.n_clusters)
        ])

        return BisectingKMeansModel(
            labels_=labels_,
            centroids_=centroids_
        )


class BisectingKMeansConfig(AlgoConfig):
    """Конфигурация алгоритма Bisecting K-Means.
    
    Attributes:
        config_space (dict): Пространство параметров:
            - n_clusters: Диапазон количества кластеров (по умолчанию (2, 15))
            - init: Методы инициализации (по умолчанию {'random', 'k-means++'})
            - max_iter: Максимальное количество итераций (по умолчанию 100)
            - tol: Порог сходимости (по умолчанию 1e-4)
    """

    def __init__(
            self, *,
            n_clusters=(2, 15),
            init=frozenset(['random', 'k-means++']),
            max_iter=20,
            tol=1e-4
    ):
        """Инициализирует конфигурацию Bisecting K-Means.
        
        Args:
            n_clusters: Диапазон количества кластеров.
            init: Доступные методы инициализации.
            max_iter: Максимальное количество итераций.
            tol: Порог сходимости.
        """
        super().__init__(
            BisectingKMeans,
            n_clusters=n_clusters,
            max_iter=max_iter,
            init=init,
            tol=tol
        )
import cupy as cp
import numpy as np
import pylibraft.config

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class MeanShiftModel(ClusteringModel):
    r"""Модель кластеризации, обученная алгоритмом MeanShift.
    
    Атрибуты:
        labels\_ (cupy.ndarray): Метки кластеров для каждой точки обучающей выборки.
        centroids\_ (cupy.ndarray): Координаты центроидов кластеров.
    """
    
    def __init__(self, labels_, centroids_):
        """Инициализация модели MeanShift.
        
        Аргументы:
            labels_ (cupy.ndarray): Метки кластеров для каждой точки.
            centroids_ (cupy.ndarray): Координаты центроидов кластеров.
        """
        super().__init__(labels_)
        self.centroids_ = centroids_

    def predict(self, X: DataFrameType) -> LabelsType:
        """Предсказание меток кластеров для новых данных.
        
        Аргументы:
            X (DataFrameType): Массив новых данных для предсказания.
            
        Возвращает:
            LabelsType: Массив предсказанных меток кластеров.
            
        Выбрасывает:
            ValueError: Если модель не была обучена (отсутствуют центроиды).
        """
        if self.centroids_ is None:
            raise ValueError("Модель еще не обучена!")

        # Оптимизированное вычисление квадратов расстояний
        labels = cp.zeros(X.shape[0], dtype=cp.int32)
        for i, x in enumerate(X):
            distances = cp.linalg.norm(self.centroids_ - x, axis=1)
            labels[i] = cp.argmin(distances)
        return labels


class MeanShift(ClusteringAlgo):
    """Реализация алгоритма MeanShift для кластеризации с использованием GPU.
    
    Алгоритм основан на сдвиге точек в направлении увеличения плотности распределения данных.
    
    Параметры:
        bandwidth (float): Ширина окна для поиска соседей (по умолчанию 1.0).
        max_iter (int): Максимальное количество итераций (по умолчанию 300).
        tol (float): Допустимое изменение центроидов для остановки (по умолчанию 1e-3).
    """
    
    def __init__(self, bandwidth=0.5, max_iter=5, tol=1e-4):
        """Инициализация алгоритма MeanShift."""
        super().__init__()
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        """Обучение модели MeanShift на переданных данных.
        
        Аргументы:
            X (cupy.ndarray): Обучающие данные для кластеризации.
            
        Возвращает:
            MeanShiftModel: Обученная модель кластеризации.
        """
        centroids = cp.random.choice(X, np.sqrt(len(X)))

        for _ in range(self.max_iter):
            max_shift = 0.0
            for i, centroid, in enumerate(centroids):
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
        """Назначение меток кластеров точкам на основе ближайшего центроида.
        
        Аргументы:
            X (cupy.ndarray): Данные для кластеризации.
            
        Возвращает:
            cupy.ndarray: Массив меток кластеров.
        """
        labels = cp.empty(X.shape[0], dtype=cp.int32)
        for i, x in enumerate(X):
            distances = cp.linalg.norm(self.centroids - x, axis=1)
            labels[i] = cp.argmin(distances, dtype=cp.int32)
        return labels


class MeanShiftConfig(AlgoConfig):
    """Конфигурация для подбора гиперпараметров алгоритма MeanShift.
    
    Параметры:
        bandwidth (tuple): Диапазон для подбора ширины окна (по умолчанию (1e-4, 1.0)).
        max_iter (tuple): Диапазон для подбора максимального числа итераций (по умолчанию (50, 300)).
        tol (tuple): Диапазон для подбора допустимого изменения центроидов (по умолчанию (1e-5, 1e-4)).
    """
    
    def __init__(
            self, *,
            bandwidth=(1e-4, 1.0),
            max_iter=5,
            tol=1e-4,
    ):
        """Инициализация конфигурации для MeanShift."""
        super().__init__(
            MeanShift,
            bandwidth=bandwidth,
            max_iter=max_iter,
            tol=tol
        )
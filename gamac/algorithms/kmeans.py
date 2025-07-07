import cupy as cp
import numpy as np
import pylibraft.config

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class KMeansModel(ClusteringModel):
    r"""Модель кластеризации, обученная алгоритмом K-средних.
    
    Атрибуты:
        labels\_ (cupy.ndarray): Метки кластеров для каждой точки обучающей выборки.
        centroids\_ (cupy.ndarray): Координаты центроидов кластеров.
    """
    
    def __init__(self, labels_, centroids_):
        """Инициализация модели KMeans.
        
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
        """
        # Вычисление квадратов норм
        x_squared = cp.sum(X**2, axis=1)[:, cp.newaxis]
        centroids_squared = cp.sum(self.centroids_**2, axis=1)[cp.newaxis, :]

        # Вычисление расстояний и определение меток
        distances = x_squared + centroids_squared - 2 * X.dot(self.centroids_.T)
        return cp.argmin(distances, axis=1)


class KMeans(ClusteringAlgo):
    """Алгоритм K-средних для кластеризации с использованием GPU.
    
    Простой метод кластеризации, который формирует k кластеров, итеративно перераспределяя
    точки к ближайшим центроидам и затем перемещая центроиды в центр новых кластеров.
    
    Параметры:
    -----------
    n_clusters: int, по умолчанию=2
        Количество кластеров, которые нужно сформировать.
    max_iter: int, по умолчанию=100
        Максимальное количество итераций алгоритма, если раньше не достигнута сходимость.
    tol: float, по умолчанию=1e-4
        Допустимое изменение центроидов для определения сходимости.
    random_state: int, по умолчанию=None
        Зерно для генератора случайных чисел для воспроизводимости.
    """
    
    def __init__(
            self,
            n_clusters=2,
            max_iter=100,
            tol=1e-4,
            random_state=None
    ):
        """Инициализация алгоритма K-средних."""
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """Обучение модели K-средних на переданных данных.
        
        Аргументы:
            X (cupy.ndarray): Обучающие данные для кластеризации.
            
        Возвращает:
            KMeansModel: Обученная модель кластеризации.
        """
        # Установка случайного зерна для воспроизводимости
        if self.random_state is not None:
            cp.random.seed(self.random_state)

        # Инициализация центроидов: случайный выбор уникальных точек из данных
        n_samples = X.shape[0]
        indices = cp.random.permutation(n_samples)[:self.n_clusters]
        self.centroids = X[indices]

        for _ in range(self.max_iter):
            # Вычисление квадратов норм для X и центроидов
            x_squared = cp.sum(X**2, axis=1)[:, cp.newaxis]
            centroids_squared = cp.sum(self.centroids**2, axis=1)[cp.newaxis, :]

            # Вычисление матрицы расстояний между точками и центроидами
            distances = x_squared + centroids_squared - 2 * X.dot(self.centroids.T)

            # Назначение меток: индекс ближайшего центроида для каждой точки
            labels = cp.argmin(distances, axis=1, dtype=cp.int32)

            # Обновление центроидов
            new_centroids = cp.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                # Выбор точек, принадлежащих текущему кластеру
                cluster_points = X[labels == i]
                if cluster_points.shape[0] > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    # Если кластер пуст, сохраняем прежний центроид
                    new_centroids[i] = self.centroids[i]

            # Проверка условия сходимости (изменение центроидов)
            centroid_shift = cp.max(cp.linalg.norm(new_centroids - self.centroids, axis=1))
            if centroid_shift <= self.tol:
                break

            self.centroids = new_centroids

        return KMeansModel(
            labels_=labels,
            centroids_=self.centroids
        )


class KMeansConfig(AlgoConfig):
    """Конфигурация для подбора гиперпараметров алгоритма K-средних.
    
    Параметры:
    -----------
    n_clusters: tuple, по умолчанию=(2, 15)
        Диапазон для подбора оптимального количества кластеров.
    max_iter: int, по умолчанию=100
        Максимальное количество итераций алгоритма.
    tol: float, по умолчанию=1e-4
        Допустимое изменение центроидов для определения сходимости.
    """
    
    def __init__(
            self, *,
            n_clusters=(2, 15),
            max_iter=100,
            tol=1e-4
    ):
        """Инициализация конфигурации для K-средних."""
        super().__init__(
            KMeans,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol
        )
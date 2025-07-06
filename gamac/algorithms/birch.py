"""Реализация алгоритма BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) для GPU.

Модуль содержит:
- Класс BirchModel: модель кластеризации BIRCH
- Класс ClusteringFeatureGPU: признаки кластеризации для GPU
- Класс CFNodeGPU: узел дерева CF (Clustering Feature)
- Класс CFTreeGPU: дерево CF для GPU
- Класс Birch: основной алгоритм BIRCH
- Класс BirchConfig: конфигурация алгоритма BIRCH
"""

import cupy as cp
import numpy as np
import pylibraft.config

from gamac.algorithms.kmeans import KMeans
from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class BirchModel(ClusteringModel):
    """Модель кластеризации BIRCH.
    
    Хранит метки кластеров, подкластеров и структуру дерева для предсказания.
    
    Attributes:
        subcluster_labels (LabelsType): Метки подкластеров.
        tree (CFTreeGPU): Построенное дерево CF.
    """

    def __init__(self, labels_, subcluster_labels, tree):
        """Инициализирует модель BIRCH.
        
        Args:
            labels_: Метки кластеров для обучающих данных.
            subcluster_labels: Метки подкластеров.
            tree: Построенное дерево CF.
        """
        super().__init__(labels_)
        self.subcluster_labels = subcluster_labels
        self.tree = tree

    def predict(self, X: DataFrameType) -> LabelsType:
        """Предсказывает метки кластеров для новых данных.
        
        Args:
            X: Данные для предсказания.
            
        Returns:
            Метки кластеров для входных данных.
        """
        subclusters = [cf.centroid().get() for cf in self.tree.root.cfs]
        labels = []
        for point in X:
            closest = np.argmin([np.linalg.norm(point.get() - sc) for sc in subclusters])
            labels.append(self.subcluster_labels[closest])
        return cp.array(labels, dtype=cp.int32)


class ClusteringFeatureGPU:
    """Признак кластеризации (CF) для GPU.
    
    Хранит статистику для группы точек:
    - n: количество точек
    - LS: линейная сумма
    - SS: квадратичная сумма
    
    Attributes:
        n (int): Количество точек в CF.
        LS (cp.ndarray): Линейная сумма точек.
        SS (cp.ndarray): Квадратичная сумма точек.
    """

    def __init__(self, point):
        """Инициализирует CF с одной точкой.
        
        Args:
            point: Начальная точка для CF.
        """
        self.n = 1
        self.LS = cp.array(point, dtype=cp.float64)
        self.SS = cp.square(point)

    def add_point(self, point):
        """Добавляет точку к CF.
        
        Args:
            point: Точка для добавления.
        """
        self.n += 1
        self.LS += point
        self.SS += cp.square(point)

    def merge(self, other):
        """Объединяет два CF.
        
        Args:
            other: Другой CF для объединения.
        """
        self.n += other.n
        self.LS += other.LS
        self.SS += other.SS

    def centroid(self):
        """Вычисляет центроид CF.
        
        Returns:
            Центроид CF.
        """
        return self.LS / self.n

    def radius(self):
        """Вычисляет радиус CF.
        
        Returns:
            Радиус CF.
        """
        return cp.sqrt(cp.sum(self.SS / self.n - cp.square(self.centroid())))


class CFNodeGPU:
    """Узел дерева CF для GPU.
    
    Attributes:
        threshold (float): Порог для объединения CF.
        branching_factor (int): Максимальное количество потомков.
        is_leaf (bool): Является ли узел листом.
        cfs (list): Список CF в узле.
        children (list): Список дочерних узлов.
        next: Ссылка на следующий узел-лист.
    """

    def __init__(self, threshold, branching_factor, is_leaf=True):
        """Инициализирует узел дерева CF.
        
        Args:
            threshold: Порог для объединения CF.
            branching_factor: Максимальное количество потомков.
            is_leaf: Является ли узел листом.
        """
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.cfs = []
        self.children = []
        self.next = None

    def split(self):
        """Разделяет узел при превышении branching_factor."""
        if self.is_leaf:
            new_node = CFNodeGPU(self.threshold, self.branching_factor, is_leaf=True)
            half = len(self.cfs) // 2
            new_node.cfs = self.cfs[half:]
            self.cfs = self.cfs[:half]
            new_node.next = self.next
            self.next = new_node

    def insert(self, cf, data):
        """Вставляет CF в узел.
        
        Args:
            cf: CF для вставки.
            data: Соответствующие данные.
        """
        if not self.is_leaf:
            closest = self.find_closest_child(data)
            self.children[closest].insert(cf, data)
        else:
            if self.cfs:
                distances = [cp.linalg.norm(cf.centroid() - existing_cf.centroid()) 
                             for existing_cf in self.cfs]
                closest = cp.argmin(cp.array(distances)).item()

                if distances[closest] <= self.threshold:
                    self.cfs[closest].merge(cf)
                else:
                    self.cfs.append(cf)
                    if len(self.cfs) > self.branching_factor:
                        self.split()
            else:
                self.cfs.append(cf)

    def find_closest_child(self, data):
        """Находит ближайший дочерний узел.
        
        Args:
            data: Точка данных.
            
        Returns:
            Индекс ближайшего дочернего узла.
        """
        centroids = [cf.centroid().get() for cf in self.cfs]
        distances = [cp.linalg.norm(data - centroid) for centroid in centroids]
        return cp.argmin(cp.array(distances)).item()


class CFTreeGPU:
    """Дерево CF для GPU.
    
    Attributes:
        threshold (float): Порог для объединения CF.
        branching_factor (int): Максимальное количество потомков.
        root (CFNodeGPU): Корневой узел дерева.
    """

    def __init__(self, threshold, branching_factor):
        """Инициализирует дерево CF.
        
        Args:
            threshold: Порог для объединения CF.
            branching_factor: Максимальное количество потомков.
        """
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.root = CFNodeGPU(threshold, branching_factor, is_leaf=True)

    def insert(self, data):
        """Вставляет точку данных в дерево.
        
        Args:
            data: Точка данных для вставки.
        """
        cf = ClusteringFeatureGPU(data)
        self.root.insert(cf, data)


class Birch(ClusteringAlgo):
    """Алгоритм BIRCH для GPU.
    
    Attributes:
        threshold (float): Порог для объединения CF.
        branching_factor (int): Максимальное количество потомков.
        n_clusters (int): Количество кластеров.
        subcluster_labels (LabelsType): Метки подкластеров.
        tree (CFTreeGPU): Построенное дерево CF.
    """

    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        """Инициализирует алгоритм BIRCH.
        
        Args:
            threshold: Порог для объединения CF.
            branching_factor: Максимальное количество потомков.
            n_clusters: Количество кластеров.
        """
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.subcluster_labels = None

    def fit(self, X):
        """Обучает модель BIRCH на данных.
        
        Args:
            X: Данные для обучения.
            
        Returns:
            Обученная модель BirchModel.
        """
        self.tree = CFTreeGPU(self.threshold, self.branching_factor)
        for point in X:
            self.tree.insert(point)

        subclusters = []
        node = self.tree.root
        while node:
            subclusters.extend([cf.centroid().get() for cf in node.cfs])
            node = node.next

        if len(subclusters) < 2:
            self.subcluster_labels = np.zeros(len(subclusters), dtype=int)
            labels_ = cp.zeros(len(X), dtype=cp.int32)
            return BirchModel(labels_=labels_, subcluster_labels=self.subcluster_labels, tree=self.tree)

        n_clusters = min(self.n_clusters, len(subclusters))
        subclusters = cp.array(subclusters, dtype=cp.float32)
        clustering = KMeans(n_clusters=n_clusters)
        self.subcluster_labels = clustering.fit(subclusters).labels_

        labels_ = []
        subcluster_centers = np.array([cf.centroid().get() for cf in self.tree.root.cfs])
        for point in X:
            point_cpu = point.get()
            closest = np.argmin(np.linalg.norm(subcluster_centers - point_cpu, axis=1))
            labels_.append(self.subcluster_labels[closest])
        labels_ = cp.array(labels_, dtype=cp.int32)

        return BirchModel(labels_=labels_, subcluster_labels=self.subcluster_labels, tree=self.tree)


class BirchConfig(AlgoConfig):
    """Конфигурация алгоритма BIRCH.
    
    Attributes:
        config_space (dict): Пространство параметров:
            - threshold: Диапазон порогов (по умолчанию (0.1, 0.9))
            - branching_factor: Диапазон branching factors (по умолчанию (10, 80))
            - n_clusters: Диапазон количества кластеров (по умолчанию (2, 15))
    """

    def __init__(
            self, *,
            threshold=(0.1, 0.9),
            branching_factor=(10, 80),
            n_clusters=(2, 15),
    ):
        """Инициализирует конфигурацию BIRCH.
        
        Args:
            threshold: Диапазон порогов.
            branching_factor: Диапазон branching factors.
            n_clusters: Диапазон количества кластеров.
        """
        super().__init__(
            Birch,
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters,
        )
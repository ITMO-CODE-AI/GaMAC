import cupy as cp
import pylibraft.config

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class ClusteringFeature:
    def __init__(self, point=None, dim=0):
        self.n_points = 0
        self.dim = dim
        self.linear_sum = cp.zeros(dim, dtype=cp.float32) if dim > 0 else cp.array([], dtype=cp.float32)
        self.squared_sum = 0.0
        if point is not None:
            self.add_point(point)

    def add_point(self, point):
        if self.dim == 0 and self.linear_sum.size == 0:
            self.dim = point.size
            self.linear_sum = cp.zeros(self.dim, dtype=cp.float32)
        
        self.n_points += 1
        self.linear_sum += point
        self.squared_sum += cp.sum(point ** 2)

    def merge(self, other):
        if self.dim == 0 and other.dim > 0:
            self.dim = other.dim
            self.linear_sum = cp.zeros(self.dim, dtype=cp.float32)
            
        self.n_points += other.n_points
        self.linear_sum += other.linear_sum
        self.squared_sum += other.squared_sum

    @property
    def centroid(self):
        return self.linear_sum / self.n_points if self.n_points > 0 else None

    @property
    def radius(self):
        if self.n_points == 0: return 0.0
        return cp.sqrt(cp.maximum(self.squared_sum/self.n_points - cp.sum(self.centroid**2, axis=0), 0))

class CFNode:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.entries = []
        self.children = []
        self.parent = None

    def insert_entry(self, cf, child=None):
        self.entries.append(cf)
        if not self.is_leaf:
            self.children.append(child)
            child.parent = self


class CFTree:
    def __init__(self, threshold, branching_factor):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.root = CFNode(is_leaf=True)
        self.dim = None

    def insert(self, X_batch):
        X_batch = cp.asarray(X_batch, dtype=cp.float32)
        if self.dim is None:
            self.dim = X_batch.shape[1]
            for cf in self.root.entries:
                cf.dim = self.dim
                cf.linear_sum = cp.zeros(self.dim, dtype=cp.float32)

        for point in X_batch:
            leaf = self._find_leaf(point)
            self._insert_into_leaf(leaf, point)

    def _find_leaf(self, point):
        node = self.root
        while not node.is_leaf:
            centroids = cp.stack([cf.centroid for cf in node.entries])
            dists = cp.sum((centroids - point)**2, axis=1)
            node = node.children[cp.argmin(dists)]
        return node

    def _insert_into_leaf(self, leaf, point):
        if not leaf.entries:
            leaf.entries.append(ClusteringFeature(point))
            return

        centroids = cp.stack([cf.centroid for cf in leaf.entries])
        dists = cp.sum((centroids - point)**2, axis=1)
        closest_idx = cp.argmin(dists).item()
        closest_cf = leaf.entries[closest_idx]

        temp_cf = ClusteringFeature()
        temp_cf.merge(closest_cf)
        temp_cf.add_point(point)
        
        if temp_cf.radius <= self.threshold:
            closest_cf.merge(ClusteringFeature(point))
            self._propagate_update(leaf)
        else:
            leaf.entries.append(ClusteringFeature(point))
            if len(leaf.entries) > self.branching_factor:
                self._split(leaf)
            self._propagate_update(leaf)

    def _split(self, node):
        entries = node.entries
        centroids = cp.stack([cf.centroid for cf in entries])

        # Векторизованный расчет попарных расстояний
        dist_matrix = cp.sum(centroids**2, axis=1)[:, None] + \
                      cp.sum(centroids**2, axis=1) - \
                      2 * centroids @ centroids.T
        cp.fill_diagonal(dist_matrix, -cp.inf)
        idx = cp.unravel_index(cp.argmax(dist_matrix), dist_matrix.shape)
        
        # Разделение на две группы
        mask = cp.sum((centroids - centroids[idx[0]])**2, axis=1) < \
               cp.sum((centroids - centroids[idx[1]])**2, axis=1)
        
        new_node = CFNode(is_leaf=node.is_leaf)
        new_entries = [entries[i] for i in cp.where(mask)[0]]
        remaining_entries = [entries[i] for i in cp.where(~mask)[0]]

        node.entries = new_entries
        new_node.entries = remaining_entries

        if node.parent is None:
            new_root = CFNode(is_leaf=False)
            new_root.insert_entry(ClusteringFeature(), node)
            self.root = new_root

        parent = node.parent
        new_parent_cf = ClusteringFeature()
        for cf in new_node.entries:
            new_parent_cf.merge(cf)
        parent.insert_entry(new_parent_cf, new_node)

        if len(parent.entries) > self.branching_factor:
            self._split(parent)

    def _propagate_update(self, node):
        while node.parent:
            parent = node.parent
            idx = parent.children.index(node)
            new_cf = ClusteringFeature()
            for cf in node.entries:
                new_cf.merge(cf)
            parent.entries[idx] = new_cf
            node = parent

    def get_subclusters(self):
        subclusters = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                subclusters.extend(node.entries)
            else:
                stack.extend(node.children)
        return subclusters


class GPUAgglomerativeClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):
        X = cp.asarray(X)
        n = X.shape[0]
        self._labels = cp.arange(n)
        dist_matrix = self._pairwise_distance(X)
        
        for _ in range(n - self.n_clusters):
            i, j = self._find_closest_pair(dist_matrix)
            self._merge_clusters(i, j, dist_matrix)
        
        self.labels_ = cp.asnumpy(self._labels)
        return self

    def _pairwise_distance(self, X):
        # Векторизованный расчет матрицы расстояний
        sq_norm = cp.sum(X**2, axis=1)
        return sq_norm[:, None] + sq_norm - 2 * X @ X.T

    def _find_closest_pair(self, dist_matrix):
        cp.fill_diagonal(dist_matrix, cp.inf)
        return cp.unravel_index(cp.argmin(dist_matrix), dist_matrix.shape)

    def _merge_clusters(self, i, j, dist_matrix):
        mask = self._labels == j
        self._labels[mask] = i
        
        # Обновление матрицы расстояний методом полной связи
        new_dists = cp.minimum(dist_matrix[i], dist_matrix[j])
        dist_matrix[i] = new_dists
        dist_matrix[:, i] = new_dists
        dist_matrix[i, i] = cp.inf

        # Удаление старого кластера
        dist_matrix[j] = cp.inf
        dist_matrix[:, j] = cp.inf


class BirchModel(ClusteringModel):
    def __init__(self, labels_, centroids_):
        super().__init__(labels_)
        self.centroids_ = centroids_

    def predict(self, X):
        """
        Предсказывает метки кластеров для новых данных
        """
        if self.subcluster_centers_ is None or self.labels_ is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите fit()")
            
        X = cp.asarray(X, dtype=cp.float32)
        
        # 1. Находим ближайшие подкластеры
        subcluster_idx = self._find_nearest_subcluster(X)
        
        # 2. Преобразуем индексы подкластеров в метки кластеров
        cluster_labels = self.labels_[subcluster_idx]
        
        return cluster_labels.get()  # Возвращаем numpy массив


class Birch(ClusteringAlgo):
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        super().__init__()
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.subcluster_centers_ = None
        self.labels_ = None

    def fit(self, X, batch_size=10000):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        self.tree = CFTree(self.threshold, self.branching_factor)
        
        # Пакетная обработка данных
        for i in range(0, len(X_gpu), batch_size):
            batch = X_gpu[i:i+batch_size]
            self.tree.insert(batch)
        
        subclusters = self.tree.get_subclusters()
        self.subcluster_centers_ = cp.stack([cf.centroid for cf in subclusters])
        
        # GPU-реализация агломеративной кластеризации
        clusterer = GPUAgglomerativeClustering(n_clusters=self.n_clusters)
        clusterer.fit(self.subcluster_centers_)
        self.labels_ = clusterer.labels_
        self.labels_ = cp.array(self.labels_, dtype=cp.int32)
        
        labels = self.predict(X)
        labels = cp.array(labels, dtype=cp.int32)
        
        print(labels)
        
        return BirchModel(
            labels_=labels,
            centroids_=self.subcluster_centers_
        )
        
    def _find_nearest_subcluster(self, X):
        """
        Находит ближайший подкластер для каждой точки
        """
        # Векторизованный расчет расстояний
        X_norm = cp.sum(X**2, axis=1, keepdims=True)
        C_norm = cp.sum(self.subcluster_centers_**2, axis=1)
        distances = X_norm + C_norm - 2 * X @ self.subcluster_centers_.T
        
        return cp.argmin(distances, axis=1)

    def _calculate_cluster_centers(self):
        """
        Вычисляет центры финальных кластеров
        """
        unique_labels = cp.unique(self.labels_)
        centers = cp.zeros((len(unique_labels), self.subcluster_centers_.shape[1]))
        
        for label in unique_labels:
            mask = self.labels_ == label
            cluster_centers = self.subcluster_centers_[mask]
            centers[label] = cluster_centers.mean(axis=0)
            
        return centers.get()
        
    def predict(self, X):
        """
        Предсказывает метки кластеров для новых данных
        """
        if self.subcluster_centers_ is None or self.labels_ is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите fit()")
            
        X = cp.asarray(X, dtype=cp.float32)
        
        # 1. Находим ближайшие подкластеры
        subcluster_idx = self._find_nearest_subcluster(X)
        
        # 2. Преобразуем индексы подкластеров в метки кластеров
        cluster_labels = self.labels_[subcluster_idx]
        
        return cluster_labels.get()  # Возвращаем numpy массив


class BirchConfig(AlgoConfig):
    def __init__(
            self, *,
            threshold=(0.1, 0.9),
            branching_factor=(10, 80),
            n_clusters=(2, 15),
    ):
        super().__init__(
            Birch,
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters,
        )

import cupy as cp
import numpy as np
import pylibraft.config
from sklearn.cluster import AgglomerativeClustering

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class BirchModel(ClusteringModel):
    def __init__(self, labels_, subcluster_labels, tree):
        super().__init__(labels_)
        self.subcluster_labels = subcluster_labels
        self.tree = tree

    def predict(self, X: DataFrameType) -> LabelsType:
        subclusters = [cf.centroid().get() for cf in self.tree.root.cfs]
        labels = []
        for point in X:
            closest = np.argmin([np.linalg.norm(point.get() - sc) for sc in subclusters])
            labels.append(self.subcluster_labels[closest])
        return cp.array(labels, dtype=cp.int32)


class ClusteringFeatureGPU:
    def __init__(self, point):
        self.n = 1
        self.LS = cp.array(point, dtype=cp.float64)
        self.SS = cp.square(point)
    
    def add_point(self, point):
        self.n += 1
        self.LS += point
        self.SS += cp.square(point)
    
    def merge(self, other):
        self.n += other.n
        self.LS += other.LS
        self.SS += other.SS
    
    def centroid(self):
        return self.LS / self.n
    
    def radius(self):
        return cp.sqrt(cp.sum(self.SS / self.n - cp.square(self.centroid())))


class CFNodeGPU:
    def __init__(self, threshold, branching_factor, is_leaf=True):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.cfs = []
        self.children = []
        self.next = None

    def split(self):
        # Упрощенная реализация split для связывания листьев
        if self.is_leaf:
            new_node = CFNodeGPU(self.threshold, self.branching_factor, is_leaf=True)
            half = len(self.cfs) // 2
            new_node.cfs = self.cfs[half:]
            self.cfs = self.cfs[:half]
            new_node.next = self.next
            self.next = new_node

    def insert(self, cf, data):
        if not self.is_leaf:
            closest = self.find_closest_child(data)
            self.children[closest].insert(cf, data)
        else:
            if self.cfs:
                distances = [cp.linalg.norm(cf.centroid() - existing_cf.centroid()) for existing_cf in self.cfs]
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
        centroids = [cf.centroid().get() for cf in self.cfs]  # Move to CPU for compatibility
        distances = [cp.linalg.norm(data - centroid) for centroid in centroids]
        return cp.argmin(cp.array(distances)).item()


class CFTreeGPU:
    def __init__(self, threshold, branching_factor):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.root = CFNodeGPU(threshold, branching_factor, is_leaf=True)

    def insert(self, data):
        cf = ClusteringFeatureGPU(data)
        self.root.insert(cf, data)


class Birch(ClusteringAlgo):
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.subcluster_labels = None
        self.labels_ = None

    def fit(self, X):
        X_gpu = cp.array(X)
        self.tree = CFTreeGPU(self.threshold, self.branching_factor)
        for point in X_gpu:
            self.tree.insert(point)

        subclusters = []
        node = self.tree.root
        while node:
            subclusters.extend([cf.centroid().get() for cf in node.cfs])
            node = node.next

        # Проверка количества подкластеров
        if len(subclusters) < 2:
            self.subcluster_labels = np.zeros(len(subclusters), dtype=int)
            self.labels_ = cp.zeros(len(X), dtype=cp.int32)
            return BirchModel(labels_=self.labels_, subcluster_labels=self.subcluster_labels, tree=self.tree)

        # Адаптация числа кластеров
        n_clusters = min(self.n_clusters, len(subclusters))
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clustering.fit(subclusters)
        self.subcluster_labels = clustering.labels_

        # Назначение меток
        self.labels_ = []
        subcluster_centers = np.array([cf.centroid().get() for cf in self.tree.root.cfs])
        for point in X_gpu:
            point_cpu = point.get()
            closest = np.argmin(np.linalg.norm(subcluster_centers - point_cpu, axis=1))
            self.labels_.append(self.subcluster_labels[closest])
        self.labels_ = cp.array(self.labels_, dtype=cp.int32)

        return BirchModel(labels_=self.labels_, subcluster_labels=self.subcluster_labels, tree=self.tree)

    def predict(self, X):
        X_gpu = cp.array(X)
        subclusters = np.array([cf.centroid().get() for cf in self.tree.root.cfs])
        labels = []
        for point in X_gpu:
            point_cpu = point.get()
            closest = np.argmin(np.linalg.norm(subclusters - point_cpu, axis=1))
            labels.append(self.subcluster_labels[closest])
        return cp.array(labels)


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

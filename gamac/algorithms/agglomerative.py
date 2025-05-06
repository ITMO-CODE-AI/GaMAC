import cupy as cp
import pylibraft.config
from typing import Optional

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class AgglomerativeClusteringModel(ClusteringModel):
    def __init__(self, labels_: LabelsType):
        super().__init__(labels_)
        self.labels_ = labels_

    def predict(self, df: DataFrameType) -> LabelsType:
        return self.labels_


class AgglomerativeClustering(ClusteringAlgo):
    def __init__(self, n_clusters=2, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X):
        # X должен быть cupy-массивом
        n_samples = X.shape[0]
        clusters = {i: [i] for i in range(n_samples)}
        distances = self._compute_distances(X)

        cluster_distances = {}
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                cluster_distances[(i, j)] = distances[i, j].item()  # .item() для извлечения числа из cupy.float

        while len(clusters) > self.n_clusters:
            c1, c2 = min(cluster_distances, key=cluster_distances.get)
            clusters[c1].extend(clusters[c2])
            del clusters[c2]

            keys_to_remove = [key for key in cluster_distances if c2 in key]
            for key in keys_to_remove:
                del cluster_distances[key]

            for c in clusters:
                if c != c1:
                    dist = self._linkage_distance(X, clusters[c1], clusters[c])
                    key = (min(c1, c), max(c1, c))
                    cluster_distances[key] = dist

        self.labels_ = cp.empty(n_samples, dtype=cp.int32)
        for cluster_id, points in enumerate(clusters.values()):
            for point in points:
                self.labels_[point] = cluster_id
        return AgglomerativeClusteringModel(labels_=self.labels_)

    def predict(self, X):
        if self.labels_ is None:
            raise Exception("Model has not been fitted yet.")
        return self.labels_

    def _compute_distances(self, X):
        sum_sq = cp.sum(X ** 2, axis=1)
        distances = cp.sqrt(cp.maximum(sum_sq[:, None] + sum_sq[None, :] - 2 * cp.dot(X, X.T), 0))
        return distances

    def _linkage_distance(self, X, cluster1, cluster2):
        if self.linkage == 'ward':
            points1 = X[cluster1]
            points2 = X[cluster2]
            centroid1 = cp.mean(points1, axis=0)
            centroid2 = cp.mean(points2, axis=0)
            dist = cp.linalg.norm(centroid1 - centroid2)
            return dist.item()
        elif self.linkage == 'single':
            min_dist = cp.inf
            for i in cluster1:
                for j in cluster2:
                    d = cp.linalg.norm(X[i] - X[j])
                    if d < min_dist:
                        min_dist = d
            return min_dist.item()
        elif self.linkage == 'complete':
            max_dist = 0
            for i in cluster1:
                for j in cluster2:
                    d = cp.linalg.norm(X[i] - X[j])
                    if d > max_dist:
                        max_dist = d
            return max_dist.item()
        else:
            raise ValueError(f"Unknown linkage type: {self.linkage}")


class AgglomerativeClusteringConfig(AlgoConfig):
    def __init__(
            self, *,
            n_clusters=(2, 15),
            linkage=frozenset(['single', 'complete', 'average'])
    ):
        super().__init__(
            AgglomerativeClustering,
            n_clusters=n_clusters,
            linkage=linkage,
        )

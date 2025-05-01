import cupy as cp
import pylibraft.config
from typing import Optional

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class AgglomerativeClusteringModel(ClusteringModel):
    def __init__(self, labels_: LabelsType):
        super().__init__(labels_)

    def predict(self, df: DataFrameType) -> LabelsType:
        raise NotImplementedError
        # distances = cp.linalg.norm(df[:, cp.newaxis] - self.X_train, axis=2)
        # nearest_indices = cp.argmin(distances, axis=1)
        # return self.labels_[nearest_indices]


class AgglomerativeClustering(ClusteringAlgo):
    def __init__(
            self,
            n_clusters: int = 2,
            linkage: str = 'single'
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage

    def _initialize_proximity(self, df: DataFrameType):
        """Инициализация матрицы попарных расстояний"""
        diff = df[:, cp.newaxis] - df
        return cp.sqrt(cp.sum(diff ** 2, axis=2))

    def _update_proximity(
            self, i: int, j: int, proximity: cp.ndarray, cluster_sizes: cp.ndarray
    ) -> None:
        """Обновление матрицы расстояний по формуле Ланса-Уильямса"""
        mask = cp.ones(proximity.shape[0], dtype=bool)
        mask[[i, j]] = False

        if self.linkage == 'single':
            proximity[i, mask] = cp.minimum(proximity[i, mask], proximity[j, mask])
        elif self.linkage == 'complete':
            proximity[i, mask] = cp.maximum(proximity[i, mask], proximity[j, mask])
        elif self.linkage == 'average':
            size_i = cluster_sizes[i]
            size_j = cluster_sizes[j]
            proximity[i, mask] = (size_i * proximity[i, mask] + 
                                  size_j * proximity[j, mask]) / (size_i + size_j)
        
        proximity[mask, i] = proximity[i, mask]
        proximity[i, i] = cp.inf

    def fit(self, df: DataFrameType) -> AgglomerativeClusteringModel:
        N, D = df.shape

        # Инициализация структур данных
        proximity = self._initialize_proximity(df)
        cluster_sizes = cp.ones(N, dtype=cp.uint32)
        children = []

        active = cp.ones(N, dtype=bool)
        clusters = [[i] for i in range(N)]

        for _ in range(N - self.n_clusters):
            # Маскировка неактивных кластеров
            masked_proximity = proximity.copy()
            masked_proximity[~active, :] = cp.inf
            masked_proximity[:, ~active] = cp.inf
            cp.fill_diagonal(masked_proximity, cp.inf)

            # Нахождение минимальной пары
            min_idx = cp.argmin(masked_proximity)
            i, j = cp.unravel_index(min_idx, masked_proximity.shape)

            # Обновление структур данных
            children.append((int(i), int(j)))
            self._update_proximity(i, j, proximity, cluster_sizes)
            cluster_sizes[i] += cluster_sizes[j]
            active[j] = False

        # Назначение меток кластеров
        labels = cp.full(N, -1, dtype=cp.int32)
        label = 0
        for idx in range(N):
            if active[idx]:
                labels[clusters[idx]] = label
                label += 1

        distances = cp.linalg.norm(df[:, cp.newaxis] - df, axis=2)
        nearest_indices = cp.argmin(distances, axis=1)
        labels_ = labels[nearest_indices]

        return AgglomerativeClusteringModel(labels_=labels_)


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

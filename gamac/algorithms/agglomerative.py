import cupy as cp
import pylibraft.config
from typing import Optional

pylibraft.config.set_output_as("cupy")


class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 2, linkage: str = 'single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_: Optional[cp.ndarray] = None
        self.children_ = []
        self._proximity: Optional[cp.ndarray] = None
        self._cluster_sizes: Optional[cp.ndarray] = None

    def _initialize_proximity(self, X: cp.ndarray) -> cp.ndarray:
        """Инициализация матрицы попарных расстояний"""
        diff = X[:, cp.newaxis] - X
        return cp.sqrt(cp.sum(diff**2, axis=2))

    def _update_proximity(self, i: int, j: int, proximity: cp.ndarray,
                          cluster_sizes: cp.ndarray) -> None:
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

    def fit(self, X: cp.ndarray) -> None:
        self.X_train = cp.asarray(X)
        n_samples = self.X_train.shape[0]
        
        # Инициализация структур данных
        self._proximity = self._initialize_proximity(self.X_train)
        self._cluster_sizes = cp.ones(n_samples, dtype=cp.int32)
        active = cp.ones(n_samples, dtype=bool)
        clusters = [[i] for i in range(n_samples)]

        for _ in range(n_samples - self.n_clusters):
            # Маскировка неактивных кластеров
            masked_proximity = self._proximity.copy()
            masked_proximity[~active, :] = cp.inf
            masked_proximity[:, ~active] = cp.inf
            cp.fill_diagonal(masked_proximity, cp.inf)

            # Нахождение минимальной пары
            min_idx = cp.argmin(masked_proximity)
            i, j = cp.unravel_index(min_idx, masked_proximity.shape)

            # Обновление структур данных
            self.children_.append((int(i), int(j)))
            self._update_proximity(i, j, self._proximity, self._cluster_sizes)
            self._cluster_sizes[i] += self._cluster_sizes[j]
            active[j] = False

        # Назначение меток кластеров
        self.labels_ = cp.full(n_samples, -1, dtype=cp.int32)
        label = 0
        for idx in range(n_samples):
            if active[idx]:
                self.labels_[clusters[idx]] = label
                label += 1

    def predict(self, X_new: cp.ndarray) -> cp.ndarray:
        """Предсказание меток для новых данных"""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
            
        X_new = cp.asarray(X_new)
        distances = cp.linalg.norm(X_new[:, cp.newaxis] - self.X_train, axis=2)
        nearest_indices = cp.argmin(distances, axis=1)
        return self.labels_[nearest_indices]

    @property
    def labels(self) -> cp.ndarray:
        return cp.asnumpy(self.labels_) if self.labels_ is not None else None

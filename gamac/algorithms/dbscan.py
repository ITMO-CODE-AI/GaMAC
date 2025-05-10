import cupy as cp
import pylibraft.config

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class DBSCANModel(ClusteringModel):
    def __init__(self, labels_, X_, eps):
        super().__init__(labels_)
        self.X_ = X_
        self.eps_sq = eps ** 2

    def predict(self, df: DataFrameType) -> LabelsType:
        df = cp.asarray(df, dtype=cp.float32)
        X_train = self.X_

        # Батчевый расчет расстояний для больших данных
        batch_size = 4096
        predictions = cp.empty(len(df), dtype=cp.int32)

        for i in range(0, len(df), batch_size):
            batch = df[i:i + batch_size]

            # Векторизованный расчет квадратов расстояний
            X_new_norm = cp.sum(batch**2, axis=1)[:, cp.newaxis]
            X_train_norm = cp.sum(X_train**2, axis=1)
            dists_sq = X_new_norm + X_train_norm - 2 * batch @ X_train.T
            dists_sq = cp.maximum(dists_sq, 0)

            # Поиск ближайших соседей в eps-окрестности
            mask = dists_sq <= self.eps_sq
            has_neighbor = cp.any(mask, axis=1)

            # Для точек без соседей - ближайший сосед
            nearest = cp.argmin(dists_sq, axis=1)
            predictions[i:i + batch_size] = cp.where(
                has_neighbor,
                self.labels_[cp.argmax(mask, axis=1)],
                self.labels_[nearest]
            )

        return predictions


class DBSCAN(ClusteringAlgo):
    def __init__(self, eps=1.0, min_samples=5):
        super().__init__()
        self.eps = eps
        self.eps_sq = eps ** 2
        self.min_samples = min_samples
        self.X = None

    def _get_neighbors(self, idx):
        """Быстрый расчет соседей с использованием матричного умножения"""
        point = self.X[idx]
        diffs = self.X - point
        dists_sq = cp.sum(diffs**2, axis=1)
        return cp.where(dists_sq <= self.eps_sq)[0]

    def _process_core_point(self, idx, visited, is_core):
        """Обработка core-точки с пакетным поиском соседей"""
        if visited[idx] or not is_core[idx]:
            return None

        cluster = []
        queue = cp.array([idx], dtype=cp.int32)
        visited[idx] = True

        while queue.size > 0:
            current = queue[0]
            queue = queue[1:]
            cluster.append(int(current))

            neighbors = self._get_neighbors(current)
            mask = ~visited[neighbors]
            new_points = neighbors[mask]

            # Пометка core-точек заранее
            core_mask = is_core[new_points]
            queue = cp.concatenate((queue, new_points[core_mask]))
            visited[new_points] = True

            # Добавление граничных точек
            cluster.extend(new_points[~core_mask].get().tolist())

        return cluster

    def fit(self, X):
        self.X = cp.asarray(X, dtype=cp.float32)
        n = len(self.X)
        visited = cp.zeros(n, dtype=bool)
        is_core = cp.zeros(n, dtype=bool)
        labels_ = cp.full(n, -1, dtype=cp.int32)

        # Предварительный расчет core-точек
        for i in range(n):
            if not visited[i]:
                neighbors = self._get_neighbors(i)
                if len(neighbors) >= self.min_samples:
                    is_core[i] = True

        # Основная кластеризация
        cluster_id = 0
        for i in range(n):
            if not visited[i] and is_core[i]:
                cluster = self._process_core_point(i, visited, is_core)
                if cluster:
                    labels_[cluster] = cluster_id
                    cluster_id += 1

        return DBSCANModel(labels_, self.X, self.eps)


class DBSCANConfig(AlgoConfig):
    def __init__(
        self,
        eps=(0.01, 1.5),
        min_samples=(5, 15)
    ):
        super().__init__(
            DBSCAN,
            eps=eps,
            min_samples=min_samples
        )
import cupy as cp
import pylibraft.config

pylibraft.config.set_output_as("cupy")

class DBSCAN:
    def __init__(self, eps=1.0, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.X = None
        self.labels_ = None
        self.clusters = []
        self.visited_samples = None
        self.neighbors = None

    def _compute_distance_matrix(self, X):
        diff = X[:, cp.newaxis, :] - X[cp.newaxis, :, :]
        return cp.linalg.norm(diff, axis=2)

    def _get_neighbors(self, sample_i):
        # Возвращаем индексы точек, расстояние до которых <= eps
        distances = self.distance_matrix[sample_i]
        neighbors = cp.where(distances <= self.eps)[0]
        return neighbors

    def _expand_cluster(self, sample_i, neighbors):
        cluster = [sample_i]
        i = 0
        while i < len(neighbors):
            neighbor_i = neighbors[i]
            if not self.visited_samples[neighbor_i]:
                self.visited_samples[neighbor_i] = True
                neighbor_neighbors = self._get_neighbors(neighbor_i)
                if len(neighbor_neighbors) >= self.min_samples:
                    # Добавляем новых соседей в список для проверки
                    neighbors = cp.unique(cp.concatenate((neighbors, neighbor_neighbors)))
            # Добавляем точку в кластер, если она еще не в нем
            if neighbor_i not in cluster:
                cluster.append(neighbor_i)
            i += 1
        return cluster

    def fit(self, X):
        self.X = cp.asarray(X, dtype=cp.float32)
        n_samples = self.X.shape[0]
        self.distance_matrix = self._compute_distance_matrix(self.X)
        self.visited_samples = cp.zeros(n_samples, dtype=bool)
        self.clusters = []

        for sample_i in range(n_samples):
            if self.visited_samples[sample_i]:
                continue
            neighbors = self._get_neighbors(sample_i)
            if len(neighbors) < self.min_samples:
                self.visited_samples[sample_i] = True
                continue
            self.visited_samples[sample_i] = True
            cluster = self._expand_cluster(sample_i, neighbors)
            self.clusters.append(cluster)

        self.labels_ = self._get_cluster_labels()

    def _get_cluster_labels(self):
        n_samples = self.X.shape[0]
        labels = cp.full(n_samples, fill_value=-1, dtype=cp.int32)  # -1 для шумов
        for cluster_id, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_id
        return labels

    def predict(self, X_new):
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit() before predict().")
        X_new = cp.asarray(X_new, dtype=cp.float32)
        # Для каждого нового образца находим ближайшую точку из обучающего набора
        diff = X_new[:, cp.newaxis, :] - self.X[cp.newaxis, :, :]
        distances = cp.linalg.norm(diff, axis=2)
        nearest_idx = cp.argmin(distances, axis=1)
        # Возвращаем метки ближайших точек
        return self.labels_[nearest_idx].get()

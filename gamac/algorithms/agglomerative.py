import cupy as cp
import pylibraft.config

pylibraft.config.set_output_as("cupy")


class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distances = None
        self.labels_ = None
        self.children = []
        self.X_train = None

    def _calculate_linkage(self, cluster1, cluster2):
        """Исправление ошибки: работа с cupy массивами"""
        pairs = cp.array([(i, j) for i in cluster1 for j in cluster2])
        i, j = pairs[:, 0], pairs[:, 1]
        dist_matrix = self.distances[i, j]

        if self.linkage == 'single':
            return cp.min(dist_matrix)
        elif self.linkage == 'complete':
            return cp.max(dist_matrix)
        elif self.linkage == 'average':
            return cp.mean(dist_matrix)
        else:
            raise ValueError(f"Unknown linkage method: {self.linkage}")

    def _calculate_distances(self, X):
        """Векторизованный расчет расстояний"""
        diff = X[:, cp.newaxis] - X
        return cp.sqrt(cp.sum(diff**2, axis=2))

    def fit(self, X):
        X = cp.asarray(X)
        self.X_train = X.copy()

        n_samples = X.shape[0]
        self.distances = self._calculate_distances(X)
        clusters = [[i] for i in range(n_samples)]

        while len(clusters) > self.n_clusters:
            min_dist = cp.inf
            best_pair = (0, 1)

            # Поиск пар кластеров
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    dist = self._calculate_linkage(clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (i, j)

            # Объединение кластеров
            merged = clusters[best_pair[0]] + clusters[best_pair[1]]
            clusters = [c for idx, c in enumerate(clusters)
                        if idx not in best_pair] + [merged]
            self.children.append(best_pair)

        # Назначение меток
        self.labels_ = cp.zeros(n_samples, dtype=cp.int32)
        for idx, cluster in enumerate(clusters):
            self.labels_[cluster] = idx
        self.labels_ = cp.asnumpy(self.labels_)

    def predict(self, X_new):
        """Оптимизированный метод предсказания"""
        if self.X_train is None:
            raise ValueError("Model not fitted yet.")
   
        X_new = cp.asarray(X_new)

        # Векторизованный расчет расстояний
        distances = cp.linalg.norm(
            X_new[:, cp.newaxis] - self.X_train,
            axis=2
        )

        # Находим ближайшие точки
        nearest_indices = cp.argmin(distances, axis=1)
        return self.labels_[cp.asnumpy(nearest_indices)]

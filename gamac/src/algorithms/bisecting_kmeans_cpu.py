import numpy as np
from sklearn.metrics import silhouette_score


class BisectingKMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels_ = None

    def _kmeans(self, X, k):
        """Базовый K-Means для деления кластера на два"""
        centers = np.random.rand(k, X.shape[1])

        for _ in range(self.max_iter):
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])

            if np.all(centers == new_centers):
                break
            centers = new_centers

        return labels, centers

    def _sse(self, X, labels):
        """Расчет SSE для каждого кластера"""
        sse = []
        for label in np.unique(labels):
            cluster = X[labels == label]
            center = cluster.mean(axis=0)
            sse.append(np.sum(np.linalg.norm(cluster - center, axis=1)**2))
        return sse

    def fit(self, X):
        clusters = [np.arange(X.shape[0])]

        while len(clusters) < self.n_clusters:
            # Выбор кластера с наибольшим SSE
            sse_values = [self._sse(X[cluster], self._kmeans(X[cluster], 2)[0]) for cluster in clusters]
            max_sse_idx = np.argmax(sse_values)

            # Обновление списка кластеров
            cluster_to_split = clusters.pop(max_sse_idx)
            labels, _ = self._kmeans(X[cluster_to_split], 2)  # Не забудьте получить метки для этого кластера
            clusters.extend([cluster_to_split[labels == i] for i in range(2)])

        # Присвоение меток кластеров
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        for i, cluster in enumerate(clusters):
            self.labels_[cluster] = i

        return self

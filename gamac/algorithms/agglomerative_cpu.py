import numpy as np
from scipy.spatial.distance import pdist, squareform


class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distances = None
        self.labels_ = None
        self.children = []

    def _calculate_linkage(self, cluster1, cluster2):
        """Вычисление расстояния между кластерами"""
        pairs = [(i, j) for i in cluster1 for j in cluster2]

        if self.linkage == 'single':
            return np.min([self.distances[i, j] for i, j in pairs])
        elif self.linkage == 'complete':
            return np.max([self.distances[i, j] for i, j in pairs])
        elif self.linkage == 'average':
            return np.mean([self.distances[i, j] for i, j in pairs])
        else:
            raise ValueError("Unknown linkage method")

    def fit(self, X):
        n_samples = X.shape[0]
        self.distances = squareform(pdist(X))  # Матрица попарных расстояний
        clusters = [[i] for i in range(n_samples)]  # Начальные кластеры

        while len(clusters) > self.n_clusters:
            # Поиск ближайших кластеров
            min_dist = np.inf
            best_pair = (0, 1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._calculate_linkage(clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (i, j)

            # Объединение кластеров
            merged = clusters[best_pair[0]] + clusters[best_pair[1]]
            clusters = [c for idx, c in enumerate(clusters) 
                        if idx not in best_pair] + [merged]
            self.children.append(best_pair)

        # Создание меток кластеров
        self.labels_ = np.zeros(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                self.labels_[point] = idx
     
        return self

import cupy as cp


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
            return cp.min([self.distances[i, j] for i, j in pairs])
        elif self.linkage == 'complete':
            return cp.max([self.distances[i, j] for i, j in pairs])
        elif self.linkage == 'average':
            return cp.mean([self.distances[i, j] for i, j in pairs])
        else:
            raise ValueError("Unknown linkage method")

    def _calculate_distances(self, X):
        """Расчет матрицы попарных расстояний"""
        n_samples = X.shape[0]
        distances = cp.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = cp.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist  # Матрица расстояний симметрична

        return distances

    def fit(self, X):
        # Перенос данных на GPU
        X = cp.asarray(X)

        n_samples = X.shape[0]
        self.distances = self._calculate_distances(X)  # Матрица попарных расстояний
        clusters = [[i] for i in range(n_samples)]  # Начальные кластеры

        while len(clusters) > self.n_clusters:
            # Поиск ближайших кластеров
            min_dist = cp.inf
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
        self.labels_ = cp.zeros(n_samples, dtype=cp.int32)
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                self.labels_[point] = idx

        # Возвращение меток на CPU
        self.labels_ = cp.asnumpy(self.labels_)

        return self

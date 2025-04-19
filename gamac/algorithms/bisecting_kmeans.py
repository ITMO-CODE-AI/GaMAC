import cupy as cp


class BisectingKMeansCupy:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers_ = None
        self.labels_ = None

    def _kmeans(self, X, k):
        """Базовый K-Means для деления кластера на два с использованием CuPy"""
        # Инициализация случайных центров
        centers = cp.random.rand(k, X.shape[1])

        for _ in range(self.max_iter):
            # Вычисление расстояний и меток
            distances = cp.linalg.norm(X[:, cp.newaxis] - centers, axis=2)
            labels = cp.argmin(distances, axis=1)

            # Расчет новых центров
            new_centers = cp.zeros_like(centers)
            for i in range(k):
                mask = (labels == i)
                if cp.any(mask):
                    new_centers[i] = X[mask].mean(axis=0)
                else:
                    new_centers[i] = centers[i]  # Если кластер пуст, оставляем прежний центр

            if cp.allclose(centers, new_centers):
                break
            centers = new_centers

        return labels, centers

    def _sse(self, X, labels):
        """Расчет общей SSE для всех кластеров с использованием CuPy"""
        total_sse = 0
        unique_labels = cp.unique(labels)
        for label in unique_labels:
            mask = (labels == label)
            cluster = X[mask]
            center = cp.mean(cluster, axis=0)
            total_sse += cp.sum((cp.linalg.norm(cluster - center, axis=1))**2).item()
        return total_sse

    def fit(self, X):
        # Перемещаем данные на GPU
        X_gpu = cp.asarray(X)

        # Инициализация списка индексов кластеров
        clusters = [cp.arange(X_gpu.shape[0])]

        while len(clusters) < self.n_clusters:
            # Выбор кластера с наибольшим SSE
            sse_values = []
            kmeans_results = []
            for i, cluster in enumerate(clusters):
                cluster_data = X_gpu[cluster]
                # Сохраняем результат _kmeans для последующего использования
                labels, centers = self._kmeans(cluster_data, 2)
                kmeans_results.append((labels, centers))
                # Общая SSE после деления на подкластеры
                sse_values.append(self._sse(cluster_data, labels))

            max_sse_idx = int(cp.argmax(cp.array(sse_values)))

            # Обновление списка кластеров
            cluster_to_split = clusters.pop(max_sse_idx)
            labels = kmeans_results[max_sse_idx][0]  # Используем сохраненный результат _kmeans
            clusters.extend([cluster_to_split[labels == i] for i in range(2)])

        # Присвоение меток кластеров
        self.labels_ = cp.zeros(X_gpu.shape[0], dtype=int)
        for i, cluster in enumerate(clusters):
            self.labels_[cluster] = i

        # Вычисление центров кластеров
        self.centers_ = cp.zeros((self.n_clusters, X_gpu.shape[1]))
        for i in range(self.n_clusters):
            mask = (self.labels_ == i)
            if cp.any(mask):
                self.centers_[i] = X_gpu[mask].mean(axis=0)

        # Возвращаем метки и центры на CPU для совместимости с sklearn API
        self.labels_ = cp.asnumpy(self.labels_)
        self.centers_ = cp.asnumpy(self.centers_)

        return self

# v0.1
import cupy as cp
import pylibraft.config
from cupyx.scipy import sparse

pylibraft.config.set_output_as("cupy")


class BisectingKMeans:
    def __init__(self, n_clusters=2, max_iter=100, init='k-means++', tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.tol = tol
        self.centers_ = None
        self.labels_ = None
        self._init_kernels()

    def _init_kernels(self):
        # Ядро для вычисления меток кластеров
        self._labels_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void compute_labels(const double* X, const double* centers, 
                                int num_points, int num_centers, int dim, 
                                int* labels) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= num_points) return;

                double min_dist = 1e308;
                int min_label = -1;

                for (int c = 0; c < num_centers; ++c) {
                    double dist = 0.0;
                    for (int d = 0; d < dim; ++d) {
                        double diff = X[idx * dim + d] - centers[c * dim + d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_label = c;
                    }
                }
                labels[idx] = min_label;
            }
        ''', 'compute_labels')

        # Ядро для вычисления SSE
        self._sse_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void compute_sse(const double* X, const double* centers, 
                            const int* labels, double* sse, 
                            int num_points, int dim) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= num_points) return;

                int label = labels[idx];
                double sum = 0.0;
                for (int d = 0; d < dim; ++d) {
                    double diff = X[idx * dim + d] - centers[label * dim + d];
                    sum += diff * diff;
                }
                atomicAdd(sse, sum);
            }
        ''', 'compute_sse')

    def _kmeans_init(self, X, k):
        """Инициализация центров методом K-Means++"""
        n_samples, n_features = X.shape
        centers = cp.empty((k, n_features), dtype=X.dtype)

        # Первый центр выбирается случайно
        first_idx = cp.random.randint(n_samples)
        centers[0] = X[first_idx]

        for i in range(1, k):
            # Вычисление расстояний до ближайшего центра
            distances = cp.linalg.norm(X[:, None] - centers[:i], axis=2)
            min_dists = cp.min(distances, axis=1)
            # Вероятности пропорциональны квадрату расстояний
            probs = min_dists ** 2
            probs /= cp.sum(probs)
            # Выбор следующего центра
            next_idx = cp.where(cp.random.rand() < cp.cumsum(probs))[0][0]
            centers[i] = X[next_idx]
        return centers

    def _kmeans(self, X, k):
        if self.init == 'k-means++':
            centers = self._kmeans_init(X, k)
        else:
            idx = cp.random.choice(X.shape[0], k, replace=False)
            centers = X[idx]

        for _ in range(self.max_iter):
            # Вычисление меток через CUDA ядро
            labels = cp.empty(X.shape[0], dtype=cp.int32)
            block_size = 256
            grid_size = (X.shape[0] + block_size - 1) // block_size
            self._labels_kernel(
                (grid_size,), (block_size,),
                (X, centers, X.shape[0], k, X.shape[1], labels)
            )

            # Обновление центров
            new_centers = cp.zeros_like(centers)
            counts = cp.zeros(k, dtype=cp.int32)
            for i in range(k):
                mask = (labels == i)
                if cp.any(mask):
                    new_centers[i] = X[mask].mean(axis=0)
                else:
                    new_centers[i] = centers[i]

            # Проверка схождения
            if cp.linalg.norm(centers - new_centers) < self.tol:
                break
            centers = new_centers

        return labels, centers

    def _sse(self, X, labels):
        # Векторизованный расчет SSE через CUDA ядро
        sse = cp.zeros(1, dtype=X.dtype)
        if len(X) == 0:
            return 0.0
        centers = cp.stack([X[labels == i].mean(axis=0) for i in cp.unique(labels)])
        block_size = 256
        grid_size = (X.shape[0] + block_size - 1) // block_size
        self._sse_kernel(
            (grid_size,), (block_size,),
            (X, centers, labels, sse, X.shape[0], X.shape[1])
        )
        return sse.item()


    def fit(self, X):
        # Оптимизация передачи данных
        if not isinstance(X, cp.ndarray):
            X_gpu = cp.asarray(X)
        else:
            X_gpu = X

        clusters = [cp.arange(X_gpu.shape[0])]

        while len(clusters) < self.n_clusters:
            sse_values = []
            kmeans_results = []

            for cluster in clusters:
                cluster_data = X_gpu[cluster]
                if len(cluster_data) < 2:
                    sse_values.append(0.0)
                    kmeans_results.append(None)
                    continue

                labels, centers = self._kmeans(cluster_data, 2)
                kmeans_results.append((labels, centers))
                sse_values.append(self._sse(cluster_data, labels))

            # Выбор кластера для разделения
            max_sse_idx = cp.argmax(cp.array(sse_values)).item()
            if sse_values[max_sse_idx] <= 0:
                break  # Не осталось кластеров для разделения

            # Обновление кластеров
            cluster_to_split = clusters.pop(max_sse_idx)
            labels = kmeans_results[max_sse_idx][0]
            clusters.extend([
                cluster_to_split[labels == 0],
                cluster_to_split[labels == 1]
            ])

        # Формирование финальных меток
        self.labels_ = cp.zeros(X_gpu.shape[0], dtype=int)
        for i, cluster in enumerate(clusters):
            self.labels_[cluster] = i
 
        # Вычисление центров
        self.centers_ = cp.stack([
            X_gpu[self.labels_ == i].mean(axis=0)
            for i in range(self.n_clusters)
        ])

        # Конвертация результатов в CPU
        self.labels_ = cp.asnumpy(self.labels_)
        self.centers_ = cp.asnumpy(self.centers_)

    def predict(self, X):
        if self.centers_ is None:
            raise ValueError("Model not fitted yet.")
        X_gpu = cp.asarray(X)
        distances = cp.linalg.norm(X_gpu[:, None] - cp.asarray(self.centers_), axis=2)
        return cp.argmin(distances, axis=1).get()

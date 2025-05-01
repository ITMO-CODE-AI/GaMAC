# # v0.1
# import cupy as cp
# import pylibraft.config
# from cupyx.scipy import sparse
#
# from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
# from gamac.data.data_pipeline import DataFrameType, LabelsType
# from gamac.kernels import MIDDLEWARE, BATCH_SIZE
#
# pylibraft.config.set_output_as("cupy")
#
#
# class BisectingKMeansModel(ClusteringModel):
#     def __init__(self, centers, labels):
#         self.centers_ = centers
#         self.labels_ = labels
#
#     def predict(self, df: DataFrameType) -> LabelsType:
#         distances = cp.linalg.norm(df[:, None] - self.centers_, axis=2)
#         return cp.argmin(distances, axis=1, dtype=cp.int32)
#
#
# class BisectingKMeans(ClusteringAlgo):
#     def __init__(
#             self,
#             n_clusters=2,
#             max_iter=100,
#             init='k-means++',
#             tol=1e-4,
#     ):
#         super().__init__(
#             n_clusters=n_clusters,
#             max_iter=max_iter,
#             init=init,
#             tol=tol,
#         )
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#         self.init = init
#         self.tol = tol
#
#     def _kmeans_init(self, X, k):
#         """Инициализация центров методом K-Means++"""
#         n_samples, n_features = X.shape
#         centers = cp.empty((k, n_features), dtype=X.dtype)
#
#         # Первый центр выбирается случайно
#         first_idx = cp.random.randint(n_samples)
#         centers[0] = X[first_idx]
#
#         for i in range(1, k):
#             # Вычисление расстояний до ближайшего центра
#             distances = cp.linalg.norm(X[:, None] - centers[:i], axis=2)
#             min_dists = cp.min(distances, axis=1)
#             # Вероятности пропорциональны квадрату расстояний
#             probs = min_dists ** 2
#             probs /= cp.sum(probs)
#             # Выбор следующего центра
#             next_idx = cp.where(cp.random.rand() < cp.cumsum(probs))[0][0]
#             centers[i] = X[next_idx]
#         return centers
#
#     def _kmeans(self, X, k):
#         N, D = X.shape
#
#         if self.init == 'k-means++':
#             centers = self._kmeans_init(X, k)
#         else:
#             idx = cp.random.choice(N, k, replace=False)
#             centers = X[idx]
#
#         for _ in range(self.max_iter):
#             # Вычисление меток через CUDA ядро
#             labels = cp.empty(N, dtype=cp.int32)
#             MIDDLEWARE.kmeans_labels(
#                 X=X,
#                 centers=centers,
#                 N=N,
#                 K=k,
#                 D=D,
#                 labels=labels
#             ).invoke(
#                 grid=((N + BATCH_SIZE - 1) // BATCH_SIZE,),
#                 blocks=(BATCH_SIZE,)
#             )
#
#             # Обновление центров
#             new_centers = cp.zeros_like(centers)
#             for i in range(k):
#                 mask = (labels == i)
#                 if cp.any(mask):
#                     new_centers[i] = X[mask].mean(axis=0)
#                 else:
#                     new_centers[i] = centers[i]
#
#             # Проверка схождения
#             if cp.linalg.norm(centers - new_centers) < self.tol:
#                 break
#             centers = new_centers
#
#         return labels, centers
#
#     def _sse(self, X, labels):
#         # Векторизованный расчет SSE через CUDA ядро
#         sse = cp.zeros(1, dtype=cp.float32)
#         if len(X) == 0:
#             return 0.0
#         N, D = X.shape
#         centers = cp.stack([X[labels == i].mean(axis=0) for i in cp.unique(labels)])
#         MIDDLEWARE.kmeans_sse(
#             X=X,
#             centers=centers,
#             labels=labels,
#             sse=sse,
#             N=N,
#             D=D,
#         ).invoke(
#             grid=((N + BATCH_SIZE - 1) // BATCH_SIZE,),
#             blocks=(BATCH_SIZE,),
#         )
#         return sse.item()
#
#     def fit(self, df: DataFrameType) -> ClusteringModel:
#         N, D = df.shape
#         clusters = [cp.arange(N)]
#
#         while len(clusters) < self.n_clusters:
#             sse_values = []
#             kmeans_results = []
#
#             for cluster in clusters:
#                 cluster_data = df[cluster]
#                 if len(cluster_data) < 2:
#                     sse_values.append(0.0)
#                     kmeans_results.append(None)
#                     continue
#
#                 labels, centers = self._kmeans(cluster_data, 2)
#                 kmeans_results.append((labels, centers))
#                 sse_values.append(self._sse(cluster_data, labels))
#
#             # Выбор кластера для разделения
#             max_sse_idx = cp.argmax(cp.array(sse_values)).item()
#             if sse_values[max_sse_idx] <= 0:
#                 break  # Не осталось кластеров для разделения
#
#             # Обновление кластеров
#             cluster_to_split = clusters.pop(max_sse_idx)
#             labels = kmeans_results[max_sse_idx][0]
#             clusters.extend([
#                 cluster_to_split[labels == 0],
#                 cluster_to_split[labels == 1]
#             ])
#
#         # Формирование финальных меток
#         labels_ = cp.zeros(N, dtype=cp.int32)
#         for i, cluster in enumerate(clusters):
#             labels_[cluster] = i
#
#         # Вычисление центров
#         centers_ = cp.stack([
#             df[labels_ == i].mean(axis=0)
#             for i in range(self.n_clusters)
#         ])
#
#         return BisectingKMeansModel(
#             centers=centers_,
#             labels=labels_
#         )
#
# class BisectingKMeansConfig(AlgoConfig):
#     def __init__(
#             self, *,
#             n_clusters=(2, 15),
#             init=frozenset(['random', 'k-means++']),
#             max_iter=100,
#             tol=1e-4
#     ):
#         super().__init__(
#             BisectingKMeans,
#             n_clusters=n_clusters,
#             max_iter=max_iter,
#             init=init,
#             tol=tol
#         )

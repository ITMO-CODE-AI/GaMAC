import cupy as cp
import pylibraft.config

from gamac.algorithms.base import ClusteringAlgo, ClusteringModel, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class AffinityPropagationModel(ClusteringModel):
    def __init__(self, labels_, centroids_):
        super().__init__(labels_)
        self.centroids_ = centroids_

    def predict(self, df: DataFrameType) -> LabelsType:
        diff = df[:, cp.newaxis] - self.centroids_
        distances = cp.linalg.norm(diff, axis=2)
        return cp.argmin(distances, axis=1)


class AffinityPropagation(ClusteringAlgo):
    def __init__(
            self,
            preference=None,
            max_iter=100,
            convergence_iter=15,
            tol=1e-6
    ):
        super().__init__(
            preference=preference,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            tol=tol,
        )
        self.preference = preference
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.tol = tol

    def fit(self, df: DataFrameType):
        N, D = df.shape

        distance_matrix = self._compute_distance_matrix(df)

        if self.preference is None:
            preference = cp.median(distance_matrix).item()
        else:
            preference = self.preference

        R = cp.zeros((N, N), dtype=cp.float32)
        A = cp.zeros((N, N), dtype=cp.float32)
        cp.fill_diagonal(R, preference)

        for iteration in range(self.max_iter):
            R_old = R.copy()

            AS = A + distance_matrix
            max_AS = cp.max(AS, axis=1, keepdims=True)
            max_AS_idx = cp.argmax(AS, axis=1)

            for i in range(N):
                for k in range(N):
                    if k == max_AS_idx[i]:
                        AS_row = AS[i].copy()
                        AS_row[k] = -cp.inf
                        second_max = cp.max(AS_row)
                        R[i, k] = distance_matrix[i, k] - second_max
                    else:
                        R[i, k] = distance_matrix[i, k] - max_AS[i]

            for k in range(N):
                Rp = cp.maximum(R[k], 0)
                Rp[k] = R[k, k]
                sum_Rp = cp.sum(Rp) - Rp[k]
                for i in range(N):
                    if i == k:
                        A[k, i] = sum_Rp
                    else:
                        A[k, i] = min(0, R[k, k] + sum_Rp - max(0, R[k, i]))

            diff = cp.abs(R - R_old)
            if cp.all(diff < self.tol):
                break

        S = R + A
        labels_ = cp.argmax(S, axis=1)

        # Определяем индексы центров кластеров (экземпляров)
        exemplars_idx = cp.unique(labels_)
        centroids_ = df[exemplars_idx]

        return AffinityPropagationModel(
            labels_=labels_,
            centroids_=centroids_,
        )

    def _compute_distance_matrix(self, X):
        diff = X[:, cp.newaxis, :] - X[cp.newaxis, :, :]
        return cp.linalg.norm(diff, axis=2)


class AffinityPropagationConfig(AlgoConfig):
    def __init__(
            self, *,
            preference=(0.0, 1.0),
            max_iter=100,
            convergence_iter=15,
            tol=1e-6
    ):
        super().__init__(
            AffinityPropagation,
            preference=preference,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            tol=tol,
        )

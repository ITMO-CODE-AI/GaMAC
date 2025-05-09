import cupy as cp
import pylibraft.config

from gamac.algorithms.base import ClusteringAlgo, ClusteringModel, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class AffinityPropagationModel(ClusteringModel):
    def __init__(self, labels_, cluster_centers_):
        super().__init__(labels_)
        self.cluster_centers_ = cluster_centers_

    def predict(self, X: DataFrameType) -> LabelsType:
        pass


class AffinityPropagation(ClusteringAlgo):
    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, preference=None):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.labels_ = None
        self.cluster_centers_indices_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        X = cp.asarray(X)
        n_samples = X.shape[0]

        # Вычисление матрицы схожести (negative squared Euclidean distance)
        S = -cp.sum((X[:, cp.newaxis, :] - X[cp.newaxis, :, :]) ** 2, axis=2)

        # Установка предпочтений
        if self.preference is None:
            cp.fill_diagonal(S, cp.median(S))
        else:
            cp.fill_diagonal(S, self.preference)

        # Инициализация матриц
        R = cp.zeros((n_samples, n_samples))  # Responsibility
        A = cp.zeros((n_samples, n_samples))  # Availability

        # Главный цикл
        for _ in range(self.max_iter):
            # 1. Обновление Responsibility (R)
            AS = A + S
            max_indices = cp.argmax(AS, axis=1)
            max_values = AS[cp.arange(n_samples), max_indices]

            # Маска для замены максимальных значений
            mask = cp.zeros_like(AS, dtype=bool)
            mask[cp.arange(n_samples), max_indices] = True
            AS_masked = cp.where(mask, -cp.inf, AS)
            second_max_values = cp.max(AS_masked, axis=1)

            new_R = S - max_values[:, cp.newaxis]
            new_R[cp.arange(n_samples), max_indices] = S[cp.arange(n_samples), max_indices] - second_max_values
            R = self.damping * R + (1 - self.damping) * new_R

            # 2. Обновление Availability (A)
            Rp = cp.maximum(R, 0)
            cp.fill_diagonal(Rp, 0)
            sum_Rp = cp.sum(Rp, axis=0)

            # Обновление недиагональных элементов
            new_A = cp.minimum(0, R + sum_Rp[:, cp.newaxis] - Rp)
            cp.fill_diagonal(new_A, sum_Rp)
            A = self.damping * A + (1 - self.damping) * new_A

        # Определение экземпляров
        exemplar_criteria = (A + R).diagonal() > 0
        self.cluster_centers_indices_ = cp.where(exemplar_criteria)[0]

        # Назначение меток
        if len(self.cluster_centers_indices_) > 0:
            distances = -S[:, self.cluster_centers_indices_]
            self.labels_ = cp.argmax(distances, axis=1)
            self.cluster_centers_ = X[self.cluster_centers_indices_]
        else:
            self.labels_ = cp.zeros(n_samples, dtype=int)
            self.cluster_centers_ = cp.array([])

        return AffinityPropagationModel(labels_=self.labels_, cluster_centers_=self.cluster_centers_)


class AffinityPropagationConfig(AlgoConfig):
    def __init__(
            self, *,
            damping=(0.3, 0.5),
            max_iter=(50, 300),
            convergence_iter=(5, 10)
    ):
        super().__init__(
            AffinityPropagation,
            damping=damping, 
            max_iter=max_iter, 
            convergence_iter=convergence_iter
        )

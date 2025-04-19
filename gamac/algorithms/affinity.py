import cupy as cp
import pylibraft.config

pylibraft.config.set_output_as("cupy")


class AffinityPropagation:
    def __init__(self, preference=None, max_iter=100, convergence_iter=15, tol=1e-6):
        self.preference = preference
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None  # Добавим хранение центроидов

    def fit(self, X):
        X = cp.asarray(X, dtype=cp.float32)
        n_samples = X.shape[0]

        distance_matrix = self._compute_distance_matrix(X)

        if self.preference is None:
            self.preference = cp.median(distance_matrix)

        R = cp.zeros((n_samples, n_samples), dtype=cp.float32)
        A = cp.zeros((n_samples, n_samples), dtype=cp.float32)
        cp.fill_diagonal(R, self.preference)

        for iteration in range(self.max_iter):
            R_old = R.copy()

            AS = A + distance_matrix
            max_AS = cp.max(AS, axis=1, keepdims=True)
            max_AS_idx = cp.argmax(AS, axis=1)

            for i in range(n_samples):
                for k in range(n_samples):
                    if k == max_AS_idx[i]:
                        AS_row = AS[i].copy()
                        AS_row[k] = -cp.inf
                        second_max = cp.max(AS_row)
                        R[i, k] = distance_matrix[i, k] - second_max
                    else:
                        R[i, k] = distance_matrix[i, k] - max_AS[i]

            for k in range(n_samples):
                Rp = cp.maximum(R[k], 0)
                Rp[k] = R[k, k]
                sum_Rp = cp.sum(Rp) - Rp[k]
                for i in range(n_samples):
                    if i == k:
                        A[k, i] = sum_Rp
                    else:
                        A[k, i] = min(0, R[k, k] + sum_Rp - max(0, R[k, i]))

            diff = cp.abs(R - R_old)
            if cp.all(diff < self.tol):
                break

        S = R + A
        self.labels_ = cp.argmax(S, axis=1).get()

        # Определяем индексы центров кластеров (экземпляров)
        exemplars_idx = cp.unique(cp.asarray(self.labels_))
        self.cluster_centers_ = X[exemplars_idx]

    def predict(self, X_new):
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() before predict().")

        X_new = cp.asarray(X_new, dtype=cp.float32)
        distances = cp.linalg.norm(X_new[:, cp.newaxis] - self.cluster_centers_, axis=2)
        nearest = cp.argmin(distances, axis=1)
        return nearest.get()

    def _compute_distance_matrix(self, X):
        diff = X[:, cp.newaxis, :] - X[cp.newaxis, :, :]
        return cp.linalg.norm(diff, axis=2)

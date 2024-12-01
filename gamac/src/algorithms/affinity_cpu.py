import numpy as np


class AffinityPropagation:
    def __init__(self, preference=None, max_iter=100, convergence_iter=15):
        self.preference = preference
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter

    def fit(self, X):
        n_samples = X.shape[0]

        # Вычисляем матрицу расстояний
        distance_matrix = self._compute_distance_matrix(X)

        # Инициализация предпочтений
        if self.preference is None:
            self.preference = np.median(distance_matrix)

        # Инициализация responsibility и availability
        R = np.zeros((n_samples, n_samples))  # Responsibility
        A = np.zeros((n_samples, n_samples))  # Availability

        # Инициализация предпочтений
        np.fill_diagonal(R, self.preference)

        for iteration in range(self.max_iter):
            # Обновление responsibility
            R_old = R.copy()
            for i in range(n_samples):
                for k in range(n_samples):
                    R[i, k] = distance_matrix[i, k] - np.max(A[i] + distance_matrix[i])

            # Обновление availability
            for k in range(n_samples):
                for i in range(n_samples):
                    A[k, i] = min(0, R[k].sum() - R[k, i]) if i != k else R[k].sum()

            # Проверка на сходимость
            if np.all(np.abs(R - R_old) < 1e-6):
                break

        # Определение кластеров
        S = R + A  # Сообщения о принадлежности
        self.labels_ = np.argmax(S, axis=1)

    def _compute_distance_matrix(self, X):
        """Вычисляет матрицу расстояний."""
        return np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)

import numpy as np
from numba import cuda


@cuda.jit
def compute_responsibility_kernel(similarity, availability, responsibility):
    """Вычисляет матрицу ответственности."""
    i, j = cuda.grid(2)
    if i < responsibility.shape[0] and j < responsibility.shape[1]:
        max_avail = -np.inf
        for k in range(responsibility.shape[1]):
            if k != j:
                max_avail = max(max_avail, availability[i, k])
        responsibility[i, j] = similarity[i, j] - max_avail


@cuda.jit
def compute_availability_kernel(responsibility, availability):
    """Вычисляет матрицу доступности."""
    i, j = cuda.grid(2)
    if i < availability.shape[0] and j < availability.shape[1]:
        if i != j:
            sum_responsibility = 0.0
            for k in range(availability.shape[0]):
                sum_responsibility += max(0.0, responsibility[k, j])
            availability[i, j] = min(0.0, responsibility[i, j] + sum_responsibility)
        else:
            availability[i, j] = responsibility[i].sum()


class AffinityPropagation:
    def __init__(self, preference=None, max_iter=100):
        self.preference = preference
        self.max_iter = max_iter

    def fit(self, X):
        n_samples = X.shape[0]

        # Вычисляем матрицу сходства
        similarity = np.exp(
            -np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2) ** 2
        )

        # Инициализация матриц ответственности и доступности
        responsibility = np.zeros((n_samples, n_samples), dtype=np.float32)
        availability = np.zeros((n_samples, n_samples), dtype=np.float32)

        # Перенос данных на GPU
        similarity_device = cuda.to_device(similarity)
        responsibility_device = cuda.to_device(responsibility)
        availability_device = cuda.to_device(availability)

        # Основной цикл алгоритма
        for _ in range(self.max_iter):
            # Вычисление ответственности
            threads_per_block = (16, 16)
            blocks_per_grid_x = (
                n_samples + (threads_per_block[0] - 1)
            ) // threads_per_block[0]
            blocks_per_grid_y = (
                n_samples + (threads_per_block[1] - 1)
            ) // threads_per_block[1]
            compute_responsibility_kernel[
                (blocks_per_grid_x, blocks_per_grid_y), threads_per_block
            ](similarity_device, availability_device, responsibility_device)

            # Вычисление доступности
            compute_availability_kernel[
                (blocks_per_grid_x, blocks_per_grid_y), threads_per_block
            ](responsibility_device, availability_device)

            # Копируем данные обратно на хост
            responsibility_device.copy_to_host(responsibility)
            availability_device.copy_to_host(availability)

        # Определение кластеров
        net_responsibility = responsibility + availability
        self.labels_ = np.argmax(net_responsibility, axis=1)

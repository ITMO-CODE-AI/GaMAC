"""Реализация алгоритма DBSCAN (Density-Based Spatial Clustering of Applications with Noise) для GPU.

Модуль содержит:
- Класс DBSCANModel: модель кластеризации DBSCAN
- Класс DBSCAN: основной алгоритм DBSCAN
- Класс DBSCANConfig: конфигурация алгоритма
"""

import cupy as cp
import pylibraft.config

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class DBSCANModel(ClusteringModel):
    """Модель кластеризации DBSCAN.
    
    Хранит метки кластеров, обучающие данные и параметр eps для предсказания.
    
    Attributes:
        X_ (cp.ndarray): Обучающие данные.
        eps_sq (float): Квадрат радиуса окрестности (для оптимизации расчетов).
    """

    def __init__(self, labels_, X_, eps):
        """Инициализирует модель DBSCAN.
        
        Args:
            labels_: Метки кластеров для обучающих данных.
            X_: Обучающие данные.
            eps: Радиус окрестности.
        """
        super().__init__(labels_)
        self.X_ = X_
        self.eps_sq = eps ** 2  # Квадрат eps для оптимизации расчетов

    def predict(self, df: DataFrameType) -> LabelsType:
        """Предсказывает метки кластеров для новых данных.
        
        Использует батчевую обработку для эффективной работы с большими данными.
        
        Args:
            df: Данные для предсказания.
            
        Returns:
            Метки кластеров для входных данных.
        """
        df = cp.asarray(df, dtype=cp.float32)
        X_train = self.X_

        # Батчевая обработка для больших данных
        batch_size = 4096
        predictions = cp.empty(len(df), dtype=cp.int32)

        for i in range(0, len(df), batch_size):
            batch = df[i:i + batch_size]

            # Оптимизированный расчет расстояний через матричные операции
            X_new_norm = cp.sum(batch**2, axis=1)[:, cp.newaxis]
            X_train_norm = cp.sum(X_train**2, axis=1)
            dists_sq = X_new_norm + X_train_norm - 2 * batch @ X_train.T
            dists_sq = cp.maximum(dists_sq, 0)  # Избегаем отрицательных значений из-за ошибок округления

            # Поиск соседей в eps-окрестности
            mask = dists_sq <= self.eps_sq
            has_neighbor = cp.any(mask, axis=1)

            # Для точек без соседей используем ближайшего соседа
            nearest = cp.argmin(dists_sq, axis=1)
            predictions[i:i + batch_size] = cp.where(
                has_neighbor,
                self.labels_[cp.argmax(mask, axis=1)],  # Метка первого найденного соседа
                self.labels_[nearest]  # Метка ближайшего соседа
            )

        return predictions


class DBSCAN(ClusteringAlgo):
    """Алгоритм DBSCAN для GPU.
    
    Алгоритм кластеризации на основе плотности, который группирует точки в кластеры,
    если они находятся в плотной области, и помечает выбросы как шум.
    
    Attributes:
        eps (float): Радиус окрестности.
        eps_sq (float): Квадрат радиуса (для оптимизации).
        min_samples (int): Минимальное количество соседей для core-точки.
        X (cp.ndarray): Обучающие данные.
    """

    def __init__(self, eps=1.0, min_samples=5):
        """Инициализирует алгоритм DBSCAN.
        
        Args:
            eps: Радиус окрестности.
            min_samples: Минимальное количество соседей для core-точки.
        """
        super().__init__()
        self.eps = eps
        self.eps_sq = eps ** 2  # Квадрат eps для оптимизации расчетов
        self.min_samples = min_samples
        self.X = None  # Будет установлено в методе fit

    def _get_neighbors(self, idx):
        """Находит всех соседей точки в eps-окрестности.
        
        Использует оптимизированный расчет расстояний.
        
        Args:
            idx: Индекс точки.
            
        Returns:
            Индексы соседей в eps-окрестности.
        """
        point = self.X[idx]
        diffs = self.X - point
        dists_sq = cp.sum(diffs**2, axis=1)
        return cp.where(dists_sq <= self.eps_sq)[0]

    def _process_core_point(self, idx, visited, is_core):
        """Обрабатывает core-точку и расширяет кластер.
        
        Args:
            idx: Индекс core-точки.
            visited: Массив посещенных точек.
            is_core: Массив флагов core-точек.
            
        Returns:
            Список индексов точек в кластере или None.
        """
        if visited[idx] or not is_core[idx]:
            return None

        cluster = []
        queue = cp.array([idx], dtype=cp.int32)
        visited[idx] = True

        while queue.size > 0:
            current = queue[0]
            queue = queue[1:]
            cluster.append(int(current))

            neighbors = self._get_neighbors(current)
            mask = ~visited[neighbors]
            new_points = neighbors[mask]

            # Добавляем только core-точки в очередь
            core_mask = is_core[new_points]
            queue = cp.concatenate((queue, new_points[core_mask]))
            visited[new_points] = True

            # Граничные точки добавляем в кластер, но не обрабатываем
            cluster.extend(new_points[~core_mask].get().tolist())

        return cluster

    def fit(self, X):
        """Обучает модель DBSCAN на данных.
        
        Args:
            X: Данные для обучения.
            
        Returns:
            Обученная модель DBSCANModel.
        """
        self.X = cp.asarray(X, dtype=cp.float32)
        n = len(self.X)
        visited = cp.zeros(n, dtype=bool)
        is_core = cp.zeros(n, dtype=bool)
        labels_ = cp.full(n, -1, dtype=cp.int32)

        # Предварительное определение core-точек
        for i in range(n):
            if not visited[i]:
                neighbors = self._get_neighbors(i)
                if len(neighbors) >= self.min_samples:
                    is_core[i] = True

        # Основной цикл кластеризации
        cluster_id = 0
        for i in range(n):
            if not visited[i] and is_core[i]:
                cluster = self._process_core_point(i, visited, is_core)
                if cluster:
                    labels_[cluster] = cluster_id
                    cluster_id += 1

        return DBSCANModel(labels_, self.X, self.eps)


class DBSCANConfig(AlgoConfig):
    """Конфигурация алгоритма DBSCAN.
    
    Attributes:
        config_space (dict): Пространство параметров:
            - eps: Диапазон радиуса окрестности (по умолчанию (0.01, 1.5))
            - min_samples: Диапазон минимального количества соседей (по умолчанию (5, 15))
    """

    def __init__(
        self,
        eps=(0.01, 1.5),
        min_samples=(5, 15)
    ):
        """Инициализирует конфигурацию DBSCAN.
        
        Args:
            eps: Диапазон радиуса окрестности.
            min_samples: Диапазон минимального количества соседей.
        """
        super().__init__(
            DBSCAN,
            eps=eps,
            min_samples=min_samples
        )
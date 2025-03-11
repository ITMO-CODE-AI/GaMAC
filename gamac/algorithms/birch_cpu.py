import numpy as np


class ClusteringFeature:
    def __init__(self, N=0, LS=None, SS=None):
        self.N = N  # количество точек
        self.LS = LS if LS is not None else np.zeros(2)  # линейная сумма
        self.SS = SS if SS is not None else 0  # сумма квадратов

    def update(self, point):
        """Обновляет кластерную функцию с новой точкой."""
        self.N += 1
        self.LS += point
        self.SS += np.dot(point, point)

    def merge(self, other):
        """Объединяет две кластерные функции."""
        self.N += other.N
        self.LS += other.LS
        self.SS += other.SS

    def radius(self):
        """Вычисляет радиус кластера."""
        return np.sqrt(self.SS / self.N - np.dot(self.LS / self.N, self.LS / self.N))


class CFNode:
    def __init__(self, branching_factor):
        self.branching_factor = branching_factor
        self.children = []
        self.cf_list = []

    def add(self, cf):
        """Добавляет CF в узел."""
        if len(self.cf_list) < self.branching_factor:
            self.cf_list.append(cf)
        else:
            # Если узел переполнен, необходимо разделить
            self.split(cf)

    def split(self, cf):
        """Разделяет узел на два при переполнении."""
        # Здесь можно использовать K-Means для разделения
        # Для простоты просто создадим новый узел
        new_node = CFNode(self.branching_factor)

        # Объединяем CF и добавляем их в новый узел
        new_node.add(cf)

        # Перемещаем часть CF из текущего узла в новый
        for existing_cf in self.cf_list:
            new_node.add(existing_cf)

        # Очищаем текущий узел и добавляем новый
        self.cf_list = [cf]
        self.children.append(new_node)


class BIRCH:
    def __init__(self, branching_factor=50, threshold=1.0):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.root = CFNode(branching_factor)

    def fit(self, X):
        """Обучает модель на данных X."""
        for point in X:
            cf = ClusteringFeature()
            cf.update(point)
            self.root.add(cf)

    def get_clusters(self):
        """Получает кластеры из дерева."""
        return [cf for cf in self.root.cf_list]

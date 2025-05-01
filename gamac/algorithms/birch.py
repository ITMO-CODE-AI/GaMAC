import cupy as cp
import pylibraft.config

from gamac.algorithms.base import ClusteringModel, ClusteringAlgo, AlgoConfig
from gamac.data.data_pipeline import DataFrameType, LabelsType

pylibraft.config.set_output_as("cupy")


class ClusteringFeature:
    def __init__(self, point):
        self.n = 1
        self.LS = point.astype(cp.float32)
        self.SS = cp.square(self.LS)

    def add_point(self, point):
        self.n += 1
        self.LS += point
        self.SS += cp.square(point)

    def merge(self, other):
        self.n += other.n
        self.LS += other.LS
        self.SS += other.SS

    @property
    def centroid(self):
        return self.LS / self.n

    def radius(self):
        return cp.sqrt(cp.sum(self.SS / self.n - cp.square(self.centroid)))


class CFNode:
    def __init__(self, threshold, branching_factor):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.subclusters = []
        self.children = []
        self.parent = None  # Важное добавление для работы split

    def insert(self, point):
        if not self.children:
            self._insert_to_leaf(point)
        else:
            self._insert_to_internal(point)

    def _insert_to_leaf(self, point):
        if not self.subclusters:
            self.subclusters.append(ClusteringFeature(point))
            return

        point_cf = ClusteringFeature(point)
        radii = cp.array([cf.radius() for cf in self.subclusters])
        new_radius = point_cf.radius()
        distances = radii + new_radius

        closest_idx = int(cp.argmin(distances).item())
        merged_radius = self.subclusters[closest_idx].radius() + new_radius

        if merged_radius <= self.threshold:
            self.subclusters[closest_idx].merge(point_cf)
        else:
            self.subclusters.append(point_cf)
            if len(self.subclusters) > self.branching_factor:
                self._split()

    def _split(self):
        # Шаг 1: Находим пару наиболее удаленных подкластеров
        subcluster_centers = cp.stack([cf.centroid for cf in self.subclusters])
        pairwise_distances = cp.linalg.norm(
            subcluster_centers[:, cp.newaxis] - subcluster_centers,
            axis=2
        )
        
        # Находим индексы максимального расстояния
        max_idx = cp.unravel_index(
            cp.argmax(pairwise_distances),
            pairwise_distances.shape
        )
        idx1, idx2 = int(max_idx[0].item()), int(max_idx[1].item())

        # Шаг 2: Создаем два новых узла
        new_node1 = CFNode(
            threshold=self.threshold,
            branching_factor=self.branching_factor
        )
        new_node2 = CFNode(
            threshold=self.threshold,
            branching_factor=self.branching_factor
        )
        new_node1.parent = self.parent
        new_node2.parent = self.parent

        # Шаг 3: Распределяем подкластеры между новыми узлами
        for cf in self.subclusters:
            dist1 = cp.linalg.norm(cf.centroid - self.subclusters[idx1].centroid)
            dist2 = cp.linalg.norm(cf.centroid - self.subclusters[idx2].centroid)
            
            if dist1 < dist2:
                new_node1.subclusters.append(cf)
            else:
                new_node2.subclusters.append(cf)

        # Шаг 4: Обновляем структуру дерева
        if self.parent:
            # Удаляем текущий узел из родителя и добавляем новые
            self.parent.children.remove(self)
            self.parent.children.append(new_node1)
            self.parent.children.append(new_node2)
        else:
            # Если это корневой узел, создаем новый корень
            new_root = CFNode(
                threshold=self.threshold,
                branching_factor=self.branching_factor
            )
            new_root.children.extend([new_node1, new_node2])
            new_node1.parent = new_root
            new_node2.parent = new_root
            self.parent = new_root

        # Шаг 5: Проверяем необходимость рекурсивного разделения
        for node in [new_node1, new_node2]:
            if len(node.subclusters) > self.branching_factor:
                node._split()


class BirchModel(ClusteringModel):
    def __init__(self, labels_, centroids_):
        super().__init__(labels_)
        self.centroids_ = centroids_

    def predict(self, df: DataFrameType) -> LabelsType:
        distances = cp.linalg.norm(
            df[:, cp.newaxis] - self.centroids_,
            axis=2
        )
        return cp.argmin(distances, axis=1)


class Birch(ClusteringAlgo):
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        super().__init__()
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters

    def fit(self, df: DataFrameType) -> BirchModel:
        root = CFNode(threshold=self.threshold, branching_factor=self.branching_factor)
        for point in df:
            root.insert(point)

        subclusters = self._get_all_subclusters(root)
        if not subclusters:
            raise ValueError("No subclusters formed during BIRCH construction")

        subcluster_centroids = cp.stack([cf.centroid for cf in subclusters])
        centroids_, labels_ = self._kmeans(subcluster_centroids)
        return BirchModel(labels_=labels_, centroids_=centroids_)

    def _get_all_subclusters(self, root):
        nodes = [root]
        subclusters = []
        while nodes:
            node = nodes.pop(0)
            subclusters.extend(node.subclusters)
            nodes.extend(node.children)
        return subclusters

    def _kmeans(self, subcluster_centroids):
        n_subclusters, n_features = subcluster_centroids.shape
        rng = cp.random.RandomState()

        # Автоматическая корректировка количества кластеров
        n_clusters = min(self.n_clusters, len(n_subclusters))

        # Инициализация центроидов K-means
        if n_subclusters <= n_clusters:
            centroids = subcluster_centroids.copy()
            if n_subclusters < n_clusters:
                additional = n_clusters - n_subclusters
                indices = rng.choice(n_subclusters, additional, replace=True)
                centroids = cp.concatenate([centroids, subcluster_centroids[indices]])
        else:
            indices = rng.choice(n_subclusters, n_clusters, replace=False)
            centroids = subcluster_centroids[indices]

        for _ in range(100):
            # Векторизованный расчет расстояний
            distances = cp.linalg.norm(
                subcluster_centroids[:, cp.newaxis] - centroids,
                axis=2
            )
            labels = cp.argmin(distances, axis=1)

            new_centroids = cp.empty((n_clusters, n_features),
                                     dtype=subcluster_centroids.dtype)
            
            for i in range(n_clusters):
                mask = (labels == i)
                if cp.any(mask):
                    new_centroids[i] = subcluster_centroids[mask].mean(axis=0)
                else:
                    # Повторная инициализация пустых кластеров
                    new_centroids[i] = subcluster_centroids[rng.randint(n_subclusters)]

            if cp.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return centroids, labels


class BirchConfig(AlgoConfig):
    def __init__(
            self, *,
            threshold=(0.1, 0.9),
            branching_factor=(10, 80),
            n_clusters=(2, 15),
    ):
        super().__init__(
            Birch,
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters,
        )

import cupy as cp
import pylibraft.config

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

    def insert(self, point):
        if not self.children:
            self._insert_to_leaf(point)
        else:
            self._insert_to_internal(point)

    def _insert_to_leaf(self, point):
        if not self.subclusters:
            self.subclusters.append(ClusteringFeature(point))
            return

        # Векторизованный расчет расстояний
        point_cf = ClusteringFeature(point)
        radii = cp.array([cf.radius() for cf in self.subclusters])
        new_radius = point_cf.radius()
        distances = radii + new_radius

        # Преобразование в целое число
        closest_idx = int(cp.argmin(distances).item())

        merged_radius = self.subclusters[closest_idx].radius() + new_radius

        if merged_radius <= self.threshold:
            self.subclusters[closest_idx].merge(point_cf)
        else:
            self.subclusters.append(point_cf)
            if len(self.subclusters) > self.branching_factor:
                self._split()

    def _split(self):
        # Заглушка для реализации разделения
        pass


class BIRCH:
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.root = CFNode(threshold, branching_factor)
        self.cluster_centers_ = None

    def fit(self, X):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        for point in X_gpu:
            self.root.insert(point)

        subclusters = self._get_all_subclusters()
        if not subclusters:
            raise ValueError("No subclusters formed during BIRCH construction")

        centroids = cp.stack([cf.centroid for cf in subclusters])
        self._kmeans(centroids)

    def _get_all_subclusters(self):
        nodes = [self.root]
        subclusters = []
        while nodes:
            node = nodes.pop(0)
            subclusters.extend(node.subclusters)
            nodes.extend(node.children)
        return subclusters

    def _kmeans(self, centroids):
        # Векторизованная реализация K-means
        for _ in range(100):
            distances = cp.linalg.norm(
                centroids[:, cp.newaxis] - centroids, 
                axis=2
            )
            labels = cp.argmin(distances, axis=0)

            new_centroids = cp.empty_like(centroids)
            for i in range(self.n_clusters):
                mask = (labels == i)
                if cp.any(mask):
                    new_centroids[i] = centroids[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            if cp.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        self.cluster_centers_ = centroids

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet")

        X_gpu = cp.asarray(X, dtype=cp.float32)
        distances = cp.linalg.norm(
            X_gpu[:, cp.newaxis] - self.cluster_centers_,
            axis=2
        )
        return cp.argmin(distances, axis=1).get()

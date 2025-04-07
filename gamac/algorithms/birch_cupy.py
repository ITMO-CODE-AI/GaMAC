import cupy as cp

class ClusteringFeature:
    def __init__(self, point):
        self.n = 1
        self.LS = cp.array(point, dtype=cp.float32)
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
        return cp.sqrt(cp.sum(self.SS/self.n - cp.square(self.centroid)))

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
        
        distances = cp.array([cf.radius() + ClusteringFeature(point).radius() 
                            for cf in self.subclusters])
        closest_idx = cp.argmin(distances)
        
        temp_cf = ClusteringFeature(point)
        merged_radius = self.subclusters[closest_idx].radius() + temp_cf.radius()
        
        if merged_radius <= self.threshold:
            self.subclusters[closest_idx].merge(temp_cf)
        else:
            self.subclusters.append(temp_cf)
            if len(self.subclusters) > self.branching_factor:
                self._split()
    
    def _split(self):
        pass  # Реализация разделения узла

class BIRCH:
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.root = CFNode(threshold, branching_factor)
        self.labels_ = None
    
    def fit(self, X):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        for point in X_gpu:
            self.root.insert(point)
        
        subclusters = self._get_all_subclusters()
        centroids = cp.array([cf.centroid.get() for cf in subclusters])
        
        # Используем cupy для K-средних
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
        # Упрощенная реализация K-средних на GPU
        for _ in range(100):
            distances = cp.linalg.norm(centroids[:, None] - centroids, axis=2)
            new_centroids = cp.empty_like(centroids)
            for i in range(self.n_clusters):
                mask = cp.argmin(distances, axis=0) == i
                new_centroids[i] = cp.mean(centroids[mask], axis=0)
            if cp.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        self.cluster_centers_ = centroids
    
    def predict(self, X):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        distances = cp.linalg.norm(X_gpu[:, None] - self.cluster_centers_, axis=2)
        return cp.argmin(distances, axis=1).get()
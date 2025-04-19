import cupy as cp
import numpy as np
from collections import defaultdict

class HDBSCAN:
    def __init__(self, min_cluster_size=5, min_samples=None, metric='euclidean',
                 alpha=1.0, cluster_selection_epsilon=0.0, 
                 cluster_selection_method='eom', allow_single_cluster=False):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.metric = metric
        self.alpha = alpha
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster

        self.labels_ = None
        self.probabilities_ = None
        self.core_distances_ = None
        self.mst_ = None
        self.single_linkage_tree_ = None

    def fit(self, X):
        X = self._validate_data(X)
        X_gpu = cp.asarray(X, dtype=cp.float32)
        
        # Step 1: Compute KNN and core distances
        knn_distances = self._knn_gpu(X_gpu)
        self.core_distances_ = knn_distances[:, -1]
        
        # Step 2: Compute mutual reachability graph
        mr_graph = self._mutual_reachability_gpu(X_gpu, self.core_distances_)
        
        # Step 3: Build minimum spanning tree
        self.mst_ = self._minimum_spanning_tree(mr_graph)
        
        # Step 4: Build single linkage tree
        self.single_linkage_tree_ = self._build_single_linkage_tree()
        
        # Step 5: Condense the tree
        self.condensed_tree_ = self._condense_tree()
        
        # Step 6: Extract clusters
        self.labels_ = self._extract_clusters()
        
        return self

    def _validate_data(self, X):
        if len(X.shape) != 2:
            raise ValueError("Input data must be 2-dimensional")
        return X.astype(np.float32)

    def _pairwise_distance(self, X):
        """Custom pairwise distance implementation"""
        sum_X = cp.sum(X**2, axis=1)
        return cp.sqrt(cp.abs(sum_X[:, None] + sum_X[None, :] - 2 * cp.dot(X, X.T)))

    def _knn_gpu(self, X_gpu):
        """KNN implementation using pure CuPy"""
        n_samples = X_gpu.shape[0]
        n_neighbors = min(self.min_samples, n_samples - 1)
        
        # Compute pairwise distances
        dist_matrix = self._pairwise_distance(X_gpu)
        
        # Find k+1 neighbors (including self)
        indices = cp.argpartition(dist_matrix, kth=n_neighbors+1, axis=1)
        indices = indices[:, :n_neighbors+1]
        
        # Get distances and sort
        rows = cp.arange(n_samples)[:, None]
        knn_distances = cp.take_along_axis(dist_matrix, indices, axis=1)
        sorted_indices = cp.argsort(knn_distances, axis=1)
        
        return knn_distances[rows, sorted_indices][:, 1:]

    def _mutual_reachability_gpu(self, X_gpu, core_distances):
        """Mutual reachability graph implementation"""
        pairwise_dists = self._pairwise_distance(X_gpu)
        core_matrix = cp.maximum(core_distances[:, None], core_distances)
        return cp.maximum(core_matrix, pairwise_dists / self.alpha)

    def _minimum_spanning_tree(self, graph):
        """Kruskal's algorithm implementation for MST"""
        n = graph.shape[0]
        edges = []
        
        # Extract upper triangle of distance matrix
        for i in range(n):
            for j in range(i+1, n):
                if graph[i, j] > 0:
                    edges.append((graph[i, j], i, j))
        
        # Sort edges by weight
        edges.sort(key=lambda x: x[0])
        
        # Union-Find data structure
        parent = list(range(n))
        mst = []
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        for weight, u, v in edges:
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                mst.append((u, v, weight))
                parent[root_u] = root_v
                
            if len(mst) == n - 1:
                break
                
        return self._mst_to_adjacency_matrix(mst, n)

    def _mst_to_adjacency_matrix(self, mst, n):
        """Convert MST edges list to adjacency matrix"""
        adj_matrix = cp.full((n, n), cp.inf)
        for u, v, weight in mst:
            adj_matrix[u, v] = weight
            adj_matrix[v, u] = weight
        return adj_matrix

    def _build_single_linkage_tree(self):
        """Build hierarchical tree from MST"""
        n = self.mst_.shape[0]
        edges = []
        
        for i in range(n):
            for j in range(i+1, n):
                weight = self.mst_[i, j]
                if weight < cp.inf:
                    edges.append((i, j, weight))
        
        edges = sorted(edges, key=lambda x: x[2])
        return self._union_find(edges, n)

    def _union_find(self, edges, n_nodes):
        """Union-Find with path compression"""
        parent = list(range(2 * n_nodes - 1))
        size = [1] * n_nodes + [0] * (n_nodes - 1)
        next_label = n_nodes
        linkage = []
        
        for u, v, weight in edges:
            root_u = self._find(parent, u)
            root_v = self._find(parent, v)
            
            if root_u != root_v:
                new_size = size[root_u] + size[root_v]
                linkage.append([root_u, root_v, weight, new_size])
                parent[root_u] = parent[root_v] = next_label
                size[next_label] = new_size
                next_label += 1
                
        return np.array(linkage)

    def _find(self, parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _condense_tree(self):
        """Condense hierarchical tree"""
        cluster_tree = defaultdict(list)
        for a, b, dist, count in self.single_linkage_tree_:
            if count >= self.min_cluster_size:
                cluster_tree[a].append((b, dist, count))
                cluster_tree[b].append((a, dist, count))
        return cluster_tree

    def _extract_clusters(self):
        """Extract clusters from condensed tree"""
        visited = set()
        clusters = []
        for node in self.condensed_tree_:
            if node not in visited:
                cluster = self._dfs(node, visited)
                if len(cluster) >= self.min_cluster_size:
                    clusters.append(cluster)
        return self._create_labels(clusters)

    def _dfs(self, node, visited):
        """Depth-first search for cluster extraction"""
        stack = [node]
        cluster = []
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                cluster.append(current)
                for neighbor, _, _ in self.condensed_tree_.get(current, []):
                    stack.append(neighbor)
        return cluster

    def _create_labels(self, clusters):
        """Create label array from clusters"""
        labels = -cp.ones(self.mst_.shape[0], dtype=cp.int32)
        for i, cluster in enumerate(clusters):
            labels[cp.array(cluster)] = i
        return labels.get()

    def predict(self, X):
        """Predict using nearest core points"""
        X_gpu = cp.asarray(X, dtype=cp.float32)
        dists = self._pairwise_distance(X_gpu)
        nearest = cp.argmin(dists, axis=1)
        return self.labels_[nearest.get()]
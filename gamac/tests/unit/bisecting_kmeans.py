import pytest
import cupy as cp
import numpy as np
from gamac.algorithms.bisecting_kmeans import (
    BisectingKMeans,
    BisectingKMeansModel,
    BisectingKMeansConfig
)

class TestBisectingKMeansModel:
    def test_predict(self):
        """Test prediction assigns points to nearest centroid"""
        centroids = cp.array([[1.0, 1.0], [4.0, 4.0]])
        model = BisectingKMeansModel(
            labels_=cp.array([0, 1], dtype=cp.int32),
            centroids_=centroids
        )
        
        test_points = cp.array([
            [1.1, 1.1],  # Cluster 0
            [3.9, 3.9],  # Cluster 1
            [2.5, 2.5]   # Should go to cluster 0 (closer to [1,1])
        ])
        
        predictions = model.predict(test_points)
        expected = cp.array([0, 1, 0])
        assert cp.all(predictions == expected)

    def test_predict_empty(self):
        """Test prediction with empty input"""
        model = BisectingKMeansModel(
            labels_=cp.array([], dtype=cp.int32),
            centroids_=cp.empty((0, 2))
        )
        assert model.predict(cp.empty((0, 2))).size == 0

class TestBisectingKMeans:
    @pytest.fixture
    def sample_data(self):
        """Generate 2D test data with 3 clear clusters"""
        cluster1 = cp.column_stack([
            cp.random.normal(loc=0, scale=0.3, size=100),
            cp.random.normal(loc=0, scale=0.3, size=100)
        ])
        cluster2 = cp.column_stack([
            cp.random.normal(loc=5, scale=0.3, size=100),
            cp.random.normal(loc=5, scale=0.3, size=100)
        ])
        cluster3 = cp.column_stack([
            cp.random.normal(loc=10, scale=0.3, size=100),
            cp.random.normal(loc=0, scale=0.3, size=100)
        ])
        return cp.concatenate([cluster1, cluster2, cluster3])

    def test_initialization(self):
        """Test initialization with different parameters"""
        bkm = BisectingKMeans(n_clusters=4, max_iter=50, init='random', tol=1e-3)
        assert bkm.n_clusters == 4
        assert bkm.max_iter == 50
        assert bkm.init == 'random'
        assert bkm.tol == 1e-3

    def test_kmeans_pp_init(self, sample_data):
        """Test k-means++ initialization"""
        bkm = BisectingKMeans(init='k-means++')
        centers = bkm._kmeans_pp_init(sample_data, 3)
        assert centers.shape == (3, 2)
        assert not cp.allclose(centers[0], centers[1])

    def test_random_init(self, sample_data):
        """Test random initialization"""
        bkm = BisectingKMeans(init='random')
        centers = bkm._random_init(sample_data, 3)
        assert centers.shape == (3, 2)
        assert len(cp.unique(centers, axis=0)) == 3

    def test_kmeans_single_iteration(self, sample_data):
        """Test single k-means iteration"""
        bkm = BisectingKMeans(max_iter=1)
        labels, centers = bkm._kmeans(sample_data, 3)
        assert labels.shape == (300,)
        assert centers.shape == (3, 2)

    def test_sse_calculation(self, sample_data):
        """Test SSE calculation"""
        bkm = BisectingKMeans()
        labels, centers = bkm._kmeans(sample_data, 3)
        sse = bkm._sse(sample_data, labels)
        assert isinstance(sse, float)
        assert sse >= 0

    def test_fit_convergence(self, sample_data):
        """Test that fitting produces correct number of clusters"""
        bkm = BisectingKMeans(n_clusters=3)
        model = bkm.fit(sample_data)
        assert len(cp.unique(model.labels_)) == 3
        assert model.centroids_.shape == (3, 2)

    def test_fit_small_data(self):
        """Test fitting with small dataset"""
        small_data = cp.array([[1, 1], [1.1, 1.1], [5, 5], [5.1, 5.1]])
        bkm = BisectingKMeans(n_clusters=2)
        model = bkm.fit(small_data)
        assert len(cp.unique(model.labels_)) == 2

    def test_fit_single_cluster(self, sample_data):
        """Test fitting with n_clusters=1"""
        bkm = BisectingKMeans(n_clusters=1)
        model = bkm.fit(sample_data)
        assert cp.all(model.labels_ == 0)
        assert model.centroids_.shape == (1, 2)
import pytest
import cupy as cp
import numpy as np
from gamac.algorithms.dbscan import DBSCAN, DBSCANModel, DBSCANConfig

class TestDBSCANModel:
    @pytest.fixture
    def sample_model(self):
        """Create a sample DBSCAN model with known clusters"""
        X = cp.array([
            [1.0, 1.0], [1.1, 1.1], [1.2, 1.2],  # Cluster 0
            [5.0, 5.0], [5.1, 5.1],               # Cluster 1
            [10.0, 10.0]                           # Noise (-1)
        ])
        labels = cp.array([0, 0, 0, 1, 1, -1], dtype=cp.int32)
        return DBSCANModel(labels_=labels, X_=X, eps=1.0)

    def test_predict_core_points(self, sample_model):
        """Test prediction for points near core samples"""
        test_points = cp.array([
            [1.05, 1.05],  # Near cluster 0
            [5.05, 5.05]    # Near cluster 1
        ])
        predictions = sample_model.predict(test_points)
        expected = cp.array([0, 1])
        assert cp.all(predictions == expected)

    def test_predict_border_points(self, sample_model):
        """Test prediction for border points"""
        test_points = cp.array([
            [1.5, 1.5],  # Border of cluster 0
            [5.5, 5.5]    # Border of cluster 1
        ])
        predictions = sample_model.predict(test_points)
        expected = cp.array([0, 1])
        assert cp.all(predictions == expected)

    def test_predict_noise_points(self, sample_model):
        """Test prediction for noise points"""
        test_points = cp.array([
            [3.0, 3.0],   # Between clusters
            [20.0, 20.0]  # Far away
        ])
        predictions = sample_model.predict(test_points)
        # Should assign to nearest cluster
        expected = cp.array([0, -1])
        assert cp.all(predictions == expected)

    def test_predict_empty_input(self, sample_model):
        """Test prediction with empty input"""
        predictions = sample_model.predict(cp.empty((0, 2)))
        assert predictions.size == 0

    def test_predict_large_input(self, sample_model):
        """Test prediction with large input (batches)"""
        test_points = cp.random.normal(loc=3.0, scale=2.0, size=(5000, 2))
        predictions = sample_model.predict(test_points)
        assert predictions.shape == (5000,)
        assert cp.all((predictions >= -1) & (predictions <= 1))

class TestDBSCAN:
    @pytest.fixture
    def sample_data(self):
        """Generate sample data with 3 clear clusters and some noise"""
        cluster1 = cp.random.normal(loc=0.0, scale=0.1, size=(50, 2))
        cluster2 = cp.random.normal(loc=5.0, scale=0.1, size=(50, 2))
        cluster3 = cp.random.normal(loc=10.0, scale=0.1, size=(50, 2))
        noise = cp.random.uniform(low=-5.0, high=15.0, size=(10, 2))
        return cp.concatenate([cluster1, cluster2, cluster3, noise])

    def test_initialization(self):
        """Test initialization with different parameters"""
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        assert dbscan.eps == 0.5
        assert dbscan.min_samples == 10
        assert dbscan.eps_sq == 0.25

    def test_get_neighbors(self):
        """Test neighbor calculation"""
        X = cp.array([
            [1.0, 1.0],
            [1.1, 1.1],
            [5.0, 5.0]
        ])
        dbscan = DBSCAN(eps=1.0)
        dbscan.X = X
        
        neighbors = dbscan._get_neighbors(0)
        assert len(neighbors) == 2
        assert 0 in neighbors
        assert 1 in neighbors

    def test_fit_clusters(self, sample_data):
        """Test fitting finds correct number of clusters"""
        dbscan = DBSCAN(eps=1.0, min_samples=5)
        model = dbscan.fit(sample_data)
        
        unique_labels = cp.unique(model.labels_)
        # Should find 3 clusters + noise (-1)
        assert len(unique_labels) == 4
        assert -1 in unique_labels.get()

    def test_fit_noise_only(self):
        """Test fitting with all noise"""
        X = cp.random.uniform(low=0.0, high=10.0, size=(100, 2))
        dbscan = DBSCAN(eps=0.1, min_samples=10)
        model = dbscan.fit(X)
        assert cp.all(model.labels_ == -1)
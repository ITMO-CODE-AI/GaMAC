import pytest
import cupy as cp
import numpy as np
from gamac.algorithms.birch import Birch, BirchModel, BirchConfig, ClusteringFeatureGPU, CFNodeGPU, CFTreeGPU

class TestClusteringFeatureGPU:
    def test_cf_operations(self):
        point = cp.array([1.0, 2.0], dtype=cp.float64)
        cf = ClusteringFeatureGPU(point)
        
        # Test initial values
        assert cf.n == 1
        assert cp.allclose(cf.LS, point)
        assert cp.allclose(cf.SS, cp.square(point))
        
        # Test add_point
        new_point = cp.array([2.0, 3.0], dtype=cp.float64)
        cf.add_point(new_point)
        assert cf.n == 2
        assert cp.allclose(cf.LS, point + new_point)
        assert cp.allclose(cf.SS, cp.square(point) + cp.square(new_point))
        
        # Test merge
        other_cf = ClusteringFeatureGPU(cp.array([3.0, 4.0], dtype=cp.float64))
        cf.merge(other_cf)
        assert cf.n == 3
        assert cp.allclose(cf.LS, point + new_point + other_cf.LS)

class TestCFNodeGPU:
    @pytest.fixture
    def sample_node(self):
        return CFNodeGPU(threshold=1.0, branching_factor=3)

    def test_insert_and_split(self, sample_node):
        data = cp.array([[1.0, 2.0], [1.1, 2.1], [1.2, 1.2], [4.3, 4.3]], dtype=cp.float64)
        
        for point in data:
            cf = ClusteringFeatureGPU(point)
            sample_node.insert(cf, point)
        
        # Check split happened
        assert len(sample_node.cfs) == 2

class TestBirchModel:
    @pytest.fixture
    def sample_model(self):
        class MockCF:
            def centroid(self):
                return cp.array([1.0, 1.0])
        
        class MockTree:
            root = type('', (), {'cfs': [MockCF(), MockCF()]})()
        
        labels = cp.array([0, 1, 0], dtype=cp.int32)
        sub_labels = np.array([0, 1])
        return BirchModel(labels_=labels, subcluster_labels=sub_labels, tree=MockTree())

    def test_predict(self, sample_model):
        X = cp.array([
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0]
        ], dtype=cp.float32)
        
        predictions = sample_model.predict(X)
        assert predictions.shape == (3,)
        assert cp.all(predictions == cp.array([0, 0, 0]))

class TestBirch:
    @pytest.fixture
    def sample_data(self):
        cluster1 = cp.random.normal(loc=0.0, scale=0.1, size=(50, 2)).astype(cp.float32)
        cluster2 = cp.random.normal(loc=5.0, scale=0.1, size=(50, 2)).astype(cp.float32)
        return cp.concatenate([cluster1, cluster2])

    def test_fit_basic(self, sample_data):
        birch = Birch(threshold=0.5, branching_factor=50, n_clusters=2)
        model = birch.fit(sample_data)
        
        assert model.labels_.shape == (100,)
        unique_labels = cp.unique(model.labels_)
        assert len(unique_labels) <= 2

    def test_fit_single_cluster(self):
        X = cp.random.normal(loc=0.0, scale=0.1, size=(100, 2)).astype(cp.float32)
        birch = Birch(n_clusters=1)
        model = birch.fit(X)
        assert len(cp.unique(model.labels_)) == 1

    def test_fit_edge_cases(self):
        # Empty input
        birch = Birch()
        birch.fit(cp.empty((0, 2)))

        # Single point
        X = cp.array([[1.0, 2.0]], dtype=cp.float32)
        birch = Birch(n_clusters=1)
        model = birch.fit(X)
        assert model.labels_[0] == 0
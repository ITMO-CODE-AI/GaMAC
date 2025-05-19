import pytest
import cupy as cp
import numpy as np
from gamac.algorithms.meanshift import MeanShift, MeanShiftModel, MeanShiftConfig

class TestMeanShiftModel:
    @pytest.fixture
    def sample_model(self):
        centroids = cp.array([[1.0, 1.0], [4.0, 4.0]], dtype=cp.float32)
        labels = cp.array([0, 1, 0, 1], dtype=cp.int32)
        return MeanShiftModel(labels_=labels, centroids_=centroids)

    def test_predict_basic(self, sample_model):
        test_points = cp.array([
            [1.1, 1.1],  # Cluster 0
            [3.9, 3.9]   # Cluster 1
        ], dtype=cp.float32)
        
        predictions = sample_model.predict(test_points)
        expected = cp.array([0, 1], dtype=cp.int32)
        assert cp.all(predictions == expected)

    def test_predict_empty_input(self, sample_model):
        predictions = sample_model.predict(cp.empty((0, 2)))
        assert predictions.shape == (0,)

    def test_unfitted_model(self):
        model = MeanShiftModel(labels_=cp.array([]), centroids_=None)
        with pytest.raises(ValueError):
            model.predict(cp.array([[1.0, 1.0]]))

class TestMeanShift:
    @pytest.fixture
    def sample_data(self):
        cluster1 = cp.random.normal(loc=0.0, scale=0.1, size=(100, 2))
        cluster2 = cp.random.normal(loc=5.0, scale=0.1, size=(100, 2))
        return cp.concatenate([cluster1, cluster2])

    def test_initialization(self):
        ms = MeanShift(bandwidth=1.5, max_iter=200, tol=1e-4)
        assert ms.bandwidth == 1.5
        assert ms.max_iter == 200
        assert ms.tol == 1e-4

    def test_fit_convergence(self, sample_data):
        ms = MeanShift(bandwidth=1.0, max_iter=1000)
        model = ms.fit(sample_data)
        assert model.centroids_.shape[0] == 2
        assert len(cp.unique(model.labels_)) == 2

    def test_centroid_merging(self):
        X = cp.array([
            [1.0, 1.0], [1.1, 1.1],
            [1.2, 1.2], [5.0, 5.0]
        ], dtype=cp.float32)
        
        ms = MeanShift(bandwidth=1.5)
        model = ms.fit(X)
        assert model.centroids_.shape[0] == 2

    def test_single_cluster(self):
        X = cp.random.normal(loc=0.0, scale=0.1, size=(100, 2))
        ms = MeanShift(bandwidth=2.0)
        model = ms.fit(X)
        assert model.centroids_.shape[0] == 1

    def test_all_noise(self):
        X = cp.random.uniform(low=0.0, high=10.0, size=(100, 2))
        ms = MeanShift(bandwidth=0.1)
        model = ms.fit(X)
        assert model.centroids_.shape[0] > 10  # Many small clusters

    def test_label_assignment(self, sample_data):
        ms = MeanShift(bandwidth=1.0)
        model = ms.fit(sample_data)
        distances = cp.linalg.norm(sample_data - model.centroids_[model.labels_], axis=1)
        assert cp.all(distances <= 1.0)
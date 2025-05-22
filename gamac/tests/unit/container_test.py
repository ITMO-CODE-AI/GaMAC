import pytest
import cupy as cp
import numpy as np
from unittest.mock import MagicMock

# Import the module containing the EstimationContainer class
from gamac.estimation.container import EstimationContainer


class TestEstimationContainer:
    @pytest.fixture
    def sample_data(self):
        # Create sample data for testing
        data = cp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=cp.float32)
        labels = cp.array([0, 1, 0, 1], dtype=cp.int32)
        uniq_labels = cp.array([0, 1], dtype=cp.int32)
        uniq_labels_arr = np.array([0, 1], dtype=np.int32)
        return {
            'data': data,
            'labels': labels,
            'uniq_labels_gpu': uniq_labels,
            'uniq_labels_arr': uniq_labels_arr
        }

    @pytest.fixture
    def sample_data_with_noise(self):
        # Create sample data with noise labels
        data = cp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=cp.float32)
        labels = cp.array([0, -1, 0, 1], dtype=cp.int32)
        return data, labels

    @pytest.fixture
    def mock_middleware(self):
        # Mock the MIDDLEWARE object
        mock = MagicMock()
        mock.get_centroids.return_value.invoke.return_value = None
        mock.get_cent_dists.return_value.invoke.return_value = None
        mock.get_sym_data.return_value.invoke.return_value = None
        mock.get_sym_dists.return_value.invoke.return_value = None
        mock.get_cent_matrix.return_value.invoke.return_value = None
        return mock

    def test_create(self, sample_data):
        container = EstimationContainer.create(
            df=sample_data['data'],
            labels=sample_data['labels']
        )

        assert container is not None
        assert cp.array_equal(container.data, sample_data['data'])
        assert cp.array_equal(container.labels, sample_data['labels'])
        assert container.n == 4
        assert container.d == 2
        assert container.k == 2

    def test_create_with_noise_below_threshold(self, sample_data_with_noise, monkeypatch):
        data, labels = sample_data_with_noise
        monkeypatch.setattr(EstimationContainer, 'NOISE_THRESHOLD', 0.5)

        container = EstimationContainer.create(data, labels)

        assert container is not None
        assert len(container.data) == 3  # One noise point removed
        assert len(container.labels) == 3
        assert container.k == 2  # Still two unique labels (0 and 1)

    def test_create_with_noise_above_threshold(self, sample_data_with_noise):
        data, labels = sample_data_with_noise

        with pytest.raises(ValueError) as excinfo:
            EstimationContainer.create(data, labels)

        assert "Received 25% objects with noise labels" in str(excinfo.value)

    def test_clusters_property(self, sample_data):
        container = EstimationContainer(**sample_data)
        clusters = container.clusters

        assert len(clusters) == 2
        assert all(isinstance(cluster, cp.ndarray) for cluster in clusters)
        assert len(clusters[0]) == 2  # Two points with label 0
        assert len(clusters[1]) == 2  # Two points with label 1

    def test_noise_threshold(self):
        # Test class variable can be overridden
        original_threshold = EstimationContainer.NOISE_THRESHOLD
        try:
            EstimationContainer.NOISE_THRESHOLD = 0.5
            assert EstimationContainer.NOISE_THRESHOLD == 0.5
        finally:
            EstimationContainer.NOISE_THRESHOLD = original_threshold

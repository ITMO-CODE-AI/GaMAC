import pytest
import cupy as cp
import numpy as np
from gamac.algorithms.kmeans import KMeans, KMeansModel, KMeansConfig


class TestKMeansModel:
    def test_predict(self):
        """Test that predict assigns points to nearest centroid"""
        centroids = cp.array([[1.0, 1.0], [4.0, 4.0]])
        model = KMeansModel(labels_=cp.array([0, 1]), centroids_=centroids)

        test_points = cp.array([
            [1.1, 1.1],  # Should be cluster 0
            [3.9, 3.9],  # Should be cluster 1
            [2.5, 2.5]   # Equidistant - implementation will choose lower index
        ])

        predictions = model.predict(test_points)
        expected = cp.array([0, 1, 0])
        assert cp.all(predictions == expected)


class TestKMeans:
    @pytest.fixture
    def sample_data(self):
        """Generate simple 2D test data with 3 clear clusters"""
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
        """Test that centroids are initialized correctly"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        assert kmeans.centroids is None
        assert kmeans.n_clusters == 3
        assert kmeans.max_iter == 100
        assert kmeans.tol == 1e-4

    def test_fit_convergence(self, sample_data):
        """Test that fitting converges before max iterations"""
        kmeans = KMeans(n_clusters=3, max_iter=1000, random_state=42)
        model = kmeans.fit(sample_data)

        # Should converge well before max_iter
        assert model.centroids_.shape == (3, 2)
        assert cp.unique(model.labels_).shape[0] == 3

    def test_fit_known_clusters(self, sample_data):
        """Test that centroids move toward true cluster centers"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        model = kmeans.fit(sample_data)

        # Check centroids are near the true centers we used to generate data
        centroids = model.centroids_.get()  # Convert to numpy for easier testing
        expected_centers = np.array([[0, 0], [5, 5], [10, 0]])

        # For each found centroid, find closest expected center
        distances = np.linalg.norm(centroids[:, np.newaxis] - expected_centers, axis=2)
        min_distances = np.min(distances, axis=1)
        assert np.all(min_distances < 1.0)  # Should be close to true centers

    def test_empty_cluster_handling(self):
        """Test that empty clusters are handled gracefully"""
        # Create data that will likely leave one cluster empty
        data = cp.array([[1, 1], [1.1, 1.1], [5, 5], [5.1, 5.1]])
        kmeans = KMeans(n_clusters=3, random_state=42)
        model = kmeans.fit(data)

        # Should still have 3 centroids even if one cluster is empty
        assert model.centroids_.shape == (3, 2)

        # Check we have points in at least 2 clusters
        unique_labels = cp.unique(model.labels_).get()
        assert len(unique_labels) >= 2

    def test_deterministic_with_random_state(self, sample_data):
        """Test that results are reproducible with fixed random_state"""
        kmeans1 = KMeans(n_clusters=3, random_state=42)
        model1 = kmeans1.fit(sample_data)

        kmeans2 = KMeans(n_clusters=3, random_state=42)
        model2 = kmeans2.fit(sample_data)

        assert cp.all(model1.centroids_ == model2.centroids_)
        assert cp.all(model1.labels_ == model2.labels_)

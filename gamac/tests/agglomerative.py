import numpy as np
from sklearn.datasets import make_blobs
import pylibraft.config

from gamac.algorithms.gpu.agglomerative import AgglomerativeClustering

pylibraft.config.set_output_as("cupy")


def gpu_test():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6)

    clustering = AgglomerativeClustering(n_clusters=3)
    clustering.fit(X)

    print(clustering.labels_)


gpu_test()

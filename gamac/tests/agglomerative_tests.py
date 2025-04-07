import sys

import numpy as np
from sklearn.datasets import make_blobs

sys.path.append('../')
# from algorithms.kmeans_cpu import KMeans
from algorithms.agglomerative_cupy import AgglomerativeClustering

# def cpu_test():
#     # Generate synthetic data
#     X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6)

#     clf = KMeans(k=3)
#     y_pred = clf.predict(X)

#     print(y_pred)


def gpu_test():
    # Generate synthetic data
    # X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6)
    X = np.random.rand(10, 3)

    clustering = AgglomerativeClustering(n_clusters=3)
    clustering.fit(X)

    print(clustering.labels_)


gpu_test()

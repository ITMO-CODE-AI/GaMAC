from ..source.silhouette import silhouette_samples_memory_saving

from math import ceil
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def sil_test():
    X, y = make_blobs(random_state=42)
    kmeans = KMeans(n_clusters=2, random_state=42)
    print(silhouette_score(X, kmeans.fit_predict(X)))
    print(silhouette_samples_memory_saving(X, kmeans.fit_predict(X)))

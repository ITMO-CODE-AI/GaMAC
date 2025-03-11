from typing import List

import numpy as np
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    MeanShift,
    AgglomerativeClustering,
    BisectingKMeans,
    Birch,
)

from gamac.src.meta.collector import DatasetForMetaCVI


class Producer:
    def __init__(self, algo, name):
        self.algo, self.name = algo, name

    def fit_predict(self, dataset: DatasetForMetaCVI) -> np.ndarray:
        return self.algo.fit_predict(dataset.data)


class ProducerProvider:
    @staticmethod
    def get_all() -> List[Producer]:
        return [
            *ProducerProvider.agglomerative(),
            *ProducerProvider.dbscan(),
            *ProducerProvider.kmeans(),
            *ProducerProvider.bisecting(),
            *ProducerProvider.birch(),
            *ProducerProvider.meanshift(),
        ]

    @staticmethod
    def kmeans():
        producers = list()
        for n_clusters in range(2, 8):
            producers.append(
                Producer(
                    KMeans(n_clusters=n_clusters, max_iter=100, random_state=42),
                    f'KMeans'
                )
            )
        return producers

    @staticmethod
    def bisecting():
        producers = list()
        for n_clusters in range(2, 8):
            producers.append(
                Producer(
                    BisectingKMeans(n_clusters=n_clusters, max_iter=100, random_state=42),
                    f'BisectingKMeans'
                )
            )
        return producers

    @staticmethod
    def birch():
        producers = list()
        for n_clusters in range(2, 8):
            for branching_factor in [20, 30, 50, 80, 120, 150]:
                for threshold in [0.1, 0.2, 0.35, 0.5, 0.7]:
                    producers.append(
                        Producer(
                            Birch(n_clusters=n_clusters, branching_factor=branching_factor, threshold=threshold),
                            f'Birch'
                        )
                    )
        return producers

    @staticmethod
    def meanshift():
        producers = list()
        for bandwidth in [0.05, 0.11, 0.18, 0.26, 0.35, 0.45, 0.56, 0.68]:
            for cluster_all in [True, False]:
                producers.append(
                    Producer(
                        MeanShift(bandwidth=bandwidth, max_iter=50, cluster_all=cluster_all, n_jobs=-1),
                        'MeanShift'
                    )
                )
        return producers

    @staticmethod
    def dbscan():
        producers = list()
        for min_samples in [5, 12, 21, 33, 55, 68]:
            for eps in [0.05, 0.12, 0.21, 0.33, 0.55, 0.68]:
                producers.append(
                    Producer(
                        DBSCAN(eps=eps, min_samples=min_samples),
                        f'DBSCAN'
                    )
                )

        return producers

    @staticmethod
    def agglomerative():
        producers = list()
        for n_clusters in range(2, 8):
            for linkage in ['ward', 'average']:
                producers.append(
                    Producer(
                        AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage),
                        f'AgglomerativeClustering'
                    )
                )
        return producers

import sys

import pandas as pd
from sklearn.datasets import make_blobs
import pylibraft.config

sys.path.append('../')
from algorithms.bisecting_kmeans import BisectingKMeans
from algorithms.agglomerative import AgglomerativeClustering
from algorithms.birch import BIRCH
from algorithms.affinity import AffinityPropagation
from algorithms.dbscan import DBSCAN
from algorithms.hdbscan import HDBSCAN

pylibraft.config.set_output_as("cupy")


def gpu_test():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6)

    bisecting_kmeans_testing(X)
    agglomerative_testing(X)
    birch_testing(X)
    affinity_testing(X)
    dbscan_testing(X)
    hdbscan_testing(X)


def bisecting_kmeans_testing(X):
    print('BisectingKMeans')
    clf = BisectingKMeans(n_clusters=3, max_iter=100)
    clf.fit(X)
    y_pred = clf.predict(X)
    print('labels', y_pred)


def agglomerative_testing(X):
    print('AgglomerativeClustering')
    for linkage in ['single', 'complete', 'average']:
        print(f'Linkage: {linkage}')
        clf = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        clf.fit(X)
        y_pred = clf.predict(X)
        print('labels', y_pred)


def birch_testing(X):
    print('BIRCH')
    clf = BIRCH(threshold=0.5, branching_factor=50, n_clusters=3)
    clf.fit(X)
    y_pred = clf.predict(X)
    print('labels', y_pred)


def affinity_testing(X):
    print('AffinityPropagation')
    clf = AffinityPropagation(max_iter=100, convergence_iter=15, tol=1e-6)
    clf.fit(X)
    y_pred = clf.predict(X)
    print('labels', y_pred)


def dbscan_testing(X):
    print('DBSCAN')
    clf = DBSCAN(eps=1.0, min_samples=5)
    clf.fit(X)
    y_pred = clf.predict(X)
    print('labels', y_pred)


def hdbscan_testing(X):
    print('HDBSCAN')
    clf = HDBSCAN()
    clf.fit(X)
    y_pred = clf.predict(X)
    print('labels', y_pred)


gpu_test()

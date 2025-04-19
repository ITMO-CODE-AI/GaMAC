import os

import pandas as pd
from sklearn.datasets import make_blobs
import pylibraft.config

from gamac.algorithms.cpu.kmeans import KMeans
from gamac.algorithms.gpu.kmeans import KMeansGPU

pylibraft.config.set_output_as("cupy")


def cpu_test():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6)

    clf = KMeans(k=3)
    y_pred = clf.predict(X)

    print(y_pred)


def gpu_test():
    X = pd.read_parquet(os.path.join('data', 'cifar_embs.parquet'), engine='pyarrow').to_numpy()
    clf = KMeansGPU(k=3)
    y_pred = clf.predict(X)

    print(y_pred)


gpu_test()

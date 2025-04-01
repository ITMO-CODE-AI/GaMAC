import os
import pandas as pd
import sys
from sklearn.datasets import make_blobs
import pylibraft.config

pylibraft.config.set_output_as("cupy")

sys.path.append('../')
print(os.getcwd())
from algorithms.kmeans_cpu import KMeans
from algorithms.kmeans_gpu_cupy import KMeansGPU

def cpu_test():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6)

    clf = KMeans(k=3)
    y_pred = clf.predict(X)

    print(y_pred)

def gpu_test():
    X = pd.read_parquet(os.path.join('data', 'cifar_embs.parquet'), engine='pyarrow').to_numpy()
    batch_size = X.shape[0]
    threads_per_block = 256
    # blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    clf = KMeansGPU(k=3)
    y_pred = clf.predict(X)

    print(y_pred)

gpu_test()

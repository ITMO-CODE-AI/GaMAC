import sys

from sklearn.datasets import make_blobs

sys.path.append('../')
from algorithms.kmeans_cpu import KMeans
from algorithms.kmeans_gpu import KMeansGpu

def cpu_test():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6)

    clf = KMeans(k=3)
    y_pred = clf.predict(X)

    print(y_pred)

def gpu_test():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6)
    
    batch_size = X.shape[0]
    threads_per_block = 256
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    y_pred = KMeansGpu.predict[blocks_per_grid, threads_per_block](X)

    print(y_pred)

gpu_test()

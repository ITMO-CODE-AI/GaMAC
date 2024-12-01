import sys

from sklearn.datasets import make_blobs

sys.path.append('../')
from algorithms.dbscan_gpu import DBSCANonGPU

def gpu_test():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6)
    
    dbscan = DBSCANonGPU()
    batch_size = X.shape[0]
    threads_per_block = 256
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    y_pred = dbscan.predict[blocks_per_grid, threads_per_block](X)

    print(y_pred)

gpu_test()
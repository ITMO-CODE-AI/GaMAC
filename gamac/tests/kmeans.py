import sys

import pandas as pd
from sklearn.datasets import make_blobs
import pylibraft.config

sys.path.append('../')
from algorithms.kmeans import KMeans

pylibraft.config.set_output_as("cupy")


def gpu_test():
    X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.6)
    # X = pd.read_parquet(os.path.join('data', 'cifar_embs.parquet'), engine='pyarrow').to_numpy()
    clf = KMeans(k=3)
    y_pred = clf.predict(X)

    print(y_pred)


gpu_test()

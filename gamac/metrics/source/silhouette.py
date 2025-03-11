from math import ceil
from numba import cuda, types
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ..utils.funcs import euclidean_distances_mean, euclidean_distances_sum, filter_list
from ..utils.checks import check_number_of_labels

THREADSPERBLOCK = 256


def silhouette_samples_memory_saving(X, labels):
    # X, labels = sklearn.utils.check_X_y(X, labels, accept_sparse=['csc', 'csr'])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    check_number_of_labels(len(le.classes_), X.shape[0])

    unique_labels = le.classes_
    n_samples_per_label = np.bincount(labels, minlength=len(unique_labels))

    intra_clust_dists = np.zeros(X.shape[0], dtype=X.dtype)

    inter_clust_dists = np.inf + intra_clust_dists

    blockspergrid = ceil(X.shape[0] / THREADSPERBLOCK)
    mask, other_mask = np.zeros(labels.shape[0]), np.zeros(labels.shape[0])
    print(unique_labels.shape, labels.shape, n_samples_per_label.shape,
          intra_clust_dists.shape, inter_clust_dists.shape, X.shape, mask.shape, other_mask.shape)
    sil_samples = silhouette_samples_memory_saving_gpu[blockspergrid, THREADSPERBLOCK](
        unique_labels, labels, n_samples_per_label, intra_clust_dists, X, inter_clust_dists, mask, other_mask)

    return sil_samples


@cuda.jit((types.float32[:], types.float32[:], types.float32[:], types.float32[:], types.float32[:, :], types.float32[:], types.float32[:], types.float32[:]))
def silhouette_samples_memory_saving_gpu(unique_labels, labels, n_samples_per_label, intra_clust_dists, X, inter_clust_dists, mask, other_mask):
    for curr_label in range(len(unique_labels)):

        # mask = labels == curr_label
        # mask = [x == curr_label for x in list_labels]
        for i, l in enumerate(labels):
            mask[i] = curr_label == l
        X = np.array(X)
        n_samples_curr_lab = n_samples_per_label[curr_label] - 1
        if n_samples_curr_lab != 0:
            intra_clust_dists[mask] = euclidean_distances_sum(
                X[mask, :]) / n_samples_curr_lab

        for other_label in range(len(unique_labels)):
            if other_label != curr_label:
                other_mask = labels == other_label
                other_distances = euclidean_distances_mean(
                    X[mask, :], X[other_mask, :])
                inter_clust_dists[mask] = np.minimum(
                    inter_clust_dists[mask], other_distances)

    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    sil_samples[n_samples_per_label.take(labels) == 1] = 0
    return sil_samples

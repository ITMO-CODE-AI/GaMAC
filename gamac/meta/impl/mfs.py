"""
CPU meta features computation
"""

import numba as nb
import numpy as np

NUM_BUCKETS = 128

@nb.njit
def dist(x, y) -> float:
    diff = x - y
    sqr = np.sum(diff * diff)
    return np.sqrt(sqr)


@nb.njit
def _generate(obj_dists: np.ndarray):
    buckets = np.array_split(obj_dists, NUM_BUCKETS)

    diff_avg = np.empty(shape=NUM_BUCKETS)
    diff_max = np.empty(shape=NUM_BUCKETS)

    dist_avg = np.empty(shape=NUM_BUCKETS)
    dist_range = np.empty(shape=NUM_BUCKETS)

    for bucket_idx, bucket in enumerate(buckets):
        diff = bucket[1:] - bucket[:-1]
        diff_avg[bucket_idx] = np.mean(diff)
        diff_max[bucket_idx] = np.max(diff)

        dist_avg[bucket_idx] = np.mean(bucket)
        dist_range[bucket_idx] = bucket[-1] - bucket[0]

    return [
        *dist_avg,
        *dist_range,
        *diff_avg,
        *diff_max,
    ]

@nb.njit
def _extract_internal(data: np.ndarray) -> np.ndarray:
    data_features, n, d_max = list(), len(data), float('-inf')

    for x_obj in data:
        obj_dists = np.empty(shape=n)

        for y_idx, y_obj in enumerate(data):
            xy_dist = dist(x_obj, y_obj)
            obj_dists[y_idx] = xy_dist
            d_max = max(d_max, xy_dist)

        obj_dists.sort()
        d_feats = _generate(obj_dists)
        data_features.append(d_feats)

    return np.array(data_features) / d_max

def compute_mfs(data: np.ndarray) -> np.ndarray:
    data_features = _extract_internal(data)
    return np.mean(data_features, axis=0)
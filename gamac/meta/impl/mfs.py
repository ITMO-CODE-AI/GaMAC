import numba as nb
import numpy as np

NUM_BUCKETS = 100

@nb.njit
def _dist(x, y) -> float:
    diff = x - y
    sqr = np.sum(diff * diff)
    return np.sqrt(sqr)

@nb.njit
def _estimate_d_max(data: np.ndarray) -> float:
    centroid = np.zeros(shape=data.shape[1])
    for x_obj in data:
        centroid += x_obj
    centroid /= len(data)

    c_dist, p_far = float('-inf'), None
    for x_obj in data:
        cx_dist = _dist(x_obj, centroid)
        if cx_dist > c_dist:
            c_dist, p_far = cx_dist, x_obj

    d_max = float('-inf')
    for x_obj in data:
        px_dist = _dist(x_obj, p_far)
        if px_dist > d_max:
            d_max = px_dist
    return d_max


@nb.njit
def _generate_from_dists(obj_dists: np.ndarray):
    diff = obj_dists[1:] - obj_dists[:-1]
    diff_regions = np.array_split(diff, NUM_BUCKETS)

    diff_mean = [np.mean(d) for d in diff_regions]
    diff_max = [np.max(d) for d in diff_regions]

    quantiles = np.array_split(obj_dists, NUM_BUCKETS)
    quantiles_mean = [np.mean(q) for q in quantiles]

    return [
        *quantiles_mean,
        *diff_mean,
        *diff_max,
    ]

@nb.njit
def _extract_internal(data: np.ndarray) -> np.ndarray:
    d_max = _estimate_d_max(data)
    data_features, n = list(), len(data)
    for x_obj in data:
        obj_dists = np.empty(shape=n)
        buckets = np.zeros(shape=NUM_BUCKETS)

        for y_idx, y_obj in enumerate(data):
            xy_dist = _dist(x_obj, y_obj) / d_max
            obj_dists[y_idx] = xy_dist

            bucket_idx = int(xy_dist * NUM_BUCKETS)
            bucket_idx = min(NUM_BUCKETS - 1, bucket_idx)
            buckets[bucket_idx] += 1

        buckets /= n
        obj_dists.sort()

        d_feats = _generate_from_dists(obj_dists)

        data_features.append([*buckets, *d_feats])

    return np.array(data_features)

def compute_mfs(data: np.ndarray) -> np.ndarray:
    data_features = _extract_internal(data)
    return np.array([
        *np.mean(data_features, axis=0),
        *np.std(data_features, axis=0),
    ])

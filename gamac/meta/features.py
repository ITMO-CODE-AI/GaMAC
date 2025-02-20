import numpy as np
from scipy.stats import tmean, tstd
from sklearn.metrics import pairwise_distances

from gamac.meta.storage import traverse_data, COMPUTED, CONTENTS

NUM_BUCKETS = 100


# @nb.jit(nopython=True)
def _extract_internal(distances: np.ndarray) -> np.ndarray:
    norm_distances = distances / np.max(distances)
    n, data_features = len(distances), list()
    for obj_dists in norm_distances:
        sorted_dists = sorted(obj_dists)
        buckets = np.array_split(sorted_dists, NUM_BUCKETS)
        obj_features = list(map(np.mean, buckets))
        data_features.append(obj_features)
    return np.array(data_features)

def extract(distances: np.ndarray) -> np.ndarray:
    data_features = _extract_internal(distances)
    return np.array([
        *tmean(data_features, axis=0),
        *tstd(data_features, axis=0),
    ])


def features(data_path):
    if COMPUTED.meta_features_exist(data_path):
        return
    data = CONTENTS.read_gen_data(data_path)
    d_matrix = pairwise_distances(data)
    meta_features = extract(d_matrix)
    COMPUTED.write_meta_features(data_path, meta_features)

if __name__ == "__main__":
    traverse_data(features)

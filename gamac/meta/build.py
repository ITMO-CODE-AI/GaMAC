"""
Persists pickle binaries for fitted meta-classifier and meta-regressor
"""

import random
from typing import Tuple, List

import numpy as np
from scipy.special import softmax
from sklearn.metrics import pairwise_distances

from gamac.meta.impl.models import ModelProvider
from gamac.meta.storage import GATHERED, traverse_data, COMPUTED, BINARIES

SMOOTH_NEIGHBOURS = 12


def get_smooth_labels(xs, ys, ws):
    dists, labels = pairwise_distances(xs), list()
    m_scores = np.array([softmax(-y) for y in ys])

    for idx, own_m in enumerate(m_scores):
        d_row = dists[idx]
        nearest = np.argsort(d_row)[:SMOOTH_NEIGHBOURS]

        neighbours_d = d_row[nearest]
        d_scores = softmax(-neighbours_d)

        neighbours_w = ws[nearest]
        w_scores = softmax(neighbours_w)

        m_matrix = m_scores[nearest]
        wd_scores = w_scores * d_scores

        label_scores = wd_scores @ m_matrix
        labels.append(np.argmax(label_scores))

    return labels

def shuffle_data(l1, l2):
    spliced = list(zip(l1, l2))
    random.shuffle(spliced)
    u1, u2 = zip(*spliced)
    v1, v2 = list(u1), list(u2)
    return np.array(v1), np.array(v2)


def collect_raw_meta_dataset() -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    pre_meta_dataset = GATHERED.read_pre_meta_dataset()
    meta_features = traverse_data(COMPUTED.read_meta_features)

    xs, ys, ws = list(), list(), list()
    for data_path, data_item in pre_meta_dataset.data.items():
        mfs = meta_features[data_path]
        vals = np.array(data_item.values)
        xs.append(mfs), ys.append(vals)
        ws.append(data_item.weight)
    return pre_meta_dataset._measures_arg, np.array(xs), np.array(ys), np.array(ws)


def build_classifier(ms, xs, ys, ws):
    extractor = ModelProvider.get_best_classifier_extractor()
    classifier = ModelProvider.get_best_xgb_classifier()

    fitted_extractor = extractor.fit(xs)

    x_reduced = fitted_extractor.transform(xs)
    z_smooth = get_smooth_labels(x_reduced, ys, ws)
    x, z = shuffle_data(x_reduced, z_smooth)

    fitted_classifier = classifier.fit(x, z)

    GATHERED.write_classifier_dataset(ms, x, z)
    BINARIES.write_classifier(fitted_extractor, fitted_classifier)


def build_regressor(ms, xs, ys):
    extractor = ModelProvider.get_best_regressor_extractor()
    regressor = ModelProvider.get_best_xgb_regressor()

    fitted_extractor = extractor.fit(xs)

    x_reduced = fitted_extractor.transform(xs)
    x, y = shuffle_data(x_reduced, ys)

    fitted_regressor = regressor.fit(x, y)

    GATHERED.write_regressor_dataset(ms, x, y)
    BINARIES.write_regressor(fitted_extractor, fitted_regressor)


def build_binaries():
    ms, xs, ys, ws = collect_raw_meta_dataset()
    build_classifier(ms, xs, ys, ws)
    build_regressor(ms, xs, ys)


if __name__ == '__main__':
    build_binaries()
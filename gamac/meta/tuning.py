import random
from typing import Tuple

import numpy as np
from scipy.special import softmax
from sklearn.metrics import pairwise_distances, make_scorer
from sklearn.model_selection import cross_validate

from gamac.meta.build import get_smooth_labels, shuffle_data, collect_raw_meta_dataset
from gamac.meta.impl.models import ModelProvider
from gamac.meta.storage import traverse_data, GATHERED, COMPUTED

# SMOOTH_NEIGHBOURS = 12
#
#
# def get_smooth_labels(xs, ys, ws):
#     dists, labels = pairwise_distances(xs), list()
#     m_scores = np.array([softmax(-y) for y in ys])
#
#     for idx, own_m in enumerate(m_scores):
#         d_row = dists[idx]
#         nearest = np.argsort(d_row)[:SMOOTH_NEIGHBOURS]
#
#         neighbours_d = d_row[nearest]
#         d_scores = softmax(-neighbours_d)
#
#         neighbours_w = ws[nearest]
#         w_scores = softmax(neighbours_w)
#
#         m_matrix = m_scores[nearest]
#         wd_scores = w_scores * d_scores
#
#         label_scores = wd_scores @ m_matrix
#         labels.append(np.argmax(label_scores))
#
#     return labels
#
# def shuffle_data(l1, l2):
#     spliced = list(zip(l1, l2))
#     random.shuffle(spliced)
#     u1, u2 = zip(*spliced)
#     v1, v2 = list(u1), list(u2)
#     return np.array(v1), np.array(v2)
#
# def collect_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     pre_meta_dataset = GATHERED.read_pre_meta_dataset()
#     meta_features = traverse_data(COMPUTED.read_meta_features)
#
#     xs, ys, ws = list(), list(), list()
#     for data_path, data_item in pre_meta_dataset.data.items():
#         mfs = meta_features[data_path]
#         vals = np.array(data_item.values)
#         xs.append(mfs), ys.append(vals)
#         ws.append(data_item.weight)
#     return np.array(xs), np.array(ys), np.array(ws)


def tune_classifier(x_orig, y_orig, weights):
    classifiers_stat = list()
    for extractor in ModelProvider.get_extractors():
        ext_model = extractor.algo.fit(x_orig)
        x_reduced = ext_model.transform(x_orig)
        z_smooth = get_smooth_labels(x_reduced, y_orig, weights)
        x, z = shuffle_data(x_reduced, z_smooth)
        for classifier in ModelProvider.get_classifiers():
            classifier_dict = cross_validate(
                classifier.algo, x, z, cv=20, scoring='f1_macro'
            )
            score = np.mean(classifier_dict['test_score'])
            result = (extractor.name, classifier.name, score)
            print(result)
            classifiers_stat.append(result)
    return classifiers_stat


def smape_loss(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    norm = np.abs(y_true) + np.abs(y_pred)
    return (diff / norm).mean()

def tune_regressors(x_orig, y_orig):
    smape_score = make_scorer(smape_loss, greater_is_better=False)
    regressors_stat = list()
    for extractor in ModelProvider.get_extractors():
        ext_model = extractor.algo.fit(x_orig)
        x_reduced = ext_model.transform(x_orig)
        x, y = shuffle_data(x_reduced, y_orig)
        for regressor in ModelProvider.get_regressors():
            regressor_dict = cross_validate(
                regressor.algo, x, y, cv=20, scoring=smape_score
            )
            score = np.mean(regressor_dict['test_score'])
            result = (extractor.name, regressor.name, score)
            print(result)
            regressors_stat.append(result)
    return regressors_stat


def tune():
    _, x_orig, y_orig, weights = collect_raw_meta_dataset()

    print("=== CLASSIFIERS ===")
    tune_classifier(x_orig, y_orig, weights)

    print("=== REGRESSORS ===")
    tune_regressors(x_orig, y_orig)

if __name__ == '__main__':
    tune()

import json
from typing import Dict, List

import numpy as np

from gamac.src.meta.storage import read_orderings, read_partitions, read_gen_data, traverse_data
from sklearn.metrics.cluster import calinski_harabasz_score, silhouette_score, davies_bouldin_score

NUM_ACCESSORS = 5

INTERNAL_MEASURES = [
    ('CH', calinski_harabasz_score),
    ('SIL', silhouette_score),
    ('DB', davies_bouldin_score),
]


def get_measure_orderings(data_path) -> Dict[str, List[int]]:
    data = read_gen_data(data_path)
    partitions = read_partitions(data_path)

    evaluations = dict()
    for measure_name, measure_fun in INTERNAL_MEASURES:
        measure_values = list()
        for partition in partitions:
            measure_value = measure_fun(data, partition)
            measure_values.append(measure_value)
        evaluations[measure_name] = np.argsort(measure_values).tolist()
    return evaluations


def norm_kendall_tau(x_order, y_order):
    n = len(x_order)
    assert len(y_order) == n

    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a, b = np.argsort(x_order), np.argsort(y_order)

    disordered = np.logical_or(
        np.logical_and(a[i] < a[j], b[i] > b[j]),
        np.logical_and(a[i] > a[j], b[i] < b[j])
    ).sum()

    return disordered / (n * (n - 1))


def build_dataset(data_path):
    orderings = read_orderings(data_path, NUM_ACCESSORS)
    measure_orderings = get_measure_orderings(data_path)

    data_result = dict()
    for measure_name, measure_ordering in measure_orderings.items():
        ordering_distances = [
            norm_kendall_tau(measure_ordering, accessor_ordering)
            for accessor_ordering in orderings
        ]
        data_result[measure_name] = np.mean(ordering_distances)

    return data_result


if __name__ == '__main__':
    meta_dataset = traverse_data(build_dataset)
    with open('pre-meta-dataset.json', 'w') as fp:
        json.dump(meta_dataset, fp)

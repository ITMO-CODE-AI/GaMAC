from typing import List

import numpy as np

from gamac.meta.impl.ord import ORDERING_DISTANCE
from gamac.meta.storage import traverse_data, COMPUTED, GATHERED, MARKUPS, AccessorResult
from gamac.meta.storage.gathered import DataOrdering

NUM_ACCESSORS = 5


def filter_accessors(accessors: List[AccessorResult]) -> List[List[int]]:
    n, evicted = len(accessors), set()

    d_matrix = np.empty(shape=(n, n), dtype=float)
    for x_idx, x_res in enumerate(accessors):
        for y_idx, y_res in enumerate(accessors):
            dist = ORDERING_DISTANCE(x_res.sorted_indices, y_res.sorted_indices)
            d_matrix[x_idx, y_idx] = dist

    while len(evicted) < n - NUM_ACCESSORS:
        remained_dists = d_matrix.sum(axis=1)
        most_dissimilar_idx = np.argmax(remained_dists)
        evicted.add(most_dissimilar_idx)
        d_matrix[most_dissimilar_idx, :] = 0
        d_matrix[:, most_dissimilar_idx] = 0

    remained_indices = set(np.arange(n).tolist()) - evicted
    return [
        a_res.sorted_indices for a_idx, a_res in enumerate(accessors)
        if a_idx in remained_indices
    ]


def order_measures(measures):
    return {m_name: np.argsort(m_vals).tolist() for m_name, m_vals in measures.items()}


def compute_orderings():
    pre_values = GATHERED.read_pre_values()
    filtered = {data_path for data_path, pre_val in pre_values.items() if pre_val > 1}

    def prepare_orderings(data_path) -> DataOrdering:
        if data_path not in filtered:
            raise FileNotFoundError()
        accessors = MARKUPS.read_markups(data_path)
        measures = COMPUTED.read_measures(data_path)
        if len(accessors) < NUM_ACCESSORS:
            raise FileNotFoundError()
        return DataOrdering(
            accessors=filter_accessors(accessors),
            measures=order_measures(measures),
        )

    orderings = traverse_data(prepare_orderings)
    GATHERED.write_orderings(orderings)


if __name__ == '__main__':
    compute_orderings()
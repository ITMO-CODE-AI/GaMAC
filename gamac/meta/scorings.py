from typing import Dict, List

import numpy as np

from gamac.meta.impl.ord import ORDERING_DISTANCE
from gamac.meta.storage import GATHERED
from gamac.meta.storage.gathered import DataOrdering, DataScoring


def calc_disagreement(
        accessors: List[List[int]],
) -> float:
    a_dists, n = list(), len(accessors)
    for i in range(1, n):
        for j in range(i):
            a_i, a_j = accessors[i], accessors[j]
            a_dist = ORDERING_DISTANCE(a_i, a_j)
            a_dists.append(a_dist)
    return np.mean(a_dists).__float__()


def calc_measures_scores(orderings: DataOrdering) -> Dict[str, float]:
    return {
        measure_name: np.median([
            ORDERING_DISTANCE(measure_ordering, accessor_ordering)
            for accessor_ordering in orderings.accessors
        ]).__float__()
        for measure_name, measure_ordering in orderings.measures.items()
    }

def compute_scorings():
    orderings, scorings = GATHERED.read_orderings(), dict()
    for data_path, data_ordering in orderings.items():
        scorings[data_path] = DataScoring(
            disagreement=calc_disagreement(data_ordering.accessors),
            measures_score=calc_measures_scores(data_ordering),
        )
    GATHERED.write_scorings(scorings)

if __name__ == '__main__':
    compute_scorings()
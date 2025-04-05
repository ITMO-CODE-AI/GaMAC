"""
Select the top of internal measures w.r.t higher correlation with accessor markups
"""

from typing import Dict, List

import numpy as np
from scipy.special import softmax

from gamac.meta.impl.cvi import INTERNAL_MEASURES
from gamac.meta.storage import GATHERED
from gamac.meta.storage.gathered import DataScoring, PreMetaDataset, PreMetaDataItem

NUM_MEASURES = 4


def estimate_popularity(
        scorings: Dict[str, DataScoring],
        measures_list: List[str]
) -> np.ndarray:
    popularity = np.zeros(shape=len(measures_list))
    for scoring in scorings.values():
        m_scores = scoring.measures_score
        scores = [m_scores[m_name] for m_name in measures_list]
        top_idx = np.argsort(scores)[:NUM_MEASURES]
        popularity[top_idx] += 1
    return popularity / len(scorings) / NUM_MEASURES


def estimate_weights(scorings: Dict[str, DataScoring]) -> Dict[str, float]:
    paths, disagreements = list(), list()
    for data_path, scoring in scorings.items():
        paths.append(data_path)
        disagreements.append(scoring.disagreement)
    ds = np.array(disagreements)
    weights = softmax(-ds / max(ds))
    return {path: weight for path, weight in zip(paths, weights)}


def select_measures(scorings: Dict[str, DataScoring]) -> List[str]:
    m_list = [m_name for m_name, _ in INTERNAL_MEASURES]

    while len(m_list) > NUM_MEASURES:
        rates = estimate_popularity(scorings, m_list)
        del m_list[np.argmin(rates)]

    return m_list


def compute_pre_meta_dataset():
    scorings = GATHERED.read_scorings()
    measures = select_measures(scorings)
    weights = estimate_weights(scorings)

    def values(scoring: DataScoring):
        return [scoring.measures_score[m_name] for m_name in measures]

    pre_meta_dataset = PreMetaDataset(
        measures=measures,
        data={
            data_path: PreMetaDataItem(
                weight=weights[data_path],
                values=values(scoring),
            )
            for data_path, scoring in scorings.items()
        }
    )

    GATHERED.write_pre_meta_dataset(pre_meta_dataset)


if __name__ == '__main__':
    compute_pre_meta_dataset()
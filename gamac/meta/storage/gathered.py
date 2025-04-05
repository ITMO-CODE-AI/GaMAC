"""
Interface to handle aposteriori datasets characteristics storage
"""

import json
from typing import List, Dict

import pandas as pd


class DataOrdering:
    def __init__(
            self,
            accessors: List[List[int]],
            measures: Dict[str, List[int]]
    ):
        self.accessors = accessors
        self.measures = measures


class DataScoring:
    def __init__(
            self,
            disagreement: float,
            measures_score: Dict[str, float]
    ):
        self.disagreement = disagreement
        self.measures_score = measures_score


class PreMetaDataItem:
    def __init__(self, weight: float, values: List[float]):
        self.weight = weight
        self.values = values


class PreMetaDataset:
    def __init__(
            self,
            measures: List[str],
            data: Dict[str, PreMetaDataItem]
    ):
        self.measures = measures
        self.data = data


class Gathered:
    def __init__(self, common_root):
        self.common_root = common_root

    @property
    def pre_value_path(self):
        return f'{self.common_root}/pre-value.csv'

    @property
    def orderings_path(self):
        return f'{self.common_root}/orderings.json'

    @property
    def scorings_path(self):
        return f'{self.common_root}/scorings.json'

    @property
    def pre_meta_path(self):
        return f'{self.common_root}/pre-meta.json'

    @property
    def classifier_data_path(self):
        return f'{self.common_root}/classifier.csv'

    @property
    def regressor_data_path(self):
        return f'{self.common_root}/regressor.csv'

    def write_orderings(self, orderings: Dict[str, DataOrdering]):
        content = dict()
        for data_path, ordering in orderings.items():
            accessors_ords = [
                str(a_ord) for a_ord in ordering.accessors
            ]
            measures_ord = {
                m_name: str(m_ord) for m_name, m_ord in ordering.measures.items()
            }
            content[data_path] = {
                'accessors': accessors_ords,
                'measures': measures_ord,
            }
        with open(self.orderings_path, "w") as fp:
            json.dump(content, fp)

    def read_orderings(self) -> Dict[str, DataOrdering]:
        with open(self.orderings_path, "r") as fp:
            content = json.load(fp)

        orderings = dict()
        for data_path, ordering in content.items():
            accessors_ords = [
                eval(a_ord) for a_ord in ordering['accessors']
            ]
            measures_ords = {
                m_name: eval(m_ord) for m_name, m_ord in ordering['measures'].items()
            }
            orderings[data_path] = DataOrdering(
                accessors=accessors_ords,
                measures=measures_ords,
            )
        return orderings

    def write_scorings(self, scorings: Dict[str, DataScoring]):
        with open(self.scores_path, 'w') as fp:
            content = {
                data_path: {
                    'disagreement': scoring.disagreement,
                    'measures_score': scoring.measures_score
                } for data_path, scoring in scorings.items()
            }
            json.dump(content, fp)

    def read_scorings(self) -> Dict[str, DataScoring]:
        with open(self.scores_path, 'r') as fp:
            content = json.load(fp)
            return {
                data_path: DataScoring(
                    disagreement=scoring['disagreement'],
                    measures_score=scoring['measures_score']
                ) for data_path, scoring in content.items()
            }

    def write_pre_meta_dataset(self, pre_meta_dataset: PreMetaDataset):
        with open(self.pre_meta_path, 'w') as fp:
            content = {
                'measures': pre_meta_dataset.measures,
                'data': {
                    data_path: {
                        'weight': item.weight,
                        'values': item.values

                    }
                    for data_path, item in pre_meta_dataset.data.items()
                }
            }
            json.dump(content, fp)

    def read_pre_meta_dataset(self) -> PreMetaDataset:
        with open(self.pre_meta_path, 'r') as fp:
            content = json.load(fp)
            return PreMetaDataset(
                measures=content['measures'],
                data={
                    data_path: PreMetaDataItem(
                        weight=item['weight'],
                        values=item['values']
                    )
                    for data_path, item in content['data'].items()
                }
            )

    def read_pre_values(self) -> Dict[str, int]:
        content = pd.read_csv(self.pre_value_path).values
        return {
            data_path: score for data_path, score in content
        }

    def write_classifier_dataset(self, measures, x, z):
        x_col = [f'mf_{idx}' for idx in range(x.shape[1])]

        class_df = pd.DataFrame(data=x, columns=x_col)
        class_df['measure'] = [measures[m_idx] for m_idx in z]

        class_df.to_csv(self.classifier_data_path, index=False)

    def write_regressor_dataset(self, measures, x, y):
        x_col = [f'mf_{idx}' for idx in range(x.shape[1])]
        reg_df = pd.DataFrame(data=x, columns=x_col)

        for m_idx, m_name in enumerate(measures):
            reg_df[m_name] = y[:, m_idx]

        reg_df.to_csv(self.regressor_data_path, index=False)

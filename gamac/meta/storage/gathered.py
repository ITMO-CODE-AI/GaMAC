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

class Gathered:
    def __init__(self, common_root):
        self.common_root = common_root

        self.orderings_path = f'{common_root}/orderings.json'
        self.scores_path = f'{common_root}/scores.json'
        self.pre_meta_path = f'{common_root}/pre-meta.json'
        self.classifier_data_path = f'{common_root}/classifier.csv'
        self.regressor_data_path = f'{common_root}/regressor.csv'

        self.pre_value_path = f'{common_root}/pre-value.csv'

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

    def write_scores(self, scores):
        with open(self.scores_path, 'w') as fp:
            json.dump(scores, fp)

    def read_scores(self):
        with open(self.scores_path, 'r') as fp:
            return json.load(fp)

    def write_pre_meta(self, pre_meta_items):
        with open(self.pre_meta_path, 'w') as fp:
            json.dump(pre_meta_items, fp)

    def read_pre_values(self) -> Dict[str, int]:
        content = pd.read_csv(self.pre_value_path).values
        return {
            data_path: score for data_path, score in content
        }
        # pre_val_dict = dict()
        # with open(PRE_VALUE_PATH) as fp:
        #     for line in fp.readlines():
        #         data_path, score = line.split(",")
        #         pre_val_dict[data_path] = int(score.replace("\n", ""))
        #
        # return pre_val_dict

    def write_classifier_dataset(self, measures, X, Z):
        x_col = [f'mf_{idx}' for idx in range(X.shape[1])]

        class_df = pd.DataFrame(data=X, columns=x_col)
        class_df['measure'] = [measures[z] for z in Z]

        class_df.to_csv(self.classifier_data_path, index=False)

    def write_regressor_dataset(self, measures, X, Y):
        x_col = [f'mf_{idx}' for idx in range(X.shape[1])]
        reg_df = pd.DataFrame(data=X, columns=x_col)

        for m_idx, m_name in enumerate(measures):
            reg_df[m_name] = Y[:, m_idx]

        reg_df.to_csv(self.regressor_data_path, index=False)

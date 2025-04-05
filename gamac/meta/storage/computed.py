"""
Interface to handle apriori dataset characteristics storage
"""

import json
import os
from typing import Dict, List

import numpy as np

MeasuresType = Dict[str, List[float]]

class Computed:
    def __init__(self, data_root):
        self.data_root = data_root

    def features_path(self, data_path):
        return f'{self.data_root}/{data_path}/features.txt'

    def measures_path(self, data_path):
        return f'{self.data_root}/{data_path}/measures.json'

    def read_meta_features(self, data_path) -> np.ndarray:
        with open(self.features_path(data_path), 'r') as fp:
            return np.array(
                eval(
                    fp.readline()
                )
            )

    def meta_features_exist(self, data_path) -> bool:
        return os.path.exists(
            self.features_path(data_path)
        )

    def write_meta_features(self, data_path, features: np.ndarray):
        with open(self.features_path(data_path), 'w') as fp:
            fp.write(
                str(
                    features.tolist()
                )
            )

    def read_measures(self, data_path) -> MeasuresType:
        with open(self.measures_path(data_path), 'r') as fp:
            content = json.load(fp)

        return {
            m_name: eval(m_val_str)
            for m_name, m_val_str in content.items()
        }

    def write_measures(self, data_path, measures: MeasuresType):
        content = {
            m_name: str(m_values) for m_name, m_values in measures.items()
        }
        with open(self.measures_path(data_path), 'w') as fp:
            json.dump(content, fp)

    def measures_exists(self, data_path) -> bool:
        return os.path.exists(
            self.measures_path(data_path)
        )

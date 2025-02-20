import json
import os
from typing import List

import numpy as np
import pandas as pd

class Contents:
    def __init__(self, data_root):
        self.data_root = data_root

    def image_path(self, data_path, partition_idx):
        return f'{self.data_root}/{data_path}/images/{partition_idx}.png'

    def gen_data_path(self, data_path):
        return f'{self.data_root}/{data_path}/gen.csv'

    def partitions_path(self, data_path):
        return f'{self.data_root}/{data_path}/partitions.csv'

    def producers_path(self, data_path):
        return f'{self.data_root}/{data_path}/producers.json'

    def read_partitions(self, data_path) -> np.ndarray:
        return pd.read_csv(
            self.partitions_path(data_path), header=None
        ).values

    def read_gen_data(self, data_path) -> np.ndarray:
        return pd.read_csv(
            self.gen_data_path(data_path), header=None
        ).values

    def write_partitions(self, data_path, partitions: List[np.ndarray]):
        pd.DataFrame(partitions).to_csv(
            self.partitions_path(data_path), header=False, index=False
        )

    def write_producers(self, data_path, producers):
        with open(self.producers_path(data_path), 'w') as fp:
            json.dump(producers, fp)

    def write_gen_data(self, data_path, gen_data: np.ndarray):
        pd.DataFrame(data=gen_data).to_csv(
            self.gen_data_path(data_path), header=False, index=False
        )

    def create_data_dir(self, data_path):
        if not os.path.exists(f'{self.data_root}/{data_path}'):
            os.mkdir(f'{self.data_root}/{data_path}')

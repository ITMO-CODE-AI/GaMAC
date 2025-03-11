import json
import os
from typing import List

import numpy as np
import pandas as pd

DATA_ROOT = "data"
ORIG_DATA_FILE = "orig.csv"
GEN_DATA_FILE = "gen.csv"
PARTITIONS_FILE = "partitions.csv"
FEATURES_FILE = "features.txt"
PRODUCERS_FILE = "producers.json"
VISUAL_ROOT = 'meta-visual'

PARTITIONS_TO_ESTIMATE = 15

def accessor_file(data_path, idx):
    return f'{DATA_ROOT}/{data_path}/accessor-{idx}.txt'

def traverse_data(handler):
    result, datasets = dict(), os.listdir(DATA_ROOT)
    for dataset in sorted(datasets):
        reducers = os.listdir(f'{DATA_ROOT}/{dataset}')
        for reducer in sorted(reducers):
            if reducer == ORIG_DATA_FILE:
                continue
            data_path = f'{dataset}/{reducer}'
            try:
                result[data_path] = handler(data_path)
                print(f'HANDLED {data_path}')
            except FileNotFoundError:
                print(f'SKIPPED {data_path}')
    return result

def read_orderings(data_path, num_accessors) -> List[List[int]]:
    orderings = list()
    for accessor_idx in range(num_accessors):
        accessor_path = accessor_file(data_path, accessor_idx)
        with open(accessor_path, 'r') as fp:
            content = fp.readline()
            ordering = eval(content)
            orderings.append(ordering)
    return orderings

def read_meta_features(data_path) -> List[float]:
    with open(f'{DATA_ROOT}/{data_path}/{FEATURES_FILE}', 'r') as fp:
        content = fp.readline()
        return eval(content)

def meta_features_exist(data_path) -> bool:
    features_path = f'{DATA_ROOT}/{data_path}/{FEATURES_FILE}'
    return os.path.exists(features_path)

def write_meta_features(data_path, features: np.ndarray):
    features_path = f'{DATA_ROOT}/{data_path}/{FEATURES_FILE}'
    with open(features_path, 'w') as fp:
        fp.write(str(features.tolist()))

def read_partitions(data_path) -> List[List[int]]:
    partitions_path = f'{DATA_ROOT}/{data_path}/{PARTITIONS_FILE}'
    content = pd.read_csv(partitions_path, header=None)
    return content.values.tolist()

def read_gen_data(data_path) -> np.ndarray:
    gen_data_path = f'{DATA_ROOT}/{data_path}/{GEN_DATA_FILE}'
    return pd.read_csv(gen_data_path, header=None).values

def write_partitions(data_path, partitions: List[np.ndarray]):
    partitions_path = f'{DATA_ROOT}/{data_path}/{PARTITIONS_FILE}'
    pd.DataFrame(partitions).to_csv(partitions_path, header=False, index=False)

def write_producers(data_path, producers):
    producers_path = f'{DATA_ROOT}/{data_path}/{PRODUCERS_FILE}'
    with open(producers_path, 'w') as fp:
        json.dump(producers, fp)

def write_gen_data(data_path, gen_data: np.ndarray):
    gen_data_path = f'{DATA_ROOT}/{data_path}/{GEN_DATA_FILE}'
    pd.DataFrame(data=gen_data).to_csv(gen_data_path, header=False, index=False)

def create_data_dir(data_path):
    if not os.path.exists(f'{DATA_ROOT}/{data_path}'):
        os.mkdir(f'{DATA_ROOT}/{data_path}')

def create_visual_dir():
    if not os.path.exists(VISUAL_ROOT):
        os.mkdir(VISUAL_ROOT)

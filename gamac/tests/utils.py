import cupy as cp
import numpy as np
import pandas as pd
import json
import os

def _read_data(data_path) -> cp.ndarray:
    data = pd.read_csv(f'{data_path}/gen.csv', header=None).values
    return cp.asarray(data, dtype=cp.float32, order='C')


def _read_partitions(data_path) -> list[cp.ndarray]:
    partitions = pd.read_csv(f'{data_path}/partitions.csv', header=None).values
    return [
        cp.asarray(partition, dtype=cp.int32)
        for partition in partitions
    ]


def _read_measures(data_path) -> dict[str, list[float]]:
    with open(f'{data_path}/measures.json') as fp:
        content = json.load(fp)
    return {
        measure_name: eval(values_str)
        for measure_name, values_str in content.items()
    }

def _read_features(data_path) -> np.ndarray:
    with open(f'{data_path}/features.txt') as fp:
        content = fp.readline()
    features_list = eval(content)
    return np.array(features_list)

def traverse_data():
    for data_name in sorted(os.listdir('data')):
        data_path = f'data/{data_name}'
        yield {
            "name": data_name,
            "data": _read_data(data_path),
            "partitions": _read_partitions(data_path),
            "measures": _read_measures(data_path),
            "features": _read_features(data_path),
        }
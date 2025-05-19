import json
import os
from unittest import TestCase

import cupy as cp
import pandas as pd
from parameterized import parameterized

from gamac.estimation.internal import InternalEvaluator, Internal


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


def traverse_data():
    for data_name in sorted(os.listdir('cvi-test')):
        data_path = f'cvi-test/{data_name}'
        data = _read_data(data_path)
        partitions = _read_partitions(data_path)
        measures = _read_measures(data_path)
        yield data_name, data, partitions, measures


class MeasuresTest(TestCase):
    @parameterized.expand([
        ['BR', 7e-9],
        ['OS', 7e-2],
        ['MCR', 7e-4],
        ['SYM', 7e-6],
    ])
    def test_internal(self, measure_name, delta):
        measure = Internal[measure_name]
        for data_name, data, partitions, measures in traverse_data():
            expected = measures[measure_name]
            fake_pivots = {measure: 0.0}
            evaluator = InternalEvaluator(data, fake_pivots)
            for idx, partition in enumerate(partitions):
                result = evaluator.evaluate(partition)[measure]
                self.assertAlmostEqual(expected[idx], result, delta=delta)

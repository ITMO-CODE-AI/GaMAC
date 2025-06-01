import time
from enum import Enum
from typing import Set, List

import cupy as cp
import numpy as np

from gamac.data.data_pipeline import DataFrameType, LabelsType
from gamac.estimation.container import EstimationContainer
from gamac.estimation.functions import br, mcr, sym, os


class Internal(Enum):
    BR = ('banfield_raftery', br)
    # C_INDEX = ('c_index', c_index)
    OS = ('os', os)
    MCR = ('mc_clain_rao', mcr)
    SYM = ('sym', sym)


EstimationResult = dict[Internal, float]


class InternalEvaluator:
    def __init__(self, df: DataFrameType, pivots: EstimationResult):
        self.df, self.pivots = df, pivots

    @staticmethod
    def create(df: DataFrameType, measures: Set[Internal]):
        random_labels = cp.random.randint(low=0, high=2, size=len(df), dtype=np.int32)
        pivots = InternalEvaluator.eval_internal(df, measures, random_labels)
        return InternalEvaluator(df, pivots)

    @staticmethod
    def create_container(df: DataFrameType, labels: LabelsType):
        return EstimationContainer.create(df, labels)

    def evaluate(self, labels: LabelsType) -> EstimationResult:
        measures = set(self.pivots.keys())
        return self.eval_internal(self.df, measures, labels)

    def rewards(self, estimates: List[EstimationResult]) -> List[float]:
        return [
            np.mean([
                self.compare(x, y) for y in estimates
            ]).__float__()
            for x in estimates
        ]

    def compare(self, x: EstimationResult, y: EstimationResult) -> float:
        return sum([
            self._compare_sym(x[measure], y[measure], pivot)
            for measure, pivot in self.pivots.items()
        ])

    def is_better(self, x: EstimationResult, y: EstimationResult) -> bool:
        return self.compare(x, y) > self.compare(y, x)

    @staticmethod
    def _compare_sym(x: float, y: float, pivot: float) -> float:
        normaliser = max(x, y) - pivot
        if normaliser < 1e-6:
            return 0.0
        return (x - y) / normaliser

    @staticmethod
    def eval_internal(df: DataFrameType, measures: Set[Internal], labels: LabelsType) -> EstimationResult:
        t_start = time.time()
        container = EstimationContainer.create(df, labels)
        values = {
            measure: measure.value[1](container) for measure in measures
        }
        serialised = {m.name: float(v) for m, v in values.items()}
        print(f"=== MEASURES {time.time() - t_start}s, {serialised} ===")
        return EstimationResult(values)


from enum import Enum
from typing import Dict, Set, List

import numpy as np
import cupy as cp

from gamac.estimation.container import EstimationContainer
from gamac.estimation.functions import br, c_index, mcr, sym
from gamac.data.data_pipeline import DataFrameType, LabelsType


class Internal(Enum):
    BR = ('banfield_raftery', br)
    C_INDEX = ('c_index', c_index)
    MCR = ('mc_clain_rao', mcr)
    SYM = ('sym', sym)


EstimationResult = dict[Internal, float]


class InternalEvaluator:
    def __init__(self, df: DataFrameType, pivots: EstimationResult):
        self.df, self.pivots = df, pivots

    @staticmethod
    def create(df: DataFrameType, measures: Set[Internal]):
        random_labels = cp.random.randint(low=0, high=2, size=len(df), dtype=np.int32)
        pivots = InternalEvaluator._eval_internal(df, measures, random_labels)
        return InternalEvaluator(df, pivots)

    @staticmethod
    def create_container(df: DataFrameType, labels: LabelsType):
        return EstimationContainer.create(df, labels)

    def evaluate(self, labels: LabelsType) -> EstimationResult:
        measures = set(self.pivots.keys())
        return self._eval_internal(self.df, measures, labels)

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
        return (x - y) / (max(x, y) - pivot)

    @staticmethod
    def _eval_internal(df: DataFrameType, measures: Set[Internal], labels: LabelsType) -> EstimationResult:
        container = EstimationContainer.create(df, labels)
        values = {
            measure: measure.value[1](container) for measure in measures
        }
        return EstimationResult(values)


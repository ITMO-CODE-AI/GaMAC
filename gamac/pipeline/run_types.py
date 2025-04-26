from abc import ABC, abstractmethod

from typing import Dict, Any

from gamac.algorithms.base import ClusteringAlgo, ClusteringModel
from gamac.estimation.internal import EstimationResult
from gamac.data.data_pipeline import LabelsType


class HistoryRun(ABC):
    def __init__(self, algo_name: str, algo_params: Dict[str, Any]):
        self.algo_name = algo_name
        self.algo_params = algo_params

    @abstractmethod
    def elapsed(self) -> float:
        pass


class FailedRun(HistoryRun):
    def __init__(
        self,
        algo_name: str,
        algo_params: Dict[str, Any],
        consumed: float,
    ):
        super().__init__(algo_name, algo_params)
        self.consumed = consumed

    def elapsed(self) -> float:
        return self.consumed


class SuccessRun(HistoryRun):
    def __init__(
        self,
        algo_name: str,
        algo_params: Dict[str, Any],
        fit_time: float,
        eval_time: float,
        estimation: EstimationResult
    ):
        super().__init__(algo_name, algo_params)
        self.fit_time, self.eval_time = fit_time, eval_time
        self.estimation = estimation

    def elapsed(self) -> float:
        return self.fit_time + self.eval_time


class Optimal:
    def __init__(
        self,
        algo: ClusteringAlgo,
        model: ClusteringModel,
        labels: LabelsType,
        estimation: EstimationResult
    ):
        self.algo, self.model = algo, model
        self.labels = labels
        self.estimation = estimation

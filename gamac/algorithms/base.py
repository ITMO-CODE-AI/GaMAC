import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from gamac.data.data_pipeline import DataFrameType, LabelsType


class ClusteringModel(ABC):
    @abstractmethod
    def predict(self, df: DataFrameType) -> LabelsType:
        pass


class ClusteringAlgo(ABC):
    def __init__(self, **kwargs):
        self.name, self.params = self.__class__.__name__, kwargs

    @abstractmethod
    def fit(self, df: DataFrameType) -> ClusteringModel:
        pass

    @abstractmethod
    def fit_predict(self, df: DataFrameType) -> LabelsType:
        return self.fit(df).predict(df)

    @staticmethod
    def make_seed(seed: Optional[int]) -> int:
        return seed if seed is not None else random.randint(0, 2 ** 32)



class AlgoConf(ABC):
    def __init__(self, builder, **kwargs):
        self.config_space, self.builder = kwargs, builder
        self.algo_name = builder.__name__

    def build(self, **kwargs) -> ClusteringAlgo:
        return self.builder(**kwargs)

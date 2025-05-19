import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from gamac.data.data_pipeline import DataFrameType, LabelsType


class ClusteringModel(ABC):
    def __init__(self, labels_: LabelsType):
        self.labels_ = labels_

    @abstractmethod
    def predict(self, df: DataFrameType) -> LabelsType:
        pass


class ClusteringAlgo(ABC):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__

    @abstractmethod
    def fit(self, df: DataFrameType) -> ClusteringModel:
        pass

    def fit_predict(self, df: DataFrameType) -> LabelsType:
        return self.fit(df).predict(df)

    @staticmethod
    def make_seed(seed: Optional[int]) -> int:
        return seed if seed is not None else random.randint(0, 2 ** 32)



class AlgoConfig(ABC):
    def __init__(self, builder, **kwargs):
        self.config_space, self.builder = kwargs, builder
        self.algo_name = builder.__name__

    def build(self, **kwargs) -> ClusteringAlgo:
        return self.builder(**kwargs)

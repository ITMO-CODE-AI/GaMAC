import numpy as np

from abc import abstractmethod, ABC
from typing import Any, Set

from gamac.data.data_pipeline import DataFrameType


class ConfigSampler(ABC):
    def __init__(self, df: DataFrameType):
        self.df = df

    def suggest(self, name, param):
        if isinstance(param, tuple) and len(param) == 2:
            return self._range_param(name, param)
        elif isinstance(param, (set, frozenset)):
            return self._categorical_param(name, param)
        elif callable(param) or isinstance(param, (int, float, bool)):
            return self._calc(param)
        else:
            raise ValueError(f"Failed to recognize parameter '{name}': {param}")

    def _range_param(self, name, param):
        lower, upper = self._calc(param[0]), self._calc(param[1])
        if isinstance(lower, int) and isinstance(upper, int):
            return self._int_param(name, lower, upper)
        elif isinstance(lower, float) and isinstance(upper, float):
            return self._float_param(name, lower, upper)
        else:
            raise ValueError(f"Expected both ints or floats (hyper parameter '{name}')")

    def _calc(self, param):
        return param(self.df) if callable(param) else param

    @abstractmethod
    def _int_param(self, name: str, lower: int, upper: int) -> int:
        pass

    @abstractmethod
    def _float_param(self, name: str, lower: float, upper: float) -> float:
        pass

    @abstractmethod
    def _categorical_param(self, name: str, values: Set[Any]) -> Any:
        pass


class RandomSampler(ConfigSampler):
    def __init__(self, df: DataFrameType):
        super().__init__(df)

    def _int_param(self, name: str, lower: int, upper: int) -> int:
        return np.random.randint(lower, upper)

    def _float_param(self, name: str, lower: float, upper: float) -> float:
        return (np.random.random() + lower) * (upper - lower)

    def _categorical_param(self, name: str, values: Set[Any]) -> Any:
        index = np.random.randint(0, len(values))
        return list(values)[index]


class OptunaSampler(ConfigSampler):
    def __init__(self, df: DataFrameType, trial):
        super().__init__(df)

        from optuna.trial import BaseTrial
        self.trial: BaseTrial = trial

    def _int_param(self, name: str, lower: int, upper: int) -> int:
        return self.trial.suggest_int(name, lower, upper)

    def _float_param(self, name: str, lower: float, upper: float) -> float:
        return self.trial.suggest_float(name, lower, upper)

    def _categorical_param(self, name: str, values: Set[Any]) -> Any:
        as_list = list(values)
        return self.trial.suggest_categorical(name, as_list)

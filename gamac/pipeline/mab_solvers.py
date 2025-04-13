import time
from abc import abstractmethod, ABC
from enum import Enum
from typing import List, Optional

import numpy as np

from gamac.estimation.internal import InternalEvaluator, EstimationResult
from gamac.data.data_pipeline import DataFrameType
from gamac.pipeline.hyper_optimisers import HyperOptimiser
from gamac.pipeline.run_types import Optimal


class MabSolver(ABC):
    def __init__(
            self,
            df: DataFrameType,
            optimisers: List[HyperOptimiser],
            evaluator: InternalEvaluator,
    ):
        self.df = df
        self.optimisers = optimisers
        self.evaluator = evaluator

        self.arms = len(optimisers)
        self.consumption = np.zeros(self.arms)
        self.usage = np.zeros(self.arms, dtype=int)

        self.history = list()
        self._time_limit, self._iter_limit = 0, 0

    def run(self, time_limit, iter_limit) -> Optimal:
        self._time_limit, self._iter_limit = time_limit, iter_limit

        self._warmup_if_needed()
        self._run_main_loop()

        best_arm = np.argmax(self._quality_rewards())
        return self.optimisers[best_arm].optimal

    @abstractmethod
    def pick_arm(self) -> int:
        pass

    def _step(self, arm):
        t_start = time.time()
        run = self.optimisers[arm].step()
        arm_time = time.time() - t_start

        self.consumption[arm] += arm_time
        self.usage[arm] += 1
        self.history.append(run)

        self._consume(arm_time)

    def _is_warmup_required(self):
        return np.min(self.usage) < HyperOptimiser.WARMUP_TRIALS or self._arms_initialised() == 0

    def _warmup_if_needed(self):
        while not self._budget_exhausted() and self._is_warmup_required():
            arm = np.argmin(self.usage)
            self._step(arm)

    def _run_main_loop(self):
        while not self._budget_exhausted():
            arm = self.pick_arm()
            self._step(arm)

    def _budget_exhausted(self):
        time_exhausted = self._time_limit is not None and self._time_limit <= 0
        iter_exhausted = self._iter_limit is not None and self._iter_limit <= 0
        return time_exhausted or iter_exhausted

    def _consume(self, time_consumed):
        if self._time_limit is not None:
            self._time_limit -= time_consumed
        if self._iter_limit is not None:
            self._iter_limit -= 1

    def _opts(self) -> List[Optional[Optimal]]:
        return [optimiser.optimal for optimiser in self.optimisers]

    def _arms_initialised(self) -> int:
        return sum([0 if opt is None else 1 for opt in self._opts()])

    def _safe_compare(self, x: Optional[Optimal], y: Optional[Optimal], fake: EstimationResult):
        x_or_fake = x.estimation if x is not None else fake
        y_or_fake = y.estimation if y is not None else fake
        return self.evaluator.compare(x_or_fake, y_or_fake)

    def _std_norm(self, rewards):
        mean, std = np.mean(rewards), np.std(rewards)
        diff = np.array(rewards) - mean
        return 0.0 if std < 1e-5 else diff / std

    def _quality_rewards(self) -> List[float]:
        fake_estimation = self.evaluator.pivots
        safe_estimations = [opt.estimation if opt is not None else fake_estimation for opt in self._opts()]
        return self.evaluator.rewards(safe_estimations)

    def _time_rewards(self):
        return 1 - self.consumption / sum(self.consumption)

    def _rewards(self) -> np.ndarray:
        return self._std_norm(self._quality_rewards()) + self._std_norm(self._time_rewards())


class RandomMab(MabSolver):
    def __init__(
            self,
            df: DataFrameType,
            optimisers: List[HyperOptimiser],
            evaluator: InternalEvaluator
    ):
        super().__init__(df, optimisers, evaluator)

    def pick_arm(self) -> int:
        return np.random.choice(np.arange(self.arms))


class SoftmaxMab(MabSolver):
    def __init__(
            self,
            df: DataFrameType,
            optimisers: List[HyperOptimiser],
            evaluator: InternalEvaluator
    ):
        super().__init__(df, optimisers, evaluator)

    def pick_arm(self) -> int:
        rewards = self._rewards()
        exp_rewards = np.exp(rewards - rewards.max())
        probs = exp_rewards / exp_rewards.sum()
        return np.random.choice(np.arange(self.arms), p=probs)


class MabSolvers(Enum):
    RANDOM = RandomMab
    SOFTMAX = SoftmaxMab

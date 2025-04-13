import time
from abc import abstractmethod, ABC
from enum import Enum
from typing import List

from gamac.algorithms.base import AlgoConf, ClusteringAlgo
from gamac.estimation.internal import InternalEvaluator, EstimationResult
from gamac.data.data_pipeline import DataFrameType
from gamac.pipeline.run_types import SuccessRun, HistoryRun, Optimal, FailedRun
from gamac.pipeline.config_samplers import ConfigSampler, OptunaSampler, RandomSampler


class HyperOptimiser(ABC):
    WARMUP_TRIALS = 3

    def __init__(
        self,
        algo_conf: AlgoConf,
        df: DataFrameType,
        evaluator: InternalEvaluator
    ):
        self.algo_conf = algo_conf
        self.df = df
        self.evaluator = evaluator
        self._runs, self.optimal = list(), None

    @property
    def success_runs(self) -> List[SuccessRun]:
        return [run for run in self._runs if isinstance(run, SuccessRun)]

    def step(self) -> HistoryRun:
        config_sampler = self._config_sampler()
        config = {
            name: config_sampler.suggest(name, space)
            for name, space in self.algo_conf.config_space.items()
        }

        algo = self.algo_conf.build(**config)
        step_result = self._eval_algo(algo)

        self._post_step_hook(step_result)
        self._runs.append(step_result)

        return step_result

    @abstractmethod
    def _config_sampler(self) -> ConfigSampler:
        pass

    def _post_step_hook(self, result: HistoryRun):
        pass

    def _is_optimal(self, estimation: EstimationResult) -> bool:
        return self.optimal is None or estimation > self.optimal.estimation

    def _eval_algo(self, algo: ClusteringAlgo) -> HistoryRun:
        algo_start = time.time()
        try:
            model, labeled_sdf = algo.fit_predict_with_model(self.df)
            fit_time = time.time() - algo_start
            estimation = self.evaluator.evaluate(labeled_sdf)
            eval_time = time.time() - fit_time

            if self._is_optimal(estimation):
                self.optimal = Optimal(algo, model, labeled_sdf, estimation)

            return SuccessRun(algo.name, algo.params, fit_time, eval_time, estimation)
        except RuntimeError:
            failed_time = time.time() - algo_start
            return FailedRun(algo.name, algo.params, failed_time)


class RandomOptimiser(HyperOptimiser):
    def __init__(
            self,
            algo_conf: AlgoConf,
            df: DataFrameType,
            evaluator: InternalEvaluator
    ):
        super().__init__(algo_conf, df, evaluator)

    def _config_sampler(self) -> ConfigSampler:
        return RandomSampler(self.df)


class OptunaOptimiser(HyperOptimiser):
    def __init__(
        self,
        algo_conf: AlgoConf,
        df: DataFrameType,
        evaluator: InternalEvaluator
    ):
        super().__init__(algo_conf, df, evaluator)

        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self._session = self._new_study()

    @staticmethod
    def _new_study():
        import optuna
        from optuna.samplers import TPESampler

        optuna_sampler = TPESampler(n_startup_trials=HyperOptimiser.WARMUP_TRIALS)
        return optuna.create_study(direction='maximize', sampler=optuna_sampler)

    def _value_for_run(self, run: HistoryRun):
        if isinstance(run, FailedRun):
            return -float('inf')
        assert isinstance(run, SuccessRun)

        return sum([
            self.evaluator.compare(run.estimation, other.estimation)
            for other in self.success_runs
        ])

    def _config_sampler(self) -> ConfigSampler:
        trial = self._session.ask()
        return OptunaSampler(self.df, trial)

    def _post_step_hook(self, result: HistoryRun):
        import optuna

        trials = self._session.get_trials(deepcopy=False)
        values = list(map(self._value_for_run, self._runs))

        recalculated_trials = [
            optuna.trial.create_trial(
                params=trial.params,
                distributions=trial.distributions,
                value=value
            ) for trial, value in zip(trials, values)
        ]

        self.session = self._new_study()
        self.session.add_trials(recalculated_trials)


class HyperOptimisers(Enum):
    RANDOM = RandomOptimiser
    OPTUNA = OptunaOptimiser

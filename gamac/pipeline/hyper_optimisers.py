import time
from abc import abstractmethod, ABC
from enum import Enum
from typing import List

from gamac.algorithms.base import AlgoConfig, ClusteringAlgo
from gamac.estimation.internal import InternalEvaluator, EstimationResult
from gamac.data.data_pipeline import DataFrameType
from gamac.pipeline.run_types import SuccessRun, HistoryRun, Optimal, FailedRun
from gamac.pipeline.config_samplers import ConfigSampler, OptunaSampler, RandomSampler


class HyperOptimiser(ABC):
    WARMUP_TRIALS = 3

    def __init__(
        self,
        algo_conf: AlgoConfig,
        df: DataFrameType,
        evaluator: InternalEvaluator
    ):
        self.algo_conf = algo_conf
        self.df = df
        self.evaluator = evaluator
        self._runs = list()
        self.optimal = None

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

        self._runs.append(step_result)
        self._post_step_hook(step_result)

        return step_result

    @abstractmethod
    def _config_sampler(self) -> ConfigSampler:
        pass

    def _post_step_hook(self, result: HistoryRun):
        pass

    def _is_optimal(self, estimation: EstimationResult) -> bool:
        return (
                self.optimal is None or
                self.evaluator.is_better(estimation, self.optimal.estimation)
        )

    def _eval_algo(self, algo: ClusteringAlgo) -> HistoryRun:
        start_timestamp = time.time()
        try:
            print('ALGO', algo)
            model = algo.fit(self.df)
            labels = model.labels_
            fit_timestamp = time.time()

            estimation = self.evaluator.evaluate(labels)
            eval_timestamp = time.time()

            if self._is_optimal(estimation):
                self.optimal = Optimal(algo, model, estimation)

            print(f"=== ALGO {fit_timestamp - start_timestamp} ===")
            return SuccessRun(
                algo_params=algo.__dict__,
                fit_time=fit_timestamp - start_timestamp,
                eval_time=eval_timestamp - fit_timestamp,
                estimation=estimation
            )
        except RuntimeError:
            failed_time = time.time() - start_timestamp
            return FailedRun(
                algo_params=algo.__dict__,
                consumed=failed_time
            )


class RandomOptimiser(HyperOptimiser):
    def __init__(
            self,
            algo_conf: AlgoConfig,
            df: DataFrameType,
            evaluator: InternalEvaluator
    ):
        super().__init__(algo_conf, df, evaluator)

    def _config_sampler(self) -> ConfigSampler:
        return RandomSampler(self.df)


class OptunaOptimiser(HyperOptimiser):
    def __init__(
        self,
        algo_conf: AlgoConfig,
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

        assert len(trials) == len(values)

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

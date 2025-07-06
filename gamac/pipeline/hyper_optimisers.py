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
    """Абстрактный базовый класс для оптимизации гиперпараметров кластеризации.
    
    Реализует общий процесс подбора параметров и оценки качества кластеризации.
    
    Атрибуты:
        WARMUP_TRIALS (int): Количество начальных пробных запусков
        algo_conf (AlgoConfig): Конфигурация алгоритма кластеризации
        df (DataFrameType): Данные для кластеризации
        evaluator (InternalEvaluator): Оценщик качества кластеризации
        _runs (List[HistoryRun]): История запусков алгоритма
        optimal (Optimal): Лучший найденный вариант
    """
    
    WARMUP_TRIALS = 3  # Количество "разогревочных" запусков

    def __init__(
        self,
        algo_conf: AlgoConfig,
        df: DataFrameType,
        evaluator: InternalEvaluator
    ):
        """Инициализация оптимизатора.
        
        Аргументы:
            algo_conf (AlgoConfig): Конфигурация алгоритма
            df (DataFrameType): Данные для кластеризации
            evaluator (InternalEvaluator): Оценщик качества
        """
        self.algo_conf = algo_conf
        self.df = df
        self.evaluator = evaluator
        self._runs = list()
        self.optimal = None

    @property
    def success_runs(self) -> List[SuccessRun]:
        """Список успешных запусков алгоритма."""
        return [run for run in self._runs if isinstance(run, SuccessRun)]

    def step(self) -> HistoryRun:
        """Выполняет один шаг оптимизации.
        
        Возвращает:
            HistoryRun: Результат выполнения шага
        """
        config_sampler = self._config_sampler()
        config = {
            name: config_sampler.suggest(name, space)
            for name, space in self.algo_conf.config_space.items()
        }

        algo = self.algo_conf.build(**config)
        step_result = self._eval_algo(algo)

        printed_algo_params = str({k: v for k, v in step_result.algo_params.items() if k not in ['centroids', 'X', 'subcluster_labels', 'tree']})
        print(f"=== ALGO {step_result.elapsed}s, {step_result.__class__.__name__}, {printed_algo_params} ===")

        self._runs.append(step_result)
        self._post_step_hook(step_result)

        return step_result

    @abstractmethod
    def _config_sampler(self) -> ConfigSampler:
        """Создает семплер параметров конфигурации.
        
        Должен быть реализован в подклассах.
        """
        pass

    def _post_step_hook(self, result: HistoryRun):
        """Метод, вызываемый после каждого шага оптимизации.
        
        Аргументы:
            result (HistoryRun): Результат выполнения шага
        """
        pass

    def _is_optimal(self, estimation: EstimationResult) -> bool:
        """Проверяет, является ли результат оптимальным.
        
        Аргументы:
            estimation (EstimationResult): Оценка качества
            
        Возвращает:
            bool: True если результат лучше текущего оптимального
        """
        return (
                self.optimal is None or
                self.evaluator.is_better(estimation, self.optimal.estimation)
        )

    def _eval_algo(self, algo: ClusteringAlgo) -> HistoryRun:
        """Выполняет оценку алгоритма кластеризации.
        
        Аргументы:
            algo (ClusteringAlgo): Алгоритм для оценки
            
        Возвращает:
            HistoryRun: Результат оценки (успешный или неудачный)
        """
        start_timestamp = time.time()
        try:
            model = algo.fit(self.df)
            labels = model.labels_
            fit_timestamp = time.time()

            estimation = self.evaluator.evaluate(labels)
            eval_timestamp = time.time()

            if self._is_optimal(estimation):
                self.optimal = Optimal(algo, model, estimation)

            return SuccessRun(
                algo_params=algo.__dict__,
                fit_time=fit_timestamp - start_timestamp,
                eval_time=eval_timestamp - fit_timestamp,
                estimation=estimation
            )
        except ValueError:
            failed_time = time.time() - start_timestamp
            return FailedRun(
                algo_params=algo.__dict__,
                consumed=failed_time
            )


class RandomOptimiser(HyperOptimiser):
    """Оптимизатор, использующий случайный поиск гиперпараметров."""
    
    def __init__(
            self,
            algo_conf: AlgoConfig,
            df: DataFrameType,
            evaluator: InternalEvaluator
    ):
        """Инициализация случайного оптимизатора."""
        super().__init__(algo_conf, df, evaluator)

    def _config_sampler(self) -> ConfigSampler:
        """Использует случайный семплер параметров."""
        return RandomSampler(self.df)


class OptunaOptimiser(HyperOptimiser):
    """Оптимизатор, использующий библиотеку Optuna для поиска гиперпараметров."""
    
    def __init__(
        self,
        algo_conf: AlgoConfig,
        df: DataFrameType,
        evaluator: InternalEvaluator
    ):
        """Инициализация оптимизатора Optuna."""
        super().__init__(algo_conf, df, evaluator)

        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self._session = self._new_study()

    @staticmethod
    def _new_study():
        """Создает новое исследование Optuna."""
        import optuna
        from optuna.samplers import TPESampler

        optuna_sampler = TPESampler(n_startup_trials=HyperOptimiser.WARMUP_TRIALS)
        return optuna.create_study(direction='maximize', sampler=optuna_sampler)

    def _value_for_run(self, run: HistoryRun) -> float:
        """Вычисляет значение целевой функции для запуска.
        
        Аргументы:
            run (HistoryRun): Результат запуска
            
        Возвращает:
            float: Значение целевой функции
        """
        if isinstance(run, FailedRun):
            return -float('inf')
        assert isinstance(run, SuccessRun)

        return sum([
            self.evaluator.compare(run.estimation, other.estimation)
            for other in self.success_runs
        ])

    def _config_sampler(self) -> ConfigSampler:
        """Использует семплер Optuna для подбора параметров."""
        trial = self._session.ask()
        return OptunaSampler(self.df, trial)

    def _post_step_hook(self, result: HistoryRun):
        """Обновляет исследование Optuna после каждого шага."""
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
    """Перечисление доступных оптимизаторов гиперпараметров."""
    
    RANDOM = RandomOptimiser  # Оптимизатор со случайным поиском
    OPTUNA = OptunaOptimiser  # Оптимизатор на основе Optuna
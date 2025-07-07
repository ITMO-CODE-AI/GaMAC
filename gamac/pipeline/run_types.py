from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from gamac.algorithms.base import ClusteringAlgo, ClusteringModel
from gamac.data.data_pipeline import LabelsType, DataFrameType
from gamac.estimation.internal import EstimationResult


class HistoryRun(ABC):
    """Абстрактный базовый класс для хранения истории запусков алгоритмов.

    Атрибуты:
        algo_params (Dict[str, Any]): Параметры алгоритма
        elapsed (float): Затраченное время в секундах
    """

    def __init__(self, algo_params: Dict[str, Any], elapsed: float):
        """Инициализация записи истории.

        Аргументы:
            algo_params (Dict[str, Any]): Параметры алгоритма
            elapsed (float): Общее затраченное время
        """
        self.algo_params = algo_params
        self.elapsed = elapsed


class FailedRun(HistoryRun):
    """Класс для хранения информации о неудачном запуске алгоритма."""

    def __init__(
        self,
        algo_params: Dict[str, Any],
        consumed: float,
    ):
        """Инициализация записи о неудачном запуске.

        Аргументы:
            algo_params (Dict[str, Any]): Параметры алгоритма
            consumed (float): Затраченное время перед ошибкой
        """
        super().__init__(algo_params, consumed)


class SuccessRun(HistoryRun):
    """Класс для хранения информации об успешном запуске алгоритма.

    Атрибуты:
        fit_time (float): Время обучения модели
        eval_time (float): Время оценки качества
        estimation (EstimationResult): Результаты оценки
    """

    def __init__(
        self,
        algo_params: Dict[str, Any],
        fit_time: float,
        eval_time: float,
        estimation: EstimationResult
    ):
        """Инициализация записи об успешном запуске.

        Аргументы:
            algo_params (Dict[str, Any]): Параметры алгоритма
            fit_time (float): Время обучения
            eval_time (float): Время оценки
            estimation (EstimationResult): Результаты оценки качества
        """
        super().__init__(algo_params, fit_time + eval_time)
        self.fit_time = fit_time  # Время обучения модели
        self.eval_time = eval_time  # Время оценки качества
        self.estimation = estimation  # Результаты оценки


class Optimal:
    """Класс для хранения информации о лучшем найденном решении.

    Атрибуты:
        algo (ClusteringAlgo): Лучший алгоритм
        model (ClusteringModel): Лучшая модель
        estimation (EstimationResult): Результаты оценки лучшего решения
    """

    def __init__(
        self,
        algo: ClusteringAlgo,
        model: ClusteringModel,
        estimation: EstimationResult
    ):
        """Инициализация записи о лучшем решении.

         Аргументы:
             algo (ClusteringAlgo): Алгоритм кластеризации
             model (ClusteringModel): Обученная модель
             estimation (EstimationResult): Оценка качества
         """
        self.algo = algo  # Алгоритм, давший лучший результат
        self.model = model  # Обученная модель кластеризации
        self.estimation = estimation  # Оценка качества кластеризации


class GamacResult:
    def __init__(
            self,
            df: DataFrameType,
            algo: ClusteringAlgo,
            model: ClusteringModel,
            estimation: EstimationResult,
            f1_score: Optional[float],
    ):
        self.df, self.algo, self.model = df, algo, model
        self.estimation, self.f1_score = estimation, f1_score

    @property
    def labels(self) -> LabelsType:
        return self.model.labels_
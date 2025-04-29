"""Основной скрипт класса Gamac"""
from typing import Optional, Set

from PIL import Image
from numpy import ndarray
from pandas import DataFrame

from gamac.algorithms.c_kernel.bisecting_kmeans import BisectingKMeansConf
from gamac.estimation.internal import Internal, InternalEvaluator
from gamac.data.data_pipeline import DataHandler, DataFrameType, LabelsType
from gamac.pipeline.hyper_optimisers import HyperOptimisers
from gamac.pipeline.mab_solvers import MabSolvers
from gamac.pipeline.cvi_predictor import CVIPredictor


class Gamac:
    """Основной пайплайн Gamac"""

    def __init__(
            self,
            mab_solver: MabSolvers = MabSolvers.SOFTMAX,
            hyper_optimiser: HyperOptimisers = HyperOptimisers.OPTUNA,
            target_measures: Optional[Set[Internal]] = None,
            time_limit: Optional[int] = None,
            iter_limit: Optional[int] = 50,
    ):
        self._mab_arg = mab_solver
        self._hyper_arg = hyper_optimiser
        self._measures_arg = target_measures
        self._algorithms = [BisectingKMeansConf()]
        self._time_limit_arg = time_limit
        self._iter_limit_arg = iter_limit

    def _check_input(
            self,
            table: Optional[DataFrame],
            text: Optional[list[str]],
            image: Optional[list[Image]],
    ):
        if not table and not text and not image:
            assert "There is not data included"
        if not table and (not text or not image):
            assert "text and image must be specified"

    def run(
            self,
            table: Optional[DataFrame],
            text: Optional[list[str]],
            image: Optional[list[Image]],
    ) -> tuple[DataFrameType, LabelsType]:
        """
        Запуск пайплайна

        Args:
            table (DataFrame): Таблица из которой брать данные
            text (list[str]): Текстовые данные
            image (list[Image]): Картинки из которых брать данные
        Returns:
            tuple[ndarray, list[int]]: Кортеж датасет и список кластеров
        """
        # self._check_input(table, text, image)
        #
        # # Обработка данных в единый датасет
        # df = self._data_handler(table, text, image)
        df = table

        # Получение рекомендации мер качества
        if self._measures_arg is None:
            measures = self._cvi_predictor(df)
        else:
            measures = self._measures_arg

        # Кластеризация датасета с применением рекомендованной меры качества
        clusters = self._auto_clustering(df, measures)

        return df, clusters

    def _data_handler(
            self,
            table: DataFrame,
            text: list[str],
            image: list[Image],
    ) -> DataFrameType:
        """
        Обработка данных в единый датасет

        Args:
            table (DataFrame): Таблица из которой брать данные
            text (list[str]): Текстовые данные
            image (list[Image]): Картинки из которых брать данные
        Returns:
            ndarray: Данные для обработки
        """
        handler = DataHandler()
        return handler.run(table, text, image)

    def _cvi_predictor(self, df: DataFrameType) -> Set[Internal]:
        """
        Получение рекомендации мер качества

        Args:
            df (ndarray): Данные для обработки
        Returns:
            str: Рекомендованная мера качества
        """
        single_prediction = CVIPredictor.run(df)
        return {single_prediction}

    def _auto_clustering(
            self,
            df: DataFrameType,
            measures: Set[Internal],
    ) -> LabelsType:
        """
        Кластеризация датасета с применением рекомендованной меры качества

        Args:
            dataset (ndarray): Преображенный датасет
            advised_measure (str): Рекомендованная мера качества
        Returns:
            list[int]: Список кластеров
        """
        evaluator = InternalEvaluator.create(df, measures)
        mab_builder, opt_builder = self._mab_arg.value, self._hyper_arg.value
        optimisers = [opt_builder(algo, df, evaluator) for algo in self._algorithms]
        mab_solver = mab_builder(df, optimisers, evaluator)

        return mab_solver.run(
            time_limit=self._time_limit_arg,
            iter_limit=self._iter_limit_arg,
        )

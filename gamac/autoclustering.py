"""Основной скрипт класса Gamac"""
import time
import cupy as cp
from typing import Optional, Set

from PIL import Image
from numpy import ndarray
from pandas import DataFrame

from gamac.algorithms.birch import BirchConfig
from gamac.algorithms.kmeans import KMeansConfig
from gamac.algorithms.bisecting_kmeans import BisectingKMeansConfig
from gamac.algorithms.meanshift import MeanShiftConfig
from gamac.algorithms.dbscan import DBSCANConfig
from gamac.algorithms.hdbscan import HDBSCANConfig
from gamac.data.data_pipeline import DataHandler, DataFrameType, LabelsType
from gamac.estimation.internal import Internal, InternalEvaluator
from gamac.pipeline.cvi_predictor import CVIPredictor
from gamac.pipeline.hyper_optimisers import HyperOptimisers
from gamac.pipeline.mab_solvers import MabSolvers
from gamac.pipeline.run_types import Optimal


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
        self._algorithms = [BisectingKMeansConfig(), MeanShiftConfig(), DBSCANConfig(),
                            HDBSCANConfig(), BirchConfig(), KMeansConfig()]
        self._time_limit_arg = time_limit
        self._iter_limit_arg = iter_limit

    def _check_input(
            self,
            table: Optional[DataFrame],
            text: Optional[list[str]],
            image: Optional[list[Image]],
    ):
        if all([var is None for var in [text, image, table]]):
            assert "There is not data included"
        if table is None and (text is None or image is None):
            assert "text and image must be specified"

    def run(
            self,
            table: Optional[DataFrame],
            text: Optional[list[str]],
            image: Optional[list[Image]],
    ) -> tuple[DataFrameType, Optimal]:
        """
        Запуск пайплайна

        Args:
            table (DataFrame): Таблица из которой брать данные
            text (list[str]): Текстовые данные
            image (list[Image]): Картинки из которых брать данные
        Returns:
            tuple[ndarray, list[int]]: Кортеж датасет и список кластеров
        """
        self._check_input(table, text, image)

        # Обработка данных в единый датасет
        df = self._data_handler(table, text, image)
        # df = table

        df = cp.array(df, dtype=cp.float32, order='C')

        # Получение рекомендации мер качества
        if self._measures_arg is None:
            measures = self._cvi_predictor(df)
        else:
            measures = self._measures_arg

        # Кластеризация датасета с применением рекомендованной меры качества
        optimal = self._auto_clustering(df, measures)

        return df, optimal

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
        meta_start = time.time()
        print("=== Started CVI prediction ===")
        cvi_predictor = CVIPredictor()
        single_prediction = cvi_predictor.run(df)
        meta_time = time.time() - meta_start
        print(f"=== Picked {single_prediction.name} in {meta_time}s ===")
        return {single_prediction}

    def _auto_clustering(
            self,
            df: DataFrameType,
            measures: Set[Internal],
    ) -> Optimal:
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

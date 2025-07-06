"""Основной скрипт класса Gamac"""
import time
import cupy as cp
from typing import Optional, Set

import torch
from PIL import Image
from pandas import DataFrame
from transformers import (
    CLIPProcessor,
    CLIPModel,
)

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
    """Основной класс Gamac для автоматической кластеризации мультимодальных данных.

    Класс реализует пайплайн для автоматического выбора алгоритма кластеризации и его гиперпараметров
    на основе многорукого бандита (MAB) и мета-обучения. Поддерживает обработку текстовых, 
    изобразительных и табличных данных.

    Основные возможности:
    - Автоматический подбор алгоритма кластеризации (K-means, DBSCAN, HDBSCAN и др.)
    - Оптимизация гиперпараметров выбранного алгоритма
    - Подбор метрик качества кластеризации (CVI)
    - Обработка мультимодальных данных (текст, изображения, таблицы)
    - Интеграция с моделями CLIP для получения векторных представлений

    Пример использования:
        >>> gamac = Gamac()
        >>> df, clusters = gamac.run(text=text_data, image=image_data)
        >>> print(clusters)

    Attributes:
        batch_size (int): Размер батча для обработки данных
        model_name (str): Название модели CLIP
        verbose (bool): Флаг вывода дополнительной информации
    """
    def __init__(
            self,
            mab_solver: MabSolvers = MabSolvers.SOFTMAX,
            hyper_optimiser: HyperOptimisers = HyperOptimisers.OPTUNA,
            target_measures: Optional[Set[Internal]] = None,
            time_limit: Optional[int] = None,
            iter_limit: Optional[int] = 50,
            batch_size: int = 32,
            model_name: str = "openai/clip-vit-large-patch14",
            verbose: bool = False
    ):
        """Запускает основной пайплайн кластеризации.

        Обрабатывает входные данные, предсказывает оптимальные метрики качества,
        выполняет автоматический подбор алгоритма кластеризации и его параметров.

        Args:
            table: Датафрейм с табличными данными (опционально)
            text: Список текстовых данных (опционально)
            image: Список изображений в формате PIL.Image (опционально)

        Returns:
            Кортеж из:
            - Обработанные данные в виде массива numpy/cupy
            - Результат кластеризации (метки кластеров)

        Raises:
            AssertionError: Если не передано ни одного типа данных или переданы
                        только текст или только изображения без таблицы
        """
        self._mab_arg = mab_solver
        self._hyper_arg = hyper_optimiser
        self._algorithms = [BisectingKMeansConfig(), MeanShiftConfig(), DBSCANConfig(),
                            HDBSCANConfig(), BirchConfig(), KMeansConfig()]
        self._time_limit_arg = time_limit
        self._iter_limit_arg = iter_limit

        self._measures_arg = target_measures

        self.batch_size = batch_size
        self.model_name = model_name
        self.verbose = verbose

        self._init_clip_model()

    def _init_clip_model(self):
        # Загрузка модели и процессора
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.model_name)

    def _check_input(
            self,
            table: Optional[DataFrame] = None,
            text: Optional[list[str]] = None,
            image: Optional[list[Image]] = None,
    ):
        if all([var is None for var in [text, image, table]]):
            assert "There is not data included"
        if table is None and (text is None or image is None):
            assert "text and image must be specified"

    def run(
            self,
            table: Optional[DataFrame] = None,
            text: Optional[list[str]] = None,
            image: Optional[list[Image]] = None,
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

        df = cp.array(df, dtype=cp.float32, order='C')

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
        handler = DataHandler(
            clip_model=self.clip_model,
            clip_processor=self.clip_processor,
            batch_size=self.batch_size,
            verbose=self.verbose,
            device=self.device
        )
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

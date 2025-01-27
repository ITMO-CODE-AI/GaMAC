"""Основной скрипт класса Gamac"""
from pandas import DataFrame
from numpy import ndarray
from PIL import Image

from gamac.pipeline.data_pipeline import DataHandler
from gamac.pipeline.meta_pipeline import MetaAdvisor
from gamac.pipeline.automl_pipeline import AutoClustering


class Gamac:
    """Основной пайплайн Gamac"""

    def __init__(self):
        self.data_pipeline = DataHandler()
        self.meta_advisor = MetaAdvisor()
        self.autocluster_pipeline = AutoClustering()

    def run(
        self, table: DataFrame,
        text: list[str], image: list[Image]
    ) -> tuple[ndarray, list[int]]:
        """
        Запуск пайплайна

        Args:
            table (DataFrame): Таблица из которой брать данные
            text (list[str]): Текстовые данные
            image (list[Image]): Картинки из которых брать данные
        Returns:
            tuple[ndarray, list[int]]: Кортеж датасет и список кластеров
        """
        if not isinstance(table, DataFrame):
            raise TypeError('Table must be a DataFrame')
        if not isinstance(text, list):
            raise TypeError('text must be a list')
        if not isinstance(image, list):
            raise TypeError('image must be a list')

        # Обработка данных в единый датасет
        dataset = self.data_processing(table, text, image)

        # Получение рекомендации мер качества
        advised_measure = self.meta_advising(dataset)

        # Кластеризация датасета с применением рекомендованной меры качества
        clusters = self.auto_clustering(dataset, advised_measure)

        return dataset, clusters

    def data_processing(self, table: DataFrame,
                        text: list[str], image: list[Image]) -> ndarray:
        """
        Обработка данных в единый датасет

        Args:
            table (DataFrame): Таблица из которой брать данные
            text (list[str]): Текстовые данные
            image (list[Image]): Картинки из которых брать данные
        Returns:
            ndarray: Данные для обработки
        """
        return self.data_pipeline.run(table, text, image)

    def meta_advising(self, dataset: ndarray) -> str:
        """
        Получение рекомендации мер качества

        Args:
            dataset (ndarray): Данные для обработки
        Returns:
            str: Рекомендованная мера качества
        """
        return self.meta_advisor.run()

    def auto_clustering(self, dataset: ndarray, advised_measure: str) -> list[int]:
        """
        Кластеризация датасета с применением рекомендованной меры качества

        Args:
            dataset (ndarray): Преображенный датасет
            advised_measure (str): Рекомендованная мера качества
        Returns:
            list[int]: Список кластеров
        """
        return self.autocluster_pipeline.run()

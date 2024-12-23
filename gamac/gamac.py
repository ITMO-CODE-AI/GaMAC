from typing import List

from pipeline.data_pipeline import DataHandler
from pipeline.meta_pipeline import MetaAdvisor
from pipeline.automl_pipeline import AutoClustering


class Gamac:
    """Основной пайплайн Gamac"""

    def __init__(self):
        self.data_pipeline = DataHandler()
        self.meta_advisor = MetaAdvisor()
        self.auto_clustering = AutoClustering()

    def run(self, table, text, image) -> List[int]:
        """
        Запуск пайплайна

        Args:
            table (str): Таблица из которой брать данные
            text (str): Текстовые данные
            image (str): Картинки из которых брать данные
        Returns:
            List(int): Список кластеров
        """
        # Обработка данных в единый датасет
        dataset = self.data_pipeline.run(table, text, image)

        # Получение рекомендации мер качества
        advised_measure = self.meta_advisor.run()

        # Кластеризация датасета с применением рекомендованной меры качества
        clusters = self.auto_clustering.run(dataset, advised_measure)

        return clusters

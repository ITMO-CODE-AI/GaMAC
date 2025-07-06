"""Модуль с абстрактными классами для реализации алгоритмов кластеризации.

Определяет базовые интерфейсы для:
- Конфигурации алгоритмов (AlgoConfig)
- Реализаций алгоритмов (ClusteringAlgo)
- Моделей кластеризации (ClusteringModel)
"""
import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from gamac.data.data_pipeline import DataFrameType, LabelsType


class ClusteringModel(ABC):
    """Абстрактный класс модели кластеризации.
    
    Содержит метки кластеров и метод для предсказания кластеров на новых данных.
    
    Attributes:
        labels_ (LabelsType): Метки кластеров, полученные при обучении модели.
    """
    def __init__(self, labels_: LabelsType):
        """Инициализирует модель с метками кластеров.
        
        Args:
            labels_: Найденные метки кластеров для обучающих данных.
        """
        self.labels_ = labels_

    @abstractmethod
    def predict(self, df: DataFrameType) -> LabelsType:
        """Предсказывает метки кластеров для новых данных.
        
        Args:
            df: Данные для кластеризации.
            
        Returns:
            Метки кластеров для входных данных.
        """
        pass


class ClusteringAlgo(ABC):
    """Абстрактный класс алгоритма кластеризации.
    
    Определяет интерфейс для алгоритмов кластеризации, включая:
    - Обучение модели (fit)
    - Обучение и предсказание в одном методе (fit_predict)
    - Генерацию случайного seed (make_seed)
    
    Attributes:
        name (str): Название класса алгоритма.
    """
    def __init__(self, **kwargs):
        """Инициализирует алгоритм с произвольными параметрами."""
        self.name = self.__class__.__name__

    @abstractmethod
    def fit(self, df: DataFrameType) -> ClusteringModel:
        """Обучает модель кластеризации на данных.
        
        Args:
            df: Данные для обучения.
            
        Returns:
            Обученная модель кластеризации.
        """
        pass

    def fit_predict(self, df: DataFrameType) -> LabelsType:
        """Обучает модель и сразу предсказывает метки кластеров.
        
        Args:
            df: Данные для кластеризации.
            
        Returns:
            Метки кластеров для входных данных.
        """
        return self.fit(df).predict(df)

    @staticmethod
    def make_seed(seed: Optional[int]) -> int:
        """Генерирует случайный seed, если не задан.
        
        Args:
            seed: Опциональное значение seed.
            
        Returns:
            Заданный seed или случайно сгенерированный.
        """
        return seed if seed is not None else random.randint(0, 2 ** 32)


class AlgoConfig(ABC):
    """Абстрактный класс конфигурации алгоритма.
    
    Содержит пространство параметров и builder для создания экземпляров алгоритмов.
    
    Attributes:
        config_space (dict): Пространство параметров алгоритма.
        builder (Callable): Функция для создания экземпляра алгоритма.
        algo_name (str): Название алгоритма.
    """
    def __init__(self, builder, **kwargs):
        """Инициализирует конфигурацию алгоритма.
        
        Args:
            builder: Функция для создания экземпляра алгоритма.
            **kwargs: Параметры конфигурации алгоритма.
        """
        self.config_space, self.builder = kwargs, builder
        self.algo_name = builder.__name__

    def build(self, **kwargs) -> ClusteringAlgo:
        """Создает экземпляр алгоритма с заданными параметрами.
        
        Args:
            **kwargs: Параметры для создания алгоритма.
            
        Returns:
            Экземпляр алгоритма кластеризации.
        """
        return self.builder(**kwargs)

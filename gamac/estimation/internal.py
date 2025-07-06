import time
from enum import Enum
from typing import Set, List

import cupy as cp
import numpy as np

from gamac.data.data_pipeline import DataFrameType, LabelsType
from gamac.estimation.container import EstimationContainer
from gamac.estimation.functions import br, mcr, sym, os


class Internal(Enum):
    """Перечисление внутренних метрик оценки качества кластеризации.
    
    Атрибуты:
        BR: Метрика Banfield-Raftery
        OS: Метрика OS (относительная разделимость)
        MCR: Метрика McClain-Rao
        SYM: Симметричная метрика
    """
    BR = ('banfield_raftery', br)
    # C_INDEX = ('c_index', c_index)
    OS = ('os', os)
    MCR = ('mc_clain_rao', mcr)
    SYM = ('sym', sym)


EstimationResult = dict[Internal, float]  # Тип для хранения результатов оценки метрик


class InternalEvaluator:
    """Класс для оценки качества кластеризации с использованием внутренних метрик.
    
    Позволяет сравнивать различные кластеризации на основе заданных метрик.
    
    Атрибуты:
        df (DataFrameType): Исходные данные
        pivots (EstimationResult): Опорные значения метрик для случайной кластеризации
    """
    
    def __init__(self, df: DataFrameType, pivots: EstimationResult):
        """Инициализация оценщика.
        
        Аргументы:
            df (DataFrameType): Исходные данные
            pivots (EstimationResult): Опорные значения метрик
        """
        self.df, self.pivots = df, pivots

    @staticmethod
    def create(df: DataFrameType, measures: Set[Internal]) -> 'InternalEvaluator':
        """Фабричный метод для создания оценщика.
        
        Вычисляет опорные значения метрик для случайной кластеризации.
        
        Аргументы:
            df (DataFrameType): Исходные данные
            measures (Set[Internal]): Набор метрик для оценки
            
        Возвращает:
            InternalEvaluator: Инициализированный оценщик
        """
        random_labels = cp.random.randint(low=0, high=2, size=len(df), dtype=np.int32)
        pivots = InternalEvaluator.eval_internal(df, measures, random_labels)
        return InternalEvaluator(df, pivots)

    @staticmethod
    def create_container(df: DataFrameType, labels: LabelsType) -> EstimationContainer:
        """Создает контейнер для вычисления метрик.
        
        Аргументы:
            df (DataFrameType): Исходные данные
            labels (LabelsType): Метки кластеров
            
        Возвращает:
            EstimationContainer: Контейнер с данными кластеризации
        """
        return EstimationContainer.create(df, labels)

    def evaluate(self, labels: LabelsType) -> EstimationResult:
        """Вычисляет значения метрик для заданной кластеризации.
        
        Аргументы:
            labels (LabelsType): Метки кластеров для оценки
            
        Возвращает:
            EstimationResult: Словарь со значениями метрик
        """
        measures = set(self.pivots.keys())
        return self.eval_internal(self.df, measures, labels)

    def rewards(self, estimates: List[EstimationResult]) -> List[float]:
        """Вычисляет оценки "полезности" для списка результатов.
        
        Аргументы:
            estimates (List[EstimationResult]): Список результатов оценки
            
        Возвращает:
            List[float]: Список оценок полезности
        """
        return [
            np.mean([
                self.compare(x, y) for y in estimates
            ]).__float__()
            for x in estimates
        ]

    def compare(self, x: EstimationResult, y: EstimationResult) -> float:
        """Сравнивает два результата оценки.
        
        Аргументы:
            x (EstimationResult): Первый результат
            y (EstimationResult): Второй результат
            
        Возвращает:
            float: Суммарная оценка сравнения по всем метрикам
        """
        return sum([
            self._compare_sym(x[measure], y[measure], pivot)
            for measure, pivot in self.pivots.items()
        ])

    def is_better(self, x: EstimationResult, y: EstimationResult) -> bool:
        """Определяет, является ли первый результат лучше второго.
        
        Аргументы:
            x (EstimationResult): Первый результат
            y (EstimationResult): Второй результат
            
        Возвращает:
            bool: True если x лучше y, иначе False
        """
        return self.compare(x, y) > self.compare(y, x)

    @staticmethod
    def _compare_sym(x: float, y: float, pivot: float) -> float:
        """Симметричное сравнение значений метрики относительно опорного значения.
        
        Аргументы:
            x (float): Первое значение метрики
            y (float): Второе значение метрики
            pivot (float): Опорное значение
            
        Возвращает:
            float: Нормализованная разница между значениями
        """
        normaliser = max(x, y) - pivot
        if normaliser < 1e-6:
            return 0.0
        return (x - y) / normaliser

    @staticmethod
    def eval_internal(df: DataFrameType, measures: Set[Internal], labels: LabelsType) -> EstimationResult:
        """Вычисляет значения внутренних метрик для заданной кластеризации.
        
        Аргументы:
            df (DataFrameType): Исходные данные
            measures (Set[Internal]): Набор метрик для вычисления
            labels (LabelsType): Метки кластеров
            
        Возвращает:
            EstimationResult: Словарь со значениями метрик
        """
        t_start = time.time()
        container = EstimationContainer.create(df, labels)
        values = {
            measure: measure.value[1](container) for measure in measures
        }
        serialised = {m.name: float(v) for m, v in values.items()}
        print(f"=== MEASURES {time.time() - t_start}s, {serialised} ===")
        return EstimationResult(values)

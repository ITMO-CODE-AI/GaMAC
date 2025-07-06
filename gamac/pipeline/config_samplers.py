import numpy as np

from abc import abstractmethod, ABC
from typing import Any, Set

from gamac.data.data_pipeline import DataFrameType


class ConfigSampler(ABC):
    """Абстрактный базовый класс для семплеров параметров конфигурации.
    
    Обеспечивает интерфейс для генерации значений параметров алгоритмов.
    
    Атрибуты:
        df (DataFrameType): Данные, на основе которых могут генерироваться параметры
    """
    
    def __init__(self, df: DataFrameType):
        """Инициализация семплера.
        
        Аргументы:
            df (DataFrameType): Входные данные для анализа
        """
        self.df = df

    def suggest(self, name: str, param: Any) -> Any:
        """Предлагает значение параметра на основе его описания.
        
        Аргументы:
            name (str): Имя параметра
            param (Any): Описание параметра (диапазон, категории и т.д.)
            
        Возвращает:
            Any: Сгенерированное значение параметра
            
        Выбрасывает:
            ValueError: Если тип параметра не распознан
        """
        if isinstance(param, tuple) and len(param) == 2:
            return self._range_param(name, param)
        elif isinstance(param, (set, frozenset)):
            return self._categorical_param(name, param)
        elif callable(param) or isinstance(param, (int, float, bool)):
            return self._calc(param)
        else:
            raise ValueError(f"Не удалось распознать параметр '{name}': {param}")

    def _range_param(self, name: str, param: tuple) -> Any:
        """Обрабатывает параметр, заданный диапазоном значений.
        
        Аргументы:
            name (str): Имя параметра
            param (tuple): Кортеж (нижняя граница, верхняя граница)
            
        Возвращает:
            Any: Сгенерированное значение параметра
            
        Выбрасывает:
            ValueError: Если границы диапазона разных типов
        """
        lower, upper = self._calc(param[0]), self._calc(param[1])
        if isinstance(lower, int) and isinstance(upper, int):
            return self._int_param(name, lower, upper)
        elif isinstance(lower, float) and isinstance(upper, float):
            return self._float_param(name, lower, upper)
        else:
            raise ValueError(f"Ожидались целые или вещественные числа (параметр '{name}')")

    def _calc(self, param: Any) -> Any:
        """Вычисляет значение параметра (если передана функция)."""
        return param(self.df) if callable(param) else param

    @abstractmethod
    def _int_param(self, name: str, lower: int, upper: int) -> int:
        """Генерирует целочисленный параметр.
        
        Аргументы:
            name (str): Имя параметра
            lower (int): Нижняя граница диапазона
            upper (int): Верхняя граница диапазона
            
        Возвращает:
            int: Сгенерированное значение
        """
        pass

    @abstractmethod
    def _float_param(self, name: str, lower: float, upper: float) -> float:
        """Генерирует вещественный параметр.
        
        Аргументы:
            name (str): Имя параметра
            lower (float): Нижняя граница диапазона
            upper (float): Верхняя граница диапазона
            
        Возвращает:
            float: Сгенерированное значение
        """
        pass

    @abstractmethod
    def _categorical_param(self, name: str, values: Set[Any]) -> Any:
        """Генерирует категориальный параметр.
        
        Аргументы:
            name (str): Имя параметра
            values (Set[Any]): Множество возможных значений
            
        Возвращает:
            Any: Выбранное значение
        """
        pass


class RandomSampler(ConfigSampler):
    """Семплер, генерирующий случайные значения параметров."""
    
    def __init__(self, df: DataFrameType):
        """Инициализация случайного семплера.
        
        Аргументы:
            df (DataFrameType): Входные данные для анализа
        """
        super().__init__(df)

    def _int_param(self, name: str, lower: int, upper: int) -> int:
        """Генерирует случайное целое число в заданном диапазоне."""
        return np.random.randint(lower, upper)

    def _float_param(self, name: str, lower: float, upper: float) -> float:
        """Генерирует случайное вещественное число в заданном диапазоне."""
        return (np.random.random() + lower) * (upper - lower)

    def _categorical_param(self, name: str, values: Set[Any]) -> Any:
        """Выбирает случайное значение из заданного множества."""
        index = np.random.randint(0, len(values))
        return list(values)[index]


class OptunaSampler(ConfigSampler):
    """Семплер, интегрированный с фреймворком Optuna для оптимизации гиперпараметров."""
    
    def __init__(self, df: DataFrameType, trial):
        """Инициализация семплера Optuna.
        
        Аргументы:
            df (DataFrameType): Входные данные для анализа
            trial: Объект trial из Optuna
        """
        super().__init__(df)
        from optuna.trial import BaseTrial
        self.trial: BaseTrial = trial

    def _int_param(self, name: str, lower: int, upper: int) -> int:
        """Генерирует целочисленный параметр через Optuna."""
        return self.trial.suggest_int(name, lower, upper)

    def _float_param(self, name: str, lower: float, upper: float) -> float:
        """Генерирует вещественный параметр через Optuna."""
        return self.trial.suggest_float(name, lower, upper)

    def _categorical_param(self, name: str, values: Set[Any]) -> Any:
        """Генерирует категориальный параметр через Optuna."""
        as_list = list(values)
        return self.trial.suggest_categorical(name, as_list)
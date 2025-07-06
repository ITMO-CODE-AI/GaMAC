import time
from abc import abstractmethod, ABC
from enum import Enum
from typing import List, Optional

import numpy as np

from gamac.estimation.internal import InternalEvaluator, EstimationResult
from gamac.data.data_pipeline import DataFrameType
from gamac.pipeline.hyper_optimisers import HyperOptimiser
from gamac.pipeline.run_types import Optimal


class MabSolver(ABC):
    """Абстрактный базовый класс для решателя Multi-Armed Bandit (MAB) задачи.
    
    Реализует стратегию выбора между различными оптимизаторами гиперпараметров.
    
    Атрибуты:
        df (DataFrameType): Данные для кластеризации
        optimisers (List[HyperOptimiser]): Список оптимизаторов ("руки" бандита)
        evaluator (InternalEvaluator): Оценщик качества кластеризации
        arms (int): Количество оптимизаторов
        consumption (np.ndarray): Массив затраченного времени на каждый оптимизатор
        usage (np.ndarray): Массив количества использований каждого оптимизатора
        history (list): История всех запусков
        _time_limit (float): Оставшийся лимит времени
        _iter_limit (int): Оставшийся лимит итераций
    """
    
    def __init__(
            self,
            df: DataFrameType,
            optimisers: List[HyperOptimiser],
            evaluator: InternalEvaluator,
    ):
        """Инициализация решателя MAB.
        
        Аргументы:
            df (DataFrameType): Данные для кластеризации
            optimisers (List[HyperOptimiser]): Список оптимизаторов
            evaluator (InternalEvaluator): Оценщик качества
        """
        self.df = df
        self.optimisers = optimisers
        self.evaluator = evaluator

        self.arms = len(optimisers)
        self.consumption = np.zeros(self.arms)
        self.usage = np.zeros(self.arms, dtype=int)

        self.history = list()
        self._time_limit, self._iter_limit = 0, 0

    def run(self, time_limit: Optional[float], iter_limit: Optional[int]) -> Optimal:
        """Основной метод запуска оптимизации.
        
        Аргументы:
            time_limit (Optional[float]): Лимит времени в секундах
            iter_limit (Optional[int]): Лимит итераций
            
        Возвращает:
            Optimal: Лучший найденный вариант кластеризации
        """
        self._time_limit, self._iter_limit = time_limit, iter_limit

        self._warmup_if_needed()
        self._run_main_loop()

        best_arm = np.argmax(self._quality_rewards())
        return self.optimisers[best_arm].optimal

    @abstractmethod
    def pick_arm(self) -> int:
        """Абстрактный метод выбора "руки" (оптимизатора) для следующего шага.
        
        Должен быть реализован в подклассах.
        
        Возвращает:
            int: Индекс выбранного оптимизатора
        """
        pass

    def _step(self, arm: int):
        """Выполняет один шаг оптимизации для выбранного оптимизатора.
        
        Аргументы:
            arm (int): Индекс оптимизатора
        """
        t_start = time.time()
        run = self.optimisers[arm].step()
        arm_time = time.time() - t_start

        self.consumption[arm] += arm_time
        self.usage[arm] += 1
        self.history.append(run)

        self._consume(arm_time)

    def _is_warmup_required(self) -> bool:
        """Проверяет, требуется ли фаза 'разогрева' оптимизаторов.
        
        Возвращает:
            bool: True если требуется дополнительный разогрев
        """
        return np.min(self.usage) < HyperOptimiser.WARMUP_TRIALS or self._arms_initialised() == 0

    def _warmup_if_needed(self):
        """Выполняет фазу 'разогрева' если это необходимо."""
        while not self._budget_exhausted() and self._is_warmup_required():
            arm = np.argmin(self.usage)
            self._step(arm)

    def _run_main_loop(self):
        """Основной цикл выполнения оптимизации."""
        while not self._budget_exhausted():
            arm = self.pick_arm()
            self._step(arm)

    def _budget_exhausted(self) -> bool:
        """Проверяет исчерпание бюджета (времени/итераций).
        
        Возвращает:
            bool: True если бюджет исчерпан
        """
        time_exhausted = self._time_limit is not None and self._time_limit <= 0
        iter_exhausted = self._iter_limit is not None and self._iter_limit <= 0
        return time_exhausted or iter_exhausted

    def _consume(self, time_consumed: float):
        """Уменьшает оставшийся бюджет.
        
        Аргументы:
            time_consumed (float): Затраченное время
        """
        if self._time_limit is not None:
            self._time_limit -= time_consumed
        if self._iter_limit is not None:
            self._iter_limit -= 1

    def _opts(self) -> List[Optional[Optimal]]:
        """Возвращает список лучших вариантов для каждого оптимизатора.
        
        Возвращает:
            List[Optional[Optimal]]: Список лучших вариантов
        """
        return [optimiser.optimal for optimiser in self.optimisers]

    def _arms_initialised(self) -> int:
        """Подсчитывает количество инициализированных оптимизаторов.
        
        Возвращает:
            int: Количество оптимизаторов с найденными решениями
        """
        return sum([0 if opt is None else 1 for opt in self._opts()])

    def _safe_compare(self, x: Optional[Optimal], y: Optional[Optimal], fake: EstimationResult) -> float:
        """Безопасное сравнение двух вариантов с подменой при отсутствии.
        
        Аргументы:
            x (Optional[Optimal]): Первый вариант
            y (Optional[Optimal]): Второй вариант
            fake (EstimationResult): Подмена для отсутствующих вариантов
            
        Возвращает:
            float: Результат сравнения
        """
        x_or_fake = x.estimation if x is not None else fake
        y_or_fake = y.estimation if y is not None else fake
        return self.evaluator.compare(x_or_fake, y_or_fake)

    def _std_norm(self, rewards: List[float]) -> np.ndarray:
        """Нормализует награды вычитанием среднего и делением на СКО.
        
        Аргументы:
            rewards (List[float]): Список наград
            
        Возвращает:
            np.ndarray: Нормализованные награды
        """
        mean, std = np.mean(rewards), np.std(rewards)
        diff = np.array(rewards) - mean
        return np.zeros(self.arms) if std < 1e-5 else diff / std

    def _quality_rewards(self) -> List[float]:
        """Вычисляет награды за качество кластеризации.
        
        Возвращает:
            List[float]: Список наград за качество
        """
        fake_estimation = self.evaluator.pivots
        safe_estimations = [opt.estimation if opt is not None else fake_estimation for opt in self._opts()]
        return self.evaluator.rewards(safe_estimations)

    def _time_rewards(self) -> np.ndarray:
        """Вычисляет награды за эффективность по времени.
        
        Возвращает:
            np.ndarray: Награды за эффективность
        """
        return 1 - self.consumption / sum(self.consumption)

    def _rewards(self) -> np.ndarray:
        """Объединяет награды за качество и эффективность.
        
        Возвращает:
            np.ndarray: Комбинированные награды
        """
        return self._std_norm(self._quality_rewards()) + self._std_norm(self._time_rewards())


class RandomMab(MabSolver):
    """Реализация случайного выбора между оптимизаторами."""
    
    def __init__(
            self,
            df: DataFrameType,
            optimisers: List[HyperOptimiser],
            evaluator: InternalEvaluator
    ):
        """Инициализация случайного решателя."""
        super().__init__(df, optimisers, evaluator)

    def pick_arm(self) -> int:
        """Случайный выбор оптимизатора.
        
        Возвращает:
            int: Случайный индекс оптимизатора
        """
        return np.random.choice(np.arange(self.arms))


class SoftmaxMab(MabSolver):
    """Реализация Softmax стратегии выбора оптимизаторов."""
    
    def __init__(
            self,
            df: DataFrameType,
            optimisers: List[HyperOptimiser],
            evaluator: InternalEvaluator
    ):
        """Инициализация Softmax решателя."""
        super().__init__(df, optimisers, evaluator)

    def pick_arm(self) -> int:
        """Выбор оптимизатора по стратегии Softmax.
        
        Возвращает:
            int: Индекс выбранного оптимизатора
        """
        rewards = self._rewards()
        exp_rewards = np.exp(rewards - rewards.max())
        probs = exp_rewards / exp_rewards.sum()
        return np.random.choice(np.arange(self.arms), p=probs)


class MabSolvers(Enum):
    """Перечисление доступных стратегий MAB."""
    
    RANDOM = RandomMab  # Случайный выбор
    SOFTMAX = SoftmaxMab  # Softmax стратегия
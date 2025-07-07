"""Основной скрипт этапа подсказки метрик кластеризации.

Модуль содержит класс для предсказания наиболее подходящих внутренних метрик
оценки качества кластеризации (Cluster Validity Indices, CVI) на основе 
анализа мета-признаков входных данных.
"""

import os
import pickle
import importlib.resources as resources

from gamac.data.data_pipeline import DataFrameType
from gamac.estimation.internal import Internal
import cupy as cp
import numpy as np

from gamac.kernels import BATCH_SIZE, MIDDLEWARE


# def load_pickle(file: str):
#     """Загружает объект из pickle-файла.

#     Аргументы:
#         file (str): Имя файла в директории bin

#     Возвращает:
#         Any: Загруженный Python-объект
#     """
#     try:
#         with resources.files("gamac.bin").joinpath(file).open('rb') as fp:
#             return pickle.load(fp)
#     except Exception:
#         real_path = os.path.realpath(__file__)
#         dir_path = os.path.dirname(real_path)
#         root_path = os.path.dirname(dir_path)
#         with open(f"{root_path}/bin/{file}", 'rb') as fp:
#             return pickle.load(fp)


def load_pickle(filename: str):
    """Загружает pickle-файл из ресурсов пакета."""
    try:
        # Проверка наличия ресурса
        resource_path = resources.files("gamac.bin").joinpath(filename)
        if not resource_path.is_file():
            available = [f.name for f in resources.files("gamac.bin").iterdir()
                         if f.is_file()]
            raise FileNotFoundError(
                f"Resource {filename} not found in gamac.bin. "
                f"Available: {available}"
            )

        with resource_path.open('rb') as fp:
            return pickle.load(fp)
    except Exception as e:
        # Дополнительная диагностика
        print(f"Error loading {filename}: {str(e)}")
        raise


class CVIPredictor:
    """Класс для предсказания оптимальных метрик оценки кластеризации.

    Использует предобученные модели для анализа мета-признаков данных
    и выбора наиболее подходящих метрик качества кластеризации.

    Атрибуты:
        BUCKETS (int): Количество корзин для гистограммного анализа
        MEASURES_BY_INDEX (list): Список метрик, соответствующих индексам модели
    """

    BUCKETS = 128  # Количество интервалов для статистики расстояний
    MEASURES_BY_INDEX = [
        Internal.OS,  # Метрика OS (относительная разделимость)
        Internal.SYM,  # Симметричная метрика
        Internal.BR,   # Метрика Banfield-Raftery
        Internal.MCR   # Метрика McClain-Rao
    ]

    def __init__(self):
        """Инициализирует предиктор, загружая предобученные модели."""
        self.extractor = load_pickle('extractor.pkl')  # Модель преобразования признаков
        self.model = load_pickle('classifier.pkl')     # Классификатор метрик

    def run(self, df: DataFrameType) -> Internal:
        """Основной метод для предсказания оптимальной метрики.
        
        Аргументы:
            df (DataFrameType): Входные данные для анализа
            
        Возвращает:
            Internal: Рекомендуемая метрика оценки качества кластеризации
        """
        meta_features = self._meta_features(df)
        transformed = self._transform(meta_features)
        return self._predict(transformed)

    def _meta_features(self, df: DataFrameType) -> np.ndarray:
        """Вычисляет мета-признаки на основе распределения расстояний.
        
        Аргументы:
            df (DataFrameType): Входные данные
            
        Возвращает:
            np.ndarray: Массив мета-признаков (нормированный)
        """
        n, dims = df.shape
        d_max = float('-inf')  # Максимальное расстояние в данных
        quotient, remainder = divmod(n, self.BUCKETS)

        # Выделение памяти на GPU
        gpu_partial_arr = cp.empty(shape=(BATCH_SIZE, n), dtype=np.float32)
        gpu_stats_arr = cp.empty(shape=(BATCH_SIZE, self.BUCKETS, 4), dtype=np.float32)

        mfs_accumulator = np.zeros(shape=(self.BUCKETS, 4), dtype=np.float32)

        iterations = n // BATCH_SIZE + (0 if n % BATCH_SIZE == 0 else 1)

        for iter_idx in range(iterations):
            print(f'=== CVI prediction iteration {iter_idx + 1}/{iterations} ====')
            batch_start = iter_idx * BATCH_SIZE
            batch_size = min(BATCH_SIZE, n - batch_start)

            # Вычисление частичных расстояний на GPU
            MIDDLEWARE.meta_dist_partial(
                N=n,
                D=dims,
                data=df,
                batch_start=batch_start,
                batch_size=batch_size,
                partial_dists=gpu_partial_arr,
            ).invoke(
                grid=(1, n),
                blocks=(batch_size, 1),
            )

            # Сортировка и поиск максимального расстояния
            gpu_partial_arr.sort(axis=1)
            cpu_d_max_arr = cp.asnumpy(gpu_partial_arr[:, -1])
            d_max = max(d_max, *cpu_d_max_arr[:batch_size])

            # Вычисление статистики по корзинам
            MIDDLEWARE.meta_dist_stat(
                Q=quotient,
                R=remainder,
                N=n,
                sorted_dists=gpu_partial_arr,
                batch_size=batch_size,
                dist_stats=gpu_stats_arr,
            ).invoke(
                grid=(1, self.BUCKETS),
                blocks=(batch_size, 1),
            )

            # Агрегация результатов
            cpu_stats_arr = cp.asnumpy(gpu_stats_arr)
            batch_accumulator = np.sum(
                cpu_stats_arr[:batch_size], axis=0
            )
            mfs_accumulator += batch_accumulator

        # Нормировка мета-признаков
        return mfs_accumulator.flatten(order='F') / d_max / n

    def _transform(self, meta_features: np.ndarray) -> np.ndarray:
        """Применяет преобразование признаков.
        
        Аргументы:
            meta_features (np.ndarray): Сырые мета-признаки
            
        Возвращает:
            np.ndarray: Преобразованные признаки
        """
        return self.extractor.transform([meta_features])[0]

    def _predict(self, transformed: np.ndarray) -> Internal:
        """Выполняет предсказание оптимальной метрики.
        
        Аргументы:
            transformed (np.ndarray): Преобразованные мета-признаки
            
        Возвращает:
            Internal: Рекомендуемая метрика
        """
        cvi_index = self.model.predict([transformed])[0]
        return self.MEASURES_BY_INDEX[cvi_index]
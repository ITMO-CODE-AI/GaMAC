import os.path
import importlib.resources as resources

from cupy import RawModule
from cupy.typing import NDArray

from gamac.data.data_pipeline import DataFrameType, LabelsType


def load_module(file: str) -> RawModule:
    """Загружает CUDA-модуль из файла.
    
    Аргументы:
        file (str): Имя файла с CUDA-кодом
        
    Возвращает:
        RawModule: Загруженный модуль CUDA
    """
    try:
        content = resources.files("gamac.kernels").joinpath(file).read_text(encoding="utf-8")
    except Exception:
        real_path = os.path.realpath(__file__)
        dir_path = os.path.dirname(real_path)
        with open(f"{dir_path}/{file}", 'r') as fp:
            content = fp.read()
    return RawModule(code=content)


class KernelInvocation:
    """Класс для инкапсуляции вызова CUDA-ядра.
    
    Атрибуты:
        kernel (RawKernel): CUDA-ядро для вызова
        args (tuple): Аргументы для передачи ядру
    """
    
    def __init__(self, kernel, args):
        """Инициализация вызова ядра.
        
        Аргументы:
            kernel: CUDA-ядро
            args: Аргументы для ядра
        """
        self.kernel, self.args = kernel, args

    def invoke(self, grid: tuple, blocks: tuple):
        """Выполняет запуск ядра на GPU.
        
        Аргументы:
            grid (tuple): Конфигурация grid для запуска ядра
            blocks (tuple): Конфигурация blocks для запуска ядра
        """
        self.kernel(grid, blocks, self.args)


class Middleware:
    """Промежуточный слой для работы с CUDA-ядрами.
    
    Обеспечивает удобный интерфейс для вызова различных ядер.
    """
    
    def __init__(self):
        """Инициализация Middleware с загрузкой всех необходимых модулей."""
        self._meta = load_module('meta-kernels.c')
        self._cvi = load_module('cvi-kernels.c')
        self._kmeans = load_module('kmeans-kernels.c')

    def meta_dist_partial(
            self, *,
            N: int,
            D: int,
            data: NDArray,
            batch_start: int,
            batch_size: int,
            partial_dists: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для вычисления частичных расстояний.
        
        Аргументы:
            N (int): Количество объектов
            D (int): Размерность данных
            data (NDArray): Массив данных
            batch_start (int): Начальный индекс батча
            batch_size (int): Размер батча
            partial_dists (NDArray): Массив для частичных расстояний
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._meta.get_function('meta_dist_partial'),
            args=(N, D, data, batch_start, batch_size, partial_dists),
        )

    def meta_dist_stat(
            self, *,
            Q: int,
            R: int,
            N: int,
            sorted_dists: NDArray,
            batch_size: int,
            dist_stats: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для статистики по расстояниям.
        
        Аргументы:
            Q (int): Первый параметр статистики
            R (int): Второй параметр статистики
            N (int): Количество объектов
            sorted_dists (NDArray): Отсортированные расстояния
            batch_size (int): Размер батча
            dist_stats (NDArray): Массив для статистики
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._meta.get_function('meta_dist_stat'),
            args=(Q, R, N, sorted_dists, batch_size, dist_stats),
        )

    def get_centroids(
            self, *,
            data: DataFrameType,
            labels: LabelsType,
            N: int,
            D: int,
            K: int,
            uniq_labels: NDArray,
            centroids: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для вычисления центроидов.
        
        Аргументы:
            data (DataFrameType): Исходные данные
            labels (LabelsType): Метки кластеров
            N (int): Количество объектов
            D (int): Размерность данных
            K (int): Количество кластеров
            uniq_labels (NDArray): Уникальные метки
            centroids (NDArray): Массив для центроидов
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._cvi.get_function('get_centroids'),
            args=(data, labels, N, D, K, uniq_labels, centroids),
        )

    def get_cent_dists(
            self, *,
            cluster: NDArray,
            cl_n: int,
            D: int,
            centroids: NDArray,
            k_idx: int,
            cent_dists: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для расстояний до центроидов.
        
        Аргументы:
            cluster (NDArray): Данные кластера
            cl_n (int): Количество объектов в кластере
            D (int): Размерность данных
            centroids (NDArray): Центроиды кластеров
            k_idx (int): Индекс текущего кластера
            cent_dists (NDArray): Массив для расстояний
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._cvi.get_function('get_cent_dists'),
            args=(cluster, cl_n, D, centroids, k_idx, cent_dists),
        )
        
    def get_cent_matrix(
            self, *,
            centroids: NDArray,
            K: int,
            D: int,
            cent_matrix: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для матрицы расстояний между центроидами.
        
        Аргументы:
            centroids (NDArray): Центроиды кластеров
            K (int): Количество кластеров
            D (int): Размерность данных
            cent_matrix (NDArray): Матрица для расстояний
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._cvi.get_function('get_cent_matrix'),
            args=(centroids, K, D, cent_matrix),
        )
    
    def get_sym_data(
            self, *,
            cluster: NDArray,
            cl_n: int,
            D: int,
            centroids: NDArray,
            k_idx: int,
            sym_data: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для симметричных данных.
        
        Аргументы:
            cluster (NDArray): Данные кластера
            cl_n (int): Количество объектов в кластере
            D (int): Размерность данных
            centroids (NDArray): Центроиды кластеров
            k_idx (int): Индекс текущего кластера
            sym_data (NDArray): Массив для симметричных данных
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._cvi.get_function('get_sym_data'),
            args=(cluster, cl_n, D, centroids, k_idx, sym_data),
        )
        
    def get_sym_dists(
            self, *,
            cluster: NDArray,
            cl_n: int,
            D: int,
            cent_dists: NDArray,
            sym_data: NDArray,
            sym_dists: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для симметричных расстояний.
        
        Аргументы:
            cluster (NDArray): Данные кластера
            cl_n (int): Количество объектов в кластере
            D (int): Размерность данных
            cent_dists (NDArray): Расстояния до центроидов
            sym_data (NDArray): Симметричные данные
            sym_dists (NDArray): Массив для симметричных расстояний
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._cvi.get_function('get_sym_dists'),
            args=(cluster, cl_n, D, cent_dists, sym_data, sym_dists),
        )

    def mcr(
            self, *,
            data: NDArray,
            N: int,
            D: int,
            labels: NDArray,
            s_w: NDArray,
            s_b: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для метрики MCR.
        
        Аргументы:
            data (NDArray): Исходные данные
            N (int): Количество объектов
            D (int): Размерность данных
            labels (NDArray): Метки кластеров
            s_w (NDArray): Массив для внутрикластерных расстояний
            s_b (NDArray): Массив для межкластерных расстояний
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._cvi.get_function('mcr'),
            args=(data, N, D, labels, s_w, s_b),
        )

    def c_index(
            self, *,
            data: NDArray,
            N: int,
            D: int,
            pairs: int,
            labels: NDArray,
            s_min_idx: int,
            s_min: NDArray,
            s_max_idx: int,
            s_max: NDArray,
            s_c: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для индекса C.
        
        Аргументы:
            data (NDArray): Исходные данные
            N (int): Количество объектов
            D (int): Размерность данных
            pairs (int): Количество пар
            labels (NDArray): Метки кластеров
            s_min_idx (int): Индекс минимального расстояния
            s_min (NDArray): Минимальные расстояния
            s_max_idx (int): Индекс максимального расстояния
            s_max (NDArray): Максимальные расстояния
            s_c (NDArray): Массив для результатов
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._cvi.get_function('c_index'),
            args=(data, N, D, pairs, labels, s_min_idx, s_min, s_max_idx, s_max, s_c),
        )

    def os(
            self, *,
            data: NDArray,
            N: int,
            D: int,
            centroids: NDArray,
            K: int,
            labels: NDArray,
            uniq_labels: NDArray,
            o_val: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для метрики OS.
        
        Аргументы:
            data (NDArray): Исходные данные
            N (int): Количество объектов
            D (int): Размерность данных
            centroids (NDArray): Центроиды кластеров
            K (int): Количество кластеров
            labels (NDArray): Метки кластеров
            uniq_labels (NDArray): Уникальные метки
            o_val (NDArray): Массив для результатов
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._cvi.get_function('os'),
            args=(data, N, D, centroids, K, labels, uniq_labels, o_val),
        )

    def external_crosstab(
            self, *,
            N: int,
            uniq_classes: NDArray,
            classes: NDArray,
            classes_k: int,
            uniq_labels: NDArray,
            labels: NDArray,
            labels_k: int,
            crosstab_matrix: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для кросс-таблицы.
        
        Аргументы:
            N (int): Количество объектов
            uniq_classes (NDArray): Уникальные классы
            classes (NDArray): Массив классов
            classes_k (int): Количество классов
            uniq_labels (NDArray): Уникальные метки
            labels (NDArray): Метки кластеров
            labels_k (int): Количество кластеров
            crosstab_matrix (NDArray): Матрица для кросс-таблицы
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._cvi.get_function('external_crosstab'),
            args=(N, uniq_classes, classes, classes_k, uniq_labels, labels, labels_k, crosstab_matrix),
        )

    def external_pairwise(
            self, *,
            N: int,
            classes: NDArray,
            labels: NDArray,
            tp_val: NDArray,
            fp_val: NDArray,
            fn_val: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для кросс-таблицы.

        Аргументы:
            N (int): Количество объектов
            uniq_classes (NDArray): Уникальные классы
            classes (NDArray): Массив классов
            classes_k (int): Количество классов
            uniq_labels (NDArray): Уникальные метки
            labels (NDArray): Метки кластеров
            labels_k (int): Количество кластеров
            crosstab_matrix (NDArray): Матрица для кросс-таблицы

        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._cvi.get_function('external_pairwise'),
            args=(N, classes, labels, tp_val, fp_val, fn_val),
        )

    def kmeans_labels(
            self, *,
            X: NDArray,
            centers: NDArray,
            N: int,
            K: int,
            D: int,
            labels: NDArray,
    ) -> KernelInvocation:
        """Создает вызов ядра для меток K-means.
        
        Аргументы:
            X (NDArray): Исходные данные
            centers (NDArray): Центроиды кластеров
            N (int): Количество объектов
            K (int): Количество кластеров
            D (int): Размерность данных
            labels (NDArray): Массив для меток
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._kmeans.get_function('kmeans_labels'),
            args=(X, centers, N, K, D, labels)
        )

    def kmeans_sse(
            self, *,
            X: NDArray,
            centers: NDArray,
            labels: NDArray,
            sse: NDArray,
            N: int,
            D: int,
    ) -> KernelInvocation:
        """Создает вызов ядра для SSE K-means.
        
        Аргументы:
            X (NDArray): Исходные данные
            centers (NDArray): Центроиды кластеров
            labels (NDArray): Метки кластеров
            sse (NDArray): Массив для SSE
            N (int): Количество объектов
            D (int): Размерность данных
            
        Возвращает:
            KernelInvocation: Объект вызова ядра
        """
        return KernelInvocation(
            kernel=self._kmeans.get_function('kmeans_sse'),
            args=(X, centers, labels, sse, N, D)
        )


MIDDLEWARE = Middleware()  # Глобальный экземпляр Middleware для использования
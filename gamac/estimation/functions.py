import cupy as cp
import numpy as np
from cupy.typing import NDArray

from gamac.estimation.container import EstimationContainer
from gamac.kernels import MIDDLEWARE, BATCH_SIZE


def mcr(container: EstimationContainer) -> float:
    """
    Вычисляет метрику межклассовой дисперсии (MCR), используя GPU вычисления.

    Параметры:
    ----------
    container : EstimationContainer
        Контейнер, содержащий данные для оценки метрики. Включает следующие поля:
            * data - входные данные
            * n - количество выборок
            * d - размерность признаков
            * labels - метки классов
            * n_w - общее число элементов внутри класса
            * n_b - общее число элементов между классами

    Возвращаемое значение:
    ----------------------
    float
        Отрицательное значение метрики межклассовой дисперсии (чем меньше абсолютная величина результата, тем лучше разделение).

    Примечания:
    -----------
    Эта функция реализует вычислительный конвейер на GPU, распределяя работу по блокам и потокам CUDA.
    Использует вспомогательные массивы `gpu_s_w` и `gpu_s_b`, рассчитывая внутриклассовую (`s_w`) и межклассовую (`s_b`) дисперсию соответственно.
    Результат представляет собой отношение средних значений внутриклассовых и межклассовых дисперсий, умноженное на минус единицу для удобства интерпретации.
    """

    # Аллокация временных буферов на GPU
    gpu_s_w = cp.empty(shape=container.n, dtype=cp.float32)
    gpu_s_b = cp.empty(shape=container.n, dtype=cp.float32)

    # Запуск ядра MCR на GPU
    MIDDLEWARE.mcr(
        data=container.data,
        N=container.n,
        D=container.d,
        labels=container.labels,
        s_w=gpu_s_w,
        s_b=gpu_s_b,
    ).invoke(
        grid=(container.n // BATCH_SIZE + 1,),
        blocks=(BATCH_SIZE,),
    )

    # Суммируем значения на GPU и извлекаем итоговые суммы
    s_w = gpu_s_w.sum().item()
    s_b = gpu_s_b.sum().item()

    # Формула расчета метрики MCR
    result = (s_w / container.n_w) / (s_b / container.n_b)
    return -result


def br(container: EstimationContainer) -> float:
    """
    Функция вычисляет критерий Байеса-Рао (BR) для кластеров данных.

    Параметры:
    ----------
    container : EstimationContainer
        Объект контейнера, хранящий следующую информацию:
            k       : Количество кластеров
            clusters: Список списков индексов точек, принадлежащих каждому кластеру
            cent_dists: Матрица расстояний от каждой точки до центра своего кластера
            n      : Общее количество объектов в наборе данных

    Возвращаемое значение:
    ---------------------
    float
        Значение критерия Байеса-Рао. Чем больше отрицательная величина, тем лучше качество разбиения на кластеры.

    Примечания:
    -----------
    Для каждого кластера рассчитывается среднее квадратичное отклонение от центра кластера (вариация).
    Затем суммируются взвешенные логарифмы вариаций всех кластеров, нормализованные общим количеством объектов.
    Если контейнер пуст (n == 0), возвращается фиксированное большое отрицательное значение (-100000).
    """

    result_acc = 0.0
    for cl_idx in range(container.k):
        # Получаем количество объектов в данном кластере
        cl_n = len(container.clusters[cl_idx])
        # Рассчитываем расстояние от каждой точки до центра текущего кластера
        cl_d = container.cent_dists[cl_idx]
        # Среднее квадратическое отклонение от центра кластера
        cl_var = (cl_d * cl_d).mean().item()
        # Добавляем константу стабилизации, чтобы избежать деления на ноль
        var_stable = 1.0 + cl_var + 1e-6
        # Накопление результатов с учётом числа объектов в кластере
        result_acc += cl_n * np.log(var_stable)
    # Нормализация результата общим числом объектов
    if container.n:
        result = result_acc / container.n
        return -result
    return -100000


def sym(container: EstimationContainer) -> float:
    """
    Вычисляет симметричную меру качества разделения кластеров (Symmetric Cluster Separation Measure).

    Параметры:
    ----------
    container : EstimationContainer
        Объект контейнера, содержащий необходимые данные для расчёта меры:
            * cent_matrix   : матрица центров кластеров
            * sym_dists     : список матриц расстояний между объектами одного кластера

    Возвращаемое значение:
    ----------------------
    float
        Симметричная мера разделения кластеров. Большее значение означает лучшее разделение.

    Примечания:
    -----------
    Мера основана на двух компонентах:
        1. Максимальное расстояние между центрами кластеров (d_k)
        2. Средняя сумма внутренних расстояний в каждом кластере (e_k)

    Результатом является соотношение максимального расстояния между центрами и средней внутренней связности.
    """
    # Максимальное расстояние между центрами кластеров
    d_k = np.max(container.cent_matrix)
    # Среднее значение сумм расстояний внутри каждого кластера
    e_k = np.mean([
        cl_sym.sum().item() for cl_sym in container.sym_dists
    ]).__float__()
    # Итоговая мера — отношение максимума внешних расстояний к среднему внутреннему расстоянию
    return d_k / e_k


def c_index(container: EstimationContainer) -> float:
    """
    Вычисляет индекс C-кластеризации (C-index) набора данных.

    Параметры:
    ----------
    container : EstimationContainer
        Объект контейнера, содержащий всю необходимую информацию для расчета индекса:
            * data    : Входные данные
            * n       : Число объектов
            * d       : Размерность пространства признаков
            * labels  : Метки классов/кластеров
            * n_w     : Число пар точек внутри кластеров

    Возвращаемое значение:
    ----------------------
    float
        Индекс C-кластеризации, нормированный в диапазоне [-1, 0]. Более высокие отрицательные значения означают лучшую кластеризацию.

    Примечания:
    -----------
    Этот метод основан на сравнении суммарных расстояний между всеми парами точек внутри и вне кластеров.
    Используется оптимизированный алгоритм на GPU для ускорения расчетов.
    Подход заключается в вычислении минимального и максимального возможных суммарных расстояний среди пар точек и оценке реального распределения расстояний относительно этих границ.
    """

    # Расчет общего количества пар и определение диапазона индекса
    pairs, n_w = container.n * (container.n - 1) // 2, container.n_w
    s_min_idx, s_max_idx = n_w, pairs - n_w
    # Временные массивы на GPU для хранения промежуточных данных
    gpu_s_c = cp.empty(shape=(1,), dtype=cp.float32)
    gpu_s_min = cp.empty(shape=n_w, dtype=cp.float32)
    gpu_s_max = cp.empty(shape=n_w, dtype=cp.float32)

    # Выполнение параллельных вычислений на GPU
    MIDDLEWARE.c_index(
        data=container.data,
        N=container.n,
        D=container.d,
        pairs=pairs,
        labels=container.labels,
        s_min_idx=s_min_idx,
        s_min=gpu_s_min,
        s_max_idx=s_max_idx,
        s_max=gpu_s_max,
        s_c=gpu_s_c,
    ).invoke(
        grid=(container.n // 16 + 1, container.n // 16 + 1),
        blocks=(16,16),
    )
    # Извлечение результатов с GPU
    s_c = gpu_s_c.item()
    s_min = gpu_s_min.sum().item()
    s_max = gpu_s_max.sum().item()
    # Финальный расчет индекса C
    result = (s_c - s_min) / (s_max - s_min)
    return -result


def f1(
        classes: NDArray,
        labels: NDArray,
) -> float:
    """Реализация F1 меры

    Args:
        classes: NDArray
        labels: NDArray
    Returns:
        float
    """
    N = len(classes)
    assert N == len(labels)
    uniq_classes = cp.unique(classes)
    classes_k = len(uniq_classes)

    uniq_labels = cp.unique(labels)
    labels_k = len(uniq_labels)

    gpu_crosstab = cp.empty(shape=(classes_k, labels_k), dtype=cp.uint32)

    MIDDLEWARE.crosstab(
        N=N,
        uniq_classes=uniq_classes,
        classes=classes,
        classes_k=classes_k,
        uniq_labels=uniq_labels,
        labels=labels,
        labels_k=labels_k,
        crosstab_matrix=gpu_crosstab
    ).invoke(
        grid=(classes_k // 16 + 1, labels_k // 16 + 1),
        blocks=(16, 16),
    )
    crosstab_matrix = cp.asnumpy(gpu_crosstab)

    a_arr, b_arr = crosstab_matrix.sum(axis=1), crosstab_matrix.sum(axis=0)

    f1_val = 0.0
    for j, nj in enumerate(b_arr):
        a_max_val = 0.0
        for i, ni in enumerate(a_arr):
            nij = crosstab_matrix[i, j]
            precision, recall = nij / ni, nij / nj
            div = precision + recall
            if div > 1e-6:
                ij_val = 2 * precision * recall / div
                a_max_val = max(a_max_val, ij_val)
        f1_val += nj / N * a_max_val
    return f1_val


def os(container: EstimationContainer):
    """
    Вычисляет показатель Optimized Silhouette (OS) для оценки качества кластеризации.

    Параметры:
    ----------
    container : EstimationContainer
        Объект, предоставляющий доступ к данным и параметрам кластеризации:
            * data          : исходные данные
            * n            : количество объектов
            * d            : размерность пространства признаков
            * centroids     : координаты центроидов кластеров
            * k            : количество кластеров
            * labels        : метки принадлежности объектов к кластерам
            * uniq_labels_gpu: уникальные метки кластеров на устройстве GPU
            * cent_matrix   : матрица расстояний между центроидами

    Возвращаемое значение:
    ----------------------
    float
        Оптимизированный коэффициент силуэта (Optimized Silhouette Score). Больше отрицательных значений соответствует лучшему качеству кластеризации.

    Примечания:
    -----------
    Показатель OS включает два основных компонента:
        1. Внутренняя компактность кластеров (O-value)
        2. Межкластерная сепарабельность (S-value)

    O-value вычисляется на GPU параллельно, тогда как S-value рассчитывается последовательным перебором ближайших соседних кластеров.
    Итоговый результат нормализуется делением первой величины на вторую и выводится с обратным знаком для улучшения интерпретируемости.
    """

    # Создаем временный массив на GPU для накопления O-значений
    o_val_gpu = cp.empty(shape=container.n, dtype=cp.float32)

    # Запуск вычислений на GPU
    MIDDLEWARE.os(
        data=container.data,
        N=container.n,
        D=container.d,
        centroids=container.centroids,
        K=container.k,
        labels=container.labels,
        uniq_labels=container.uniq_labels_gpu,
        o_val=o_val_gpu,
    ).invoke(
        grid=(container.n // BATCH_SIZE + 1,),
        blocks=(BATCH_SIZE,),
    )
    # Сумма накопленных O-значений
    o_val = o_val_gpu.sum().item()
    # Последовательный расчёт S-значения
    s_val = 0.0
    for x_idx, x_row in enumerate(container.cent_matrix):
        s_x = float('inf') # Минимальная дистанция до ближайшего соседнего кластера
        for y_idx, y in enumerate(x_row):
            if x_idx == y_idx:
                continue
            s_x = min(s_x, y)
        s_val += s_x
    # Вычисляем финальный результат и возвращаем его с обратным знаком
    result = o_val / s_val
    return -result

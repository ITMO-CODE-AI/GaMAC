import cupy as cp
import numpy as np
import pandas as pd
import json
import os


def _read_data(data_path: str) -> cp.ndarray:
    """Читает данные из CSV-файла и преобразует в массив cupy.
    
    Аргументы:
        data_path (str): Путь к директории с данными
        
    Возвращает:
        cp.ndarray: Массив данных в формате cupy (float32)
        
    Пример:
        >>> data = _read_data('../data/dataset1')
        >>> type(data)
        <class 'cupy.ndarray'>
    """
    data = pd.read_csv(f'{data_path}/gen.csv', header=None).values
    return cp.asarray(data, dtype=cp.float32, order='C')


def _read_partitions(data_path: str) -> list[cp.ndarray]:
    """Читает разбиения данных из CSV-файла.
    
    Аргументы:
        data_path (str): Путь к директории с данными
        
    Возвращает:
        list[cp.ndarray]: Список разбиений (каждое как cupy массив)
        
    Пример:
        >>> partitions = _read_partitions('../data/dataset1')
        >>> len(partitions)
        5
    """
    partitions = pd.read_csv(f'{data_path}/partitions.csv', header=None).values
    return [
        cp.asarray(partition, dtype=cp.int32)
        for partition in partitions
    ]


def _read_measures(data_path: str) -> dict[str, list[float]]:
    """Читает метрики качества из JSON-файла.
    
    Аргументы:
        data_path (str): Путь к директории с данными
        
    Возвращает:
        dict[str, list[float]]: Словарь с метриками качества
        
    Пример:
        >>> measures = _read_measures('../data/dataset1')
        >>> 'silhouette' in measures
        True
    """
    with open(f'{data_path}/measures.json') as fp:
        content = json.load(fp)
    return {
        measure_name: eval(values_str)
        for measure_name, values_str in content.items()
    }


def _read_features(data_path: str) -> np.ndarray:
    """Читает признаки данных из текстового файла.
    
    Аргументы:
        data_path (str): Путь к директории с данными
        
    Возвращает:
        np.ndarray: Массив признаков
        
    Пример:
        >>> features = _read_features('../data/dataset1')
        >>> features.shape
        (10,)
    """
    with open(f'{data_path}/features.txt') as fp:
        content = fp.readline()
    features_list = eval(content)
    return np.array(features_list)


def traverse_data():
    """Генератор для обхода всех наборов данных в директории.
    
    Yields:
        dict: Словарь с информацией о наборе данных:
            - name: название набора
            - data: массив данных
            - partitions: список разбиений
            - measures: метрики качества
            - features: признаки
            
    Пример использования:
        for dataset in traverse_data():
            print(dataset['name'], dataset['data'].shape)
    """
    real_path = os.path.realpath(__file__)
    tests_path = os.path.dirname(real_path)
    for data_name in sorted(os.listdir(f'{tests_path}/data')):
        data_path = f'{tests_path}/data/{data_name}'
        yield {
            "name": data_name,
            "data": _read_data(data_path),
            "partitions": _read_partitions(data_path),
            "measures": _read_measures(data_path),
            "features": _read_features(data_path),
        }
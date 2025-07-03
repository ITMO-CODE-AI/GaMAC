"""Основной скрипт этапа предобработки данных"""
import torch
import numpy as np
from cupy.typing import NDArray
from numpy import ndarray
from pandas import DataFrame
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
)

from gamac.data.table_preprocessing import table_preprocessing
from gamac.data.modal_preprocessing import get_clip_embeddings

DataFrameType = NDArray
LabelsType = NDArray


class DataHandler:
    """Класс обработчика данных для дальнейшей кластеризации"""

    def __init__(self, clip_model, clip_processor, device, batch_size: int = 32, verbose: bool = False):
        """
        Инициализация класса

        Args:
            batch_size (int): размер батча для обработки
        """
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

    def table_preprocess(self, table: DataFrame) -> ndarray:
        """
        Предобработка таблицы

        Args:
            table (DataFrame): табличные данные
        Returns:
            ndarray
        """
        return table_preprocessing(df=table, verbose=self.verbose)

    def text_image_preprocess(self, text: list[str], image: list[Image]) -> ndarray:
        """
        Предобработка текста и изображения

        Args:
            text (list[str]): Список текстовых данных
            image (list[Image]): Список изображений
        Returns:
            ndarray
        """
        return get_clip_embeddings(
            model=self.clip_model,
            processor=self.clip_processor,
            img_inputs=image,
            txt_inputs=text,
            batch_size=self.batch_size,
            device=self.device
        )

    def concat_dataset(self, table_dataset: ndarray, img_txt_dataset: ndarray) -> ndarray:
        """
        Конкатенация двух датафреймов

        Args:
            table_dataset (ndarray): массив таблички
            img_txt_dataset (ndarray): массив текста и изображений
        Returns:
            ndarray: конкатенированный датафрейм
        """

        return np.concatenate((table_dataset, img_txt_dataset), axis=1)

    def run(self, table: DataFrame, text: list[str], image: list[Image]) -> DataFrameType:
        """
        Запуск обработки
        Args:
            table (DataFrame): датафрейм таблицы
            text (list[str]): Список текстовых данных
            image (list[Image]): Список изображений
        Returns:
            np.array: единый датафрейм
        """
        dataset = None
        table_dataset = None
        img_txt_dataset = None

        if table is not None:
            table_dataset = self.table_preprocess(table)

        if text is not None and image is not None:
            img_txt_dataset = self.text_image_preprocess(text, image)

        if table_dataset is not None and img_txt_dataset is not None:
            dataset = self.concat_dataset(table_dataset, img_txt_dataset)

        return dataset if dataset is not None else table_dataset if table_dataset is not None else img_txt_dataset

from typing import List

from numpy import ndarray
from pandas import DataFrame
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
)

from gamac.data.table_preprocessing import table_preprocessing
from gamac.data.modal_preprocessing import get_clip_embeddings


class DataHandler:
    """Класс обработчика данных для дальнейшей кластеризации"""

    def __init__(self, device: str = "cuda"):
        """
        Инициализация класса

        Args:
            device (str, optional): Defaults to "cuda".
        """
        # Проверка на len в img_inputs/txt_inputs
        self.modal_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", device_map=device
        )
        self.modal_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", device_map=device
        )
        self.device = device

    def table_preprocess(self, table: DataFrame) -> ndarray:
        """
        Предобработка таблицы

        Args:
            table (DataFrame): табличные данные
        Returns:
            ndarray
        """
        return table_preprocessing(input_dataframe=table)

    def text_image_preprocess(self, text: List[str], image: List[Image]) -> ndarray:
        """
        Предобработка текста и изображения

        Args:
            text (List[str]): Список текстовых данных
            image (List[Image]): Список изображений
        Returns:
            ndarray
        """
        return get_clip_embeddings(
            model=self.model,
            processor=self.processor,
            img_inputs=image,
            txt_inputs=text,
            batch=1,
            device=self.device,
        )

    def run(self, table: DataFrame, text: List[str], image: List[Image]):
        """
        Запуск обработки
        Args:
            table (DataFrame): датафрейм таблицы
            text (List[str]): Список текстовых данных
            image (List[Image]): Список изображений
        Returns:
            np.array: единый датафрейм
        """
        table_dataset = self.table_preprocess(table)
        img_txt_dataset = self.text_image_preprocess(text, image)

        dataset = self.concat_dataset(table_dataset, img_txt_dataset)

        return dataset

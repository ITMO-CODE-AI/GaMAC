import pytest
import torch
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock

from gamac.data.modal_preprocessing import get_clip_embeddings


@pytest.fixture
def sample_images():
    """Фикстура с тестовыми изображениями"""
    return [
        Image.new('RGB', (224, 224), color='red'),
        Image.new('RGB', (224, 224), color='blue')
    ]


@pytest.fixture
def sample_texts():
    """Фикстура с тестовыми текстами"""
    return ["red image", "blue image"]


def test_input_length_mismatch(sample_images):
    """Тест на несовпадение количества изображений и текстов"""
    with pytest.raises(ValueError, match="Количество изображений и текстовых описаний должно совпадать"):
        get_clip_embeddings("openai/clip-vit-base-patch32", sample_images, ["only one text"])


def test_model_loading_cpu():
    """Тест загрузки модели на CPU"""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('transformers.CLIPModel.from_pretrained') as mock_model, \
         patch('transformers.CLIPProcessor.from_pretrained') as mock_processor:

        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        get_clip_embeddings("openai/clip-vit-base-patch32", [], [])

        mock_model.assert_called_once()
        mock_processor.assert_called_once()


def test_model_loading_gpu():
    """Тест загрузки модели на GPU"""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('transformers.CLIPModel.from_pretrained') as mock_model, \
         patch('transformers.CLIPProcessor.from_pretrained') as mock_processor:

        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        get_clip_embeddings("openai/clip-vit-base-patch32", [], [])

        mock_model.return_value.to.assert_called_with('cuda')

from typing import List

from PIL import Image
import numpy as np
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
)


def get_clip_embeddings(model: CLIPModel, processor: CLIPProcessor, device,
                        img_inputs: List[Image], txt_inputs: List[str], batch_size: int = 32):
    """Получение CLIP эмбеддингов

    Args:
        model (CLIPModel)
        processor (CLIPProcessor)
        img_inputs: список изображений
        txt_inputs: список текстовых описаний
        batch_size (int): размер батча для обработки
    """
    # Проверка на совпадение длины входных данных
    if len(img_inputs) != len(txt_inputs):
        raise ValueError("Количество изображений и текстовых описаний должно совпадать")

    # Предварительное выделение памяти для эмбеддингов
    with torch.no_grad():
        # Получаем размерность эмбеддингов на тестовом батче
        test_inputs = processor(
            text=txt_inputs[:1],
            images=img_inputs[:1],
            return_tensors="pt"
        ).to(device)
        test_outputs = model(**test_inputs)

        text_embed_dim = test_outputs.text_embeds.shape[1]
        image_embed_dim = test_outputs.image_embeds.shape[1]
        total_embed_dim = text_embed_dim + image_embed_dim
        embeds = np.zeros((len(txt_inputs), total_embed_dim))

        # Обработка батчами
        for i in range(0, len(txt_inputs), batch_size):
            batch_text = txt_inputs[i:i+batch_size]
            batch_images = img_inputs[i:i+batch_size]

            inputs = processor(
                text=batch_text,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            outputs = model(**inputs)

            # Конкатенация и сохранение эмбеддингов
            batch_embeds = torch.cat(
                (outputs.text_embeds, outputs.image_embeds),
                dim=1
            ).cpu().numpy()

            embeds[i:i+batch_size] = batch_embeds

    # Очистка памяти
    torch.cuda.empty_cache()

    return embeds

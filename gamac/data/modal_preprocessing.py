from typing import List

from PIL import Image
import numpy as np
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
)


def get_clip_embeddings(model_name: str, img_inputs: List[Image], txt_inputs: List[str], batch_size: int = 32):
    """Получение CLIP эмбеддингов

    Args:
        model_name (str): название модели: "openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32", "../models/CLIP-GmP-ViT-L-14"
        img_inputs: список изображений
        txt_inputs: список текстовых описаний
        batch_size (int): размер батча для обработки
    """
    # Проверка на совпадение длины входных данных
    if len(img_inputs) != len(txt_inputs):
        raise ValueError("Количество изображений и текстовых описаний должно совпадать")

    # Загрузка модели и процессора
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

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
    del model, processor
    torch.cuda.empty_cache()

    return embeds

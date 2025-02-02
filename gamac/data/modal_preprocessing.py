from typing import List

from PIL import Image
import numpy as np
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
)


def get_clip_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    img_inputs: List[Image],
    txt_inputs: List[str],
    batch: int = 1,
    device: str = "cuda",
):
    """
    Получение эмбеддингов image+text

    Args:
        model (CLIPModel): модель CLIP
        processor (CLIPProcessor): процессор CLIP
            Пример: "openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32", "../models/CLIP-GmP-ViT-L-14"
        img_inputs (List[Image]) - image inputs
        txt_inputs (List[str]) - text inputs
        batch (int, optional): Defaults to 1.
        device (str, optional): Defaults to "cuda".

    Returns:
        np.array: возврат эмбеддингов картинка+текст
    """
    for i in range(0, len(txt_inputs), batch):
        inputs = processor(
            text=txt_inputs[i : i + batch],
            images=img_inputs[i : i + batch],
            return_tensors="pt",
        )
        model.to(device)
        inputs = inputs.to(device)

        outputs = model(**inputs)

        # Проверка на len в img_inputs/txt_inputs
        if i == 0:
            embeds = np.zeros(
                (
                    len(img_inputs),
                    outputs.text_embeds.shape[1] + outputs.image_embeds.shape[1],
                )
            )

        for k in range(0, batch):
            embeds[i + k] = list(outputs.text_embeds[k].detach().cpu().numpy()) + list(
                outputs.image_embeds[k].detach().cpu().numpy()
            )

        torch.cuda.empty_cache()

    # очистка памяти GPU
    del model, processor, inputs
    torch.cuda.empty_cache()

    return embeds

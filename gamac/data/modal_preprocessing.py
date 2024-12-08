import numpy as np
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
)


def get_clip_embeddings(
    model_name: str, img_inputs, txt_inputs, batch: int = 1, device: str = "cuda"
):
    """
    Получение эмбеддингов image+text

    Args:
        name (str): название модели: "openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32", "../models/CLIP-GmP-ViT-L-14"
        img_inputs: image inputs
        txt_inputs: text inputsd
        batch (int, optional): Defaults to 1.
        device (str, optional): Defaults to "cuda".

    Returns:
        np.array: возврат эмбеддингов картинка+текст
    """
    # Проверка на len в img_inputs/txt_inputs
    model = CLIPModel.from_pretrained(model_name, device_map=device)
    processor = CLIPProcessor.from_pretrained(model_name, device_map=device)

    for i in range(0, len(txt_inputs), batch):
        inputs = processor(
            text=txt_inputs[i : i + batch],
            images=img_inputs[i : i + batch],
            return_tensors="pt",
        )
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

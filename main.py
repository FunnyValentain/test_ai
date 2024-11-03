import tensorflow as tf
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Загрузка предобученной модели CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


def generate_clothing_description(image_path):
    # Загрузка изображения
    image = Image.open(image_path)

    # Обработка изображения
    inputs = processor(text=["a photo of clothing", "a photo of a shirt", "a photo of a dress",
                             "a photo of pants", "a photo of a jacket", "a photo of shoes"],
                       images=image, return_tensors="pt", padding=True)

    # Получение выходных данных из модели
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # Получаем логиты для изображений
    probs = logits_per_image.softmax(dim=1)  # Применяем softmax для получения вероятностей

    # Получаем описание с наивысшей вероятностью
    descriptions = ["a photo of clothing", "a photo of a shirt", "a photo of a dress",
                    "a photo of pants", "a photo of a jacket", "a photo of shoes"]

    max_prob_index = torch.argmax(probs)
    return descriptions[max_prob_index.item()]


# Пример использования

image_path = input('Укажите путь к фотке ') # Укажите путь к вашему изображению
description = generate_clothing_description(image_path)
print(f"Description: {description}")
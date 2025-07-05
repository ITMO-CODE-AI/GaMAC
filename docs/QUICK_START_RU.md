[![README](https://img.shields.io/badge/README-md-red.svg)](../README_RU.md)
[![ru](https://img.shields.io/badge/lang-ru-red.svg)](QUICK_START_RU.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](QUICK_START.md)

## Быстрый старт

1. Установка зависимостей
    <code>pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128</code>

2. Запуск Gamac

Основные примеры лежат в [notebooks](notebooks/)

2.1. Автоматическая кластеризация табличных, визуальных и текстовых данных
---
```
from torchvision.datasets import CIFAR100
from gamac.autoclustering import Gamac

# Import data
cifar100 = CIFAR100('../data/cifar', download=True, train=False)

cifar_txt = [f'a photo of {cifar100.classes[img[1]]}' for img in cifar100][:100]
cifar_img = [img[0] for img in cifar100][:100]
cifar_table = pd.DataFrame(cifar100.targets[:100])

# Run Gamac
df, best_model = Gamac().run(table=cifar_table, text=cifar_txt, image=cifar_img)

# Get a best model and dataset clusters
print(f'best_model.model: {best_model.model}')
print(f'clusters: {best_model.model.labels_}')
```
---
2.2. Автоматическая кластеризация табличных данных
---
```
import pandas as pd
from sklearn.datasets import load_iris
from gamac.autoclustering import Gamac

# Import data
data = load_iris(as_frame=True)
table = data['data']

# Run Gamac
df, best_model = Gamac().run(table=table, text=None, image=None)

# Get a best model and dataset clusters
print(f'best_model.model: {best_model.model}')
print(f'clusters: {best_model.model.labels_}')
```
---
2.3. Автоматическая кластеризация визуальных и текстовых данных
---
```
from torchvision.datasets import CIFAR100
from gamac.autoclustering import Gamac

# Import data
cifar100 = CIFAR100('../data/cifar', download=True, train=False)

cifar_txt = [f'a photo of {cifar100.classes[img[1]]}' for img in cifar100][:100]
cifar_img = [img[0] for img in cifar100][:100]

# Run Gamac
df, best_model = Gamac().run(table=None, text=cifar_txt, image=cifar_img)

# Get a best model and dataset clusters
print(f'best_model.model: {best_model.model}')
print(f'clusters: {best_model.model.labels_}')
```
---
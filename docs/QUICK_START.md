[![README](https://img.shields.io/badge/README-md-blue.svg)](../README.md)
[![ru](https://img.shields.io/badge/lang-ru-red.svg)](QUICK_START_RU.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](QUICK_START.md)

## Quick Start

1. Install all project requirements
    <code>pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128</code>

2. Run Gamac

Check examples in [notebooks](notebooks/)

2.1. Autoclustering with table, text and image data
---
```
from torchvision.datasets import CIFAR100
from gamac.autoclustering import Gamac

# Import data
cifar100 = CIFAR100('../data/cifar', download=True, train=False)

cifar_txt = [f'a photo of {cifar100.classes[img[1]]}' for img in cifar100][:100]
cifar_img = [img[0] for img in cifar100][:100]
cifar_table = pd.DataFrame(cifar100.targets[:100])

df, best_model = Gamac().run(table=cifar_table, text=cifar_txt, image=cifar_img)

print(f'best_model.model: {best_model.model}')
print(f'clusters: {best_model.model.labels_}')
```
---
2.2. Autoclustering with only table data
---
```
import pandas as pd
from sklearn.datasets import load_iris
from gamac.autoclustering import Gamac

# Import data
data = load_iris(as_frame=True)
table = data['data']

df, best_model = Gamac().run(table=table, text=None, image=None)

print(f'best_model.model: {best_model.model}')
print(f'clusters: {best_model.model.labels_}')
```
---
2.3. Autoclustering with only text and image data
---
```
from torchvision.datasets import CIFAR100
from gamac.autoclustering import Gamac

# Import data
cifar100 = CIFAR100('../data/cifar', download=True, train=False)

cifar_txt = [f'a photo of {cifar100.classes[img[1]]}' for img in cifar100][:100]
cifar_img = [img[0] for img in cifar100][:100]

df, best_model = Gamac().run(table=None, text=cifar_txt, image=cifar_img)

print(f'best_model.model: {best_model.model}')
print(f'clusters: {best_model.model.labels_}')
```
---

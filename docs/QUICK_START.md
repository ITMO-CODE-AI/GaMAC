[![README](https://img.shields.io/badge/README-md-blue.svg)](../README.md)
[![ru](https://img.shields.io/badge/lang-ru-red.svg)](QUICK_START_RU.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](QUICK_START.md)

## Quick Start


1. Run Gamac

Check examples in following [notebook](https://github.com/ITMO-CODE-AI/GaMAC/blob/develop/notebooks/basic_example.ipynb)

1.1. Autoclustering with table, text and image data
---
```
from torchvision.datasets import CIFAR100
from gamac.autoclustering import Gamac

# Import data
cifar100 = CIFAR100('../data/cifar', download=True, train=False)

cifar_txt = [f'a photo of {cifar100.classes[img[1]]}' for img in cifar100]
cifar_img = [img[0] for img in cifar100]
cifar_table = pd.DataFrame(cifar100.targets)

result = Gamac().run(table=cifar_table, text=cifar_txt, image=cifar_img)

print(f'result.model: {result.model}')
print(f'clusters: {result.model.labels_}')
```
---
1.2. Autoclustering with only table data
---
```
import pandas as pd
from sklearn.datasets import load_digits
from gamac.autoclustering import Gamac

# Import data
data = load_digits(as_frame=True)
table = data['data']

result = Gamac().run(table=table, text=None, image=None)

print(f'result.model: {result.model}')
print(f'clusters: {result.model.labels_}')
```
---
1.3. Autoclustering with only text and image data
---
```
from torchvision.datasets import CIFAR100
from gamac.autoclustering import Gamac

# Import data
cifar100 = CIFAR100('../data/cifar', download=True, train=False)

cifar_txt = [f'a photo of {cifar100.classes[img[1]]}' for img in cifar100]
cifar_img = [img[0] for img in cifar100]

result = Gamac().run(table=None, text=cifar_txt, image=cifar_img)

print(f'result.model: {result.model}')
print(f'clusters: {result.model.labels_}')
```
---

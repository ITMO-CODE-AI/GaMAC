[![README](https://img.shields.io/badge/README-md-red.svg)](../README_RU.md)
[![ru](https://img.shields.io/badge/lang-ru-red.svg)](QUICK_START_RU.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](QUICK_START.md)

## Быстрый старт

1. Установка зависимостей
    <code>pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128</code>

2. Запуск Gamac
---
2.1. Автоматическая кластеризация табличных, визуальных и текстовых данных
---
```
import pandas as pd
from PIL import Image
from gamac.autoclustering import Gamac

# Import data
data: pd.DataFrame = ... # table data
image: list[Image] = ... # image data
text: list[str] = ... # text data

df, optimal = Gamac().run(table=data, text=text, image=image)

print(f'optimal.model: {optimal.model}')
print(f'clusters: {optimal.model.labels_}')
```
---
2.2. Автоматическая кластеризация табличных данных
---
```
import pandas as pd
from PIL import Image
from gamac.autoclustering import Gamac

# Import data
data: pd.DataFrame = ... # table data

df, optimal = Gamac().run(table=data, text=None, image=None)

print(f'optimal.model: {optimal.model}')
print(f'clusters: {optimal.model.labels_}')
```
---
2.3. Автоматическая кластеризация визуальных и текстовых данных
---
```
import pandas as pd
from PIL import Image
from gamac.autoclustering import Gamac

# Import data
image: list[Image] = ... # image data
text: list[str] = ... # text data

df, optimal = Gamac().run(table=None, text=text, image=image)

print(f'optimal.model: {optimal.model}')
print(f'clusters: {optimal.model.labels_}')
```

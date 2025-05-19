<p align="center"><h1 align="center">GaMAC</h1></p>
<p align="center">
	<a href="https://itmo.ru/"><img src="https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg"></a>
	<img src="https://img.shields.io/github/license/CTLab-ITMO/CoolPrompt?style=BadgeStyleOptions.DEFAULT&logo=opensourceinitiative&logoColor=white&color=blue" alt="license">
</p>
<p align="center">
</p>
<br>

[![ru](https://img.shields.io/badge/lang-ru-red.svg)](README_RU.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)


---
## Интро

<overview>
GaMAC это модуль на языке Python для автоматизированного машинного обучения задачам кластеризации. Проект был запущен в 2024 году Лабораторией Искусственного Интеллекта Факультета Информационных Технологий и Программирования Университета ИТМО, с тех пор в настоящее время ведется работа над проектом.
</overview>


При поддержке [Фонда Содействия Инновациям](https://fasie.ru/).

![fasie-icon](docs/fasie.svg)



## Содержание

* [Описание](info/OVERVIEW.md)
* [Установка](info/DEPLOY.md)
* [Quick Start](info/QUICK_START.md)
* [Глоссарий](info/GLOSSARY.md)
* [Применение](info/CASE.md)



---

### Минимальные требования

* Ubuntu 22.04 / WSL
* 4 CPU cores, 16 GB RAM;
* GPU: NVIDIA, CUDA 12.6 support, GPU memory size: 10 Gb
* Python>=3.12

### Python зависимости

Список зависимостей находится в [requirements.txt](../requirements.txt).

### Установка и настройка зависимостей

```bash
git clone https://github.com/ITMO-CODE-AI/GaMAC.git
cd GaMAC

pip install -r requirements.txt
```

---

## Быстрый старт

1. Установка зависимостей
    <code>pip install -r requirements.txt</code>

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
---

## Лицензия

Данный проект находится под лицензией Apache 2.0. Подробнее можно ознакомиться в [LICENSE](https://github.com/ITMO-CODE-AI/GaMAC/blob/feature/unit_test_algo/LICENSE).

---

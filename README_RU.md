<p align="center"><h1 align="center">GaMAC   <img src="docs/gamac_itmo.jpg" width="44px" height="44px"></h1></p>
<p align="center">
	<a href="https://itmo.ru/"><img src="https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg"></a>
	<img src="https://img.shields.io/github/license/CTLab-ITMO/CoolPrompt?style=BadgeStyleOptions.DEFAULT&logo=opensourceinitiative&logoColor=white&color=blue" alt="license">
</p>
<p align="center">
</p>
<br>

[![ru](https://img.shields.io/badge/lang-ru-red.svg)](README_RU.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![CI/CD job for 4090](https://github.com/ITMO-CODE-AI/GaMAC/actions/workflows/ci_cd_4090.yml/badge.svg)](https://github.com/ITMO-CODE-AI/GaMAC/actions/workflows/ci_cd_4090.yml)
[![CI/CD job for 4070](https://github.com/ITMO-CODE-AI/GaMAC/actions/workflows/ci_cd_4070.yml/badge.svg)](https://github.com/ITMO-CODE-AI/GaMAC/actions/workflows/ci_cd_4070.yml)
[![CI/CD job for 3070](https://github.com/ITMO-CODE-AI/GaMAC/actions/workflows/ci_cd_3070.yml/badge.svg)](https://github.com/ITMO-CODE-AI/GaMAC/actions/workflows/ci_cd_3070.yml)

---
## GaMAC

<overview>
GaMAC это модуль на языке Python для автоматизированного машинного обучения задач кластеризации c GPU ускорением. 
Проект был запущен в 2024 году Лабораторией Искусственного Интеллекта Факультета Информационных Технологий и Программирования Университета ИТМО, с тех пор в настоящее время ведется работа над проектом.
</overview>


При поддержке [Фонда Содействия Инновациям](https://fasie.ru/).

![fasie-icon](docs/fasie.svg)



## Содержание

* [Описание](docs/OVERVIEW_RU.md)
* [Установка](docs/DEPLOY_RU.md)
* [Quick Start](docs/QUICK_START_RU.md)
* [Глоссарий](docs/GLOSSARY_RU.md)
* [Применение](docs/CASE_RU.md)

---

### Каталог проекта

```
├── data	# Внешние данные
├── docs	# Документация проекта
├── gamac   # Модуль проекта
|   ├── algorithms	# Имплементация алгоритмов кластеризации
|   ├── bin	# Файлы моделей
|   ├── data	# Модуль обработки данных
|   ├── estimation	# Модуль оценки результатов кластеризации
|   ├── meta	# Модуль мета-классификатора
|   |   ├── accessors	# Данные разметки
|   |   ├── impl	# Имплементация мета-классификатора
|   |   └── storage	# Память мета-классификатора
|   ├── pipeline	# Модуль поиска алгоритмов
|   ├── tests	# Модуль тестов
|   |   ├── data	# Данные для тестов
|   |   └── unit	# Юнит-тесты
|   └── autoclustering.py	# Основной скрипт интерфейс GaMAC
├── notebooks	# Ноутбуки проекта
|   ├── examples	# Примеры запуска GaMAC
|   |   ├── basic_example.ipynb	# Базовые примеры запуска GaMAC
|   |   └── example_on_realdata.ipynb	# Примеры запуска на промышленных кейсах
|   └── experiments	# Эксперименты разработки
|   |   ├── cvi_accuracy.ipynb	# Оценка мета-классификатора
|   |   ├── embedder_testing.ipynb	# Оценка различных кодировщиков изображений-текста
|   |   ├── experiment_on_optimizers.ipynb	# Оценка различных методов поиска
|       └── devops	# DevOps эксперименты
├── flake8
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README_RU.md	# README на русском
├── README.md	# README на английском
└── requirements.txt
```

### Минимальные требования

* Ubuntu 22.04 / WSL
* 4 CPU cores, 16 GB RAM;
* GPU: NVIDIA, CUDA 12.8 support, GPU memory size: 10 Gb
* Python>=3.12

### Python зависимости

Список зависимостей находится в [requirements.txt](requirements.txt).

### Установка и настройка зависимостей

Через pip
```bash
pip install -U --extra-index-url https://test.pypi.org/simple/ Gamac --extra-index-url https://download.pytorch.org/whl/cu128
```

Через git
```bash
git clone https://github.com/ITMO-CODE-AI/GaMAC.git
cd GaMAC

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

---

## Быстрый старт

1. Запуск Gamac

Основные примеры лежат в [ноутбуке](https://github.com/ITMO-CODE-AI/GaMAC/blob/develop/notebooks/examples/basic_example.ipynb)

1.1. Автоматическая кластеризация табличных, визуальных и текстовых данных
---
```
from torchvision.datasets import CIFAR100
from gamac.autoclustering import Gamac

# Import data
cifar100 = CIFAR100('../data/cifar', download=True, train=False)

cifar_txt = [f'a photo of {cifar100.classes[img[1]]}' for img in cifar100]
cifar_img = [img[0] for img in cifar100]
cifar_table = pd.DataFrame(cifar100.targets)

# Run Gamac
result = Gamac().run(table=cifar_table, text=cifar_txt, image=cifar_img)

# Get a best model and dataset clusters
print(f'best_model.model: {result.model}')
print(f'clusters: {result.model.labels_}')
```
---
1.2. Автоматическая кластеризация табличных данных
---
```
import pandas as pd
from sklearn.datasets import load_digits
from gamac.autoclustering import Gamac

# Import data
data = load_digits(as_frame=True)
table = data['data']

# Run Gamac
result = Gamac().run(table=table, text=None, image=None)

# Get a best model and dataset clusters
print(f'best_model.model: {result.model}')
print(f'clusters: {result.model.labels_}')
```
---
1.3. Автоматическая кластеризация визуальных и текстовых данных
---
```
from torchvision.datasets import CIFAR100
from gamac.autoclustering import Gamac

# Import data
cifar100 = CIFAR100('../data/cifar', download=True, train=False)

cifar_txt = [f'a photo of {cifar100.classes[img[1]]}' for img in cifar100]
cifar_img = [img[0] for img in cifar100]

# Run Gamac
result = Gamac().run(table=None, text=cifar_txt, image=cifar_img)

# Get a best model and dataset clusters
print(f'best_model.model: {result.model}')
print(f'clusters: {result.model.labels_}')
```
---

## Прикладное использование

1. Компьютерное зрение и анализ изображений
- Анализ изображений с точки зрения отдельных параметров (цветов, яркости, контрастности и т.д.)
- Автоматическая категоризация товаров по визуальным признакам (e-commerce).

2. Обработка естественного языка (NLP)
- Выявление шаблонов в социальных сетях (анализ настроений, тематические тренды).

3. Биоинформатика и медицинская диагностика
- Выявление различных типов клеток и тканей (например, в гистологии).
- Анализ геномных данных для выявления паттернов мутаций.

4. Финансы и финтех
- Сегментация клиентов банков для определения групп связанных заемщиков.
- Обнаружение аномалий в транзакциях (мошенничество, отмывание денег).

5. Рекомендательные системы
- Кластеризация контента (фильмы, музыка, товары) для улучшения рекомендаций.

6. Маркетинг и поведенческая аналитика
- Сегментация аудитории для таргетированной рекламы.

7. Геопространственный анализ
- Кластеризация точек интереса (POI) для урбанистики и логистики.

## Abl


## Лицензия

Данный проект находится под лицензией Apache 2.0. Подробнее можно ознакомиться в [LICENSE](https://github.com/ITMO-CODE-AI/GaMAC/blob/feature/unit_test_algo/LICENSE).

---

## Дополнительная информация

* [О Jupyter notebooks](https://github.com/ITMO-CODE-AI/GaMAC/blob/develop/notebooks/About%20notebooks.md)

* [О DevOps](https://github.com/ITMO-CODE-AI/GaMAC/blob/develop/notebooks/experiments/devops/About%20DevOps.md)

---
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
GaMAC is a Python module for automated machine learning on clustering tasks with a GPU acceleraion. 
The project was started in 2024 by ITMO AI Laboratory of Information Technologies and Programming Faculty, and since then we are currently working on this project. 
</overview>

Sponsored by [Foundation for Promotion of Innovation](https://fasie.ru/).

![fasie-icon](docs/fasie.svg)



## Contents

* [Description](docs/OVERVIEW.md)
* [Deploy](docs/DEPLOY.md)
* [Quick Start](docs/QUICK_START.md)
* [Glossary](docs/GLOSSARY.md)
* [Use Case](docs/CASE.md)


---

### Project catalog

```
├── data	# External datasets
├── docs	# Project documentation
├── gamac   # Project module
|   ├── algorithms	# Implementations of clustering algorithms
|   ├── bin	# Models files
|   ├── data	# Data processing module
|   ├── estimation	# Estimation of clustering results module
|   ├── meta	# Meta-learning module
|   |   ├── accessors	# Markup data
|   |   ├── impl	# Meta-learning implementation
|   |   └── storage	# Meta-learning storage
|   ├── pipeline	# Autoclustering optimization module
|   ├── tests	# Tests
|   |   ├── data	# Data for tests
|   |   └── unit	# Unit-tests
|   └── autoclustering.py	# Autoclustering main interface script
├── notebooks	# Project notebooks
|   ├── examples	# Examples of running GaMAC
|   |   ├── basic_example.ipynb	# Basic examples of GaMAC
|   |   ├── cvi_accuracy.ipynb	# Meta-learning implementation
|   |   └── storage	# Meta-learning storage
|   ├── experiments
├── package-lock.json
├── README.md
└── src
    ├── app.js
    ├── models.js
    ├── routes.js
    └── utils
        ├── another.js
        ├── constants.js
        └── index.js
```


### Minimal requirements

* Ubuntu 22.04 / WSL
* 4 CPU cores, 16 GB RAM;
* GPU: NVIDIA, CUDA 12.8 support, GPU memory size: 10 Gb
* Python>=3.12

### Python dependencies

List of dependencies can be found in [requirements.txt](requirements.txt).

### Installation and dependencies setup

With pip
```bash
pip install -U --extra-index-url https://test.pypi.org/simple/ Gamac --extra-index-url https://download.pytorch.org/whl/cu128
```

With git
```bash
git clone https://github.com/ITMO-CODE-AI/GaMAC.git
cd GaMAC

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

---

## Quick Start

1. Run Gamac

Check examples in following [notebook](https://github.com/ITMO-CODE-AI/GaMAC/blob/develop/notebooks/examples/basic_example.ipynb)

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

## Practical applications
1. Computer Vision and Image Analysis
- Image analysis based on specific parameters (colors, brightness, contrast, etc.).
- Automatic product categorization by visual features (e-commerce).

2. Natural Language Processing (NLP)
- Pattern detection in social media (sentiment analysis, thematic trends).

3. Bioinformatics and Medical Diagnostics
- Identification of different cell and tissue types (e.g., in histology).
- Genomic data analysis for mutation pattern detection.

4. Finance and Fintech
- Bank customer segmentation to identify groups of related borrowers.
- Anomaly detection in transactions (fraud, money laundering).

5. Recommender Systems
- Content clustering (movies, music, products) to improve recommendations.

6. Marketing and Behavioral Analytics
- Audience segmentation for targeted advertising.

7. Geospatial Analysis
- Clustering points of interest (POI) for urban planning and logistics.



## License

This project is protected under the Apache 2.0 License. For more details, refer to the [LICENSE](https://github.com/ITMO-CODE-AI/GaMAC/blob/feature/unit_test_algo/LICENSE) file.

---

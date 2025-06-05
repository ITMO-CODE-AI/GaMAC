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
[![CD job for 4070](https://github.com/ITMO-CODE-AI/GaMAC/actions/workflows/cd_4070.yml/badge.svg)](https://github.com/ITMO-CODE-AI/GaMAC/actions/workflows/cd_4070.yml)
[![CD job for 3070](https://github.com/ITMO-CODE-AI/GaMAC/actions/workflows/cd_3070.yml/badge.svg)](https://github.com/ITMO-CODE-AI/GaMAC/actions/workflows/cd_3070.yml)

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

### Minimal requirements

* Ubuntu 22.04 / WSL
* 4 CPU cores, 16 GB RAM;
* GPU: NVIDIA, CUDA 12.8 support, GPU memory size: 10 Gb
* Python>=3.12

### Python dependencies

List of dependencies can be found in [requirements.txt](requirements.txt).

### Installation and dependencies setup

```bash
git clone https://github.com/ITMO-CODE-AI/GaMAC.git
cd GaMAC

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

---

## Quick Start

1. Install all project requirements
    <code>pip install -r requirements.txt</code>

2. Run Gamac
   
2.1. Autoclustering with table, text and image data
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
2.2. Autoclustering with only table data
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
2.3. Autoclustering with only text and image data
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

## License

This project is protected under the Apache 2.0 License. For more details, refer to the [LICENSE](https://github.com/ITMO-CODE-AI/GaMAC/blob/feature/unit_test_algo/LICENSE) file.

---

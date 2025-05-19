<p align="center"><h1 align="center">GaMAC</h1><img src="docs/gamac_itmo.jpg" width="50px" height="50px"></p>
<p align="center">
	<a href="https://itmo.ru/"><img src="https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg"></a>
	<img src="https://img.shields.io/github/license/CTLab-ITMO/CoolPrompt?style=BadgeStyleOptions.DEFAULT&logo=opensourceinitiative&logoColor=white&color=blue" alt="license">
	
</p>
<p align="center">
</p>
<br>

[![ru](https://img.shields.io/badge/lang-en-red.svg)](README_RU.md)
[![en](https://img.shields.io/badge/lang-pt--br-green.svg)](README.md)


---
## Overview

<overview>
GaMAC is a Python module for automated machine learning on clustering tasks. The project was started in 2024 by ITMO AI Laboratory of Information Technologies and Programming Faculty, and since then we are currently working on this project. 
</overview>

Sponsored by [Foundation for Promotion of Innovation](https://fasie.ru/).

![fasie-icon](info/fasie.svg)



## Содержание

* [Description](info/OVERVIEW.md)
* [Quick Start](info/QUICK_START.md)
* [Гайд по развертыванию](info/DEPLOY.md)
* [API-спецификация](info/API.md)
* [Glossary](info/GLOSSARY.md)
* [Use Case](info/CASE.md)


---

## Quick Start

- Install all project requirements
    <code>pip install -r requirements.txt</code>

- Run Gamac
1. Autoclustering with table, text and image data
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

## License

This project is protected under the Apache 2.0 License. For more details, refer to the [LICENSE](https://github.com/ITMO-CODE-AI/GaMAC/blob/feature/unit_test_algo/LICENSE) file.

---


## Contacts

**WIP**

## Citation

**WIP**

---


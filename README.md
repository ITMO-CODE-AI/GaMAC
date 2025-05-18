<p align="center"><h1 align="center">GaMAC</h1></p>
<p align="center">
	<a href="https://itmo.ru/"><img src="https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg"></a>
	<img src="https://img.shields.io/github/license/CTLab-ITMO/CoolPrompt?style=BadgeStyleOptions.DEFAULT&logo=opensourceinitiative&logoColor=white&color=blue" alt="license">
</p>
<p align="center">
	</p>
<br>


---
## Overview

<overview>
GaMAC is a Python module for automated machine learning on clustering tasks. The project was started in 2024 by ITMO AI Laboratory of Information Technologies and Programming Faculty, and since then we are currently working on this project. 
</overview>

---

## Quick Start

- Install all project requirements
    <code>pip install -r requirements.txt</code>

- Run Gamac
1. With table, text and image data
---
```
from gamac.autoclustering import Gamac

import pandas as pd
from PIL import Image

data: pd.DataFrame = ...
image: list[Image] = ...
text: list[str] = ...

df, optimal = Gamac().run(table=data, text=text, image=image)

print(f'optimal.model: {optimal.model}')
print(f'clusters: {optimal.model.labels_}')
```

## Data


---

## License

This project is protected under the Apache 2.0 License. For more details, refer to the [LICENSE]() file.

---


## Contacts

**WIP**

## Citation

**WIP**

---


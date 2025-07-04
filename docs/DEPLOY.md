[![README](https://img.shields.io/badge/README-md-blue.svg)](../README.md)
[![ru](https://img.shields.io/badge/lang-ru-red.svg)](DEPLOY_RU.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](DEPLOY.md)

### Minimal requirements

* Ubuntu 22.04 / WSL
* 4 CPU cores, 16 GB RAM;
* GPU: NVIDIA, CUDA 12.6 support, GPU memory size: 10 Gb
* Python>=3.12

### Python dependencies

List of dependencies can be found in [requirements](../requirements.txt) directory.

### Installation and dependencies setup of the library

```bash
git clone https://github.com/ITMO-CODE-AI/GaMAC.git
cd GaMAC

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

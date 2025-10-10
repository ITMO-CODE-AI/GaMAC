[![README](https://img.shields.io/badge/README-md-red.svg)](../README_RU.md)
[![ru](https://img.shields.io/badge/lang-ru-red.svg)](DEPLOY_RU.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](DEPLOY.md)

### Минимальные требования

* Ubuntu 22.04 / WSL
* Четырехядерный центральный процессор, 16 Гб оперативной памяти
* GPU: NVIDIA, поддержка cuda 12.8, минимальный размер графической памяти 10 ГБ
* Python>=3.12

### Python зависимости

Список зависимостей находится в папке [requirements](../requirements.txt)

### Установка и настройка зависимостей библиотеки

Преднастройка для работы с pyrfr
```bash
sudo apt install swig libboost-all-dev python3.12-dev
```

Установка GaMAC

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

### Запуск юнит тестов

Выполните следующую команду из корня репозитория:

```bash
python -m pytest gamac/tests/unit/ -v
```
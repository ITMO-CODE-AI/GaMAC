FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    python3 -m pip install --extra-index-url=https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

COPY gamac gamac
COPY devops devops
COPY test-data test-data

CMD . .venv/bin/activate && \
    python devops/test.py
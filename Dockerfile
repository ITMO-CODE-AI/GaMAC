FROM nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12

USER root

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

USER 1001

COPY requierements.txt requierements.txt

RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    python3 -m pip install --extra-index-url=https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu126 -r requierements.txt

COPY gamac gamac

CMD . .venv/bin/activate && cd gamac/tests && exec k_means_tests.py
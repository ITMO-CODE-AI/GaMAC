FROM nvidia/cuda:12.5.1-devel-ubuntu24.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

COPY pyCUDA_requierements.txt pyCUDA_requierements.txt

RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    python3 -m pip install -r pyCUDA_requierements.txt

COPY sandbox sandbox

CMD . .venv/bin/activate && exec python sandbox/kmeans_gpu_pycuda.py
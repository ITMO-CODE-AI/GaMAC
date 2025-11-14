FROM altairzero/gamac:2025.10.14.12.53

RUN . .venv/bin/activate && \
    python3 -m pip install jupyter pycuda minio

COPY test-data all-data

COPY test-notebooks all-notebooks

EXPOSE 8888

ENV NAME World

CMD . .venv/bin/activate && \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
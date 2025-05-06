import time

import cupy as cp
import pandas as pd

from gamac.autoclustering import Gamac


def main():
    data = pd.read_csv('test-data/gen.csv')
    data = data.head(1000).values
    gpu_data = cp.array(data, dtype=cp.float32)
    df, optimal = Gamac().run(table=gpu_data, text=None, image=None)

    print(f'optimal.model: {optimal.model}')
    print(f'clusters: {optimal.model.labels_}')


if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)

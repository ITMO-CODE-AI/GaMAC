import time

import cupy as cp
import pandas as pd

from gamac.autoclustering import Gamac


def main():
    data = pd.read_csv('test-data/gen.csv').head(1000).values
    gpu_data = cp.array(data, dtype=cp.float32)
    df, clusters = Gamac().run(table=gpu_data, text=None, image=None)

    print(df)
    print(f'clusters: {clusters}')


if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)

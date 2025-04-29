import time

import cupy as cp
import pandas as pd

from gamac.autoclustering import Gamac


def main():
    data = pd.read_csv('test-data/gen.csv').values
    gpu_data = cp.array(data, dtype=cp.float32)
    Gamac().run(table=gpu_data, text=None, image=None)

if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)
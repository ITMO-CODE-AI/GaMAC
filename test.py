import time

import pandas as pd

from gamac.autoclustering import Gamac


def main():
    data = pd.read_csv('test-data/gen.csv').head(1000)
    df, optimal = Gamac().run(table=data, text=None, image=None)

    print(f'optimal.model: {optimal.model}')
    print(f'clusters: {optimal.model.labels_}')


if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)

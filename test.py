import time

import pandas as pd

from gamac.autoclustering import Gamac
from gamac.estimation.internal import Internal
from gamac.estimation.functions import f1

def main():
    # data = pd.read_csv('test-data/gen.csv').head(1000)
    data = pd.read_csv('test-data/OpenML_100-plants-margin_orig.csv')
    data = data.drop("class", axis=1)
    # df, optimal = Gamac(target_measures=(Internal.SYM, Internal.BR, Internal.OS)).run(table=data, text=None, image=None)
    df, optimal = Gamac().run(table=data, text=None, image=None)

    print(f'optimal.model: {optimal.model}')
    print(f'clusters: {optimal.model.labels_}')
    # print(f'F-1 score: {f1()}')

if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)

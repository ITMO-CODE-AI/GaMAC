import time
import os
import pandas as pd

from gamac.autoclustering import Gamac
from gamac.estimation.internal import Internal
from gamac.estimation.functions import f1

DATA = os.getenv('DATA', "gen.csv") 

def main():
    data = pd.read_csv(f'test-data/{DATA}')
    data = data.drop("class", errors='ignore', axis=1)
    # df, optimal = Gamac(target_measures=(Internal.SYM, Internal.BR, Internal.OS)).run(table=data, text=None, image=None)
    df, optimal = Gamac().run(table=data, text=None, image=None)

    print(f'optimal.model: {optimal.model}')
    print(f'clusters: {optimal.model.labels_}')

if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)

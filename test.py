import time
import os
import pandas as pd

from gamac.autoclustering import Gamac
from gamac.estimation.internal import Internal
from gamac.estimation.functions import f1

DATA = os.getenv('DATA', "gen.csv")
TARGET_MEASURES = os.getenv('TARGET_MEASURES', "BR,OS,MCR,SYM")

def main():
    measures = {"BR": Internal.BR, "OS": Internal.OS, "MCR": Internal.MCR, "SYM": Internal.SYM}
    used_measures = [measures[x] for x in TARGET_MEASURES.split(sep=',')]
    data = pd.read_csv(f'test-data/{DATA}')
    data = data.drop("class", errors='ignore', axis=1)
    df, optimal = Gamac(target_measures=tuple(used_measures)).run(table=data, text=None, image=None)
    #df, optimal = Gamac().run(table=data, text=None, image=None)
    print(f'used data: {DATA}')
    print(f'used measures: {used_measures}')
    print(f'optimal.model: {optimal.model}')
    print(f'clusters: {optimal.model.labels_}')

if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)

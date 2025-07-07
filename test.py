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
    # used_measures = TARGET_MEASURES.split(sep=',')
    if '.csv' in DATA:
        data = pd.read_csv(f'test-data/{DATA}')
    elif '.parquet':
        pd.read_parquet(f'test-data/{DATA}')
    if 'class' in data.columns:
        classes = data['class'].tolist()
    data = data.drop('class', errors='ignore', axis=1)
    print(f'used data: {DATA}')
    print(f'used measures: {used_measures}')
    result = Gamac(target_measures=tuple(used_measures)).run(table=data, text=None, image=None, classes=classes)
    #df, optimal = Gamac().run(table=data, text=None, image=None)
    print(f'optimal.model: {result.model}')
    print(f'clusters: {result.model.labels_}')
    print(f'F1 score: {result.f1_score}')


if __name__ == '__main__':
    start = time.time()
    main()
    print('Work time:', time.time() - start)

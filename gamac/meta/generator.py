import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from gamac.src.meta.collector import DatasetForMetaCVI, DatasetInfoCollector
from gamac.src.meta.producers import ProducerProvider
from gamac.src.meta.reducers import ReducerProvider


def launch(data_name):
    dataframe = pd.read_csv(f'data/{data_name}/orig.csv')
    if 'class' in dataframe.columns:
        dataframe = dataframe.drop(columns=['class'])
    data = dataframe.values
    normalised = MinMaxScaler().fit_transform(data)

    for reducer in ReducerProvider.get_all():
        print(f"===== {reducer.name} =====")
        try:
            dataset = DatasetForMetaCVI(data_name, reducer, normalised)
        except ValueError:
            continue
        collector = DatasetInfoCollector(dataset)
        for producer in ProducerProvider.get_all():
            print(f"--- {producer.name} ---")
            partition = producer.fit_predict(dataset)
            collector.save(partition, producer)
        collector.persist()


if __name__ == "__main__":
    pass

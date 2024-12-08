import random

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from gamac.src.meta.reducers import Reducer
from gamac.src.meta.render import COLORS, scatter_image
from gamac.src.meta.storage import create_data_dir, write_gen_data, write_partitions, write_producers, \
    PARTITIONS_TO_ESTIMATE, DATA_ROOT


class DatasetForMetaCVI:
    def __init__(self, name: str, reducer: Reducer, original: np.ndarray):
        self.name, self.reducer = name, reducer
        self.data = reducer.fit_transform(original)
        self.data = MinMaxScaler().fit_transform(self.data)
        self.data_path = f'{name}/{reducer.name}'

        create_data_dir(self.data_path)
        write_gen_data(self.data_path, self.data)


class DatasetInfoCollector:

    def __init__(self, dataset: DatasetForMetaCVI):
        self.dataset = dataset
        self.registered = list()

    def save(self, partition, producer):
        if np.max(partition) > 8 or np.max(partition) < 1 or self._is_too_noisy(partition):
            print("INVALID LABELS")
        else:
            self.registered.append((partition, producer))

    def _is_too_noisy(self, partition):
        noise = [1 for label in partition if label == -1]
        return len(noise) / len(partition) > 0.1

    def persist(self):
        random.shuffle(self.registered)
        chosen_indices = self._choose_most_different()
        producers, partitions = list(), list()
        for idx, chose_index in enumerate(chosen_indices):
            partition, producer = self.registered[chose_index]
            producers.append(
                {
                    "algo": producer.name,
                    "params": producer.algo.get_params()
                }
            )
            partitions.append(partition)
            self._scatter(partition, idx)
        write_producers(self.dataset.data_path, producers)
        write_partitions(self.dataset.data_path, partitions)

    def _choose_most_different(self):
        n = len(self.registered)
        print(f"OBTAINED {n} PARTITIONS for {self.dataset.data_path}")
        similarity_matrix = np.zeros((n, n))
        for x_idx in range(n):
            for y_idx in range(x_idx):
                x, y = self.registered[x_idx][0], self.registered[y_idx][0]
                score = metrics.fowlkes_mallows_score(x, y)
                similarity_matrix[x_idx, y_idx] = score
                similarity_matrix[y_idx, x_idx] = score
        evicted = set()
        while len(evicted) < n - PARTITIONS_TO_ESTIMATE:
            most_similar_idx = np.argmax(similarity_matrix)
            cur_evicted = most_similar_idx % n
            evicted.add(cur_evicted)
            similarity_matrix[cur_evicted, :] = 0
            similarity_matrix[:, cur_evicted] = 0

        return set(np.arange(n).tolist()) - evicted

    def _scatter(self, labels: np.ndarray, p_idx: int):
        x, y = self.dataset.data[:, 0], self.dataset.data[:, 1]
        colors = [COLORS[label] for label in labels]
        img_path = f'{DATA_ROOT}/{self.dataset.data_path}/img-{p_idx}.png'
        scatter_image(x, y, colors, img_path)

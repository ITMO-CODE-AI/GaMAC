"""Основной скрипт этапа подсказки мер"""
import os
import pickle

from gamac.data.data_pipeline import DataFrameType
from gamac.estimation.internal import Internal
import cupy as cp
import numpy as np

from gamac.kernels import BATCH_SIZE, MIDDLEWARE


def load_pickle(file: str):
    real_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(real_path)
    root_path = os.path.dirname(dir_path)
    with open(f"{root_path}/bin/{file}", 'rb') as fp:
        return pickle.load(fp)


class CVIPredictor:
    """summary"""

    BUCKETS = 128
    MEASURES_BY_INDEX = [
        Internal.OS,
        Internal.SYM,
        Internal.BR,
        Internal.MCR
    ]

    def __init__(self):
        self.extractor = load_pickle('extractor.pkl')
        self.model = load_pickle('classifier.pkl')

    def run(self, df: DataFrameType) -> Internal:
        meta_features = self._meta_features(df)
        transformed = self._transform(meta_features)
        return self._predict(transformed)

    def _meta_features(self, df: DataFrameType) -> np.ndarray:
        n, dims = df.shape
        d_max = float('-inf')
        quotient, remainder = divmod(n, self.BUCKETS)

        gpu_partial_arr = cp.empty(shape=(BATCH_SIZE, n), dtype=np.float32)
        gpu_stats_arr = cp.empty(shape=(BATCH_SIZE, self.BUCKETS, 4), dtype=np.float32)

        mfs_accumulator = np.zeros(shape=(self.BUCKETS, 4), dtype=np.float32)

        iterations = n // BATCH_SIZE + (0 if n % BATCH_SIZE == 0 else 1)

        for iter_idx in range(iterations):
            print(f'=== CVI prediction iteration {iter_idx + 1}/{iterations} ====')
            batch_start = iter_idx * BATCH_SIZE
            batch_size = min(BATCH_SIZE, n - batch_start)

            MIDDLEWARE.meta_dist_partial(
                N=n,
                D=dims,
                data=df,
                batch_start=batch_start,
                batch_size=batch_size,
                partial_dists=gpu_partial_arr,
            ).invoke(
                grid=(1, n),
                blocks=(batch_size, 1),
            )

            gpu_partial_arr.sort(axis=1)
            cpu_d_max_arr = cp.asnumpy(gpu_partial_arr[:, -1])
            d_max = max(d_max, *cpu_d_max_arr[:batch_size])

            MIDDLEWARE.meta_dist_stat(
                Q=quotient,
                R=remainder,
                N=n,
                sorted_dists=gpu_partial_arr,
                batch_size=batch_size,
                dist_stats=gpu_stats_arr,
            ).invoke(
                grid=(1, self.BUCKETS),
                blocks=(batch_size, 1),
            )

            cpu_stats_arr = cp.asnumpy(gpu_stats_arr)
            batch_accumulator = np.sum(
                cpu_stats_arr[:batch_size], axis=0
            )
            mfs_accumulator += batch_accumulator

        return mfs_accumulator.flatten(order='F') / d_max / n

    def _transform(self, meta_features: np.ndarray) -> np.ndarray:
        return self.extractor.transform([meta_features])[0]

    def _predict(self, transformed: np.ndarray) -> Internal:
        cvi_index = self.model.predict([transformed])[0]
        return self.MEASURES_BY_INDEX[cvi_index]

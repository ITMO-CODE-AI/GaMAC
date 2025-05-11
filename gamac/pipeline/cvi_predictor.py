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

    @staticmethod
    def run(df: DataFrameType) -> Internal:
        meta_features = CVIPredictor._meta_features(df)
        transformed = CVIPredictor._transform(meta_features)
        return CVIPredictor._predict(transformed)

    @staticmethod
    def _meta_features(df: DataFrameType) -> np.ndarray:
        n, dims = df.shape
        d_max = float('-inf')
        buckets = CVIPredictor.BUCKETS
        quotient, remainder = divmod(n, buckets)

        gpu_d_max_arr = cp.empty(shape=(BATCH_SIZE,), dtype=np.float32)
        gpu_sorted_arr = cp.empty(shape=(BATCH_SIZE, n), dtype=np.float32)
        gpu_stats_arr = cp.empty(shape=(BATCH_SIZE, buckets, 4), dtype=np.float32)

        mfs_accumulator = np.zeros(shape=(buckets, 4), dtype=np.float32)

        iterations = n // BATCH_SIZE + (0 if n % BATCH_SIZE == 0 else 1)

        for iter_idx in range(iterations):
            print(f'=== CVI prediction iteration {iter_idx + 1}/{iterations} ====')
            batch_start = iter_idx * BATCH_SIZE
            batch_size = min(BATCH_SIZE, n - batch_start)

            MIDDLEWARE.meta_dist_sort(
                N=n,
                D=dims,
                data=df,
                batch_start=batch_start,
                batch_size=batch_size,
                sorted_dists=gpu_sorted_arr,
                max_dists=gpu_d_max_arr
            ).invoke(
                grid=(1, n),
                blocks=(batch_size, 1),
            )

            cpu_d_max_arr = cp.asnumpy(gpu_d_max_arr)
            batch_max = max(cpu_d_max_arr[:batch_size])
            d_max = max(d_max, batch_max)

            MIDDLEWARE.meta_dist_stat(
                Q=quotient,
                R=remainder,
                N=n,
                sorted_dists=gpu_sorted_arr,
                batch_size=batch_size,
                dist_stats=gpu_stats_arr,
            ).invoke(
                grid=(1, buckets),
                blocks=(batch_size, 1),
            )

            cpu_stats_arr = cp.asnumpy(gpu_stats_arr)
            batch_accumulator = np.sum(
                cpu_stats_arr[:batch_size], axis=0
            )
            mfs_accumulator += batch_accumulator

        return mfs_accumulator.flatten(order='F') / d_max / n

    @staticmethod
    def _transform(meta_features: np.ndarray) -> np.ndarray:
        extractor = load_pickle('classifier-extractor.pkl')
        return extractor.transform([meta_features])[0]

    @staticmethod
    def _predict(transformed: np.ndarray) -> Internal:
        model = load_pickle('classifier-model.pkl')
        cvi_index = model.predict([transformed])[0]
        return CVIPredictor.MEASURES_BY_INDEX[cvi_index]

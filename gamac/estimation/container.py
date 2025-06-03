import cupy as cp
import numpy as np
from cupy.typing import NDArray

from gamac.kernels import MIDDLEWARE, BATCH_SIZE


class EstimationContainer:
    NOISE_THRESHOLD = 0.1

    def __init__(
            self,
            data: NDArray,
            labels: NDArray,
            uniq_labels_gpu: NDArray,
            uniq_labels_arr: np.ndarray,
    ):
        self.data = data
        self.labels = labels
        self.uniq_labels_gpu = uniq_labels_gpu
        self.uniq_labels_arr = uniq_labels_arr
        self.n, self.d = data.shape
        self._clusters = None
        self._centroids = None
        self._cent_dists = None
        self._sym_dists = None
        self._cent_matrix = None

    @staticmethod
    def _denoise(df, labels):
        denoised = labels != -1
        denoised_labels = labels[denoised]
        denoised_ratio = len(denoised_labels) / len(labels)
        noise_ratio = 1.0 - denoised_ratio
        if noise_ratio > EstimationContainer.NOISE_THRESHOLD:
            noice_perc = int(noise_ratio * 100)
            raise ValueError(f"Received {noice_perc}% objects with noise labels")
        return df[denoised], denoised_labels

    @staticmethod
    def _check_max_clusters(data, uniq_labels):
        n, k = len(data), len(uniq_labels)
        max_clusters = int(np.cbrt(2 * n)) + 1
        if k > max_clusters:
            raise ValueError(f"Received too many ({k}) clusters for dataset of {n} objects")

    @staticmethod
    def create(df, labels):
        uniq_labels_gpu = cp.unique(labels)
        uniq_labels_arr = uniq_labels_gpu.get()

        if -1 in uniq_labels_arr:
            _data, _labels = EstimationContainer._denoise(df, labels)
            _uniq_labels_arr = uniq_labels_arr[uniq_labels_arr != -1]
            EstimationContainer._check_max_clusters(_data, _uniq_labels_arr)
            _uniq_labels_gpu = cp.asarray(_uniq_labels_arr, dtype=cp.int32)
            return EstimationContainer(
                data=_data,
                labels=_labels,
                uniq_labels_gpu=_uniq_labels_gpu,
                uniq_labels_arr=_uniq_labels_arr,
            )
        else:
            EstimationContainer._check_max_clusters(df, uniq_labels_arr)
            return EstimationContainer(
                data=df,
                labels=labels,
                uniq_labels_gpu=uniq_labels_gpu,
                uniq_labels_arr=uniq_labels_arr,
            )

    @property
    def k(self) -> int:
        return len(self.uniq_labels_arr)

    @property
    def n_w(self) -> int:
        n_w = 0
        for cl in self.clusters:
            cl_n = len(cl)
            n_w += int(cl_n * (cl_n - 1) / 2)
        return n_w

    @property
    def n_b(self) -> int:
        n = len(self.data)
        return n * (n - 1) - self.n_w

    @property
    def clusters(self):
        if self._clusters is None:
            self._clusters = [
                self.data[self.labels == label]
                for label in self.uniq_labels_arr
            ]
        return self._clusters

    @property
    def centroids(self) -> list:
        if self._centroids is None:
            centroids = cp.empty(shape=(self.k, self.d), dtype=cp.float32)
            MIDDLEWARE.get_centroids(
                data=self.data,
                labels=self.labels,
                N=self.n,
                D=self.d,
                K=self.k,
                uniq_labels=self.uniq_labels_gpu,
                centroids=centroids
            ).invoke(
                grid=(self.k // 16 + 1, self.d // 16 + 1),
                blocks=(16, 16),
            )
            self._centroids = centroids
        return self._centroids

    @property
    def cent_dists(self) -> list:
        if self._cent_dists is None:
            cent_dists_list = list()
            for k_idx, cluster in enumerate(self.clusters):
                cl_n = len(cluster)
                cent_dists = cp.empty(shape=cl_n, dtype=cp.float32)

                MIDDLEWARE.get_cent_dists(
                    cluster=cluster,
                    cl_n=cl_n,
                    D=self.d,
                    centroids=self.centroids,
                    k_idx=k_idx,
                    cent_dists=cent_dists,
                ).invoke(
                    grid=(cl_n // BATCH_SIZE + 1,),
                    blocks=(BATCH_SIZE,),
                )

                cent_dists_list.append(cent_dists)
            self._cent_dists = cent_dists_list
        return self._cent_dists

    @property
    def sym_dists(self) -> list:
        if self._sym_dists is None:
            sym_dists_list = list()
            for k_idx in range(self.k):
                cluster = self.clusters[k_idx]
                cent_dists = self.cent_dists[k_idx]
                cl_n = len(cluster)

                sym_data = cp.empty(shape=cluster.shape, dtype=cp.float32)
                MIDDLEWARE.get_sym_data(
                    cluster=cluster,
                    cl_n=cl_n,
                    D=self.d,
                    centroids=self.centroids,
                    k_idx=k_idx,
                    sym_data=sym_data,
                ).invoke(
                    grid=(cl_n // BATCH_SIZE + 1, self.d),
                    blocks=(BATCH_SIZE, 1),
                )

                sym_dists = cp.empty(shape=cl_n, dtype=cp.float32)
                MIDDLEWARE.get_sym_dists(
                    cluster=cluster,
                    cl_n=cl_n,
                    D=self.d,
                    cent_dists=cent_dists,
                    sym_data=sym_data,
                    sym_dists=sym_dists,
                ).invoke(
                    grid=(cl_n // BATCH_SIZE + 1,),
                    blocks=(BATCH_SIZE,),
                )

                sym_dists_list.append(sym_dists)
            self._sym_dists = sym_dists_list
        return self._sym_dists

    @property
    def cent_matrix(self):
        if self._cent_matrix is None:
            cent_matrix = cp.empty(shape=(self.k, self.k), dtype=cp.float32)
            MIDDLEWARE.get_cent_matrix(
                centroids=self.centroids,
                K=self.k,
                D=self.d,
                cent_matrix=cent_matrix,
            ).invoke(
                grid=(self.k // 16 + 1, self.k // 16 + 1),
                blocks=(16, 16),
            )
            self._cent_matrix = cp.asnumpy(cent_matrix)
        return self._cent_matrix


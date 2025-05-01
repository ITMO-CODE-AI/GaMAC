import os.path

from cupy import RawModule
from cupy.typing import NDArray

from gamac.data.data_pipeline import DataFrameType, LabelsType


def load_module(file: str):
    real_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(real_path)
    with open(f"{dir_path}/{file}", 'r') as fp:
        content = fp.read()
    return RawModule(code=content)


class KernelInvocation:
    def __init__(self, kernel, args):
        self.kernel, self.args = kernel, args

    def invoke(self, grid, blocks):
        self.kernel(grid, blocks, self.args)


class Middleware:
    def __init__(self):
        self._meta = load_module('meta-kernels.c')
        self._cvi = load_module('cvi-kernels.c')
        self._kmeans = load_module('kmeans-kernels.c')

    def get_centroids(
            self, *,
            data: DataFrameType,
            labels: LabelsType,
            N: int,
            D: int,
            K: int,
            centroids: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('get_centroids'),
            args=(data, labels, N, D, K, centroids),
        )

    def get_cent_dists(
            self, *,
            cluster: NDArray,
            cl_n: int,
            D: int,
            centroids: NDArray,
            k_idx: int,
            cent_dists: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('get_cent_dists'),
            args=(cluster, cl_n, D, centroids, k_idx, cent_dists),
        )
        
    def get_cent_matrix(
            self, *,
            centroids: NDArray,
            K: int,
            D: int,
            cent_matrix: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('get_cent_matrix'),
            args=(centroids, K, D, cent_matrix),
        )
    
    def get_sym_data(
            self, *,
            cluster: NDArray,
            cl_n: int,
            D: int,
            centroids: NDArray,
            k_idx: int,
            sym_data: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('get_sym_data'),
            args=(cluster, cl_n, D, centroids, k_idx, sym_data),
        )
        
    def get_sym_dists(
            self, *,
            cluster: NDArray,
            cl_n: int,
            D: int,
            cent_dists: NDArray,
            sym_data: NDArray,
            sym_dists: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('get_sym_dists'),
            args=(cluster, cl_n, D, cent_dists, sym_data, sym_dists),
        )

    def mcr(
            self, *,
            data: NDArray,
            N: int,
            D: int,
            labels: NDArray,
            s_w: NDArray,
            s_b: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('mcr'),
            args=(data, N, D, labels, s_w, s_b),
        )

    def c_index(
            self, *,
            data: NDArray,
            N: int,
            D: int,
            pairs: int,
            labels: NDArray,
            s_min_idx: int,
            s_min: NDArray,
            s_max_idx: int,
            s_max: NDArray,
            s_c: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('c_index'),
            args=(data, N, D, pairs, labels, s_min_idx, s_min, s_max_idx, s_max, s_c),
        )

    # def c_index(
    #         self, *,
    #         data: NDArray,
    #         N: int,
    #         D: int,
    #         pairs: int,
    #         batch_start: int,
    #         labels: NDArray,
    #         s_min_idx: int,
    #         s_min: NDArray,
    #         s_max_idx: int,
    #         s_max: NDArray,
    #         s_c: NDArray,
    # ) -> KernelInvocation:
    #     return KernelInvocation(
    #         kernel=self._cvi.get_function('c_index'),
    #         args=(data, N, D, pairs, batch_start, labels, s_min_idx, s_min, s_max_idx, s_max, s_c),
    #     )

    def crosstab(
            self, *,
            N: int,
            classes: NDArray,
            classes_k: int,
            labels: NDArray,
            labels_k: int,
            crosstab_matrix: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('crosstab'),
            args=(N, classes, classes_k, labels, labels_k, crosstab_matrix),
        )

    def kmeans_labels(
            self, *,
            X: NDArray,
            centers: NDArray,
            N: int,
            K: int,
            D: int,
            labels: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._kmeans.get_function('kmeans_labels'),
            args=(X, centers, N, K, D, labels)
        )

    def kmeans_sse(
            self, *,
            X: NDArray,
            centers: NDArray,
            labels: NDArray,
            sse: NDArray,
            N: int,
            D: int,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._kmeans.get_function('kmeans_sse'),
            args=(X, centers, labels, sse, N, D)
        )

MIDDLEWARE = Middleware()

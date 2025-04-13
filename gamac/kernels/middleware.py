from cupy import RawModule
from cupy.typing import NDArray

from gamac.data.data_pipeline import DataFrameType, LabelsType


def load_module(file: str):
    with open(file, 'r') as fp:
        content = fp.read()
    return RawModule(content)


class KernelInvocation:
    def __init__(self, kernel, args):
        self.kernel, self.args = kernel, args

    def invoke(self, grid, blocks):
        self.kernel(grid, blocks, self.args)


class Middleware:
    def __init__(self):
        self._meta = load_module('meta-kernels.c')
        self._cvi = load_module('cvi-kernels.c')

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
            batch_start: int,
            labels: NDArray,
            s_min_idx: int,
            s_min: NDArray,
            s_max_idx: int,
            s_max: NDArray,
            s_c: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('c_index'),
            args=(data, N, D, pairs, batch_start, labels, s_min_idx, s_min, s_max_idx, s_max, s_c),
        )

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

MIDDLEWARE = Middleware()

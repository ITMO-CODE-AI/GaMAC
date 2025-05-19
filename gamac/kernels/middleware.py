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

    def meta_dist_sort(
            self, *,
            N: int,
            D: int,
            data: NDArray,
            batch_start: int,
            batch_size: int,
            sorted_dists: NDArray,
            max_dists: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._meta.get_function('meta_dist_sort'),
            args=(N, D, data, batch_start, batch_size, sorted_dists, max_dists),
        )

    def meta_dist_stat(
            self, *,
            Q: int,
            R: int,
            N: int,
            sorted_dists: NDArray,
            batch_size: int,
            dist_stats: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._meta.get_function('meta_dist_stat'),
            args=(Q, R, N, sorted_dists, batch_size, dist_stats),
        )

    def get_centroids(
            self, *,
            data: DataFrameType,
            labels: LabelsType,
            N: int,
            D: int,
            K: int,
            uniq_labels: NDArray,
            centroids: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('get_centroids'),
            args=(data, labels, N, D, K, uniq_labels, centroids),
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

    def os(
            self, *,
            data: NDArray,
            N: int,
            D: int,
            centroids: NDArray,
            K: int,
            labels: NDArray,
            uniq_labels: NDArray,
            o_val: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('os'),
            args=(data, N, D, centroids, K, labels, uniq_labels, o_val),
        )


    def crosstab(
            self, *,
            N: int,
            uniq_classes: NDArray,
            classes: NDArray,
            classes_k: int,
            uniq_labels: NDArray,
            labels: NDArray,
            labels_k: int,
            crosstab_matrix: NDArray,
    ) -> KernelInvocation:
        return KernelInvocation(
            kernel=self._cvi.get_function('crosstab'),
            args=(N, uniq_classes, classes, classes_k, uniq_labels, labels, labels_k, crosstab_matrix),
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

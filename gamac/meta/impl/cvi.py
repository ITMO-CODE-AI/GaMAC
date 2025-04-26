"""
CPU internal measures implementations
"""

from typing import List, Callable

import numba
import numpy as np


def _get_centroid(data: np.ndarray) -> np.ndarray:
    return np.mean(data, axis=0)


@numba.njit
def _dist(x, y) -> float:
    diff = x - y
    sqr = np.sum(diff * diff)
    return np.sqrt(sqr)


@numba.njit
def _dist_ps(x_obj, data, centroid):
    x_sym = 2 * centroid - x_obj
    d_near = min([_dist(x_sym, x_obj) for x_obj in data])
    return (d_near + 1e-6) * _dist(x_obj, centroid)


@numba.njit
def _get_cent_dists(cluster_data: np.ndarray, centroid: np.ndarray):
    return np.array([_dist(cl_obj, centroid) for cl_obj in cluster_data])


@numba.njit
def _get_sym_dists(cluster_data: np.ndarray, centroid: np.ndarray):
    return np.array([_dist_ps(cl_obj, cluster_data, centroid) for cl_obj in cluster_data])


@numba.njit
def _get_n_w(clusters: List[np.ndarray]) -> int:
    n_w = 0
    for cl in clusters:
        cl_n = len(cl)
        n_w += int(cl_n * (cl_n - 1) / 2)
    return n_w


@numba.njit
def _get_n_b(clusters: List[np.ndarray]) -> int:
    n = 0
    for cl in clusters:
        n += len(cl)
    return n * (n - 1) - _get_n_w(clusters)


@numba.njit
def _get_intra_stat(cluster_data: np.ndarray):
    cluster_n, sum_dist = len(cluster_data), 0.0
    if cluster_n < 2:
        return 1e-6, 1e-6, 1e-6, 1e-6
    max_dist, min_dist = float('-inf'), float('inf')
    for x_obj_idx in range(1, cluster_n):
        for y_obj_idx in range(x_obj_idx):
            x_obj, y_obj = cluster_data[x_obj_idx], cluster_data[y_obj_idx]
            xy_dist = _dist(x_obj, y_obj)
            max_dist = max(xy_dist, max_dist)
            min_dist = min(xy_dist, min_dist)
            sum_dist += xy_dist
    pairs = cluster_n * (cluster_n - 1) / 2
    return min_dist, sum_dist / pairs, max_dist, sum_dist


@numba.njit
def _get_inter_stat(x: np.ndarray, y: np.ndarray):
    x_n, y_n, sum_dist = len(x), len(y), 0.0
    max_dist, min_dist = float('-inf'), float('inf')
    for x_obj in x:
        for y_obj in y:
            xy_dist = _dist(x_obj, y_obj)
            min_dist = min(xy_dist, min_dist)
            max_dist = max(xy_dist, max_dist)
            sum_dist += xy_dist
    pairs = x_n * y_n
    return min_dist, sum_dist / pairs, max_dist, sum_dist


@numba.njit
def _get_wgss(cent_dists: List[np.array]):
    wgss = 0.0
    for cent_dist in cent_dists:
        wgss += np.sum(cent_dist * cent_dist)
    return wgss


class CachedValues:
    def __init__(self, min_val: float, mean_val: float, max_val: float, sum_val: float):
        self.min_val = min_val
        self.mean_val = mean_val
        self.max_val = max_val
        self.sum_val = sum_val


class ArrayStat(CachedValues):
    def __init__(self, arr: np.ndarray):
        super().__init__(
            min_val=np.min(arr),
            mean_val=np.mean(arr),
            max_val=np.max(arr),
            sum_val=np.sum(arr),
        )
        self.arr = arr


class Clusters:
    def __init__(
            self,
            all_data: np.ndarray,
            global_centroid: np.ndarray,
            clusters_data: List[np.ndarray],
            centroids: List[np.ndarray],
            cent_dists: List[ArrayStat],
            sym_dists: List[ArrayStat],
            cartesian_stats: List[List[CachedValues]],
    ):
        self.all_data = all_data
        self.global_centroid = global_centroid

        self.clusters_data = clusters_data
        self.centroids = centroids

        self.cent_dists = cent_dists
        self.sym_dists = sym_dists

        self.cartesian_stats = cartesian_stats

        self.n, self.k = len(all_data), len(clusters_data)
        self.amounts = [len(cl_data) for cl_data in clusters_data]

    @property
    def intra_stats(self) -> List[CachedValues]:
        return [cl_stat[cl_idx] for cl_idx, cl_stat in enumerate(self.cartesian_stats)]

    @staticmethod
    def create(data: np.ndarray, labels: List[int]):
        k = max(labels) + 1
        clusters_data = [list() for _ in range(k)]
        filtered_data = list()

        for x, y in zip(data, labels):
            if y >= 0:
                clusters_data[y].append(x)
                filtered_data.append(x)
        clusters_data = [np.array(cl_data) for cl_data in clusters_data]

        centroids = list()
        cent_dists, sym_dists = list(), list()

        cartesian_stats = [[None for _ in range(k)] for _ in range(k)]

        for cl_idx, cl_data in enumerate(clusters_data):
            cl_centroid = _get_centroid(cl_data)
            centroids.append(cl_centroid)

            cartesian_stats[cl_idx][cl_idx] = CachedValues(*_get_intra_stat(cl_data))

            cent_dists.append(
                ArrayStat(_get_cent_dists(cl_data, cl_centroid))
            )

            sym_dists.append(
                ArrayStat(_get_sym_dists(cl_data, cl_centroid))
            )

        for x_idx in range(1, k):
            for y_idx in range(x_idx):
                x, y = clusters_data[x_idx], clusters_data[y_idx]
                inter_stat = CachedValues(*_get_inter_stat(x, y))
                cartesian_stats[x_idx][y_idx] = inter_stat
                cartesian_stats[y_idx][x_idx] = inter_stat

        return Clusters(
            all_data=np.array(filtered_data),
            global_centroid=_get_centroid(data),
            clusters_data=clusters_data,
            centroids=centroids,
            cent_dists=cent_dists,
            sym_dists=sym_dists,
            cartesian_stats=cartesian_stats,
        )


def ch(clusters: Clusters) -> float:
    global_sum = np.sum([
        cl_n * _dist(clusters.global_centroid, cl_centroid)
        for cl_n, cl_centroid in zip(clusters.amounts, clusters.centroids)
    ])
    cluster_sum = np.sum([cent_dist.sum_val for cent_dist in clusters.cent_dists])
    return (clusters.n - clusters.k) / (clusters.k - 1) * global_sum / cluster_sum


def _gen_db(clusters: Clusters, diam: Callable[[int], float]) -> float:
    clusters_sum = 0.0
    for x_idx in range(clusters.k):
        x_result = float('-inf')
        for y_idx in range(clusters.k):
            if x_idx == y_idx:
                continue
            x_cent = clusters.centroids[x_idx]
            y_cent = clusters.centroids[y_idx]
            comp = diam(x_idx) + diam(y_idx)
            sep = _dist(x_cent, y_cent)
            x_result = max(comp / sep, x_result)
        clusters_sum += x_result
    return clusters_sum / clusters.k


def _gen_db_star(clusters: Clusters, diam: Callable[[int], float]) -> float:
    clusters_sum = 0.0
    for x_idx in range(clusters.k):
        sep, comp = float('-inf'), float('inf')
        for y_idx in range(clusters.k):
            if x_idx == y_idx:
                continue
            x_cent = clusters.centroids[x_idx]
            y_cent = clusters.centroids[y_idx]
            sep = max(diam(x_idx) + diam(y_idx), sep)
            comp = min(_dist(x_cent, y_cent), comp)
        clusters_sum += sep / comp
    return clusters_sum / clusters.k


def db(clusters: Clusters) -> float:
    def _diam(cl_idx: int):
        return clusters.cent_dists[cl_idx].mean_val

    result = _gen_db(clusters, _diam)
    return -result  # to invert monotonicity


def db_star(clusters: Clusters) -> float:
    def _diam(cl_idx: int):
        return clusters.cent_dists[cl_idx].mean_val

    result = _gen_db_star(clusters, _diam)
    return -result  # to invert monotonicity


def _gen_dunn(
        clusters: Clusters,
        cluster_diam: Callable[[int], float],
        cluster_dist: Callable[[int, int], float]
) -> float:
    sep = float('inf')
    for x_idx in range(1, clusters.k):
        for y_idx in range(x_idx):
            sep = min(cluster_dist(x_idx, y_idx), sep)
    comp = max([cluster_diam(cl_idx) for cl_idx in range(clusters.k)])
    return sep / comp


def _get_dunn_diam1(clusters: Clusters) -> Callable[[int], float]:
    def _dunn_diam1(cl_idx: int) -> float:
        return clusters.cartesian_stats[cl_idx][cl_idx].max_val

    return _dunn_diam1


# def _dunn_diam2(cluster: MCluster) -> float:
#     res_diam = 0.0
#     for x_obj in cluster.data:
#         for y_obj in cluster.data:
#             res_diam += dist(x_obj, y_obj)
#     return res_diam

def _get_dunn_diam3(clusters: Clusters) -> Callable[[int], float]:
    def _dunn_diam3(cl_idx: int) -> float:
        return 2.0 * clusters.cent_dists[cl_idx].mean_val

    return _dunn_diam3


def _get_dunn_dist1(clusters: Clusters) -> Callable[[int, int], float]:
    def _dunn_dist1(x_idx: int, y_idx: int) -> float:
        return clusters.cartesian_stats[x_idx][y_idx].min_val

    return _dunn_dist1


def _get_dunn_dist3(clusters: Clusters) -> Callable[[int, int], float]:
    def _dunn_dist3(x_idx: int, y_idx: int) -> float:
        return clusters.cartesian_stats[x_idx][y_idx].mean_val

    return _dunn_dist3


def _get_dunn_dist4(clusters: Clusters) -> Callable[[int, int], float]:
    def _dunn_dist4(x_idx: int, y_idx: int) -> float:
        x_cent = clusters.centroids[x_idx]
        y_cent = clusters.centroids[y_idx]
        return _dist(x_cent, y_cent)

    return _dunn_dist4


# def _dunn_dist5(x: MCluster, y: MCluster) -> float:
#     return (np.sum(x.dist_to_centroid) + np.sum(y.dist_to_centroid)) / (x.n + y.n)


def dunn11(clusters: Clusters) -> float:
    _diam = _get_dunn_diam1(clusters)
    _dist = _get_dunn_dist1(clusters)
    return _gen_dunn(clusters, _diam, _dist)


def dunn31(clusters: Clusters) -> float:
    _diam = _get_dunn_diam1(clusters)
    _dist = _get_dunn_dist3(clusters)
    return _gen_dunn(clusters, _diam, _dist)


def dunn41(clusters: Clusters) -> float:
    _diam = _get_dunn_diam1(clusters)
    _dist = _get_dunn_dist4(clusters)
    return _gen_dunn(clusters, _diam, _dist)


# def dunn51(data: np.ndarray, clusters: List[MCluster]) -> float:
#     return _gen_dunn(clusters, _dunn_diam1, _dunn_dist5)


# def dunn12(data: np.ndarray, clusters: List[MCluster]) -> float:
#     return _gen_dunn(clusters, _dunn_diam2, _dunn_dist1)
#
#
# def dunn32(data: np.ndarray, clusters: List[MCluster]) -> float:
#     return _gen_dunn(clusters, _dunn_diam2, _dunn_dist3)
#
#
# def dunn42(data: np.ndarray, clusters: List[MCluster]) -> float:
#     return _gen_dunn(clusters, _dunn_diam2, _dunn_dist4)


# def dunn52(data: np.ndarray, clusters: List[MCluster]) -> float:
#     return _gen_dunn(clusters, _dunn_diam2, _dunn_dist5)


def dunn13(clusters: Clusters) -> float:
    _diam = _get_dunn_diam3(clusters)
    _dist = _get_dunn_dist1(clusters)
    return _gen_dunn(clusters, _diam, _dist)


def dunn33(clusters: Clusters) -> float:
    _diam = _get_dunn_diam3(clusters)
    _dist = _get_dunn_dist3(clusters)
    return _gen_dunn(clusters, _diam, _dist)


def dunn43(clusters: Clusters) -> float:
    _diam = _get_dunn_diam3(clusters)
    _dist = _get_dunn_dist4(clusters)
    return _gen_dunn(clusters, _diam, _dist)


# def dunn53(data: np.ndarray, clusters: List[MCluster]) -> float:
#     return _gen_dunn(clusters, _dunn_diam3, _dunn_dist5)


def score(clusters: Clusters) -> float:
    wcd = np.sum([cent_dist.mean_val for cent_dist in clusters.cent_dists])
    bcd = np.sum([
        cl_n * _dist(cl_centroid, clusters.global_centroid)
        for cl_n, cl_centroid in zip(clusters.amounts, clusters.centroids)
    ]) / (clusters.n * clusters.k)
    return 1.0 - 1.0 / np.exp(np.exp(bcd - wcd))


@numba.njit
def _nb_sil(all_data: np.ndarray, clusters_data: List[np.ndarray]) -> float:
    sil_res, k = 0.0, len(clusters_data)
    for x_idx, x in enumerate(clusters_data):
        x_n = len(x)
        for x_obj in x:
            a_val, b_val = 0.0, float('inf')
            for y_idx, y in enumerate(clusters_data):
                y_n = len(y)
                if x_idx == y_idx:
                    for y_obj in y:
                        a_val += _dist(x_obj, y_obj)
                    a_val /= x_n
                else:
                    b_y = 0.0
                    for y_obj in y:
                        b_y += _dist(x_obj, y_obj)
                    b_val = min(b_y / y_n, b_val)
            sil_res += (b_val - a_val) / max(b_val, a_val)
    return sil_res / len(all_data)


def sil(clusters: Clusters) -> float:
    return _nb_sil(
        all_data=clusters.all_data,
        clusters_data=clusters.clusters_data
    )


def sv(clusters: Clusters) -> float:
    s_val, v_val = 0.0, 0.0
    for x_idx in range(clusters.k):
        sorted_dists = sorted(clusters.cent_dists[x_idx].arr)
        edge_idx = int(clusters.amounts[x_idx] * 0.1) + 1
        v_val += np.mean(sorted_dists[-edge_idx:]) + 1e-6

        s_c, x_cent = float('inf'), clusters.centroids[x_idx]
        for y_idx in range(clusters.k):
            if x_idx == y_idx:
                continue
            y_cent = clusters.centroids[y_idx]
            s_c = min(_dist(y_cent, x_cent), s_c)
        s_val += s_c
    return s_val / v_val


@numba.njit
def _nb_os(clusters_data: List[np.ndarray], centroids: List[np.ndarray]) -> float:
    o_val, s_val = 0.0, 0.0
    for x_idx, x in enumerate(clusters_data):
        x_centroid = centroids[x_idx]
        for x_obj in x:
            x_a, o_xs = _dist(x_obj, x_centroid), []
            for y_idx, y in enumerate(clusters_data):
                if x_idx == y_idx:
                    continue
                y_centroid = centroids[y_idx]
                x_b_j = _dist(x_obj, y_centroid)
                threshold = (x_b_j - x_a) / (x_b_j + x_a)
                o_x_j = x_a / x_b_j
                if threshold < 0.4 and o_x_j > 0.1:
                    o_xs.append(o_x_j)
            if len(o_xs) > 0:
                o_val += sum(o_xs) / len(o_xs)

        s_c = float('inf')
        for y_idx, y in enumerate(clusters_data):
            if x_idx == y_idx:
                continue
            y_centroid = centroids[y_idx]
            s_c = min(_dist(y_centroid, x_centroid), s_c)
        s_val += s_c
    return o_val / s_val


def os(clusters: Clusters) -> float:
    result = _nb_os(
        clusters_data=clusters.clusters_data,
        centroids=clusters.centroids,
    )
    return -result  # to invert monotonicity


@numba.njit
def _nb_c_index(
        all_data: np.ndarray,
        clusters_data: List[np.ndarray],
        intra_sums: List[float],
) -> float:
    s_c, s_all = sum(intra_sums), []
    n = len(all_data)
    n_w = _get_n_w(clusters_data)
    for x_obj_idx in range(1, n):
        for y_obj_idx in range(x_obj_idx):
            x_obj, y_obj = all_data[x_obj_idx], all_data[y_obj_idx]
            s_all.append(_dist(x_obj, y_obj))
    s_all.sort()
    s_min = sum(s_all[:n_w])
    s_max = sum(s_all[-n_w:])
    return (s_c - s_min) / (s_max - s_min)


def c_index(clusters: Clusters) -> float:
    result = _nb_c_index(
        all_data=clusters.all_data,
        clusters_data=clusters.clusters_data,
        intra_sums=[cl_stat.sum_val for cl_stat in clusters.intra_stats],
    )
    return -result  # to invert monotonicity


@numba.njit
def _get_kl_arr(clusters_data: List[np.ndarray]):
    k, kl_arr = len(clusters_data), list()
    for x_idx in range(1, k):
        for x_obj in clusters_data[x_idx]:
            for y_idx in range(x_idx):
                for y_obj in clusters_data[y_idx]:
                    kl_arr.append(_dist(x_obj, y_obj))
    return sorted(kl_arr)


@numba.njit
def _bin_search(sorted_arr, value) -> int:
    l, r = 0, len(sorted_arr) - 1
    while l <= r:
        m = (l + r) // 2
        m_val = sorted_arr[m]
        if m_val < value:
            l = m + 1
        elif m_val > value:
            r = m - 1
        else:
            return m
    return l


@numba.njit
def _nb_gamma_index(
        clusters_data: List[np.ndarray],
) -> float:
    d_l, kl_arr = 0, _get_kl_arr(clusters_data)
    for cl in clusters_data:
        cl_n = len(cl)
        for x_obj_idx in range(1, cl_n):
            for y_obj_idx in range(x_obj_idx):
                x, y = cl[x_obj_idx], cl[y_obj_idx]
                d_l += _bin_search(kl_arr, _dist(x, y))
    return d_l / (_get_n_w(clusters_data) * _get_n_b(clusters_data))


def gamma_index(clusters: Clusters) -> float:
    result = _nb_gamma_index(
        clusters_data=clusters.clusters_data,
    )
    return -result  # to invert monotonicity


@numba.njit
def _nb_cop(
        all_data: np.ndarray,
        clusters_data: List[np.ndarray],
        mean_cent_dists: List[float],
) -> float:
    cop_res, k = 0.0, len(clusters_data)
    for x_idx in range(k):
        x, x_sep = clusters_data[x_idx], float('inf')
        x_comp = mean_cent_dists[x_idx]
        for y_idx in range(k):
            if x_idx == y_idx:
                continue
            for y_obj in clusters_data[y_idx]:
                y_sep = float('-inf')
                for x_obj in x:
                    y_sep = max(_dist(y_obj, x_obj), y_sep)
                x_sep = min(x_sep, y_sep)
        cop_res += len(x) * x_comp / x_sep
    return cop_res / len(all_data)


def cop(clusters: Clusters) -> float:
    result = _nb_cop(
        all_data=clusters.all_data,
        clusters_data=clusters.clusters_data,
        mean_cent_dists=[c_dist.mean_val for c_dist in clusters.cent_dists]
    )
    return -result  # to invert monotonicity


@numba.njit
def _nb_cs(clusters_data: List[np.ndarray], centroids: List[np.ndarray], intra_maxs: List[float], ) -> float:
    comp, k = 0.0, len(clusters_data)
    for cl_idx, cl_data in enumerate(clusters_data):
        comp += intra_maxs[cl_idx] / len(cl_data)

    d_centroids = np.full(shape=(k, k), fill_value=float('inf'))
    for x_idx in range(1, k):
        for y_idx in range(x_idx):
            d_c = _dist(centroids[x_idx], centroids[y_idx])
            d_centroids[x_idx, y_idx] = d_c
            d_centroids[y_idx, x_idx] = d_c
    sep = 0.0
    for c_idx in range(k):
        sep += min(d_centroids[c_idx])
    return comp / sep


def cs(clusters: Clusters) -> float:
    result = _nb_cs(
        clusters_data=clusters.clusters_data,
        centroids=clusters.centroids,
        intra_maxs=[cl_stat.max_val for cl_stat in clusters.intra_stats],
    )
    return -result  # to invert monotonicity


def sym(clusters: Clusters) -> float:
    d_k = float('-inf')
    for x_idx in range(1, clusters.k):
        for y_idx in range(x_idx):
            x_cent = clusters.centroids[x_idx]
            y_cent = clusters.centroids[y_idx]
            d_k = max(_dist(x_cent, y_cent), d_k)

    e_k = np.mean([
        cl_sym.sum_val for cl_sym in clusters.sym_dists
    ])
    return d_k / e_k


def sym_db(clusters: Clusters) -> float:
    def _diam(cl_idx: int) -> float:
        return clusters.sym_dists[cl_idx].mean_val

    result = _gen_db(clusters, _diam)
    return -result  # to invert monotonicity


def sym_db_star(clusters: Clusters) -> float:
    def _diam(cl_idx: int) -> float:
        return clusters.sym_dists[cl_idx].mean_val

    result = _gen_db_star(clusters, _diam)
    return -result  # to invert monotonicity


def sym_11(clusters: Clusters) -> float:
    def _diam(cl_idx: int) -> float:
        return clusters.sym_dists[cl_idx].max_val

    _dist = _get_dunn_dist1(clusters)

    return _gen_dunn(clusters, _diam, _dist)


def sym_33(clusters: Clusters) -> float:
    def _diam(cl_idx: int) -> float:
        return 2.0 * clusters.sym_dists[cl_idx].mean_val

    _dist = _get_dunn_dist3(clusters)
    return _gen_dunn(clusters, _diam, _dist)


def banfield_raftery(clusters: Clusters) -> float:
    br_res = 0.0
    for cl_idx in range(clusters.k):
        cl_d = clusters.cent_dists[cl_idx].arr
        cl_var = np.mean(cl_d * cl_d)
        var_stable = 1.0 + cl_var + 1e-6  # Added 1 and 1e-6 to stabilize, originally just used np.log(cl_var)
        br_res += clusters.amounts[cl_idx] * np.log(var_stable)
    result = br_res / clusters.n  # originally index is not normalised
    return -result  # to invert monotonicity


@numba.njit
def _nb_mc_clain_rao(clusters_data: List[np.ndarray], inter_sums: np.ndarray) -> float:
    s_w, s_b = 0.0, 0.0
    k = len(clusters_data)
    for x_idx in range(k):
        for y_idx in range(x_idx + 1):
            inter_sum_val = inter_sums[x_idx, y_idx]
            if x_idx == y_idx:
                s_w += inter_sum_val
            else:
                s_b += inter_sum_val
    n_w, n_b = _get_n_w(clusters_data), _get_n_b(clusters_data)
    return (s_w / n_w) / (s_b / n_b)


def mc_clain_rao(clusters: Clusters) -> float:
    result = _nb_mc_clain_rao(
        clusters_data=clusters.clusters_data,
        inter_sums=np.array([
            [pair_stat.sum_val for pair_stat in cl_row]
            for cl_row in clusters.cartesian_stats
        ])
    )
    return -result  # to invert monotonicity


def pakhira_bandyopadhyay_maulik(clusters: Clusters) -> float:
    d_b = float('-inf')
    for x_idx in range(1, clusters.k):
        for y_idx in range(x_idx):
            x_cent = clusters.centroids[x_idx]
            y_cent = clusters.centroids[y_idx]
            d_b = max(_dist(x_cent, y_cent), d_b)
    e_w, e_t = 0.0, 0.0
    for cent_dist in clusters.cent_dists:
        e_w += cent_dist.sum_val

    for obj in clusters.all_data:
        e_t += _dist(obj, clusters.global_centroid)

    val = (e_t * d_b) / (clusters.k * e_w)
    return val * val


def ray_turi(clusters: Clusters) -> float:
    wgss = _get_wgss([c_dist.arr for c_dist in clusters.cent_dists])
    delta = float('inf')
    for x_idx in range(1, clusters.k):
        for y_idx in range(x_idx):
            x_cent = clusters.centroids[x_idx]
            y_cent = clusters.centroids[y_idx]
            delta = min(_dist(x_cent, y_cent), delta)
    result = wgss / (clusters.n * delta * delta)
    return -result  # to invert monotonicity


@numba.njit
def _nb_wemmert_gancarski(
        all_data: np.ndarray,
        clusters_data: List[np.ndarray],
        centroids: List[np.ndarray],
) -> float:
    wg_res = 0.0
    for x_idx, x in enumerate(clusters_data):
        r_cl, x_cent = 0.0, centroids[x_idx]
        for x_obj_idx, x_obj in enumerate(x):
            r_x, r_y = _dist(x_obj, x_cent), float('inf')
            for y_idx, y in enumerate(clusters_data):
                if x_idx == y_idx:
                    continue
                y_centroid = centroids[y_idx]
                r_y = min(_dist(x_obj, y_centroid), r_y)
            r_cl += r_x / (r_y + 1e-6)
        wg_res += max(0, len(x) - r_cl)
    return wg_res / len(all_data)


def wemmert_gancarski(clusters: Clusters) -> float:
    return _nb_wemmert_gancarski(
        all_data=clusters.all_data,
        clusters_data=clusters.clusters_data,
        centroids=clusters.centroids,
    )


@numba.njit
def _nb_point_biserial(
        all_data: np.ndarray,
        clusters_data: List[np.ndarray],
        inter_sums: np.ndarray
) -> float:
    s_w, s_b = 0.0, 0.0
    k = len(clusters_data)
    for x_idx in range(k):
        for y_idx in range(x_idx + 1):
            inter_sum_val = inter_sums[x_idx, y_idx]
            if x_idx == y_idx:
                s_w += inter_sum_val
            else:
                s_b += inter_sum_val
    n_w, n_b = _get_n_w(clusters_data), _get_n_b(clusters_data)
    normaliser = np.sqrt(n_w * n_b) / len(all_data)
    return (s_w / n_w - s_b / n_b) * normaliser


def point_biserial(clusters: Clusters) -> float:
    return _nb_point_biserial(
        all_data=clusters.all_data,
        clusters_data=clusters.clusters_data,
        inter_sums=np.array([
            [pair_stat.sum_val for pair_stat in cl_row]
            for cl_row in clusters.cartesian_stats
        ])
    )


@numba.njit
def _nb_xie_beni(all_data: np.ndarray, cent_dists: List[np.ndarray], inter_mins: np.ndarray) -> float:
    wgss = _get_wgss(cent_dists)
    k, delta = len(cent_dists), float('inf')

    for x_idx in range(1, k):
        for y_idx in range(x_idx):
            inter_min_val = inter_mins[x_idx, y_idx]
            delta = min(inter_min_val, delta)
    return wgss / (len(all_data) * delta * delta)


def xie_beni(clusters: Clusters) -> float:
    result = _nb_xie_beni(
        all_data=clusters.all_data,
        cent_dists=[c_dist.arr for c_dist in clusters.cent_dists],
        inter_mins=np.array([
            [pair_stat.min_val for pair_stat in cl_row]
            for cl_row in clusters.cartesian_stats
        ])
    )
    return -result  # to invert monotonicity

# def s_dbw(data: np.ndarray, clusters: List[MCluster]) -> float:
#     k, dim = len(clusters), data.shape[1]
#
#     def get_sigma_norm(arr, centroid):
#         sigma = np.zeros(shape=dim)
#         for obj in arr:
#             diff = obj - centroid
#             sigma += diff * diff
#         return np.linalg.norm(sigma / len(data))
#
#     global_centroid = get_centroid(data)
#     global_sigma = get_sigma_norm(data, global_centroid)
#
#     sigma_x = [get_sigma_norm(x.data, x.centroid) for x in clusters]
#     stddev = np.sqrt(sum(sigma_x)) / k
#
#     scat = np.mean([s_x / global_sigma for s_x in sigma_x])
#
#     def one_or_zero(d_val):
#         return 1 if d_val < stddev else 0
#
#     def f_dist(x, y):
#         d = dist(x, y)
#         return one_or_zero(d)
#
#     def den_cl(cl: MCluster):
#         den_cl_val = 0
#         for d_val in cl.dist_to_centroid:
#             den_cl_val += one_or_zero(d_val)
#         return den_cl_val
#
#     def den_xy(x: MCluster, y: MCluster):
#         den_xy_val = 0
#         mid_cent = (x.centroid + y.centroid) / 2
#         for x_obj in x.data:
#             den_xy_val += f_dist(x_obj, mid_cent)
#         for y_obj in y.data:
#             den_xy_val += f_dist(y_obj, mid_cent)
#         return den_xy_val
#
#     density = 0.0
#     den_clusters = [den_cl(x) for x in clusters]
#     for x_idx in range(1, k):
#         for y_idx in range(x_idx):
#             x, y = clusters[x_idx], clusters[y_idx]
#             den_x, den_y = den_clusters[x_idx], den_clusters[y_idx]
#             divider = max(den_x, den_y)
#             if divider > 1e-8:
#                 density += den_xy(x, y) / divider
#     density /= k * (k - 1) / 2
#
#     result = scat + density
#     return -result  # to invert monotonicity

INTERNAL_MEASURES = [
    ('CH', ch),
    ('DB', db),
    ('DB_STAR', db_star),
    ('DUNN_11', dunn11),
    ('DUNN_31', dunn31),
    ('DUNN_41', dunn41),
    ('DUNN_13', dunn13),
    ('DUNN_33', dunn33),
    ('DUNN_43', dunn43),
    ('SCORE', score),
    ('SIL', sil),
    ('SV', sv),
    ('OS', os),
    ('C_INDEX', c_index),
    ('GAMMA_INDEX', gamma_index),
    ('COP', cop),
    ('CS', cs),
    ('SYM', sym),
    ('SYM_DB', sym_db),
    ('SYM_DB_STAR', sym_db_star),
    ('SYM_11', sym_11),
    ('SYM_33', sym_33),
    ('BR', banfield_raftery),
    ('MCR', mc_clain_rao),
    ('PBM', pakhira_bandyopadhyay_maulik),
    ('RT', ray_turi),
    ('WG', wemmert_gancarski),
    ('PB', point_biserial),
    ('XB', xie_beni),
    # ('S_DBW', s_dbw)
]
import cupy as cp
import numpy as np
from cupy.typing import NDArray

from gamac.estimation.container import EstimationContainer
from gamac.kernels import MIDDLEWARE, BATCH_SIZE


def mcr(container: EstimationContainer) -> float:
    gpu_s_w = cp.empty(shape=container.n, dtype=np.float32)
    gpu_s_b = cp.empty(shape=container.n, dtype=np.float32)

    MIDDLEWARE.mcr(
        data=container.data,
        N=container.n,
        D=container.d,
        labels=container.labels,
        s_w=gpu_s_w,
        s_b=gpu_s_b,
    ).invoke(
        grid=(container.n // BATCH_SIZE + 1,),
        blocks=(BATCH_SIZE,),
    )

    s_w = gpu_s_w.sum().item()
    s_b = gpu_s_b.sum().item()

    result = (s_w / container.n_w) / (s_b / container.n_b)
    return -result


def br(container: EstimationContainer) -> float:
    result_acc = 0.0
    for cl_idx in range(container.k):
        cl_n = len(container.clusters[cl_idx])
        cl_d = container.cent_dists[cl_idx]
        cl_var = (cl_d * cl_d).mean().item()
        var_stable = 1.0 + cl_var + 1e-6
        result_acc += cl_n * np.log(var_stable)
    if container.n:
        result = result_acc / container.n
        print(result_acc, container.n)
        return -result
    return -100000


def sym(container: EstimationContainer) -> float:
    d_k = np.max(container.cent_matrix)
    e_k = np.mean([
        cl_sym.sum().item() for cl_sym in container.sym_dists
    ])
    return d_k / e_k


# def c_index(container: EstimationContainer) -> float:
#     print("start")
#     pairs, n_w = container.n * (container.n - 1) // 2, container.n_w
#     s_min_idx, s_max_idx = n_w, pairs - n_w
#
#     alloc_size = container.n * BATCH_SIZE
#     gpu_s_c = cp.empty(shape=(1,), dtype=np.float32)
#     gpu_s_min = cp.empty(shape=alloc_size, dtype=np.float32)
#     gpu_s_max = cp.empty(shape=alloc_size, dtype=np.float32)
#
#     q, r = divmod(pairs, alloc_size)
#     iterations = q + (0 if r == 0 else 1)
#     s_min_acc, s_max_acc = 0.0, 0.0
#
#     for iter_idx in range(iterations):
#         print(f"{iter_idx} / {iterations}")
#         gpu_s_min.fill(0.0)
#         gpu_s_max.fill(0.0)
#
#         batch_start = iter_idx * alloc_size
#         rest_pairs = pairs - batch_start
#
#         if alloc_size <= rest_pairs:
#             blocks = container.n
#         else:
#             blocks = rest_pairs // BATCH_SIZE + 1
#
#         MIDDLEWARE.c_index(
#             data=container.data,
#             N=container.n,
#             D=container.d,
#             pairs=pairs,
#             batch_start=batch_start,
#             labels=container.labels,
#             s_min_idx=s_min_idx,
#             s_min=gpu_s_min,
#             s_max_idx=s_max_idx,
#             s_max=gpu_s_max,
#             s_c=gpu_s_c,
#         ).invoke(
#             grid=(blocks,),
#             blocks=(BATCH_SIZE,),
#         )
#
#         s_min_acc += gpu_s_min.sum()
#         s_max_acc += gpu_s_max.sum()
#
#     s_c = gpu_s_c[0]
#     result = (s_c - s_min_acc) / (s_max_acc - s_min_acc)
#     return -result


def c_index(container: EstimationContainer) -> float:
    pairs, n_w = container.n * (container.n - 1) // 2, container.n_w
    s_min_idx, s_max_idx = n_w, pairs - n_w

    gpu_s_c = cp.empty(shape=(1,), dtype=np.float32)
    gpu_s_min = cp.empty(shape=n_w, dtype=np.float32)
    gpu_s_max = cp.empty(shape=n_w, dtype=np.float32)

    MIDDLEWARE.c_index(
        data=container.data,
        N=container.n,
        D=container.d,
        pairs=pairs,
        labels=container.labels,
        s_min_idx=s_min_idx,
        s_min=gpu_s_min,
        s_max_idx=s_max_idx,
        s_max=gpu_s_max,
        s_c=gpu_s_c,
    ).invoke(
        grid=(container.n // 16 + 1, container.n // 16 + 1),
        blocks=(16,16),
    )

    s_c = gpu_s_c.item()
    s_min = gpu_s_min.sum().item()
    s_max = gpu_s_max.sum().item()
    result = (s_c - s_min) / (s_max - s_min)
    return -result


def f1(
        classes: NDArray,
        classes_k: int,
        labels: NDArray,
        labels_k: int,
) -> float:
    N = len(labels)
    gpu_crosstab = cp.empty(shape=(classes_k, labels_k), dtype=np.uint32)
    MIDDLEWARE.crosstab(
        N=N,
        classes=classes,
        classes_k=classes_k,
        labels=labels,
        labels_k=labels_k,
        crosstab_matrix=gpu_crosstab
    ).invoke(
        grid=(classes_k // 16 + 1, labels_k // 16 + 1),
        blocks=(16, 16),
    )
    crosstab_matrix = cp.asnumpy(gpu_crosstab)

    a_arr = crosstab_matrix.sum(axis=1)
    b_arr = crosstab_matrix.sum(axis=0)

    f1_val = 0.0
    for j, nj in enumerate(b_arr):
        a_max_val = 0.0
        for i, ni in enumerate(a_arr):
            nij = crosstab_matrix[i, j]
            precision, recall = nij / ni, nij / nj
            div = precision + recall
            if div > 1e-6:
                ij_val = 2 * precision * recall / div
                a_max_val = max(a_max_val, ij_val)
        f1_val += nj / N * a_max_val
    return f1_val

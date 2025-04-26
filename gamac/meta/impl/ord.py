"""
Functions to compute distance between two orderings
"""

from typing import List

import numpy as np


def norm_kendall_tau(
        x_order: List[int],
        y_order: List[int]
) -> float:
    n = len(x_order)
    assert len(y_order) == n

    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a, b = np.argsort(x_order), np.argsort(y_order)

    disordered = np.logical_or(
        np.logical_and(a[i] < a[j], b[i] > b[j]),
        np.logical_and(a[i] > a[j], b[i] < b[j])
    ).sum()

    return disordered / (n * (n - 1))


def norm_f_dist(
        x_order: List[int],
        y_order: List[int]
) -> float:
    n = len(x_order)
    assert len(y_order) == n

    diff = np.array(x_order) - np.array(y_order)
    f_value = np.sum(np.abs(diff))
    f_max = n * n / 2 if n % 2 == 0 else (n - 1) * (n + 1) / 2
    return f_value / f_max


def lin_f_dist(
        pretend_ordering: List[int],
        ground_ordering: List[int]
) -> float:
    n = len(ground_ordering)
    assert len(pretend_ordering) == n

    result, mid = 0.0, n // 2
    if n % 2 == 0:
        weights = [abs(idx - mid) for idx in range(n + 1)]
        del weights[mid]
        flin_max = 2 * mid * (mid + 1) * (2 * mid + 1) / 3 - mid * (mid + 1)
    else:
        weights = [abs(idx - mid) for idx in range(n)]
        flin_max = 2 * mid * (mid + 1) * (2 * mid + 1) / 3
    for idx in np.argsort(ground_ordering):
        x, y = pretend_ordering[idx], ground_ordering[idx]
        result += weights[idx] * abs(x - y)
    return result / flin_max

def sq_f_dist(
        pretend_ordering: List[int],
        ground_ordering: List[int]
) -> float:
    n = len(ground_ordering)
    assert len(pretend_ordering) == n


    result, mid = 0.0, n // 2
    if n % 2 == 0:
        weights = [abs(idx - mid) ** 2 for idx in range(n + 1)]
        del weights[mid]
        fsq_max = (mid * (mid + 1)) ** 2 - (mid * (mid + 1) * (2 * mid + 1)) / 3
    else:
        weights = [abs(idx - mid) ** 2 for idx in range(n)]
        fsq_max = (mid * (mid + 1)) ** 2
    for idx in np.argsort(ground_ordering):
        x, y = pretend_ordering[idx], ground_ordering[idx]
        result += weights[idx] * abs(x - y)
    return result / fsq_max


ORDERING_DISTANCE = sq_f_dist

from numba import cuda, types
import numpy as np

from ..utils.funcs import delta, sigma

@cuda.jit(())
def dunn(k_list):
    s = np.ones([len(k_list), len(k_list)])
    d = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            s[k, l] = sigma(k_list[k], k_list[l])
            d[k] = delta(k_list[k])
            di = np.min(s) / np.max(d)
    return di

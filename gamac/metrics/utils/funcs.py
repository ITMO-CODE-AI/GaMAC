from numba import cuda
import numpy as np


@cuda.jit(device=True)
def sigma(ck, cl):
    values = np.ones([len(ck), len(cl)])
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i] - cl[j])
    return np.min(values)


@cuda.jit(device=True)
def sigma3(ck, cl):
    values = np.ones([len(ck), len(cl)])
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i] - cl[j])
    return values / (len(ck) * len(cl))


@cuda.jit(device=True)
def delta(ci):
    values = np.zeros([len(ci), len(ci)])
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])
    return np.max(values)


@cuda.jit(device=True)
def delta3(ci):
    values = np.zeros([len(ci), len(ci)])
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])
    return 2 * values / (len(ci))


@cuda.jit(device=True)
def euclidean_distances_numba(X, Y=None, Y_norm_squared=None):
    XX_ = (X * X).sum(axis=1)
    XX = XX_.reshape((1, -1))

    if X is Y:
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = Y_norm_squared
    else:
        YY_ = np.sum(Y * Y, axis=1)
        YY = YY_.reshape((1, -1))

    distances = np.dot(X, Y.T)
    distances *= -2
    distances += XX
    distances += YY
    distances = np.maximum(distances, 0)

    return np.sqrt(distances)


@cuda.jit(device=True)
def euclidean_distances_sum(X, Y=None):
    if Y is None:
        Y = X
    Y_norm_squared = (Y ** 2).sum(axis=1)
    sums = np.zeros((len(X)))
    for i in range(len(X)):
        base_row = X[i, :]
        sums[i] = euclidean_distances_numba(base_row.reshape(1, -1), Y, Y_norm_squared=Y_norm_squared).sum()

    return sums


@cuda.jit(device=True)
def euclidean_distances_mean(X, Y=None):
    if Y is None:
        Y = X
    Y_norm_squared = (Y ** 2).sum(axis=1)
    means = np.zeros((len(X)))
    for i in range(len(X)):
        base_row = X[i, :]
        means[i] = euclidean_distances_numba(base_row.reshape(1, -1), Y, Y_norm_squared=Y_norm_squared).mean()

    return means


@cuda.jit(device=True)
def filter_list(a, b):
    c = []
    for x in a:
        c.append(x == b)
    return c

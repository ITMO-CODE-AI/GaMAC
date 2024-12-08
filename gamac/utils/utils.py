import math
from numba import cuda
import numpy as np


def normalize(X, axis=-1, order=2):
    """Normalize the dataset X"""
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def cpu_distance(x1, x2):
    """Calculates the l2 distance between two vectors"""
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)
@cuda.jit(device=True)
def gpu_distance(x1, x2):
    """Calculates the l2 distance between two vectors"""
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)
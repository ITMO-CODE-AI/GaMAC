import math
# from cupyx.scipy.spatial.distance import euclidean
import cupy as cp
from cuvs.distance import pairwise_distance
import numpy as np
import pylibraft.config

pylibraft.config.set_output_as("cupy")

def normalize(X, axis=-1, order=2):
    """Normalize the dataset X"""
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def cpu_distance(x1, x2):
    """Calculates the l2 distance between two vectors on CPU"""
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)

def gpu_cupy_distance(x1, x2):
    """Calculates the l2 distance between two vectors on GPU"""
    x1, x2 = cp.asarray(x1), cp.asarray(x2)
    return pairwise_distance(x1, x2, metric="euclidean")

def gpu_distance(x1, x2):
    """Calculates the l2 distance between two vectors on CPU"""
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)
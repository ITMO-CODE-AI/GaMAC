"""
Named containers for clustering algorithms and feature extractors
"""

import numpy as np


class Reducer:
    def __init__(self, name, algo):
        self.name, self.algo = name, algo

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        result = self.algo.fit_transform(x)
        if result.size != np.isfinite(result).sum():
            raise ValueError("Got incorrect transformed values")
        return result


class Producer:
    def __init__(self, algo, name):
        self.algo, self.name = algo, name

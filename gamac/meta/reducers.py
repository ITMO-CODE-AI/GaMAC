from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding, SpectralEmbedding
from umap import UMAP


class Reducer:
    def __init__(self, name, algo):
        self.name, self.algo = name, algo

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        result = self.algo.fit_transform(x)
        if result.size != np.isfinite(result).sum():
            raise ValueError("Got incorrect transformed values")
        return result


class ReducerProvider:
    @staticmethod
    def get_all() -> List[Reducer]:
        reducers = {
            **ReducerProvider.pca(),
            **ReducerProvider.mlle(),
            **ReducerProvider.spec(),
            **ReducerProvider.tsne(),
            **ReducerProvider.umap(),
            **ReducerProvider.mds(),
        }
        return [Reducer(name, algo) for name, algo in reducers.items()]

    @staticmethod
    def pca():
        return {
            'pca': PCA(n_components=2, random_state=42)
        }

    @staticmethod
    def mlle():
        return {
            'mlle1': LocallyLinearEmbedding(n_neighbors=20, method='modified', random_state=42),
            'mlle2': LocallyLinearEmbedding(n_neighbors=50, method='modified', random_state=42),
        }

    @staticmethod
    def spec():
        return {
            'spec1': SpectralEmbedding(affinity='rbf', gamma=1e-3, random_state=42),
            'spec2': SpectralEmbedding(affinity='nearest_neighbors', n_neighbors=20, random_state=42),
            'spec3': SpectralEmbedding(affinity='nearest_neighbors', n_neighbors=50, random_state=42),
        }

    @staticmethod
    def tsne():
        return {
            'tsne1': TSNE(perplexity=10, random_state=42),
            'tsne2': TSNE(perplexity=30, random_state=42),
            'tsne3': TSNE(perplexity=50, random_state=42),
        }

    @staticmethod
    def mds():
        return {
            'mds': MDS(metric=True, max_iter=100, n_init=2, random_state=42)
        }

    @staticmethod
    def umap():
        return {
            'umap1': UMAP(n_neighbors=15, min_dist=0.2, init='pca', random_state=42),
            'umap2': UMAP(n_neighbors=50, min_dist=0.5, init='pca', random_state=42),
            'umap3': UMAP(n_neighbors=80, min_dist=0.7, init='pca', random_state=42)
        }

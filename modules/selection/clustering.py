"""
Clusterers: group items by their embedding vectors.

A Clusterer only maps vectors to cluster labels; what is embedded and how the
clusters are used is up to the strategy. Swap in another algorithm (HDBSCAN,
agglomerative, ...) by implementing fit().
"""
from abc import ABC, abstractmethod

import numpy as np


class Clusterer(ABC):
    @abstractmethod
    def fit(self, vectors: np.ndarray, k: int) -> list[int]:
        """
        vectors : (n_items, dim) array, L2-normalised rows
        k       : requested number of clusters (implementations may return fewer)
        Returns one integer cluster label per row.
        """


class KMeansClusterer(Clusterer):
    """
    k-means on normalised embeddings (Euclidean distance on unit vectors is a
    monotone function of cosine distance). Deterministic for a fixed seed.
    """

    def __init__(self, seed: int = 0):
        self._seed = seed

    def fit(self, vectors: np.ndarray, k: int) -> list[int]:
        n = len(vectors)
        if n == 0:
            return []
        # Identical texts give identical vectors; k cannot exceed the distinct points.
        k = max(1, min(k, n, len(np.unique(vectors.round(6), axis=0))))
        if k == 1:
            return [0] * n
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, n_init=10, random_state=self._seed).fit_predict(vectors).tolist()

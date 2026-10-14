"""
Sub-interfaces for RetrievalEvaluator implementations.

A concrete evaluator wires together:
    encoder    — FeatureEncoder: turns a pool item's features into vectors
    calculator — MetricsCalculator: diversity over the free (non-searched) features
    surprise   — SurpriseCalculator: how many items fall outside the searched concepts
"""
from abc import ABC, abstractmethod


class FeatureEncoder(ABC):
    @abstractmethod
    def encode_item(self, item: dict):
        """
        Encode one feature dict {feature_name: value | [values]} into
        {feature_name: vector | float}.
        """

    def encode_pool(self, items: list) -> list:
        """
        Encode all pool items. Override for batch efficiency (default: loop over encode_item).
        """
        return [self.encode_item(item) for item in items]


class MetricsCalculator(ABC):
    @abstractmethod
    def compute(self, item_features: list, pinned: set) -> dict:
        """
        item_features : encoded feature dicts, one per item
        pinned        : names of the features the query searched on (excluded)
        Returns {"diversity": float} — average pairwise spread across free features.
        """


class SurpriseCalculator(ABC):
    @abstractmethod
    def compute(self, pool: list, query: dict) -> float:
        """
        pool  : pool items whose "properties" is an Artifact dict
        query : {feature_name: [concept URIs]} — the searched concepts per feature
        Returns surprise in [0, 1]; 0 when every item stays within the searched concepts.
        """

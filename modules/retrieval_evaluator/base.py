"""
Sub-interfaces for RetrievalEvaluator implementations.

A concrete evaluator wires together a FeatureEncoder and a MetricsCalculator:
    encoder    — turns a query string or pool item into a feature vector
    calculator — takes the encoded vectors and returns pool-level metric scores
"""
from abc import ABC, abstractmethod


class FeatureEncoder(ABC):
    @abstractmethod
    def encode_query(self, query: dict):
        """
        Encode pinned query features into a FeatureMap.
        query: {feature_name: raw_value} for each pinned feature.
        """

    @abstractmethod
    def encode_item(self, item: dict):
        """
        Encode a pool item into a FeatureMap.
        item: { id, title, score, properties: {...} }
        """

    def encode_pool(self, items: list) -> list:
        """
        Encode all pool items. Override for batch efficiency (default: loop over encode_item).
        """
        return [self.encode_item(item) for item in items]


class MetricsCalculator(ABC):
    @abstractmethod
    def compute(self, query_features, item_features: list) -> dict:
        """
        Compute pool-level metrics from encoded feature maps.

        query_features  : output of FeatureEncoder.encode_query() — pinned features only
        item_features   : list of FeatureEncoder.encode_item/pool() outputs, one per item

        Returns {"diversity": float, "surprise": float}
          diversity — average pairwise spread across free (non-pinned) features
          surprise  — normalised entropy over pinned-feature deviation counts
        """

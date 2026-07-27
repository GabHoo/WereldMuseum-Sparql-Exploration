from modules.base import RetrievalEvaluator
from modules.retrieval_evaluator.base import FeatureEncoder, MetricsCalculator


class MyFeatureEncoder(FeatureEncoder):
    """
    Stub encoder — replace with your embedding / feature extraction logic.
    Input:  {feature_name: raw_value}  (plain dict, any keys)
    Output: {feature_name: np.ndarray | float}  (same keys, encoded)
    Text values → embedding vector; numeric values → float passthrough.
    """

    def encode_query(self, query: dict) -> dict:
        raise NotImplementedError

    def encode_item(self, item: dict) -> dict:
        raise NotImplementedError


class MyMetricsCalculator(MetricsCalculator):
    """
    Stub calculator — implement diversity and surprise from encoded dicts.
    Features are matched by key name; missing keys in an item are skipped.
    """

    def compute(self, query_features: dict, item_features: list[dict]) -> dict:  # noqa: ARG002
        raise NotImplementedError


class MyRetrievalEvaluator(RetrievalEvaluator):
    """
    Stub — copy this file, wire your encoder + calculator, implement score(), then:
      1. Add an entry to REGISTRY["retrieval_evaluator"] in config.py
      2. Set RETRIEVAL_EVALUATOR=<your-key> in .env
    """

    def __init__(self):
        self.encoder    = MyFeatureEncoder()
        self.calculator = MyMetricsCalculator()

    def score(self, pool: list[dict], query: dict) -> dict:
        """
        pool  : list of pool items {id, title, score, properties: {...}}
        query : pinned feature dict {feature_name: raw_value}; {} → all features free
        Returns {"diversity": float, "surprise": float}
        """
        if not pool:
            return {"diversity": 0.0, "surprise": 0.0}
        # extract feature dicts from items however makes sense for your implementation
        feature_dicts = [item.get("properties", {}) for item in pool]
        query_encoded = self.encoder.encode_query(query)
        items_encoded = self.encoder.encode_pool(feature_dicts)
        return self.calculator.compute(query_encoded, items_encoded)

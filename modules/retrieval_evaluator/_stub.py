from modules.base import RetrievalEvaluator
from modules.retrieval_evaluator.base import FeatureEncoder, MetricsCalculator, SurpriseCalculator


class MyFeatureEncoder(FeatureEncoder):
    """
    Stub encoder — replace with your embedding / feature extraction logic.
    Input:  {feature_name: value | [values]}  (plain dict, any keys)
    Output: {feature_name: np.ndarray | float}  (same keys, encoded)
    """

    def encode_item(self, item: dict) -> dict:
        raise NotImplementedError


class MyMetricsCalculator(MetricsCalculator):
    """
    Stub calculator — implement diversity over the free (non-searched) features.
    Features are matched by key name; missing keys in an item are skipped.
    """

    def compute(self, item_features: list[dict], pinned: set) -> dict:  # noqa: ARG002
        raise NotImplementedError


class MySurpriseCalculator(SurpriseCalculator):
    """
    Stub — score how far the pool strays from the searched concepts.
    query is {feature_name: [concept URIs]}; items carry <feature>_uris.
    """

    def compute(self, pool: list, query: dict) -> float:  # noqa: ARG002
        raise NotImplementedError


class MyRetrievalEvaluator(RetrievalEvaluator):
    """
    Stub — copy this file, wire your parts, implement score(), then:
      1. Add an entry to REGISTRY["retrieval_evaluator"] in config.py
      2. Add a branch in _build_retrieval_evaluator() in config.py
      3. Set RETRIEVAL_EVALUATOR=<your-key> in .env
    """

    def __init__(self):
        self.encoder    = MyFeatureEncoder()
        self.calculator = MyMetricsCalculator()
        self.surprise   = MySurpriseCalculator()

    def score(self, pool: list[dict], query: dict) -> dict:
        """
        pool  : list of pool items {id, title, score, properties: {...}}
        query : {feature_name: [concept URIs]}; {} → all features free
        Returns {"diversity": float, "surprise": float}
        """
        if not pool:
            return {"diversity": 0.0, "surprise": 0.0}
        feature_dicts = [item.get("properties", {}) for item in pool]
        result = self.calculator.compute(self.encoder.encode_pool(feature_dicts), set(query))
        result["surprise"] = self.surprise.compute(pool, query)
        return result

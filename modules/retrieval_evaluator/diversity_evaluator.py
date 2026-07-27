"""
Diversity-aware retrieval evaluator.

Each pool item's properties dict is used directly as its feature dict.
Non-semantic fields (URIs, internal identifiers) are stripped before encoding.
Features are matched across items by key name — no schema declaration needed.
Adding a new SPARQL field: it appears automatically in diversity computation.

Usage (CLI):
    python -m modules.retrieval_evaluator.diversity_evaluator path/to/export.json
    python -m modules.retrieval_evaluator.diversity_evaluator path/to/export.json --pinned typeLabel=masker

Environment:
    EMBEDDING_MODEL  — override the sentence-transformer model name
"""
import json
import os
import sys

from modules.base import RetrievalEvaluator
from modules.retrieval_evaluator.base import FeatureEncoder, MetricsCalculator

# Fields from SPARQL results that carry no semantic content for embedding.
# URIs and internal identifiers are excluded; human-readable labels are kept.
_NOISE_KEYS = frozenset({"url_photo", "artifact", "specificType"})


def _extract_features(item: dict) -> dict:
    """
    Build a flat feature dict from a pool item.
    Pulls all properties except noise keys, then adds title from the item root.
    """
    features = {
        k: v
        for k, v in item.get("properties", {}).items()
        if k not in _NOISE_KEYS and isinstance(v, str) and v
    }
    title = item.get("title", "").strip()
    if title and title != "Untitled":
        features["title"] = title
    return features


class DiversityEvaluator(RetrievalEvaluator):
    """
    Evaluates a retrieved pool by diversity and surprise.

    encoder    : encodes {feature_name: raw_value} dicts into {feature_name: vector|float}
    calculator : computes pool-level metrics from the encoded dicts
    """

    def __init__(self, encoder: FeatureEncoder, calculator: MetricsCalculator):
        self.encoder    = encoder
        self.calculator = calculator

    def score(self, pool: list[dict], query: dict) -> dict:
        """
        pool  : list of normalised pool items {id, title, score, properties}
        query : pinned feature dict {feature_name: raw_value}; {} → all features free
        Returns {"diversity": float, "surprise": float}
        """
        if not pool:
            return {"diversity": 0.0, "surprise": 0.0}

        feature_dicts  = [_extract_features(item) for item in pool]
        query_encoded  = self.encoder.encode_query(query)
        items_encoded  = self.encoder.encode_pool(feature_dicts)
        return self.calculator.compute(query_encoded, items_encoded)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m modules.retrieval_evaluator.diversity_evaluator"
            " <export_or_pool.json> [--pinned feat=value ...]"
        )
        sys.exit(1)

    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Accept both a raw pool list and a full export dict {meta, items}
    if isinstance(data, list):
        pool     = data
        nl_query = ""
    else:
        pool     = data.get("items", [])
        nl_query = data.get("meta", {}).get("query", "")

    # Parse --pinned key=value pairs
    query_features: dict = {}
    pinned_mode = False
    for arg in sys.argv[2:]:
        if arg == "--pinned":
            pinned_mode = True
            continue
        if pinned_mode and "=" in arg:
            k, v = arg.split("=", 1)
            query_features[k.strip()] = v.strip()

    print(f"Pool    : {len(pool)} items")
    print(f"NL query: {nl_query!r}")
    print(f"Pinned  : {query_features or '(none — all features free)'}")
    print()

    from modules.retrieval_evaluator.feature_encoder import MultilingualFeatureEncoder
    from modules.retrieval_evaluator.metrics_calculator import EntropyDiversityCalculator

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    print(f"Loading model: {model_name} …")
    encoder    = MultilingualFeatureEncoder(model_name=model_name)
    calculator = EntropyDiversityCalculator()
    evaluator  = DiversityEvaluator(encoder=encoder, calculator=calculator)

    print("Computing metrics …")
    result = evaluator.score(pool, query_features)
    print(json.dumps(result, indent=2))

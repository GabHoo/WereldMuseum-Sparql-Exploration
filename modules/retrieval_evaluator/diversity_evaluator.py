"""
Diversity-aware retrieval evaluator.

Each pool item's properties dict is used directly as its feature dict.
Non-semantic fields (URIs, internal identifiers) are stripped before encoding.
Features are matched across items by key name — no schema declaration needed.
Adding a new SPARQL field: it appears automatically in diversity computation.

Usage (CLI):
    python -m modules.retrieval_evaluator.diversity_evaluator path/to/export.json
    python -m modules.retrieval_evaluator.diversity_evaluator path/to/export.json \
        --pinned type=https://hdl.handle.net/20.500.11840/termmaster10069604

Environment:
    EMBEDDING_MODEL  — override the sentence-transformer model name
"""
import json
import os
import sys

from modules.base import RetrievalEvaluator
from modules.retrieval_evaluator.base import FeatureEncoder, MetricsCalculator, SurpriseCalculator

# Artifact fields (modules/artifact.py) that carry no semantic content for
# embedding: identifiers, URLs, concept URIs, measurements. Long free-text descriptions are
# also skipped to keep per-search embedding cost low.
_NOISE_KEYS = frozenset({
    "id", "images", "inventory_number", "dimensions", "description",
    "type_uris", "material_uris", "place_uris",
})


def _extract_features(item: dict) -> dict:
    """
    Build a feature dict from a pool item: its Artifact fields minus noise keys.
    Values are str or list[str]; empty values are dropped.
    """
    features = {}
    for k, v in item.get("properties", {}).items():
        if k in _NOISE_KEYS:
            continue
        if isinstance(v, str) and v.strip():
            features[k] = v
        elif isinstance(v, list):
            vals = [x for x in v if isinstance(x, str) and x.strip()]
            if vals:
                features[k] = vals
    return features


class DiversityEvaluator(RetrievalEvaluator):
    """
    Evaluates a retrieved pool on two separate axes:

    encoder + calculator : diversity — embedding spread over the free features
    surprise             : share of items outside the searched concepts (by URI)
    """

    def __init__(self, encoder: FeatureEncoder, calculator: MetricsCalculator,
                 surprise: SurpriseCalculator):
        self.encoder    = encoder
        self.calculator = calculator
        self.surprise   = surprise

    def score(self, pool: list[dict], query: dict) -> dict:
        """
        pool  : list of normalised pool items {id, title, score, properties}
        query : {feature_name: [concept URIs]} searched per feature; {} → all features free
        Returns {"diversity": float, "surprise": float | None} — surprise is
        None if the searched concepts could not be expanded (endpoint error).
        """
        if not pool:
            return {"diversity": 0.0, "surprise": 0.0}

        items_encoded = self.encoder.encode_pool([_extract_features(item) for item in pool])
        result = self.calculator.compute(items_encoded, set(query))
        try:
            result["surprise"] = self.surprise.compute(pool, query)
        except Exception:
            result["surprise"] = None
        return result


def build_default(knowledge_base) -> "DiversityEvaluator":
    """The standard evaluator: multilingual encoder, pairwise diversity, concept surprise."""
    from modules.retrieval_evaluator.feature_encoder import MultilingualFeatureEncoder
    from modules.retrieval_evaluator.metrics_calculator import PairwiseDiversityCalculator
    from modules.retrieval_evaluator.surprise_calculator import ConceptSurpriseCalculator

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    return DiversityEvaluator(
        encoder=MultilingualFeatureEncoder(model_name=model_name),
        calculator=PairwiseDiversityCalculator(),
        surprise=ConceptSurpriseCalculator(knowledge_base),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m modules.retrieval_evaluator.diversity_evaluator"
            " <export_or_pool.json> [--pinned feature=<concept URI> ...]"
        )
        sys.exit(1)

    from dotenv import load_dotenv
    from modules.knowledge_base.sparql_endpoint import SPARQLEndpoint
    load_dotenv()

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

    # Parse --pinned feature=<concept URI> pairs (repeat a feature for several concepts)
    query_features: dict[str, list[str]] = {}
    pinned_mode = False
    for arg in sys.argv[2:]:
        if arg == "--pinned":
            pinned_mode = True
            continue
        if pinned_mode and "=" in arg:
            k, v = arg.split("=", 1)
            query_features.setdefault(k.strip(), []).append(v.strip())

    print(f"Pool    : {len(pool)} items")
    print(f"NL query: {nl_query!r}")
    print(f"Pinned  : {query_features or '(none — all features free)'}")
    print()

    kb = SPARQLEndpoint(os.getenv("SPARQL_ENDPOINT_URL", ""))
    print("Loading model …")
    evaluator = build_default(kb)
    print("Computing metrics …")
    print(json.dumps(evaluator.score(pool, query_features), indent=2))

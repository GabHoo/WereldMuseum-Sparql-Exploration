"""
EntropyDiversityCalculator — pool-level diversity and surprise metrics.

Operates on plain dicts of {feature_name: np.ndarray | float}.
Features are matched across items by key name; missing keys are skipped.

surprise  = normalised Shannon entropy over per-item pinned-feature deviation counts
diversity = average pairwise cosine/numeric distance across free features
"""
import math
from collections import Counter

import numpy as np

from modules.retrieval_evaluator.base import MetricsCalculator


class EntropyDiversityCalculator(MetricsCalculator):
    """
    Computes pool-level surprise and diversity from encoded feature dicts.

    thresholds : {feature_name: float} deviation cutoffs (τ_f per feature).
                 Features not listed use default_threshold.
    default_threshold : fallback τ_f for any feature not in thresholds.
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        default_threshold: float = 0.30,
    ):
        self._thresholds = thresholds or {}
        self._default    = default_threshold

    def _tau(self, name: str) -> float:
        return self._thresholds.get(name, self._default)

    def compute(self, query_features: dict, item_features: list[dict]) -> dict:
        if not item_features:
            return {"diversity": 0.0, "surprise": 0.0}

        pinned = set(query_features.keys())
        free   = {name for fm in item_features for name in fm} - pinned

        surprise  = self._surprise(query_features, item_features, pinned)
        diversity = self._diversity(item_features, free)
        return {"diversity": round(diversity, 6), "surprise": round(surprise, 6)}

    # ── Surprise: pinned-feature entropy ─────────────────────────────────────

    def _surprise(self, query_features, item_features, pinned) -> float:
        k = len(pinned)
        if k == 0:
            return 0.0

        d_per_item: list[int] = []
        for fm in item_features:
            d = 0
            for name in pinned:
                q_val = query_features.get(name)
                i_val = fm.get(name)
                if q_val is None or i_val is None:
                    continue
                if self._distance(q_val, i_val) > self._tau(name):
                    d += 1
            d_per_item.append(d)

        counter = Counter(d_per_item)
        n = len(d_per_item)
        H = -sum((c / n) * math.log(c / n) for c in counter.values() if c > 0)
        return H / math.log(k + 1)

    # ── Diversity: free-feature intra-pool spread ─────────────────────────────

    def _diversity(self, item_features, free) -> float:
        if not free or len(item_features) < 2:
            return 0.0

        per_feature: list[float] = []
        for name in free:
            vals = [fm[name] for fm in item_features if fm.get(name) is not None]
            if len(vals) < 2:
                continue
            if isinstance(vals[0], (int, float)):
                per_feature.append(self._avg_pairwise_numeric(vals))
            else:
                per_feature.append(self._avg_pairwise_cosine(np.stack(vals)))

        return float(np.mean(per_feature)) if per_feature else 0.0

    # ── Distance helpers ──────────────────────────────────────────────────────

    def _distance(self, a, b) -> float:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b))
        # L2-normalised vectors → cosine_dist = 1 − dot(a, b)
        return float(1.0 - np.dot(np.asarray(a), np.asarray(b)))

    def _avg_pairwise_cosine(self, vecs: np.ndarray) -> float:
        S   = vecs @ vecs.T
        n   = vecs.shape[0]
        idx = np.triu_indices(n, k=1)
        return float(np.mean(1.0 - S[idx]))

    def _avg_pairwise_numeric(self, vals: list) -> float:
        arr = np.array(vals, dtype=float)
        n   = len(arr)
        idx = np.triu_indices(n, k=1)
        return float(np.mean(np.abs(arr[:, None] - arr[None, :])[idx]))

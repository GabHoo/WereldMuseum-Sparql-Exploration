"""
PairwiseDiversityCalculator — pool-level diversity over free features.

Operates on plain dicts of {feature_name: np.ndarray | float}.
Features are matched across items by key name; missing keys are skipped.

diversity = average pairwise cosine/numeric distance per free (non-searched)
            feature, averaged over free features
"""
import numpy as np

from modules.retrieval_evaluator.base import MetricsCalculator


class PairwiseDiversityCalculator(MetricsCalculator):
    def compute(self, item_features: list[dict], pinned: set) -> dict:
        free = {name for fm in item_features for name in fm} - set(pinned)
        return {"diversity": round(self._diversity(item_features, free), 6)}

    def _diversity(self, item_features, free) -> float:
        if not free or len(item_features) < 2:
            return 0.0

        per_feature: list[float] = []
        for name in free:
            vals = [fm[name] for fm in item_features if fm.get(name) is not None]
            # A feature can mix kinds (e.g. a title that is just a number);
            # use whichever kind most items have and skip the rest.
            nums = [v for v in vals if isinstance(v, (int, float))]
            vecs = [v for v in vals if not isinstance(v, (int, float))]
            if len(vecs) >= len(nums) and len(vecs) >= 2:
                per_feature.append(self._avg_pairwise_cosine(np.stack(vecs)))
            elif len(nums) >= 2:
                per_feature.append(self._avg_pairwise_numeric(nums))

        return float(np.mean(per_feature)) if per_feature else 0.0

    def _avg_pairwise_cosine(self, vecs: np.ndarray) -> float:
        # Vectors are L2-normalised, so cosine distance = 1 − dot product.
        S   = vecs @ vecs.T
        idx = np.triu_indices(vecs.shape[0], k=1)
        return float(np.mean(1.0 - S[idx]))

    def _avg_pairwise_numeric(self, vals: list) -> float:
        arr = np.array(vals, dtype=float)
        idx = np.triu_indices(len(arr), k=1)
        return float(np.mean(np.abs(arr[:, None] - arr[None, :])[idx]))

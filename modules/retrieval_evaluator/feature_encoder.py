"""
MultilingualFeatureEncoder — encodes any {key: value} feature dict.

Accepts a plain dict of feature values (strings or numbers) and returns an
encoded dict of the same keys with values replaced by:
  - np.ndarray (L2-normalised, float32) for text values
  - float for numeric values (detected by trying float(v))
  - key omitted if value is None / empty string

Model: paraphrase-multilingual-mpnet-base-v2
  Works on Dutch and 50+ languages. L2-normalised output means
  cosine_distance(a, b) = 1 − dot(a, b) (no extra step needed).
  ~420 MB, cached in ~/.cache/torch/sentence_transformers on first use.

  Alternative for CPU-constrained environments:
  paraphrase-multilingual-MiniLM-L12-v2  (384-dim, ~2× faster)
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from modules.retrieval_evaluator.base import FeatureEncoder

_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def _is_numeric(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class MultilingualFeatureEncoder(FeatureEncoder):
    """
    Encodes feature dicts using a multilingual sentence-transformer.

    Both query and item features are plain {key: value} dicts.
    The encoder returns a dict with the same keys; values become:
      - np.ndarray for text
      - float for numeric
      - key absent for None / empty string

    An internal text cache deduplicates embeddings across calls so that
    repeated label values (e.g. many items with the same typeLabel) are
    only embedded once per server lifetime.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model = SentenceTransformer(model_name)
        self._text_cache: dict[str, np.ndarray] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def encode_query(self, query: dict) -> dict:
        """Encode a query feature dict. query = {feature_name: raw_value}."""
        return self._encode_dict(query)

    def encode_item(self, item: dict) -> dict:
        """Encode a single feature dict (extracted from a pool item externally)."""
        return self._encode_dict(item)

    def encode_pool(self, items: list) -> list:
        """
        Batch-encode a list of feature dicts (already extracted from pool items).
        Collects all unique text values across the list and embeds them in one call.
        """
        # Collect unseen text values from all items
        unseen: list[str] = []
        for feat_dict in items:
            for v in feat_dict.values():
                if isinstance(v, str) and v and not _is_numeric(v) and v not in self._text_cache:
                    unseen.append(v)

        unique_unseen = list(dict.fromkeys(unseen))
        if unique_unseen:
            vecs = self._model.encode(
                unique_unseen,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vec in zip(unique_unseen, vecs):
                self._text_cache[text] = vec

        return [self._encode_dict(fd) for fd in items]

    # ── Private ───────────────────────────────────────────────────────────────

    def _encode_dict(self, feat_dict: dict) -> dict:
        result = {}
        texts_to_embed: list[str] = []

        for k, v in feat_dict.items():
            if v is None or v == "":
                continue
            if _is_numeric(v):
                result[k] = float(v)
            elif isinstance(v, str):
                if v in self._text_cache:
                    result[k] = self._text_cache[v]
                else:
                    texts_to_embed.append(v)
                    result[k] = v  # placeholder, replaced below

        if texts_to_embed:
            unique = list(dict.fromkeys(texts_to_embed))
            vecs = self._model.encode(
                unique,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vec in zip(unique, vecs):
                self._text_cache[text] = vec
            # Replace placeholders
            result = {
                k: (self._text_cache[v] if isinstance(v, str) and v in self._text_cache else v)
                for k, v in result.items()
            }

        return result

"""
MultilingualFeatureEncoder — encodes any {key: value} feature dict.

Accepts a dict of feature values and returns an encoded dict of the same keys:
  - text value          → np.ndarray (L2-normalised, float32)
  - list of text values → mean of the values' embeddings, re-normalised, so a
                          multi-valued field (e.g. two materials) is one vector
                          that sits between its values
  - numeric value       → float (detected by trying float(v))
  - None / "" / []      → key omitted

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

    Item features are plain {key: value} dicts.
    The encoder returns a dict with the same keys; values become:
      - np.ndarray for text, and for lists of text (mean of the embeddings)
      - float for numeric
      - key absent for None / empty string / empty list

    An internal text cache deduplicates embeddings across calls so that
    repeated label values (e.g. many items with the same type) are
    only embedded once per server lifetime.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model = SentenceTransformer(model_name)
        self._text_cache: dict[str, np.ndarray] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def encode_item(self, item: dict) -> dict:
        """Encode a single feature dict (extracted from a pool item externally)."""
        return self._encode_dict(item)

    def encode_pool(self, items: list) -> list:
        """
        Batch-encode a list of feature dicts (already extracted from pool items).
        Collects all unique text values across the list and embeds them in one call.
        """
        texts = [t for fd in items for v in fd.values() for t in self._texts_of(v)]
        self._embed_missing(texts)
        return [self._encode_dict(fd) for fd in items]

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _texts_of(v) -> list[str]:
        """Text values inside v that need an embedding (numbers excluded)."""
        if isinstance(v, str):
            return [v] if v and not _is_numeric(v) else []
        if isinstance(v, list):
            return [x for x in v if isinstance(x, str) and x]
        return []

    def _embed_missing(self, texts: list[str]) -> None:
        unique = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        if not unique:
            return
        vecs = self._model.encode(unique, normalize_embeddings=True, show_progress_bar=False)
        for text, vec in zip(unique, vecs):
            self._text_cache[text] = vec

    def _encode_dict(self, feat_dict: dict) -> dict:
        self._embed_missing([t for v in feat_dict.values() for t in self._texts_of(v)])
        result = {}
        for k, v in feat_dict.items():
            if v is None or v == "" or v == []:
                continue
            if isinstance(v, list):
                vecs = [self._text_cache[t] for t in self._texts_of(v)]
                if vecs:
                    mean = np.mean(vecs, axis=0)
                    norm = np.linalg.norm(mean)
                    result[k] = (mean / norm if norm > 0 else mean).astype(np.float32)
            elif _is_numeric(v):
                result[k] = float(v)
            elif isinstance(v, str):
                result[k] = self._text_cache[v]
        return result

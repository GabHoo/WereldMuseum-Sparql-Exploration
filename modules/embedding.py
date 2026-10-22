"""
Shared text embedding.

One sentence-transformer model per model name is loaded per process and shared
by every module that needs embeddings (the retrieval evaluator's encoder, the
cluster selection strategy, ...), so enabling several of them does not load the
~420 MB model twice.

TextEmbedder adds a per-text cache: each distinct string is embedded once per
server lifetime.
"""
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@lru_cache(maxsize=None)
def load_sentence_model(model_name: str = DEFAULT_MODEL):
    """Load (once per process) and return the SentenceTransformer for model_name."""
    from sentence_transformers import SentenceTransformer  # heavy import, only when used
    return SentenceTransformer(model_name)


class TextEmbedder:
    """Embeds strings into L2-normalised vectors (cosine similarity = dot product)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model = load_sentence_model(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), one normalised row per text."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self._model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            for text, vec in zip(missing, vecs):
                self._cache[text] = vec
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 0))

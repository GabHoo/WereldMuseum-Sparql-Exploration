"""
Shared sentence-transformer text model.

One model per model name is loaded per process and shared by every module that
embeds text (the metadata_text embedding, the retrieval evaluator's encoder,
...), so enabling several of them does not load the ~420 MB model twice.
"""
from functools import lru_cache

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@lru_cache(maxsize=None)
def load_sentence_model(model_name: str = DEFAULT_MODEL):
    """Load (once per process) and return the SentenceTransformer for model_name."""
    from sentence_transformers import SentenceTransformer  # heavy import, only when used
    return SentenceTransformer(model_name)

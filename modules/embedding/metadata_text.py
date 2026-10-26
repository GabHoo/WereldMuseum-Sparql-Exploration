"""
MetadataTextEncoder — embeds an object's metadata as one line of text.

The text is the configured Artifact fields joined into one line by
metadata_linearization (EMBEDDING_TEXT_FIELDS, default "title"), e.g.
"HOUTEN MASKER · maskers (kleding) · hout, kleurstof · Ivoorkust"; objects
whose fields are all empty become "(untitled)". The model is the multilingual
sentence-transformer EMBEDDING_MODEL, shared with the RetrievalEvaluator.
One view: "text". Fast: a 100-object pool takes about a second.

Each distinct text is embedded once and cached: the same text then always gets
the very same vector. (Encoded in different batches, the model's output differs
in the last digits, which would split e.g. all "(untitled)" objects over
several clusters.)
"""
import numpy as np

from modules.embedding.base import ObjectEncoder
from modules.embedding.sentence_model import DEFAULT_MODEL, load_sentence_model
from modules.selection.utils import metadata_linearization

_UNTITLED = "(untitled)"


class MetadataTextEncoder(ObjectEncoder):
    def __init__(self, model_name: str = DEFAULT_MODEL, fields: str = "title"):
        self.views = ("text",)
        self._linearize = metadata_linearization(fields)   # checks the field names now
        self._model = load_sentence_model(model_name)
        self._cache: dict[str, np.ndarray] = {}   # text → vector

    def object_text(self, properties: dict) -> str:
        return self._linearize({"properties": properties}) or _UNTITLED

    def encode(self, objects: list) -> list:
        texts = [o["text"] for o in objects]
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self._model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            for text, vec in zip(missing, vecs):
                self._cache[text] = np.asarray(vec, dtype=np.float32)
        return [{"text": self._cache[t]} for t in texts]

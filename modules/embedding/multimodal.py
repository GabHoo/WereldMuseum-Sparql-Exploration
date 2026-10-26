"""
MultimodalEncoder — object embeddings with EmbeddingGemma 2 (EMBEDDING=multimodal).

google/embeddinggemma-2 maps text, images (and video/audio, unused here) into
one shared 768-d space, alone or combined. Per object it computes the
configured views (config.py passes just EMBEDDING_VIEW):
    fused — "<metadata text> <|image|>" with the object's image, one input
    image — the image alone (no prefix, as the model card prescribes)
    text  — the metadata text alone
Text uses the model's document format: "title: <title> | text: <fields>".

Images cost most of the time: each becomes EMBEDDING_IMAGE_TOKENS vision
tokens (model default 280; 140 or 70 is ~2× / ~4× faster at a cosine of ~0.98 /
~0.97 to the default vectors). Text alone is ~100× cheaper.

Matryoshka: vectors can be truncated to 512 / 256 / 128 dims (EMBEDDING_DIM)
and are re-normalised after truncation. Never load the model in float16 — the
model card warns it silently produces NaN or degraded embeddings.

Needs transformers 5.19+ (EmbeddingGemma2 model + processor classes).
"""
import numpy as np

from modules.embedding.base import VIEWS, ObjectEncoder

DEFAULT_MODEL = "google/embeddinggemma-2"

# Artifact fields (modules/artifact.py) that go into the "text:" part of the
# document, in this order. Identifiers, URIs, images and dimensions carry no
# semantic content; the title has its own "title:" slot.
_TEXT_FIELDS = ("type", "material", "place", "date", "maker", "intended_use", "description")
_MAX_DESCRIPTION_CHARS = 500


class MultimodalEncoder(ObjectEncoder):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        views: tuple = ("fused",),
        dim: int | None = None,
        device: str | None = None,
        dtype: str = "auto",
        batch_size: int = 8,
        image_tokens: int | None = None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        unknown = set(views) - set(VIEWS)
        if unknown or not views:
            raise ValueError(f"EMBEDDING_VIEW must be one of {VIEWS}, got {views!r}")
        self.views = tuple(views)
        self._dim = dim
        self._batch_size = batch_size

        if dtype == "auto":
            # bfloat16 is fast on GPUs; on CPU / MPS float32 is the safe choice.
            dtype = "bfloat16" if torch.cuda.is_available() else "float32"
        if dtype not in ("bfloat16", "float32"):
            raise ValueError("EMBEDDING_DTYPE must be bfloat16 or float32 (float16 breaks this model)")

        # Load only the encoders we need: never audio, vision only for image views.
        config_kwargs = {"audio_config": None}
        if self.views == ("text",):
            config_kwargs["vision_config"] = None

        # Vision tokens per image (70 | 140 | 280 | 560 | 1120); None = model default.
        processor_kwargs = {"max_soft_tokens": image_tokens, "image_seq_length": image_tokens} if image_tokens else None

        self._model = SentenceTransformer(
            model_name,
            device=device or None,
            model_kwargs={"dtype": getattr(torch, dtype)},
            config_kwargs=config_kwargs,
            processor_kwargs=processor_kwargs,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def object_text(self, properties: dict) -> str:
        """Document-format text: 'title: <title | none> | text: type: …; material: …'."""
        parts = []
        for name in _TEXT_FIELDS:
            v = properties.get(name)
            vals = [x for x in (v if isinstance(v, list) else [v]) if isinstance(x, str) and x.strip()]
            if not vals:
                continue
            text = ", ".join(vals)
            if name == "description":
                text = text[:_MAX_DESCRIPTION_CHARS]
            parts.append(f"{name.replace('_', ' ')}: {text}")
        title = (properties.get("title") or "").strip() or "none"
        return f"title: {title} | text: {'; '.join(parts)}"

    def encode(self, objects: list) -> list:
        results = [{} for _ in objects]
        for view in self.views:
            idx, inputs = [], []
            for i, obj in enumerate(objects):
                inp = self._input(view, obj)
                if inp is not None:
                    idx.append(i)
                    inputs.append(inp)
            if not inputs:
                continue
            vecs = self._model.encode(
                inputs,
                prompt="",                   # text is already in document format
                batch_size=self._batch_size,
                truncate_dim=self._dim,
                normalize_embeddings=True,   # applied after truncation
                show_progress_bar=False,
            )
            for i, vec in zip(idx, vecs):
                results[i][view] = np.asarray(vec, dtype=np.float32)
        return results

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _input(view: str, obj: dict):
        text, image = obj.get("text") or "", obj.get("image_path")
        if view == "text":
            return text or None
        if view == "image":
            return {"image": [image]} if image else None
        # fused: without a usable image, fall back to the text alone so every
        # object still gets a fused vector (same embedding space).
        if image:
            return {"text": f"{text} <|image|>", "image": [image]}
        return text or None


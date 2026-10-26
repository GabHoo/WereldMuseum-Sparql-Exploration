"""
EmbeddingGemmaEncoder — object representations with EmbeddingGemma 2.

google/embeddinggemma-2 maps text, images (and video/audio, unused here) into
one shared 768-d space, alone or combined. Per object it computes the
configured views:
    fused — "<metadata text> <|image|>" with the object's image, one input
    image — the image alone (no prefix, as the model card prescribes)
    text  — the metadata text alone
Text uses the model's document format: "title: <title> | text: <fields>".

Images cost most of the time: each becomes REPRESENTATION_IMAGE_TOKENS vision
tokens (model default 280; 140 or 70 is ~2× / ~4× faster at a cosine of ~0.98 /
~0.97 to the default vectors). Text alone is ~100× cheaper.

Matryoshka: vectors can be truncated to 512 / 256 / 128 dims (REPRESENTATION_DIM)
and are re-normalised after truncation. Never load the model in float16 — the
model card warns it silently produces NaN or degraded embeddings.

Usage (CLI):
    python -m modules.representation.embeddinggemma path/to/export.json [--k 3]

Prints, per view, each object's nearest neighbours within the export.

Environment: REPRESENTATION_MODEL, REPRESENTATION_VIEWS, REPRESENTATION_DIM,
REPRESENTATION_DEVICE, REPRESENTATION_DTYPE, REPRESENTATION_BATCH_SIZE,
REPRESENTATION_IMAGE_TOKENS
"""
import json
import os
import sys

import numpy as np

from modules.representation.base import VIEWS, ObjectEncoder

_DEFAULT_MODEL = "google/embeddinggemma-2"

# Artifact fields (modules/artifact.py) that go into the "text:" part of the
# document, in this order. Identifiers, URIs, images and dimensions carry no
# semantic content; the title has its own "title:" slot.
_TEXT_FIELDS = ("type", "material", "place", "date", "maker", "intended_use", "description")
_MAX_DESCRIPTION_CHARS = 500


class EmbeddingGemmaEncoder(ObjectEncoder):
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
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
            raise ValueError(f"REPRESENTATION_VIEWS must be a subset of {VIEWS}, got {views!r}")
        self.views = tuple(views)
        self._dim = dim
        self._batch_size = batch_size

        if dtype == "auto":
            # bfloat16 is fast on GPUs; on CPU / MPS float32 is the safe choice.
            dtype = "bfloat16" if torch.cuda.is_available() else "float32"
        if dtype not in ("bfloat16", "float32"):
            raise ValueError("REPRESENTATION_DTYPE must be bfloat16 or float32 (float16 breaks this model)")

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


def build_from_env() -> EmbeddingGemmaEncoder:
    views = tuple(v.strip() for v in os.getenv("REPRESENTATION_VIEWS", "fused").split(",") if v.strip())
    dim = os.getenv("REPRESENTATION_DIM", "").strip()
    image_tokens = os.getenv("REPRESENTATION_IMAGE_TOKENS", "").strip()
    return EmbeddingGemmaEncoder(
        model_name=os.getenv("REPRESENTATION_MODEL", _DEFAULT_MODEL),
        views=views,
        dim=int(dim) if dim else None,
        device=os.getenv("REPRESENTATION_DEVICE", "").strip() or None,
        dtype=os.getenv("REPRESENTATION_DTYPE", "auto").strip() or "auto",
        batch_size=int(os.getenv("REPRESENTATION_BATCH_SIZE", "8")),
        image_tokens=int(image_tokens) if image_tokens else None,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m modules.representation.embeddinggemma <export_or_pool.json> [--k 3]")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()
    import config

    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    pool = data if isinstance(data, list) else data.get("items", [])
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 3

    print(f"Pool: {len(pool)} items")
    print("Loading model …")
    from modules.representation.memory_bank import RepresentationService
    service = RepresentationService(build_from_env(), config._build_image_cache())
    print("Encoding …")
    service.on_search(pool, search_id="cli")
    service.wait("search", timeout=None)

    titles = {item["id"]: item.get("title") or item["id"] for item in pool}
    for view in service.encoder.views:
        ids, mat = service.search_bank.matrix(view)
        print(f"\n── {view} ({len(ids)} vectors) ──")
        if len(ids) < 2:
            continue
        sims = mat @ mat.T
        np.fill_diagonal(sims, -np.inf)
        for i, item_id in enumerate(ids):
            nn = np.argsort(-sims[i])[:k]
            neighbours = ", ".join(f"{titles[ids[j]][:40]} ({sims[i, j]:.2f})" for j in nn)
            print(f"{titles[item_id][:50]:50}  →  {neighbours}")

"""
Interface for object encoders used by the Representation module.

An ObjectEncoder turns collection objects (metadata text + a local image file)
into vectors, one per *view*:
    fused — text and image in one multimodal input
    image — the image alone
    text  — the metadata text alone
Which views are computed is configured (REPRESENTATION_VIEWS); every vector is
L2-normalised, so cosine similarity is a dot product.

RepresentationService (memory_bank.py) calls the encoder from a background
thread and keeps the vectors in two memory banks (search results / selection).
"""
from abc import ABC, abstractmethod

VIEWS = ("fused", "image", "text")


class ObjectEncoder(ABC):
    views: tuple = ("fused",)   # subset of VIEWS, set by the constructor

    @abstractmethod
    def encode(self, objects: list) -> list:
        """
        objects : [{"id": str, "text": str, "image_path": str | None}]
        Returns one dict per object, in order: {view: np.ndarray (float32, L2-normalised)}.
        A view is omitted when its input is missing (e.g. "image" without an image).
        """

    def object_text(self, properties: dict) -> str:
        """
        Text to embed for an Artifact dict (pool item "properties").
        Override to change which fields are used or how they are formatted.
        """
        return properties.get("title") or ""

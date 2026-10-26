from modules.embedding.base import ObjectEncoder


class MyObjectEncoder(ObjectEncoder):
    """
    Stub — copy this file, rename the class, implement encode(), then:
      1. Add an entry to REGISTRY["embedding"] in config.py
      2. Add a branch in _build_encoder() in config.py if it needs constructor args
      3. Set EMBEDDING=<your-key> in .env
    Every embedding-based selection strategy (clustering, rerank) can then use it.
    """

    def __init__(self, views: tuple = ("text",)):
        self.views = views

    def encode(self, objects: list) -> list:
        """
        objects : [{"id": str, "text": str, "image_path": str | None}]
                  text       — from object_text(properties)
                  image_path — local JPEG thumbnail of the object's first real image
                               (None when needs_images is False, i.e. text views only)
        Return per object {view: L2-normalised float32 vector} for view in self.views;
        omit a view whose input is missing.
        """
        raise NotImplementedError

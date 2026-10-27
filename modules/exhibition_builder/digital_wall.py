"""
Digital wall exhibition builder.

build() serves an editor page (templates/digital_wall.html, static/digital_wall/)
where the curated objects can be dragged onto a gallery wall, annotated with
pen and text, and combined with props (pillar, frame). The composed scene is
then sent to render(), which asks the OpenAI image-edit API to turn it into a
photorealistic museum exhibition — one faithful version plus re-imaginings in
different gallery styles.

Rendering needs OPENAI_API_KEY in .env; composing the wall works without it.
"""
import base64

import openai
from flask import render_template
from openai import OpenAI

from modules.base import ExhibitionBuilder

DEFAULT_DESCRIPTION = "an arrangement of museum artifacts"

# Variation 0 is the faithful render of the user's arrangement; variations 1+
# reimagine it in these styles. The editor caps the count at 1 + len(...).
VARIATION_STYLES = [
    "a dramatic black-box gallery with focused spotlights, deep shadows and pools of warm light",
    "a bright, airy contemporary gallery with white walls, skylights and minimalist plinths",
    "an immersive gallery themed around the artifacts’ culture of origin, with rich textures, "
    "patterned walls and ambient colored light",
    "a grand classical museum hall with high ceilings, stone columns, parquet floors "
    "and ornate glass display cases",
]


def build_prompt(description: str, variation: int) -> str:
    """Image-edit prompt: 0 = faithful render, 1+ = a re-imagining in VARIATION_STYLES."""
    desc = description or DEFAULT_DESCRIPTION
    if variation == 0:
        return (
            "Transform this image into a photorealistic museum exhibition layout. "
            f"The scene shows: {desc}. "
            "Keep the artifacts the same. Make sure the arrangement is similar and recognizable; "
            "render the surrounding gallery space, lighting, walls, as a real museum gallery."
        )
    style = VARIATION_STYLES[(variation - 1) % len(VARIATION_STYLES)]
    return (
        "Reimagine this arrangement of museum artifacts as a photorealistic museum exhibition. "
        f"The scene shows: {desc}. "
        "Keep the same artifacts recognizable and in a similar arrangement, but creatively redesign "
        f"the exhibition as {style}. Be inventive with the gallery architecture, lighting and layout."
    )


class DigitalWall(ExhibitionBuilder):
    """Wall editor for the curated selection + OpenAI photorealistic render."""

    def __init__(self, api_key: str, image_model: str, image_size: str):
        self._client = OpenAI(api_key=api_key) if api_key else None
        self.image_model = image_model
        self.image_size = image_size

    def build(self, export_data: dict) -> str:
        return render_template("digital_wall.html", export_data=export_data)

    def render(self, scene_png: bytes, description: str, variation: int) -> bytes:
        if self._client is None:
            raise RuntimeError("OPENAI_API_KEY is not set in .env — needed to render the exhibition.")
        try:
            result = self._client.images.edit(
                model=self.image_model,
                image=("scene.png", scene_png, "image/png"),
                prompt=build_prompt(description, variation),
                size=self.image_size,
            )
        except openai.APIError as e:
            msg = f"OpenAI error: {e}"
            if getattr(e, "code", None) == "model_not_found":
                msg += " — try another DIGITAL_WALL_IMAGE_MODEL in .env (e.g. gpt-image-1)."
            raise RuntimeError(msg) from e

        b64 = result.data[0].b64_json if result.data else None   # gpt-image models always return base64
        if not b64:
            raise RuntimeError("No image returned by the OpenAI API.")
        return base64.b64decode(b64)

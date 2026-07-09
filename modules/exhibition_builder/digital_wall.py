"""
Digital wall exhibition builder — skeleton implementation.

Takes the JSON produced by the curation tool and lays it out as a
digital wall exhibition. Implementation to be filled in by a colleague.

Usage (CLI):
    python -m modules.exhibition_builder.digital_wall path/to/export.json
"""

import json
import sys

from modules.base import ExhibitionBuilder


class DigitalWall(ExhibitionBuilder):
    """Renders a curated selection as a digital wall exhibition."""

    def build(self, export_data: dict) -> None:
        raise NotImplementedError("DigitalWall.build() is not yet implemented.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m modules.exhibition_builder.digital_wall <export.json>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded export: {len(data.get('items', []))} items, query={data.get('meta', {}).get('query', '')!r}")
    builder = DigitalWall()
    builder.build(data)

import json

from modules.base import Exporter


class JSONExporter(Exporter):
    """Exports the curated selection as a pretty-printed JSON file."""

    def export(self, items: list, meta: dict) -> bytes:
        clean = [
            {"id": item["id"], "title": item["title"], "properties": item.get("properties", {})}
            for item in items
        ]
        payload = {"meta": meta, "items": clean}
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

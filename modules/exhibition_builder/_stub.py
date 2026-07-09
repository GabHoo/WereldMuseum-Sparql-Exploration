from modules.base import ExhibitionBuilder


class MyExhibitionBuilder(ExhibitionBuilder):
    """
    Stub — copy this file, rename the class, implement build(), then:
      1. Add an entry to REGISTRY["exhibition_builder"] in config.py
      2. Set EXHIBITION_BUILDER=<your-key> in .env
    """

    def build(self, export_data: dict) -> None:
        """
        export_data : dict with keys:
            "meta"  — { query, sparql, timestamp }
            "items" — list of { id, title, properties: {...} }

        Produce whatever output your exhibition format requires
        (HTML page, PDF, API call to a wall renderer, …).
        """
        raise NotImplementedError

from modules.base import ExhibitionBuilder


class MyExhibitionBuilder(ExhibitionBuilder):
    """
    Stub — copy this file, rename the class, implement build(), then:
      1. Add an entry to REGISTRY["exhibition_builder"] in config.py
      2. Set EXHIBITION_BUILDER=<your-key> in .env
    """

    def build(self, export_data: dict) -> str:
        """
        export_data : dict with keys:
            "meta"  — { query, sparql, timestamp }
            "items" — list of { id, title, properties: {...} }

        Return the exhibition as an HTML page; app.py serves it at /exhibition
        when the user clicks *Build Exhibition*. Optionally override render()
        if the page composes scenes that should be turned into images.
        """
        raise NotImplementedError

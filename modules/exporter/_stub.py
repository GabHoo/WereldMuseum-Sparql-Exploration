from modules.base import Exporter


class MyExporter(Exporter):
    """
    Stub — copy this file, rename the class, implement export(), then:
      1. Add an entry to REGISTRY["exporter"] in config.py
      2. Set EXPORTER=<your-key> in .env
    """

    def export(self, items: list, meta: dict) -> bytes:
        """
        items : list of selected pool items (score field is already stripped).
                Each item: { id, title, properties: {...} }
        meta  : { query, sparql, timestamp }

        Return the file content as bytes. Flask sends this as a download.
        Set the correct MIME type and filename in app.py's /export route
        if your format differs from JSON.
        """
        raise NotImplementedError

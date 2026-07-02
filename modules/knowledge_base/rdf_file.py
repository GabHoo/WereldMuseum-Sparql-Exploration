from rdflib import Graph

from modules.base import KnowledgeBase


class RDFFile(KnowledgeBase):
    """
    Loads a local RDF file (Turtle, N-Triples, RDF/XML, JSON-LD, …)
    and executes SPARQL queries in-process via rdflib.

    Module-specific config (set via .env):
        RDF_FILE_PATH : path to the local RDF file (e.g. collection.ttl)

    Supports any format rdflib can parse; format is auto-detected from the
    file extension. The graph is loaded once at startup.
    """

    def __init__(self, file_path: str):
        if not file_path:
            raise ValueError("RDF_FILE_PATH must be set when KNOWLEDGE_BASE=rdf_file")
        self._graph = Graph()
        self._graph.parse(file_path)

    def execute(self, sparql: str) -> list:
        results = self._graph.query(sparql)
        rows = []
        for row in results:
            item = {}
            for var in results.vars:
                val = row[var]
                item[str(var)] = str(val) if val is not None else ""
            rows.append(item)
        return rows

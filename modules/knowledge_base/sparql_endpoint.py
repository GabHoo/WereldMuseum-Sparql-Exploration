from SPARQLWrapper import SPARQLWrapper, JSON

from modules.base import KnowledgeBase


class SPARQLEndpoint(KnowledgeBase):
    """
    Executes SPARQL queries against a remote HTTP endpoint.

    Module-specific config (set via .env):
        SPARQL_ENDPOINT_URL : URL of the SPARQL endpoint
    """

    def __init__(self, endpoint_url: str):
        self._endpoint_url = endpoint_url

    def execute(self, sparql: str) -> list:
        wrapper = SPARQLWrapper(self._endpoint_url)
        wrapper.setQuery(sparql)
        wrapper.setReturnFormat(JSON)
        results = wrapper.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        return [{var: row[var]["value"] for var in row} for row in bindings]

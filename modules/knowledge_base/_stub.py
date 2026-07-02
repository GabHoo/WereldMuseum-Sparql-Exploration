from modules.base import KnowledgeBase


class MyKnowledgeBase(KnowledgeBase):
    """
    Stub — copy this file, rename the class, implement execute(), then:
      1. Add an entry to REGISTRY["knowledge_base"] in config.py
      2. Add a builder branch in _build_knowledge_base() in config.py
      3. Set KNOWLEDGE_BASE=<your-key> in .env
    """

    def execute(self, sparql: str) -> list:
        """
        sparql : a valid SPARQL SELECT query string

        Must return a list of dicts, one per result row.
        Keys are the variable names from the SELECT clause; values are strings.
        Example: [{"artifact": "https://...", "artifacttitle": "Mask", ...}, ...]
        """
        raise NotImplementedError

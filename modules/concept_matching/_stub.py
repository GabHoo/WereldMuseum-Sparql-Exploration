from modules.base import ConceptMatcher


class MyConceptMatcher(ConceptMatcher):
    """
    Stub — copy this file, rename the class, implement match(), then:
      1. Add an entry to REGISTRY["concept_matcher"] in config.py
      2. If it needs constructor args, add a branch in _build_concept_matcher()
      3. Set CONCEPT_MATCHER=<your-key> in .env

    Any matcher plugs into every concept-based QueryGeneration (e.g. concept_query),
    so you can swap thesaurus lookup for embeddings or an LLM without touching SPARQL.
    """

    def match(self, nl_query: str) -> list:
        """
        nl_query : free text typed by the user

        Must return a list of {"uri": str, "label": str | None, "feature": str},
        where feature is one of the shared feature keys (type | material | place).
        """
        raise NotImplementedError

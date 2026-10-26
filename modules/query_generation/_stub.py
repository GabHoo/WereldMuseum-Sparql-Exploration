from modules.base import QueryGeneration


class MyQueryGeneration(QueryGeneration):
    """
    Stub — copy this file, rename the class, implement convert(), then:
      1. Add an entry to REGISTRY["query_generation"] in config.py
      2. Set QUERY_GENERATION=<your-key> in .env

    If your implementation works from free text, consider taking a
    ConceptMatcher in the constructor (see concept_query.py) instead of
    doing the text → URI matching here, so matchers stay swappable.
    Optionally override match_concepts() / get_query_features() too.
    """

    def convert(self, nl_query: str, context: dict) -> str:
        """
        nl_query : user input string (keyword or free text)
        context  : dict from config — contains whatever graph-specific data
                   this implementation needs, e.g.:
                     context["sparql_template"]   → SPARQL template override
                     context["schema_snippet"]    → ontology snippet for LLM prompts

        Must return a valid SPARQL SELECT query string.
        """
        raise NotImplementedError

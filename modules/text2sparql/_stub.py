from modules.base import Text2SPARQL


class MyText2SPARQL(Text2SPARQL):
    """
    Stub — copy this file, rename the class, implement convert(), then:
      1. Add an entry to REGISTRY["text2sparql"] in config.py
      2. Set TEXT2SPARQL=<your-key> in .env
    """

    def convert(self, nl_query: str, context: dict) -> str:
        """
        nl_query : user input string (category name, keyword, or free text)
        context  : dict from config.TEXT2SPARQL_CONTEXT — contains whatever
                   graph-specific data this implementation needs, e.g.:
                     context["categories"]        → {label: uri}
                     context["sparql_template"]   → SPARQL string with {category_uri}
                     context["keyword_sparql_template"] → SPARQL string with {keyword}
                     context["schema_snippet"]    → ontology snippet for LLM prompts

        Must return a valid SPARQL SELECT query string.
        """
        raise NotImplementedError

"""
ConceptQuery — turns matched concepts into a SPARQL query.

This module only decides *what to fetch* for a set of concepts. Deciding
*which concepts the user meant* is delegated to a ConceptMatcher
(modules/concept_matching/), injected via config, so matchers and query
templates can be swapped independently.

Configure via .env:
    QUERY_GENERATION=concept_query
    CONCEPT_MATCHER=thesaurus

context["sparql_template"] (optional) overrides the outer template at call
time; it must use {required_pattern} as the placeholder and SELECT ?artifact.
"""
from modules.base import ConceptMatcher, QueryGeneration

# Find query: returns only the URIs of matching artifacts. Their metadata is
# fetched afterwards by ArtifactFetcher, so LIMIT counts artifacts, not rows.
# {required_pattern} is filled by convert() with one anchor block per feature.
_SELECT_TEMPLATE = """
    PREFIX la:   <https://linked.art/ns/terms/>
    PREFIX crm:  <http://www.cidoc-crm.org/cidoc-crm/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT DISTINCT ?artifact
    WHERE {{
      {required_pattern}
      FILTER EXISTS {{
        ?artifact crm:P65_shows_visual_item/la:digitally_shown_by/la:access_point ?image .
      }}
    }}
    LIMIT 100
"""

# Per-feature required anchor pattern. Keys are the shared feature names,
# which are also Artifact field names (modules/artifact.py).
_REQUIRED_PATTERNS = {
    "type": (
        "VALUES ?chosenType {{ {uri_values} }}\n"
        "      ?chosenType skos:narrower* ?specificType .\n"
        "      ?artifact crm:P2_has_type ?specificType ."
    ),
    "material": (
        "VALUES ?chosenMat {{ {uri_values} }}\n"
        "      ?chosenMat skos:narrower* ?specificMat .\n"
        "      ?artifact crm:P45_consists_of ?specificMat ."
    ),
    "place": (
        "VALUES ?chosenPlace {{ {uri_values} }}\n"
        "      ?chosenPlace skos:narrower* ?specificPlace .\n"
        "      ?artifact crm:P108i_was_produced_by ?prod .\n"
        "      ?prod crm:P7_took_place_at ?specificPlace ."
    ),
}


class ConceptQuery(QueryGeneration):
    def __init__(self, matcher: ConceptMatcher):
        self._matcher = matcher
        self._last: tuple[str, list] | None = None  # (nl_query, concepts)

    def _concepts(self, nl_query: str) -> list:
        # /search calls match_concepts, get_query_features and convert with
        # the same query; run the matcher once and reuse the result.
        # Read the cache once into a local, so a concurrent request replacing
        # self._last can't hand us another query's concepts.
        last = self._last
        if last is not None and last[0] == nl_query:
            return last[1]
        concepts = self._matcher.match(nl_query)
        self._last = (nl_query, concepts)
        return concepts

    def match_concepts(self, nl_query: str) -> list:
        return [
            {"label": c["label"], "uri": c["uri"], "feature": c["feature"]}
            for c in self._concepts(nl_query)
            if c["label"] is not None
        ]

    def get_query_features(self, nl_query: str) -> dict:
        features: dict[str, list[str]] = {}
        for c in self._concepts(nl_query):
            if c["label"] is not None:
                features.setdefault(c["feature"], []).append(c["label"])
        return features

    def convert(self, nl_query: str, context: dict) -> str:
        concepts = self._concepts(nl_query)
        if not concepts:
            raise ValueError(
                f"No concepts matched {nl_query!r}. "
                "Try different or simpler keywords."
            )

        # Group URIs by feature (type | material | place).
        # Unlabelled URIs (ancestors outside the vocabulary) are still queried.
        by_feature: dict[str, list[str]] = {}
        for c in concepts:
            by_feature.setdefault(c["feature"], []).append(c["uri"])

        # One required block per dimension → AND semantics across dimensions.
        parts = []
        for feature, uris in by_feature.items():
            pattern = _REQUIRED_PATTERNS.get(feature)
            if pattern:
                parts.append(pattern.format(uri_values=" ".join(f"<{u}>" for u in uris)))

        required_pattern = "\n      ".join(parts)
        template = context.get("sparql_template") or _SELECT_TEMPLATE
        return template.format(required_pattern=required_pattern)

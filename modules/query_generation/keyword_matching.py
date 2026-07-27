import csv
import pathlib
import re

from modules.base import QueryGeneration

# Outer template: fetches all semantic features for every matched artifact.
# {required_pattern} is filled by convert() with one or more anchor blocks
# (each block pins a VALUES clause to the searched feature dimension).
_SELECT_TEMPLATE = """
    PREFIX la:   <https://linked.art/ns/terms/>
    PREFIX crm:  <http://www.cidoc-crm.org/cidoc-crm/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dct:  <http://purl.org/dc/terms/>
    SELECT
      ?artifact
      ?artifacttitle
      ?url_photo
      (GROUP_CONCAT(DISTINCT ?typeLabelRaw; separator=" | ") AS ?typeLabel)
      (GROUP_CONCAT(DISTINCT ?materiaalRaw; separator=" | ") AS ?materiaal)
      (GROUP_CONCAT(DISTINCT ?herkomstRaw;  separator=" | ") AS ?herkomst)
    WHERE {{
      {required_pattern}

      OPTIONAL {{
        ?artifact crm:P2_has_type ?typeNode .
        ?typeNode skos:altLabel ?typeLabelRaw .
      }}
      OPTIONAL {{
        ?artifact crm:P45_consists_of ?matNode .
        ?matNode skos:altLabel ?materiaalRaw .
      }}
      OPTIONAL {{
        ?artifact crm:P108i_was_produced_by ?prodEvent .
        ?prodEvent crm:P7_took_place_at ?placeNode .
        ?placeNode skos:prefLabel ?herkomstRaw .
      }}
      OPTIONAL {{ ?artifact dct:title ?artifacttitle . }}
      OPTIONAL {{
        ?artifact a crm:E22_Human-Made_Object .
        ?artifact crm:P65_shows_visual_item ?vi .
        ?vi la:digitally_shown_by ?o2 .
        ?o2 la:access_point ?url_photo .
      }}
      FILTER(BOUND(?url_photo))
    }}
    GROUP BY ?artifact ?artifacttitle ?url_photo
    LIMIT 100
"""

# Per-feature-category required anchor pattern.
# ?prodReq is distinct from the ?prodEvent in the OPTIONAL retrieval block.
_REQUIRED_PATTERNS = {
    "typeLabel": (
        "VALUES ?chosenType {{ {uri_values} }}\n"
        "      ?chosenType skos:narrower* ?specificType .\n"
        "      ?artifact crm:P2_has_type ?specificType ."
    ),
    "materiaal": (
        "VALUES ?chosenMat {{ {uri_values} }}\n"
        "      ?chosenMat skos:narrower* ?specificMat .\n"
        "      ?artifact crm:P45_consists_of ?specificMat ."
    ),
    "herkomst": (
        "VALUES ?chosenHerkomst {{ {uri_values} }}\n"
        "      ?chosenHerkomst skos:narrower* ?specificHerkomst .\n"
        "      ?artifact crm:P108i_was_produced_by ?prodReq .\n"
        "      ?prodReq crm:P7_took_place_at ?specificHerkomst ."
    ),
}

_FILLER_WORDS = {"and", "the", "a", "an", "of", "in", "to", "with", "for", "from"}

_ANCESTOR_QUERY = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?label WHERE {{
  <{uri}> skos:broader* ?ancestor .
  ?ancestor skos:prefLabel ?label .
}}
"""

# Top-level thesaurus category label → feature key used in query feature dicts.
# Match is case-insensitive substring so minor label variations don't break it.
_CATEGORY_FEATURE_MAP = {
    "objecttrefwoord":       "typeLabel",
    "culturele herkomst":    "herkomst",
    "geografische herkomst": "herkomst",
    "materiaal en techniek": "materiaal",
}


def _load_thesaurus(*paths: str) -> dict:
    """
    Load thesaurus CSV files into a dict keyed by concept URI.
    Expected columns: concept, prefLabel, broad, altLabels (pipe-separated).
    Missing files are silently skipped.
    """
    result = {}
    for path in paths:
        p = pathlib.Path(path)
        if not p.exists():
            continue
        with open(p, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                uri = row["concept"]
                labels = [row["prefLabel"]]
                for alt in row.get("altLabels", "").split(" | "):
                    if alt.strip():
                        labels.append(alt.strip())
                result[uri] = {"parent": row.get("broad", ""), "labels": labels}
    return result


def _get_coherent_superconcept(concepts: list, thesaurus: dict) -> set:
    """
    Walk the broader-concept tree upward until a single ancestor concept covers
    multiple of the matched concepts. Returns the shared ancestor(s), or the
    original set if no convergence is found.
    """
    frontier = set(concepts)
    while len(frontier) > 1:
        parents: dict[str, int] = {}
        for concept in frontier:
            parent = thesaurus.get(concept, {}).get("parent", "")
            if parent:
                parents[parent] = parents.get(parent, 0) + 1
        shared = {uri for uri, count in parents.items() if count > 1}
        if not shared:
            break
        frontier = shared
    return frontier


def _prompt_to_uris(prompt: str, thesaurus: dict) -> set:
    """
    Split the prompt into words, skip filler words, match each against thesaurus
    labels, and return a set of concept URIs — collapsed to coherent superconcepts
    where possible.
    """
    words = [w.lower() for w in re.split(r'\s+', prompt.strip()) if w]
    words = [w for w in words if w not in _FILLER_WORDS]

    overall: set = set()
    for word in words:
        matches = [
            uri for uri, entry in thesaurus.items()
            if any(word in label.lower() for label in entry["labels"])
        ]
        if matches:
            coherent = _get_coherent_superconcept(matches, thesaurus)
            overall.update(coherent if coherent else matches)
    return overall

def word_to_lemma(word):
    # todo
    return word

class KeywordMatching(QueryGeneration):
    """
    QueryGeneration that matches free-text keywords against a thesaurus and
    generates a SPARQL query covering all matched concept URIs in one go.

    Configure via .env:
        QUERY_GENERATION=keyword_matching
        KEYWORD_MATCHING_THESAURUS_FILES=path/to/t1.csv,path/to/t2.csv,...

    The thesaurus CSV format is graph-agnostic: any collection can supply its
    own files as long as they have the expected columns.
    context["sparql_template"] (optional) overrides the built-in template at
    call time; it must use {uri_values} as the placeholder.
    """

    def __init__(self, thesaurus_files: list = None, sparql_endpoint_url: str = ""):
        self._thesaurus = _load_thesaurus(*(thesaurus_files or []))
        self._endpoint_url = sparql_endpoint_url
        self._type_cache: dict[str, str] = {}

    def match_concepts(self, nl_query: str) -> list:
        uris = _prompt_to_uris(nl_query, self._thesaurus)
        return [
            {"label": self._thesaurus[uri]["labels"][0], "uri": uri}
            for uri in uris
            if uri in self._thesaurus
        ]

    def _concept_feature_key(self, uri: str) -> str:
        """Walk skos:broader* to find which top-level category owns this concept."""
        if uri in self._type_cache:
            return self._type_cache[uri]
        key = "typeLabel"  # fallback when endpoint not configured or query fails
        if self._endpoint_url:
            try:
                from SPARQLWrapper import SPARQLWrapper, JSON
                sparql = SPARQLWrapper(self._endpoint_url)
                sparql.setQuery(_ANCESTOR_QUERY.format(uri=uri))
                sparql.setReturnFormat(JSON)
                results = sparql.query().convert()
                for binding in results["results"]["bindings"]:
                    label_lower = binding["label"]["value"].lower()
                    for cat_label, feature_key in _CATEGORY_FEATURE_MAP.items():
                        if cat_label in label_lower:
                            key = feature_key
                            break
                    else:
                        continue
                    break
            except Exception:
                pass
        self._type_cache[uri] = key
        return key

    def get_query_features(self, nl_query: str) -> dict:
        uris = _prompt_to_uris(nl_query, self._thesaurus)
        features: dict[str, list[str]] = {}
        for uri in uris:
            if uri not in self._thesaurus:
                continue
            label = self._thesaurus[uri]["labels"][0]
            key = self._concept_feature_key(uri)
            features.setdefault(key, []).append(label)
        return {k: " | ".join(v) for k, v in features.items()}

    def convert(self, nl_query: str, context: dict) -> str:
        uris = _prompt_to_uris(nl_query, self._thesaurus)
        if not uris:
            raise ValueError(
                f"No thesaurus concepts matched {nl_query!r}. "
                "Try different or simpler keywords."
            )

        # Group URIs by their anchor dimension; culturele + geografische → herkomst.
        by_anchor: dict[str, list[str]] = {}
        for uri in uris:
            key = self._concept_feature_key(uri)
            by_anchor.setdefault(key, []).append(uri)

        # Build one required pattern block per anchor dimension (AND semantics).
        parts = []
        for anchor, anchor_uris in by_anchor.items():
            pattern = _REQUIRED_PATTERNS.get(anchor)
            if pattern:
                parts.append(pattern.format(uri_values=" ".join(f"<{u}>" for u in anchor_uris)))

        required_pattern = "\n      ".join(parts) if parts else ""
        template = context.get("sparql_template") or _SELECT_TEMPLATE
        return template.format(required_pattern=required_pattern)


if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    files = sys.argv[1:] or [
        "collections/wereld_museum/thesaurus1.csv",
        "collections/wereld_museum/thesaurus2.csv",
        "collections/wereld_museum/thesaurus3.csv",
    ]
    endpoint_url = os.getenv("SPARQL_ENDPOINT_URL", "")
    km = KeywordMatching(thesaurus_files=files, sparql_endpoint_url=endpoint_url)
    print(f"Loaded {len(km._thesaurus)} concepts from {len(files)} file(s).")
    if endpoint_url:
        print(f"Ancestor classification enabled ({endpoint_url})")
    else:
        print("Ancestor classification disabled (set SPARQL_ENDPOINT_URL in .env to enable)")
    print()
    try:
        while True:
            prompt = input("Query: ").strip()
            if not prompt:
                continue
            concepts = km.match_concepts(prompt)
            if concepts:
                features = km.get_query_features(prompt)
                print(f"  matched concepts ({len(concepts)}):")
                for c in concepts:
                    print(f"    {c['label']:40s}  {c['uri']}")
                print(f"  query features: {features}")
            else:
                print("  → no matches")
            print()
    except KeyboardInterrupt:
        pass

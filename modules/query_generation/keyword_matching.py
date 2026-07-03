import csv
import pathlib
import re

from modules.base import QueryGeneration

# Same ontology as CategorySelect but uses VALUES instead of BIND so multiple
# concept URIs can be passed in a single query.
_SPARQL_TEMPLATE = """
    PREFIX la:   <https://linked.art/ns/terms/>
    PREFIX crm:  <http://www.cidoc-crm.org/cidoc-crm/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dct:  <http://purl.org/dc/terms/>
    SELECT
      ?specificType
      (GROUP_CONCAT(DISTINCT ?typeLabelRaw; separator=" | ") AS ?typeLabel)
      ?artifact
      ?artifacttitle
      ?url_photo
    WHERE {{
      VALUES ?chosenCategory {{ {uri_values} }}
      ?chosenCategory skos:narrower* ?specificType .
      ?artifact crm:P2_has_type ?specificType .
      OPTIONAL {{ ?specificType skos:altLabel ?typeLabelRaw . }}
      OPTIONAL {{ ?artifact dct:title ?artifacttitle . }}
      OPTIONAL {{
        ?artifact a crm:E22_Human-Made_Object .
        ?artifact crm:P65_shows_visual_item ?vi .
        ?vi la:digitally_shown_by ?o2 .
        ?o2 la:access_point ?url_photo .
      }}
    }}
    GROUP BY ?specificType ?artifact ?artifacttitle ?url_photo
    LIMIT 100
"""

_FILLER_WORDS = {"and", "the", "a", "an", "of", "in", "to", "with", "for", "from"}


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

    def __init__(self, thesaurus_files: list = None):
        self._thesaurus = _load_thesaurus(*(thesaurus_files or []))

    def match_concepts(self, nl_query: str) -> list:
        uris = _prompt_to_uris(nl_query, self._thesaurus)
        return [
            {"label": self._thesaurus[uri]["labels"][0], "uri": uri}
            for uri in uris
            if uri in self._thesaurus
        ]

    def convert(self, nl_query: str, context: dict) -> str:
        uris = _prompt_to_uris(nl_query, self._thesaurus)
        if not uris:
            raise ValueError(
                f"No thesaurus concepts matched {nl_query!r}. "
                "Try different or simpler keywords."
            )
        template = context.get("sparql_template") or _SPARQL_TEMPLATE
        uri_values = " ".join(f"<{u}>" for u in uris)
        return template.format(uri_values=uri_values)


if __name__ == "__main__":
    import sys
    files = sys.argv[1:] or [
        "collections/wereld_museum/thesaurus1.csv",
        "collections/wereld_museum/thesaurus2.csv",
        "collections/wereld_museum/thesaurus3.csv",
    ]
    thesaurus = _load_thesaurus(*files)
    print(f"Loaded {len(thesaurus)} concepts from {len(files)} file(s).")
    try:
        while True:
            prompt = input("Query: ").strip()
            if not prompt:
                continue
            uris = _prompt_to_uris(prompt, thesaurus)
            if uris:
                print(f"  → {len(uris)} concept URI(s):")
                for u in uris:
                    print(f"    {u}")
            else:
                print("  → no matches")
    except KeyboardInterrupt:
        pass

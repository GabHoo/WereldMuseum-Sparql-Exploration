"""
ThesaurusMatcher — maps free-text keywords to thesaurus concept URIs.

Algorithm (unchanged from the original KeywordMatching):
  1. Split the prompt into words, drop filler words.
  2. For each word, collect every concept whose labels contain it (substring).
  3. Collapse each word's matches to a coherent superconcept by walking the
     `broad` column upward while two or more matches share a parent.
  4. Classify each resulting URI into a feature dimension by walking
     skos:broader* on the live endpoint to its top-level category.

Configure via .env:
    CONCEPT_MATCHER=thesaurus
    KEYWORD_MATCHING_THESAURUS_FILES=path/to/t1.csv,path/to/t2.csv,...
    SPARQL_ENDPOINT_URL=...   (for step 4; without it everything is type)

The thesaurus CSV format is graph-agnostic: columns
concept, prefLabel, broad, altLabels (altLabels separated by " | ").
"""
import csv
import pathlib
import re

from modules.base import ConceptMatcher

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
    "objecttrefwoord":       "type",
    "culturele herkomst":    "place",
    "geografische herkomst": "place",
    "materiaal en techniek": "material",
}

_DEFAULT_FEATURE = "type"  # when the endpoint is unset or the lookup fails


def _load_thesaurus(*paths: str) -> dict:
    """
    Load thesaurus CSV files into a dict keyed by concept URI.
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
    Generalise the concepts one word matched to their common ancestor(s).

    Each round replaces the set by the parents that two or more of its
    concepts share; a concept whose parent no other concept shares drops out.
    Stops when one concept is left or no parent is shared. E.g. "masker"
    matches many mask concepts, which collapse to "maskers (kleding)".
    Returns the original set if no two concepts share a parent.
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

    A word matches a label that *contains* it, not the other way round, and
    words are not lemmatised: "houten" (wooden) matches labels containing
    "houten", not the material "hout".
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
    # todo: not used yet. Meant to reduce a word to its base form before
    # matching (e.g. "houten" → "hout", "maskers" → "masker").
    return word


class ThesaurusMatcher(ConceptMatcher):
    """Keyword → thesaurus concepts, by label substring match (see the module docstring)."""

    def __init__(self, thesaurus_files: list = None, sparql_endpoint_url: str = ""):
        self._thesaurus = _load_thesaurus(*(thesaurus_files or []))
        self._endpoint_url = sparql_endpoint_url
        self._feature_cache: dict[str, str] = {}

    def match(self, nl_query: str) -> list:
        uris = sorted(_prompt_to_uris(nl_query, self._thesaurus))
        return [
            {
                "uri":     uri,
                "label":   self._thesaurus[uri]["labels"][0] if uri in self._thesaurus else None,
                "feature": self._feature_for(uri),
            }
            for uri in uris
        ]

    def _feature_for(self, uri: str) -> str:
        """Walk skos:broader* to find which top-level category owns this concept."""
        if uri in self._feature_cache:
            return self._feature_cache[uri]
        key = _DEFAULT_FEATURE
        if self._endpoint_url:
            try:
                from SPARQLWrapper import SPARQLWrapper, JSON
                sparql = SPARQLWrapper(self._endpoint_url)
                sparql.setQuery(_ANCESTOR_QUERY.format(uri=uri))
                sparql.setReturnFormat(JSON)
                results = sparql.query().convert()
                # One row per ancestor label. The first label that names a
                # top-level category decides; the for/else breaks out of both
                # loops on that first hit.
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
                pass   # endpoint unreachable: keep _DEFAULT_FEATURE
        self._feature_cache[uri] = key
        return key


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
    matcher = ThesaurusMatcher(thesaurus_files=files, sparql_endpoint_url=endpoint_url)
    print(f"Loaded {len(matcher._thesaurus)} concepts from {len(files)} file(s).")
    if endpoint_url:
        print(f"Feature classification enabled ({endpoint_url})")
    else:
        print("Feature classification disabled (set SPARQL_ENDPOINT_URL in .env to enable)")
    print()
    try:
        while True:
            prompt = input("Query: ").strip()
            if not prompt:
                continue
            concepts = matcher.match(prompt)
            if concepts:
                print(f"  matched concepts ({len(concepts)}):")
                for c in concepts:
                    print(f"    {str(c['label']):40s}  {c['feature']:10s}  {c['uri']}")
            else:
                print("  → no matches")
            print()
    except KeyboardInterrupt:
        pass

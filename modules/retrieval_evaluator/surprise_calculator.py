"""
ConceptSurpriseCalculator — how many items fall outside the searched concepts.

For each feature the query searched on (e.g. type = "maskers (kleding)"), the
searched concepts are expanded with everything below them in the thesaurus
(skos:narrower*), the same expansion the find query uses. Then, per item:

    deviates on f  ⇔  the item has at least one concept for f (<f>_uris),
                      and none of them is a searched concept or below one

    d(item)   = number of searched features the item deviates on
    surprise  = mean over the pool of d(item) / k      (k = number of searched features)

So surprise is 0 exactly when no item deviates, and grows with how many items
deviate and on how many features. An item with no value for a feature (not
recorded) does not count as deviating on it. Concepts are compared by URI, so
differences in label wording never count as surprise.

The expansion runs one SPARQL query per searched feature, cached per concept set.
"""
from modules.base import KnowledgeBase
from modules.retrieval_evaluator.base import SurpriseCalculator

_EXPAND_QUERY = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?d WHERE {{
  VALUES ?c {{ {uris} }}
  ?c skos:narrower* ?d .
}}
"""


class ConceptSurpriseCalculator(SurpriseCalculator):
    def __init__(self, knowledge_base: KnowledgeBase):
        self._kb = knowledge_base
        self._expanded: dict[frozenset, frozenset] = {}

    def compute(self, pool: list, query: dict) -> float:
        searched = {f: uris for f, uris in query.items() if uris}
        k = len(searched)
        if k == 0 or not pool:
            return 0.0

        covered = {f: self._expand(uris) for f, uris in searched.items()}
        total = 0.0
        for item in pool:
            props = item.get("properties", {})
            d = 0
            for f, allowed in covered.items():
                values = props.get(f"{f}_uris") or []
                if values and not any(v in allowed for v in values):
                    d += 1
            total += d / k
        return round(total / len(pool), 6)

    def deviating_features(self, item: dict, query: dict) -> list[str]:
        """Searched features on which one item falls outside the searched concepts."""
        props = item.get("properties", {})
        out = []
        for f, uris in query.items():
            values = props.get(f"{f}_uris") or []
            if uris and values and not any(v in self._expand(uris) for v in values):
                out.append(f)
        return out

    def _expand(self, uris: list[str]) -> frozenset:
        key = frozenset(uris)
        if key not in self._expanded:
            rows = self._kb.execute(_EXPAND_QUERY.format(uris=" ".join(f"<{u}>" for u in key)))
            self._expanded[key] = key | {r["d"] for r in rows if r.get("d")}
        return self._expanded[key]

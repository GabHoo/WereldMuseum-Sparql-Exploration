"""
Helpers shared by selection strategies.

    unseen(pool, history)             items not shown yet
    largest_remainder(weights, total) split slots proportionally, in whole numbers
    metadata_linearization(fields)    item → one text line from Artifact fields
"""
from typing import Callable

from modules.artifact import Artifact
from modules.base import SelectionHistory


def unseen(pool: list, history: SelectionHistory) -> list:
    """Pool items that have not been shown yet, in pool order."""
    seen = set(history.seen)
    return [item for item in pool if item["id"] not in seen]


def largest_remainder(weights: dict, total: int) -> dict:
    """
    Split `total` slots over keys in proportion to their weights, as whole
    numbers that add up to `total` (largest remainder method).
    """
    s = sum(weights.values())
    if total <= 0 or s <= 0:
        return {k: 0 for k in weights}
    exact = {k: total * w / s for k, w in weights.items()}
    out = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def metadata_linearization(fields: str = "title", sep: str = " · ") -> Callable[[dict], str]:
    """
    Return a function that turns a pool item into one line of text, e.g. for
    embedding: the given Artifact fields in order, joined with `sep`; list
    fields contribute all values joined with ", "; empty fields are skipped.

        linearize = metadata_linearization("title,type,material,place")
        linearize(item)  →  "HOUTEN MASKER · maskers (kleding) · hout, kleurstof · Ivoorkust"

    Field names are checked against Artifact here, so a typo fails at startup
    instead of silently producing empty text.
    """
    names = [f.strip() for f in fields.split(",") if f.strip()]
    unknown = set(names) - set(Artifact.field_names())
    if unknown:
        raise ValueError(f"Unknown Artifact field(s) {sorted(unknown)} in {fields!r}. "
                         f"Known: {Artifact.field_names()}")

    def linearize(item: dict) -> str:
        props = item.get("properties", {})
        parts = []
        for name in names:
            v = props.get(name)
            if isinstance(v, list):
                v = ", ".join(x.strip() for x in v if isinstance(x, str) and x.strip())
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        return sep.join(parts)

    linearize.fields = names
    return linearize

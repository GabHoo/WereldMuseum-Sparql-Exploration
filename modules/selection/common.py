"""
Small helpers shared by selection strategies.
"""
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

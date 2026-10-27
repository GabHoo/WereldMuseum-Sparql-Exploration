import random

from modules.base import SelectionHistory, SelectionStrategy


class RandomStrategy(SelectionStrategy):
    """
    Baseline selection: pick n unseen items at random.
    Ignores the curator's feedback; clustering and rerank are the strategies
    that use it. No embedding needed.
    """

    def select(self, pool: list, n: int, history: SelectionHistory) -> list:
        seen = set(history.seen)
        unseen = [item for item in pool if item["id"] not in seen]
        return random.sample(unseen, min(n, len(unseen)))

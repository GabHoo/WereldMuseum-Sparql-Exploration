import random

from modules.base import SelectionHistory, SelectionStrategy


class RandomStrategy(SelectionStrategy):
    """
    Baseline selection: pick n unseen items at random.
    Ignores accepted/rejected history — swap this for an active-learning
    implementation that uses history to re-score the pool.
    """

    def select(self, pool: list, n: int, history: SelectionHistory) -> list:
        seen = set(history.seen)
        unseen = [item for item in pool if item["id"] not in seen]
        return random.sample(unseen, min(n, len(unseen)))

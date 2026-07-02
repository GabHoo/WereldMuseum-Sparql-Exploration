from modules.base import SelectionHistory, SelectionStrategy


class MySelectionStrategy(SelectionStrategy):
    """
    Stub — copy this file, rename the class, implement select(), then:
      1. Add an entry to REGISTRY["selection"] in config.py
      2. Set SELECTION=<your-key> in .env
    """

    def select(self, pool: list, n: int, history: SelectionHistory) -> list:
        """
        pool    : all items from the SPARQL query, each a dict with:
                    id         (str)
                    title      (str)
                    score      (float, default 1.0 — you may mutate this)
                    properties (dict of raw SPARQL fields)
        n       : number of items to return
        history : SelectionHistory with:
                    accepted   list of item IDs the user said yes to
                    rejected   list of item IDs the user said no to
                    seen       list of all item IDs ever shown (never return these)

        Return n items from pool whose id is NOT in history.seen.
        You may mutate item["score"] — scores are sent back to the browser
        and included in the next call, so rankings accumulate across rounds.
        """
        raise NotImplementedError

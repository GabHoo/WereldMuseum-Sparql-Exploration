from modules.base import SelectionHistory, SelectionStrategy
from modules.selection.utils import unseen


class MySelectionStrategy(SelectionStrategy):
    """
    Stub — copy this file, rename the class, implement select(), then:
      1. Add an entry to REGISTRY["selection"] in config.py
      2. If it needs constructor args, add a branch in _build_selection()
      3. Set SELECTION=<your-key> in .env

    Reusable pieces:
      modules/selection/utils.py       unseen(), largest_remainder(), metadata_linearization()
      modules/selection/clustering.py  Clusterer / KMeansClusterer
      modules/embedding.py             TextEmbedder (shared embedding model)
    See title_cluster.py for a complete example.
    """

    def prepare(self, pool: list) -> None:
        """
        Optional. Called once per /search, before the first select().
        Do per-search work here (embedding, clustering, scoring) and store
        small JSON-serialisable results in item["selection"] — the pool goes
        to the browser and comes back with every /next call. Never store vectors.
        """

    def select(self, pool: list, n: int, history: SelectionHistory) -> list:
        """
        pool    : pool items {id, title, score, properties, selection?}
                  properties is the Artifact dict (modules/artifact.py)
        n       : number of items to return
        history : SelectionHistory with
                    accepted   ids the user kept so far in this search
                    rejected   ids shown but not kept
                    seen       every id shown so far (never return these)

        Return up to n items from unseen(pool, history).
        """
        candidates = unseen(pool, history)
        raise NotImplementedError

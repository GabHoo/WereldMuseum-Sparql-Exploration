"""
ClusteringStrategy — cluster the pool, show one of each, then steer towards
the clusters the user keeps from, while always keeping some exploration.

Works on the vectors of the Embedding module (EMBEDDING=metadata_text |
multimodal), in the EMBEDDING_VIEW space; config.py injects the service.

prepare(pool)   once per search
    1. Wait up to `wait` seconds for the pool's vectors (app.py has queued
       them). metadata_text is ready in about a second; multimodal takes ~30 s.
    2. If every item has a vector: cluster them (Clusterer), aiming for
       `n_clusters` clusters (default: twice the batch size, so clusters remain
       unshown after the first batch), and store on each item:
           item["selection"] = {"cluster": <k>, "cluster_label": <most common title>}
       Vectors are not stored on the items.
    Otherwise the pool stays unclustered: batches are random until a later
    select() finds every vector ready and clusters then. The annotations travel
    with the pool, so this happens once per search.

select(pool, n, history)   every batch
    r        = min(1, shuffle_step · empty_streak) — grows with every batch in a
               row in which nothing was kept; resets to 0 after a keep.
    steered  = round(steering_weight · (1 − r) · n) slots, only once the user
               has kept something: shared over the clusters of the kept items
               (history.accepted), in proportion to how many were kept from each.
    random   = round(r · remaining) slots: drawn uniformly from all unseen items,
               ignoring clusters.
    explore  = the rest: one item per cluster, least-shown clusters first
               (so never-shown clusters come first), cycling until full.
    Within a cluster, items are picked at random among the unseen ones.
    Batch order: steered, then exploration, then random.

steering_weight 0 → never steer; 1 → only the kept clusters (exploration still
fills in when those clusters run out). shuffle_step 0 → empty batches change
nothing; 0.25 → after 4 empty batches in a row everything is random.
"""
import logging
import random
from collections import Counter, defaultdict

from modules.base import SelectionHistory, SelectionStrategy
from modules.selection.clustering import Clusterer
from modules.selection.utils import empty_streak, largest_remainder, unseen

log = logging.getLogger(__name__)

_UNTITLED = "(untitled)"


class ClusteringStrategy(SelectionStrategy):
    def __init__(
        self,
        embedding,
        clusterer: Clusterer,
        view: str = "text",
        n_clusters: int = 18,
        steering_weight: float = 0.6,
        shuffle_step: float = 0.25,
        wait: float = 3.0,
        seed: int | None = None,
    ):
        if not 0.0 <= steering_weight <= 1.0:
            raise ValueError(f"steering_weight must be in [0, 1], got {steering_weight}")
        if not 0.0 <= shuffle_step <= 1.0:
            raise ValueError(f"shuffle_step must be in [0, 1], got {shuffle_step}")
        if embedding is None:
            log.warning("SELECTION=clustering needs EMBEDDING; selecting at random")
        elif view not in embedding.encoder.views:
            raise ValueError(f"EMBEDDING_VIEW={view!r} is not computed by this embedding "
                             f"{list(embedding.encoder.views)}")
        self._embedding = embedding
        self._clusterer = clusterer
        self._view      = view
        self._k         = n_clusters
        self._w         = steering_weight
        self._step      = shuffle_step
        self._wait      = wait
        self._rng       = random.Random(seed)
        self.last_debug: dict = {}   # what the last select() did, for the run log

    # ── Per search ───────────────────────────────────────────────────────────

    def prepare(self, pool: list) -> None:
        if not pool or self._embedding is None:
            return
        self._embedding.wait([item["id"] for item in pool], self._wait)
        self._cluster(pool)

    def _cluster(self, pool: list) -> bool:
        """Annotate the items with clusters if every one has a vector; else False."""
        ids, vectors = self._embedding.matrix([item["id"] for item in pool], self._view)
        if len(ids) < len(pool):
            return False
        labels = self._clusterer.fit(vectors, self._k)

        titles_by_label: dict[int, list[str]] = defaultdict(list)
        for item, label in zip(pool, labels):
            titles_by_label[label].append(item.get("title") or _UNTITLED)
        names = {label: Counter(ts).most_common(1)[0][0] for label, ts in titles_by_label.items()}

        for item, label in zip(pool, labels):
            item["selection"] = {
                **item.get("selection", {}),
                "cluster":       int(label),
                "cluster_label": names[label],
            }
        return True

    # ── Per batch ────────────────────────────────────────────────────────────

    def select(self, pool: list, n: int, history: SelectionHistory) -> list:
        clustered = all("cluster" in item.get("selection", {}) for item in pool)
        if not clustered and not (self._embedding and self._cluster(pool)):
            # Vectors not ready yet (or no embedding): random, like a first batch.
            self.last_debug = {"mode": "random",
                               "reason": "pool not encoded yet" if self._embedding else "no embedding"}
            cands = unseen(pool, history)
            return self._rng.sample(cands, min(n, len(cands)))
        self.last_debug = {"mode": "clustering", "clusters": len({i["selection"]["cluster"] for i in pool})}

        cluster_of = {item["id"]: item["selection"]["cluster"] for item in pool}
        by_cluster: dict[str, list] = defaultdict(list)
        for item in unseen(pool, history):
            by_cluster[cluster_of[item["id"]]].append(item)
        for items in by_cluster.values():
            self._rng.shuffle(items)

        picks: list = []
        # Randomness grows with every batch in a row without a keep.
        r = min(1.0, self._step * empty_streak(history, n))

        # 1. Steer: more from the clusters the user kept items from.
        kept = Counter(cluster_of[i] for i in history.accepted if i in cluster_of)
        kept = {c: m for c, m in kept.items() if by_cluster.get(c)}
        if kept:
            for c, m in largest_remainder(kept, round(self._w * (1 - r) * n)).items():
                picks += [by_cluster[c].pop() for _ in range(min(m, len(by_cluster[c])))]

        # 2. Reserve pure-random slots (filled last, from whatever is left).
        n_random = round(r * (n - len(picks)))

        # 3. Explore: one per cluster, least-shown clusters first, cycling.
        shown = Counter(cluster_of[i] for i in history.seen if i in cluster_of)
        while len(picks) < n - n_random and any(by_cluster.values()):
            order = sorted(
                (c for c, items in by_cluster.items() if items),
                key=lambda c: (shown[c], self._rng.random()),
            )
            for c in order:
                if len(picks) >= n - n_random:
                    break
                picks.append(by_cluster[c].pop())
                shown[c] += 1

        # 4. Random: uniform over all remaining unseen items, ignoring clusters.
        rest = [item for items in by_cluster.values() for item in items]
        picks += self._rng.sample(rest, min(n - len(picks), len(rest)))
        return picks

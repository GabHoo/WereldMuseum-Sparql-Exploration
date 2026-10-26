"""
MetadataTextualClusteringStrategy — cluster the pool, show one of each, then steer
towards the clusters the user keeps from, while always keeping some exploration.

prepare(pool)   once per search
    1. Linearize each item's metadata into one text (metadata_linearization),
       from the configured Artifact fields (default: the title).
    2. Embed the texts (shared TextEmbedder) and cluster them (Clusterer),
       aiming for `n_clusters` clusters (default: twice the batch size, so
       clusters remain unshown after the first batch).
    3. Store on each item:  item["selection"] = {"cluster": "<search-id>:<k>",
                                                "cluster_label": <most common text>}
       The search id prefix keeps clusters of different searches apart when the
       browser merges pools (multi-category search). Vectors are not stored.

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
import hashlib
import random
from collections import Counter, defaultdict
from typing import Callable

from modules.base import SelectionHistory, SelectionStrategy
from modules.embedding import TextEmbedder
from modules.selection.clustering import Clusterer
from modules.selection.utils import empty_streak, largest_remainder, metadata_linearization, unseen

_UNTITLED = "(untitled)"


class MetadataTextualClusteringStrategy(SelectionStrategy):
    def __init__(
        self,
        embedder: TextEmbedder,
        clusterer: Clusterer,
        linearize: Callable[[dict], str] | None = None,
        n_clusters: int = 18,
        steering_weight: float = 0.6,
        shuffle_step: float = 0.25,
        seed: int | None = None,
    ):
        if not 0.0 <= steering_weight <= 1.0:
            raise ValueError(f"steering_weight must be in [0, 1], got {steering_weight}")
        if not 0.0 <= shuffle_step <= 1.0:
            raise ValueError(f"shuffle_step must be in [0, 1], got {shuffle_step}")
        self._embedder  = embedder
        self._clusterer = clusterer
        self._linearize = linearize or metadata_linearization("title")
        self._k         = n_clusters
        self._w         = steering_weight
        self._step      = shuffle_step
        self._rng       = random.Random(seed)

    # ── Per search ───────────────────────────────────────────────────────────

    def prepare(self, pool: list) -> None:
        if not pool:
            return
        texts  = [self._linearize(item) or _UNTITLED for item in pool]
        labels = self._clusterer.fit(self._embedder.embed(texts), self._k)

        search_id = hashlib.sha1("".join(sorted(i["id"] for i in pool)).encode()).hexdigest()[:6]
        texts_by_label: dict[int, list[str]] = defaultdict(list)
        for text, label in zip(texts, labels):
            texts_by_label[label].append(text)
        names = {label: Counter(ts).most_common(1)[0][0] for label, ts in texts_by_label.items()}

        for item, label in zip(pool, labels):
            item["selection"] = {
                **item.get("selection", {}),
                "cluster":       f"{search_id}:{label}",
                "cluster_label": names[label],
            }

    # ── Per batch ────────────────────────────────────────────────────────────

    def select(self, pool: list, n: int, history: SelectionHistory) -> list:
        if any("cluster" not in item.get("selection", {}) for item in pool):
            self.prepare(pool)  # e.g. a pool built before this strategy was enabled

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

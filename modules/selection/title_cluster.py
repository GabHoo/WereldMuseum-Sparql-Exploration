"""
TitleClusterStrategy — cluster the pool, show one of each, then steer
towards the clusters the user keeps from, while always keeping some exploration.

prepare(pool)   once per search
    1. Linearize each item's metadata into one text (metadata_linearization),
       from the configured Artifact fields (default: the title).
    2. Embed the texts (shared TextEmbedder) and cluster them (Clusterer),
       aiming for `n_clusters` clusters (default: the batch size).
    3. Store on each item:  item["selection"] = {"cluster": "<search-id>:<k>",
                                                "cluster_label": <most common text>}
       The search id prefix keeps clusters of different searches apart when the
       browser merges pools (multi-category search). Vectors are not stored.

select(pool, n, history)   every batch
    steered  = round(steering_weight · n) slots, only once the user has kept
               something: shared over the clusters of the kept items
               (history.accepted), in proportion to how many were kept from each.
    explore  = the remaining slots: one item per cluster, least-shown clusters
               first, cycling until full — so the first batch shows one item
               from every cluster, and later batches keep covering the pool.
    Within a cluster, items are picked at random among the unseen ones.
    Steered items come first in the batch, exploration items after them.

steering_weight 0 → pure exploration; 1 → only the kept clusters (exploration
still fills in when those clusters run out).
"""
import hashlib
import random
from collections import Counter, defaultdict
from typing import Callable

from modules.base import SelectionHistory, SelectionStrategy
from modules.embedding import TextEmbedder
from modules.selection.clustering import Clusterer
from modules.selection.utils import largest_remainder, metadata_linearization, unseen

_UNTITLED = "(untitled)"


class TitleClusterStrategy(SelectionStrategy):
    def __init__(
        self,
        embedder: TextEmbedder,
        clusterer: Clusterer,
        linearize: Callable[[dict], str] | None = None,
        n_clusters: int = 9,
        steering_weight: float = 0.6,
        seed: int | None = None,
    ):
        if not 0.0 <= steering_weight <= 1.0:
            raise ValueError(f"steering_weight must be in [0, 1], got {steering_weight}")
        self._embedder  = embedder
        self._clusterer = clusterer
        self._linearize = linearize or metadata_linearization("title")
        self._k         = n_clusters
        self._w         = steering_weight
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

        # 1. Steer: more from the clusters the user kept items from.
        kept = Counter(cluster_of[i] for i in history.accepted if i in cluster_of)
        kept = {c: m for c, m in kept.items() if by_cluster.get(c)}
        if kept:
            for c, m in largest_remainder(kept, round(self._w * n)).items():
                picks += [by_cluster[c].pop() for _ in range(min(m, len(by_cluster[c])))]

        # 2. Explore: one per cluster, least-shown clusters first, cycling.
        shown = Counter(cluster_of[i] for i in history.seen if i in cluster_of)
        while len(picks) < n and any(by_cluster.values()):
            order = sorted(
                (c for c, items in by_cluster.items() if items),
                key=lambda c: (shown[c], self._rng.random()),
            )
            for c in order:
                if len(picks) >= n:
                    break
                picks.append(by_cluster[c].pop())
                shown[c] += 1

        return picks

"""
EmbeddingService — object vectors for embedding-based selection, computed in
the background and cached per object id.

    ensure(items, priority)   queue these pool items for encoding (cached ones are skipped)
    preload(items)            the same at the lowest priority (demo warm-up)
    vector(id, view)          one item's vector, or None while it is not encoded
    matrix(ids, view)         (the ids that have a vector, array len × d), in the given order
    wait(ids, timeout)        block until these ids are encoded (or no longer queued)
    status()                  {enabled, views, cached, queued}

The server keeps no per-curator state here. Which pool is current, what was
kept and what was marked not relevant come with every request (/search,
/next), so several tabs or curators can work side by side; the cache only
holds vectors, which are the same for everyone.

Encoding runs on one daemon worker thread, batch by batch. Requests are served
by priority — FEEDBACK (kept items and negatives), DISPLAYED (the batch on
screen, feedback on it comes next), POOL (the rest of the results), PRELOAD —
and within a priority the newest request first, so a new search is encoded
before the leftovers of an older one, and a pool sent again with /next moves
to the front again. Images come from ImageCache: the same on-disk thumbnails
the UI shows, with placeholders already filtered. A batch that fails is not
cached, so the next request queues it again.

Usage (CLI, nearest neighbours within an export, for the configured EMBEDDING):
    python -m modules.embedding.service path/to/export.json [--k 3]
"""
import heapq
import itertools
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from modules.embedding.base import ObjectEncoder

log = logging.getLogger(__name__)

FEEDBACK  = 0   # kept items and negatives: the strategy needs them for this batch
DISPLAYED = 1   # the batch on screen: feedback on it comes next
POOL      = 2   # the rest of the search results
PRELOAD   = 3   # cache warm-up: only when nothing else is waiting
_IMAGE_FETCHERS = 4   # parallel thumbnail downloads per batch, as the browser preload


class EmbeddingService:
    def __init__(self, encoder: ObjectEncoder, image_cache, batch_size: int = 8, cache_size: int = 5000):
        self.encoder = encoder
        self._images = image_cache
        self._batch_size = batch_size
        self._cache_size = cache_size
        self._cache: OrderedDict[str, dict] = OrderedDict()   # id → {view: vec}, LRU
        self._heap: list = []                  # (priority, -seq, id); stale entries are skipped
        self._queued: dict[str, tuple] = {}    # id → (priority, seq, item) of its live heap entry
        self._busy: set = set()                # ids being encoded right now
        self._seq = itertools.count()
        self._cond = threading.Condition()     # guards everything above
        self._fetchers = ThreadPoolExecutor(_IMAGE_FETCHERS, thread_name_prefix="embedding-images")
        threading.Thread(target=self._worker, name="embedding-worker", daemon=True).start()

    # ── Requests ──────────────────────────────────────────────────────────────

    def ensure(self, items: list, priority: int = POOL) -> int:
        """
        Queue the items that have no vector yet; returns how many were queued.
        Items already queued more urgently stay where they are; at the same or
        a lower urgency they move to the front of their priority.
        """
        queued = 0
        with self._cond:
            for item in items:
                item_id = item["id"]
                if item_id in self._cache:
                    self._cache.move_to_end(item_id)
                    continue
                if item_id in self._busy:
                    continue
                current = self._queued.get(item_id)
                if current is not None and current[0] < priority:
                    continue
                seq = next(self._seq)
                self._queued[item_id] = (priority, seq, item)
                heapq.heappush(self._heap, (priority, -seq, item_id))
                queued += 1
            if queued:
                self._cond.notify_all()
        return queued

    def preload(self, items: list) -> int:
        """Queue items for encoding into the cache after all other work."""
        return self.ensure(items, PRELOAD)

    # ── Reading vectors ───────────────────────────────────────────────────────

    def vector(self, item_id: str, view: str) -> np.ndarray | None:
        """An item's vector for view; None if it is not encoded (yet)."""
        with self._cond:
            vecs = self._cache.get(item_id)
            return vecs.get(view) if vecs else None

    def matrix(self, ids: list, view: str) -> tuple:
        """(the ids that have a vector for view, array n × d), in the order of ids."""
        with self._cond:
            pairs = [(i, self._cache[i][view]) for i in ids
                     if i in self._cache and view in self._cache[i]]
        if not pairs:
            return [], np.zeros((0, 0), dtype=np.float32)
        found, vecs = zip(*pairs)
        return list(found), np.stack(vecs)

    def wait(self, ids: list, timeout: float | None = 30.0) -> bool:
        """Block until none of ids is queued or being encoded; False on timeout."""
        ids = set(ids)
        with self._cond:
            return self._cond.wait_for(
                lambda: not any(i in self._queued or i in self._busy for i in ids), timeout)

    def status(self) -> dict:
        with self._cond:
            return {
                "enabled": True,
                "views":   list(self.encoder.views),
                "cached":  len(self._cache),
                "queued":  len(self._queued) + len(self._busy),
            }

    # ── Worker ────────────────────────────────────────────────────────────────

    def _next_batch(self) -> list:
        """Pop up to batch_size items, most urgent and newest first (holding the lock)."""
        batch = []
        while self._heap and len(batch) < self._batch_size:
            _, neg_seq, item_id = heapq.heappop(self._heap)
            current = self._queued.get(item_id)
            if current is None or current[1] != -neg_seq:
                continue   # stale: re-queued since, or already encoded
            del self._queued[item_id]
            self._busy.add(item_id)
            batch.append(current[2])
        return batch

    def _worker(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._heap)
                batch = self._next_batch()
            if not batch:
                continue
            try:
                vectors = self._encode(batch)
            except Exception:
                log.exception("Embedding batch failed (%d items)", len(batch))
                vectors = None
            with self._cond:
                for i, item in enumerate(batch):
                    self._busy.discard(item["id"])
                    if vectors is not None:
                        self._cache[item["id"]] = vectors[i]
                        self._cache.move_to_end(item["id"])
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                self._cond.notify_all()

    def _encode(self, batch: list) -> list:
        paths = (list(self._fetchers.map(self._image_path, batch))
                 if self.encoder.needs_images else [None] * len(batch))
        return self.encoder.encode([{
            "id":         item["id"],
            "text":       self.encoder.object_text(item.get("properties", {})),
            "image_path": path,
        } for item, path in zip(batch, paths)])

    def _image_path(self, item: dict) -> str | None:
        """Local thumbnail of the item's first image that is not a placeholder."""
        for url in item.get("properties", {}).get("images", []):
            if not self._images.is_allowed(url):
                continue
            path, _ = self._images.thumbnail(url)
            if path is not None:
                return str(path)
        return None


if __name__ == "__main__":
    import json
    import os
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m modules.embedding.service <export_or_pool.json> [--k 3]")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()
    import config

    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    pool = data if isinstance(data, list) else data.get("items", [])
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 3
    key = os.getenv("EMBEDDING", "").strip() or "metadata_text"

    print(f"Pool: {len(pool)} items · embedding: {key}")
    print("Loading model …")
    service = config.build_embedding(key)
    print("Encoding …")
    service.ensure(pool)
    service.wait([item["id"] for item in pool], timeout=None)

    titles = {item["id"]: item.get("title") or item["id"] for item in pool}
    for view in service.encoder.views:
        ids, mat = service.matrix([item["id"] for item in pool], view)
        print(f"\n── {view} ({len(ids)} vectors) ──")
        if len(ids) < 2:
            continue
        sims = mat @ mat.T
        np.fill_diagonal(sims, -np.inf)
        for i, item_id in enumerate(ids):
            nn = np.argsort(-sims[i])[:k]
            neighbours = ", ".join(f"{titles[ids[j]][:40]} ({sims[i, j]:.2f})" for j in nn)
            print(f"{titles[item_id][:50]:50}  →  {neighbours}")

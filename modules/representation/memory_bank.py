"""
Memory banks of object representations, filled in the background.

Two banks per session:
    search bank    — the pool of the current query; replaced on every new query
    selection bank — the objects the curator kept; persists across queries

RepresentationService is what app.py (and future consumers such as a
re-ranking SelectionStrategy) talks to:

    on_search(pool, search_id)   new query → clear + refill the search bank.
                                 Calls with the same search_id add to it instead.
    prioritize(ids)              encode these search-bank items (the displayed
                                 batch) before the rest of the pool.
    sync_selection(items)        mirror the browser's full "selected" list.
    preload(items)               encode items into the vector cache only (no
                                 bank), after all other work (demo warm-up).
    search_bank / selection_bank MemoryBank: matrix(view), centroid(view), get(id), …
    vector(id, view)             one item's vector (from a bank or the cache)
    similarity(view, ids, bank)  cosine similarity of items to a bank
    wait(bank, timeout)          block until a bank is fully encoded
    status()                     progress of both banks

Encoding runs on one daemon worker thread, batch by batch. Selection batches
go before search batches; batches of an older query are skipped. Vectors are
cached per object id (LRU), so a kept item — normally already encoded as a
search result — is never encoded twice. Images come from ImageCache: the same
on-disk thumbnails the UI shows, with placeholders already filtered.

Only one session ("default") is used for now: one curator per server process.
"""
import itertools
import logging
import queue
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from modules.representation.base import ObjectEncoder

log = logging.getLogger(__name__)

_PRIORITY_SELECTION = 0
_PRIORITY_DISPLAYED = 1   # the batch on screen: feedback needs it encoded first
_PRIORITY_SEARCH = 2
_PRIORITY_PRELOAD = 3     # cache warm-up: only when nothing else is waiting
_IMAGE_FETCHERS = 4   # parallel thumbnail downloads per batch, as the browser preload


class MemoryBank:
    """
    {id: {"item": pool item, "vectors": {view: vec} | None}} — None while pending.
    Thread-safe; insertion order is kept, so matrix() rows follow the pool order.
    """

    def __init__(self):
        self._entries: dict[str, dict] = {}
        self._lock = threading.Lock()

    def add(self, items: list, vectors: dict | None = None) -> None:
        """Add pool items (vectors pending unless given in vectors={id: {view: vec}})."""
        vectors = vectors or {}
        with self._lock:
            for item in items:
                if item["id"] not in self._entries:
                    self._entries[item["id"]] = {"item": item, "vectors": vectors.get(item["id"])}

    def set_vectors(self, item_id: str, vectors: dict) -> None:
        """Fill in an item's vectors; ignored if the item is no longer in the bank."""
        with self._lock:
            if item_id in self._entries:
                self._entries[item_id]["vectors"] = vectors

    def remove(self, ids) -> None:
        with self._lock:
            for item_id in ids:
                self._entries.pop(item_id, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def ids(self) -> list:
        with self._lock:
            return list(self._entries)

    def get(self, item_id: str) -> dict | None:
        with self._lock:
            return self._entries.get(item_id)

    def pending(self) -> list:
        """Pool items whose vectors are not computed yet."""
        with self._lock:
            return [e["item"] for e in self._entries.values() if e["vectors"] is None]

    def matrix(self, view: str) -> tuple:
        """(ids, array n × d) of the items that have a vector for view."""
        with self._lock:
            pairs = [(i, e["vectors"][view]) for i, e in self._entries.items()
                     if e["vectors"] and view in e["vectors"]]
        if not pairs:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids, vecs = zip(*pairs)
        return list(ids), np.stack(vecs)

    def centroid(self, view: str) -> np.ndarray | None:
        """Re-normalised mean vector of the bank, or None if it has no vectors."""
        _, mat = self.matrix(view)
        if not len(mat):
            return None
        mean = mat.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean

    def status(self) -> dict:
        with self._lock:
            total = len(self._entries)
            done = sum(e["vectors"] is not None for e in self._entries.values())
        return {"total": total, "done": done, "ready": done == total}


class _Session:
    def __init__(self):
        self.search_bank = MemoryBank()
        self.selection_bank = MemoryBank()
        self.search_id = None
        self.generation = 0   # bumped on every new query; older search batches are skipped
        self.lock = threading.Lock()


class RepresentationService:
    def __init__(self, encoder: ObjectEncoder, image_cache, batch_size: int = 8, cache_size: int = 5000):
        self.encoder = encoder
        self._images = image_cache
        self._batch_size = batch_size
        self._cache_size = cache_size
        self._cache: OrderedDict[str, dict] = OrderedDict()   # id → {view: vec}
        self._cache_lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()   # FIFO order within one priority
        self._fetchers = ThreadPoolExecutor(_IMAGE_FETCHERS, thread_name_prefix="representation-images")
        self._idle = threading.Condition()
        self._busy = 0                  # batches queued or being encoded
        threading.Thread(target=self._worker, name="representation-worker", daemon=True).start()

    # ── Banks ─────────────────────────────────────────────────────────────────

    def session(self, session: str = "default") -> _Session:
        with self._sessions_lock:
            return self._sessions.setdefault(session, _Session())

    @property
    def search_bank(self) -> MemoryBank:
        return self.session().search_bank

    @property
    def selection_bank(self) -> MemoryBank:
        return self.session().selection_bank

    def on_search(self, pool: list, search_id: str | None = None, session: str = "default") -> None:
        """
        Register a new search result pool. A different (or missing) search_id
        starts a new query: the search bank is cleared and pending work for the
        old query is dropped. The same search_id adds to the current bank.
        """
        s = self.session(session)
        with s.lock:
            if search_id is None or search_id != s.search_id:
                s.search_id = search_id
                s.generation += 1
                s.search_bank.clear()
            generation = s.generation
            s.search_bank.add(pool, self._cached(item["id"] for item in pool))
        self._enqueue(_PRIORITY_SEARCH, session, "search", generation, s.search_bank.pending())

    def prioritize(self, ids: list, session: str = "default") -> int:
        """
        Encode these search-bank items before the rest of the pool; returns how
        many were queued. Unknown or already encoded ids are skipped. The items'
        original pool batches stay queued but find them cached, so nothing is
        encoded twice. The batch being encoded right now is not interrupted.
        """
        s = self.session(session)
        with s.lock:
            generation = s.generation
            items = [e["item"] for e in map(s.search_bank.get, ids) if e and e["vectors"] is None]
        self._enqueue(_PRIORITY_DISPLAYED, session, "search", generation, items)
        return len(items)

    def sync_selection(self, items: list, session: str = "default") -> dict:
        """
        Make the selection bank hold exactly items (the browser's full selected
        list): add new ones (vectors from the cache when possible), drop the rest.
        """
        s = self.session(session)
        wanted = {item["id"] for item in items}
        s.selection_bank.remove([i for i in s.selection_bank.ids() if i not in wanted])
        s.selection_bank.add(items, self._cached(wanted))
        self._enqueue(_PRIORITY_SELECTION, session, "selection", None, s.selection_bank.pending())
        return self.status(session)

    def preload(self, items: list, session: str = "default") -> int:
        """Queue items for encoding into the cache only; returns how many were queued."""
        with self._cache_lock:
            todo = [item for item in items if item["id"] not in self._cache]
        self._enqueue(_PRIORITY_PRELOAD, session, "cache", None, todo)
        return len(todo)

    # ── Consumer helpers ──────────────────────────────────────────────────────

    def similarity(self, view: str, ids: list, bank: MemoryBank) -> np.ndarray:
        """
        Cosine similarity (len(ids) × len(bank)) of the given items to every
        vector in bank, for view. Rows of items without a vector are NaN;
        columns follow bank.matrix(view)[0].
        """
        _, mat = bank.matrix(view)
        out = np.full((len(ids), len(mat)), np.nan, dtype=np.float32)
        vecs = [self.vector(i, view) for i in ids]
        for row, vec in enumerate(vecs):
            if vec is not None and len(mat):
                out[row] = mat @ vec
        return out

    def vector(self, item_id: str, view: str, session: str = "default") -> np.ndarray | None:
        """An item's vector for view, from either bank or the cache; None if not encoded."""
        s = self.session(session)
        for bank in (s.search_bank, s.selection_bank):
            entry = bank.get(item_id)
            if entry and entry["vectors"] and view in entry["vectors"]:
                return entry["vectors"][view]
        return self._cached([item_id]).get(item_id, {}).get(view)

    def wait(self, bank: str = "search", timeout: float | None = 30.0, session: str = "default") -> bool:
        """Block until bank ("search" | "selection") is fully encoded; False on timeout."""
        target = getattr(self.session(session), f"{bank}_bank")
        with self._idle:
            return self._idle.wait_for(lambda: target.status()["ready"] or self._busy == 0, timeout) \
                and target.status()["ready"]

    def status(self, session: str = "default") -> dict:
        s = self.session(session)
        return {
            "enabled": True,
            "views": list(self.encoder.views),
            "search": s.search_bank.status(),
            "selection": s.selection_bank.status(),
            "cached": len(self._cache),
        }

    # ── Worker ────────────────────────────────────────────────────────────────

    def _cached(self, ids) -> dict:
        """Cached vectors for ids (marks them recently used)."""
        with self._cache_lock:
            out = {}
            for i in ids:
                if i in self._cache:
                    self._cache.move_to_end(i)
                    out[i] = self._cache[i]
            return out

    def _enqueue(self, priority: int, session: str, bank: str, generation, items: list) -> None:
        for start in range(0, len(items), self._batch_size):
            with self._idle:
                self._busy += 1
            batch = items[start:start + self._batch_size]
            self._queue.put((priority, next(self._seq), session, bank, generation, batch))

    def _worker(self) -> None:
        while True:
            _, _, session, bank, generation, batch = self._queue.get()
            try:
                self._process(session, bank, generation, batch)
            except Exception:
                log.exception("Representation batch failed (%d items)", len(batch))
                # Mark as done-without-vectors so status() can reach ready.
                if bank != "cache":
                    target = getattr(self.session(session), f"{bank}_bank")
                    for item in batch:
                        target.set_vectors(item["id"], {})
            finally:
                with self._idle:
                    self._busy -= 1
                    self._idle.notify_all()

    def _process(self, session: str, bank: str, generation, batch: list) -> None:
        s = self.session(session)
        if bank == "search" and generation != s.generation:
            return   # a newer query has replaced this pool
        found = self._cached(item["id"] for item in batch)
        todo = [item for item in batch if item["id"] not in found]
        if todo:
            image_paths = list(self._fetchers.map(self._image_path, todo))
            objects = [{
                "id":         item["id"],
                "text":       self.encoder.object_text(item.get("properties", {})),
                "image_path": path,
            } for item, path in zip(todo, image_paths)]
            vectors = self.encoder.encode(objects)
            with self._cache_lock:
                for item, vecs in zip(todo, vectors):
                    found[item["id"]] = self._cache[item["id"]] = vecs
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        # Fill both banks: an item kept while still pending gets its vectors too
        # (also for preloaded items that a search has since added to a bank).
        for item_id, vecs in found.items():
            s.search_bank.set_vectors(item_id, vecs)
            s.selection_bank.set_vectors(item_id, vecs)

    def _image_path(self, item: dict) -> str | None:
        """Local thumbnail of the item's first image that is not a placeholder."""
        if self.encoder.views == ("text",):
            return None
        for url in item.get("properties", {}).get("images", []):
            if not self._images.is_allowed(url):
                continue
            path, _ = self._images.thumbnail(url)
            if path is not None:
                return str(path)
        return None

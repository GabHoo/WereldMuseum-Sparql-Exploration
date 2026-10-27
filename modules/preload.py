"""
Preloader — warms the caches for a fixed list of demo queries at startup.

For each query in PRELOAD_QUERIES (a JSON list of strings), in a background
thread:
    1. run it through the configured QueryGeneration + KnowledgeBase +
       ArtifactFetcher, and store the fetched artifacts on disk (PoolCache),
       so a later /search for the same query skips the endpoint;
    2. download each artifact's first thumbnail (ImageCache, on disk);
    3. score the pool with the RetrievalEvaluator, if any, to warm its
       in-memory caches (otherwise the first /search pays ~10 s);
    4. queue the pool for encoding into the Embedding module's vector cache
       (in memory, lowest priority: live searches go first).

The pool cache is keyed by the generated SPARQL, so any wording that yields
the same query is served from disk. Only the preloader writes to it.

Cache layout (persists across restarts; delete to re-fetch):
    <cache_dir>/<sha1(sparql)>.json   {"sparql": ..., "artifacts": [Artifact.to_dict()]}
"""
import hashlib
import json
import logging
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

from modules.artifact import Artifact

log = logging.getLogger(__name__)

_THUMB_FETCHERS = 4


class PoolCache:
    """The fetched artifacts per find query, on disk. get() returns None when not cached."""

    def __init__(self, cache_dir: str):
        self._dir = pathlib.Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, sparql: str) -> list[Artifact] | None:
        path = self._path(sparql)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return [Artifact.from_dict(d) for d in data["artifacts"]]

    def put(self, sparql: str, artifacts: list[Artifact]) -> None:
        path = self._path(sparql)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"sparql": sparql, "artifacts": [a.to_dict() for a in artifacts]},
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: readers never see a half-written file

    def _path(self, sparql: str) -> pathlib.Path:
        return self._dir / f"{hashlib.sha1(sparql.encode('utf-8')).hexdigest()}.json"


class Preloader:
    """Runs the steps in the module docstring for each query, once, in a background thread."""

    def __init__(self, queries: list[str], modules: dict, pool_cache: PoolCache, build_pool):
        self._queries = queries
        self._modules = modules
        self._pools = pool_cache
        self._build_pool = build_pool   # app._build_pool: artifacts → pool items
        self._status = {"queries": len(queries), "done": 0, "failed": [], "objects": 0, "running": False}

    def start(self) -> None:
        threading.Thread(target=self._run, name="preload", daemon=True).start()

    def status(self) -> dict:
        return dict(self._status)

    def _run(self) -> None:
        self._status["running"] = True
        for query in self._queries:
            try:
                pool = self._preload(query)
                self._status["objects"] += len(pool)
                log.info("Preloaded %r: %d objects", query, len(pool))
            except Exception:
                log.exception("Preloading %r failed", query)
                self._status["failed"].append(query)
            self._status["done"] += 1
        self._status["running"] = False

    def _preload(self, query: str) -> list:
        """Warm every cache for one query (steps 1–4 above); returns its pool."""
        sparql = self._modules["query_generation"].convert(query, {})
        artifacts = self._pools.get(sparql)
        if artifacts is None:
            rows = self._modules["knowledge_base"].execute(sparql)
            ids = [r["artifact"] for r in rows if r.get("artifact")]
            artifacts = self._modules["artifact_fetcher"].fetch(ids)
            self._pools.put(sparql, artifacts)

        images = self._modules["image_cache"]
        with ThreadPoolExecutor(_THUMB_FETCHERS, thread_name_prefix="preload-images") as pool:
            list(pool.map(lambda a: self._first_thumbnail(images, a), artifacts))

        pool = self._build_pool(artifacts)   # after the thumbnails: drops placeholder-only artifacts
        # Score once as /search does: fills the evaluator's in-memory caches
        # (text embeddings, concept expansions), ~10 s per pool when cold.
        evaluator = self._modules.get("retrieval_evaluator")
        if evaluator and pool:
            evaluator.score(pool, self._modules["query_generation"].get_query_features(query))
        embedding = self._modules.get("embedding")
        if embedding:
            embedding.preload(pool)
        return pool

    @staticmethod
    def _first_thumbnail(images, artifact: Artifact) -> None:
        """Download thumbnails until the first one that is not a placeholder."""
        for url in artifact.images:
            if images.is_allowed(url) and images.thumbnail(url)[0] is not None:
                return

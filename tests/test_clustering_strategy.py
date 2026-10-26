"""
Tests for ClusteringStrategy on synthetic vectors (no model needed).

Run: python -m pytest tests/  or  python -m tests.test_clustering_strategy
"""
import numpy as np

from modules.base import SelectionHistory
from modules.selection.clustering import KMeansClusterer
from modules.selection.clustering_strategy import ClusteringStrategy

DIM = 16


class FakeEmbedding:
    """The parts of EmbeddingService the strategy uses; vectors appear once 'encoded'."""

    def __init__(self, vectors: dict, ready: bool = True):
        self.encoder = type("Encoder", (), {"views": ("text",)})()
        self._vectors = vectors
        self.ready = ready
        self.waited = []

    def matrix(self, ids, view):
        found = [i for i in ids if self.ready and i in self._vectors]
        if not found:
            return [], np.zeros((0, 0), dtype=np.float32)
        return found, np.stack([self._vectors[i] for i in found])

    def wait(self, ids, timeout=None):
        self.waited.append(timeout)
        return self.ready


def _clusters(sizes: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, size in sizes.items():
        centre = rng.normal(size=DIM) * 3
        for i in range(size):
            v = centre + 0.1 * rng.normal(size=DIM)
            out[f"{name}{i}"] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def _pool(vectors):
    return [{"id": i, "title": i[0].upper(), "score": 1.0} for i in vectors]


def _strategy(emb, **kw):
    return ClusteringStrategy(emb, KMeansClusterer(), n_clusters=3, seed=0, **kw)


def test_first_batch_one_per_cluster_when_vectors_ready():
    vecs = _clusters({"a": 10, "b": 10, "c": 10})
    pool, emb = _pool(vecs), FakeEmbedding(vecs)
    strategy = _strategy(emb, wait=2.0)
    strategy.prepare(pool)
    assert emb.waited == [2.0]
    assert {item["selection"]["cluster_label"] for item in pool} == {"A", "B", "C"}
    batch = strategy.select(pool, 3, SelectionHistory())
    assert sorted(item["id"][0] for item in batch) == ["a", "b", "c"]


def test_first_batch_random_until_vectors_ready():
    vecs = _clusters({"a": 10, "b": 10, "c": 10})
    pool, emb = _pool(vecs), FakeEmbedding(vecs, ready=False)
    strategy = _strategy(emb)
    strategy.prepare(pool)                                   # waited, but still encoding
    assert not any("selection" in item for item in pool)
    first = strategy.select(pool, 3, SelectionHistory())
    assert len(first) == 3 and strategy.last_debug["reason"] == "pool not encoded yet"

    emb.ready = True                                         # encoding finished meanwhile
    seen = [item["id"] for item in first]
    strategy.select(pool, 3, SelectionHistory(seen=seen))
    assert strategy.last_debug["mode"] == "clustering"
    assert all("cluster" in item["selection"] for item in pool)   # annotations travel with the pool


def test_steers_to_kept_cluster():
    vecs = _clusters({"a": 10, "b": 10, "c": 10})
    pool = _pool(vecs)
    strategy = _strategy(FakeEmbedding(vecs), steering_weight=1.0)
    strategy.prepare(pool)
    batch = strategy.select(pool, 4, SelectionHistory(accepted=["a0"], seen=["a0", "b0", "c0"]))
    assert all(item["id"].startswith("a") for item in batch)


def test_without_embedding_selects_at_random():
    vecs = _clusters({"a": 10})
    pool = _pool(vecs)
    strategy = ClusteringStrategy(None, KMeansClusterer())
    strategy.prepare(pool)
    batch = strategy.select(pool, 4, SelectionHistory(seen=["a0"]))
    assert len(batch) == 4 and "a0" not in {item["id"] for item in batch}
    assert strategy.last_debug == {"mode": "random", "reason": "no embedding"}


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")

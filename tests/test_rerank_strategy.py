"""
Tests for RerankStrategy on synthetic vectors (no model needed).

Run: python -m pytest tests/  or  python -m tests.test_rerank_strategy
"""
import random

import numpy as np

from modules.base import SelectionHistory
from modules.selection.rerank_strategy import RerankStrategy, _mmr, _score_knn, _score_rocchio

DIM = 16


class FakeEmbedding:
    """The parts of EmbeddingService the strategy uses, over fixed vectors (None = not encoded yet)."""

    def __init__(self, vectors: dict, view: str = "fused"):
        self.encoder = type("Encoder", (), {"views": (view,)})()
        self._vectors = vectors

    def vector(self, item_id, view):
        return self._vectors.get(item_id)

    def matrix(self, ids, view):
        found = [i for i in ids if self._vectors.get(i) is not None]
        if not found:
            return [], np.zeros((0, 0), dtype=np.float32)
        return found, np.stack([self._vectors[i] for i in found])

    def wait(self, ids, timeout=None):
        return True


def _clusters(sizes: dict, seed: int = 0) -> dict:
    """{id: unit vector}: one tight cluster per name, ids '<name><i>'."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, size in sizes.items():
        centre = rng.normal(size=DIM)
        for i in range(size):
            v = centre + 0.1 * rng.normal(size=DIM)
            out[f"{name}{i}"] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def _strategy(rep, **kw):
    kw = {"explore": 0.0, "wait": 0.0, "rng": random.Random(0), **kw}
    return RerankStrategy(rep, **kw)


def _pool(vectors):
    return [{"id": i, "score": 1.0} for i in vectors]


# ── Pure helpers ──────────────────────────────────────────────────────────────

def test_rocchio_ranks_positive_cluster_first():
    vecs = _clusters({"a": 5, "b": 5})
    ids = list(vecs)
    mat = np.stack([vecs[i] for i in ids])
    scores = _score_rocchio(mat, mat[:1], np.zeros((0, 0)), beta=1.0, gamma=0.25)
    top = {ids[j] for j in np.argsort(-scores)[:5]}
    assert top == {f"a{i}" for i in range(5)}


def test_negatives_push_items_down():
    vecs = _clusters({"a": 3, "b": 3})
    mat = np.stack(list(vecs.values()))
    pos = mat[:1]
    without = _score_rocchio(mat, pos, np.zeros((0, 0)), 1.0, 0.25)
    with_neg = _score_rocchio(mat, pos, mat[3:4], 1.0, 0.25)
    # b items are pushed down more than a items
    assert (without[3:] - with_neg[3:]).min() > (without[:3] - with_neg[:3]).max()


def test_knn_serves_two_themes():
    vecs = _clusters({"a": 4, "b": 4, "c": 4})
    ids = list(vecs)
    mat = np.stack(list(vecs.values()))
    pos = np.stack([vecs["a0"], vecs["b0"]])
    scores = _score_knn(mat, pos, np.zeros((0, 0)), 1.0, 0.25, k=1)
    top = {ids[j][0] for j in np.argsort(-scores)[:8]}
    assert top == {"a", "b"}


def test_mmr_spreads_over_clusters():
    vecs = _clusters({"a": 6, "b": 6})
    mat = np.stack(list(vecs.values()))
    scores = np.array([1.0] * 6 + [0.8] * 6, dtype=np.float32)
    picked = _mmr(mat, scores, 4, lam=0.5)
    assert {j < 6 for j in picked} == {True, False}
    assert _mmr(mat, scores, 4, lam=1.0) == [0, 1, 2, 3]


# ── Strategy ──────────────────────────────────────────────────────────────────

def test_feedback_steers_next_batch():
    vecs = _clusters({"a": 20, "b": 20})
    rep = FakeEmbedding(vecs)
    pool = _pool(vecs)
    history = SelectionHistory(accepted=["a0"], rejected=["b0"], seen=["a0", "b0"])
    batch = _strategy(rep, mmr_lambda=1.0).select(pool, 10, history)
    assert len(batch) == 10
    assert all(item["id"].startswith("a") for item in batch)
    assert not {"a0", "b0"} & {item["id"] for item in batch}


def test_exploration_takes_its_share():
    vecs = _clusters({"a": 20, "b": 20})
    strategy = _strategy(FakeEmbedding(vecs), explore=0.2)
    batch = strategy.select(_pool(vecs), 10, SelectionHistory(accepted=["a0"], seen=["a0"]))
    assert len(batch) == 10
    assert len(strategy.last_debug["explored_ids"]) == 2
    assert len(strategy.last_debug["ranked_ids"]) == 8


def test_kept_under_earlier_query_counts_as_positive():
    vecs = _clusters({"a": 20, "b": 20})
    rep = FakeEmbedding(vecs)
    # b0 was kept under an earlier query: the browser sends it in accepted
    batch = _strategy(rep, mmr_lambda=1.0).select(_pool(vecs), 5, SelectionHistory(accepted=["b0"], seen=["b0"]))
    assert all(item["id"].startswith("b") for item in batch)


def test_cross_query_off_ignores_kept_outside_pool():
    vecs = _clusters({"a": 10, "b": 10})
    rep = FakeEmbedding(vecs)
    pool = [p for p in _pool(vecs) if p["id"] != "b0"]   # b0 came from an earlier query
    strategy = _strategy(rep, cross_query=False)
    strategy.select(pool, 5, SelectionHistory(accepted=["b0"]))
    assert strategy.last_debug["mode"] == "random"


def test_negatives_are_never_shown():
    vecs = _clusters({"a": 10, "b": 10})
    rep = FakeEmbedding(vecs)
    # b1..b3 were marked not relevant under an earlier query, so never seen here
    history = SelectionHistory(accepted=["a0"], rejected=["b1", "b2", "b3"], seen=["a0"])
    batch = _strategy(rep, explore=1.0).select(_pool(vecs), 19, history)
    assert not {"b1", "b2", "b3"} & {item["id"] for item in batch}
    assert len(batch) == 16


def test_cross_query_off_ignores_negatives_outside_pool():
    vecs = _clusters({"a": 10, "b": 10})
    rep = FakeEmbedding(vecs)
    pool = [p for p in _pool(vecs) if p["id"] != "b0"]   # b0 came from an earlier query
    history = SelectionHistory(rejected=["b0"])
    on = _strategy(rep)
    on.select(pool, 5, history)
    assert on.last_debug["n_neg"] == 1
    off = _strategy(rep, cross_query=False)
    off.select(pool, 5, history)
    assert off.last_debug["mode"] == "random"


def test_positive_wins_over_negative():
    vecs = _clusters({"a": 10, "b": 10})
    rep = FakeEmbedding(vecs)
    strategy = _strategy(rep)
    strategy.select(_pool(vecs), 5, SelectionHistory(accepted=["a0"], rejected=["a0", "b0"], seen=["a0"]))
    assert strategy.last_debug["n_pos"] == 1
    assert strategy.last_debug["n_neg"] == 1


def test_unencoded_candidates_are_explored():
    vecs = _clusters({"a": 10})
    vecs.update({f"x{i}": None for i in range(10)})   # still queued for encoding
    strategy = _strategy(FakeEmbedding(vecs))
    hist = SelectionHistory(accepted=["a0"], seen=["a0"])
    batch = strategy.select(_pool(vecs), 15, hist)
    assert len(batch) == 15
    assert len(strategy.last_debug["ranked_ids"]) == 9
    assert strategy.last_debug["coverage"] == round(9 / 19, 3)


def test_fallbacks_are_random():
    vecs = _clusters({"a": 10})
    pool = _pool(vecs)
    no_rep = RerankStrategy(None, rng=random.Random(0))
    assert len(no_rep.select(pool, 5, SelectionHistory(seen=["a0"]))) == 5
    no_feedback = _strategy(FakeEmbedding(vecs))
    batch = no_feedback.select(pool, 5, SelectionHistory())
    assert len(batch) == 5 and no_feedback.last_debug["mode"] == "random"


def test_scores_are_written_to_items():
    vecs = _clusters({"a": 5, "b": 5})
    pool = _pool(vecs)
    _strategy(FakeEmbedding(vecs)).select(pool, 3, SelectionHistory(accepted=["a0"], seen=["a0"]))
    scores = {p["id"]: p["score"] for p in pool}
    assert scores["a1"] > scores["b1"]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")

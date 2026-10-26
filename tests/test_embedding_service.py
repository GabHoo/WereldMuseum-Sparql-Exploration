"""
Tests for EmbeddingService with a fake encoder (no model needed).

Run: python -m pytest tests/  or  python -m tests.test_embedding_service
"""
import threading

import numpy as np

from modules.embedding.base import ObjectEncoder
from modules.embedding.service import DISPLAYED, FEEDBACK, POOL, EmbeddingService

DIM = 8


class FakeEncoder(ObjectEncoder):
    """Text-only encoder that records the order it encodes in; can be paused or made to fail."""

    def __init__(self):
        self.views = ("text",)
        self.order: list = []
        self.gate = threading.Event()
        self.gate.set()
        self.started = threading.Event()   # set once encode() has been entered
        self.fail_ids: set = set()

    def object_text(self, properties):
        return properties.get("title", "")

    def encode(self, objects):
        self.started.set()
        self.gate.wait()
        if any(o["id"] in self.fail_ids for o in objects):
            raise RuntimeError("model error")
        self.order += [o["id"] for o in objects]
        out = []
        for o in objects:
            v = np.random.default_rng(abs(hash(o["id"])) % 2**32).normal(size=DIM).astype(np.float32)
            out.append({"text": v / np.linalg.norm(v)})
        return out


def _items(prefix, n):
    return [{"id": f"{prefix}{i}", "properties": {"title": f"{prefix} {i}"}} for i in range(n)]


def _service(batch_size=4):
    encoder = FakeEncoder()
    return encoder, EmbeddingService(encoder, image_cache=None, batch_size=batch_size)


def test_ensure_encodes_and_caches():
    encoder, svc = _service()
    pool = _items("a", 10)
    svc.ensure(pool)
    assert svc.wait([i["id"] for i in pool], timeout=5)
    ids, mat = svc.matrix([i["id"] for i in pool], "text")
    assert ids == [i["id"] for i in pool] and mat.shape == (10, DIM)
    assert svc.ensure(pool) == 0          # cached: nothing queued again
    assert len(encoder.order) == 10


def test_priorities_and_newest_first():
    encoder, svc = _service(batch_size=2)
    encoder.gate.clear()                  # hold the worker while queueing
    svc.ensure(_items("busy", 2))         # the worker takes this batch and waits on the gate
    assert encoder.started.wait(5)
    old, new = _items("old", 4), _items("new", 4)
    svc.ensure(old, POOL)
    svc.ensure(new, POOL)                 # a newer search: before the older one's leftovers
    svc.ensure(new[3:], DISPLAYED)        # the batch on screen: before the rest
    svc.ensure([old[0]], FEEDBACK)        # a kept item: first of all
    encoder.gate.set()
    assert svc.wait([i["id"] for i in old + new], timeout=5)
    order = encoder.order[2:]             # after the batch that was already running
    assert order[0] == "old0"
    assert order[1] == "new3"
    assert order.index("new0") < order.index("old1")


def test_two_tabs_do_not_interfere():
    # Tab A searches, tab B searches something else, tab A asks for its next batch:
    # A's pool still gets (and keeps) its vectors; nothing per tab is replaced.
    encoder, svc = _service()
    a, b = _items("a", 6), _items("b", 6)
    svc.ensure(a)
    svc.ensure(b)
    svc.ensure(a)                         # /next from tab A
    assert svc.wait([i["id"] for i in a + b], timeout=5)
    assert len(svc.matrix([i["id"] for i in a], "text")[0]) == 6
    assert len(svc.matrix([i["id"] for i in b], "text")[0]) == 6


def test_failed_batch_is_not_cached_and_retried():
    encoder, svc = _service(batch_size=8)
    pool = _items("x", 3)
    encoder.fail_ids = {"x1"}
    svc.ensure(pool)
    assert svc.wait([i["id"] for i in pool], timeout=5)   # waiters return after a failure
    assert svc.vector("x0", "text") is None
    encoder.fail_ids = set()
    assert svc.ensure(pool) == 3
    assert svc.wait([i["id"] for i in pool], timeout=5)
    assert svc.vector("x1", "text") is not None


def test_status_and_unknown_ids():
    encoder, svc = _service()
    assert svc.wait(["never-queued"], timeout=0.1)        # unknown ids do not block
    assert svc.vector("never-queued", "text") is None
    svc.ensure(_items("s", 3))
    svc.wait(["s0", "s1", "s2"], timeout=5)
    assert svc.status() == {"enabled": True, "views": ["text"], "cached": 3, "queued": 0}


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")

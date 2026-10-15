"""
RelevanceFeedbackStrategy — re-ranks the unseen pool after every batch, using
the curator's feedback and the Representation module's object vectors.

Feedback (accumulates over the whole query):
    positives P  history.accepted ∪ the selection bank (the kept objects).
                 cross_query=False keeps only kept objects in the current pool.
    negatives N  history.rejected − P. "Not kept" is a weak signal, so N
                 weighs less than P (gamma < beta).

Scoring of each unseen candidate c (vectors are L2-normalised, dot = cosine):
    rocchio  s(c) = c · (beta·centroid(P) − gamma·centroid(N))
    knn      s(c) = mean top-k sim(c, P) − gamma · mean top-k sim(c, N)
             (better when the curator collects several themes at once)

A batch of n: round(explore·n) random candidates (exploration), the rest picked
greedily by MMR — argmax lambda·s(c) − (1−lambda)·max sim(c, picked) — so the
batch is not ten near-duplicates. The batch is shuffled, so ranked and explored
items are not told apart by position.

Vectors are filled in the background, so not every candidate has one yet:
only encoded candidates are ranked, and free slots are filled at random. The
strategy waits at most `wait` seconds for the feedback items' own vectors
(app.py prioritises the batch on screen, so they are usually ready).

Without feedback, or without the Representation module, it selects at random.

Usage (CLI, offline simulation against the random baseline):
    python -m modules.selection.relevance_feedback <pool.json | export.json | runs.log> \\
        [--target <concept URI>] [--batches 8] [--n 10] [--runs 5]
"""
import logging
import random
import sys
import time

import numpy as np

from modules.base import SelectionHistory, SelectionStrategy

log = logging.getLogger(__name__)

_POLL = 0.05   # seconds between checks while waiting for feedback vectors


# ── Scoring and batch composition (pure numpy) ───────────────────────────────

def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _score_rocchio(cands: np.ndarray, pos: np.ndarray, neg: np.ndarray,
                   beta: float, gamma: float) -> np.ndarray:
    """cands m × d, pos p × d, neg q × d (p or q may be 0) → scores (m,)."""
    query = np.zeros(cands.shape[1], dtype=np.float32)
    if len(pos):
        query += beta * _unit(pos.mean(axis=0))
    if len(neg):
        query -= gamma * _unit(neg.mean(axis=0))
    return cands @ query


def _mean_top_k(sims: np.ndarray, k: int) -> np.ndarray:
    """Row-wise mean of the k largest values of sims (m × p)."""
    k = min(k, sims.shape[1])
    return np.sort(sims, axis=1)[:, -k:].mean(axis=1)


def _score_knn(cands: np.ndarray, pos: np.ndarray, neg: np.ndarray,
               beta: float, gamma: float, k: int) -> np.ndarray:
    scores = np.zeros(len(cands), dtype=np.float32)
    if len(pos):
        scores += beta * _mean_top_k(cands @ pos.T, k)
    if len(neg):
        scores -= gamma * _mean_top_k(cands @ neg.T, k)
    return scores


def _mmr(cands: np.ndarray, scores: np.ndarray, n: int, lam: float) -> list:
    """Indices of n candidates picked greedily by maximal marginal relevance."""
    picked: list = []
    max_sim = np.full(len(cands), -np.inf, dtype=np.float32)
    available = np.ones(len(cands), dtype=bool)
    for _ in range(min(n, len(cands))):
        redundancy = np.where(np.isfinite(max_sim), max_sim, 0.0)
        gain = np.where(available, lam * scores - (1 - lam) * redundancy, -np.inf)
        best = int(np.argmax(gain))
        picked.append(best)
        available[best] = False
        max_sim = np.maximum(max_sim, cands @ cands[best])
    return picked


# ── Strategy ──────────────────────────────────────────────────────────────────

class RelevanceFeedbackStrategy(SelectionStrategy):
    def __init__(self, representation=None, view: str = "fused", scoring: str = "rocchio",
                 beta: float = 1.0, gamma: float = 0.25, k: int = 3, mmr_lambda: float = 0.7,
                 explore: float = 0.2, wait: float = 1.0, cross_query: bool = True,
                 rng: random.Random | None = None):
        if scoring not in ("rocchio", "knn"):
            raise ValueError(f"Unknown RERANK_SCORING: {scoring!r} (rocchio | knn)")
        if representation is None:
            log.warning("SELECTION=relevance_feedback needs REPRESENTATION; selecting at random")
        elif view not in representation.encoder.views:
            raise ValueError(f"RERANK_VIEW={view!r} is not in REPRESENTATION_VIEWS "
                             f"{list(representation.encoder.views)}")
        self.rep = representation
        self.view = view
        self.scoring = scoring
        self.beta, self.gamma, self.k = beta, gamma, k
        self.mmr_lambda = mmr_lambda
        self.explore = explore
        self.wait = wait
        self.cross_query = cross_query
        self._rng = rng or random.Random()
        self.last_debug: dict = {}   # what the last select() did, for the run log

    def select(self, pool: list, n: int, history: SelectionHistory) -> list:
        seen = set(history.seen)
        cands = [item for item in pool if item["id"] not in seen]
        if self.rep is None:
            self.last_debug = {"mode": "random", "reason": "no representation"}
            return self._rng.sample(cands, min(n, len(cands)))

        pos_ids, neg_ids = self._feedback(pool, history)
        if not pos_ids and not neg_ids:
            self.last_debug = {"mode": "random", "reason": "no feedback"}
            return self._rng.sample(cands, min(n, len(cands)))

        self._wait_for(pos_ids + neg_ids)
        pos = self._stack(pos_ids)
        neg = self._stack(neg_ids)
        if not len(pos) and not len(neg):
            self.last_debug = {"mode": "random", "reason": "feedback not encoded",
                               "n_pos": len(pos_ids), "n_neg": len(neg_ids)}
            return self._rng.sample(cands, min(n, len(cands)))

        # Candidates with a vector are ranked; the rest can only be explored.
        bank_ids, bank_mat = self.rep.search_bank.matrix(self.view)
        row = {i: r for r, i in enumerate(bank_ids)}
        by_id = {item["id"]: item for item in cands}
        enc_ids = [i for i in by_id if i in row]
        enc_mat = bank_mat[[row[i] for i in enc_ids]] if enc_ids else np.zeros((0, 0), np.float32)

        if self.scoring == "knn":
            scores = _score_knn(enc_mat, pos, neg, self.beta, self.gamma, self.k) if enc_ids else np.zeros(0)
        else:
            scores = _score_rocchio(enc_mat, pos, neg, self.beta, self.gamma) if enc_ids else np.zeros(0)
        for item_id, score in zip(enc_ids, scores):
            by_id[item_id]["score"] = round(float(score), 4)

        n = min(n, len(cands))
        n_explore = round(self.explore * n)
        ranked = [enc_ids[i] for i in _mmr(enc_mat, scores, n - n_explore, self.mmr_lambda)] if enc_ids else []
        rest = [i for i in by_id if i not in set(ranked)]
        explored = self._rng.sample(rest, n - len(ranked))

        self.last_debug = {
            "mode":        self.scoring,
            "view":        self.view,
            "n_pos":       len(pos), "n_neg": len(neg),
            "coverage":    round(len(enc_ids) / len(cands), 3) if cands else 0.0,
            "ranked_ids":  ranked,
            "explored_ids": explored,
            "scores":      {i: by_id[i]["score"] for i in ranked},
            "score_range": [round(float(scores.min()), 4), round(float(scores.max()), 4)] if len(scores) else None,
        }
        batch = [by_id[i] for i in ranked + explored]
        self._rng.shuffle(batch)
        return batch

    # ── Private ───────────────────────────────────────────────────────────────

    def _feedback(self, pool: list, history: SelectionHistory) -> tuple:
        """(positive ids, negative ids), each without duplicates."""
        kept = self.rep.selection_bank.ids()
        if not self.cross_query:
            in_pool = {item["id"] for item in pool}
            kept = [i for i in kept if i in in_pool]
        pos = list(dict.fromkeys(list(history.accepted) + kept))
        pos_set = set(pos)
        neg = [i for i in dict.fromkeys(history.rejected) if i not in pos_set]
        return pos, neg

    def _pending(self, ids: list) -> list:
        """Ids still queued for encoding in either bank (unknown ids never will be)."""
        out = []
        for i in ids:
            entry = self.rep.search_bank.get(i) or self.rep.selection_bank.get(i)
            if entry is not None and entry["vectors"] is None:
                out.append(i)
        return out

    def _wait_for(self, ids: list) -> None:
        deadline = time.monotonic() + self.wait
        pending = self._pending(ids)
        while pending and time.monotonic() < deadline:
            time.sleep(_POLL)
            pending = self._pending(pending)

    def _stack(self, ids: list) -> np.ndarray:
        vecs = [v for v in (self.rep.vector(i, self.view) for i in ids) if v is not None]
        return np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)


# ── CLI: offline simulation ───────────────────────────────────────────────────

_CONCEPT_FIELDS = ("type_uris", "material_uris", "place_uris")


def _load_pool(path: str) -> list:
    """A pool list, an export {meta, items}, or a run log (last search event's pool)."""
    import json
    with open(path, encoding="utf-8") as f:
        if path.endswith((".log", ".jsonl")):
            pools = [r["pool"] for r in map(json.loads, filter(str.strip, f)) if r.get("event") == "search"]
            if not pools:
                raise SystemExit(f"No search event in {path}")
            return pools[-1]
        data = json.load(f)
    return data if isinstance(data, list) else data.get("items", [])


def _concepts(item: dict) -> set:
    props = item.get("properties", {})
    return {uri for field in _CONCEPT_FIELDS for uri in props.get(field) or []}


def _simulate(strategy, rep, pool: list, relevant: set, batches: int, n: int) -> list:
    """Per batch: (acceptance rate, cumulative recall, diversity of the batch)."""
    rep.sync_selection([])
    rep.wait("selection", timeout=None)
    history, kept, rows = SelectionHistory(), [], []
    for _ in range(batches):
        shown = strategy.select(pool, n, history)
        if not shown:
            break
        hits = [item for item in shown if item["id"] in relevant]
        kept += hits
        rep.sync_selection(kept)
        history.seen += [item["id"] for item in shown]
        history.accepted += [item["id"] for item in hits]
        history.rejected += [item["id"] for item in shown if item["id"] not in relevant]
        vecs = [v for v in (rep.vector(item["id"], strategy.view) for item in shown) if v is not None]
        if len(vecs) >= 2:
            mat = np.stack(vecs)
            sims = mat @ mat.T
            diversity = float(1 - sims[np.triu_indices(len(vecs), 1)].mean())
        else:
            diversity = float("nan")
        rows.append((len(hits) / len(shown), len(kept) / max(len(relevant), 1), diversity))
    # Pool exhausted early: pad, so runs line up per batch.
    return rows + [(float("nan"),) * 3] * (batches - len(rows))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m modules.selection.relevance_feedback <pool.json | export.json | runs.log>"
              " [--target <concept URI>] [--batches 8] [--n 10] [--runs 5]")
        sys.exit(1)

    from collections import Counter

    from dotenv import load_dotenv
    load_dotenv()
    import config
    from modules.representation.embeddinggemma import build_from_env
    from modules.representation.memory_bank import RepresentationService

    def _arg(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    pool = _load_pool(sys.argv[1])
    batches, n, runs = int(_arg("--batches", 8)), int(_arg("--n", 10)), int(_arg("--runs", 5))
    counts = Counter(uri for item in pool for uri in _concepts(item))
    target = _arg("--target", None)
    if target is None:
        print(f"Pool: {len(pool)} items. Pick a --target concept; most common in this pool:")
        for uri, count in counts.most_common(15):
            print(f"  {count:4d}  {uri}")
        sys.exit(0)
    relevant = {item["id"] for item in pool if target in _concepts(item)}
    print(f"Pool: {len(pool)} items, {len(relevant)} relevant to {target}")
    if not relevant:
        sys.exit(1)

    print("Loading model …")
    rep = RepresentationService(build_from_env(), config._build_image_cache())
    print("Encoding the pool …")
    rep.on_search(pool, search_id="cli")
    rep.wait("search", timeout=None)

    kwargs = config.rerank_kwargs()
    strategies = {
        "random":             lambda seed: RelevanceFeedbackStrategy(None, rng=random.Random(seed), **kwargs),
        "relevance_feedback": lambda seed: RelevanceFeedbackStrategy(rep, rng=random.Random(seed), **kwargs),
    }
    logging.getLogger(__name__).setLevel(logging.ERROR)   # the random baseline's warning
    for name, make in strategies.items():
        results = np.array([_simulate(make(seed), rep, pool, relevant, batches, n) for seed in range(runs)])
        mean = np.nanmean(results, axis=0)
        print(f"\n── {name} (mean of {runs} runs) ──")
        print("batch  accept  recall  diversity")
        for b, (acc, rec, div) in enumerate(mean, 1):
            print(f"{b:5d}  {acc:6.2f}  {rec:6.2f}  {div:9.3f}")
